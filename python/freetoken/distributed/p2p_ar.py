"""Two-rank one-shot P2P all-reduce for small messages.

NCCL's LL ring protocol costs ~30-40us per 5-20 KiB all-reduce on a two-GPU
PCIe box (no NVLink); a decode step pays it twice per layer (~97 calls -> the
single largest latency item of a batch-size-1 step). This replaces those calls
with ONE triton kernel per all-reduce:

    stage x into my IPC-exported buffer
      -> system-scope RELEASE store of ``seq`` into the peer's flag
      -> spin with an ACQUIRE load on my flag until the peer's ``seq`` lands
      -> sum x + peer's staged buffer (fp32 add of two bf16 -> bit-identical
         to ncclSum's fp32 accumulation over two ranks)

Everything is device-side off a monotonic device sequence counter, so the
kernel is CUDA-graph capturable and replays keep both ranks' flags in step
(SPMD: every rank runs the same call sequence, eager warmups and replays
alike; capture itself records without executing, so counters stay aligned).

Buffers are raw ``cudaMalloc`` (never the torch pool): the engine runs with
expandable_segments, whose VMM allocations ``cudaIpcGetMemHandle`` rejects.
Messages above ``max_bytes``, non-bf16, or non-contiguous inputs fall back to
the wrapped NCCL impl, so prefill-sized reductions are untouched.

Set FREETOKEN_P2P_AR=0 to keep plain NCCL everywhere (A/B or fallback).
"""

from __future__ import annotations

import ctypes
import os
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from freetoken.utils import init_logger

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from .info import DistributedInfo

logger = init_logger(__name__)

# Flag/data split of one IPC block: [flag int64][pad to 128B][data max_bytes].
_FLAG_BYTES = 128
# One CTA covers the whole message: decode reductions here are <= 20 KiB
# (bs x hidden bf16); a single launch keeps the protocol trivially ordered.
_MAX_BYTES = 32 * 1024
_BLOCK = _MAX_BYTES // 2  # bf16 elements


@triton.jit
def _p2p_one_shot_sum_bf16(
    x_ptr,        # *bf16 local partial (in-place: also the output)
    out_ptr,      # *bf16 result (may alias x_ptr)
    my_data_addr,   # int64: this rank's IPC data region
    peer_data_addr, # int64: peer's IPC data region (mapped)
    my_flag_addr,   # int64: this rank's flag (peer writes it)
    peer_flag_addr, # int64: peer's flag (this rank writes it)
    counter_addr,   # int64: local monotonic launch counter
    n,              # elements
    BLOCK: tl.constexpr,
):
    seq = tl.atomic_add(counter_addr.to(tl.pointer_type(tl.int64)), 1) + 1
    my_data = my_data_addr.to(tl.pointer_type(tl.bfloat16))
    peer_data = peer_data_addr.to(tl.pointer_type(tl.bfloat16))
    my_flag = my_flag_addr.to(tl.pointer_type(tl.int64))
    peer_flag = peer_flag_addr.to(tl.pointer_type(tl.int64))
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(my_data + offs, x, mask=mask)
    # release: the staged data is visible system-wide before the flag lands
    tl.atomic_xchg(peer_flag, seq, sem="release", scope="sys")
    while tl.atomic_add(my_flag, 0, sem="acquire", scope="sys") < seq:
        pass
    y = tl.load(peer_data + offs, mask=mask)
    summed = x.to(tl.float32) + y.to(tl.float32)
    tl.store(out_ptr + offs, summed.to(tl.bfloat16), mask=mask)


@triton.jit
def _p2p_probe_store(addr, val):
    """Write one int64 into a (peer-mapped) address; the build probe uses this to
    verify the P2P data plane actually delivers -- some PCIe topologies accept the
    IPC mapping and silently black-hole cross-GPU writes."""
    tl.store(addr.to(tl.pointer_type(tl.int64)), val)


def _ipc_handle_bytes(handle) -> bytes:
    return ctypes.string_at(handle.getPtr(), 64)


def _ipc_handle_from_bytes(data: bytes):
    from cuda.bindings import runtime as rt

    handle = rt.cudaIpcMemHandle_t()
    ctypes.memmove(handle.getPtr(), data, 64)
    return handle


class P2POneShotAllReducer:
    """Symmetric two-rank reducer; ``try_build`` is the only constructor callers use."""

    def __init__(
        self,
        rank: int,
        my_ptr: int,
        peer_ptr: int,
        counter_ptr: int,
        max_bytes: int,
        owns: tuple,
    ) -> None:
        self.rank = rank
        self._my_ptr = my_ptr
        self._peer_ptr = peer_ptr
        self._counter_ptr = counter_ptr
        self.max_bytes = max_bytes
        self.max_elems = max_bytes // 2
        self._owns = owns  # keep cudaMalloc'd pointers + handles alive

    # ------------------------------------------------------------------ build

    @classmethod
    def try_build(
        cls,
        tp_info: "DistributedInfo",
        tp_cpu_group: "ProcessGroup",
        max_bytes: int = _MAX_BYTES,
    ) -> "P2POneShotAllReducer | None":
        """Build the reducer, or return None (logged) leaving NCCL in place.

        Never raises. The collective sequence is FIXED (payload exchange, then
        verdict exchange) no matter where -- or whether -- a rank fails locally,
        so a one-sided failure can never strand the peer in a collective. The
        verdict covers a real cross-GPU write probe: IPC can open cleanly on
        topologies whose P2P data plane black-holes remote writes (observed on a
        2-GPU PCIe box across host bridges), which would otherwise hang the first
        spin-wait kernel forever.
        """
        if tp_info.size != 2:
            return None  # v1 protocol is pairwise; larger worlds keep NCCL
        if torch.cuda.device_count() < 2:
            return None
        try:
            return cls._build(tp_info, tp_cpu_group, min(max_bytes, _MAX_BYTES))
        except Exception as exc:  # noqa: BLE001 -- any gap must degrade, not kill boot
            logger.warning(f"P2P one-shot all-reduce unavailable ({exc}); keeping NCCL")
            return None

    @classmethod
    def _build(cls, tp_info, tp_cpu_group, max_bytes: int):
        from cuda.bindings import runtime as rt

        rank = tp_info.rank
        device = torch.cuda.current_device()
        block_bytes = _FLAG_BYTES + max_bytes

        def _ck(res):
            err = res[0] if isinstance(res, tuple) else res
            if int(err) != 0:
                raise RuntimeError(str(err))
            return res[1] if isinstance(res, tuple) and len(res) > 1 else None

        # ---- stage 1: allocate + export, ALWAYS reaching the payload exchange
        my_ptr = counter_ptr = peer_ptr = None
        handle = None
        try:
            # zeroed flag+data block and a private launch counter, outside the
            # torch pool (expandable_segments' VMM memory cannot be IPC-exported)
            my_ptr = _ck(rt.cudaMalloc(block_bytes))
            _ck(rt.cudaMemset(my_ptr, 0, block_bytes))
            counter_ptr = _ck(rt.cudaMalloc(8))
            _ck(rt.cudaMemset(counter_ptr, 0, 8))
            handle = _ck(rt.cudaIpcGetMemHandle(my_ptr))
            payload = (True, device, _ipc_handle_bytes(handle))
        except Exception as exc:  # noqa: BLE001
            payload = (False, device, None)
            local_err: Exception | None = exc
        else:
            local_err = None
        gathered: list = [None, None]
        torch.distributed.all_gather_object(gathered, payload, group=tp_cpu_group)
        peer_ok, peer_dev, peer_handle_bytes = gathered[1 - rank]

        # ---- stage 2: open + functional probe, ALWAYS reaching the verdict exchange
        verdict = False
        if local_err is None and peer_ok and peer_dev != device:
            try:
                peer_ptr = int(_ck(
                    rt.cudaIpcOpenMemHandle(
                        _ipc_handle_from_bytes(peer_handle_bytes),
                        rt.cudaIpcMemLazyEnablePeerAccess,
                    )
                ))
                verdict = cls._probe_data_plane(rank, int(my_ptr), peer_ptr)
                if not verdict:
                    local_err = RuntimeError("P2P write probe: remote stores do not land")
            except Exception as exc:  # noqa: BLE001
                local_err = exc
                verdict = False
        elif local_err is None:
            local_err = RuntimeError(f"peer verdict {peer_ok!r} on device {peer_dev}")
        verdicts: list = [None, None]
        torch.distributed.all_gather_object(verdicts, verdict, group=tp_cpu_group)
        if all(verdicts):
            return cls(
                rank, int(my_ptr), peer_ptr, int(counter_ptr), max_bytes,
                owns=(my_ptr, counter_ptr, handle, peer_handle_bytes),
            )
        # ---- unanimous decline: release whatever this rank mapped
        try:
            if peer_ptr is not None:
                rt.cudaIpcCloseMemHandle(peer_ptr)
            if my_ptr is not None:
                rt.cudaFree(my_ptr)
            if counter_ptr is not None:
                rt.cudaFree(counter_ptr)
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(str(local_err or "peer declined the P2P path"))

    @staticmethod
    def _probe_data_plane(rank: int, my_ptr: int, peer_ptr: int) -> bool:
        """Cross-GPU round trip: each rank stores a magic value into the PEER's
        data region, barrier, then reads its own region back. True only when the
        remote store actually landed."""
        from cuda.bindings import runtime as rt

        magic = 0x5032_0000 | rank
        _p2p_probe_store[(1,)](peer_ptr + _FLAG_BYTES, magic, num_warps=1)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        host = torch.zeros(1, dtype=torch.int64)
        err, = rt.cudaMemcpy(
            host.data_ptr(), my_ptr + _FLAG_BYTES, 8,
            rt.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        if int(err) != 0:
            return False
        got = int(host[0])
        want = 0x5032_0000 | (1 - rank)
        ok = got == want
        if not ok:
            logger.warning(
                f"P2P probe rank{rank}: expected {want:#x} from the peer, read {got:#x}"
            )
        # leave the flag region clean for the real protocol
        err2, = rt.cudaMemset(my_ptr + _FLAG_BYTES, 0, 8)
        return ok and int(err2) == 0

    # -------------------------------------------------------------------- run

    def all_reduce_(self, x: torch.Tensor) -> torch.Tensor:
        n = x.numel()
        assert n <= self.max_elems and x.dtype == torch.bfloat16 and x.is_contiguous()
        _p2p_one_shot_sum_bf16[(1,)](
            x,
            x,
            self._my_ptr + _FLAG_BYTES,
            self._peer_ptr + _FLAG_BYTES,
            self._my_ptr,
            self._peer_ptr,
            self._counter_ptr,
            n,
            BLOCK=_BLOCK,
            num_warps=32,
        )
        return x

    def close(self) -> None:
        from cuda.bindings import runtime as rt

        my_ptr, counter_ptr = self._owns[0], self._owns[1]
        rt.cudaIpcCloseMemHandle(self._peer_ptr)
        rt.cudaFree(my_ptr)
        rt.cudaFree(counter_ptr)


def p2p_ar_enabled() -> bool:
    return os.getenv("FREETOKEN_P2P_AR", "1").strip().lower() not in {"0", "false", "no", "off"}
