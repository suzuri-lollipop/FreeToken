"""Two-rank one-shot all-reduce over a shared host-memory rendezvous.

NCCL's LL ring costs ~26-37us per 5-20 KiB all-reduce on a two-GPU PCIe box;
a decode step pays it twice per layer (~97 calls -> the single largest latency
item after the dense-weight GEMVs). Front-end stream memops do NOT beat it
(measured: their wait-wakeup granularity puts the round trip at the same
~26us), so this path spins IN-KERNEL on host-mapped flags (~1us PCIe poll
granularity) and keeps the whole exchange to two tiny single-CTA kernels:

    K1  seq = ++ctr (device counter, replay-safe)
        store x -> my_data[seq % 2]        (mapped host memory, own PCIe link)
        bar + system-release: R_me = seq   (flag carries the SEQ, not a bit:
                                            level polls can never miss or
                                            double-consume a pulse)
    K2  spin until R_peer >= seq (acquire, ~1us polls of host memory)
        out = x + peer_data[seq % 2]       (fp32 add of two bf16, in-place;
                                            bit-identical to ncclSum over two
                                            ranks)

Parity double-buffering makes the consumed handshake unnecessary: a producer's
K1(N+2) (reusing slot N%2) is stream-ordered behind its own K2(N+1), which
passed R_peer >= N+1 -- set by the peer's K1(N+1), itself ordered after the
peer's K2(N) that read my slot N%2. Seq-numbered flags need no ack/reset.

Both ranks execute the same call sequence (SPMD), so the device counters and
host flags advance in lockstep; the counter lives in device memory, hence the
kernel sequence is CUDA-graph capturable and each replay alternates slots
correctly. No GPU<->GPU P2P is used (some PCIe topologies accept the IPC
mapping yet black-hole remote writes -- see p2p_ar's probe); host memory is
the rendezvous, reachable from both GPUs over their own links.

Set FREETOKEN_SHM_AR=0 to keep plain NCCL.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import time
from typing import TYPE_CHECKING

import torch
import triton
import triton.language as tl

from freetoken.utils import init_logger

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from .info import DistributedInfo

logger = init_logger(__name__)

_MAX_BYTES = 32 * 1024
_BLOCK = 16384  # bf16 elements per CTA pass; n <= 16384 -> single pass
# Host block layout: [R0 R1][ctr pad][data0 (2 x 32KB parity slots... )]
# flags at word 0/1; the device seq counter is a separate cudaMalloc'd word.
_R_OFF = 0            # int64 words: R0 at +0, R1 at +8
_DATA_OFF = 4096
_SHM_SIZE = _DATA_OFF + 4 * _MAX_BYTES  # 2 ranks x 2 parity slots


@triton.jit
def _shm_ar_stage_kernel(
    x_ptr, data_addr, r_addr, ctr_addr, n,
    slot_bytes: tl.constexpr, BLOCK: tl.constexpr,
):
    """Single CTA: bump seq, stage x into my_data[seq%2], release-signal R_me=seq."""
    seq = tl.load(ctr_addr.to(tl.pointer_type(tl.int64))) + 1
    tl.store(ctr_addr.to(tl.pointer_type(tl.int64)), seq)
    slot = (seq % 2) * slot_bytes
    dst = (data_addr + slot).to(tl.pointer_type(tl.bfloat16))
    offs = tl.arange(0, BLOCK)
    m = offs < n
    v = tl.load(x_ptr + offs, mask=m)
    tl.store(dst + offs, v, mask=m)
    # CTA-wide ordering, then one system-scope release publish of the seq flag
    tl.debug_barrier()
    lane0 = tl.arange(0, 1)
    tl.atomic_xchg(r_addr.to(tl.pointer_type(tl.int64)) + lane0, seq,
                   sem="release", scope="sys")


@triton.jit
def _shm_ar_sum_kernel(
    x_ptr, peer_data_addr, r_peer_addr, ctr_addr, n,
    slot_bytes: tl.constexpr, BLOCK: tl.constexpr,
):
    """Single CTA: wait for the peer's seq, then out = x + peer_data[seq%2]."""
    seq = tl.load(ctr_addr.to(tl.pointer_type(tl.int64)))
    rp = r_peer_addr.to(tl.pointer_type(tl.int64))
    while tl.atomic_add(rp, 0, sem="acquire", scope="sys") < seq:
        pass
    slot = (seq % 2) * slot_bytes
    peer = (peer_data_addr + slot).to(tl.pointer_type(tl.bfloat16))
    offs = tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m).to(tl.float32)
    y = tl.load(peer + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(x_ptr + offs, (x + y).to(tl.bfloat16), mask=m)


class ShmOneShotAllReducer:
    def __init__(self, rank: int, base: int, counter: torch.Tensor, owns: tuple) -> None:
        self.rank = rank
        self._base = base  # device-visible alias of the mapped shm block
        self._counter = counter  # device int64 [1]: the seq source (graph-safe)
        self.max_bytes = _MAX_BYTES
        self.max_elems = _MAX_BYTES // 2
        self._owns = owns  # fd/mmap/host_addr kept alive for the process lifetime

    # ------------------------------------------------------------------ build

    @classmethod
    def try_build(
        cls,
        tp_info: "DistributedInfo",
        tp_cpu_group: "ProcessGroup",
    ) -> "ShmOneShotAllReducer | None":
        """Build + functionally probe, or return None (logged) leaving NCCL.

        The collective sequence is FIXED (name broadcast, build-verdict gather,
        barrier, probe-verdict gather) and runs identically on both ranks
        whatever fails locally, so a one-sided failure can never strand the
        peer in a collective. The probe has a wall-clock deadline: a protocol
        stall degrades to "unavailable" instead of hanging the boot.
        """
        if tp_info.size != 2:
            return None
        try:
            return cls._build(tp_info, tp_cpu_group)
        except Exception as exc:  # noqa: BLE001 -- degrade, never kill boot
            logger.warning(f"shm one-shot all-reduce unavailable ({exc}); keeping NCCL")
            return None

    @classmethod
    def _build(cls, tp_info, tp_cpu_group):
        from cuda.bindings import runtime as rt

        rank = tp_info.rank
        device = torch.cuda.current_device()

        # ---- collective 1: agree on the shm name
        name = [f"/freetoken_shm_ar_{os.getpid()}_{device}" if rank == 0 else None]
        torch.distributed.broadcast_object_list(name, src=0, group=tp_cpu_group)
        path = "/dev/shm" + name[0]
        keep = os.getenv("FREETOKEN_SHM_AR_KEEP", "0") == "1"

        local = None
        err: Exception | None = None
        try:
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            os.ftruncate(fd, _SHM_SIZE)
            mm = mmap.mmap(fd, _SHM_SIZE)
            host_addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
            if rank == 0:
                words = (ctypes.c_int64 * 2).from_address(host_addr + _R_OFF)
                words[0] = 0
                words[1] = 0
            res = rt.cudaHostRegister(host_addr, _SHM_SIZE, 0x3)  # Mapped|Portable
            if int(res[0] if isinstance(res, tuple) else res) != 0:
                raise RuntimeError(f"cudaHostRegister: {res}")
            dres = rt.cudaHostGetDevicePointer(host_addr, 0)
            if int(dres[0]) != 0:
                raise RuntimeError(f"cudaHostGetDevicePointer: {dres[0]}")
            counter = torch.zeros(1, dtype=torch.int64, device=f"cuda:{device}")
            local = cls(rank, int(dres[1]), counter, owns=(fd, mm, host_addr))
            # Default to single-token (bs=1) reductions: on a 2-GPU PCIe box the
            # bs>=2 decode graphs also carry CPU-MoE front-end WAIT nodes, and a
            # graph that mixes those with the resident K2 spin kernel reproducibly
            # blocks the NEXT cudaGraphLaunch of that exec (driver-level; bisected
            # via FREETOKEN_SHM_AR_MAX_KIB: 5 KiB = bs1-only is stable, >=10 KiB
            # hangs the first bs>=2 step). bs=1 is also where the win is largest
            # (the step is shortest, so AR latency dominates). Raise the knob only
            # on stacks where the mixed-graph launch behaves.
            kib = int(os.getenv("FREETOKEN_SHM_AR_MAX_KIB", "5") or "5")
            local.max_bytes = min(_MAX_BYTES, max(4, kib) * 1024)
            local.max_elems = local.max_bytes // 2
        except Exception as exc:  # noqa: BLE001
            err = exc

        # ---- collective 2: build verdict (both ranks always reach it)
        verdicts: list = [None, None]
        torch.distributed.all_gather_object(verdicts, err is None, group=tp_cpu_group)
        if not all(verdicts):
            if local is not None:
                local._teardown()
            if rank == 0:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            raise RuntimeError(err or "peer failed the shm setup")
        # both ranks mapped the block; the name can go (mappings survive unlink)
        if rank == 0 and not keep:
            try:
                os.unlink(path)
            except OSError:
                pass
        if keep:
            logger.info(f"shm all-reduce block kept at {path} (debug)")

        # ---- collective 3: rank0's flag init visible before any probe op
        torch.distributed.barrier(group=tp_cpu_group)

        # ---- functional probe: two real rounds (parity alternation), deadline-
        # bounded, host-verified. A protocol stall must degrade, not hang boot.
        ok = False
        stalled = False
        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            done = torch.cuda.Event()
            xs = []
            with torch.cuda.stream(side):
                for v in (1.0, 2.0):
                    x = torch.full((1, 128), v, dtype=torch.bfloat16,
                                   device=f"cuda:{device}") * (rank + 1)
                    local.all_reduce_(x)
                    xs.append((x, v * 3.0))
                done.record(side)
            deadline = time.monotonic() + 15.0
            while not done.query():
                if time.monotonic() > deadline:
                    stalled = True
                    raise RuntimeError("probe rounds did not complete (protocol stall)")
                time.sleep(0.005)
            ok = all(bool(torch.all(x == want).item()) for x, want in xs)
        except Exception as exc:  # noqa: BLE001
            err = exc
        # ---- collective 4: probe verdict
        probe_verdicts: list = [None, None]
        torch.distributed.all_gather_object(probe_verdicts, ok, group=tp_cpu_group)
        if not all(probe_verdicts):
            if stalled:
                # A parked spin kernel may still reference the registered
                # mapping: leaking it (process lifetime) is the only safe
                # teardown; unregistering under a live kernel is undefined.
                logger.error(
                    "shm all-reduce probe STALLED; leaving the mapping registered "
                    "and falling back to NCCL"
                )
            else:
                local._teardown()
            raise RuntimeError(f"shm all-reduce probe failed ({err})")
        logger.info_rank0(
            "shm one-shot all-reduce probe OK (host rendezvous, in-kernel seq flags)"
        )
        return local

    def _teardown(self) -> None:
        from cuda.bindings import runtime as rt

        fd, mm, host_addr = self._owns
        try:
            rt.cudaHostUnregister(host_addr)
        except Exception:  # noqa: BLE001
            pass
        mm.close()
        os.close(fd)

    # -------------------------------------------------------------------- run

    def all_reduce_(self, x: torch.Tensor) -> torch.Tensor:
        n = x.numel()
        assert n <= self.max_elems and x.dtype == torch.bfloat16 and x.is_contiguous()
        peer = 1 - self.rank
        base = self._base
        my_data = base + _DATA_OFF + self.rank * 2 * _MAX_BYTES
        peer_data = base + _DATA_OFF + peer * 2 * _MAX_BYTES
        ctr = self._counter
        _shm_ar_stage_kernel[(1,)](
            x, my_data, base + _R_OFF + 8 * self.rank, ctr.data_ptr(), n,
            slot_bytes=_MAX_BYTES, BLOCK=_BLOCK, num_warps=32,
        )
        _shm_ar_sum_kernel[(1,)](
            x, peer_data, base + _R_OFF + 8 * peer, ctr.data_ptr(), n,
            slot_bytes=_MAX_BYTES, BLOCK=_BLOCK, num_warps=32,
        )
        return x


def shm_ar_enabled() -> bool:
    return os.getenv("FREETOKEN_SHM_AR", "1").strip().lower() not in {"0", "false", "no", "off"}
