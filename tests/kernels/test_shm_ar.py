"""SHM one-shot all-reduce kernel mechanics (single-GPU self-loop).

The cross-rank protocol needs two processes; here the two kernels are driven
against ONE mapped-host block with the peer aliased to self, which exercises
everything device-side: the seq counter, parity slot alternation, the
system-scope release/acquire flag handoff and CUDA-graph capturability. The
full two-rank protocol (bit-identity vs ncclSum, stall-free SPMD lockstep) is
validated by the engine boot probe on TP=2 servers.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.distributed.shm_ar import (
    _BLOCK,
    _DATA_OFF,
    _MAX_BYTES,
    _R_OFF,
    _SHM_SIZE,
    _shm_ar_stage_kernel,
    _shm_ar_sum_kernel,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _rig():
    """A pinned host block standing in for the /dev/shm mapping + device counter."""
    host = torch.zeros(_SHM_SIZE // 8, dtype=torch.int64, pin_memory=True)
    ctr = torch.zeros(1, dtype=torch.int64, device="cuda")
    return host, ctr


def _round(x, host, ctr, rank=0, peer=0):
    """One all-reduce round against the self-looped block (peer == self)."""
    base = host.data_ptr()
    n = x.numel()
    _shm_ar_stage_kernel[(1,)](
        x, base + _DATA_OFF + rank * 2 * _MAX_BYTES, base + _R_OFF + 8 * rank,
        ctr.data_ptr(), n, slot_bytes=_MAX_BYTES, BLOCK=_BLOCK, num_warps=32,
    )
    _shm_ar_sum_kernel[(1,)](
        x, base + _DATA_OFF + peer * 2 * _MAX_BYTES, base + _R_OFF + 8 * peer,
        ctr.data_ptr(), n, slot_bytes=_MAX_BYTES, BLOCK=_BLOCK, num_warps=32,
    )


@CUDA
def test_self_loop_doubles_and_alternates_parity():
    host, ctr = _rig()
    for it in range(8):
        x = torch.full((1, 2560), float(it + 1), dtype=torch.bfloat16, device="cuda")
        _round(x, host, ctr)
        torch.cuda.synchronize()
        # self-loop: x + x, with the seq/parity advancing every round
        assert torch.equal(x, torch.full_like(x, 2.0 * (it + 1))), it
        assert int(ctr) == it + 1
        slot = int(ctr) % 2
        staged = host.view(torch.uint8)[_DATA_OFF + slot * _MAX_BYTES:]
        # the staged bytes are the PRE-sum value (bf16 it+1)
        first = staged[:2].view(torch.bfloat16).item()
        assert first == float(it + 1)


@CUDA
def test_seq_flag_blocks_until_published():
    """The sum kernel's acquire-spin only passes on the staged seq: run sum
    first with the flag behind and confirm it waits (we publish from the host
    side mid-spin), proving the wait is real, not incidental."""
    host, ctr = _rig()
    x = torch.randn(2, 2560, dtype=torch.bfloat16, device="cuda")
    base = host.data_ptr()
    n = x.numel()
    # bump the counter WITHOUT staging/publishing, then launch sum: it must spin
    ctr.fill_(1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _shm_ar_sum_kernel[(1,)](
            x, base + _DATA_OFF, base + _R_OFF, ctr.data_ptr(), n,
            slot_bytes=_MAX_BYTES, BLOCK=_BLOCK, num_warps=32,
        )
    import time

    time.sleep(0.2)
    assert not stream.query(), "sum must still be spinning on the unpublished seq"
    # publish seq=1 from the host (mapped memory is host-writable)
    host_view = host.numpy()
    host_view[0] = 1  # R0 = seq
    stream.synchronize()  # now it completes
    assert int(ctr) == 1


@CUDA
def test_graph_capture_replays_alternate_slots():
    host, ctr = _rig()
    x = torch.randn(4, 2560, dtype=torch.bfloat16, device="cuda")
    _round(x, ctr=ctr, host=host)  # eager warm (compiles + primes)
    torch.cuda.synchronize()
    x0 = x.clone()
    graph = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(graph):
            _round(x, host, ctr)
    torch.cuda.current_stream().wait_stream(s)
    x.copy_(x0)
    ctr.fill_(1)  # the captured round continues from seq 1
    for i in range(5):
        graph.replay()
        torch.cuda.synchronize()
        # each replay doubles in place (self-loop): x *= 2, seq advances
        x0 = (x0.float() * 2).to(torch.bfloat16)
        assert torch.equal(x, x0), i
        assert int(ctr) == 2 + i
