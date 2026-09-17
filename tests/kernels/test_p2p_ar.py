"""P2P one-shot all-reduce kernel semantics (loopback: peer == self).

The cross-process IPC protocol itself is exercised end-to-end by the TP engine
(two ranks); here the kernel's math, masking, sequence-counter continuity and
CUDA-graph capturability are pinned without a second GPU: with peer == self
the release/acquire flag round-trip degenerates to same-device ordering and
the sum is exactly x + x.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.distributed.p2p_ar import _FLAG_BYTES, P2POneShotAllReducer

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _loopback_reducer(max_bytes: int = 32 * 1024) -> P2POneShotAllReducer:
    buf = torch.zeros((max_bytes // 2 + 64,), dtype=torch.bfloat16, device="cuda")
    counter = torch.zeros(1, dtype=torch.int64, device="cuda")
    ptr = buf.data_ptr()
    return P2POneShotAllReducer(
        rank=0,
        my_ptr=ptr,
        peer_ptr=ptr,  # loopback
        counter_ptr=counter.data_ptr(),
        max_bytes=max_bytes,
        owns=(buf, counter),
    )


@CUDA
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_loopback_sum_matches_double(rows):
    ar = _loopback_reducer()
    x = torch.randn(rows, 2560, dtype=torch.bfloat16, device="cuda")
    expected = (x.float() * 2).to(torch.bfloat16)
    out = ar.all_reduce_(x)
    torch.cuda.synchronize()
    assert out.data_ptr() == x.data_ptr()  # in-place contract
    assert torch.equal(x, expected)


@CUDA
def test_loopback_repeats_keep_counter_in_step():
    ar = _loopback_reducer()
    x = torch.randn(1, 2560, dtype=torch.bfloat16, device="cuda")
    last = 0.0
    for i in range(50):
        last = float(i % 7 + 1)
        x.fill_(last)
        ar.all_reduce_(x)
    torch.cuda.synchronize()
    assert torch.equal(x, torch.full_like(x, last * 2))
    assert int(ar._owns[1][0]) == 50


@CUDA
def test_loopback_survives_cuda_graph_capture_and_replay():
    ar = _loopback_reducer()
    x = torch.randn(4, 2560, dtype=torch.bfloat16, device="cuda")
    # eager warmup (matches the engine's capture flow)
    ar.all_reduce_(x.clone())
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        ar.all_reduce_(x)  # eager pre-run on the capture stream's device context
    torch.cuda.current_stream().wait_stream(stream)
    x.normal_()
    frozen = (x.float() * 2).to(torch.bfloat16)
    with torch.cuda.graph(graph):
        ar.all_reduce_(x)
    x.zero_()
    x.copy_((frozen.float() / 2).to(torch.bfloat16))
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    # every replay doubles in place; 10 replays of v -> v * 2^10 would overflow
    # bf16 range only for large v -- assert the exact chain instead
    v = (frozen.float() / 2).to(torch.bfloat16)
    for _ in range(10):
        v = (v.float() * 2).to(torch.bfloat16)
    assert torch.equal(x, v.expand_as(x))


@CUDA
def test_impl_falls_back_above_threshold_and_other_dtypes():
    from freetoken.distributed.impl import P2POneShotDistributedImpl

    seen = []

    class _Recorder:
        def all_reduce(self, t):
            seen.append(t.numel() * t.element_size())
            return t

        def all_gather(self, t):
            raise AssertionError

    ar = _loopback_reducer(max_bytes=4096)
    impl = P2POneShotDistributedImpl(inner=_Recorder(), ar=ar, max_elems=ar.max_elems)
    small = torch.ones(1, 1024, dtype=torch.bfloat16, device="cuda")  # 2 KiB
    impl.all_reduce(small)
    assert not seen and torch.equal(small, torch.full_like(small, 2.0))
    big = torch.ones(1, 4096, dtype=torch.bfloat16, device="cuda")  # 8 KiB > cap
    impl.all_reduce(big)
    assert seen == [8192]
    f32 = torch.ones(4, dtype=torch.float32, device="cuda")
    impl.all_reduce(f32)
    assert seen == [8192, 16]
