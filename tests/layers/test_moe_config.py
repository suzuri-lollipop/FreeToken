"""MoEConfig's block-aligned fp8 TP shard (the MTP expert placement prerequisite).

fp8 128x128 block scales can never straddle two ranks, so the intermediate dim
shards by whole blocks -- possibly unevenly (the Qwen4Exp MTP head's 640 rows =
5 blocks split 256/384 over TP2). Every other quant kind keeps the even split.
"""

import pytest

from freetoken.layers.quantization.moe.base import MoEConfig, block_aligned_range
from freetoken.layers.quantization.scheme import fp8_block_scheme, nvfp4_scheme


def _cfg(scheme, intermediate=640, rank=0, size=1):
    return MoEConfig(
        num_experts=8, hidden=2560, intermediate=intermediate, top_k=10,
        tp_rank=rank, tp_size=size, scheme=scheme,
    )


def test_block_aligned_range_covers_every_row_exactly_once():
    assert [block_aligned_range(640, 128, r, 2) for r in range(2)] == [(0, 256), (256, 640)]
    assert [block_aligned_range(640, 128, r, 4) for r in range(4)] == [
        (0, 128), (128, 256), (256, 384), (384, 640)]
    # an even block count stays a uniform split
    assert [block_aligned_range(512, 128, r, 2) for r in range(2)] == [(0, 256), (256, 512)]
    assert block_aligned_range(640, 128, 0, 1) == (0, 640)
    with pytest.raises(AssertionError):
        block_aligned_range(641, 128, 0, 1)


def test_local_intermediate_is_block_aligned_only_for_fp8_block():
    fb = fp8_block_scheme("bf16")
    assert _cfg(fb, rank=0, size=2).local_intermediate == 256
    assert _cfg(fb, rank=1, size=2).local_intermediate == 384
    assert _cfg(fb, rank=0, size=1).local_intermediate == 640
    nv = nvfp4_scheme(input_scale=True)
    assert _cfg(nv, rank=0, size=2).local_intermediate == 320
    assert _cfg(nv, rank=1, size=2).local_intermediate == 320


def test_fp8_block_kernel_accepts_tp_with_a_block_multiple_intermediate():
    from freetoken.layers.quantization.moe.fp8_block import TritonFp8BlockMoEKernel

    kernel = TritonFp8BlockMoEKernel()
    assert kernel.unusable_reason(_cfg(fp8_block_scheme("bf16"), rank=1, size=2)) is None
    bad = _cfg(fp8_block_scheme("bf16"), intermediate=600, rank=0, size=2)
    reason = kernel.unusable_reason(bad)
    assert reason is not None and "multiple of 128" in reason
