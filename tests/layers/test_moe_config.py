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


# ---------------------------------------------------------------------------
# Uneven (bandwidth-weighted) nvfp4 expert shard: a slow PCIe link banks fewer
# bytes per expert so its slot cache holds more experts and its per-step miss
# traffic shrinks (engine resolves the weights from the live H2D probe).
# ---------------------------------------------------------------------------

from freetoken.layers.quantization.moe.base import weighted_block_ranges  # noqa: E402
from freetoken.layers.quantization.moe.nvfp4 import tp_piece, tp_rows  # noqa: E402


def _wcfg(scheme=None, intermediate=640, hidden=2560, rank=0, size=2,
          weights=(0.6, 0.4), strategy="offload", decode_target="gpu"):
    from freetoken.layers.quantization.scheme import nvfp4_scheme

    return MoEConfig(
        num_experts=8, hidden=hidden, intermediate=intermediate, top_k=10,
        tp_rank=rank, tp_size=size, scheme=scheme or nvfp4_scheme(input_scale=True),
        strategy=strategy, decode_target=decode_target, shard_weights=weights,
    )


def test_weighted_block_ranges_are_proportional_and_exact():
    assert weighted_block_ranges(640, 16, (0.6, 0.4)) == [(0, 384), (384, 640)]
    assert weighted_block_ranges(640, 16, (0.4, 0.6)) == [(0, 256), (256, 640)]
    # largest-remainder: 24.8/15.2 groups -> the bigger fraction gets the spare group
    assert weighted_block_ranges(640, 16, (0.62, 0.38)) == [(0, 400), (400, 640)]
    # three ranks, 40 groups: 13.33 each, remainder group to rank order on ties
    r = weighted_block_ranges(640, 16, (1.0, 1.0, 1.0))
    assert r == [(0, 224), (224, 432), (432, 640)]
    # deterministic: same input, same output
    assert weighted_block_ranges(640, 16, (0.6, 0.4)) == [(0, 384), (384, 640)]


def test_weighted_block_ranges_reject_bad_grain():
    assert weighted_block_ranges(641, 16, (0.6, 0.4)) is None      # not block-aligned
    assert weighted_block_ranges(16, 16, (0.6, 0.4)) is None      # fewer groups than ranks
    assert weighted_block_ranges(640, 16, (0.0, 1.0)) is None     # non-positive weight
    # a rank that would receive zero groups falls back to None (caller splits evenly)
    assert weighted_block_ranges(48, 16, (0.999, 0.001)) is None


def test_uneven_range_only_for_offloaded_nvfp4_tp():
    from freetoken.layers.quantization.scheme import fp8_block_scheme, nvfp4_scheme

    nv = nvfp4_scheme(input_scale=True)
    assert _wcfg(nv, rank=0).local_intermediate_range == (0, 384)
    assert _wcfg(nv, rank=1).local_intermediate_range == (384, 640)
    assert _wcfg(nv, rank=0).local_intermediate == 384
    assert _wcfg(nv, rank=1).local_intermediate == 256
    # every gate that must keep the even split
    assert _wcfg(nv, rank=0, strategy="resident").local_intermediate == 320
    assert _wcfg(nv, rank=0, strategy="hybrid").local_intermediate == 320
    assert _wcfg(nv, rank=0, decode_target="cpu").local_intermediate == 320
    assert _wcfg(nv, rank=0, weights=None).local_intermediate == 320
    assert _wcfg(nv, rank=0, weights=(1.0,)).local_intermediate == 320
    assert _wcfg(nv, size=1, rank=0).local_intermediate == 640
    # fp8-block keeps its own block-aligned split regardless of the weights
    fb = fp8_block_scheme("bf16")
    assert _wcfg(fb, rank=0).local_intermediate_range == (0, 256)
    assert _wcfg(fb, rank=1).local_intermediate_range == (256, 640)


def test_uneven_tp_shard_flag():
    from freetoken.layers.quantization.scheme import nvfp4_scheme

    nv = nvfp4_scheme(input_scale=True)
    assert _wcfg(nv, rank=0).uneven_tp_shard is True
    assert _wcfg(nv, rank=0, weights=None).uneven_tp_shard is False
    # equal weights resolve to the even split -> not uneven
    assert _wcfg(nv, rank=0, weights=(0.5, 0.5)).uneven_tp_shard is False


def test_tp_piece_and_rows_take_the_range():
    import torch

    total = 640
    # plain row piece ([1, I, k]) narrows to [lo, hi)
    p = torch.arange(total, dtype=torch.int32).reshape(1, total, 1)
    assert torch.equal(tp_piece(p, 1, 384, 640, total).reshape(-1),
                       torch.arange(384, 640, dtype=torch.int32))
    # a scale grid thinner by the group slices in its own units
    s = torch.arange(32 * 40, dtype=torch.int32).reshape(1, 32, 40)
    assert torch.equal(tp_piece(s, 2, 384, 640, total), s[:, :, 24:40])
    # fused [gate | up] rows cut BOTH halves
    f = torch.arange(2 * total, dtype=torch.int32).reshape(1, 2 * total, 1)
    got = tp_rows(f, 1, 384, 640, total).reshape(-1)
    assert torch.equal(got, torch.cat([torch.arange(384, 640),
                                       torch.arange(total + 384, total + 640)]).to(torch.int32))
    # something that does not tile `total` is returned whole (per-tensor globals)
    g = torch.ones(1, 1)
    assert torch.equal(tp_piece(g, 1, 384, 640, total), g)


def test_pack_slices_the_uneven_range_and_the_ranks_cover_the_expert():
    import torch

    from freetoken.layers.quantization.moe.nvfp4 import TritonNvfp4MoEKernel

    FP8 = torch.float8_e4m3fn
    kernel = TritonNvfp4MoEKernel()
    I, H = 64, 32  # 4 scale groups of 16 rows
    g = torch.Generator().manual_seed(7)

    def rnd(*shape, dtype=torch.uint8):
        if dtype == torch.uint8:
            return torch.randint(0, 255, shape, generator=g, dtype=torch.uint8)
        return torch.randn(shape, generator=g).to(dtype)

    pieces = {
        "gate": rnd(1, I, H // 2), "up": rnd(1, I, H // 2),
        "gate_scale": rnd(1, I, H // 16, dtype=FP8), "up_scale": rnd(1, I, H // 16, dtype=FP8),
        "gate_global": rnd(1, 1, dtype=torch.float16), "up_global": rnd(1, 1, dtype=torch.float16),
        "down": rnd(1, H, I // 2), "down_scale": rnd(1, H, I // 16, dtype=FP8),
        "down_global": rnd(1, 1, dtype=torch.float16),
    }

    def packed(rank, size, weights):
        cfg = _wcfg(intermediate=I, hidden=H, rank=rank, size=size, weights=weights)
        layout = kernel.layout(cfg)
        out = {role: torch.empty((1, *spec.shape), dtype=spec.dtype)
               for role, spec in layout.items() if not spec.resident}
        kernel.pack({k: v.clone() for k, v in pieces.items()}, cfg, out)
        return cfg, out

    # weights (0.75, 0.25) over 4 groups -> rank0 rows [0,48), rank1 rows [48,64)
    cfg0, r0 = packed(0, 2, (0.75, 0.25))
    cfg1, r1 = packed(1, 2, (0.75, 0.25))
    assert (cfg0.local_intermediate, cfg1.local_intermediate) == (48, 16)
    _, full = packed(0, 1, None)

    # gate rows, up rows (fused), scales and the down column slice all match the
    # full-expert pack at the rank's row range
    lo0, hi0, lo1, hi1 = 0, 48, 48, 64
    assert torch.equal(r0["gate_up"][0, :hi0], full["gate_up"][0, lo0:hi0])
    assert torch.equal(r0["gate_up"][0, hi0:], full["gate_up"][0, I + lo0:I + hi0])
    assert torch.equal(r1["gate_up"][0, :16], full["gate_up"][0, lo1:hi1])
    assert torch.equal(r1["gate_up"][0, 16:], full["gate_up"][0, I + lo1:I + hi1])
    assert torch.equal(r0["gate_up_scale"][0, :hi0], full["gate_up_scale"][0, lo0:hi0])
    assert torch.equal(r1["gate_up_scale"][0, :16], full["gate_up_scale"][0, lo1:hi1])
    assert torch.equal(r0["down"], full["down"][:, :, : hi0 // 2])
    assert torch.equal(r1["down"], full["down"][:, :, lo1 // 2:])
    assert torch.equal(r0["down_scale"], full["down_scale"][:, :, : hi0 // 16])
    assert torch.equal(r1["down_scale"], full["down_scale"][:, :, lo1 // 16:])
    # per-row global scales broadcast to the local row count
    assert r0["gate_up_global"].shape == (1, 2 * 48)
    assert r1["gate_up_global"].shape == (1, 2 * 16)
    assert r0["down_global"].shape == (1, H)
