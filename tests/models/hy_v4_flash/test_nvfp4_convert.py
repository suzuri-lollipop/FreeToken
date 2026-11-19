"""NVFP4 conversion guards.

The round-trip tests pin the converter's packing / scale convention against the
engine's dequant kernel (a nibble-order or global-scale drift would still produce
plausible-looking numbers): random weights, the e2m1 tie rules, zero blocks, and
the fp8 block dequant the attention projections go through.

The serving smoke test runs a converted checkpoint through the offload expert
path and the disk-streamed engram tables (env-gated, needs_weights).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from .common import requires_cuda

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts"))
from convert_hy4_fp8_to_nvfp4 import dequant_fp8_block, dequant_nvfp4, quantize_nvfp4  # noqa: E402

pytestmark = requires_cuda


def _engine_dequant(packed, scale, g):
    from freetoken.kernel.triton.nvfp4_dequant import dequant_nvfp4 as kernel

    out = kernel(
        packed.unsqueeze(0), scale.unsqueeze(0),
        g.reshape(1).expand(packed.shape[0]).contiguous().unsqueeze(0),
        torch.zeros(1, dtype=torch.int32, device=packed.device),
    )
    return out[0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_roundtrip_matches_engine_kernel(seed):
    torch.manual_seed(seed)
    w = (torch.randn(128, 256, device="cuda") * 0.05).to(torch.bfloat16)
    packed, scale, g = quantize_nvfp4(w)
    ref = dequant_nvfp4(packed, scale, g).to(torch.bfloat16)
    eng = _engine_dequant(packed, scale, g)
    torch.testing.assert_close(ref, eng, atol=2e-3, rtol=2e-3)
    # NVFP4 error stays bounded relative to the tensor amax
    err = (ref.float() - w.float()).abs().max().item()
    assert err < 0.15 * w.float().abs().max().item()


def test_low_nibble_first():
    # two adjacent elements with distinct codes: the byte must read lo=even element
    w = torch.zeros(1, 16, device="cuda", dtype=torch.bfloat16)
    w[0, 0] = 6.0  # code 7
    w[0, 1] = 0.5  # code 1
    packed, scale, g = quantize_nvfp4(w)
    # global = amax/(448*6); block scale = amax_block/(6*g) = 448 -> codes exact
    assert int(packed[0, 0].item()) == (7 | (1 << 4))


def test_ties_round_to_even_code():
    from convert_hy4_fp8_to_nvfp4 import _e2m1_codes

    # exact midpoints between e2m1 values: RNE picks the even code
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, -0.25, -2.5], device="cuda")
    codes = _e2m1_codes(x)
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    vals = torch.where(codes >= 8, -1.0, 1.0) * lut[(codes & 7).long()]
    want = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, 0.0, -2.0], device="cuda")
    torch.testing.assert_close(vals, want, atol=0, rtol=0)


def test_zero_weight():
    w = torch.zeros(16, 32, device="cuda", dtype=torch.bfloat16)
    packed, scale, g = quantize_nvfp4(w)
    assert dequant_nvfp4(packed, scale, g).abs().max().item() == 0.0


def test_fp8_block_dequant():
    torch.manual_seed(3)
    w = (torch.randn(256, 384, device="cuda") * 0.1).to(torch.float8_e4m3fn)
    scale = (torch.rand(2, 3, device="cuda") * 0.01 + 0.001).to(torch.bfloat16)
    deq = dequant_fp8_block(w, scale)
    want = w.float() * scale.float().repeat_interleave(128, 0)[:256].repeat_interleave(128, 1)[:, :384]
    torch.testing.assert_close(deq.float(), want.to(torch.bfloat16).float(), atol=0, rtol=0)


@pytest.mark.needs_weights
@pytest.mark.skipif(not os.environ.get("FREETOKEN_HY4_NVFP4_MODEL"), reason="set FREETOKEN_HY4_NVFP4_MODEL to a converted checkpoint")
def test_converted_checkpoint_serves():
    from .harness import TinyEngine

    ck = os.environ["FREETOKEN_HY4_NVFP4_MODEL"]
    eng = TinyEngine(ck, None, quantized=True, max_seq_len=512)
    assert eng.config.expert_quant == "nvfp4"
    tokens = [100, 200, 300, 400, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
    req = eng.new_request(0, tokens, max_new=8)
    logits = eng.prefill([req])
    assert torch.isfinite(logits).all()
    eng.finish_prefill([req])
    tok = int(logits[-1].argmax().item())
    for _ in range(4):
        logits = eng.decode([req], [tok])
        assert torch.isfinite(logits).all()
        tok = int(logits[-1].argmax().item())
