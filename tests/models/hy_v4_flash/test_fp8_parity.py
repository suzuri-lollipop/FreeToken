"""FP8 (block-fp8) serving parity: the released FP8 layout -- stacked block-fp8
experts through the offload banks + block-fp8 attention projections -- against the
eager reference computing on the same weights dequantized to bf16.

The family serves block-fp8 weight-only (W8A16, ``fp8_w8a16``): Hy4-flash is
bf16-trained and its activations degenerate under the kernels' default dynamic
W8A8 quantization, so both phases keep bf16 activations and the plain bf16
reference is the parity expectation. Uses the released attention/expert widths
(the fp8 block kernels need 128-multiple geometry); conventions and the
noise-floor comparison follow test_real_dims_parity.
"""

from __future__ import annotations

import os

import pytest
import torch

from .common import requires_cuda, write_tiny_fp8_checkpoint
from .harness import TinyEngine
from .reference import Cache
from .test_reference_parity import REAL_DIMS, SelectionBridge, _reference, _tokens

pytestmark = requires_cuda


@pytest.fixture(scope="module")
def fp8_checkpoint(tmp_path_factory):
    folder = tmp_path_factory.mktemp("hy4-fp8-parity")
    dequantized = write_tiny_fp8_checkpoint(str(folder), seed=17, indexer_weight_scale=1.0, **REAL_DIMS)
    return str(folder), dequantized


def test_fp8_parity(fp8_checkpoint, monkeypatch):
    import freetoken.layers.quantization.linear.unquantized as unq

    import models.hy_v4_flash.reference as refmod

    monkeypatch.setattr(unq, "_W8A16_ENABLED", False)
    monkeypatch.setattr(refmod, "ROPE_FP32", True)
    monkeypatch.setattr(refmod, "SOFTMAX_FP32", True)
    folder, dequantized = fp8_checkpoint
    eng = TinyEngine(folder, None, quantized=True, moe_strategy="offload", max_seq_len=1024)
    assert eng.config.expert_quant == "fp8_block"
    ref = _reference(folder, dequantized)
    ref32 = _reference(folder, {k: v.float() if v.is_floating_point() else v for k, v in dequantized.items()})
    bridge = SelectionBridge(eng, ref)
    bridge32 = SelectionBridge(eng, ref32)
    try:
        tokens = _tokens(40, 9)
        req = eng.new_request(0, tokens, max_new=8)
        got = eng.prefill([req])
        cache, cache32 = Cache(), Cache()
        ids = torch.tensor([tokens], device="cuda")
        pos = torch.arange(40, device="cuda").unsqueeze(0)
        want = ref.forward(ids, pos, cache)[0, -1:]
        want32 = ref32.forward(ids, pos, cache32)[0, -1:]
        # 2.5x margin: besides independent bf16 roundings, the W8A16 kernels
        # dequantize fp8 weights in fp32 inside the K-loop while the reference
        # rounds them to bf16 first -- a small systematic precision component
        rel_err = ((got - want).norm() / want.norm().clamp(min=1e-9)).item()
        rel_noise = ((want - want32).norm() / want.norm().clamp(min=1e-9)).item()
        assert rel_err <= max(2.5 * rel_noise, 0.02), (
            f"fp8 prefill exceeds bf16 noise floor: rel err {rel_err:.4f}, reference spread {rel_noise:.4f}"
        )
        eng.finish_prefill([req])
        full = list(tokens)
        for step in range(4):
            token = int(want[-1].argmax().item())
            full.append(token)
            got = eng.decode([req], [token])[0]
            full_t = torch.tensor([full], device="cuda")
            want = ref.generate_step(torch.tensor([[token]], device="cuda"), len(full) - 1, cache, full_t)[0, 0]
            want32 = ref32.generate_step(torch.tensor([[token]], device="cuda"), len(full) - 1, cache32, full_t)[0, 0]
            rel_err = ((got - want).norm() / want.norm().clamp(min=1e-9)).item()
            rel_noise = ((want - want32).norm() / want.norm().clamp(min=1e-9)).item()
            assert rel_err <= max(2.5 * rel_noise, 0.02), (
                f"fp8 decode {step} exceeds bf16 noise floor: rel err {rel_err:.4f}, reference spread {rel_noise:.4f}"
            )
        assert bridge.total > 0
        rate = bridge.flips / bridge.total
        assert rate <= 0.10, f"fp8 selection flip rate {bridge.flips}/{bridge.total} = {rate:.3f}"
    finally:
        bridge.restore()
        bridge32.restore()


@pytest.mark.needs_weights
@pytest.mark.skipif(not os.environ.get("FREETOKEN_HY4_FP8_MODEL"), reason="set FREETOKEN_HY4_FP8_MODEL to the FP8 checkpoint")
def test_fp8_checkpoint_serves():
    ck = os.environ["FREETOKEN_HY4_FP8_MODEL"]
    eng = TinyEngine(ck, None, quantized=True, moe_strategy="offload", max_seq_len=512)
    assert eng.config.expert_quant == "fp8_block"
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
