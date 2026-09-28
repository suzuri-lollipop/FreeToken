"""Triton sampling kernels (kernel/triton/sampling.py) vs torch references.

Sampling is the silent-corruption class: a wrong softmax still sums to ~1 and a
wrong top-k still returns plausible token ids, so the draw kernels' RNG is left
opaque on purpose and the parity here pins only what a correct sampler must
satisfy for EVERY draw: argmax under top_k=1, membership in the exact torch
top-k/top-p support, exact probability-mass invariants, and determinism under
an explicit seed.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.kernel.triton.sampling import (
    sampling_from_probs,
    softmax,
    top_k_renorm_probs,
    top_k_sampling_from_probs,
    top_k_top_p_sampling_from_probs,
    top_p_renorm_probs,
    top_p_sampling_from_probs,
)

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

_DEV = "cuda"
_ATOL = 1e-4


def _logits(b: int, v: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device=_DEV).manual_seed(seed)
    return torch.randn(b, v, generator=g, device=_DEV)


def _nucleus_set(probs: torch.Tensor, top_p: float) -> list[set[int]]:
    """The smallest set of largest-probability tokens whose mass reaches top_p."""
    order = torch.argsort(probs, dim=-1, descending=True)
    ps = probs.gather(-1, order)
    cut = (ps.cumsum(-1) >= top_p).to(torch.long).argmax(-1)
    return [
        set(order[row, : cut[row] + 1].tolist()) for row in range(probs.size(0))
    ]


def test_softmax_empty_batch_is_an_identity_clone():
    empty = torch.empty((0, 16), device="cpu")
    out = softmax(empty)
    assert out.shape == (0, 16)
    assert out.dtype == torch.float32


@requires_gpu
def test_softmax_matches_torch_with_per_row_temperatures():
    logits = _logits(4, 2048, 1)
    t = torch.tensor([0.5, 1.0, 2.0, 0.1], device=_DEV)
    ref = torch.softmax(logits / t.unsqueeze(1), dim=-1)
    assert torch.allclose(softmax(logits, t), ref, atol=_ATOL)


@requires_gpu
def test_softmax_scalar_and_default_temperatures():
    logits = _logits(3, 1024, 2)
    assert torch.allclose(softmax(logits, 0.7), torch.softmax(logits / 0.7, -1), atol=_ATOL)
    assert torch.allclose(softmax(logits), torch.softmax(logits, -1), atol=_ATOL)


@requires_gpu
def test_top_k_one_is_always_the_argmax():
    logits = _logits(4, 4096, 3)
    expected = logits.argmax(-1)
    for temperature in (None, 1.0, 0.7):
        probs = softmax(logits, temperature)
        assert torch.equal(top_k_sampling_from_probs(probs, 1, seed=11), expected)
        # per-row k as a tensor takes the same path
        k = torch.ones(4, device=_DEV, dtype=torch.int32)
        assert torch.equal(top_k_sampling_from_probs(probs, k, seed=11), expected)
        assert torch.equal(top_k_top_p_sampling_from_probs(probs, 1, 1.0, seed=11), expected)


@requires_gpu
def test_top_k_draw_stays_inside_the_exact_torch_topk():
    probs = softmax(_logits(4, 8192, 5), 0.9)
    for k in (5, 100):
        out = top_k_sampling_from_probs(probs, k, seed=k)
        ref_sets = torch.topk(probs, k, dim=-1).indices
        for row in range(probs.size(0)):
            assert out[row].item() in ref_sets[row].tolist()


@requires_gpu
def test_top_k_renorm_has_exactly_the_topk_support_and_renormalized_mass():
    probs = softmax(_logits(2, 2048, 6), 1.3)
    renorm = top_k_renorm_probs(probs, 32)
    ref_sets = torch.topk(probs, 32, dim=-1).indices
    for row in range(2):
        kept = set(renorm[row].nonzero().flatten().tolist())
        assert kept == set(ref_sets[row].tolist())
        assert torch.allclose(renorm[row][renorm[row] > 0], probs[row][sorted(kept)] / probs[row][sorted(kept)].sum(), atol=_ATOL)


@requires_gpu
def test_top_p_draw_stays_inside_the_torch_nucleus():
    probs = softmax(_logits(4, 8192, 7), 0.8)
    out = top_p_sampling_from_probs(probs, 0.9, seed=23)
    for row, nucleus in enumerate(_nucleus_set(probs, 0.9)):
        assert out[row].item() in nucleus


@requires_gpu
def test_top_p_renorm_mass_invariants():
    probs = softmax(_logits(2, 4096, 8), 1.0)
    for top_p in (0.3, 0.6, 0.95):
        out = top_p_renorm_probs(probs, top_p)
        assert torch.allclose(out.sum(-1), torch.ones(2, device=_DEV), atol=_ATOL)
        kept_mass = (probs * (out > 0)).sum(-1)
        # renorm never keeps LESS mass than the requested nucleus
        assert (kept_mass >= top_p * (1 - 1e-4) - 1e-6).all()
        # nothing gains mass from nowhere
        assert ((out > 0).float() <= (probs > 0).float()).all()

    assert torch.allclose(top_p_renorm_probs(probs, 1.0), probs, atol=_ATOL)


@requires_gpu
def test_one_hot_probabilities_draw_their_only_mass():
    b, v = 4, 1024
    probs = torch.zeros(b, v, device=_DEV)
    wins = torch.tensor([7, 1011, 3, 500], device=_DEV)
    probs[torch.arange(b, device=_DEV), wins] = 1.0

    assert torch.equal(sampling_from_probs(probs, seed=1), wins)
    assert torch.equal(top_k_sampling_from_probs(probs, 10, seed=1), wins)
    assert torch.equal(top_p_sampling_from_probs(probs, 0.9, seed=1), wins)
    assert torch.equal(top_k_top_p_sampling_from_probs(probs, 5, 0.5, seed=1), wins)


@requires_gpu
def test_same_seed_is_deterministic_and_outputs_are_int32():
    probs = softmax(_logits(2, 4096, 9), 1.0)
    a = sampling_from_probs(probs, seed=42)
    b = sampling_from_probs(probs, seed=42)
    assert torch.equal(a, b)
    assert a.dtype == torch.int32
    assert ((a >= 0) & (a < 4096)).all()


@requires_gpu
def test_bf16_probs_are_cast_before_drawing():
    logits = _logits(2, 2048, 10).to(torch.bfloat16)
    probs = softmax(logits, 1.0).to(torch.bfloat16)
    out = top_k_sampling_from_probs(probs, 1)
    assert torch.equal(out, probs.float().argmax(-1))