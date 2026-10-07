"""MTP sampled-verification math (engine/spec_sample.py), plus two GPU checks.

Greedy MTP is a comparison, sampled MTP is a rejection test, and the difference is whether
the user's temperature is honored or merely approximated. A cheap stand-in -- accept the
draft when it looks probable, keep the target's argmax otherwise -- produces fluent text at
the wrong distribution, which is exactly the kind of bug no end-to-end test notices. So the
rule is pinned here, on host, without a model:

  * the truncation is the canonical one (temperature, then top-k, then top-p, renormalized)
    and applies to the draft and the target identically, because a lopsided ``q``/``p`` pair
    pushes the ratio past 1 on the trimmed tail and puts the bias back;
  * the emitted token follows ``q`` for ANY draft head, including a deliberately bad one;
  * the acceptance probability is ``sum(min(q, p))``, i.e. the throughput gate measures a
    quantity with a known ceiling rather than an arbitrary score;
  * the draws come from uniforms handed in, so a failure is reproducible and the same on
    every TP rank (an in-graph RNG would not be).

The wiring on top of this (the eager step and the scheduler gate) is
tests/engine/test_spec_graph.py and tests/scheduler/test_spec_drain.py.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.core import SamplingParams
from freetoken.engine.spec_sample import (
    debug_traced,
    draw_probs,
    expected_acceptance,
    rejection_sample,
    sampling_enabled,
    trunc_renorm_probs,
)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


# ----------------------------------------------------------------- truncation


def test_no_truncation_is_plain_softmax():
    logits = torch.randn(3, 17)
    got = trunc_renorm_probs(logits, temperature=1.0, top_k=-1, top_p=1.0)
    assert torch.allclose(got, torch.softmax(logits, dim=-1))
    assert torch.allclose(got.sum(-1), torch.ones(3), atol=1e-6)


def test_temperature_divides_logits():
    logits = torch.randn(2, 11)
    got = trunc_renorm_probs(logits, temperature=2.0, top_k=-1, top_p=1.0)
    assert torch.allclose(got, torch.softmax(logits / 2.0, dim=-1), atol=1e-7)


def test_top_k_keeps_k_and_renormalizes():
    logits = torch.tensor([[3.0, 1.0, 0.0, -2.0]])
    got = trunc_renorm_probs(logits, temperature=1.0, top_k=2, top_p=1.0)
    want = torch.softmax(torch.tensor([[3.0, 1.0]]), dim=-1)
    assert got[0, 2:].tolist() == [0.0, 0.0]
    assert torch.allclose(got[0, :2], want, atol=1e-7)
    assert got.sum(-1).item() == pytest.approx(1.0, abs=1e-6)


def test_top_p_cuts_at_the_cumulative_mass():
    # exact powers of two keep the cumulative boundary free of float noise
    probs = trunc_renorm_probs(torch.log(torch.tensor([[0.5, 0.25, 0.125, 0.0625, 0.0625]])),
                              temperature=1.0, top_k=-1, top_p=0.8)
    assert (probs[0, :3] > 0).all()
    assert probs[0, 3:].tolist() == [0.0, 0.0]
    assert probs.sum(-1).item() == pytest.approx(1.0, abs=1e-6)


def test_top_k_then_top_p_over_survivors():
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]])
    got = trunc_renorm_probs(logits, temperature=1.0, top_k=3, top_p=0.5)
    want = torch.softmax(logits[:, :3], dim=-1)
    want = want.masked_fill((want.cumsum(-1) - want) >= 0.5, 0.0)
    assert torch.allclose(got[:, :3], want / want.sum(), atol=1e-7)
    assert got[0, 3:].tolist() == [0.0, 0.0]


def test_truncation_never_reorders_the_top():
    # a greedy request never reaches this module, but the top of a truncated row must still
    # be the argmax -- the argmax path and this one have to agree on the leading token
    logits = torch.tensor([[0.4, 0.1, 0.9, 0.7]])
    assert trunc_renorm_probs(logits, 0.7, top_k=2).argmax(-1).item() == logits.argmax(-1).item()


def test_vocab_sized_top_k_is_no_truncation():
    logits = torch.randn(1, 9)
    assert torch.allclose(
        trunc_renorm_probs(logits, 1.0, top_k=9), torch.softmax(logits, dim=-1), atol=1e-7
    )


# ------------------------------------------------- truncation, on the real kernels
#
# Above is the host reference; the engine calls the same kernels the normal sampler does
# (flashinfer when installed, the Triton fallback otherwise -- the switch in spec_sample
# mirrors engine/sample.py). Those kernels can be JIT-built on first use, so these checks run
# against the Triton implementation, which needs no toolchain beyond Triton itself. A filter
# that trims differently on GPU than on host is a distribution bug that only shows up as text
# nobody can reproduce.


def _use_triton_kernels(monkeypatch):
    monkeypatch.setattr("freetoken.kernel.backend.is_flashinfer_installed", lambda: False)


@needs_cuda
@pytest.mark.parametrize("temperature,top_k,top_p", [
    (1.0, -1, 1.0),
    (0.7, 32, 0.9),
    (2.0, -1, 0.5),
])
def test_truncation_kernels_match_the_reference(monkeypatch, temperature, top_k, top_p):
    _use_triton_kernels(monkeypatch)
    torch.manual_seed(3)
    logits = torch.randn(32, 4096) * 3
    got = trunc_renorm_probs(logits.cuda(), temperature, top_k=top_k, top_p=top_p)
    want = trunc_renorm_probs(logits, temperature, top_k=top_k, top_p=top_p)
    torch.testing.assert_close(got.cpu(), want, atol=2e-5, rtol=2e-4)


def test_expected_acceptance_matches_the_tighter_of_two_uniform_draws():
    # sum(min(q, p)) is the acceptance probability; a zero p (the first verify of a request)
    # can never accept, and identical densities always do
    a = torch.full((1, 4), 0.25)
    assert expected_acceptance(a, torch.full((1, 4), 0.5)).item() == pytest.approx(1.0)
    assert expected_acceptance(a, torch.zeros(1, 4)).item() == pytest.approx(0.0)
    assert expected_acceptance(a, a).item() == pytest.approx(1.0)


@needs_cuda
def test_rejection_identity_holds_with_the_kernels(monkeypatch):
    # the identity tested on host, run through the kernels the sampled step actually calls
    _use_triton_kernels(monkeypatch)
    torch.manual_seed(4)
    device = torch.device("cuda")
    vocab = 4096
    q = trunc_renorm_probs(torch.randn(256, vocab, device=device) * 3, 1.0, top_k=64, top_p=0.9)
    p = torch.full_like(q, 1.0 / vocab)
    emitted = rejection_sample(
        q, p, draw_probs(p, torch.rand(256, device=device)),
        torch.rand(256, device=device), torch.rand(256, device=device))[0]
    emitted_p = q.gather(1, emitted.view(-1, 1)).squeeze(1)
    assert (emitted_p > 0).all()  # nothing leaves the truncated support
    # every emitted row came out of the target's own density, so what comes out carries q's
    # concentrated mass rather than p's uniform one
    assert emitted_p.mean().item() > q.mean().item() * 8


# ----------------------------------------------------------------- draws


def test_draw_inverts_the_cdf():
    probs = torch.tensor([[0.5, 0.25, 0.25, 0.0]])
    u = torch.tensor([0.0, 0.49, 0.5, 0.74, 0.75, 0.999, 2.0])
    got = draw_probs(probs.expand(7, 4), u).tolist()
    # the boundary belongs to the earlier bucket; past the total mass the draw clamps in
    assert got == [0, 0, 0, 1, 1, 2, 3]


def test_draw_is_deterministic_in_the_uniform():
    probs = torch.softmax(torch.randn(64, 32), dim=-1)
    u = torch.rand(64)
    assert torch.equal(draw_probs(probs, u), draw_probs(probs, u))


# ----------------------------------------------------------------- rejection


def test_matching_drafts_always_accept():
    # the draft head IS the target: the ratio is 1 everywhere, so nothing may be replaced
    logits = torch.randn(1, 40)
    q = trunc_renorm_probs(logits, 0.8, top_k=8, top_p=0.9)
    draft = draw_probs(q, torch.tensor([0.3]))
    emit, accept, ratio = rejection_sample(q, q.clone(), draft, torch.tensor([0.999]),
                                           torch.tensor([0.5]))
    assert bool(accept) and ratio.item() == pytest.approx(1.0)
    assert emit.item() == draft.item()


def test_zero_draft_probability_rejects_without_nan():
    q = torch.tensor([[0.6, 0.4, 0.0]])
    p = torch.tensor([[0.0, 0.5, 0.5]])
    emit, accept, ratio = rejection_sample(
        q, p, torch.tensor([0]), torch.tensor([0.5]), torch.tensor([0.5])
    )
    assert not bool(accept) and ratio.item() == 0.0
    assert emit.item() == 0  # only token 0 survives in (q - p)+
    assert torch.isfinite(ratio).all()


def test_residual_fallback_stays_in_vocab_when_distributions_agree():
    q = trunc_renorm_probs(torch.randn(1, 16), 1.0, top_k=4)
    trimmed = int((q[0] == 0).nonzero()[0])  # a drafted id the target refused outright
    emit, accept, _ = rejection_sample(q, q.clone(), torch.tensor([trimmed]),
                                       torch.tensor([0.5]), torch.tensor([0.5]))
    assert not bool(accept)  # q == p leaves an empty residual to draw from
    assert 0 <= emit.item() < 16  # and that draw still has to stay inside the row


def test_rejection_recovers_the_target_distribution_for_a_bad_draft():
    # the guarantee this phase rests on: emitted ~ q even when p is nothing like q
    torch.manual_seed(1234)
    vocab = 6
    q = torch.softmax(torch.randn(1, vocab) * 2.0, dim=-1)
    p = torch.softmax(torch.randn(1, vocab) * -3.0, dim=-1)
    trials = 200_000
    q_b, p_b = q.expand(trials, vocab), p.expand(trials, vocab)
    gen = torch.Generator().manual_seed(7)
    draft = draw_probs(p_b, torch.rand(trials, generator=gen))
    emit, _, _ = rejection_sample(
        q_b, p_b, draft, torch.rand(trials, generator=gen), torch.rand(trials, generator=gen)
    )
    counts = torch.bincount(emit, minlength=vocab).float() / trials
    assert torch.allclose(counts, q.squeeze(0), atol=0.01)


def test_acceptance_rate_equals_sum_of_minimums():
    torch.manual_seed(99)
    vocab = 5
    q = torch.softmax(torch.randn(1, vocab) * 1.5, dim=-1)
    p = torch.softmax(torch.randn(1, vocab) * 2.0, dim=-1)
    trials = 200_000
    q_b, p_b = q.expand(trials, vocab), p.expand(trials, vocab)
    gen = torch.Generator().manual_seed(11)
    draft = draw_probs(p_b, torch.rand(trials, generator=gen))
    _, accept, _ = rejection_sample(
        q_b, p_b, draft, torch.rand(trials, generator=gen), torch.rand(trials, generator=gen)
    )
    assert accept.float().mean().item() == pytest.approx(
        expected_acceptance(q, p).item(), abs=0.01
    )


def test_rejection_is_reproducible_from_its_uniforms():
    torch.manual_seed(5)
    q = trunc_renorm_probs(torch.randn(8, 64), 1.0, top_k=16, top_p=0.95)
    p = trunc_renorm_probs(torch.randn(8, 64), 1.0, top_k=16, top_p=0.95)
    args = (q, p, draw_probs(p, torch.rand(8)), torch.rand(8), torch.rand(8))
    assert torch.equal(rejection_sample(*args)[0], rejection_sample(*args)[0])


# ----------------------------------------------------------------- gate


def test_gate_tracks_params_and_the_kill_switch():
    assert sampling_enabled(SamplingParams(temperature=1.0, top_k=20, top_p=0.95))
    assert not sampling_enabled(SamplingParams(temperature=0.0))  # greedy keeps the argmax path
    assert not sampling_enabled(SamplingParams(temperature=1.0, top_k=1))  # is_greedy
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("FREETOKEN_MTP_SAMPLING", "0")
        assert not sampling_enabled(SamplingParams(temperature=1.0, top_k=20))


def test_truncation_is_row_independent():
    """The sampled step truncates both target rows in one pass, so per-row and batched calls
    must agree bit for bit -- the equivalence the batching leans on, and the reason the
    spec step can spend one launch set instead of two."""
    torch.manual_seed(11)
    x = torch.randn(3, 64)
    for top_k, top_p in ((8, 0.9), (64, 1.0), (-1, 0.5), (1, 1.0)):
        batched = trunc_renorm_probs(x, 0.7, top_k, top_p)
        rows = torch.cat([trunc_renorm_probs(x[i : i + 1], 0.7, top_k, top_p) for i in range(3)])
        assert torch.equal(batched, rows), (top_k, top_p)
        assert torch.allclose(batched.sum(-1), torch.ones(3), atol=1e-5)


def test_the_trace_switch_reads_zero_as_off():
    """The trace synchronizes the device on every spec step, so a value that only LOOKED
    disabled would cost the throughput the trace is used to measure."""
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("FREETOKEN_MTP_DEBUG", raising=False)
        assert not debug_traced()
        mp.setenv("FREETOKEN_MTP_DEBUG", "0")
        assert not debug_traced()
        mp.setenv("FREETOKEN_MTP_DEBUG", "1")
        assert debug_traced()


def test_the_carried_density_belongs_to_the_request():
    """Rejection sampling is unbiased only against the density the draft was DRAWN from, so
    that row cannot be engine-wide: two sampled requests alternate single-request spec
    batches, and the second step would overwrite the p the first is about to test against."""
    from freetoken.core import Req
    from freetoken.engine.engine import Engine

    eng = Engine.__new__(Engine)
    eng.device = torch.device("cpu")

    def req():
        return Req(input_ids=torch.arange(4, dtype=torch.int32), table_idx=0, cached_len=0,
                   output_len=4, uid=1, sampling_params=SamplingParams(temperature=0.7),
                   cache_handle=None)

    a, b = req(), req()
    first = eng._spec_draft_probs(a, 8)
    assert first is a.spec_draft_probs and first.shape == (1, 8)
    assert eng._spec_draft_probs(b, 8) is not first
    assert eng._spec_draft_probs(a, 8) is first

