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


def test_degenerate_top_p_is_clamped_like_the_main_sampler():
    # Sampler.prepare clamps top_p into [1e-6, 1.0]; the API hands the raw knob over, and an
    # unclamped p <= 0 renorms the row to all zeros -- the draw then comes back as the vocab's
    # last token instead of the target's top one, and the rejection test divides by NaN
    logits = torch.tensor([[3.0, 1.0, 0.0, -2.0]])
    want = trunc_renorm_probs(logits, temperature=1.0, top_k=-1, top_p=1e-6)
    assert want.argmax(-1).item() == 0 and (want[0, 1:] == 0).all()
    for p in (0.0, -1.0):
        got = trunc_renorm_probs(logits, temperature=1.0, top_k=-1, top_p=p)
        assert torch.isfinite(got).all()
        assert got.sum(-1).item() == pytest.approx(1.0, abs=1e-6)
        assert torch.allclose(got, want)  # exactly the main sampler's clamped distribution


def test_top_p_above_one_is_no_truncation():
    logits = torch.randn(1, 9)
    assert torch.allclose(
        trunc_renorm_probs(logits, 1.0, top_p=1.5), torch.softmax(logits, dim=-1), atol=1e-7
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
    (0.7, -1, 0.0),  # the degenerate knob: the clamp must reach the kernel path too
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


def test_knob_tensors_are_cached_and_clamped():
    """The per-row knob tensors are rebuilt twice per sampled step (q and p_next); the cache
    turns that into zero allocs after the first, with the same skip rules and clamps."""
    from freetoken.engine.spec_sample import MIN_TOP_P, _knob_tensors

    x = torch.randn(2, 11)
    first = _knob_tensors(x, 0.7, 5, 0.9)
    second = _knob_tensors(x, 0.7, 5, 0.9)
    assert all(a is b for a, b in zip(first, second))  # cached by host value
    t, k, p = first
    assert t.tolist() == [pytest.approx(0.7)] * 2
    assert k.tolist() == [5, 5] and k.dtype == torch.int32
    assert p.tolist() == [pytest.approx(0.9)] * 2

    # no-op knobs build nothing; degenerate ones land on Sampler.prepare's bounds
    assert _knob_tensors(x, 1.0, -1, 1.0)[1:] == (None, None)
    assert _knob_tensors(x, 1.0, 11, 1.0)[1] is None       # k >= vocab: no truncation
    _, _, p0 = _knob_tensors(x, 1.0, -1, 0.0)
    assert p0.tolist() == [pytest.approx(MIN_TOP_P, rel=1e-5)] * 2

    # a tensor knob cannot be keyed by value and never pollutes the cache
    tv = _knob_tensors(x, torch.full((2,), 0.7), 5, 0.9)
    assert _knob_tensors(x, torch.full((2,), 0.7), 5, 0.9)[0] is not tv[0]


@needs_cuda
def test_both_knobs_take_one_fused_renorm_launch(monkeypatch):
    """top_k AND top_p active (the checkpoint's own defaults) must reach the fused triton
    renorm -- one launch, no intermediate full-vocab row, whatever backend supplied the
    softmax -- and stay numerically the two-stage truncation it replaces."""
    import freetoken.kernel.triton.sampling as tri
    from freetoken.engine import spec_sample

    calls = []
    real = tri.top_k_top_p_renorm_probs

    def spy(probs, top_k, top_p):
        calls.append((probs.shape, top_k.tolist(), top_p.tolist()))
        return real(probs, top_k, top_p)

    monkeypatch.setattr(spec_sample, "_KNOB_CACHE", {})
    monkeypatch.setattr(tri, "top_k_top_p_renorm_probs", spy)
    _use_triton_kernels(monkeypatch)  # the softmax backend; the fused renorm is triton either way
    torch.manual_seed(5)
    logits = torch.randn(2, 4096, device="cuda") * 3
    got = trunc_renorm_probs(logits, 1.0, top_k=20, top_p=0.95)
    assert len(calls) == 1 and calls[0][1] == [20, 20]
    want = trunc_renorm_probs(logits, 1.0, top_k=-1, top_p=1.0)  # plain softmax start point
    staged = tri.top_k_renorm_probs(
        want, torch.full((2,), 20, dtype=torch.int32, device="cuda"))
    staged = tri.top_p_renorm_probs(staged, torch.full((2,), 0.95, device="cuda"))
    torch.testing.assert_close(got, staged, atol=2e-5, rtol=2e-4)
    assert torch.allclose(got.sum(-1), torch.ones(2, device="cuda"), atol=1e-4)


@needs_cuda
def test_spec_stage_round_trips_uniforms_and_draft():
    """One pinned row carries the step's four uniforms (bit-exact through the int64 view)
    and the draft id, replacing three pageable copies the eager step used to serialize on."""
    from freetoken.engine.engine import Engine

    eng = Engine.__new__(Engine)
    eng.device = torch.device("cuda")
    uniforms = torch.rand(4)
    u, draft = eng._spec_stage(uniforms, 4242)
    assert u.device.type == "cuda" and draft.device.type == "cuda"
    torch.cuda.synchronize()
    assert torch.equal(u.cpu(), uniforms)  # bit-exact: same fp32 bits, int64 carrier
    assert draft.tolist() == [4242] and draft.dtype == torch.int64
    # the buffers are reused, so a second stage must not alias stale values into a consumer
    u2, d2 = eng._spec_stage(torch.rand(4), 7)
    assert u2.data_ptr() == u.data_ptr() and d2.tolist() == [7]


def test_graph_knob_values_express_skips_as_no_op_values():
    """A capture cannot skip a launch, so a disabled filter is STAGED as its no-op value:
    k = vocab keeps every token, p = 1.0 keeps all mass, and the clamps match the host path
    (Sampler.prepare's bounds) so a graphed step truncates exactly like the eager one."""
    from freetoken.engine.spec_sample import MIN_TEMPERATURE, MIN_TOP_P, graph_knob_values

    params = SamplingParams(temperature=1.0, top_k=20, top_p=0.95)
    assert graph_knob_values(params, 1000) == (1.0, 20, 0.95)
    # disabled / degenerate knobs become the no-op or clamped value
    assert graph_knob_values(SamplingParams(temperature=0.5), 1000) == (0.5, 1000, 1.0)
    assert graph_knob_values(
        SamplingParams(temperature=1.0, top_k=5000, top_p=1.5), 1000)[1:] == (1000, 1.0)
    t, _, p = graph_knob_values(
        SamplingParams(temperature=1e-9, top_k=20, top_p=0.0), 1000)
    assert t == MIN_TEMPERATURE and p == MIN_TOP_P


def test_sampled_graph_switch_reads_zero_as_off(monkeypatch):
    from freetoken.engine.spec_sample import sampled_graph_enabled

    monkeypatch.delenv("FREETOKEN_MTP_SAMPLED_GRAPH", raising=False)
    assert sampled_graph_enabled()
    monkeypatch.setenv("FREETOKEN_MTP_SAMPLED_GRAPH", "0")
    assert not sampled_graph_enabled()
    monkeypatch.setenv("FREETOKEN_MTP_SAMPLED_GRAPH", "1")
    assert sampled_graph_enabled()


@needs_cuda
def test_fused_renorm_is_the_identity_at_the_no_op_knobs(monkeypatch):
    """The graphed body always runs the fused renorm, so its no-op point (k = vocab,
    p = 1.0) must hand the softmax back unchanged -- that is what lets one capture serve
    requests whose filters the eager path would have skipped entirely."""
    import freetoken.kernel.triton.sampling as tri

    _use_triton_kernels(monkeypatch)
    torch.manual_seed(7)
    logits = torch.randn(2, 4096, device="cuda") * 3
    probs = trunc_renorm_probs(logits, 1.0, top_k=-1, top_p=1.0)  # plain softmax
    k = torch.full((2,), 4096, dtype=torch.int32, device="cuda")
    p = torch.ones(2, device="cuda")
    got = tri.top_k_top_p_renorm_probs(probs, k, p)
    torch.testing.assert_close(got, probs, atol=1e-6, rtol=1e-5)
    assert torch.allclose(got.sum(-1), torch.ones(2, device="cuda"), atol=1e-6)

