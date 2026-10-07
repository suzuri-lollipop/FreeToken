"""Engine-side MTP residual stash gates (engine/engine.py:_spec_stash_eligible and
_stash_spec_residual), CPU-only.

Prefill decides here whether a request may draft later: the rows the target forward did
not run the MTP head on are handed to the first decode step as a prologue, which is what
keeps acceptance at 100% on the prompt instead of starting from a cold head. Two failure
modes are silent and expensive, so they are pinned on the host-side state only:

  * stashing anyway when the verify could never run (chunked prefill, shared prefix, no
    GDN pool, sampling turned off for non-greedy requests, multimodal spans). The prologue
    then runs the head over rows nothing verified, and the first mismatch costs a rollback
    with no committed state to roll back to -- the same wrongness the pre-MTP loop had, paid
    for with a draft model.
  * stashing an unbounded row span. The stash is prompt-sized, so a long-prompt request
    that never gets to draft (it stays in a mixed batch for its whole life) pins host
    memory for nothing. The cap therefore tracks max_extend_tokens, not just a constant.

Nothing here touches CUDA, so a wrong gate still lets the model answer correctly -- with
worse tokens-per-second and a little more resident host memory. The scheduler side of the
same state machine (why a stash gets dropped later) is tests/scheduler/test_spec_drain.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.engine.engine import Engine
from freetoken.scheduler.prefill import ChunkedReq


def _engine(max_extend_tokens: int = 8192, speculative: str = "mtp") -> Engine:
    eng = Engine.__new__(Engine)
    eng.config = SimpleNamespace(max_extend_tokens=max_extend_tokens, speculative=speculative)
    return eng


def _prefill_batch(*reqs) -> Batch:
    batch = Batch(reqs=list(reqs), phase="prefill")
    # padded_reqs is what the forward actually ran; the stash must cover exactly those rows
    batch.padded_reqs = list(reqs)
    return batch


def _req(prompt_len: int, *, chunked: bool = False, temperature: float = 0.0,
         cached_len: int = 0, mm_items=None, gdn_slot: int | None = 0) -> Req:
    cls = ChunkedReq if chunked else Req
    req = cls(input_ids=torch.arange(prompt_len + 1, dtype=torch.int32), table_idx=0,
              cached_len=cached_len, output_len=0, uid=1,
              sampling_params=SamplingParams(temperature=temperature), cache_handle=None)
    # a chunk mid-prefill stops short of the last prompt token; a cold one runs all of it
    req.device_len = prompt_len - 2 if chunked else prompt_len
    req.mm_items = mm_items
    req.linear_slot_idx = gdn_slot
    return req


def _stash(req, rows: int | None = None, **kw):
    """Run the gate over a one-request prefill batch; returns the residual it read."""
    batch = _prefill_batch(req)
    n = rows if rows is not None else max(8, int(req.input_ids.numel()))
    residual = torch.zeros(n, 4, dtype=torch.float32)
    residual.view(-1).copy_(torch.arange(n * 4, dtype=torch.float32))
    _engine(**kw)._stash_spec_residual(batch, residual)
    return residual


def test_cold_greedy_prompt_stashes_its_rows():
    """The whole prompt ran under one forward with no shared prefix: the stash is a view of
    this batch's residual, so the prologue can walk the head up to the live position for
    free -- and no host copy is paid for rows the caller still owns."""
    req = _req(6)
    residual = _stash(req)
    assert req.spec_off is False and req.spec_residual is not None
    # one row per prompt token this forward ran, aliasing the residual it came from
    assert req.spec_residual.shape[0] == req.extend_len
    assert req.spec_residual._base is residual
    assert torch.equal(req.spec_residual, residual[:req.extend_len])


def test_multi_request_batch_copies_before_the_residual_is_reused():
    """A padded prefill batch shares one residual buffer that the next forward overwrites,
    so more than one request means a real copy per request."""
    a, b = _req(4), _req(4)
    batch = _prefill_batch(a, b)
    _engine()._stash_spec_residual(batch, torch.zeros(8, 4))
    assert a.spec_residual is not None and a.spec_residual._base is None
    assert a.spec_residual.shape[0] == a.extend_len


def test_every_decline_is_named_and_drops_the_stash():
    """Each reason costs something different, and the log line is the only place the field
    number for "why is MTP not helping this workload" comes from."""
    cases = {
        "chunked": _req(6, chunked=True),
        "prefix_hit": _req(6, cached_len=2),
        "no_gdn_pool": _req(6, gdn_slot=None),
        "mm": _req(6, mm_items=[{"fake": "image"}]),
    }
    for reason, req in cases.items():
        _stash(req)
        assert req.spec_off is True, reason
        assert req.spec_off_reason == reason
        assert req.spec_residual is None
        # the rows and the cap the gate saw, handed over for the scheduler's decline line:
        # by the time that prefill drains, complete_one() has moved cached_len to the whole
        # prompt and extend_len reads 0, so the live request no longer describes the decision
        assert "prompt=" in req.spec_off_detail, reason
        assert "stash_budget=" in req.spec_off_detail, reason


def test_sampled_request_stashes_until_the_kill_switch_says_otherwise(monkeypatch):
    """A non-greedy request drafts too: its verify rejects by ratio instead of comparing ids,
    so the prologue rows are still worth the host memory. FREETOKEN_MTP_SAMPLING=0 restores
    the old decline, which is the A/B switch the gate keeps for exactly that reason."""
    sampled = _req(6, temperature=0.7)
    residual = _stash(sampled)
    assert sampled.spec_off is False and sampled.spec_residual is not None
    assert torch.equal(sampled.spec_residual, residual[:sampled.extend_len])

    monkeypatch.setenv("FREETOKEN_MTP_SAMPLING", "0")
    declined = _req(6, temperature=0.7)
    _stash(declined)
    assert declined.spec_off is True and declined.spec_off_reason == "nongreedy"
    assert declined.spec_residual is None


def test_stash_cap_follows_the_prefill_chunk_budget(monkeypatch):
    """The cap is min(env, max_extend_tokens): a prompt that arrives in pieces bigger than
    the prefill chunk is chunked anyway, and a stash larger than one forward is a lie."""
    monkeypatch.delenv("FREETOKEN_MTP_MAX_STASH_TOKENS", raising=False)
    over = _req(8199)  # numel 8200 > min(16384 default, 8192 max_extend_tokens)
    _stash(over)
    assert over.spec_off and over.spec_off_reason == "over_budget"

    monkeypatch.setenv("FREETOKEN_MTP_MAX_STASH_TOKENS", "64")
    capped = _req(70)
    _stash(capped, max_extend_tokens=32)
    assert capped.spec_off and capped.spec_off_reason == "over_budget"

    fits = _req(30)
    _stash(fits, max_extend_tokens=32)
    assert not fits.spec_off and fits.spec_residual is not None


def test_the_forward_gate_and_the_stash_gate_read_sampling_the_same_way(monkeypatch):
    """The forward-level gate decides whether the residual rows exist at all.

    It once tested ``is_greedy`` while this file's per-request gate had moved to
    ``spec_supported``: a sampled batch then computed no residual, so no draft was ever
    produced and the request decoded regularly forever -- and because the batch-level gate
    names no request, not even a decline was counted. Both gates must read the same switch.
    """
    sampled = _req(6, temperature=0.7)
    eng = _engine()
    assert eng._spec_stash_eligible(_prefill_batch(sampled), False)
    _stash(sampled)
    assert sampled.spec_off is False and sampled.spec_residual is not None

    monkeypatch.setenv("FREETOKEN_MTP_SAMPLING", "0")
    killed = _req(6, temperature=0.7)
    assert not _engine()._spec_stash_eligible(_prefill_batch(killed), False)
    _stash(killed)
    assert killed.spec_off and killed.spec_off_reason == "nongreedy"


def test_the_forward_gate_only_pays_for_a_batch_someone_can_draft_with():
    """The stash is a [T, hc*hidden] clone of the chunk, so it is bought only when a request in
    this batch can spend it: a request already off, an image row, a captured prefill chunk (no
    residual comes out of a graph replay), or an engine with no speculative decoding at all."""
    eng = _engine()
    off = _req(6)
    off.spec_off = True
    assert not eng._spec_stash_eligible(_prefill_batch(off), False)
    image = _req(6, mm_items=[{"fake": "image"}])
    assert not eng._spec_stash_eligible(_prefill_batch(image), False)
    assert not eng._spec_stash_eligible(_prefill_batch(_req(6)), True)
    assert not _engine(speculative="none")._spec_stash_eligible(_prefill_batch(_req(6)), False)
    # one draftable request in a batch of declined ones is enough to produce the rows
    declined, greedy = _req(6), _req(6)
    declined.spec_off = True
    assert eng._spec_stash_eligible(_prefill_batch(declined, greedy), False)
