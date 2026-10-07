"""Engine-side MTP residual stash gates (engine/engine.py:_spec_stash_eligible and
_stash_spec_residual), CPU-only.

Prefill decides here whether a request may draft later: the rows the target forward did
not run the MTP head on are handed to the first decode step as a prologue, which is what
keeps acceptance at 100% on the prompt instead of starting from a cold head. That pass
wants a hole-free span starting at position 0, so chunked prefills ACCUMULATE and only
genuinely missing rows decline. Two failure modes are silent and expensive, so they are
pinned on the host-side state only:

  * stashing anyway when the verify could never run (a shared prefix, no GDN pool, sampling
    turned off by the kill switch, multimodal spans). The prologue then runs the head over
    rows nothing verified, and the first mismatch costs a rollback with no committed state to
    roll back to -- the same wrongness the pre-MTP loop had, paid for with a draft model.
  * stashing an unbounded row span. The stash is prompt-sized and becomes the catch-up pass,
    so FREETOKEN_MTP_MAX_STASH_TOKENS bounds both the memory a long prompt pins until its
    first decode step spends it and the time that step pays. It deliberately no longer
    follows the prefill chunk size.

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
    number for "why is MTP not helping this workload" comes from. A chunked prompt is NOT in
    this list: its chunks accumulate (test_chunked_prefill_accumulates_its_rows)."""
    cases = {
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


def test_the_stash_budget_counts_rows_not_one_chunk(monkeypatch):
    """Phase 1 tied the cap to max_extend_tokens because a stash could not span chunks. Now it
    can, so the cap is only the memory and time a long prompt buys: rows of prompt residual."""
    monkeypatch.delenv("FREETOKEN_MTP_MAX_STASH_TOKENS", raising=False)
    wide = _req(12000)  # 12000 rows: more than one chunk, less than the 16384 default
    _stash(wide, max_extend_tokens=32)
    assert not wide.spec_off and wide.spec_residual.shape[0] == 12000

    monkeypatch.setenv("FREETOKEN_MTP_MAX_STASH_TOKENS", "64")
    capped = _req(70)
    _stash(capped)
    assert capped.spec_off and capped.spec_off_reason == "over_budget"
    assert capped.spec_residual is None

    fits = _req(30)
    _stash(fits)
    assert not fits.spec_off and fits.spec_residual is not None


def test_chunked_prefill_accumulates_its_rows(monkeypatch):
    """The catch-up pass needs every row the request ran, not one forward's worth, so each chunk
    adds its rows: that is what puts a long cold prompt on the spec path at all."""
    monkeypatch.delenv("FREETOKEN_MTP_MAX_STASH_TOKENS", raising=False)
    first = _req(8, chunked=True)  # cached_len 0, extend_len 6
    _stash(first, rows=6)
    assert first.spec_off is False and first.spec_residual.shape[0] == 6

    first.cached_len = 6  # the chunk committed exactly what the stash already covers
    first.device_len = 8  # the final chunk: a plain Req with cached_len > 0
    _stash(first, rows=2)
    stash = first.spec_residual
    assert first.spec_off is False and stash.shape[0] == 8
    assert stash._base is None  # a concatenation, not an alias into either forward's buffer
    assert stash[:6, 0].tolist() == [4 * i for i in range(6)]  # the first chunk's rows, in order


def test_a_hole_in_the_span_declines_as_prefix_hit(monkeypatch):
    """Rows the target served from cache were never run, so no stash can cover them -- and rows
    that advanced while nothing was stashed are the same defect from the other side."""
    monkeypatch.delenv("FREETOKEN_MTP_MAX_STASH_TOKENS", raising=False)
    hit = _req(6, cached_len=2)
    _stash(hit)
    assert hit.spec_off and hit.spec_off_reason == "prefix_hit"
    assert "stashed=0" in hit.spec_off_detail

    req = _req(8, chunked=True)
    _stash(req, rows=6)
    req.cached_len = 9  # committed rows the head never saw a residual for
    req.device_len = 11
    _stash(req, rows=2)
    assert req.spec_off and req.spec_off_reason == "prefix_hit"
    # an intermediate chunk's input_ids is a prefix slice, so the line must not call its length
    # the prompt: 9 rows in would otherwise read as a 9-token prompt
    assert "rows=9" in req.spec_off_detail and "prompt=" not in req.spec_off_detail
    assert req.spec_residual is None  # the partial span goes with the decision


def test_cold_start_turns_those_declines_into_no_stash(monkeypatch):
    """With FREETOKEN_MTP_COLD_START the span rule stops being an eligibility test: the request
    keeps its right to draft, it just gets no catch-up pass -- so the gate drops whatever partial
    span it holds (a span with a hole at the END catches the head up on nothing useful) and
    records no reason at all."""
    import freetoken.engine.engine as eng_mod

    monkeypatch.delenv("FREETOKEN_MTP_MAX_STASH_TOKENS", raising=False)
    monkeypatch.setattr(eng_mod, "mtp_cold_start", lambda: True)

    hit = _req(6, cached_len=2)
    _stash(hit)
    assert not hit.spec_off and hit.spec_residual is None
    assert hit.spec_off_reason == ""

    req = _req(8, chunked=True)
    _stash(req, rows=6)
    assert req.spec_residual.shape[0] == 6  # still contiguous so far: the span is kept
    req.cached_len = 9
    req.device_len = 11
    _stash(req, rows=2)
    assert not req.spec_off and req.spec_residual is None


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
