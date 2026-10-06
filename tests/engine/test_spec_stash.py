"""Engine-side MTP residual stash gate (engine/engine.py:_stash_spec_residual), CPU-only.

Prefill decides here whether a request may draft later: the rows the target forward did
not run the MTP head on are handed to the first decode step as a prologue, which is what
keeps acceptance at 100% on the prompt instead of starting from a cold head. Two failure
modes are silent and expensive, so they are pinned on the host-side state only:

  * stashing anyway when the verify could never run (chunked prefill, shared prefix, no
    GDN pool, temperature>0, multimodal spans). The prologue then runs the head over rows
    nothing verified, and the first mismatch costs a rollback with no committed state to
    roll back to -- the same wrongness the pre-MTP loop had, paid for with a draft model.
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


def _engine(max_extend_tokens: int = 8192) -> Engine:
    eng = Engine.__new__(Engine)
    eng.config = SimpleNamespace(max_extend_tokens=max_extend_tokens)
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
        "nongreedy": _req(6, temperature=0.7),
        "mm": _req(6, mm_items=[{"fake": "image"}]),
    }
    for reason, req in cases.items():
        _stash(req)
        assert req.spec_off is True, reason
        assert req.spec_off_reason == reason
        assert req.spec_residual is None


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