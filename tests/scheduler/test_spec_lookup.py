"""Lookup spec on the scheduler side: eligibility, staging, and the acceptance gate.

The heavy lifting (the trigram index and the gate's window) is covered in
``tests/engine/test_lookup_drafter.py``; these cases pin the scheduler contract: a single
greedy request whose suffix repeats gets a draft staged exactly like MTP's (pool + host buffer
+ device_len bump + verify phase), a request without a match keeps the regular decode step,
and the terminal declines are counted instead of silently skipped.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.engine.lookup import LookupState
from freetoken.scheduler.scheduler import Scheduler


def _req(prompt, *, temperature=0.0, max_tokens=64, cached_len=None):
    tokens = torch.tensor(prompt, dtype=torch.int32)
    return Req(
        input_ids=tokens,
        table_idx=0,
        cached_len=len(prompt) - 1 if cached_len is None else cached_len,
        output_len=max_tokens,
        uid=11,
        sampling_params=SamplingParams(temperature=temperature),
        cache_handle=None,
    )


def _sched():
    calls = {"reject": [], "alloc": [], "free": []}

    def alloc(n):
        calls["alloc"].append(n)
        return [3 + len(calls["alloc"]) - 1]

    sched = SimpleNamespace(
        config=SimpleNamespace(speculative="lookup"),
        engine=SimpleNamespace(
            linear_state_pool=SimpleNamespace(
                alloc=alloc, free=lambda idx: calls["free"].append(idx)
            )
        ),
        token_pool=torch.zeros(4, 128, dtype=torch.int32),
        _spec_reject=lambda req, reason: (
            calls["reject"].append(reason),
            setattr(req, "spec_off", True),
            Scheduler._release_spec_scratch(sched, req),
        ),
        _count_spec_rejection=lambda reason, req=None: calls["reject"].append(reason),
        _spec_win_verifies=0,
        _spec_win_accepted=0,
        _spec_verifies=0,
        _spec_accepted=0,
        _report_spec_tally=lambda force=False: None,
        _release_spec_scratch=lambda req: Scheduler._release_spec_scratch(sched, req),
        _release_lookup_scratch_all=lambda reqs: Scheduler._release_lookup_scratch_all(
            sched, reqs
        ),
        _as_lookup_batch_multi=lambda b: Scheduler._as_lookup_batch_multi(sched, b),
    )
    return sched, calls


def _batch(req):
    return SimpleNamespace(reqs=[req], spec_mode=None, spec_source="mtp", phase="decode")


def test_repeat_stages_one_draft_like_mtp():
    # history 1..9 then 1,2,3 repeated: the suffix ending at the current row (3) has an
    # earlier occurrence whose continuation is 4.
    req = _req([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3], cached_len=11)
    req.device_len = 12
    req.lookup = LookupState(max_draft=1, min_context=0)
    sched, calls = _sched()
    batch = _batch(req)

    out = Scheduler._as_lookup_batch(sched, batch, req)

    assert out is batch
    assert batch.spec_mode == "verify" and batch.spec_source == "lookup"
    assert batch.spec_sample is False and batch.phase == "prefill"
    assert batch.spec_draft_id == 4
    assert req.device_len == 13
    assert req.spec_draft == 4
    assert req.input_ids.tolist()[-1] == 4
    assert int(sched.token_pool[0, 12]) == 4
    assert req.spec_slot_idx == 3 and calls["alloc"] == [1]
    assert calls["reject"] == []


def test_no_repeat_keeps_the_regular_decode_step():
    req = _req([1, 2, 3, 4, 5, 6, 7], cached_len=6)
    req.device_len = 7
    sched, calls = _sched()
    assert Scheduler._as_lookup_batch(sched, _batch(req), req) is None
    assert req.spec_draft is None and req.device_len == 7
    assert calls["reject"] == []  # a miss is not a decline: the next step looks again


def test_sampling_is_declined_before_any_draft():
    req = _req([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3], temperature=0.7, cached_len=11)
    req.device_len = 12
    sched, calls = _sched()
    assert Scheduler._as_lookup_batch(sched, _batch(req), req) is None
    assert calls["reject"] == ["nongreedy"]
    assert req.spec_off


def test_one_output_row_left_is_declined():
    req = _req([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3], max_tokens=1, cached_len=11)
    req.device_len = 12
    sched, calls = _sched()
    assert Scheduler._as_lookup_batch(sched, _batch(req), req) is None
    assert calls["reject"] == ["no_room"]


def test_second_verify_uses_the_committed_token_not_the_staged_draft():
    # after an accept the host buffer holds y1 at the old draft slot and y2 at the next row;
    # the index must see both before proposing again (device_len is the committed view).
    req = _req([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3], cached_len=11)
    req.device_len = 12
    req.lookup = LookupState(min_context=0)
    sched, _ = _sched()
    Scheduler._as_lookup_batch(sched, _batch(req), req)  # stages draft 4
    req.input_ids[12] = 4
    req.append_host(torch.tensor([5], dtype=torch.int32))
    req.cached_len, req.device_len = 13, 14  # accept: y1=4, y2=5 placed
    Scheduler._as_lookup_batch(sched, _batch(req), req)
    assert req.lookup.propose() is None or req.lookup.last_match >= 3


def test_the_gate_turns_the_request_off_after_a_bad_window():
    req = _req([1, 2, 3, 4], cached_len=3)
    req.device_len = 4
    req.lookup = LookupState(accept_window=8, min_acceptance=0.5, min_context=0)
    sched, calls = _sched()
    for _ in range(8):
        Scheduler._tally_lookup_verify(sched, req, 0, 1)
    assert req.spec_off
    assert req.lookup is None  # _release_spec_scratch drops it when spec_off is set
    assert calls["reject"] == ["low_acceptance"]
    assert sched._spec_verifies == 8 and sched._spec_accepted == 0

def test_three_drafts_are_staged_in_order():
    req = _req([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3], cached_len=11)
    req.device_len = 12
    req.lookup = LookupState(max_draft=3, min_context=0)
    sched, calls = _sched()
    batch = _batch(req)

    out = Scheduler._as_lookup_batch(sched, batch, req)

    assert out is batch
    assert batch.spec_draft_ids == (4, 5, 6)
    assert batch.spec_draft_id == 4
    assert batch.spec_prefix_len == 11
    assert req.device_len == 15
    assert req.input_ids.tolist()[-3:] == [4, 5, 6]
    assert sched.token_pool[0, 12:15].tolist() == [4, 5, 6]


def _drain_stub(req, spec, draft_ids):
    calls = {"copy": [], "release": [], "slot_freed": []}
    sched = SimpleNamespace(
        config=SimpleNamespace(page_size=64),
        decode_manager=SimpleNamespace(remove_req=lambda r: None),
        _free_req_resources=lambda r: None,
        engine=SimpleNamespace(linear_state_pool=SimpleNamespace(
            copy_from=lambda src, dst: calls["copy"].append((src, dst)),
            free=lambda slot: calls["slot_freed"].append(slot))),
        token_pool=torch.zeros(4, 128, dtype=torch.int32),
        cache_manager=SimpleNamespace(release_paged=lambda info: calls["release"].append(info)),
        eos_token_ids={2},
        _match_stop_str=lambda r: None,
        toolcall_anchor_id=None,
        finished_reqs=set(),
        _tally_lookup_verify=lambda r, a, d: calls.update(
            tally=(r.uid, a, d)),
        _release_spec_scratch=lambda r: Scheduler._release_spec_scratch(sched, r),
        _count_spec_rejection=lambda reason, r=None: None,
        _report_spec_tally=lambda force=False: None,
        _spec_verifies=0,
        _spec_accepted=0,
        _spec_win_verifies=0,
        _spec_win_accepted=0,
    )
    sched._restore_spec_prefix = lambda b, **kw: Scheduler._restore_spec_prefix(sched, b, **kw)
    sched._drain_spec_single = lambda b, sp, r, f: Scheduler._drain_spec_single(
        sched, b, sp, r, f
    )
    sched._drain_lookup_one = lambda b, item, sp, r, f: Scheduler._drain_lookup_one(
        sched, b, item, sp, r, f
    )
    sched._emit_spec_tokens = lambda req, toks, skip, r, f: Scheduler._emit_spec_tokens(
        sched, req, toks, skip, r, f
    )
    req.spec_slot_idx = 3
    req.linear_slot_idx = 1
    batch = SimpleNamespace(
        reqs=[req], spec_mode="verify", spec_source="lookup", spec_pages=None,
        spec_draft_ids=tuple(draft_ids), spec_prefix_len=11,
    )
    reply, finished = [], set()
    Scheduler._drain_spec(sched, batch, spec, reply, finished)
    return reply, finished, calls


def test_full_accept_emits_every_draft_and_the_bonus():
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6]
    req = _req(prompt, cached_len=11)
    req.device_len = 15
    reply, finished, calls = _drain_stub(
        req, {"ids": [4, 5, 6, 7], "accept": 3, "full": True, "y1": 4, "y2": 5},
        (4, 5, 6),
    )
    assert [m.next_token for m in reply] == [4, 5, 6, 7]
    assert req.cached_len == 15 and req.device_len == 16
    assert req.input_ids.tolist()[-1] == 7
    assert calls["tally"] == (11, 3, 3)
    assert calls["copy"] == [] and not finished


def test_partial_accept_keeps_only_the_verified_first_token():
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6]
    req = _req(prompt, cached_len=11)
    req.device_len = 15
    reply, finished, calls = _drain_stub(
        req, {"ids": [4, 9, 9, 9], "accept": 1, "full": False, "y1": 4, "y2": 9},
        (4, 5, 6),
    )
    assert [m.next_token for m in reply] == [4]
    assert req.cached_len == 12 and req.device_len == 13
    assert req.input_ids.tolist()[-1] == 4 and req.input_ids.numel() == 13
    assert calls["copy"] == [(3, 1)]
    assert calls["tally"] == (11, 1, 3)


def test_short_history_does_not_draft_until_it_grows():
    req = _req([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3], cached_len=11)
    req.device_len = 12
    req.lookup = LookupState(min_context=64)
    sched, calls = _sched()
    batch = _batch(req)

    assert Scheduler._as_lookup_batch(sched, batch, req) is None
    assert not batch.spec_mode and req.device_len == 12
    assert calls["reject"] == []


def test_full_accept_that_exactly_fills_the_budget_emits_every_token():
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6]
    req = _req(prompt, cached_len=11, max_tokens=1)  # max_device_len = 16
    req.device_len = 15
    reply, finished, _ = _drain_stub(
        req, {"ids": [4, 5, 6, 7], "accept": 3, "full": True, "y1": 4, "y2": 5},
        (4, 5, 6),
    )
    assert [m.next_token for m in reply] == [4, 5, 6, 7]
    assert [m.finished for m in reply] == [False, False, False, True]
    assert reply[-1].finish_reason == "length"
    assert req.device_len == req.max_device_len == 16


def test_parallel_batch_releases_a_scratch_slot_left_from_a_lone_step():
    # lookup reserves one GDN scratch slot; a request that drafted while alone must not park
    # it through the whole parallel stretch. _as_spec_batch runs on every decode batch, so
    # the mixed path is where the slot goes back to the free-list.
    a = _req([1, 2, 3], cached_len=2)
    b = _req([4, 5, 6], cached_len=2)
    a.spec_slot_idx, b.spec_slot_idx = 3, 4
    a.spec_draft, b.spec_draft = 7, 8
    sched, calls = _sched()
    batch = SimpleNamespace(reqs=[a, b])

    assert Scheduler._as_spec_batch(sched, batch) is None
    assert a.spec_slot_idx is None and b.spec_slot_idx is None
    assert calls["free"] == [3, 4]
    assert calls["reject"] == []


def test_two_eligible_requests_are_staged_together():
    a = _req([1, 2, 3, 4, 1, 2, 3], cached_len=6)
    b = _req([1, 2, 3, 4, 1, 2, 3], cached_len=6)
    a.device_len = b.device_len = 7
    a.table_idx, b.table_idx = 0, 1
    a.lookup = LookupState(max_draft=2, min_context=0)
    b.lookup = LookupState(max_draft=2, min_context=0)
    sched, calls = _sched()
    batch = SimpleNamespace(reqs=[a, b], spec_mode=None, spec_source="mtp", phase="decode")

    out = Scheduler._as_spec_batch(sched, batch)

    assert out is batch
    assert batch.spec_mode == "verify" and batch.spec_source == "lookup"
    assert len(batch.spec_items) == 2
    assert [item["req"].table_idx for item in batch.spec_items] == [0, 1]
    assert [item["drafts"] for item in batch.spec_items] == [(4, 1), (4, 1)]
    assert [item["prefix"] for item in batch.spec_items] == [6, 6]
    assert a.device_len == b.device_len == 9
    assert a.spec_slot_idx == 3 and b.spec_slot_idx == 4
    assert calls["alloc"] == [1, 1] and calls["free"] == []


def test_four_eligible_requests_are_staged_together(monkeypatch):
    from freetoken import env as env_mod

    monkeypatch.setattr(env_mod.ENV, "LOOKUP_MAX_BATCH", SimpleNamespace(value=4))
    reqs = [_req([1, 2, 3, 4, 1, 2, 3], cached_len=6) for _ in range(4)]
    for i, req in enumerate(reqs):
        req.device_len = 7
        req.table_idx = i
        req.lookup = LookupState(max_draft=2, min_context=0)
    sched, calls = _sched()
    batch = SimpleNamespace(reqs=reqs, spec_mode=None, spec_source="mtp", phase="decode")

    out = Scheduler._as_spec_batch(sched, batch)

    assert out is batch
    assert len(batch.spec_items) == 4
    assert [item["req"].table_idx for item in batch.spec_items] == [0, 1, 2, 3]
    assert [item["drafts"] for item in batch.spec_items] == [(4, 1)] * 4
    assert [r.spec_slot_idx for r in reqs] == [3, 4, 5, 6]
    assert calls["alloc"] == [1, 1, 1, 1] and calls["free"] == []


def test_multi_drain_accepts_and_rejects_independently():
    calls = {"copy": [], "release": [], "tally": []}
    sched = SimpleNamespace(
        config=SimpleNamespace(page_size=64),
        decode_manager=SimpleNamespace(remove_req=lambda r: None),
        _free_req_resources=lambda r: None,
        engine=SimpleNamespace(linear_state_pool=SimpleNamespace(
            copy_from=lambda src, dst: calls["copy"].append((src, dst)),
            free=lambda slot: calls["slot_freed"].append(slot))),
        token_pool=torch.zeros(4, 128, dtype=torch.int32),
        cache_manager=SimpleNamespace(release_paged=lambda info: calls["release"].append(info)),
        eos_token_ids={2},
        _match_stop_str=lambda r: None,
        toolcall_anchor_id=None,
        finished_reqs=set(),
        _tally_lookup_verify=lambda r, a, d: calls["tally"].append((r.uid, a, d)),
        _release_spec_scratch=lambda r: Scheduler._release_spec_scratch(sched, r),
        _count_spec_rejection=lambda reason, r=None: None,
        _report_spec_tally=lambda force=False: None,
        _spec_verifies=0,
        _spec_accepted=0,
        _spec_win_verifies=0,
        _spec_win_accepted=0,
    )
    sched._restore_spec_prefix = lambda b, **kw: Scheduler._restore_spec_prefix(sched, b, **kw)
    sched._drain_lookup_one = lambda b, item, sp, r, f: Scheduler._drain_lookup_one(
        sched, b, item, sp, r, f
    )
    sched._emit_spec_tokens = lambda req, toks, skip, r, f: Scheduler._emit_spec_tokens(
        sched, req, toks, skip, r, f
    )
    a = _req([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6], cached_len=11)
    b = _req([1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6], cached_len=11)
    a.uid, b.uid = 11, 12
    a.table_idx, b.table_idx = 0, 1
    a.device_len = b.device_len = 15
    a.spec_slot_idx, b.spec_slot_idx = 3, 4
    a.linear_slot_idx, b.linear_slot_idx = 1, 2
    items = [
        {"req": a, "drafts": (4, 5, 6), "prefix": 11},
        {"req": b, "drafts": (4, 5, 6), "prefix": 11},
    ]
    payloads = [
        {"ids": [4, 5, 6, 7], "accept": 3, "full": True, "y1": 4, "y2": 5},
        {"ids": [4, 9, 9, 9], "accept": 1, "full": False, "y1": 4, "y2": 9},
    ]
    batch = SimpleNamespace(
        reqs=[a, b], spec_mode="verify", spec_source="lookup", spec_pages=None,
        spec_items=items, spec_prefix_len=11,
    )
    reply, finished = [], set()
    Scheduler._drain_spec(sched, batch, {"items": payloads}, reply, finished)

    assert [m.uid for m in reply].count(11) == 4
    assert [m.uid for m in reply].count(12) == 1
    assert a.cached_len == 15 and a.device_len == 16
    assert b.cached_len == 12 and b.device_len == 13
    assert b.input_ids.tolist()[-1] == 4
    assert calls["tally"] == [(11, 3, 3), (12, 1, 3)]
    assert calls["copy"] == [(4, 2)]  # only the partial request needs the snapshot


def test_batch_above_the_lookup_ceiling_falls_back_and_releases_slots():
    reqs = [_req([1, 2, 3], cached_len=2) for _ in range(4)]
    for i, req in enumerate(reqs):
        req.table_idx = i
        req.spec_slot_idx = 10 + i
    sched, calls = _sched()
    batch = SimpleNamespace(reqs=reqs)

    assert Scheduler._as_spec_batch(sched, batch) is None
    assert [r.spec_slot_idx for r in reqs] == [None] * 4
    assert calls["free"] == [10, 11, 12, 13]
