"""_drain_spec bookkeeping (MTP): the accept/reject host-side state machine,
CPU-only with a stub scheduler and real Req objects.

The invariants under test are the ones the two-token verify lives or dies by:
accept completes once and appends two tokens (the live slot already holds the
state after both rows); reject restores the scratch slot's post-row-0 state,
commits exactly one input token (cached_len = device_len - 1), repairs the
draft's pool slot with y1, keeps the fresh row-0 draft and releases only the
pages beyond the committed prefix; an EOS in the first emitted token drops the
bonus token and finishes the request.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.scheduler import Scheduler

P = 5  # position of the last placed token entering a verify step


def _req(host_len=P + 1, max_extra=20):
    return Req(
        input_ids=torch.arange(host_len, dtype=torch.int32),
        table_idx=0, cached_len=P, output_len=host_len + max_extra - P - 1 + 10,
        uid=7, sampling_params=SamplingParams(temperature=0.0), cache_handle=None,
    )


def _setup(req, mode="verify", pages=None):
    calls = {"copy_from": [], "release": [], "freed": [], "removed": [], "slot_freed": []}
    pool = torch.zeros(4, 256, dtype=torch.int32)
    pool[0, P + 1] = 999  # the draft's staged pool slot (verify bump wrote it)
    pool[0, P + 2] = 42   # the engine's next_tokens_gpu write (Scheduler._forward)

    def free_resources(r):
        # mirrors the real _free_req_resources spec cleanup (it delegates to
        # _release_spec_scratch, which the bound method below runs for real)
        calls["freed"].append(r)
        sched._release_spec_scratch(r)
        r.table_idx = -1

    sched = SimpleNamespace(
        config=SimpleNamespace(page_size=64),
        decode_manager=SimpleNamespace(remove_req=lambda r: calls["removed"].append(r)),
        _free_req_resources=free_resources,
        engine=SimpleNamespace(linear_state_pool=SimpleNamespace(
            copy_from=lambda src, dst: calls["copy_from"].append((src, dst)),
            free=lambda slot: calls["slot_freed"].append(slot))),
        token_pool=pool,
        cache_manager=SimpleNamespace(
            release_paged=lambda info: calls["release"].append(info)),
        eos_token_ids={2},
        _match_stop_str=lambda r: None,
        toolcall_anchor_id=None,
        finished_reqs=set(),
    )
    req.spec_slot_idx = 3
    sched._restore_spec_prefix = lambda b: Scheduler._restore_spec_prefix(sched, b)
    sched._count_spec_rejection = lambda reason, r=None: Scheduler._count_spec_rejection(sched, reason, r)
    sched._log_spec_decline = lambda reason, r: Scheduler._log_spec_decline(sched, reason, r)
    sched._decline_detail = lambda r: Scheduler._decline_detail(sched, r)
    sched._spec_reject_summary = lambda: Scheduler._spec_reject_summary(sched)
    sched._spec_coverage = lambda: Scheduler._spec_coverage(sched)
    sched._release_spec_scratch = lambda r: Scheduler._release_spec_scratch(sched, r)
    sched._spec_reject = lambda r, reason: Scheduler._spec_reject(sched, r, reason)
    sched._tally_verify = lambda r, a: Scheduler._tally_verify(sched, r, a)
    req.linear_slot_idx = 1
    batch = SimpleNamespace(reqs=[req], spec_mode=mode, spec_pages=pages)
    sched._report_spec_tally = lambda force=False: Scheduler._report_spec_tally(sched, force)
    return sched, batch, calls, pool


def _drain(sched, batch, spec):
    reply, finished = [], set()
    Scheduler._drain_spec(sched, batch, spec, reply, finished)
    return reply, finished


def test_accept_completes_once_and_appends_two_tokens():
    # _as_spec_batch staged the draft onto the host buffer already (slot P+1 = y1)
    req = _req(host_len=P + 2)
    req.input_ids[P + 1] = 41
    req.device_len = P + 2  # the verify bump
    sched, batch, calls, pool = _setup(req)
    reply, finished = _drain(sched, batch,
                             {"y1": 41, "y2": 42, "accept": True, "draft": 77})
    assert req.input_ids.tolist()[-2:] == [41, 42]
    assert req.cached_len == P + 2 and req.device_len == P + 3
    assert req.spec_draft == 77
    assert [m.next_token for m in reply] == [41, 42]
    assert not any(m.finished for m in reply) and not finished
    assert calls["copy_from"] == [] and calls["release"] == []
    assert int(pool[0, P + 2]) == 42  # the engine's pool write landed at the new position


def test_reject_restores_first_token_state_and_keeps_a_fresh_draft():
    req = _req(host_len=P + 2)
    req.input_ids[P + 1] = 999  # the staged draft, to be overwritten with y1
    req.device_len = P + 2
    req.spec_draft = 999
    sched, batch, calls, pool = _setup(req, pages=[(0, 1, 2)])
    reply, finished = _drain(sched, batch,
                             {"y1": 41, "y2": 4242, "accept": False, "draft": 77})
    assert calls["copy_from"] == [(3, 1)]  # scratch -> live
    assert int(pool[0, P + 1]) == 41  # the draft's pool slot repaired with the true token
    assert calls["release"] == [[(0, 1, 2)]]
    assert req.spec_draft == 77
    assert req.input_ids.tolist()[-1] == 41 and req.input_ids.numel() == P + 2  # slot repaired, not appended
    # the pre-step lens: the replay step re-reads rows [P, P+1] = [x_P, y1]
    assert req.cached_len == P + 1 and req.device_len == P + 2
    assert [m.next_token for m in reply] == [41] and not finished


def test_eos_on_the_first_token_drops_the_bonus_and_finishes():
    req = _req(host_len=P + 2)
    req.input_ids[P + 1] = 2  # the staged draft == y1 == eos
    req.device_len = P + 2
    sched, batch, calls, _pool = _setup(req)
    reply, finished = _drain(sched, batch,
                             {"y1": 2, "y2": 42, "accept": True, "draft": 77})
    assert [m.next_token for m in reply] == [2]
    assert reply[0].finished and reply[0].finish_reason == "stop"
    assert req in finished and calls["removed"] == [req] and calls["freed"] == [req]
    assert req.spec_slot_idx is None  # the scratch slot was released with the request


def test_length_finish_on_the_second_token():
    req = Req(input_ids=torch.arange(P + 2, dtype=torch.int32), table_idx=0,
              cached_len=P, output_len=0, uid=7,
              sampling_params=SamplingParams(temperature=0.0), cache_handle=None)
    # max_device_len = P+2: the staged draft already exhausts the budget, so the
    # accept's complete_one leaves no room and the first emitted token finishes
    req.device_len = P + 2
    sched, batch, _calls, _pool = _setup(req)
    reply, finished = _drain(sched, batch,
                             {"y1": 41, "y2": 42, "accept": True, "draft": 77})
    assert [m.next_token for m in reply] == [41]
    assert reply[0].finished and reply[0].finish_reason == "length"
    assert req in finished


def _upgrade_sched(req):
    writes = {}

    class _Pool(dict):
        def __setitem__(self, key, value):
            writes[key] = value
            super().__setitem__(key, value)

    sched = SimpleNamespace(
        config=SimpleNamespace(speculative="mtp"),
        engine=SimpleNamespace(linear_state_pool=SimpleNamespace(
            alloc=lambda n: [5], free=lambda slot: None)),
        token_pool=_Pool(),
        _build_spec_prologue=lambda r: "PROLOGUE",
        _spec_rejections={},
        _spec_decline_lines={},
    )
    sched._spec_reject = lambda r, reason: Scheduler._spec_reject(sched, r, reason)
    sched._count_spec_rejection = (
        lambda reason, r=None: Scheduler._count_spec_rejection(sched, reason, r))
    sched._log_spec_decline = lambda reason, r: Scheduler._log_spec_decline(sched, reason, r)
    sched._decline_detail = lambda r: Scheduler._decline_detail(sched, r)
    sched._release_spec_scratch = lambda r: Scheduler._release_spec_scratch(sched, r)
    return sched, writes


def _spec_req(host_len, output_len, cached_len, device_len=None, **spec):
    req = Req(input_ids=torch.arange(host_len, dtype=torch.int32), table_idx=0,
              cached_len=cached_len, output_len=output_len, uid=7,
              sampling_params=SamplingParams(temperature=0.0), cache_handle=None)
    req.device_len = device_len if device_len is not None else host_len
    for k, v in spec.items():
        setattr(req, k, v)
    return req


def test_verify_needs_room_for_two_tokens():
    from freetoken.core import Batch
    from freetoken.scheduler.scheduler import Scheduler

    # remain_len == 1: an accept would append two tokens past the budget -> stay regular, and the
    # draft is declined as the end of the generation rather than as a lost batch (see no_room)
    req = _spec_req(P + 1, 1, P, spec_draft=42, spec_slot_idx=5)
    sched, writes = _upgrade_sched(req)
    batch = Batch(reqs=[req], phase="decode")
    assert Scheduler._as_spec_batch(sched, batch) is None
    assert req.device_len == P + 1 and batch.spec_mode is None and not writes
    assert req.spec_off and req.spec_off_reason == "no_room"
    assert req.spec_slot_idx is None  # the scratch slot came back with the draft

    # remain_len == 2: the verify upgrade stages the draft and bumps the row count
    req2 = _spec_req(P + 1, 2, P, spec_draft=42, spec_slot_idx=5)
    sched2, writes2 = _upgrade_sched(req2)
    batch2 = Batch(reqs=[req2], phase="decode")
    out = Scheduler._as_spec_batch(sched2, batch2)
    assert out is batch2 and batch2.spec_mode == "verify" and batch2.phase == "prefill"
    assert req2.device_len == P + 2 and writes2[(0, P + 1)] == 42
    # the draft is staged on the HOST buffer too (the PLE extend fill reads it)
    assert req2.input_ids.numel() == P + 2 and int(req2.input_ids[P + 1]) == 42



@pytest.mark.parametrize("position,pages,released", [
    (63, [(0, 1, 2)], [[(0, 1, 2)]]),
    (64, [(0, 1, 2)], []),
    (65, None, []),
])
def test_reject_keeps_only_pages_containing_committed_inputs(position, pages, released):
    req = _spec_req(position + 2, 10, position, spec_slot_idx=3)
    sched, batch, calls, pool = _setup(req, pages=pages)
    _drain(sched, batch, {"y1": 41, "y2": 42, "accept": False, "draft": 77})
    assert req.cached_len == position + 1
    assert req.device_len == position + 2
    assert calls["release"] == released
    sched.config.speculative = "mtp"
    from freetoken.core import Batch
    following = Batch(reqs=[req], phase="decode")
    assert Scheduler._as_spec_batch(sched, following) is following
    assert following.spec_mode == "verify"
    assert req.extend_len == 2
    assert int(pool[0, position + 2]) == 77


@pytest.mark.parametrize("aborted", [False, True])
def test_reject_or_abort_at_page_boundary_releases_uncommitted_page(aborted):
    req = _spec_req(65, 10, 63, spec_slot_idx=3)
    req.aborted = aborted
    sched, batch, calls, _ = _setup(req, pages=[(0, 1, 2)])
    reply, finished = _drain(sched, batch,
                             {"y1": 2, "y2": 42, "accept": False, "draft": 77})
    assert req in finished
    assert req.cached_len == 64
    assert calls["copy_from"] == [(3, 1)]
    assert calls["release"] == [[(0, 1, 2)]]
    assert calls["freed"] == [req]
    assert [r.next_token for r in reply] == ([] if aborted else [2])


def _sched_for_batch(batch):
    sched = Scheduler.__new__(Scheduler)
    sched.config = SimpleNamespace(speculative="mtp", prefill_decode_interval=0)
    sched._prefill_debt = 0.0
    sched._prefill_debt_s = 0.0
    sched._chunk_target_s = 0.0
    sched._chunk_ema_spt = None
    sched.prefill_budget = 64
    sched._prefill_streak = 0
    sched._spec_rejections = {}
    sched.token_pool = {}
    sched._build_spec_prologue = lambda r: "PROLOGUE"
    sched.engine = SimpleNamespace(linear_state_pool=SimpleNamespace(free=lambda s: None))
    sched.prefill_manager = SimpleNamespace(schedule_next_batch=lambda budget: None)
    sched.decode_manager = SimpleNamespace(schedule_next_batch=lambda: batch)
    sched._prepare_batch = lambda b: b
    sched._report_prompt_admissions = lambda b: None
    return sched


def test_a_mixed_batch_resyncs_a_started_draft_not_kills_it():
    """A decode batch that grew past one request used to END drafting for every request that
    had already started (the reason read as batch_killed). The head's KV is still valid for the
    rows it saw, so only the staged draft -- now aimed at a committed position -- is worthless:
    drop it and keep drafting."""
    from freetoken.core import Batch

    started = _spec_req(P + 1, 10, P, spec_draft=42)
    started.spec_head_len = P  # the head consumed the prompt; its KV covers [0, P)
    waiting = _spec_req(P + 1, 10, P, spec_residual=torch.ones(2, 4))
    waiting.spec_draft = 7  # never reached a spec step; the draft is stale
    batch = Batch(reqs=[started, waiting], phase="decode")
    sched = _sched_for_batch(batch)
    assert sched._schedule_next_batch() is batch

    assert not started.spec_off and started.spec_draft is None
    assert waiting.spec_draft is None and waiting.spec_residual is not None
    # counted once, and only when a draft was actually thrown away
    assert sched._spec_rejections == {"resync": 1}
    sched._on_regular_decode(started)
    assert sched._spec_rejections == {"resync": 1}

    # both requests are drafting again the moment they run alone
    sched.engine.linear_state_pool.alloc = lambda n: list(range(n))
    alone = Batch(reqs=[waiting], phase="decode")
    alone.padded_reqs = [waiting]
    upgraded = sched._as_spec_batch(alone)
    assert upgraded is not None and upgraded.spec_mode == "prologue_decode"
    reseed = Batch(reqs=[started], phase="decode")
    reseed.padded_reqs = [started]
    assert sched._as_spec_batch(reseed) is reseed
    assert reseed.spec_mode == "prologue_decode"
    assert reseed.spec_prologue is None  # a re-seed: no catch-up pass behind it
    assert sched._spec_cold_seeds == 1


def test_stash_beyond_the_lag_bound_is_released():
    """A request that never runs alone must not pin its prompt-sized stash for its whole
    life: past the lag bound its drafts are not worth the second row."""
    from freetoken.scheduler.scheduler import _MTP_RESYNC_LAG as lag

    def drained(extra):
        # the head never saw the rows decoded since the stash was written
        return _spec_req(P + extra + 1, 400, P + extra, spec_residual=torch.ones(P, 4))

    late = drained(lag + 1)
    _sched_for_batch(None)._on_regular_decode(late)
    assert late.spec_off and late.spec_off_reason == "lag_overflow"
    assert late.spec_residual is None

    in_time = drained(1)
    _sched_for_batch(None)._on_regular_decode(in_time)
    assert not in_time.spec_off and in_time.spec_residual is not None


def test_scratch_slot_survives_a_reject_and_dies_with_the_draft():
    """The scratch slot is allocated once per request and reused: a reject copies it back
    into the live slot and keeps it for the next step's clone. What must not happen is it
    outliving the draft -- _spec_reject releases it, or the next admission cannot get it."""
    req = _req(host_len=P + 2)
    req.device_len = P + 2
    req.spec_draft = 999
    sched, batch, calls, _ = _setup(req, pages=[(0, 1, 2)])
    _drain(sched, batch, {"y1": 41, "y2": 4242, "accept": False, "draft": 77})
    assert calls["copy_from"] == [(3, 1)] and req.spec_slot_idx == 3

    sched._spec_reject(req, "low_acceptance")
    assert calls["slot_freed"] == [3] and req.spec_slot_idx is None


def test_low_acceptance_disables_further_drafting(monkeypatch):
    """With FREETOKEN_MTP_RESUME_AFTER=0 the floor is terminal -- the pre-suspend behaviour,
    kept as the arm an A/B runs against: every verify pays a second target row, so a request
    whose drafts keep missing stops paying for them."""
    import freetoken.scheduler.scheduler as sched_mod
    from freetoken.scheduler.scheduler import _MTP_ACCEPT_WINDOW

    monkeypatch.setattr(sched_mod, "_MTP_RESUME_AFTER", 0)
    req = _spec_req(P + 2, 10, P, spec_draft=7, spec_residual=torch.ones(2, 4))
    sched, _, _, _ = _setup(req)
    for _ in range(_MTP_ACCEPT_WINDOW - 1):
        sched._tally_verify(req, False)
    assert not req.spec_off
    sched._tally_verify(req, False)
    assert req.spec_off and req.spec_off_reason == "low_acceptance"
    assert req.spec_draft is None and req.spec_slot_idx is None
    assert req.spec_residual is None
    assert sched._spec_rejections.get("low_acceptance") == 1

    # a draft that keeps landing stays on, and the window resets
    good = _spec_req(P + 2, 10, P, spec_draft=7)
    sched, _, _, _ = _setup(good)
    for _ in range(2 * _MTP_ACCEPT_WINDOW):
        sched._tally_verify(good, True)
    assert not good.spec_off
    assert good.spec_verifies == 0  # window rolled over clean


def test_a_missed_window_stands_down_and_probes_again(monkeypatch):
    """One window is a small sample (at 0.45 true acceptance a 16-window lands under a 0.35
    floor ~20% of the time), so missing it pauses drafting instead of ending it -- and the
    regular steps it falls back to count the pause away without filing the lag as terminal."""
    import freetoken.scheduler.scheduler as sched_mod
    from freetoken.core import Batch
    from freetoken.scheduler.scheduler import _MTP_ACCEPT_WINDOW as window

    monkeypatch.setattr(sched_mod, "_MTP_RESUME_AFTER", 3)
    monkeypatch.setattr(sched_mod, "_MTP_REJECT_STREAK", 0)  # isolate the slow window from the streak
    sched = _sched_for_batch(None)
    sched.engine.linear_state_pool.alloc = lambda n: list(range(n))
    req = _spec_req(P + 2, 10, P, spec_draft=7, spec_head_len=P)
    for _ in range(window):
        sched._tally_verify(req, False)
    assert req.spec_suspend == 3 and not req.spec_off
    assert sched._spec_rejections == {"low_acceptance": 1}

    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    for left in (2, 1, 0):
        assert sched._as_spec_batch(batch) is None  # no upgrade while standing down
        batch.spec_mode = None
        sched._on_regular_decode(req)
        assert req.spec_suspend == left
    assert not req.spec_off  # the pause never became batch_killed / lag_overflow
    assert sched._spec_rejections == {"low_acceptance": 1}
    assert sched._as_spec_batch(batch) is batch  # resumed on the stale draft: one wasted verify
    assert batch.spec_mode == "verify"

def test_a_long_suspend_reseeds_instead_of_verifying_a_stale_draft(monkeypatch):
    """A stand-down longer than the resync lag leaves a hole the head never sees, and the
    staged draft (with its carried density) was computed that many committed tokens ago.
    Resuming on it verifies against the wrong position, so the countdown's end drops both
    and the next alone step cold-seeds a fresh draft -- counted, never terminal."""
    import freetoken.scheduler.scheduler as sched_mod
    from freetoken.core import Batch
    from freetoken.scheduler.scheduler import _MTP_ACCEPT_WINDOW as window

    monkeypatch.setattr(sched_mod, "_MTP_RESUME_AFTER", 10)
    monkeypatch.setattr(sched_mod, "_MTP_RESYNC_LAG", 4)
    monkeypatch.setattr(sched_mod, "_MTP_REJECT_STREAK", 0)
    sched = _sched_for_batch(None)
    sched.engine.linear_state_pool.alloc = lambda n: list(range(n))
    req = _spec_req(P + 2, 60, P, spec_draft=7, spec_head_len=P, spec_slot_idx=5)
    req.spec_draft_probs = torch.zeros(1, 4)
    for _ in range(window):
        sched._tally_verify(req, False)
    assert req.spec_suspend == 10 and not req.spec_off

    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    for _ in range(10):
        assert sched._as_spec_batch(batch) is None  # no upgrade while standing down
        batch.spec_mode = None
        sched._on_regular_decode(req)
    assert req.spec_draft is None and req.spec_draft_probs is None
    assert sched._spec_rejections == {"low_acceptance": 1, "suspend_hole": 1}
    assert req.spec_slot_idx == 5  # the scratch slot survives: the head KV is still usable
    assert not req.spec_off and req.spec_suspend_missed == 0
    out = sched._as_spec_batch(batch)
    assert out is batch and batch.spec_mode == "prologue_decode"
    assert batch.spec_prologue is None  # a re-seed: no catch-up pass behind it
    assert sched._spec_cold_seeds == 1


def test_a_short_suspend_still_resumes_on_the_stale_draft(monkeypatch):
    """Within the lag bound the hole is one the head already tolerates (a mixed batch
    leaves the same), so the staged draft is kept and its one wasted verify stays the
    cheaper half of the trade."""
    import freetoken.scheduler.scheduler as sched_mod
    from freetoken.core import Batch
    from freetoken.scheduler.scheduler import _MTP_ACCEPT_WINDOW as window

    monkeypatch.setattr(sched_mod, "_MTP_RESUME_AFTER", 3)
    monkeypatch.setattr(sched_mod, "_MTP_RESYNC_LAG", 8)
    monkeypatch.setattr(sched_mod, "_MTP_REJECT_STREAK", 0)
    sched = _sched_for_batch(None)
    sched.engine.linear_state_pool.alloc = lambda n: list(range(n))
    req = _spec_req(P + 2, 60, P, spec_draft=7, spec_head_len=P)
    for _ in range(window):
        sched._tally_verify(req, False)
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    for _ in range(3):
        assert sched._as_spec_batch(batch) is None
        batch.spec_mode = None
        sched._on_regular_decode(req)
    assert req.spec_draft == 7
    assert "suspend_hole" not in sched._spec_rejections
    assert sched._as_spec_batch(batch) is batch and batch.spec_mode == "verify"



def test_an_auto_off_reject_still_restores_the_scratch_slot(monkeypatch):
    """The low-acceptance auto-off frees the scratch slot, and a reject restores FROM it.

    Tallying before the commit therefore rolls back with `spec_slot_idx` already None: the GDN
    pool turns that into a tensor-shape error raised on the engine stream mid-decode, which
    takes the scheduler process down with it."""
    import freetoken.scheduler.scheduler as sched_mod
    from freetoken.scheduler.scheduler import _MTP_ACCEPT_WINDOW as window

    monkeypatch.setattr(sched_mod, "_MTP_RESUME_AFTER", 0)  # terminal arm: the slot must outlive it
    req = _spec_req(P + 2, 10, P, spec_draft=999, spec_verifies=window - 1)
    req.device_len = P + 2
    sched, batch, calls, _pool = _setup(req)
    _drain(sched, batch, {"y1": 41, "y2": 4242, "accept": False, "draft": 77})
    assert calls["copy_from"] == [(3, 1)]  # restored while the slot was still its own
    assert req.spec_off and req.spec_off_reason == "low_acceptance"
    assert calls["slot_freed"] == [3] and req.spec_slot_idx is None
    assert req.spec_draft is None  # the draft went with the decision, after the commit


def test_declines_are_counted_once_per_request():
    """The reason counters are what the periodic acceptance line prints as "declined:", the
    only field number for why MTP is not helping a workload. A request declined on every
    step (a batch that never empties to one) must contribute one, not one per step."""
    sched = _sched_for_batch(None)
    assert sched._spec_reject_summary() == "-"

    req = _spec_req(P + 1, 10, P, spec_draft=7)
    req.spec_head_len = P  # started: every later regular decode re-declines it
    for _ in range(5):
        sched._on_regular_decode(req)
    assert sched._spec_rejections == {"resync": 1}

    sched._count_spec_rejection("nongreedy")
    assert sched._spec_reject_summary() == "nongreedy=1 resync=1"


def test_no_scratch_slot_declines_instead_of_raising():
    """LinearStatePool.alloc raises when the free-list is empty, and the scheduler has no
    way back out mid-decode -- so the raise must become a decline for that one request."""
    from freetoken.core import Batch

    req = _spec_req(P + 1, 10, P, spec_residual=torch.ones(P, 4))
    sched = _sched_for_batch(None)

    def boom(n):
        raise RuntimeError("LinearStatePool exhausted")

    sched.engine.linear_state_pool.alloc = boom
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    assert sched._as_spec_batch(batch) is None
    assert req.spec_off and req.spec_off_reason == "exhausted"
    assert req.spec_residual is None  # the stash went with the decision


def _spec_logs(monkeypatch):
    """Capture what the scheduler prints.

    The MTP lines are the only field number for whether drafting ran and why not, so their
    existence -- not just the counters behind them -- is part of the behaviour under test."""
    import freetoken.scheduler.scheduler as sched_mod

    lines: list[str] = []
    monkeypatch.setattr(sched_mod.logger, "info_rank0", lambda msg: lines.append(msg))
    return lines


def test_a_decline_prints_where_it_is_counted(monkeypatch):
    """The decline breakdown used to live only inside the verify tally, so a workload where
    nothing ever qualified printed nothing and read exactly like MTP being switched off."""
    lines = _spec_logs(monkeypatch)
    sched = _sched_for_batch(None)
    req = _spec_req(P + 1, 10, P)
    sched._spec_reject(req, "no_stash")
    assert any(
        l.startswith("MTP spec: declined no_stash") and f"req {req.uid}" in l for l in lines
    )
    n = len(lines)
    sched._spec_reject(req, "batch_killed")  # already off: one count, one line
    assert len(lines) == n and sched._spec_rejections == {"no_stash": 1}


def test_the_gates_own_numbers_reach_the_decline_line(monkeypatch):
    """The prefill forward decides before the drain has advanced the request, so it hands its
    snapshot over; the line must quote that rather than the drained (and now misleading) state."""
    lines = _spec_logs(monkeypatch)
    sched = _sched_for_batch(None)
    req = _spec_req(P + 1, 10, P)
    req.spec_off_detail = "prompt=35905 new=8192 cached=27712 stash_budget=8192"
    sched._count_spec_rejection("prefix_hit", req)
    line = next(l for l in lines if "declined prefix_hit" in l)
    assert "prompt=35905" in line and "stash_budget=8192" in line
    assert req.spec_off_detail == ""  # consumed: a later decline describes itself


def test_decline_lines_are_bounded_per_reason(monkeypatch):
    """One line per request is the design, but a workload of thousands of declined requests
    still needs a ceiling -- the counters, not the log, carry the total."""
    import freetoken.scheduler.scheduler as sched_mod

    lines = _spec_logs(monkeypatch)
    monkeypatch.setattr(sched_mod, "_MTP_DECLINE_LOG_CAP", 2)
    sched = _sched_for_batch(None)
    for i in range(5):
        req = _spec_req(P + 1, 10, P)
        req.uid = 100 + i
        sched._count_spec_rejection("prefix_hit", req)
    assert len([l for l in lines if "declined prefix_hit" in l]) == 2
    assert any("further requests without a line" in l for l in lines)
    assert sched._spec_rejections["prefix_hit"] == 5


def test_the_tally_prints_below_the_verify_window(monkeypatch):
    """A clock-driven flush is what makes a handful of verifies readable at all: the window
    alone would leave a short run silent for the rest of its life."""
    from freetoken.scheduler.scheduler import _MTP_ACCEPT_WINDOW

    lines = _spec_logs(monkeypatch)
    req = _spec_req(P + 2, 10, P, spec_draft=7)
    sched, _, _, _ = _setup(req)
    sched._tally_verify(req, True)
    sched._report_spec_tally()  # the drain's throttled call
    assert any("MTP spec: acceptance 1/1 = 1.000" in l for l in lines)

    before = len(lines)
    sched._report_spec_tally()
    sched._report_spec_tally(force=True)  # run_when_idle polls: nothing new to say
    assert len(lines) == before

    # the window still prints on its own, and labels what moved since the last line
    for _ in range(_MTP_ACCEPT_WINDOW):
        sched._tally_verify(req, True)
    assert any("(last " in l for l in lines[before:])


def test_a_fresh_stash_at_the_last_row_is_no_room_not_batch_killed():
    """Why a request stopped drafting is the whole point of the breakdown: a draft that never
    started with one row of budget left ends for lack of room, and must not be filed under
    concurrency -- and its prompt-sized stash should go with the decision, not with the finish."""
    from freetoken.core import Batch

    req = _spec_req(P + 1, 1, P, spec_residual=torch.ones(P, 4))
    sched = _sched_for_batch(None)
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    assert sched._as_spec_batch(batch) is None
    assert req.spec_off and req.spec_off_reason == "no_room"
    assert req.spec_residual is None
    assert sched._spec_rejections == {"no_room": 1}


def test_a_short_miss_streak_buys_back_the_expensive_steps():
    """Measured on live traffic: the next draft lands 0.51 after an accept and ~0.15 after three
    straight misses, while a verify costs ~2.1 decode steps. The slow window cannot see that
    within its own sample, so the streak stands down for a few steps first."""
    from freetoken.scheduler.scheduler import _MTP_SKIP_STEPS

    req = _spec_req(P + 2, 10, P, spec_draft=7)
    sched, _, _, _ = _setup(req)
    for _ in range(2):
        sched._tally_verify(req, False)
    assert not req.spec_suspend and getattr(sched, "_spec_rejections", {}) == {}
    sched._tally_verify(req, False)  # third straight miss
    assert req.spec_suspend == _MTP_SKIP_STEPS and not req.spec_off
    assert req.spec_rejects == 0  # the streak itself is consumed, not left armed
    assert sched._spec_rejections == {"streak": 1}

    req.spec_suspend = 0  # an accept on the way back out resets it
    sched._tally_verify(req, True)
    assert req.spec_rejects == 0
    sched._tally_verify(req, False)
    sched._tally_verify(req, False)
    assert not req.spec_suspend and sched._spec_rejections == {"streak": 1}


def test_a_request_the_forward_never_stashed_is_declined_not_dropped():
    """The batch-level forward gate names no request, so a request reaching a single-request
    decode batch with neither a draft nor a stash was off-spec in silence. Counting it is the
    only way the breakdown stops being empty exactly when drafting is not running."""
    from freetoken.core import Batch

    req = _spec_req(P + 1, 10, P)
    sched = _sched_for_batch(None)
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    assert sched._as_spec_batch(batch) is None
    assert req.spec_off and req.spec_off_reason == "no_stash"
    assert sched._spec_rejections == {"no_stash": 1}


def test_cold_start_seeds_a_draft_with_nothing_to_catch_up_on(monkeypatch):
    """The catch-up pass is an optimization, not a precondition: with the switch on, a request
    whose prompt rows are gone still drafts, seeded by the head pass of its own first spec
    step. Counted, because those drafts are worth less than a warmed-up one."""
    import freetoken.scheduler.scheduler as sched_mod
    from freetoken.core import Batch

    monkeypatch.setattr(sched_mod, "mtp_cold_start", lambda: True)
    req = _spec_req(P + 1, 10, P)  # no draft, no stash, the head has never run
    sched = _sched_for_batch(None)
    sched.engine.linear_state_pool.alloc = lambda n: list(range(n))
    batch = Batch(reqs=[req], phase="decode")
    batch.padded_reqs = [req]
    assert sched._as_spec_batch(batch) is batch
    assert batch.spec_mode == "prologue_decode"
    assert batch.spec_prologue is None  # no catch-up batch: this step's head pass seeds it
    assert not req.spec_off and sched._spec_rejections == {}
    assert sched._spec_cold_seeds == 1


def test_only_a_spec_batch_forces_the_serial_drain():
    """The accept dependency belongs to the spec batch, not to --speculative mtp as a whole:
    a prefill or regular decode batch must keep the overlap pipeline, while both spec shapes
    (the catch-up step that produces the first draft, and the verify that decides on it) feed
    the next batch's positions, pages and token_pool rows from the drained payload."""
    from types import SimpleNamespace as NS

    drain = Scheduler._needs_serial_drain
    assert not drain(None)
    assert not drain((NS(batch=NS(spec_mode=None)), NS()))
    assert not drain((NS(batch=NS()), NS()))  # no spec_mode attribute at all
    assert drain((NS(batch=NS(spec_mode="verify")), NS()))
    assert drain((NS(batch=NS(spec_mode="prologue_decode")), NS()))


def test_the_reply_snapshot_is_a_copy_and_only_when_drafting_is_on():
    """/v1/stats reads these counters after the replies carrying them have been serialized, so
    the snapshot must not alias the live map the next decline keeps writing into."""
    sched = _sched_for_batch(None)
    req = _spec_req(P + 2, 10, P, spec_draft=7)
    sched._tally_verify(req, True)
    verifies, accepted, declines, steps, decode_steps, cold = sched._spec_snapshot()
    assert (verifies, accepted) == (1, 1) and declines == {}
    assert (steps, decode_steps, cold) == (0, 0, 0)
    declines["chunked"] = 1
    assert "chunked" not in sched._spec_rejections
    sched.config.speculative = "none"
    assert sched._spec_snapshot() is None


def test_the_coverage_split_is_what_separates_a_bad_head_from_a_bad_policy():
    """Acceptance says nothing about how many steps drafted: a 0.45 measured over 16 verifies
    of a 256-token answer is a switched-off request, not a weak head."""
    sched = _sched_for_batch(None)
    assert sched._spec_coverage() == ""  # no steps yet: the field is absent, not 0/0
    sched._spec_steps, sched._spec_decode_steps = 61, 131
    assert sched._spec_coverage() == ", drafted 61/192 (0.32)"
    sched._spec_cold_seeds = 7
    assert sched._spec_coverage().endswith("(0.32) cold=7")
    assert "drafted 61/192" in sched._spec_status()


def test_a_decline_only_run_reports_without_dividing_by_zero(monkeypatch):
    """A run where nothing ever drafted still has a breakdown worth printing -- and printing it
    raised ZeroDivisionError on the idle flush, which took the whole backend worker down with
    it (live: --speculative mtp with the stash budget at zero died as soon as the queue drained).
    """
    lines = _spec_logs(monkeypatch)
    sched = _sched_for_batch(None)
    sched._count_spec_rejection("no_stash")
    sched._report_spec_tally(force=True)
    assert any("MTP spec: no verifies" in l and "no_stash=1" in l for l in lines)
    assert "no verifies" in sched._spec_status()


def _cfg_sched():
    from types import SimpleNamespace as NS

    from freetoken.scheduler.scheduler import Scheduler

    sched = NS(config=NS(model_config=None), engine=NS(model=None), _spec_report_s=10.0)
    sched._report_spec_config = lambda: Scheduler._report_spec_config(sched)
    return sched


def test_startup_names_the_declines_the_switches_have_removed(monkeypatch):
    """The start-up line is the only MTP statement made before traffic exists, so it has to say
    which declines the knobs in effect would produce -- and which they cancel."""
    import freetoken.scheduler.scheduler as sched_mod

    lines = _spec_logs(monkeypatch)
    monkeypatch.setattr(sched_mod, "mtp_cold_start", lambda: False)
    _cfg_sched()._report_spec_config()
    line = [l for l in lines if "MTP spec: enabled" in l][0]
    assert "single-request decode batches" in line
    assert "prefix-hit, over-budget and image rows never draft" in line

    lines.clear()
    monkeypatch.setattr(sched_mod, "mtp_cold_start", lambda: True)
    _cfg_sched()._report_spec_config()
    assert "draft cold" in [l for l in lines if "MTP spec: enabled" in l][0]
