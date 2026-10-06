"""Scheduling fairness under multi-user load.

Two related fixes:

* ``Scheduler._schedule_next_batch`` interleaves: after ``prefill_decode_interval``
  consecutive prefill steps it hands one step to the running decodes, so a long
  chunked prompt cannot stall everyone's inter-token latency. ``0`` disables it.
* The wall-clock debt counts the chunk still IN FLIGHT, not only the chunks already
  drained: the drain that books a chunk runs after the decision, so the accrued debt
  alone lets one slow chunk stall a waiting decode for two chunks.
* ``PrefillManager.schedule_next_batch`` backfills: a pending request that does
  not fit this step no longer blocks smaller requests queued behind it; the
  deferred ones keep their arrival order and retry next step.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch


def _stub_scheduler(interval: int, decode_runnable: bool = True, debt_s: float = 0.0,
                    debt: float = 0.0, chunk_target: float = 0.0,
                    chunk_ema: float | None = None, budget: int = 99):
    from freetoken.scheduler.scheduler import Scheduler

    calls: list[str] = []
    budgets: list[int] = []
    prefill_batch = SimpleNamespace(is_prefill=True, prompt_admissions=[])
    decode_batch = SimpleNamespace(is_prefill=False, prompt_admissions=[])

    def prefill(_budget):
        calls.append("prefill")
        budgets.append(_budget)
        return prefill_batch

    def decode():
        calls.append("decode")
        return decode_batch

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_budget = budget
    scheduler.config = SimpleNamespace(prefill_decode_interval=interval)
    scheduler._prefill_streak = 0
    scheduler._prefill_debt = debt
    scheduler._prefill_debt_s = debt_s
    scheduler._debt_discharge = None
    scheduler._last_data = None  # set per step by _inflight / _overlap_phases
    scheduler._chunk_target_s = chunk_target
    scheduler._chunk_ema_spt = chunk_ema
    scheduler.prefill_manager = SimpleNamespace(schedule_next_batch=prefill)
    scheduler.decode_manager = SimpleNamespace(
        schedule_next_batch=decode, runnable=decode_runnable
    )
    scheduler._prepare_batch = lambda batch: batch
    scheduler._report_prompt_admissions = lambda batch: None
    scheduler.recorded_budgets = budgets
    return scheduler, calls


def _phases(scheduler, n: int) -> list[str]:
    phases = []
    for _ in range(n):
        batch = type(scheduler)._schedule_next_batch(scheduler)
        phases.append("p" if batch.is_prefill else "d")
    return phases


def test_decode_gets_one_step_after_a_prefill_streak():
    scheduler, _calls = _stub_scheduler(interval=2)
    assert _phases(scheduler, 6) == ["p", "p", "d", "p", "p", "d"]


def test_interval_zero_restores_prefill_first_behavior():
    scheduler, _calls = _stub_scheduler(interval=0)
    assert _phases(scheduler, 6) == ["p"] * 6


def test_interleave_off_when_no_decode_is_running():
    scheduler, _calls = _stub_scheduler(interval=2, decode_runnable=False)
    assert _phases(scheduler, 6) == ["p"] * 6


def test_time_debt_forces_decode_before_the_count_interval():
    # count-based guard disabled (interval=0); the accrued wall-clock debt alone
    # must hand the first step to decode, then reset so prefill resumes.
    scheduler, _calls = _stub_scheduler(interval=0, debt_s=2.0, debt=7.3)
    assert _phases(scheduler, 4) == ["d", "p", "p", "p"]
    assert scheduler._prefill_debt == 0.0


def test_time_debt_zero_never_forces_decode():
    scheduler, _calls = _stub_scheduler(interval=0, debt_s=0.0, debt=1e9)
    assert _phases(scheduler, 4) == ["p"] * 4


def test_time_debt_below_threshold_keeps_prefilling():
    scheduler, _calls = _stub_scheduler(interval=0, debt_s=2.0, debt=1.5)
    assert _phases(scheduler, 4) == ["p"] * 4


def test_debt_not_owed_when_no_decode_is_running():
    scheduler, _calls = _stub_scheduler(
        interval=0, debt_s=2.0, debt=7.3, decode_runnable=False
    )
    assert _phases(scheduler, 4) == ["p"] * 4


# ------------------------------------------------ P1b: the chunk still in flight counts too


def _inflight(scheduler, seconds: float, phase: str = "prefill", spec_mode=None,
              scheduled_at: float | None = None):
    """Makes ``_last_data`` the batch overlap_loop launched but has not drained yet."""
    import time as _time

    if scheduled_at is None:
        # a decode batch is never stamped (core.py: Batch.scheduled_at defaults to 0.0)
        scheduled_at = _time.perf_counter() - seconds if phase == "prefill" else 0.0
    scheduler._last_data = (
        SimpleNamespace(  # ForwardData = (ForwardInput, ForwardOutput); only .batch matters
            batch=SimpleNamespace(
                is_prefill=phase == "prefill", spec_mode=spec_mode, prompt_admissions=[],
                scheduled_at=scheduled_at, log_new_tokens=0,
            ),
        ),
    )
    return scheduler._last_data[0].batch


def _overlap_phases(scheduler, n: int, chunk_s: float) -> list[str]:
    """Steps the way ``overlap_loop`` does: the decision sees the batch launched last step
    as still in flight, and the drain a few lines later books (or discharges) it."""
    from freetoken.scheduler.scheduler import Scheduler

    phases = []
    for _ in range(n):
        launched = type(scheduler)._schedule_next_batch(scheduler)
        phases.append("p" if launched.is_prefill else "d")
        if scheduler._last_data is not None:
            Scheduler._account_prefill_debt(scheduler, scheduler._last_data[0].batch)
        _inflight(scheduler, chunk_s, phase="prefill" if launched.is_prefill else "decode")
    return phases


def test_inflight_chunk_concedes_once_and_bounds_the_stall():
    # every chunk runs 3s against a 2s budget: one concede per chunk, never two in a row
    scheduler, calls = _stub_scheduler(interval=0, debt_s=2.0)
    assert _overlap_phases(scheduler, 6, chunk_s=3.0) == ["p", "d", "p", "d", "p", "d"]
    assert scheduler._prefill_debt == 0.0  # each chunk paid once, none re-opened the debt
    assert calls == ["prefill", "decode"] * 3


def test_inflight_term_concedes_one_step_before_the_drain_would():
    # 0.5s chunks vs a 2s budget: drained time alone only owes after the 5th chunk books it,
    # the in-flight term sees it at the 4th decision -- that one step is the user-visible gap
    scheduler, _calls = _stub_scheduler(interval=0, debt_s=2.0)
    assert _overlap_phases(scheduler, 6, chunk_s=0.5) == ["p", "p", "p", "p", "d", "p"]


def test_count_guard_unchanged_when_debt_is_off():
    # the in-flight term must not leak in while the debt is disabled
    scheduler, _calls = _stub_scheduler(interval=2)
    assert _overlap_phases(scheduler, 6, chunk_s=3.0) == ["p", "p", "d", "p", "p", "d"]


def test_inflight_guard_matches_the_accrual_guard():
    cases = [
        dict(phase="prefill", spec_mode="verify"),  # a spec step rides phase="prefill"
        dict(phase="decode"),                       # decode batches are unstamped
        dict(phase="prefill", scheduled_at=0.0),    # unstamped prefill: never elapsed
    ]
    for case in cases:
        scheduler, _calls = _stub_scheduler(interval=0, debt_s=2.0)
        _inflight(scheduler, 9.0, **case)
        assert _phases(scheduler, 3) == ["p"] * 3, case


def test_discharge_is_one_shot_and_scoped_to_its_own_batch():
    from freetoken.scheduler.scheduler import Scheduler

    scheduler, _calls = _stub_scheduler(interval=0, debt_s=2.0)
    paid = _inflight(scheduler, 3.0)
    type(scheduler)._schedule_next_batch(scheduler)  # concedes: P buys it
    assert scheduler._debt_discharge is paid
    Scheduler._account_prefill_debt(scheduler, paid)  # P's drain: nothing re-owed
    assert scheduler._prefill_debt == 0.0
    assert scheduler._debt_discharge is None
    later = _inflight(scheduler, 3.0)
    Scheduler._account_prefill_debt(scheduler, later)  # the next chunk still pays
    assert scheduler._prefill_debt > 2.9


def test_inflight_helper_is_gated_before_it_touches_last_data():
    # the drain scheduler of test_cost_accounting_core / test_spec_drain has no _last_data;
    # the debt gate being first is what keeps them off the new path
    from freetoken.scheduler.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._prefill_debt_s = 0.0
    scheduler.decode_manager = SimpleNamespace(runnable=True)
    assert Scheduler._inflight_prefill(scheduler) == (None, 0.0)
    assert not hasattr(scheduler, "_last_data")


def _stub_drain_scheduler(debt_s: float, decode_runnable: bool, chunk_target: float = 0.0):
    from freetoken.scheduler.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._prefill_debt = 0.0
    scheduler._prefill_debt_s = debt_s
    scheduler._debt_discharge = None
    scheduler._chunk_target_s = chunk_target
    scheduler._chunk_ema_spt = None
    scheduler.decode_manager = SimpleNamespace(runnable=decode_runnable)
    return scheduler


def test_account_prefill_debt_accrues_schedule_to_drain_window():
    import time as _time

    from freetoken.scheduler.scheduler import Scheduler

    scheduler = _stub_drain_scheduler(debt_s=2.0, decode_runnable=True)
    batch = SimpleNamespace(
        is_prefill=True, spec_mode=None, scheduled_at=_time.perf_counter() - 7.3
    )
    Scheduler._account_prefill_debt(scheduler, batch)
    assert 7.2 < scheduler._prefill_debt < 8.0


def test_account_prefill_debt_skips_non_prefill_spec_and_idle():
    import time as _time

    from freetoken.scheduler.scheduler import Scheduler

    stamp = _time.perf_counter() - 5.0
    cases = [
        SimpleNamespace(is_prefill=False, spec_mode=None, scheduled_at=stamp),
        SimpleNamespace(is_prefill=True, spec_mode="verify", scheduled_at=stamp),
        SimpleNamespace(is_prefill=True, spec_mode=None, scheduled_at=0.0),
    ]
    for batch in cases:
        scheduler = _stub_drain_scheduler(debt_s=2.0, decode_runnable=True)
        Scheduler._account_prefill_debt(scheduler, batch)
        assert scheduler._prefill_debt == 0.0
    # debt is only "owed" while somebody waits: no runnable decode -> no accrual
    scheduler = _stub_drain_scheduler(debt_s=2.0, decode_runnable=False)
    Scheduler._account_prefill_debt(
        scheduler, SimpleNamespace(is_prefill=True, spec_mode=None, scheduled_at=stamp)
    )
    assert scheduler._prefill_debt == 0.0


def _build_managers(num_pages):
    from freetoken.core import Context, get_global_ctx, set_global_ctx

    try:
        get_global_ctx()
    except AssertionError:
        set_global_ctx(Context(page_size=1))

    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.prefill import PrefillManager
    from freetoken.scheduler.table import TableManager

    page_table = torch.zeros((8, 64), dtype=torch.int32, device="cpu")
    cache_manager = CacheManager(num_pages=num_pages, page_size=1, page_table=page_table, type="radix")
    table_manager = TableManager(max_running_reqs=8, page_table=page_table)
    decode_manager = DecodeManager(page_size=1)
    return cache_manager, table_manager, PrefillManager(cache_manager, table_manager, decode_manager)


def test_small_request_backfills_behind_a_too_large_one():
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    cache_manager, table_manager, prefill_manager = _build_managers(num_pages=16)
    big = PendingReq(
        uid=1,
        input_ids=torch.arange(100, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=4),
    )
    small = PendingReq(
        uid=2,
        input_ids=torch.arange(8, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=4),
    )
    prefill_manager.pending_list = [big, small]

    batch = prefill_manager.schedule_next_batch(8)

    assert batch is not None  # old head-of-line break returned None here
    assert [req.uid for req in batch.reqs] == [2]
    assert [req.uid for req in prefill_manager.pending_list] == [1]  # deferred, retried next step


def test_deferred_requests_keep_arrival_order():
    from freetoken.core import SamplingParams
    from freetoken.scheduler.utils import PendingReq

    cache_manager, table_manager, prefill_manager = _build_managers(num_pages=16)
    big_a = PendingReq(
        uid=1,
        input_ids=torch.arange(100, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=4),
    )
    big_b = PendingReq(
        uid=2,
        input_ids=torch.arange(99, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=4),
    )
    small = PendingReq(
        uid=3,
        input_ids=torch.arange(4, dtype=torch.int32),
        sampling_params=SamplingParams(max_tokens=4),
    )
    prefill_manager.pending_list = [big_a, big_b, small]

    batch = prefill_manager.schedule_next_batch(4)

    assert [req.uid for req in batch.reqs] == [3]
    assert [req.uid for req in prefill_manager.pending_list] == [1, 2]


# ---------------------------------------------------------------- P2: adaptive chunk budget


def test_adaptive_budget_shrinks_only_while_decode_waits():
    from freetoken.scheduler.scheduler import Scheduler

    # 7.3s / 8192tok measured -> 0.89 ms/tok; target 0.5s -> 561 -> align64 -> 512
    scheduler, _ = _stub_scheduler(
        interval=0, debt_s=0.0, chunk_target=0.5, chunk_ema=7.3 / 8192, budget=8192
    )
    assert Scheduler._adaptive_prefill_budget(scheduler) == 512
    # no victim waiting -> the solo huge prompt keeps the full configured chunk
    scheduler.decode_manager.runnable = False
    assert Scheduler._adaptive_prefill_budget(scheduler) == 8192


def test_adaptive_budget_floor_and_cap():
    from freetoken.scheduler.scheduler import Scheduler

    scheduler, _ = _stub_scheduler(
        interval=0, debt_s=0.0, chunk_target=0.5, chunk_ema=1.0 / 256, budget=8192
    )
    # 0.5s / 3.9ms-per-tok = 128 -> align 64 -> floor 128
    assert Scheduler._adaptive_prefill_budget(scheduler) == 128
    scheduler._chunk_ema_spt = 0.5 / 32  # even slower -> still floored
    assert Scheduler._adaptive_prefill_budget(scheduler) == 128
    scheduler._chunk_ema_spt = 0.5 / 100000  # absurdly fast
    assert Scheduler._adaptive_prefill_budget(scheduler) == 8192  # capped at budget
    # a budget smaller than the floor clamps the floor (never exceeds the config)
    small, _ = _stub_scheduler(interval=0, debt_s=0.0, chunk_target=0.5,
                               chunk_ema=1.0 / 256, budget=99)
    assert Scheduler._adaptive_prefill_budget(small) == 99


def test_adaptive_budget_off_without_target_or_ema():
    from freetoken.scheduler.scheduler import Scheduler

    scheduler, _ = _stub_scheduler(interval=0, debt_s=0.0)
    assert Scheduler._adaptive_prefill_budget(scheduler) == 99
    scheduler, _ = _stub_scheduler(interval=0, debt_s=0.0, chunk_target=0.5)
    assert scheduler._chunk_ema_spt is None
    assert Scheduler._adaptive_prefill_budget(scheduler) == 99


def test_schedule_next_batch_uses_the_adaptive_budget():
    from freetoken.scheduler.scheduler import Scheduler

    scheduler, calls = _stub_scheduler(
        interval=0, debt_s=0.0, chunk_target=0.5, chunk_ema=7.3 / 8192, budget=8192
    )
    _phases(scheduler, 2)
    assert calls == ["prefill", "prefill"]
    assert scheduler.recorded_budgets == [512, 512]


def test_observe_chunk_time_updates_ema():
    import time as _time

    from freetoken.scheduler.scheduler import Scheduler

    scheduler = _stub_drain_scheduler(debt_s=0.0, decode_runnable=True, chunk_target=0.5)
    batch = SimpleNamespace(
        is_prefill=True, spec_mode=None,
        scheduled_at=_time.perf_counter() - 7.3, log_new_tokens=8192,
    )
    Scheduler._observe_chunk_time(scheduler, batch)
    assert scheduler._chunk_ema_spt is not None
    first = scheduler._chunk_ema_spt
    assert abs(first - 7.3 / 8192) < 0.0005
    # second sample moves the EMA 30% toward itself
    batch2 = SimpleNamespace(
        is_prefill=True, spec_mode=None,
        scheduled_at=_time.perf_counter() - 2.0, log_new_tokens=512,
    )
    Scheduler._observe_chunk_time(scheduler, batch2)
    expected = 0.7 * first + 0.3 * (batch2_spt := 2.0 / 512)
    assert abs(scheduler._chunk_ema_spt - expected) < 0.0005
    # spec batches and zero-token batches never feed the EMA
    before = scheduler._chunk_ema_spt
    Scheduler._observe_chunk_time(scheduler, SimpleNamespace(
        is_prefill=True, spec_mode="verify",
        scheduled_at=_time.perf_counter() - 9.0, log_new_tokens=2))
    Scheduler._observe_chunk_time(scheduler, SimpleNamespace(
        is_prefill=True, spec_mode=None,
        scheduled_at=_time.perf_counter() - 9.0, log_new_tokens=0))
    assert scheduler._chunk_ema_spt == before


def test_effective_debt_follows_chunk_target_in_adaptive_mode():
    from freetoken.scheduler.scheduler import Scheduler

    scheduler, _ = _stub_scheduler(
        interval=0, debt_s=2.0, chunk_target=0.5, chunk_ema=7.3 / 8192
    )
    # adaptive + victim waiting -> concede after (nearly) every shrunken chunk
    assert Scheduler._effective_debt_s(scheduler) == 0.25
    scheduler.decode_manager.runnable = False
    assert Scheduler._effective_debt_s(scheduler) == 2.0
    scheduler.decode_manager.runnable = True
    scheduler._chunk_target_s = 0.0
    assert Scheduler._effective_debt_s(scheduler) == 2.0
    scheduler._chunk_target_s, scheduler._chunk_ema_spt = 0.5, None
    assert Scheduler._effective_debt_s(scheduler) == 2.0
