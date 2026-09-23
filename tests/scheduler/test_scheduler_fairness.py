"""Scheduling fairness under multi-user load.

Two related fixes:

* ``Scheduler._schedule_next_batch`` interleaves: after ``prefill_decode_interval``
  consecutive prefill steps it hands one step to the running decodes, so a long
  chunked prompt cannot stall everyone's inter-token latency. ``0`` disables it.
* ``PrefillManager.schedule_next_batch`` backfills: a pending request that does
  not fit this step no longer blocks smaller requests queued behind it; the
  deferred ones keep their arrival order and retry next step.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch


def _stub_scheduler(interval: int, decode_runnable: bool = True):
    from freetoken.scheduler.scheduler import Scheduler

    calls: list[str] = []
    prefill_batch = SimpleNamespace(is_prefill=True, prompt_admissions=[])
    decode_batch = SimpleNamespace(is_prefill=False, prompt_admissions=[])

    def prefill(_budget):
        calls.append("prefill")
        return prefill_batch

    def decode():
        calls.append("decode")
        return decode_batch

    scheduler = Scheduler.__new__(Scheduler)
    scheduler.prefill_budget = 99
    scheduler.config = SimpleNamespace(prefill_decode_interval=interval)
    scheduler._prefill_streak = 0
    scheduler.prefill_manager = SimpleNamespace(schedule_next_batch=prefill)
    scheduler.decode_manager = SimpleNamespace(
        schedule_next_batch=decode, runnable=decode_runnable
    )
    scheduler._prepare_batch = lambda batch: batch
    scheduler._report_prompt_admissions = lambda batch: None
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
