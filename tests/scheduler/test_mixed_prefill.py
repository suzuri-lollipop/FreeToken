"""Mixed prefill+decode batches (FREETOKEN_MIXED_PREFILL_TOKENS).

While a prompt is being prefilled, the running requests ride the chunk's forward as
extend_len==1 rows of a phase="prefill" batch (Scheduler._mix_decodes) so their decode
never waits for a dedicated prefill step. The scheduling decisions are covered in
test_scheduler_fairness.py; this file covers the per-request bookkeeping that must
branch on ``Batch.decoding_reqs`` instead of on phase:

* the drain samples one token per row and commits ONLY the prompt side to the radix
  tree (a live decode committing its mid-generation prefix would double-lock pages);
* each SWA eviction driver runs on its own side of a mixed batch;
* the prefill time of a mixed window is charged to the prompt alone, the window never
  feeds the chunk-size EMA, and a mixed batch owes no decode-stall debt.

CPU-only: real Req/Batch/CacheManager objects, no Engine, no GPU.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.message import DetokenizeMsg
from freetoken.scheduler.scheduler import Scheduler


def _req(uid: int, prompt_len: int = 13) -> Req:
    req = Req(
        input_ids=torch.arange(prompt_len, dtype=torch.int32),
        table_idx=uid, cached_len=prompt_len - 1, output_len=prompt_len + 20,
        uid=uid, sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
        cache_handle=None,
    )
    req.complete_one()  # the forward placed its sampled row and advanced the window
    return req


def _last_data(batch, tokens: list[int]):
    return (
        SimpleNamespace(batch=batch),
        (None,
         torch.tensor(tokens, dtype=torch.int32),
         SimpleNamespace(synchronize=lambda: None)),
    )


def _drain_stub(cache_req_calls: list[int]):
    import contextlib

    cache_manager = SimpleNamespace(
        lazy_free_region=contextlib.nullcontext,
        cache_req=lambda req, *, finished: cache_req_calls.append(req.uid),
        evictable_kv_pages=0,
        evictable_mamba_slots=0,
        evictable_swa_tokens=0,
    )
    sent: list[DetokenizeMsg] = []
    stub = SimpleNamespace(
        cache_manager=cache_manager,
        decode_manager=SimpleNamespace(remove_req=lambda r: None, running_reqs=set()),
        prefill_manager=SimpleNamespace(pending_list=[]),
        status_reporter=SimpleNamespace(report_batch=lambda *_, **__: None),
        finished_reqs=set(),
        eos_token_ids=set(),
        toolcall_anchor_id=None,
        config=SimpleNamespace(page_size=64, speculative="none"),
        engine=SimpleNamespace(run_pending_host_fill=lambda: None),
        send_result=sent.extend,
        _kv_usage_pages=lambda: (0, 0),
        _mamba_slot_usage=lambda: None,
        _swa_token_usage=lambda: None,
        _moe_residency=lambda: (0, 0, 0),
        _gpu_mem_bytes=lambda: 0,
        _match_stop_str=lambda _req: None,
        _account_prefill_debt=lambda _batch: None,
        _charge_prefill_time=lambda _batch: None,
        _observe_chunk_time=lambda _batch: None,
        _pending_abort_acks=set(),
        _last_data=None,
        _free_req_resources=lambda _req: None,
        spec=None,
        replies=sent,
    )
    stub._commit_tokens = lambda req, tokens: Scheduler._commit_tokens(stub, req, tokens)
    return stub, sent


def test_mixed_drain_replies_for_all_rows_but_commits_only_the_prompt():
    prompt_req = _req(uid=1)
    decode_req = _req(uid=2)
    batch = Batch(reqs=[prompt_req, decode_req], phase="prefill")
    batch.decoding_reqs = [decode_req]
    commits: list[int] = []
    stub, sent = _drain_stub(commits)

    Scheduler._process_last_data(stub, _last_data(batch, [42, 43]))

    # the ride-along row emitted its token like a decode step...
    assert [m.uid for m in sent] == [1, 2]
    assert [m.next_tokens for m in sent] == [[42], [43]]
    # ...but only the prompt completion committed a prefix: a live decode's
    # mid-generation prefix must never be radix-committed by the batch it merely rode
    assert commits == [1]


def test_mixed_drain_pure_prefill_still_commits():
    # guard regression: without decoding_reqs the commit path is untouched
    prompt_req = _req(uid=1)
    batch = Batch(reqs=[prompt_req], phase="prefill")
    commits: list[int] = []
    stub, _sent = _drain_stub(commits)

    Scheduler._process_last_data(stub, _last_data(batch, [42]))

    assert commits == [1]


def test_free_window_slots_drivers_split_by_side():
    calls: dict[str, list[int]] = {"dec": [], "ext": []}

    def maybe_free(reqs, *, forward_iter):
        calls["dec"] = [r.uid for r in reqs]

    def extend(reqs):
        calls["ext"] = [r.uid for r in reqs]

    stub = SimpleNamespace(
        _forward_iter=8,
        cache_manager=SimpleNamespace(
            maybe_free_swa_out_of_window=maybe_free,
            free_swa_out_of_window_extend=extend,
        ),
    )
    prompt, decode = _req(uid=1), _req(uid=2)

    mixed = Batch(reqs=[prompt, decode], phase="prefill")
    mixed.decoding_reqs = [decode]
    Scheduler._free_window_slots(stub, mixed)
    assert calls == {"dec": [2], "ext": [1]}
    assert decode.decode_batch_idx == 1 and prompt.decode_batch_idx == 0

    calls["dec"].clear(), calls["ext"].clear()
    Scheduler._free_window_slots(stub, Batch(reqs=[prompt], phase="prefill"))
    assert calls == {"dec": [], "ext": [1]}
    assert prompt.decode_batch_idx == 0  # prefill side never enters the decode cadence

    calls["dec"].clear(), calls["ext"].clear()
    Scheduler._free_window_slots(stub, Batch(reqs=[decode], phase="decode"))
    assert calls == {"dec": [2], "ext": []}


def test_charge_prefill_time_ignores_the_decode_side():
    prompt, decode = _req(uid=1), _req(uid=2)
    prompt.prefill_chunk_tokens = 100
    decode.prefill_chunk_tokens = 5000  # stale snapshot from its own admission
    batch = SimpleNamespace(
        is_prefill=True,
        scheduled_at=time.perf_counter() - 0.5,
        reqs=[prompt, decode], decoding_reqs=[decode],
    )

    Scheduler._charge_prefill_time(SimpleNamespace(), batch)

    assert 450.0 < prompt.prefill_ms < 550.0  # the whole window is the prompt's
    assert decode.prefill_ms == 0.0


def test_mixed_batch_owes_no_stall_debt():
    def debt_after(batch):
        stub = SimpleNamespace(
            _prefill_debt_s=2.0, _prefill_debt=0.0, _debt_discharge=None,
            decode_manager=SimpleNamespace(runnable=True),
        )
        # tp1 stub: no broadcast clock, so the real _sched_now reads the local clock
        stub._sched_now = lambda: Scheduler._sched_now(stub)
        Scheduler._account_prefill_debt(stub, batch)
        return stub._prefill_debt

    stamp = time.perf_counter() - 7.3

    def batch(decoding):
        return SimpleNamespace(
            is_prefill=True,
            scheduled_at=stamp, decoding_reqs=decoding,
        )

    assert debt_after(batch(None)) > 1.0  # a dedicated chunk still accrues
    assert debt_after(batch([_req(uid=2)])) == 0.0


def test_mixed_in_flight_never_concedes():
    mixed = SimpleNamespace(
        is_prefill=True,
        scheduled_at=time.perf_counter() - 9.0, decoding_reqs=[object()],
    )
    stub = SimpleNamespace(
        _prefill_debt_s=2.0, _last_data=(SimpleNamespace(batch=mixed), None),
        decode_manager=SimpleNamespace(runnable=True),
    )
    assert Scheduler._inflight_prefill(stub) == (None, 0.0)


def test_observe_chunk_time_skips_mixed_windows():
    batch = SimpleNamespace(
        is_prefill=True,
        scheduled_at=time.perf_counter() - 1.0, log_new_tokens=512,
        decoding_reqs=[object()],
    )
    stub = SimpleNamespace(_chunk_target_s=0.5, _chunk_ema_spt=None)
    Scheduler._observe_chunk_time(stub, batch)
    assert stub._chunk_ema_spt is None
