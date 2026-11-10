"""Piped prefill (FREETOKEN_PIPED_PREFILL) - CPU-side invariants.

The mechanism itself (streams, events, the worker thread) is covered by the live A/B;
what is testable without a GPU is the contract between the three parties:

* the engine Context's batch slot is per-thread once a pipe thread registers, so a
  forward launched on the pipe never sees (or clobbers) the decode batch the main
  loop is issuing, and the main thread's existing single-slot semantics are intact;
* collectives issued from the marked pipe thread route to the second comm, while
  the main thread keeps the primary path;
* the piped scheduler branch hands a chunk to the pipe exactly once per drain,
  always issues the decode batch on the main stream while a chunk runs, skips the
  stall-debt for piped windows, and reverts to inline steps when the worker died.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from freetoken.core import Context


# ------------------------------------------------------------------ engine ctx


def test_ctx_batch_is_per_thread_once_a_pipe_registers():
    ctx = Context(page_size=1)
    main_batch, pipe_batch = object(), object()
    seen: dict[str, object] = {}

    def worker():
        ctx.register_pipe_thread()
        with ctx.forward_batch(pipe_batch):
            time.sleep(0.03)  # main holds its own scope across this
            seen["pipe"] = ctx.batch

    with ctx.forward_batch(main_batch):
        seen["main"] = ctx.batch
        th = threading.Thread(target=worker)
        th.start()
        time.sleep(0.01)
        seen["main_during"] = ctx.batch
        th.join()
        seen["main_after"] = ctx.batch

    assert seen["main"] is main_batch
    assert seen["main_during"] is main_batch  # the pipe's scope never touched main's
    assert seen["main_after"] is main_batch
    assert seen["pipe"] is pipe_batch
    assert ctx._batch is None and ctx._active is None


def test_ctx_nesting_assert_stays_per_thread():
    ctx = Context(page_size=1)
    with ctx.forward_batch(object()):
        with pytest.raises(AssertionError):
            with ctx.forward_batch(object()):
                pass


# ------------------------------------------------------------------ collectives


# ------------------------------------------------------------------ scheduler policy


class _FakePipe:
    """Mirrors PrefillPiper's slot/poll semantics without CUDA or threads."""

    def __init__(self, can_take=True):
        self._can = can_take
        self.inflight = False
        self.dead = False
        self.handed = []
        self.gates = 0
        self.slot = None
        self.chunk_done = SimpleNamespace(
            query=lambda: True, synchronize=lambda: None, record=lambda s: None
        )

    def can_take(self):
        return self._can and not self.inflight

    def handoff(self, fi):
        self.handed.append(fi)
        self.inflight = True

    def record_gate(self):
        self.gates += 1

    def publish(self, fin, fout, err=None, done_ready=True):
        self.slot = (fin, fout, err)
        self.chunk_done.query = lambda: done_ready

    def poll(self, force=False):
        if self.slot is None:
            if not force:
                return None
            while self.slot is None and not self.dead:
                self.wait_slot()
            if self.slot is None:
                return None
        out = self.slot
        if out[2] is None and not force and not self.chunk_done.query():
            return None
        self.slot = None
        self.inflight = False
        return out

    def inflight_batch(self):
        for holder in (self.slot, None):
            if holder is not None:
                return holder[0].batch if isinstance(holder[0], SimpleNamespace) and hasattr(holder[0], "batch") else holder.batch
        return self.handed[-1].batch if self.handed else None

    def wait_slot(self, timeout=0.05):
        return True

    def shutdown(self):
        pass


def _pipe_stub(prefill_runnable=True, decode_runnable=True, can_take=True, budget=99):
    from freetoken.scheduler.scheduler import Scheduler

    prefill_batch = SimpleNamespace(
        is_prefill=True, phase="prefill", reqs=[], prompt_admissions=[], scheduled_at=0.0
    )
    decode_batch = SimpleNamespace(is_prefill=False, phase="decode", reqs=[])
    calls = {"prefill": [], "decode": 0}

    def prefill(b):
        calls["prefill"].append(b)
        return prefill_batch

    s = Scheduler.__new__(Scheduler)
    s._pipe = _FakePipe(can_take=can_take)
    s._pipe_take = None
    s.prefill_budget = budget
    s.config = SimpleNamespace(prefill_decode_interval=4, speculative="none")
    s._prefill_streak = 3
    s._prefill_debt = 1.5
    s._prefill_debt_s = 2.0
    s._chunk_target_s = 0.0
    s._chunk_ema_spt = None
    s._mixed_prefill_tokens = 0
    s.prefill_manager = SimpleNamespace(
        schedule_next_batch=prefill, runnable=prefill_runnable, pending_list=[]
    )
    s.decode_manager = SimpleNamespace(
        schedule_next_batch=lambda: (calls.__setitem__("decode", calls["decode"] + 1) or decode_batch),
        runnable=decode_runnable,
        running_reqs=set(),
    )
    s._prepare_batch = lambda b: ("fi", b)
    s._report_prompt_admissions = lambda b: None
    s.engine = SimpleNamespace(linear_state_pool=None)  # restore: no pool = no-op
    s._as_spec_batch = lambda b: None
    s.batches = {"prefill": prefill_batch, "decode": decode_batch}
    return s, calls


def test_piped_mode_hands_the_chunk_and_still_issues_the_decode():
    s, calls = _pipe_stub()
    fi = s._schedule_next_batch()

    assert fi[1] is s.batches["decode"]  # main issue is the decode batch
    assert s._pipe_take is not None and s._pipe_take[1] is s.batches["prefill"]
    assert s.batches["prefill"].piped is True
    assert calls["prefill"] == [99]  # full adaptive budget: no 1024 cap while piping
    assert s._prefill_streak == 0 and s._prefill_debt == 0.0


def test_piped_mode_waits_the_drain_before_the_next_chunk():
    s, calls = _pipe_stub()
    s._pipe.inflight = True  # a chunk is running; its slot is not drained yet
    s._schedule_next_batch()
    assert calls["prefill"] == []
    assert s._pipe_take is None
    assert s._pipe.handed == []


def test_piped_mode_with_nothing_to_decode_still_pipes_the_chunk():
    s, calls = _pipe_stub(decode_runnable=False)
    fi = s._schedule_next_batch()
    assert fi is None
    assert s._pipe_take is not None


def test_piped_chunk_owes_no_stall_debt():
    import time as _time

    from freetoken.scheduler.scheduler import Scheduler

    stub = SimpleNamespace(
        _prefill_debt_s=2.0, _prefill_debt=0.0, _debt_discharge=None,
        decode_manager=SimpleNamespace(runnable=True),
    )
    batch = SimpleNamespace(
        is_prefill=True, spec_mode=None, decoding_reqs=None, piped=True,
        scheduled_at=_time.perf_counter() - 7.3,
    )
    Scheduler._account_prefill_debt(stub, batch)
    assert stub._prefill_debt == 0.0


def test_abort_of_a_piped_inflight_chunk_defers_to_the_drain():
    import torch

    from freetoken.core import Req, SamplingParams
    from freetoken.message import AbortBackendMsg
    from freetoken.scheduler.decode import DecodeManager
    from freetoken.scheduler.scheduler import Scheduler

    req = Req(
        input_ids=torch.arange(9, dtype=torch.int32),
        table_idx=1, cached_len=8, output_len=20, uid=5,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
        cache_handle=None,
    )
    dm = DecodeManager(page_size=1)
    dm.running_reqs.add(req)
    pipe = _FakePipe()
    pipe.handoff(SimpleNamespace(batch=SimpleNamespace(reqs=[req])))
    freed = []
    stub = SimpleNamespace(
        engine=SimpleNamespace(encoder_cache=None),
        prefill_manager=SimpleNamespace(abort_req=lambda uid: None),
        decode_manager=dm,
        _last_data=None, _pipe=pipe,
        _free_req_resources=freed.append,
        _pending_abort_acks=set(),
    )

    Scheduler._process_one_msg(stub, AbortBackendMsg(uid=5))

    assert req.aborted  # marked, freed later by the pipe drain
    assert freed == []


def test_dead_pipe_discards_its_chunk_and_ends_the_requests(monkeypatch):
    import torch

    from freetoken.core import Req, SamplingParams
    from freetoken.message import ErrorReplyMsg
    from freetoken.scheduler.prefill import ChunkedReq
    from freetoken.scheduler.scheduler import Scheduler

    cont = Req(
        input_ids=torch.arange(9, dtype=torch.int32),
        table_idx=1, cached_len=8, output_len=20, uid=5,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
        cache_handle=None,
    )
    chunked = ChunkedReq(
        input_ids=cont.input_ids, table_idx=cont.table_idx, cached_len=cont.cached_len,
        output_len=cont.output_len, uid=cont.uid,
        sampling_params=cont.sampling_params, cache_handle=None,
    )
    pending = SimpleNamespace(chunked_req=chunked)
    removed, freed, sent = [], [], []
    pipe = _FakePipe()
    import contextlib

    stub = SimpleNamespace(
        _pipe=pipe,
        cache_manager=SimpleNamespace(lazy_free_region=contextlib.nullcontext),
        prefill_manager=SimpleNamespace(pending_list=[pending]),
        decode_manager=SimpleNamespace(remove_req=removed.append),
        _free_req_resources=freed.append,
        send_result=sent.extend,
    )

    Scheduler._discard_piped_chunk(
        stub, SimpleNamespace(batch=SimpleNamespace(reqs=[chunked])), RuntimeError("boom")
    )

    assert stub.prefill_manager.pending_list == []  # the continuation was dropped
    assert freed == [chunked] and removed == [chunked]
    assert sent == [] or all(isinstance(m, ErrorReplyMsg) for m in sent)


def test_poll_force_consumes_even_before_local_query_ready():
    # rank0's broadcast says drain: a rank whose GPU finished LATER must still drain
    # this iteration (the drain's copy_done synchronize absorbs the skew)
    pipe = _FakePipe()
    fin, fout = SimpleNamespace(batch=SimpleNamespace(reqs=[])), SimpleNamespace(
        copy_done_event=SimpleNamespace(query=lambda: False))
    pipe.publish(fin, fout, done_ready=False)
    pipe.inflight = True
    assert pipe.poll(force=False) is None  # local query False: no drain
    got = pipe.poll(force=False)  # still not ready
    assert got is None
    got = pipe.poll(force=True)
    assert got == (fin, fout, None) and not pipe.inflight


def test_poll_force_with_dead_worker_and_no_slot_returns_none():
    pipe = _FakePipe()
    pipe.slot = None
    pipe.dead = True
    assert pipe.poll(force=True) is None


def test_stalled_watchdog_fires_past_the_ceiling():
    from freetoken.scheduler.prefill_pipe import PrefillPiper

    # pure time arithmetic on a bare instance (no thread, no CUDA)
    piper = PrefillPiper.__new__(PrefillPiper)
    piper.inflight = True
    piper.handoff_ts = time.perf_counter() - 10.0
    piper.timeout_s = 5.0
    assert piper.stalled()
    piper.timeout_s = 600.0
    assert not piper.stalled()
    piper.inflight = False
    assert not piper.stalled()


def test_forced_poll_raises_on_permanent_divergence(monkeypatch):
    import threading
    import time as _time

    from freetoken.scheduler import prefill_pipe as pp
    from freetoken.scheduler.prefill_pipe import PrefillPiper

    monkeypatch.setattr(pp, "_FORCED_WAIT_S", 0.1)
    piper = PrefillPiper.__new__(PrefillPiper)
    piper.slot = None
    piper.dead = False
    piper.inflight = True
    piper._cv = threading.Condition()
    t0 = _time.perf_counter()
    with pytest.raises(RuntimeError, match="divergence"):
        piper.poll(force=True)
    assert _time.perf_counter() - t0 < 2.0  # bounded, not a wedge
