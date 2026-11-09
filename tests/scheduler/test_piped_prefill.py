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


def test_pipe_thread_routes_collectives_to_the_second_comm(monkeypatch):
    from freetoken.distributed import impl

    class _Rec:
        def __init__(self, tag):
            self.tag = tag
            self.seen = []

        def all_reduce(self, x):
            self.seen.append(self.tag)
            return x

        def all_gather(self, x):
            return x

    primary, pipe = _Rec("primary"), _Rec("pipe")
    comm = impl.DistributedCommunicator()
    monkeypatch.setattr(impl.DistributedCommunicator, "plugins", [primary])
    monkeypatch.setattr(impl, "_PIPE_IMPL", pipe)

    comm.all_reduce("t")
    assert primary.seen == ["primary"] and pipe.seen == []

    def worker():
        impl.mark_pipe_thread()
        comm.all_reduce("t")
        comm.all_gather("t")

    th = threading.Thread(target=worker)
    th.start()
    th.join()
    assert pipe.seen == ["pipe"]  # only the marked thread's calls moved
    assert primary.seen == ["primary"]


# ------------------------------------------------------------------ scheduler policy


class _FakePipe:
    def __init__(self, can_take=True):
        self._can = can_take
        self.inflight = False
        self.dead = False
        self.handed = []
        self.gates = 0
        self.poll_result = None
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

    def poll(self):
        out, self.poll_result = self.poll_result, None
        if out is not None and out[2] is None:
            self.inflight = False
        return out

    def inflight_batch(self):
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
