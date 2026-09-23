"""_drain_spec bookkeeping (MTP Phase 1): the accept/reject/replay host-side state
machine, CPU-only with a stub scheduler and real Req objects.

The invariants under test are the ones the reject-replay loop lives or dies by:
accept completes once and appends two tokens; reject keeps the PRE-step lens
(cached_len P, device_len P+2) so the replay re-reads [x_P, y1] from the pool,
restores the GDN/PLE snapshot, repairs the draft's pool slot and releases the
recorded pages; a replay canary miss fails loudly; an EOS in the first emitted
token drops the bonus token and finishes the request.
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
    calls = {"copy_from": [], "release": [], "freed": [], "removed": []}
    pool = torch.zeros(4, 64, dtype=torch.int32)
    pool[0, P + 1] = 999  # the draft's staged pool slot (verify bump wrote it)
    pool[0, P + 2] = 42   # the engine's next_tokens_gpu write (Scheduler._forward)

    def free_resources(r):
        # mirrors the real _free_req_resources spec cleanup (slot release is pool-side
        # and covered by test_hybrid_cache_manager's roundtrip)
        calls["freed"].append(r)
        r.spec_slot_idx = None
        r.spec_residual = None
        r.spec_draft = None
        r.spec_replay = False
        r.table_idx = -1

    sched = SimpleNamespace(
        decode_manager=SimpleNamespace(remove_req=lambda r: calls["removed"].append(r)),
        _free_req_resources=free_resources,
        engine=SimpleNamespace(linear_state_pool=SimpleNamespace(
            copy_from=lambda src, dst: calls["copy_from"].append((src, dst)))),
        token_pool=pool,
        cache_manager=SimpleNamespace(
            release_paged=lambda info: calls["release"].append(info)),
        eos_token_ids={2},
        _match_stop_str=lambda r: None,
        toolcall_anchor_id=None,
        finished_reqs=set(),
    )
    req.spec_slot_idx = 3
    req.linear_slot_idx = 1
    batch = SimpleNamespace(reqs=[req], spec_mode=mode, spec_pages=pages)
    return sched, batch, calls, pool


def _drain(sched, batch, spec):
    reply, finished = [], set()
    Scheduler._drain_spec(sched, batch, spec, reply, finished)
    return reply, finished


def test_accept_completes_once_and_appends_two_tokens():
    req = _req()
    req.device_len = P + 2  # the verify bump
    sched, batch, calls, pool = _setup(req)
    reply, finished = _drain(sched, batch,
                             {"y1": 41, "y2": 42, "accept": True, "draft": 77})
    assert req.input_ids.tolist()[-2:] == [41, 42]
    assert req.cached_len == P + 2 and req.device_len == P + 3
    assert req.spec_draft == 77 and not req.spec_replay
    assert [m.next_token for m in reply] == [41, 42]
    assert not any(m.finished for m in reply) and not finished
    assert calls["copy_from"] == [] and calls["release"] == []
    assert int(pool[0, P + 2]) == 42  # the engine's pool write landed at the new position


def test_reject_restores_repairs_and_keeps_pre_step_lens():
    req = _req()
    req.device_len = P + 2
    req.spec_draft = 999
    sched, batch, calls, pool = _setup(req, pages=[(0, 1, 2)])
    reply, finished = _drain(sched, batch,
                             {"y1": 41, "y2": 4242, "accept": False, "draft": 77})
    assert calls["copy_from"] == [(3, 1)]  # scratch -> live
    assert int(pool[0, P + 1]) == 41  # the draft's pool slot repaired with the true token
    assert calls["release"] == [[(0, 1, 2)]]
    assert req.spec_replay and req.spec_draft is None
    assert req.input_ids.tolist()[-1] == 41 and req.input_ids.numel() == P + 2
    # the pre-step lens: the replay step re-reads rows [P, P+1] = [x_P, y1]
    assert req.cached_len == P and req.device_len == P + 2
    assert [m.next_token for m in reply] == [41] and not finished


def test_replay_accept_appends_only_the_bonus_token():
    req = _req(host_len=P + 2)  # the reject drain already appended y1
    req.device_len = P + 2
    req.spec_replay = True
    sched, batch, calls, _pool = _setup(req, mode="replay")
    reply, finished = _drain(sched, batch,
                             {"y1": 41, "y2": 43, "accept": True, "draft": 88})
    assert req.input_ids.tolist()[-1] == 43 and req.input_ids.numel() == P + 3
    assert req.cached_len == P + 2 and req.device_len == P + 3
    assert req.spec_draft == 88 and not req.spec_replay
    assert [m.next_token for m in reply] == [43]


def test_replay_canary_miss_fails_loudly():
    req = _req(host_len=P + 2)
    req.device_len = P + 2
    sched, batch, _calls, _pool = _setup(req, mode="replay")
    with pytest.raises(RuntimeError, match="canary"):
        _drain(sched, batch, {"y1": 41, "y2": 43, "accept": False, "draft": 88})


def test_eos_on_the_first_token_drops_the_bonus_and_finishes():
    req = _req()
    req.device_len = P + 2
    sched, batch, calls, _pool = _setup(req)
    reply, finished = _drain(sched, batch,
                             {"y1": 2, "y2": 42, "accept": True, "draft": 77})
    assert [m.next_token for m in reply] == [2]
    assert reply[0].finished and reply[0].finish_reason == "stop"
    assert req in finished and calls["removed"] == [req] and calls["freed"] == [req]
    assert req.spec_slot_idx is None  # the scratch slot was released with the request


def test_length_finish_on_the_second_token():
    req = Req(input_ids=torch.arange(P + 1, dtype=torch.int32), table_idx=0,
              cached_len=P, output_len=1, uid=7,
              sampling_params=SamplingParams(temperature=0.0), cache_handle=None)
    # max_device_len = P+2: appending y1 exhausts the budget, the bonus token is dropped
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
        engine=SimpleNamespace(linear_state_pool=SimpleNamespace(alloc=lambda n: [5])),
        token_pool=_Pool(),
        _build_spec_prologue=lambda r: "PROLOGUE",
    )
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

    # remain_len == 1: an accept would append two tokens past the budget -> stay regular
    req = _spec_req(P + 1, 1, P, spec_draft=42, spec_slot_idx=5)
    sched, writes = _upgrade_sched(req)
    batch = Batch(reqs=[req], phase="decode")
    assert Scheduler._as_spec_batch(sched, batch) is None
    assert req.device_len == P + 1 and batch.spec_mode is None and not writes

    # remain_len == 2: the verify upgrade stages the draft and bumps the row count
    req2 = _spec_req(P + 1, 2, P, spec_draft=42, spec_slot_idx=5)
    sched2, writes2 = _upgrade_sched(req2)
    batch2 = Batch(reqs=[req2], phase="decode")
    out = Scheduler._as_spec_batch(sched2, batch2)
    assert out is batch2 and batch2.spec_mode == "verify" and batch2.phase == "prefill"
    assert req2.device_len == P + 2 and writes2[(0, P + 1)] == 42


def test_replay_runs_even_with_one_token_of_room():
    # post-reject lenses (cached P, device P+2, host holds y1): the two-row replay is the
    # ONLY consumer of that state, so the remain>=2 verify gate must not apply to it.
    from freetoken.core import Batch
    from freetoken.scheduler.scheduler import Scheduler

    req = _spec_req(P + 2, 1, P, device_len=P + 2, spec_replay=True, spec_slot_idx=5)
    assert req.remain_len == 1
    sched, _writes = _upgrade_sched(req)
    batch = Batch(reqs=[req], phase="decode")
    out = Scheduler._as_spec_batch(sched, batch)
    assert out is batch and batch.spec_mode == "replay" and batch.phase == "prefill"
    assert req.device_len == P + 2  # no bump: the replay rows are [x_P, y1]
