"""A uid whose stream generator never starts (client gone before starlette ran the
response body) used to keep its ack_map/event_map entries for the process lifetime,
with listen() appending the whole token stream into the orphaned list. The sweep
reaps it; a started stream must never be touched by the sweep."""

import asyncio
import contextlib
from types import SimpleNamespace

import pytest

from freetoken.message import AbortMsg
from freetoken.server.api_server import FrontendManager
from freetoken.server.generation import GenSpec, send_tokenize_or_discard, submit_generation


class _FakeSend:
    def __init__(self):
        self.msgs = []

    async def put(self, msg):
        self.msgs.append(msg)


def _manager(send):
    mgr = FrontendManager(
        config=SimpleNamespace(served_model_name="unit-model"),
        send_tokenizer=send,
        recv_tokenizer=None,
        maintenance_state="serving",
    )
    # no listener tasks in a unit test: send_one must not start listen()/sweep
    mgr.initialized = True
    return mgr


def test_unstarted_uid_is_reaped_and_the_backend_is_told():
    send = _FakeSend()
    mgr = _manager(send)
    aborted = []
    # no listener in a unit test: record on_abort instead of waiting for the
    # backend's abort ack to transit listen() and heal _inflight (as in production)
    mgr.stats = SimpleNamespace(
        on_new_user=lambda uid: None, on_abort=aborted.append, observe=lambda msg: None
    )
    uid = mgr.new_user()
    # what listen() does to an orphan: every sampled reply piles up unread
    mgr.ack_map[uid].append(SimpleNamespace(finished=False))
    mgr.ack_map[uid].append(SimpleNamespace(finished=True))

    reaped = asyncio.run(mgr.reap_orphan_uids(ttl_s=0))

    assert reaped == [uid]
    assert uid not in mgr.ack_map and uid not in mgr.event_map and uid not in mgr.uid_created
    assert aborted == [uid]
    assert [m for m in send.msgs if isinstance(m, AbortMsg)]


def test_started_stream_is_never_reaped():
    send = _FakeSend()
    mgr = _manager(send)
    uid = mgr.new_user()

    async def go():
        gen = mgr.wait_for_ack(uid)
        step = asyncio.ensure_future(gen.__anext__())  # runs the prologue, parks on the event
        await asyncio.sleep(0.05)
        reaped = await mgr.reap_orphan_uids(ttl_s=0)
        step.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await step
        return reaped

    assert asyncio.run(go()) == []
    assert uid not in mgr.uid_created  # the prologue dropped the marker
    assert uid not in mgr.ack_map  # and the generator's finally still owns cleanup


def test_sweep_loop_reaps_after_the_ttl():
    send = _FakeSend()
    mgr = _manager(send)
    uid = mgr.new_user()

    async def go():
        task = asyncio.create_task(mgr.sweep_orphan_uids(interval_s=0.01, ttl_s=0.0))
        await asyncio.sleep(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert uid not in mgr.ack_map and uid not in mgr.event_map


def test_send_failure_between_new_user_and_the_backend_discards_the_uid():
    discarded = []

    async def boom(msg):
        raise RuntimeError("zmq context terminated")

    state = SimpleNamespace(
        new_user=lambda: 7,
        send_one=boom,
        discard_user=discarded.append,
    )
    with pytest.raises(RuntimeError):
        asyncio.run(send_tokenize_or_discard(state, 7, object()))
    assert discarded == [7]


def test_submit_generation_discards_on_send_failure():
    discarded = []

    async def boom(msg):
        raise RuntimeError("socket closed")

    state = SimpleNamespace(
        new_user=lambda: 3,
        send_one=boom,
        discard_user=discarded.append,
    )
    spec = GenSpec(messages=[{"role": "user", "content": "hi"}], sampling_params=SimpleNamespace())
    with pytest.raises(RuntimeError):
        asyncio.run(submit_generation(spec, state))
    assert discarded == [3]
