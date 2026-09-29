"""ZMQ queue plumbing (utils/mp.py): real round-trips over the transport the engine uses.

The queues are the scheduler/tokenizer IPC backbone; both ends live behind
encoder/decoder callables, so a subscription filter, msgpack option or
socket-type mixup fails only in production. Each queue owns its own context, so a
real endpoint stands in for the production transport: a unix socket on tmp_path, or
loopback TCP on Windows, where the libzmq wheels answer `Protocol not supported` for
ipc:// in either spelling -- including the one this file would otherwise hardcode.
PUB delivery needs the SUB peer fully connected, so the pub/sub test re-sends until
received instead of racing the connect.
"""

from __future__ import annotations

import asyncio
import os
import time

import msgpack
import pytest
import zmq

from freetoken.utils.mp import (
    ZmqAsyncPullQueue,
    ZmqAsyncPushQueue,
    ZmqPubQueue,
    ZmqPullQueue,
    ZmqPushQueue,
    ZmqSubQueue,
    use_zmq_event_loop,
    zmq_tcp_ports,
)


def _addr(tmp_path, name):
    if os.name == "nt":
        return f"tcp://127.0.0.1:{zmq_tcp_ports(1)[0]}"
    return f"ipc://{tmp_path / name}.sock"


def test_push_pull_roundtrip(tmp_path):
    addr = _addr(tmp_path, "pushpull")
    push = ZmqPushQueue(addr, create=True, encoder=lambda o: {"x": o})
    pull = ZmqPullQueue(addr, create=False, decoder=lambda d: d["x"] * 2)
    try:
        push.put(21)
        assert pull.get() == 42
    finally:
        push.stop()
        pull.stop()


def _recv(sub, expected):
    try:
        got = sub.get()
    except zmq.Again:
        return False
    # a stale in-flight copy of an earlier payload drains like any message
    return got == expected


def test_pub_sub_receives_everything_when_blank_subscribed(tmp_path):
    addr = _addr(tmp_path, "pubsub")
    pub = ZmqPubQueue(addr, create=True, encoder=lambda o: {"x": o})
    sub = ZmqSubQueue(addr, create=False, decoder=lambda d: d["x"])
    sub.socket.setsockopt(zmq.RCVTIMEO, 200)
    try:
        # the queue's own setsockopt SUBSCRIBE "" is what makes delivery work at all;
        # the raw path carries pre-encoded msgpack that get() decodes back
        for expected, send in (
            ("raw", lambda: pub.put_raw(msgpack.packb({"x": "raw"}))),
            ("hello", lambda: pub.put("hello")),
        ):
            deadline = time.monotonic() + 5.0
            while not _recv(sub, expected):
                assert time.monotonic() < deadline, f"SUB never delivered {expected!r}"
                send()
    finally:
        pub.stop()
        sub.stop()


def test_decode_separates_receive_from_decoding(tmp_path):
    addr = _addr(tmp_path, "decode")
    push = ZmqPushQueue(addr, create=True, encoder=lambda o: {"v": o})
    pull = ZmqPullQueue(addr, create=False, decoder=lambda d: d["v"])
    try:
        push.put(7)
        raw = pull.get_raw()
        assert pull.decode(raw) == 7
    finally:
        push.stop()
        pull.stop()


def test_stop_of_unconnected_socket_is_safe(tmp_path):
    q = ZmqPullQueue(_addr(tmp_path, "idle"), create=True, decoder=lambda d: d)
    q.stop()


def test_async_pair_roundtrip_on_the_selected_loop(tmp_path):
    """The frontend reads worker replies through zmq.asyncio, whose sockets register themselves
    on ``loop.add_reader`` -- missing from the proactor loop Windows picks by default, which left
    a Windows serve prefilling requests it could never answer. ``use_zmq_event_loop`` is the seam,
    and this round-trip is what it exists to make work."""
    addr = _addr(tmp_path, "async")

    async def roundtrip():
        send = ZmqAsyncPushQueue(addr, create=True, encoder=lambda o: {"x": o})
        recv = ZmqAsyncPullQueue(addr, create=False, decoder=lambda d: d["x"])
        try:
            await send.put(21)
            return await asyncio.wait_for(recv.get(), timeout=10)
        finally:
            send.stop()
            recv.stop()

    use_zmq_event_loop()
    assert asyncio.run(roundtrip()) == 21


@pytest.mark.skipif(os.name != "nt", reason="POSIX loops already have add_reader")
def test_windows_loop_seam_drops_the_proactor_loop():
    """Asserted by type, not by a round-trip: pyzmq can also shim the proactor loop when
    tornado happens to be installed, and a green test must not rest on that accident."""
    use_zmq_event_loop()
    loop = asyncio.new_event_loop()
    try:
        assert isinstance(loop, asyncio.SelectorEventLoop)
    finally:
        loop.close()