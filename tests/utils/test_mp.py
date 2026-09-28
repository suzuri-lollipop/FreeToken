"""ZMQ queue plumbing (utils/mp.py): real round-trips over ipc sockets.

The queues are the scheduler/tokenizer IPC backbone; both ends live behind
encoder/decoder callables, so a subscription filter, msgpack option or
socket-type mixup fails only in production. Each queue owns its own context,
so ipc:// (a real endpoint on tmp_path) stands in for the production transport.
PUB delivery needs the SUB peer fully connected, so the pub/sub test re-sends
until received instead of racing the connect.
"""

from __future__ import annotations

import time

import msgpack
import zmq

from freetoken.utils.mp import ZmqPullQueue, ZmqPushQueue, ZmqPubQueue, ZmqSubQueue


def _addr(tmp_path, name):
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