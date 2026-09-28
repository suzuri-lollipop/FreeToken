"""Bounded log ring cursor and fan-out semantics (daemon/logring.py).

The ring feeds the daemon's `?since=` log stream: seqs must stay monotonic
across eviction (the cursor is the all-time count, not a buffer index) and a
broken subscriber must never kill the tailer. All expectations come from the
documented contract - seq >= cursor inclusive, next cursor is the all-time
count - not from this implementation's internal bookkeeping.
"""

from __future__ import annotations

from freetoken.daemon.logring import LogRing


def test_seqs_assigned_monotonically():
    ring = LogRing(capacity=2)
    recs = [ring.append(f"line{i}") for i in range(3)]
    assert [r["seq"] for r in recs] == [0, 1, 2]


def test_since_returns_selected_records_and_alltime_cursor():
    ring = LogRing()
    for i in range(5):
        ring.append(f"line{i}")
    recs, cursor = ring.since(3)
    assert [r["seq"] for r in recs] == [3, 4]
    assert cursor == 5


def test_cursor_survives_eviction():
    ring = LogRing(capacity=3)
    for i in range(8):
        ring.append(f"line{i}")
    # buffer holds seqs 5..7 but the cursor is the all-time count
    recs, cursor = ring.since(0)
    assert [r["seq"] for r in recs] == [5, 6, 7]
    assert cursor == 8
    assert ring.cursor() == 8


def test_since_does_not_replay_evicted_records():
    ring = LogRing(capacity=3)
    for i in range(4):
        ring.append(f"line{i}")
    # seq 0 is evicted: polling from the login cursor must not return it
    recs, cursor = ring.since(0)
    assert [r["seq"] for r in recs] == [1, 2, 3]
    assert cursor == 4


def test_since_at_fresh_cursor_returns_empty():
    ring = LogRing()
    assert ring.since(0) == ([], 0)
    assert ring.since(100) == ([], 0)


def test_append_fans_out_to_subscribers():
    ring = LogRing()
    got: list[dict] = []
    ring.subscribe(got.append)
    rec = ring.append("hello")
    assert got == [rec]


def test_unsubscribe_stops_delivery():
    ring = LogRing()
    got: list[dict] = []
    ring.subscribe(got.append)
    ring.unsubscribe(got.append)
    ring.append("silent")
    assert got == []
    assert ring.subscriber_count() == 0


def test_raising_subscriber_does_not_break_the_ring():
    ring = LogRing()

    def boom(rec):
        raise RuntimeError("subscriber bug")

    healthy: list[dict] = []
    ring.subscribe(boom)
    ring.subscribe(healthy.append)
    rec = ring.append("still works")
    assert healthy == [rec]


def test_appended_records_carry_kind_and_ts():
    ring = LogRing()
    rec = ring.append("progress!", kind="progress", ts=12.5)
    assert rec["kind"] == "progress"
    assert rec["ts"] == 12.5
    assert rec["text"] == "progress!"