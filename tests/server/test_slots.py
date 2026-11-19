"""The per-slot view (/slots + /metrics live blocks) is bookkeeping that fails silently: a
wrong slot index, a rate that looks like a rate, a busy row stranded after a crash. These
pin the slot assignment/reuse order, the sliding-window arithmetic against hand-computed
values, the abort/crash cleanup, the history width across a slot-count change, and that one
/metrics doc serves both the bench (requests) and the dashboard (live/engine/history).

Driven through the real StatsTracker feed (the listen() chokepoint) with real UserReply
objects on a pinned clock, the way the sibling accounting tests pin their clock.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

# Same shim as the sibling server tests: the venv may hold a non-editable install.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from freetoken.message import UserReply  # noqa: E402
from freetoken.server.control_api import register_control_routes  # noqa: E402
from freetoken.server.slots import build_metrics_blocks  # noqa: E402
from freetoken.server.stats import StatsTracker  # noqa: E402


@pytest.fixture
def clock(monkeypatch):
    """One pinned monotonic clock for both modules (stats forwards to slots; the aggregate
    window and the per-slot windows must read the same time)."""
    t = [100.0]
    monkeypatch.setattr("freetoken.server.stats.time.monotonic", lambda: t[0])
    monkeypatch.setattr("freetoken.server.slots.time.monotonic", lambda: t[0])
    return t


def _admit(uid: int, prompt: int, cached: int = 0) -> UserReply:
    return UserReply(uid=uid, incremental_output="", finished=False,
                     prompt_tokens_delta=prompt, cached_tokens=cached)


def _tokens(uid: int, n: int, finished: bool = False, timings: dict | None = None) -> UserReply:
    return UserReply(uid=uid, incremental_output="x" * n, finished=finished,
                     completion_tokens_delta=n, timings=timings)


def _row(rows: list[dict], slot: int) -> dict:
    return next(r for r in rows if r["id"] == slot)


def test_slots_assign_in_admission_order_and_reuse_the_lowest_free(clock):
    tr = StatsTracker(slots_total=2)
    for uid in (1, 2, 3):
        tr.on_new_user(uid)
    tr.observe(_admit(1, 500))
    tr.observe(_admit(2, 300))
    # The engine only admits a third request once a table slot frees; until its admission
    # reply arrives it is queue depth, not a row.
    assert tr.slots.stats() == {"total": 2, "active": 2, "queue_depth": 1}
    rows = tr.slots.rows(4096)
    assert [_row(rows, i)["state"] for i in (0, 1)] == ["prefill", "prefill"]
    assert _row(rows, 0)["prompt_tokens"] == 500

    clock[0] += 1.0
    tr.observe(_tokens(1, 4, finished=True))
    tr.observe(_admit(3, 200))
    rows = tr.slots.rows(4096)
    # Slot 0 freed and went to the newly admitted request; slot 1 never moved.
    assert _row(rows, 0)["prompt_tokens"] == 200 and _row(rows, 0)["state"] == "prefill"
    assert _row(rows, 1)["prompt_tokens"] == 300
    assert tr.slots.stats() == {"total": 2, "active": 2, "queue_depth": 0}


def test_rates_match_hand_computed_window_arithmetic(clock):
    tr = StatsTracker(window_s=5.0, slots_total=1)
    tr.on_new_user(1)
    tr.observe(_admit(1, 500, cached=100))          # t=100: admission
    clock[0] = 101.0
    tr.observe(_tokens(1, 1))                        # first token: prefill span 1 s
    clock[0] = 102.0
    tr.observe(_tokens(1, 10))
    row = _row(tr.slots.rows(4096), 0)
    # 11 tokens over the 101->102 span; prefill priced the 400 newly read tokens over 1 s.
    assert row["tok_s"] == 11.0
    assert row["prefill_tok_s"] == 400.0
    assert row["ttft_s"] == 1.0
    assert row["generated"] == 11 and row["prefill_done"] == 500
    assert row["ctx_used"] == 511 and row["reused"] == 100
    # A full prefix hit reads nothing new: no fabricated prefill rate. Its lone fresh token
    # has no window span yet: the mean with the span floored (1 / 0.25), never 1 / ~0.
    tr.observe(_tokens(1, 1, finished=True))         # free slot 0
    tr.on_new_user(2)
    tr.observe(_admit(2, 300, cached=300))
    clock[0] = 103.0
    tr.observe(_tokens(2, 1))
    row = _row(tr.slots.rows(4096), 0)
    assert row["prefill_tok_s"] is None
    assert row["tok_s"] == 4.0
    # Idle decay: past the window the rate is zero, not the last known value.
    clock[0] = 200.0
    assert _row(tr.slots.rows(4096), 0)["tok_s"] == 0.0
    assert tr.decode_tps() == 0.0


def test_abort_releases_the_slot_and_a_late_terminal_reply_is_inert(clock):
    tr = StatsTracker(slots_total=1)
    tr.on_new_user(1)
    tr.observe(_admit(1, 100))
    tr.on_abort(1)                                   # client gone / engine died
    assert tr.slots.stats()["active"] == 0
    # The abort ack (or a sampled token racing it) arrives after the release: it must not
    # resurrect the row nor double-free the slot.
    tr.observe(_tokens(1, 3))
    tr.observe(_tokens(1, 1, finished=True))
    assert tr.slots.stats()["active"] == 0
    tr.on_new_user(2)
    tr.observe(_admit(2, 50))
    assert _row(tr.slots.rows(4096), 0)["prompt_tokens"] == 50


def test_idle_row_keeps_the_last_finished_requests_ctx_and_drafts(clock):
    tr = StatsTracker(slots_total=1)
    tr.on_new_user(1)
    tr.observe(_admit(1, 100))
    clock[0] += 1.0
    tr.observe(_tokens(1, 20, finished=True, timings={"draft_n": 100, "draft_n_accepted": 60}))
    row = _row(tr.slots.rows(4096), 0)
    assert row["state"] == "idle" and row["is_processing"] is False
    assert row["ctx_used"] == 120
    assert (row["drafts_offered"], row["drafts_accepted"]) == (100, 60)
    assert row["tok_s"] == 0.0                       # idle rows keep the grid continuous


def test_history_adapts_to_the_slot_count_and_reset_clears_it(clock):
    tr = StatsTracker(slots_total=2)
    tr.on_new_user(1)
    tr.observe(_admit(1, 10))
    clock[0] += 1.0
    tr.observe(_tokens(1, 5))
    clock[0] += 1.0                                  # a real window span before sampling
    tr.sample_history()
    hist = tr.slots.history()
    assert len(hist["tok_s"]) == 1 and len(hist["slot_tok_s"]) == 2
    assert hist["slot_tok_s"][0][-1] == 5.0 and hist["slot_tok_s"][1][-1] == 0.0
    # The readiness meta reports the engine's effective cap: a wider grid pads the past
    # with idle zeros instead of misaligning the old points.
    tr.set_slots_total(3)
    hist = tr.slots.history()
    assert len(hist["slot_tok_s"]) == 3 and hist["slot_tok_s"][2] == [0.0]
    assert hist["slot_tok_s"][0] == [5.0]
    tr.reset_slots()
    assert tr.slots.history() == {"tok_s": [], "slot_tok_s": [[], [], []]}
    assert tr.slots.stats()["active"] == 0


def test_live_state_word_follows_the_slots_and_the_gate(clock):
    tr = StatsTracker(slots_total=2)
    state = SimpleNamespace(config=SimpleNamespace(served_model_name="m", max_seq_len=4096),
                            stats=tr, maintenance_state="serving")
    assert build_metrics_blocks(state)["live"]["state"] == "idle"
    tr.on_new_user(1)
    tr.observe(_admit(1, 100))
    live = build_metrics_blocks(state)["live"]
    assert live["state"] == "reading" and live["tok_s"] is None
    assert live["slots"] == {"total": 2, "active": 1, "queue_depth": 0}
    clock[0] += 1.0
    tr.observe(_tokens(1, 3))
    clock[0] += 1.0
    tr.observe(_tokens(1, 3))                        # 6 tokens over a 1 s window span
    live = build_metrics_blocks(state)["live"]
    assert live["state"] == "generating" and live["tok_s"] == 6.0
    state.maintenance_state = "loading"
    assert build_metrics_blocks(state)["live"]["state"] == "loading"
    state.maintenance_state = "failed"
    assert build_metrics_blocks(state)["live"]["state"] == "unloaded"
    # A bare state (older backend stand-in) reports an idle engine, never raises.
    bare = build_metrics_blocks(SimpleNamespace())
    assert bare["live"]["state"] == "idle" and bare["history"] == {"tok_s": [], "slot_tok_s": []}


def test_one_metrics_doc_serves_bench_and_dashboard(clock):
    tr = StatsTracker(slots_total=2)
    tr.on_new_user(1)
    tr.observe(_admit(1, 100))
    state = SimpleNamespace(config=SimpleNamespace(served_model_name="m", max_seq_len=4096),
                            stats=tr, maintenance_state="serving")
    app = FastAPI()
    register_control_routes(app, lambda: state)
    client = TestClient(app)

    doc = client.get("/metrics").json()
    # The bench's phase pool keeps its shape...
    assert doc["kind"] == "freetoken" and isinstance(doc["requests"], list)
    # ...and the dashboard relay finds basalt's blocks on the same doc.
    assert doc["engine"]["model"] == "m" and doc["engine"]["max_context"] == 4096
    assert doc["live"]["slots"]["total"] == 2
    assert isinstance(doc["time"], float)
    assert set(doc["history"]) == {"tok_s", "slot_tok_s"}

    rows = client.get("/slots").json()
    assert len(rows) == 2
    assert rows[0]["n_ctx"] == 4096 and rows[0]["state"] == "prefill"
    assert rows[1]["state"] == "idle"
