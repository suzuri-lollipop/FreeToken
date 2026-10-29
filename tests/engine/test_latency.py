"""LatencyProbe window aggregation and dump records (engine/latency.py), CPU-only.

The probe runs inside the serving loop where a crash kills the server and a wrong
number silently misleads every calibration that reads [latency]. What fails SILENTLY
is the accounting: window percentages that no longer sum near the wall time, a window
that never resets (percentages drift to 0 forever), the per-window reject/verify
tally printing cumulative counts, or the JSONL schema drifting under the launcher.
Those are pure host logic, pinned here without an engine. The call-site gating in
the scheduler loop needs a live loop (the e2e gate's job).
"""

from __future__ import annotations

import importlib
import json
import time
from types import SimpleNamespace

import pytest


@pytest.fixture
def latency_mod(monkeypatch):
    from freetoken.engine import latency

    yield latency
    monkeypatch.undo()
    importlib.reload(latency)


def _force_window(probe) -> None:
    probe._t0 = time.monotonic() - 10 * probe.every_s


class _Sched:
    def __init__(self):
        self.decode_manager = SimpleNamespace(running_reqs=[1, 2])
        self.prefill_manager = SimpleNamespace(pending_list=[3])
        self._spec_verifies = 10
        self._spec_accepted = 6
        self._spec_rejections = {"low_prob": 2, "resync": 1}


def test_gate_off_by_default(latency_mod):
    assert latency_mod.probe() is None


def test_gate_on_via_env(latency_mod, monkeypatch):
    monkeypatch.setenv("FREETOKEN_LATENCY", "1")
    importlib.reload(latency_mod)
    assert latency_mod.probe() is not None


def test_no_dump_before_window(tmp_path):
    from freetoken.engine.latency import LatencyProbe

    probe = LatencyProbe(every_s=60, out_path=str(tmp_path / "lat.jsonl"))
    probe.host_phase("drain", 0.001)
    probe.maybe_dump()
    assert not (tmp_path / "lat.jsonl").exists()


def test_window_stats_reset_and_jsonl(tmp_path):
    from freetoken.engine.latency import LatencyProbe

    out = tmp_path / "lat.jsonl"
    probe = LatencyProbe(every_s=0.5, out_path=str(out))
    probe.host_phase("recv", 0.010)
    probe.host_phase("sched", 0.002)
    for i in range(1, 101):
        probe.host_phase("drain", i * 0.001)
    _force_window(probe)
    probe.maybe_dump()

    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["loops"] == 100  # counted by the drain beats, one per loop
    drain = rec["phases"]["drain"]
    assert drain["n"] == 100
    assert abs(drain["max_ms"] - 100.0) < 1e-6
    assert abs(drain["p95_ms"] - 96.0) < 1e-6
    assert abs(drain["mean_ms"] - 50.5) < 0.01
    # wall is ~10 s, drain sum is 50.5 s of overlapping beats -> capped share is fine
    assert rec["phases"]["recv"]["pct"] == pytest.approx(0.1, abs=0.2)
    assert 0.0 <= rec["rest_pct"] < 100.0

    # window reset: second dump sees only the new samples
    probe.host_phase("recv", 0.005)
    _force_window(probe)
    probe.maybe_dump()
    rec2 = json.loads(out.read_text(encoding="utf-8").splitlines()[1])
    assert rec2["loops"] == 0
    assert "drain" not in rec2["phases"]  # empty phases drop out after the reset
    assert rec2["phases"]["recv"]["n"] == 1


def test_gauges_and_spec_tally_are_window_deltas(tmp_path):
    from freetoken.engine.latency import LatencyProbe

    out = tmp_path / "lat.jsonl"
    probe = LatencyProbe(every_s=0.5, out_path=str(out))
    sched = _Sched()
    probe.attach(sched)
    probe.attach(sched)  # idempotent
    _force_window(probe)
    probe.maybe_dump()

    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert rec["gauges"] == {"running": 2, "waiting": 1}
    assert rec["spec"]["verifies"] == 10
    assert rec["spec"]["accepted"] == 6
    assert rec["spec"]["rejections"] == {"low_prob": 2, "resync": 1}

    sched._spec_verifies = 18
    sched._spec_accepted = 9
    sched._spec_rejections = {"low_prob": 3, "resync": 1}
    _force_window(probe)
    probe.maybe_dump()
    rec2 = json.loads(out.read_text(encoding="utf-8").splitlines()[1])
    assert rec2["spec"]["verifies"] == 8
    assert rec2["spec"]["accepted"] == 3
    assert rec2["spec"]["rejections"] == {"low_prob": 1}


def test_dump_survives_broken_gauges(tmp_path):
    from freetoken.engine.latency import LatencyProbe

    probe = LatencyProbe(every_s=0.5, out_path=str(tmp_path / "lat.jsonl"))
    probe.attach(object())  # no managers, no counters: getattr defaults carry it
    probe.host_phase("sched", 0.001)
    _force_window(probe)
    probe.maybe_dump()  # must not raise into the serving loop
    assert (tmp_path / "lat.jsonl").exists()
