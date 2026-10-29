"""Env-gated host-side latency breakdown for the scheduler loop (FREETOKEN_LATENCY=1).

Answers "where does the time go" per window: what share of wall time the loop spends in
recv (idle/message wait), sched (batch build), fwd_issue (launch), and drain (waiting on
the previous batch's GPU work), plus queue depth and the spec-verify tally. Host-only
perf_counter, never a device sync -- the decode replay parks at the PLE lookup WAIT until
the post-drain fill signals, so any sync here would deadlock the loop (same rule as
moe/_debug_stats, whose host phases mirror these names when both flags are on).

Dumps one [latency] log line per FREETOKEN_LATENCY_EVERY_S at the loop's post-fill safe
point; FREETOKEN_LATENCY_OUT additionally appends each window as one JSON line for
tooling (the launcher calibration reads it). Diagnostics only: production runs carry a
``probe() is None`` check per call site.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque

from freetoken.utils import init_logger

logger = init_logger(__name__)

ENABLED = os.getenv("FREETOKEN_LATENCY", "0").strip() not in ("", "0", "false", "no")

# window length in seconds between [latency] dumps
def _every_s() -> float:
    # a bad value must not take the server down at import: fall back to the default
    try:
        return float(os.getenv("FREETOKEN_LATENCY_EVERY_S", "") or 10)
    except ValueError:
        return 10.0


EVERY_S = _every_s()
# optional JSONL sink for the per-window records (empty = log line only)
OUT = os.getenv("FREETOKEN_LATENCY_OUT", "").strip()

# bounded per-window samples behind the percentiles (a 10 s window at ~100 Hz loops
# fits; overflow drops the oldest, which only coarsens p95 on pathological loops)
_CAP = 1024


class LatencyProbe:
    """Per-window host-phase accumulators; dump resets the window."""

    def __init__(self, every_s: float = EVERY_S, out_path: str = OUT) -> None:
        self.every_s = max(every_s, 0.5)
        self.out_path = out_path or None
        self._sched = None
        self._t0 = time.monotonic()
        self._loops = 0
        self._phases: dict[str, deque] = {}
        self._sum: dict[str, float] = {}
        self._spec_base = (0, 0)  # cumulative verify counters at the last dump
        self._rej_base: dict[str, int] = {}

    def attach(self, sched) -> None:
        """Bind the scheduler once so the dump can read live depth/spec counters."""
        if self._sched is None:
            self._sched = sched

    def host_phase(self, name: str, dt: float) -> None:
        if name == "drain":
            self._loops += 1
        ms = dt * 1e3
        dq = self._phases.get(name)
        if dq is None:
            dq = self._phases[name] = deque(maxlen=_CAP)
        dq.append(ms)
        self._sum[name] = self._sum.get(name, 0.0) + ms

    def maybe_dump(self) -> None:
        now = time.monotonic()
        win = now - self._t0
        if win < self.every_s:
            return
        self._t0 = now
        try:
            rec = self._build(win)
        except Exception as exc:  # noqa: BLE001 -- diagnostics must never kill serving
            logger.warning(f"[latency] dump failed: {exc}")
            return
        self._reset()
        parts = [f"[latency] win={win:.1f}s loops={rec['loops']}"]
        g = rec["gauges"]
        parts.append(f"run={g.get('running', '-')} wait={g.get('waiting', '-')}")
        parts.append(f"rest={rec['rest_pct']:.0f}%")
        for name, st in sorted(rec["phases"].items(), key=lambda kv: -kv[1]["pct"]):
            parts.append(
                f"{name} {st['pct']:.0f}% (n={st['n']} mean={st['mean_ms']:.2f}"
                f" p95={st['p95_ms']:.2f}ms)"
            )
        spec = rec["spec"]
        if spec["verifies"] or spec["rejections"]:
            rate = (
                f"{100 * spec['accepted'] / spec['verifies']:.0f}%"
                if spec["verifies"] else "-"
            )
            parts.append(
                f"spec {spec['verifies']}v acc={rate} rej={spec['rejections']}"
            )
        logger.info_rank0(" ".join(parts))
        if self.out_path:
            try:
                with open(self.out_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
            except OSError as exc:
                logger.warning(f"[latency] out write failed: {exc}")

    def _build(self, win: float) -> dict:
        wall_ms = max(win * 1e3, 1e-9)
        phases = {}
        for name, dq in self._phases.items():
            if not dq:
                continue
            s = sorted(dq)
            phases[name] = {
                "n": len(s),
                "mean_ms": round(sum(s) / len(s), 3),
                "p50_ms": round(s[len(s) // 2], 3),
                "p95_ms": round(s[min(len(s) - 1, int(0.95 * len(s)))], 3),
                "max_ms": round(s[-1], 3),
                "pct": round(100.0 * self._sum[name] / wall_ms, 1),
            }
        rest = wall_ms - sum(self._sum.values())
        sched = self._sched
        gauges: dict = {}
        spec = {"verifies": 0, "accepted": 0, "rejections": {}}
        if sched is not None:
            try:
                gauges["running"] = len(sched.decode_manager.running_reqs)
                gauges["waiting"] = len(sched.prefill_manager.pending_list)
            except AttributeError:
                pass
            v = getattr(sched, "_spec_verifies", 0)
            a = getattr(sched, "_spec_accepted", 0)
            spec["verifies"] = v - self._spec_base[0]
            spec["accepted"] = a - self._spec_base[1]
            self._spec_base = (v, a)
            rejections = getattr(sched, "_spec_rejections", None) or {}
            d = self._rej_base
            for k, n in rejections.items():
                if n - d.get(k, 0) > 0:
                    spec["rejections"][k] = n - d.get(k, 0)
                d[k] = n
        return {
            "ts": time.time(),
            # every TP rank appends to the same out file; pid separates the streams
            "pid": os.getpid(),
            "win_s": round(win, 3),
            "loops": self._loops,
            "gauges": gauges,
            "rest_pct": round(100.0 * max(rest, 0.0) / wall_ms, 1),
            "phases": phases,
            "spec": spec,
        }

    def _reset(self) -> None:
        self._loops = 0
        self._sum.clear()
        for dq in self._phases.values():
            dq.clear()


_PROBE: LatencyProbe | None = LatencyProbe() if ENABLED else None


def probe() -> LatencyProbe | None:
    return _PROBE
