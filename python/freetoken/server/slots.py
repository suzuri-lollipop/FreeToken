"""Per-request slot reporting for GET /slots and the live/history blocks of GET /metrics.

The shape is basalt's slot_table, so llm-launcher's dashboard draws FreeToken with the same
per-slot rows (state / tok/s / prompt / generated / context / draft accept) it already draws
for basalt. FreeToken schedules continuous batches rather than fixed request slots: the
engine's table indices never reach the frontend, so the view is rebuilt here basalt-style --
the oldest admitted requests hold the lowest free slot indices until they end, the rest
wait -- from the same UserReply stream that feeds StatsTracker (admission = the prompt-tokens
reply, decode = completion deltas). No scheduler or wire change.

Prefill progress inside a chunked prompt is not on the reply stream, so a reading row reports
its prompt size with prefill_done=None (unknown, never a fabricated progress). The slot's
prefill rate is measured once the first token lands: newly computed prompt tokens over the
admission->first-token span, the same window ReqStats.prompt_ms covers. Draft counters ride
the terminal reply's timings, so an idle slot shows its last finished request's accept numbers
(basalt's slot_last semantics).

The 1 Hz history ring is pushed by StatsTracker.sample_loop from a daemon thread; every other
caller runs on the event loop. The lock keeps the two apart.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

HISTORY_LEN = 60
# basalt's RATE_MIN_SPAN_S: below this span there is no rate yet -- a lone fresh sample
# would otherwise divide by ~0 and read as an astronomical tok/s on the dashboard.
_MIN_RATE_SPAN_S = 0.25


@dataclass
class _Req:
    uid: int
    created_at: float
    state: str = "queued"  # queued -> prefill -> decoding
    slot: int | None = None
    admitted_at: float | None = None
    prompt_tokens: int = 0
    cached_tokens: int = 0
    generated: int = 0
    first_token_at: float | None = None
    prefill_tok_s: float | None = None
    # Same bounded sliding window as StatsTracker's aggregate one, per request.
    decode: deque = field(default_factory=lambda: deque(maxlen=4096))


class SlotTracker:
    def __init__(self, total: int = 4, window_s: float = 5.0) -> None:
        self._lock = threading.Lock()
        self._total = max(1, int(total))
        self._window_s = window_s
        self._reqs: dict[int, _Req] = {}
        self._by_slot: dict[int, int] = {}  # slot -> uid
        self._last: dict[int, dict] = {}  # slot -> last finished request's ctx / draft counts
        self._history: deque = deque(maxlen=HISTORY_LEN)

    # ------------------------------------------------------------------ feed (event loop)

    def on_new_user(self, uid: int, now: float | None = None) -> None:
        t = time.monotonic() if now is None else now
        with self._lock:
            if uid not in self._reqs:
                self._reqs[uid] = _Req(uid=uid, created_at=t)

    def on_abort(self, uid: int) -> None:
        """An abort was dispatched: the slot is as good as free (the engine releases it on
        the abort ack). Release now so a crashed backend -- whose terminal ack never arrives
        -- cannot strand a busy row; fail_pending_generations aborts every pending uid."""
        with self._lock:
            req = self._reqs.get(uid)
            if req is not None:
                self._release(req, None)

    def observe(self, reply: Any, now: float | None = None) -> None:
        t = time.monotonic() if now is None else now
        uid = getattr(reply, "uid", None)
        if uid is None:
            return
        with self._lock:
            req = self._reqs.get(uid)
            if req is None:
                return
            prompt_delta = getattr(reply, "prompt_tokens_delta", 0)
            if prompt_delta > 0 and req.admitted_at is None:
                req.admitted_at = t
                req.prompt_tokens = prompt_delta
                req.cached_tokens = getattr(reply, "cached_tokens", 0)
                req.slot = self._take_slot()
                if req.slot is not None:
                    self._by_slot[req.slot] = uid
                # An admitted request without a free row (slot total below the engine's real
                # cap: meta lost or still loading) stays counted as running, just not drawn.
                req.state = "prefill"
            completion_delta = getattr(reply, "completion_tokens_delta", 0)
            if completion_delta > 0:
                if req.first_token_at is None:
                    req.first_token_at = t
                    if req.admitted_at is not None:
                        span = t - req.admitted_at
                        newly_read = req.prompt_tokens - req.cached_tokens
                        req.prefill_tok_s = (
                            round(newly_read / span, 1) if newly_read > 0 and span > 1e-6 else None
                        )
                    req.state = "decoding"
                req.generated += completion_delta
                req.decode.append((t, completion_delta))
            if getattr(reply, "finished", False):
                self._release(req, reply)

    def _take_slot(self) -> int | None:
        return next((i for i in range(self._total) if i not in self._by_slot), None)

    def _release(self, req: _Req, reply: Any) -> None:
        if req.slot is not None:
            self._by_slot.pop(req.slot, None)
            timings = getattr(reply, "timings", None) or {}
            self._last[req.slot] = {
                "ctx": req.prompt_tokens + req.generated,
                "offered": timings.get("draft_n"),
                "accepted": timings.get("draft_n_accepted"),
            }
        self._reqs.pop(req.uid, None)

    # ------------------------------------------------------------------ reads

    def set_total(self, total: int) -> None:
        """The engine's effective max_running_req (readiness meta). The engine may clamp the
        requested cap at load; history adapts to the new width at read time."""
        if total > 0:
            with self._lock:
                self._total = int(total)

    def reset(self) -> None:
        """The engine generation died: drop every row and the old engine's speed history."""
        with self._lock:
            self._reqs.clear()
            self._by_slot.clear()
            self._last.clear()
            self._history.clear()

    def stats(self) -> dict:
        with self._lock:
            active = sum(1 for r in self._reqs.values() if r.state != "queued")
            queued = sum(1 for r in self._reqs.values() if r.admitted_at is None)
            return {"total": self._total, "active": active, "queue_depth": queued}

    def activity(self) -> tuple[int, int]:
        """(prefilling, decoding) request counts, for the live state word."""
        with self._lock:
            prefill = sum(1 for r in self._reqs.values() if r.state == "prefill")
            decoding = sum(1 for r in self._reqs.values() if r.state == "decoding")
            return prefill, decoding

    def _rate(self, req: _Req, now: float) -> float:
        cutoff = now - self._window_s
        while req.decode and req.decode[0][0] < cutoff:
            req.decode.popleft()
        if not req.decode:
            return 0.0
        total = sum(n for _ts, n in req.decode)
        span = now - req.decode[0][0]
        if span >= _MIN_RATE_SPAN_S:
            return total / span
        # Too fresh to have a window rate: the mean since the first token, span floored
        # (basalt's fallback), never total / ~0.
        if req.first_token_at is not None:
            return req.generated / max(_MIN_RATE_SPAN_S, now - req.first_token_at)
        return 0.0

    def slot_rates(self, now: float | None = None) -> tuple[float, ...]:
        t = time.monotonic() if now is None else now
        with self._lock:
            rates = []
            for i in range(self._total):
                req = self._reqs.get(self._by_slot.get(i))
                rates.append(self._rate(req, t) if req is not None and req.state == "decoding" else 0.0)
            return tuple(rates)

    def rows(self, n_ctx: int, now: float | None = None) -> list[dict]:
        """basalt /slots rows: one per slot, idle rows kept (the grid never changes shape)."""
        t = time.monotonic() if now is None else now
        with self._lock:
            rows = []
            for i in range(self._total):
                uid = self._by_slot.get(i)
                req = self._reqs.get(uid) if uid is not None else None
                last = self._last.get(i, {})
                row = {
                    "id": i, "n_ctx": n_ctx, "is_processing": req is not None, "state": "idle",
                    "tok_s": 0.0, "generated": 0, "prompt_tokens": 0, "prefill_done": 0,
                    "ctx_used": last.get("ctx", 0), "reused": None, "ttft_s": None,
                    "prefill_tok_s": None,
                    "drafts_offered": last.get("offered"), "drafts_accepted": last.get("accepted"),
                }
                if req is not None:
                    decoding = req.state == "decoding"
                    row.update(
                        state=req.state,
                        tok_s=round(self._rate(req, t), 1) if decoding else 0.0,
                        generated=req.generated,
                        prompt_tokens=req.prompt_tokens,
                        prefill_done=req.prompt_tokens if decoding else None,
                        ctx_used=req.prompt_tokens + req.generated,
                        reused=req.cached_tokens,
                        ttft_s=(
                            round(req.first_token_at - req.created_at, 2)
                            if decoding and req.first_token_at is not None else None
                        ),
                        prefill_tok_s=req.prefill_tok_s,
                    )
                rows.append(row)
            return rows

    # ------------------------------------------------------------------ history (sampler thread)

    def push_history(self, agg_tok_s: float, rates: tuple[float, ...]) -> None:
        with self._lock:
            self._history.append((round(agg_tok_s, 1), tuple(round(r, 1) for r in rates)))

    def history(self) -> dict:
        with self._lock:
            points = list(self._history)
            return {
                "tok_s": [p[0] for p in points],
                # Points sampled under a narrower slot count pad with 0.0 (the slot was idle
                # before it existed); a wider past is trimmed to the current width.
                "slot_tok_s": [
                    [(p[1][i] if i < len(p[1]) else 0.0) for p in points]
                    for i in range(self._total)
                ],
            }


def build_metrics_blocks(state: Any) -> dict:
    """The engine / live / history / time blocks of GET /metrics, in basalt's shape (the
    launcher's Strata-family relay reads exactly these). Never raises: a bare state reports
    an idle engine with no history."""
    cfg = getattr(state, "config", None)
    stats = getattr(state, "stats", None)
    model = getattr(cfg, "served_model_name", None) or ""
    n_ctx = int(getattr(cfg, "max_seq_len", 0) or 0)
    engine = {"model": model, "model_label": model, "max_context": n_ctx}
    slots = getattr(stats, "slots", None)
    if slots is None:
        return {
            "engine": engine,
            "live": {"state": "idle", "tok_s": None, "prefill_tok_s_mean": None,
                     "parallel": 0, "running": 0, "active": 0, "queued": 0,
                     "slots": {"total": 0, "active": 0, "queue_depth": 0}},
            "history": {"tok_s": [], "slot_tok_s": []},
            "time": time.time(),
        }
    st = slots.stats()
    n_prefill, n_decoding = slots.activity()
    maintenance = getattr(state, "maintenance_state", "serving")
    if maintenance == "loading":
        live_state = "loading"
    elif maintenance == "failed":
        live_state = "unloaded"
    elif n_decoding:
        live_state = "generating"
    elif n_prefill:
        live_state = "reading"
    else:
        live_state = "idle"
    busy = live_state in ("reading", "generating")
    # Combined decode = sum of the guarded per-slot rates (basalt's _tok_s), so the "all
    # slots" line is exactly the sum of the slot lines and a lone fresh sample cannot spike.
    decode_tps = sum(slots.slot_rates())
    prefill_tps = stats.prefill_tps() if hasattr(stats, "prefill_tps") else 0.0
    return {
        "engine": engine,
        "live": {
            "state": live_state,
            "tok_s": round(decode_tps, 1) if live_state == "generating" else None,
            "prefill_tok_s_mean": round(prefill_tps, 1) if busy else None,
            "parallel": st["total"],
            "running": st["active"],
            "active": st["active"],
            "queued": st["queue_depth"],
            "slots": st,
        },
        "history": slots.history(),
        "time": time.time(),
    }
