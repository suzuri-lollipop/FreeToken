"""Idle clock keeper: periodic tiny GPU work while the engine is idle.

Why: consumer/workstation GPUs park at ~180MHz SM clocks when idle and take
~100-300ms to ramp back to boost. A request arriving after idle pays that ramp
inside its prefill (measured: first request after 3min idle = 481ms vs 150-180ms
back-to-back on a 3-token cached-prefix prefill). A millisecond-scale matmul
burst every N ms keeps the clock domain awake at <0.5% duty.

Design constraints that shaped this:
- NO model forwards, NO NCCL collectives: a dummy decode replay would need
  every TP rank in lockstep and would touch scheduler/graph/PLE state. A local
  matmul on a private side stream needs no coordination -- each rank keeps its
  own clocks.
- Runs on a daemon thread with preallocated buffers and torch.mm(out=): steady
  state does zero allocations, holds the GIL only for the launch (~10s of us),
  and never interferes with CUDA-graph replay (separate stream, no allocator
  interaction beyond the one-time buffer alloc).
- Only pulses while genuinely idle: the scheduler loop calls notify() on every
  iteration that issued a batch; a pulse is skipped when real work ran within
  the last interval.

Enable with FREETOKEN_IDLE_KEEPALIVE_MS=<ms> (default 0 = off). Recommended
500-2000ms. The tradeoff while idle: a few watts of GPU power vs the ramp
latency on the next request.
"""
from __future__ import annotations

import os
import threading
import time

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_ENV = "FREETOKEN_IDLE_KEEPALIVE_MS"
# 2048^3 bf16 GEMM x N: ~1ms at boost clocks, a few ms at parked clocks --
# long enough to hold the boost state, short enough to stay <0.5% duty.
_MM_DIM = 2048
_MM_ITERS = 8


def keepalive_interval_ms() -> int:
    raw = os.getenv(_ENV, "0").strip()
    try:
        ms = int(raw)
    except ValueError:
        raise ValueError(f"{_ENV} must be an integer (ms), got {raw!r}")
    return max(0, ms)


class ClockKeeper:
    """Side-stream matmul pulser. start() spawns the thread; stop() joins it."""

    def __init__(self, device: torch.device, interval_ms: int) -> None:
        self.device = device
        self.interval_ms = interval_ms
        self._last_activity = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- scheduler-side hook ---------------------------------------------------
    def notify(self) -> None:
        """Mark real engine activity; pulses are suppressed for one interval."""
        self._last_activity = time.monotonic()

    # -- lifecycle --------------------------------------------------------------
    def start(self) -> None:
        if self.interval_ms <= 0 or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="clock-keeper"
        )
        self._thread.start()
        logger.info_rank0(
            f"Idle clock keeper enabled: {_MM_ITERS}x{_MM_DIM}^3 matmul burst "
            f"every {self.interval_ms}ms while idle"
        )

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
            self._thread = None

    # -- pulse loop ---------------------------------------------------------------
    def _run(self) -> None:
        try:
            with torch.cuda.device(self.device), torch.inference_mode():
                stream = torch.cuda.Stream(device=self.device)
                a = torch.randn(_MM_DIM, _MM_DIM, dtype=torch.bfloat16, device=self.device)
                b = torch.randn(_MM_DIM, _MM_DIM, dtype=torch.bfloat16, device=self.device)
                c = torch.empty(_MM_DIM, _MM_DIM, dtype=torch.bfloat16, device=self.device)
                interval = self.interval_ms / 1000.0
                while not self._stop.is_set():
                    self._stop.wait(interval)
                    if self._stop.is_set():
                        break
                    if time.monotonic() - self._last_activity < interval:
                        continue  # real work is keeping the clocks up
                    with torch.cuda.stream(stream):
                        for _ in range(_MM_ITERS):
                            torch.mm(a, b, out=c)
        except Exception as exc:  # noqa: BLE001 -- a keeper fault must never kill serving
            logger.warning(f"clock keeper stopped: {exc!r}")
