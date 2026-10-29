"""Backend startup/liveness supervision, decoupled from run_api_server so it is unit
testable without spawning real processes.

The scheduler/tokenizer worker processes are spawned by launch.start_subprocess, which
hands back a BackendHandle. ``drain_ready`` blocks until every worker has acked ready
(applying interleaved ("progress", desc, done, total) messages to a LoadProgress),
after which the caller flips the maintenance gate to "serving". Task 3 adds liveness
watching on top of this.

``revive`` adds strata's guarantee on top of liveness: a backend that dies or hangs while
the HTTP frontend keeps answering is restarted in-process instead of leaving the server
half-alive (see run_backend_supervisor).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Callable, List

logger = logging.getLogger(__name__)

# Strata's engine-restart policy (serve/server.py: RESTART_RETRY_S, restart(tries=3)).
RESTART_TRIES = 3
RESTART_RETRY_S = 15.0
# Strata's stall watchdog constants (serve/server.py: STALL_CPU_EPS_S, STALL_IO_EPS_B).
STALL_CPU_EPS_S = 0.5
STALL_IO_EPS_B = 1 << 20
# Shortest span over which "no CPU and no disk I/O" is accepted as proof of a hang; the
# window opens when the backend first falls silent, so this only trims a single sample.
STALL_MIN_WINDOW_S = 5.0


def phase_slug(desc: str) -> str:
    """Normalize a progress-bar desc into the stable /health phase enum. Order matters:
    the expert bars ("Loading experts …", "Loading expert banks …") both contain 'expert'
    and must win over the generic 'weight' match ("Loading weights (FTW)")."""
    d = (desc or "").lower()
    if "expert" in d:
        return "expert_banks"
    if "weight" in d:
        return "weights"
    if "graph" in d or "warm" in d:
        return "warmup"
    return "other"


@dataclass
class LoadProgress:
    desc: str = ""
    done_bytes: int = 0
    total_bytes: int = 0

    def update(self, desc: str, done: int, total: int) -> None:
        self.desc = desc
        self.done_bytes = done
        self.total_bytes = total

    def reset(self) -> None:
        """Start a fresh load (a revived backend reports its own bars from zero)."""
        self.desc = ""
        self.done_bytes = 0
        self.total_bytes = 0

    @property
    def phase(self) -> str:
        return phase_slug(self.desc)


@dataclass
class BackendHandle:
    """What start_subprocess hands back so run_api_server can supervise the workers it
    no longer blocks on."""

    ack_queue: Any
    processes: List[Any] = field(default_factory=list)
    expected_acks: int = 0


class WorkerDied(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _first_dead(processes: List[Any]) -> Any | None:
    for p in processes:
        try:
            if not p.is_alive():
                return p
        except Exception:  # noqa: BLE001 — treat an unqueryable handle as alive
            continue
    return None


def _as_error(msg: Any) -> str | None:
    """The reason a dying worker pushed as ("error", reason), or None for any other ack — lets
    the supervisor report the real cause (e.g. a config ValueError) rather than the generic
    "exited during load"."""
    if isinstance(msg, tuple) and msg and msg[0] == "error":
        return str(msg[1]) if len(msg) > 1 else "backend worker failed during load"
    return None


def _drain_pending_error(get: Callable[[float], Any], attempts: int = 5, timeout: float = 0.02) -> str | None:
    """After a worker death, poll the ack queue briefly for a final ("error", reason) ack still
    in flight (the queue feeder flushes on process exit), so the real reason wins over the
    generic message. Bounded so a genuinely reason-less death still fails fast."""
    for _ in range(attempts):
        try:
            msg = get(timeout)
        except Empty:
            continue
        err = _as_error(msg)
        if err is not None:
            return err
    return None


def _proc_cpu_io(pid: Any) -> "tuple[float, int] | None":
    """(CPU seconds, block-I/O bytes) one process has accumulated, read from /proc so the
    watch needs no dependency and no cooperation from the worker. The counters cover every
    thread, and a scheduler parked in a GPU sync spins there, which is what makes a silent
    but working engine look different from a hung one."""
    try:
        with open(f"/proc/{int(pid)}/stat", "rb") as fh:
            fields = fh.read().rsplit(b")", 1)[1].split()
        clk_tck = os.sysconf("SC_CLK_TCK") or 100
        cpu_s = (int(fields[11]) + int(fields[12])) / clk_tck
        io_bytes = 0
        with open(f"/proc/{int(pid)}/io", "rb") as fh:
            for line in fh:
                key, _, value = line.partition(b":")
                if key in (b"read_bytes", b"write_bytes"):
                    io_bytes += int(value.strip())
        return cpu_s, io_bytes
    except (OSError, ValueError, IndexError):
        return None  # gone, or no readable /proc: no reading, same as strata without psutil


def summed_cpu_io(processes: List[Any]) -> "tuple[float, int] | None":
    """Backend-wide activity: every worker's sample added. None if ANY worker cannot be
    read, because a partial sum would understate motion and end a working engine."""
    total_cpu = 0.0
    total_io = 0
    for p in processes:
        pid = getattr(p, "pid", None)
        if pid is None:
            return None
        sample = _proc_cpu_io(pid)
        if sample is None:
            return None
        total_cpu += sample[0]
        total_io += sample[1]
    return total_cpu, total_io


class FrozenEngine:
    """Strata's stall watchdog (serve/server.py ``_stall_check``) for a backend whose
    processes are alive but hung: silent for ``timeout_s`` AND no CPU and no disk I/O
    anywhere in the backend during that window.

    Silence alone is not proof - a long prefill says nothing while it runs - so the counters
    decide, and a backend that owes no answer is never watched at all. The parent cannot see
    GPU work, so the CPU term carries it. ``timeout_s`` <= 0 turns the watch off."""

    def __init__(
        self,
        timeout_s: float,
        *,
        cpu_eps_s: float = STALL_CPU_EPS_S,
        io_eps_b: int = STALL_IO_EPS_B,
        min_window_s: float = STALL_MIN_WINDOW_S,
        sample: Callable[[List[Any]], "tuple[float, int] | None"] = summed_cpu_io,
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.cpu_eps_s = cpu_eps_s
        self.io_eps_b = io_eps_b
        self.min_window_s = min_window_s
        self._sample = sample
        self._base: "tuple[float, tuple[float, int]] | None" = None
        self._quiet = 0.0
        self._warned = False

    def reset(self) -> None:
        """Forget a generation's history (called when the backend is replaced)."""
        self._base = None
        self._quiet = 0.0
        self._warned = False

    def check(self, processes: List[Any], *, busy: bool, quiet_s: float) -> str | None:
        """``busy``: the backend owes an answer for at least one request. ``quiet_s``: seconds
        since anything came back from it. Returns a reason once the silence cannot be
        explained by work; None while it can (or while there is nothing to decide about)."""
        if self.timeout_s <= 0 or not busy:
            self.reset()
            return None
        now = time.monotonic()
        if self._base is not None and quiet_s < self._quiet:
            self._base = None  # the silence clock rewound: something answered since the baseline
        self._quiet = quiet_s
        if self._base is None:
            base = self._sample(processes)
            if base is None:
                return None  # no reading, so nothing may be declared frozen
            self._base = (now, base)
            self._warned = False
            return None
        if quiet_s < self.timeout_s:
            return None
        window = now - self._base[0]
        current = self._sample(processes)
        if current is None:
            return None
        base_cpu, base_io = self._base[1]
        if current[0] - base_cpu > self.cpu_eps_s or current[1] - base_io > self.io_eps_b:
            # Silent but working: merely slow, so look again after a fresh window (and say so once).
            self._base = (now, current)
            if not self._warned:
                self._warned = True
                logger.warning(
                    "backend has answered nothing for %.0f s but is still using CPU or disk", quiet_s
                )
            return None
        if window < self.min_window_s:
            return None
        return (
            f"the engine answered nothing for {quiet_s:.0f} s and used no CPU or disk "
            "in that time (frozen)"
        )


def drain_ready(
    handle: BackendHandle,
    progress: LoadProgress,
    get: Callable[[float], Any] | None = None,
    poll: float = 1.0,
    on_meta: Callable[[dict], None] | None = None,
) -> None:
    """Block until ``expected_acks`` non-progress acks arrive, applying ("progress", …)
    tuples to ``progress``. Between acks, poll worker liveness; raise WorkerDied if a worker
    exits before readiness (e.g. CUDA OOM / corrupt weights during load).

    ("meta", payload) tuples carry optional backend metadata (per-unit cache byte costs);
    they are forwarded to ``on_meta`` and do NOT count toward readiness. meta is optional --
    an engine build that never emits it must not stall this drain, so nothing waits for it."""
    if get is None:
        def get(timeout: float) -> Any:
            return handle.ack_queue.get(timeout=timeout)

    ready = 0
    while ready < handle.expected_acks:
        try:
            msg = get(poll)
        except Empty:
            dead = _first_dead(handle.processes)
            if dead is not None:
                # The worker may have pushed its real failure reason just before exiting; give
                # the queue a brief window to surface it, else fall back to the generic message.
                reason = _drain_pending_error(get)
                raise WorkerDied(reason or f"backend worker {getattr(dead, 'name', '?')} exited during load")
            continue
        err = _as_error(msg)
        if err is not None:
            raise WorkerDied(err)
        if isinstance(msg, tuple) and msg and msg[0] == "progress":
            _tag, desc, done, total = msg
            progress.update(desc, done, total)
        elif isinstance(msg, tuple) and msg and msg[0] == "meta":
            if on_meta is not None:
                on_meta(msg[1] if len(msg) > 1 else {})
        else:
            ready += 1


def run_backend_supervisor(
    handle: BackendHandle,
    progress: LoadProgress,
    on_ready: Callable[[], None],
    on_failure: Callable[[str], None] | None = None,
    poll: float = 1.0,
    on_meta: Callable[[dict], None] | None = None,
    is_shutting_down: Callable[[], bool] | None = None,
    revive: Callable[[str, int], BackendHandle | None] | None = None,
    restart_tries: int = RESTART_TRIES,
    restart_retry_s: float = RESTART_RETRY_S,
    restart_when_asked: Callable[[], bool] | None = None,
    frozen: Callable[[], str | None] | None = None,
    on_restart: Callable[[str, int, int], None] | None = None,
    on_give_up: Callable[[str], None] | None = None,
) -> None:
    """Drain to readiness (watching liveness), flip the gate via ``on_ready``, then watch
    forever: a worker that dies AFTER ready (a scheduler crash while the frontend keeps
    answering 200) also fires ``on_failure`` — closing the half-alive-engine gap.

    ``on_meta`` receives the optional ("meta", …) backend metadata during the drain.

    ``is_shutting_down`` distinguishes a crash from an orderly stop: on SIGTERM/SIGINT
    (``ft serve`` ^C, or the desktop's stop) the api server sets a shutting-down flag before
    the workers exit, so their expected deaths are NOT misreported. When the flag is set at
    the moment a death is observed, the supervisor returns silently — no ``on_failure``, no
    ERROR log, no "failed" latch. A death seen while the flag is clear is a real crash and
    fires ``on_failure`` as before.

    ``revive(reason, attempt)`` is what makes a stopped engine come back instead of leaving a
    server that answers 200 over a dead model: on any death or hang the caller has already
    closed the gate (``on_failure``), and it then rebuilds the workers and hands back a new
    BackendHandle, which is drained to readiness and watched again from the top. When it
    returns None the attempt failed: try again after ``restart_retry_s``, up to
    ``restart_tries`` CONSECUTIVE failures (the count resets once a backend has stayed up that
    long). Out of tries, ``on_give_up`` fires and the supervisor waits for
    ``restart_when_asked()`` - strata's laziness, where the next request starts the engine
    again - rather than exiting or spinning on a failure that will not heal. Without ``revive``
    the supervisor reports once and returns, as before.

    ``frozen()`` is polled on the same tick and returns a reason when the backend has hung
    without exiting (see FrozenEngine); ``on_restart`` reports each revive attempt."""

    def _shutting_down() -> bool:
        return bool(is_shutting_down is not None and is_shutting_down())

    def _sleep(seconds: float) -> bool:
        """Wait out a back-off, noticing an orderly stop. True means: stop supervising."""
        deadline = time.monotonic() + max(0.0, seconds)
        while True:
            if _shutting_down():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(poll, remaining))

    def _watch(current: BackendHandle) -> str | None:
        """Watch a ready backend. Returns why it stopped, or None for an orderly stop."""
        while True:
            dead = _first_dead(current.processes)
            if dead is not None:
                if _shutting_down():
                    # Orderly stop in progress: the worker exit is expected - stay quiet.
                    return None
                return f"backend worker {getattr(dead, 'name', '?')} exited"
            if frozen is not None:
                try:
                    hung = frozen()
                except Exception:  # noqa: BLE001 - a broken probe must never end a working engine
                    logger.exception("frozen-engine probe failed")
                    hung = None
                if hung:
                    if _shutting_down():
                        return None
                    return str(hung)
            time.sleep(poll)

    state = {"attempt": 0, "ready_at": 0.0, "failed_at": 0.0}

    def _recover(reason: str) -> BackendHandle | None:
        """Restart the backend until it is back, or until there is no reason to keep trying.
        None means: stop supervising."""
        while not _shutting_down():
            if state["attempt"] > 0 and _sleep(restart_retry_s):
                return None
            state["attempt"] += 1
            if on_restart is not None:
                on_restart(reason, state["attempt"], restart_tries)
            try:
                revived = revive(reason, state["attempt"])
            except Exception:  # noqa: BLE001 - a failed rebuild is a retry, not a dead supervisor
                logger.exception("backend revive attempt %d raised", state["attempt"])
                revived = None
            if revived is not None:
                logger.warning("backend restarted (attempt %d) after: %s", state["attempt"], reason)
                return revived
            logger.error(
                "backend revive attempt %d of %d failed: %s",
                state["attempt"], restart_tries, reason,
            )
            if state["attempt"] >= restart_tries:
                if restart_when_asked is None:
                    return None
                if on_give_up is not None:
                    on_give_up(reason)
                logger.error(
                    "backend will not start (%s); answering 503 until a request asks again", reason
                )
                while not _shutting_down():
                    try:
                        if restart_when_asked():
                            break
                    except Exception:  # noqa: BLE001 - a failed nudge must not hang the wait
                        logger.exception("restart request check failed")
                    time.sleep(poll)
                if _shutting_down():
                    return None
                state["attempt"] = 0
        return None

    while True:
        try:
            drain_ready(handle, progress, poll=poll, on_meta=on_meta)
        except WorkerDied as exc:
            # A worker dying mid-load during an orderly shutdown is expected, not a load failure.
            if _shutting_down():
                return
            reason = exc.message
        else:
            on_ready()
            state["ready_at"] = time.monotonic()
            reason = _watch(handle)
            if reason is None:
                return
        # A backend that reached readiness on its own and then stayed up starts a new streak;
        # one that dies during (or right after) a revive keeps spending the old budget, so a
        # doomed restart cannot loop forever every restart_retry_s.
        if (
            state["attempt"] > 0
            and state["ready_at"] > state["failed_at"]
            and time.monotonic() - state["ready_at"] >= restart_retry_s
        ):
            state["attempt"] = 0
        state["failed_at"] = time.monotonic()
        if on_failure is not None:
            on_failure(reason)
        if revive is None:
            return
        handle = _recover(reason)
        if handle is None:
            return
