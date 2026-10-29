from __future__ import annotations

import queue

import pytest
from queue import Empty as _Empty

from freetoken.utils import progress
from freetoken.server.supervisor import (
    BackendHandle,
    FrozenEngine,
    LoadProgress,
    drain_ready,
    phase_slug,
    run_backend_supervisor,
)


def test_drain_ready_counts_ready_acks_and_applies_progress():
    q: "queue.Queue" = queue.Queue()
    q.put(("progress", "Loading weights (FTW)", 5, 10))
    q.put("Scheduler is ready")
    q.put(("progress", "Loading experts (parallel)", 8, 8))
    q.put("tokenizer ready")
    q.put("detokenizer ready")
    handle = BackendHandle(ack_queue=q, processes=[], expected_acks=3)
    progress = LoadProgress()

    drain_ready(handle, progress)

    assert progress.total_bytes == 8
    assert progress.done_bytes == 8
    assert progress.phase == "expert_banks"
    assert q.empty()


def test_drain_ready_forwards_meta_without_counting_it_ready():
    """("meta", payload) is optional backend metadata: forwarded to on_meta, but it must NOT
    count toward expected_acks (else a meta-emitting engine would flip ready one ack early)."""
    q: "queue.Queue" = queue.Queue()
    q.put(("meta", {"kv_bytes_per_token": 42}))
    q.put("Scheduler is ready")
    q.put("tokenizer ready")
    handle = BackendHandle(ack_queue=q, processes=[], expected_acks=2)
    seen: dict = {}

    drain_ready(handle, LoadProgress(), on_meta=lambda m: seen.update(m))

    assert seen == {"kv_bytes_per_token": 42}
    assert q.empty()  # both real acks consumed; meta did not short-count them


def test_drain_ready_ignores_meta_when_no_callback():
    """An engine that emits meta while the caller passes no on_meta must not stall or error."""
    q: "queue.Queue" = queue.Queue()
    q.put(("meta", {"kv_bytes_per_token": 7}))
    q.put("Scheduler is ready")
    handle = BackendHandle(ack_queue=q, processes=[], expected_acks=1)

    drain_ready(handle, LoadProgress())  # no on_meta

    assert q.empty()


def test_drain_ready_detects_worker_death_during_load():
    import queue

    import pytest

    from freetoken.server.supervisor import WorkerDied

    class DeadProc:
        name = "freetoken-TP0-scheduler"

        def is_alive(self) -> bool:
            return False

    q: "queue.Queue" = queue.Queue()  # never receives a ready ack
    handle = BackendHandle(ack_queue=q, processes=[DeadProc()], expected_acks=1)
    with pytest.raises(WorkerDied):
        drain_ready(handle, LoadProgress(), get=lambda _t: (_ for _ in ()).throw(_Empty()))


def test_drain_ready_raises_the_real_reason_from_an_error_ack():
    """A worker that pushes ("error", reason) just before dying surfaces THAT reason (e.g. a
    config ValueError), not the generic "exited during load"."""
    import queue

    import pytest

    from freetoken.server.supervisor import WorkerDied

    q: "queue.Queue" = queue.Queue()
    q.put(("error", "ValueError: --moe-backend 'hybrid' cannot compute q4_0 experts on the CPU"))
    handle = BackendHandle(ack_queue=q, processes=[], expected_acks=1)
    with pytest.raises(WorkerDied) as exc:
        drain_ready(handle, LoadProgress())
    assert "q4_0" in str(exc.value)


def test_supervisor_reports_the_worker_error_reason_via_on_failure():
    """End to end: an ("error", reason) ack from a dying worker reaches on_failure verbatim, so
    the desktop failure modal shows the actionable cause instead of "exited during load"."""
    import queue

    class DeadProc:
        name = "freetoken-TP0-scheduler"

        def is_alive(self) -> bool:
            return False

    q: "queue.Queue" = queue.Queue()
    q.put(("error", "ValueError: bad checkpoint config"))
    handle = BackendHandle(ack_queue=q, processes=[DeadProc()], expected_acks=1)
    seen: dict = {}
    from freetoken.server.supervisor import run_backend_supervisor

    run_backend_supervisor(
        handle,
        LoadProgress(),
        on_ready=lambda: seen.setdefault("ready", True),
        on_failure=lambda m: seen.setdefault("failure", m),
        poll=0.01,
    )
    assert "ready" not in seen
    assert seen["failure"] == "ValueError: bad checkpoint config"


def test_supervisor_reports_failure_on_startup_death():
    import queue

    class DeadProc:
        name = "freetoken-detokenizer-0"

        def is_alive(self) -> bool:
            return False

    q: "queue.Queue" = queue.Queue()
    handle = BackendHandle(ack_queue=q, processes=[DeadProc()], expected_acks=1)
    seen: dict = {}
    from freetoken.server.supervisor import run_backend_supervisor

    run_backend_supervisor(
        handle,
        LoadProgress(),
        on_ready=lambda: seen.setdefault("ready", True),
        on_failure=lambda m: seen.setdefault("failure", m),
        poll=0.01,
    )
    assert "ready" not in seen
    assert "detokenizer" in seen["failure"]


def test_supervisor_detects_post_ready_death():
    import queue

    class Proc:
        name = "freetoken-TP0-scheduler"

        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

    proc = Proc()
    q: "queue.Queue" = queue.Queue()
    q.put("scheduler ready")
    handle = BackendHandle(ack_queue=q, processes=[proc], expected_acks=1)
    seen: dict = {}
    from freetoken.server.supervisor import run_backend_supervisor

    def on_ready() -> None:
        seen["ready"] = True
        proc._alive = False  # die right after readiness

    run_backend_supervisor(
        handle, LoadProgress(), on_ready=on_ready,
        on_failure=lambda m: seen.setdefault("failure", m), poll=0.01,
    )
    assert seen.get("ready") is True
    assert "scheduler" in seen["failure"]


def test_supervisor_silent_on_post_ready_death_during_shutdown():
    """An orderly stop (SIGTERM/^C) sets a shutting-down flag before the workers exit. A
    post-ready death observed while that flag is set is EXPECTED — the watchdog must return
    silently: no on_failure, so no ERROR log and no "failed" latch during a clean stop."""
    import queue

    class Proc:
        name = "freetoken-TP0-scheduler"

        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

    proc = Proc()
    q: "queue.Queue" = queue.Queue()
    q.put("scheduler ready")
    handle = BackendHandle(ack_queue=q, processes=[proc], expected_acks=1)
    seen: dict = {}
    shutting_down = {"v": False}
    from freetoken.server.supervisor import run_backend_supervisor

    def on_ready() -> None:
        seen["ready"] = True
        shutting_down["v"] = True  # stop requested…
        proc._alive = False        # …and the worker exits as part of that stop

    run_backend_supervisor(
        handle, LoadProgress(), on_ready=on_ready,
        on_failure=lambda m: seen.setdefault("failure", m), poll=0.01,
        is_shutting_down=lambda: shutting_down["v"],
    )
    assert seen.get("ready") is True
    assert "failure" not in seen  # graceful stop: the death was not reported


def test_supervisor_silent_on_startup_death_during_shutdown():
    """A worker dying mid-load while an orderly stop is already in progress must not be
    reported as a load failure either."""
    import queue

    class DeadProc:
        name = "freetoken-detokenizer-0"

        def is_alive(self) -> bool:
            return False

    q: "queue.Queue" = queue.Queue()  # never receives a ready ack
    handle = BackendHandle(ack_queue=q, processes=[DeadProc()], expected_acks=1)
    seen: dict = {}
    from freetoken.server.supervisor import run_backend_supervisor

    run_backend_supervisor(
        handle,
        LoadProgress(),
        on_ready=lambda: seen.setdefault("ready", True),
        on_failure=lambda m: seen.setdefault("failure", m),
        poll=0.01,
        is_shutting_down=lambda: True,  # stop already requested before load finished
    )
    assert "ready" not in seen
    assert "failure" not in seen  # silenced: expected exit during shutdown


# ---------------------------------------------------------------------------
# revive: a backend that dies or hangs AFTER readiness is rebuilt in place, so the server
# never settles into answering requests over a dead engine.
# ---------------------------------------------------------------------------


def test_supervisor_revives_the_backend_after_a_post_ready_death():
    """The state this exists for: the API is up, the engine is gone. The supervisor must not
    return there - it fires on_failure (which closes the gate), rebuilds, drains the NEW
    handle to readiness and keeps watching, so the gate opens again on its own."""
    class Proc:
        name = "freetoken-TP0-scheduler"

        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

    first, second = Proc(), Proc()
    q1: "queue.Queue" = queue.Queue()
    q1.put("scheduler ready")
    handle = BackendHandle(ack_queue=q1, processes=[first], expected_acks=1)
    q2: "queue.Queue" = queue.Queue()
    q2.put("scheduler ready")
    revived = BackendHandle(ack_queue=q2, processes=[second], expected_acks=1)
    seen: dict = {"failures": [], "revives": [], "restarts": [], "ready": 0}
    shutting_down = {"v": False}

    def on_ready() -> None:
        seen["ready"] += 1
        if seen["ready"] == 1:
            first._alive = False  # the engine dies while the API keeps answering
        else:
            # The new generation is up and watched; end the run as an orderly stop would.
            second._alive = False
            shutting_down["v"] = True

    def revive(reason: str, attempt: int):
        seen["revives"].append((reason, attempt))
        return revived

    run_backend_supervisor(
        handle, LoadProgress(), on_ready=on_ready,
        on_failure=lambda m: seen["failures"].append(m), poll=0.01,
        is_shutting_down=lambda: shutting_down["v"],
        revive=revive, restart_retry_s=0.0,
        on_restart=lambda r, a, t: seen["restarts"].append((r, a, t)),
    )
    assert seen["failures"] == ["backend worker freetoken-TP0-scheduler exited"]
    assert seen["revives"][0][1] == 1
    assert "exited" in seen["revives"][0][0]
    assert seen["restarts"] == [(seen["failures"][0], 1, 3)]
    assert seen["ready"] == 2  # ready, dead, ready again - without a second server


def test_supervisor_gives_up_then_lets_a_request_start_another_round():
    """Out of tries the serve stays alive answering a retryable 503 rather than exiting or
    re-spawning forever; the next request (``restart_when_asked``) buys a fresh round."""
    class Proc:
        name = "freetoken-TP0-scheduler"

        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

    first = Proc()
    q1: "queue.Queue" = queue.Queue()
    q1.put("scheduler ready")
    handle = BackendHandle(ack_queue=q1, processes=[first], expected_acks=1)
    seen: dict = {"attempts": [], "ready": 0, "give_up": 0}
    calls = {"revive": 0, "asked_after": 0}
    shutting_down = {"v": False}

    def on_ready() -> None:
        seen["ready"] += 1
        if seen["ready"] == 1:
            first._alive = False
        else:
            shutting_down["v"] = True

    def revive(reason: str, attempt: int):
        calls["revive"] += 1
        seen["attempts"].append(attempt)
        if calls["revive"] <= 3:
            return None  # three failed rounds: weights will not load
        q2: "queue.Queue" = queue.Queue()
        q2.put("scheduler ready")
        back = Proc()
        back._alive = False  # end the run the way an orderly stop would, right after revival
        return BackendHandle(ack_queue=q2, processes=[back], expected_acks=1)

    def asked() -> bool:
        # A client shows up a couple of ticks after the give-up, which is when the give-up
        # latch is expected to be armed and waiting.
        calls["asked_after"] += 1
        return calls["asked_after"] >= 2

    run_backend_supervisor(
        handle, LoadProgress(), on_ready=on_ready, on_failure=lambda m: None, poll=0.01,
        is_shutting_down=lambda: shutting_down["v"],
        revive=revive, restart_tries=3, restart_retry_s=0.0, restart_when_asked=asked,
        on_give_up=lambda _r: seen.__setitem__("give_up", seen["give_up"] + 1),
    )
    assert seen["give_up"] == 1
    assert seen["attempts"] == [1, 2, 3, 1]  # the re-arm starts a new streak
    assert seen["ready"] == 2


# ---------------------------------------------------------------------------
# FrozenEngine: the hung-but-alternatively-invisible backend strata calls a stall.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cpu_step", "io_step", "expect_reason"),
    [
        (0.0, 0, True),                      # silent AND idle: hung, revive it
        (5.0, 0, False),                     # silent but on the CPU: merely slow
        (0.0, 1 << 30, False),               # silent but reading/writing: merely slow
        (0.4, 1 << 20 - 1, True),            # inside both epsilons: still no work
    ],
)
def test_frozen_engine_declares_a_hang_only_without_any_work(cpu_step, io_step, expect_reason):
    samples = iter([(10.0, 5000), (10.0 + cpu_step, 5000 + io_step)])
    watch = FrozenEngine(0.05, min_window_s=0.0, sample=lambda _p: next(samples))

    assert watch.check([], busy=True, quiet_s=0.0) is None  # the first look only takes a baseline
    reason = watch.check([], busy=True, quiet_s=60.0)
    assert (reason is not None) is expect_reason
    if expect_reason:
        assert "frozen" in reason


def test_frozen_engine_never_decides_on_nothing_to_decide():
    def no_sample(_p):
        return None

    def never(_p):
        raise AssertionError("sampled while it should not be")

    assert FrozenEngine(0.0, sample=never).check([], busy=True, quiet_s=999.0) is None
    assert FrozenEngine(1.0, sample=never).check([], busy=False, quiet_s=999.0) is None
    blind = FrozenEngine(1.0, min_window_s=0.0, sample=no_sample)
    assert blind.check([], busy=True, quiet_s=0.0) is None
    assert blind.check([], busy=True, quiet_s=999.0) is None  # no /proc reading, no verdict

    # A new generation starts from nothing: no carry-over baseline, no stale warning latch.
    watch = FrozenEngine(0.05, min_window_s=0.0, sample=lambda _p: (1.0, 1))
    assert watch.check([], busy=True, quiet_s=0.0) is None
    watch.reset()
    assert watch.check([], busy=True, quiet_s=0.0) is None
    assert watch.check([], busy=True, quiet_s=999.0) is not None


# ---------------------------------------------------------------------------
# progress sink: the byte_bar -> set_progress_sink pipe drain_ready consumes, and the
# phase_slug normalization that labels the three serve bars.
# ---------------------------------------------------------------------------


def test_phase_slug_normalizes_the_three_serve_bars():
    assert phase_slug("Loading weights (FTW)") == "weights"
    assert phase_slug("Loading experts (parallel)") == "expert_banks"
    assert phase_slug("Loading expert banks (FTW)") == "expert_banks"
    assert phase_slug("something else") == "other"
    assert phase_slug("") == "other"


def test_byte_bar_emits_to_installed_sink_then_stops_after_clear():
    seen: list[tuple[str, int, int]] = []
    progress.set_progress_sink(lambda desc, done, total: seen.append((desc, done, total)))
    try:
        bar = progress.byte_bar(total=100, desc="Loading weights (FTW)")
        bar.update(100)  # full jump always emits
        bar.close()
    finally:
        progress.set_progress_sink(None)
    assert seen and seen[-1] == ("Loading weights (FTW)", 100, 100)

    # After clearing, a new bar must not emit.
    seen.clear()
    bar = progress.byte_bar(total=100, desc="Loading weights (FTW)")
    bar.update(100)
    bar.close()
    assert seen == []
