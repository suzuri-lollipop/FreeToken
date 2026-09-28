"""Single-instance pidfile and persisted serve state (daemon/pidfile.py).

The lock is the daemon's last line of defense against two serves writing into
one state/status tree, and ServeStateStore is what a restarted daemon re-adopts
a running serve from. A wrong flock flag or an unset starttime fails silently
until a second daemon corrupts state or re-adopts a different serve as its own.
"""

from __future__ import annotations

import os

import pytest

from freetoken.daemon.pidfile import (
    AlreadyRunning,
    ServeState,
    ServeStateStore,
    SingleInstance,
)


def test_acquire_writes_the_pid_and_release_drops_the_lock(tmp_path):
    lock_path = tmp_path / "run" / "ft.pid"
    with SingleInstance(str(lock_path)) as lock:
        assert os.path.isfile(lock_path)
        assert lock_path.read_text() == f"{os.getpid()}\n"

    # released: a second instance can acquire the same path
    with SingleInstance(str(lock_path)) as lock2:
        assert lock_path.read_text() == f"{os.getpid()}\n"


def test_second_acquire_while_held_raises_already_running(tmp_path):
    lock_path = tmp_path / "ft.pid"
    first = SingleInstance(str(lock_path))
    first.acquire()
    try:
        with pytest.raises(AlreadyRunning):
            SingleInstance(str(lock_path)).acquire()
    finally:
        first.release()


def test_double_release_is_a_noop(tmp_path):
    # release() must tolerate an already-closed fd (defensive OSError path)
    lock = SingleInstance(str(tmp_path / "ft.pid"))
    lock.acquire()
    lock.release()
    lock.release()  # no error on double release


def test_state_store_roundtrips_the_full_state(tmp_path):
    store = ServeStateStore(str(tmp_path / "state.json"))
    state = ServeState(
        model="model/dir",
        port=30000,
        pid=1234,
        args=["--port", "30000"],
        starttime=1717171717,
        log_path="/tmp/serve.log",
    )
    store.save(state)
    assert store.load() == state


def test_state_store_load_defaults_missing_optionals(tmp_path):
    store = ServeStateStore(str(tmp_path / "state.json"))
    store.save(ServeState(model="m", port=1, pid=2, args=[]))
    loaded = store.load()
    assert loaded == ServeState(model="m", port=1, pid=2, args=[])


def test_state_store_load_returns_none_for_missing_or_corrupt_file(tmp_path):
    store = ServeStateStore(str(tmp_path / "state.json"))
    assert store.load() is None

    (tmp_path / "state.json").write_text("{corrupt")
    assert store.load() is None


def test_state_store_load_returns_none_for_wrong_types(tmp_path):
    store = ServeStateStore(str(tmp_path / "state.json"))
    (tmp_path / "state.json").write_text('{"model": "m", "port": "not-int", "pid": 2}')
    assert store.load() is None

    (tmp_path / "state.json").write_text('{"port": 1, "pid": 2}')  # missing model
    assert store.load() is None


def test_state_store_clear_removes_file_and_tolerates_absence(tmp_path):
    store = ServeStateStore(str(tmp_path / "state.json"))
    store.clear()  # nothing to remove: no error
    store.save(ServeState(model="m", port=1, pid=2, args=[]))
    store.clear()
    assert store.load() is None