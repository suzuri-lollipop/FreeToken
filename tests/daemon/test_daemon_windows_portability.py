"""Windows-portability tests for the daemon's OS layer, runnable on Linux.

The Windows branches are exercised by monkeypatching the dispatch flags
(``osproc._IS_WINDOWS``, ``os.name``) and replacing ``osproc._win`` with a stub, plus a
few properties (torch-free import, safe degradation off nt) that run for real here."""

from __future__ import annotations

import os
import signal
import subprocess

import pytest

from freetoken.daemon import osproc, winproc
from freetoken.daemon.pidfile import AlreadyRunning, SingleInstance


@pytest.fixture
def win_mode(monkeypatch):
    """Force osproc down its Windows branch with a recording stub for winproc."""

    class WinStub:
        def __init__(self):
            self.calls: list[tuple] = []
            self.alive = True
            self.starttime = 99
            self.cmdline: list[str] = []
            self.tree: list[int] = []
            self.working_set = 4096

        def pid_alive(self, pid):
            self.calls.append(("pid_alive", pid))
            return self.alive

        def read_starttime(self, pid):
            self.calls.append(("read_starttime", pid))
            return self.starttime

        def read_cmdline(self, pid):
            self.calls.append(("read_cmdline", pid))
            return self.cmdline

        def tree_pids(self, root):
            self.calls.append(("tree_pids", root))
            return self.tree

        def signal_tree(self, pid, code):
            self.calls.append(("signal_tree", pid, code))

        def set_oom_score_adj(self, *a):
            raise AssertionError("must not be reached")

        def working_set_bytes(self, pid):
            self.calls.append(("working_set_bytes", pid))
            return self.working_set

    stub = WinStub()
    monkeypatch.setattr(osproc, "_IS_WINDOWS", True)
    monkeypatch.setattr(osproc, "_HAS_PROC", False)
    monkeypatch.setattr(osproc, "_win", stub)
    # Any POSIX pid touch is a dispatch bug: make it explode.
    monkeypatch.setattr(osproc.os, "kill", lambda *a: pytest.fail("os.kill on the Windows path"))
    return stub


# --------------------------------------------------------------------------- winproc


def test_winproc_degrades_off_windows():
    """On a POSIX host every winproc helper returns its safe default instead of raising --
    osproc only dispatches here when os.name == 'nt', but the import-safety net says the
    module must at least load anywhere."""
    assert os.name != "nt"  # this box is not the target; the guards below are the point
    assert winproc.pid_alive(1) is False
    assert winproc.read_starttime(1) is None
    assert winproc.working_set_bytes(1) == 0
    assert winproc.snapshot_ppid_map() is None
    assert winproc.read_cmdline(1) == []
    assert winproc.tree_pids(1) == []
    winproc.signal_tree(1, 15)  # no-op


def test_build_tree_is_transitive_and_root_first():
    # 5's parent is 6, so it is NOT under root 1; the tree is 1 -> {2,3}, 2 -> {4}.
    ppid_of = {1: 0, 2: 1, 3: 1, 4: 2, 5: 6}
    tree = winproc.build_tree(1, ppid_of)
    assert tree[0] == 1
    assert set(tree) == {1, 2, 3, 4}
    assert winproc.build_tree(99, ppid_of) == []  # unknown root


def test_build_tree_survives_ppid_cycles():
    assert winproc.build_tree(1, {1: 2, 2: 1}) == [1, 2]


# --------------------------------------------------------------------------- osproc dispatch


def test_pid_alive_dispatches_to_winproc(win_mode):
    assert osproc.pid_alive(4242) is True
    assert win_mode.calls == [("pid_alive", 4242)]


def test_starttime_and_footprint_dispatch(win_mode):
    win_mode.tree = [7, 8, 9]
    assert osproc.read_starttime(7) == 99
    assert osproc.read_pss_bytes(7) == 4096
    assert osproc.tree_pids(7) == [7, 8, 9]


def test_signal_group_becomes_tree_terminate(win_mode):
    osproc.signal_group(7, signal.SIGTERM)
    assert ("signal_tree", 7, int(signal.SIGTERM)) in win_mode.calls


def test_oom_score_always_fails_on_windows(win_mode):
    assert osproc.set_oom_score_adj(7, 500) is False


def test_serve_identity_full_argv_check(win_mode):
    win_mode.cmdline = ["python", "-m", "freetoken.cli", "serve", "--model", "M", "--port", "1919"]
    assert osproc.is_ft_serve_on_port(7, 1919) is True
    assert osproc.is_ft_serve_on_port(7, 1920) is False
    win_mode.cmdline = ["notepad.exe"]
    assert osproc.is_ft_serve_on_port(7, 1919) is False


def test_serve_identity_degrades_without_argv(win_mode):
    """A pid we cannot read argv for (protected process, 32/64-bit mismatch) still adopts on
    liveness + start time -- the documented degradation."""
    assert osproc.is_ft_serve_on_port(7, 1919, starttime=99) is True
    assert osproc.is_ft_serve_on_port(7, 1919, starttime=1) is False  # PID reuse guard wins
    win_mode.alive = False
    assert osproc.is_ft_serve_on_port(7, 1919) is False


# --------------------------------------------------------------------------- pidfile lock


def test_single_instance_lock_roundtrip(tmp_path):
    path = str(tmp_path / "daemon.pid")
    first = SingleInstance(path)
    first.acquire()
    try:
        assert int(open(path).read().strip()) == os.getpid()
        rival = SingleInstance(path)
        with pytest.raises(AlreadyRunning):
            rival.acquire()
    finally:
        first.release()
    again = SingleInstance(path)
    again.acquire()  # release must drop the lock, even in-process
    again.release()


# --------------------------------------------------------------------------- spawn flags


def test_spawn_serve_uses_process_group_flag_on_windows(monkeypatch, tmp_path):
    captured = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            captured.update(kwargs)
            self.pid = os.getpid()

    log = str(tmp_path / "serve.log")
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    # CREATE_NEW_PROCESS_GROUP exists only on real Windows; the nt value is 0x200.
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, raising=False)
    monkeypatch.setattr(os, "name", "nt", raising=False)
    from freetoken.daemon import serve_manager

    serve_manager.spawn_serve(["python", "x"], log)
    assert captured["creationflags"] == 0x200


def test_spawn_serve_plain_on_posix(monkeypatch, tmp_path):
    captured = {}

    class FakePopen:
        def __init__(self, argv, **kwargs):
            captured.update(kwargs)
            self.pid = os.getpid()

    log = str(tmp_path / "serve.log")
    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    from freetoken.daemon import serve_manager

    serve_manager.spawn_serve(["python", "x"], log)
    assert "creationflags" not in captured


# --------------------------------------------------------------------------- guards


def test_host_banks_windows_guards(monkeypatch):
    """mlock and madvise degrade to the documented PAGEABLE path instead of raising."""
    pytest.importorskip("torch")  # host_banks imports torch at module scope
    from freetoken.moe import host_banks

    with pytest.raises(OSError, match="Windows"):
        monkeypatch.setattr(os, "name", "nt", raising=False)
        host_banks._os_lock(0, 4096)
