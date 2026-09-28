"""Defensive /proc helpers (daemon/osproc.py).

The supervisor trusts these to decide who to signal, adopt and account for;
the expectations come from the /proc formats themselves (stat's comm field may
contain spaces and parens, smaps_rollup reports kB, cmdline is NUL-separated).
A wrong parse here does not raise - it silently adopts a recycled pid or
charges a serve its worker's memory twice over.
"""

from __future__ import annotations

import freetoken.daemon.osproc as osproc


def test_pid_alive_probe_semantics(monkeypatch):
    def fake_kill(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(osproc.os, "kill", fake_kill)
    assert osproc.pid_alive(123) is False
    assert osproc.pid_alive(0) is False  # negative/zero never probe
    assert osproc.pid_alive(-5) is False


def test_pid_alive_eprm_is_alive(monkeypatch):
    def fake_kill(pid, sig):
        raise PermissionError

    monkeypatch.setattr(osproc.os, "kill", fake_kill)
    assert osproc.pid_alive(123) is True


def test_read_cmdline_splits_nul_argv(monkeypatch):
    monkeypatch.setattr(osproc, "_read_proc", lambda pid, name: "ft\x00serve\x00--port\x0012345\x00")
    assert osproc.read_cmdline(9) == ["ft", "serve", "--port", "12345"]


def test_read_cmdline_missing_is_empty(monkeypatch):
    monkeypatch.setattr(osproc, "_read_proc", lambda pid, name: None)
    assert osproc.read_cmdline(9) == []


def test_stat_parser_handles_comm_with_spaces_and_parens(monkeypatch):
    # (comm) may itself contain spaces and parens; fields 3..22 follow (state=3,
    # ppid=4, pgrp=5, ..., starttime=22 is the clock-tick count)
    raw = "123 (my (weird) serve) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 424242 0 0\n"
    monkeypatch.setattr(osproc, "_read_proc", lambda pid, name: raw)

    assert osproc.proc_pgid(123) == 2  # stat field 3 is state, field 5 is pgid
    assert osproc.read_starttime(123) == 424242  # field 22


def test_stat_parser_short_or_unparsable_returns_none(monkeypatch):
    monkeypatch.setattr(osproc, "_read_proc", lambda pid, name: "123 (serve) S only")
    assert osproc.proc_pgid(123) is None
    assert osproc.read_starttime(123) is None

    monkeypatch.setattr(osproc, "_read_proc", lambda pid, name: "no-paren-here")
    assert osproc.proc_pgid(123) is None


def test_read_pss_bytes_converts_kb(monkeypatch):
    smaps = "Rss: 1000 kB\nPss: 250 kB\nPss_Dirty: 10 kB\n"
    monkeypatch.setattr(osproc, "_read_proc", lambda pid, name: smaps)
    assert osproc.read_pss_bytes(123) == 250 * 1024


def test_read_pss_bytes_missing_is_zero(monkeypatch):
    monkeypatch.setattr(osproc, "_read_proc", lambda pid, name: None)
    assert osproc.read_pss_bytes(123) == 0


def test_tree_pids_collects_process_group_members(monkeypatch):
    monkeypatch.setattr(osproc, "_HAS_PROC", True)
    monkeypatch.setattr(osproc, "_iter_all_pids", lambda: iter([10, 11, 12, 13]))
    monkeypatch.setattr(osproc, "proc_pgid", lambda pid: 10 if pid != 13 else 99)
    monkeypatch.setattr(osproc, "pid_alive", lambda pid: True)

    # pid 10 itself has pgid 10; pid 13 belongs to an unrelated group
    assert osproc.tree_pids(10) == [10, 11, 12]


def test_tree_pids_falls_back_to_root_pid_off_proc(monkeypatch):
    monkeypatch.setattr(osproc, "_HAS_PROC", False)
    monkeypatch.setattr(osproc, "pid_alive", lambda pid: True)
    assert osproc.tree_pids(77) == [77]

    monkeypatch.setattr(osproc, "pid_alive", lambda pid: False)
    assert osproc.tree_pids(77) == []


def test_signal_group_uses_killpg_when_pid_is_the_group_leader(monkeypatch):
    sent = []

    def fake_killpg(pgid, sig):
        sent.append(("killpg", pgid, sig))

    monkeypatch.setattr(osproc, "proc_pgid", lambda pid: pid)
    monkeypatch.setattr(osproc.os, "killpg", fake_killpg)
    osproc.signal_group(42, 15)
    assert sent == [("killpg", 42, 15)]


def test_signal_group_falls_back_to_single_pid_when_groups_differ(monkeypatch):
    sent = []

    def fake_kill(pid, sig):
        sent.append(("kill", pid, sig))

    monkeypatch.setattr(osproc, "proc_pgid", lambda pid: pid + 1)
    monkeypatch.setattr(osproc.os, "kill", fake_kill)
    osproc.signal_group(42, 9)
    assert sent == [("kill", 42, 9)]


def test_argv_port_reads_all_spellings():
    assert osproc._argv_port(["--port", "8080"]) == 8080
    assert osproc._argv_port(["-p", "8081"]) == 8081
    assert osproc._argv_port(["--port=8082"]) == 8082
    assert osproc._argv_port(["serve", "--no-some-flag"]) == 1919
    assert osproc._argv_port(["--port", "not-a-number", "-p", "7"]) == 7


def test_set_oom_score_adj_reports_success_and_failure(monkeypatch):
    state = {"allow": False}

    class fake_open:
        def __init__(self, path, mode):
            if not state["allow"]:
                raise PermissionError

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def write(self, text):
            pass

    # set_oom_score_adj calls the builtin `open` name inside osproc
    monkeypatch.setattr(osproc, "open", lambda path, mode="r": fake_open(path, mode), raising=False)
    assert osproc.set_oom_score_adj(1, 100) is False
    state["allow"] = True
    assert osproc.set_oom_score_adj(1, 100) is True


def test_is_ft_serve_on_port_identity_checks(monkeypatch):
    monkeypatch.setattr(osproc, "_HAS_PROC", True)
    monkeypatch.setattr(osproc, "pid_alive", lambda pid: True)
    monkeypatch.setattr(osproc, "read_starttime", lambda pid: 424242)
    monkeypatch.setattr(
        osproc,
        "read_cmdline",
        lambda pid: ["/usr/bin/python", "-m", "freetoken.cli", "serve", "--port", "8080"],
    )

    assert osproc.is_ft_serve_on_port(9, 8080, starttime=424242) is True
    # reused pid: starttime differs -> refuse, whatever argv says
    assert osproc.is_ft_serve_on_port(9, 8080, starttime=111) is False
    # not our serve invocation at all
    monkeypatch.setattr(osproc, "read_cmdline", lambda pid: ["bash"])
    assert osproc.is_ft_serve_on_port(9, 8080) is False
    # same pid, new port -> not the serve we asked about
    monkeypatch.setattr(
        osproc, "read_cmdline", lambda pid: ["python", "-m", "freetoken.cli", "serve", "--port", "9090"]
    )
    assert osproc.is_ft_serve_on_port(9, 8080) is False