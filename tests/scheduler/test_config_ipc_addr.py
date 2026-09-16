"""zmq ipc endpoint construction (torch-free part of SchedulerConfig)."""

from __future__ import annotations

import os

from freetoken.scheduler.config import zmq_ipc_endpoint


def test_posix_addr_is_tmp():
    assert zmq_ipc_endpoint(0, ".pid=42") == "ipc:///tmp/freetoken_0.pid=42"


def test_slots_stay_distinct():
    a = {zmq_ipc_endpoint(i, ".pid=1") for i in range(5)}
    assert len(a) == 5


def test_windows_addr_uses_tempdir(monkeypatch):
    monkeypatch.setattr(os, "name", "nt", raising=False)
    import tempfile

    import freetoken.scheduler.config as cfg

    monkeypatch.setattr(tempfile, "gettempdir", lambda: "C:\\Users\\me\\AppData\\Local\\Temp")
    addr = cfg.zmq_ipc_endpoint(2, ".pid=7")
    assert addr == "ipc://C:\\Users\\me\\AppData\\Local\\Temp/freetoken_2.pid=7"
    assert "/tmp" not in addr
