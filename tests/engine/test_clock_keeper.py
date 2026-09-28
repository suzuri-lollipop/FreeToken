"""Idle clock keeper config and lifecycle (engine/clock_keeper.py).

The env parse decides whether the anti-park pulser runs at all, and the
lifecycle guards decide how many threads it owns. Both fail silently - a bad
parse just disables the keepalive, a leaked thread just spins forever - so
these are pinned on the plain host logic; no GPU matmul is ever launched.
"""

from __future__ import annotations

import threading

import pytest
import torch

from freetoken.engine import clock_keeper
from freetoken.engine.clock_keeper import ClockKeeper, _ENV


def test_default_interval_is_zero(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert clock_keeper.keepalive_interval_ms() == 0


def test_interval_parses_int_with_whitespace(monkeypatch):
    monkeypatch.setenv(_ENV, " 500 ")
    assert clock_keeper.keepalive_interval_ms() == 500


def test_negative_interval_clamps_to_zero(monkeypatch):
    monkeypatch.setenv(_ENV, "-20")
    assert clock_keeper.keepalive_interval_ms() == 0


def test_invalid_interval_raises(monkeypatch):
    monkeypatch.setenv(_ENV, "fast")
    with pytest.raises(ValueError, match="must be an integer"):
        clock_keeper.keepalive_interval_ms()


def test_start_spawns_exactly_one_thread_and_stop_clears_it(monkeypatch):
    started = threading.Event()
    monkeypatch.setattr(ClockKeeper, "_run", lambda self: started.set())

    keeper = ClockKeeper(torch.device("cpu"), interval_ms=100)
    keeper.start()  # never double-spawns even when called repeatedly
    keeper.start()
    assert keeper._thread is not None
    assert started.wait(timeout=5)
    keeper.stop()
    assert keeper._thread is None


def test_zero_interval_never_starts(monkeypatch):
    started = threading.Event()
    monkeypatch.setattr(ClockKeeper, "_run", lambda self: started.set())

    keeper = ClockKeeper(torch.device("cpu"), interval_ms=0)
    keeper.start()
    assert keeper._thread is None
    assert not started.is_set()


def test_stop_without_start_is_a_noop():
    ClockKeeper(torch.device("cpu"), interval_ms=100).stop()