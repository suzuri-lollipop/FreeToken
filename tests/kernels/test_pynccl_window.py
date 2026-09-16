"""The PyNCCL symmetric-window switch (kernel/pynccl.py).

The window is registered over the same cuMem staging buffer the collective runs on, and that
combination never completes its first all-reduce on some two-GPU rigs; it stays off by default.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    from freetoken.env import ENV

    monkeypatch.delenv("NCCL_WIN_ENABLE", raising=False)
    knob = ENV.PYNCCL_SYMMETRIC_WINDOW
    saved = knob.value
    yield
    knob.value = saved


def test_the_window_is_off_by_default():
    from freetoken.kernel.pynccl import configure_pynccl_window

    assert configure_pynccl_window() is False
    import os

    assert os.environ["NCCL_WIN_ENABLE"] == "0"


def test_an_explicit_nccl_win_enable_is_not_overwritten():
    import os

    os.environ["NCCL_WIN_ENABLE"] = "1"
    from freetoken.kernel.pynccl import configure_pynccl_window

    assert configure_pynccl_window() is True
    assert os.environ["NCCL_WIN_ENABLE"] == "1"


def test_the_knob_turns_the_window_on():
    from freetoken.env import ENV
    from freetoken.kernel.pynccl import configure_pynccl_window

    ENV.PYNCCL_SYMMETRIC_WINDOW.value = True
    assert configure_pynccl_window() is True
    import os

    assert os.environ["NCCL_WIN_ENABLE"] == "1"
