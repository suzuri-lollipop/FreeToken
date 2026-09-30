"""expandable_segments enablement: the VMM probe and its fallback.

On stacks whose CUDA does not implement the Virtual Memory Management API (notably WSL2), the
first expandable-segment allocation dies with an opaque cudaErrorUnknown. The engine probes the
path with a real allocation and re-sets the default allocator when it fails. A torch build
compiled without the CUDA driver API never gets that far: it takes the setting, warns that the
platform does not support it, and keeps the default allocator. The allocator calls are
monkeypatched so the logic runs without a GPU.
"""
import warnings

import torch

from freetoken.engine import engine as engine_mod
from freetoken.engine.engine import _ensure_expandable_segments


def _clear_alloc_conf(monkeypatch):
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)


def test_user_allocator_config_is_respected(monkeypatch):
    calls = []
    monkeypatch.setattr(engine_mod, "_set_allocator_settings", calls.append)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:False")
    _ensure_expandable_segments(torch.device("cuda"))
    assert calls == []


def test_enables_and_keeps_when_the_probe_allocation_succeeds(monkeypatch):
    calls = []
    monkeypatch.setattr(engine_mod, "_set_allocator_settings", calls.append)
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: None)
    _clear_alloc_conf(monkeypatch)
    _ensure_expandable_segments(torch.device("cuda"))
    assert calls == ["expandable_segments:True"]


def test_falls_back_to_the_default_allocator_when_the_probe_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(engine_mod, "_set_allocator_settings", calls.append)

    def broken_empty(*a, **kw):
        raise RuntimeError("CUDA driver error: unknown error")

    monkeypatch.setattr(torch, "empty", broken_empty)
    _clear_alloc_conf(monkeypatch)
    _ensure_expandable_segments(torch.device("cuda"))
    assert calls == ["expandable_segments:True", "expandable_segments:False"]


def test_continues_without_probe_when_the_setting_api_is_missing(monkeypatch):
    def missing(*a, **kw):
        raise AttributeError("no such allocator API in this torch build")

    monkeypatch.setattr(engine_mod, "_set_allocator_settings", missing)
    _clear_alloc_conf(monkeypatch)
    _ensure_expandable_segments(torch.device("cuda"))  # must not raise


def test_undoes_the_setting_when_the_build_does_not_support_it(monkeypatch):
    """The setting is accepted and ignored by a driver-API-less build, which only says so in a
    warning -- logging 'enabled' there is a lie, and the undo keeps the default allocator."""
    calls = []
    monkeypatch.setattr(engine_mod, "_set_allocator_settings", calls.append)

    def unsupported_empty(*a, **kw):
        warnings.warn("expandable_segments not supported on this platform", UserWarning)

    monkeypatch.setattr(torch, "empty", unsupported_empty)
    _clear_alloc_conf(monkeypatch)
    _ensure_expandable_segments(torch.device("cuda"))
    assert calls == ["expandable_segments:True", "expandable_segments:False"]


def test_an_unrelated_probe_warning_is_not_swallowed(monkeypatch):
    calls = []
    monkeypatch.setattr(engine_mod, "_set_allocator_settings", calls.append)

    def noisy_empty(*a, **kw):
        warnings.warn("something else entirely", UserWarning)

    monkeypatch.setattr(torch, "empty", noisy_empty)
    _clear_alloc_conf(monkeypatch)
    with warnings.catch_warnings(record=True) as out:
        warnings.simplefilter("always")
        _ensure_expandable_segments(torch.device("cuda"))
    assert any("something else entirely" in str(w.message) for w in out)
