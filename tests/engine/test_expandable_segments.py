"""expandable_segments enablement: the VMM probe and its fallback.

On stacks whose CUDA does not implement the Virtual Memory Management API (notably
WSL2), the first expandable-segment allocation dies with an opaque cudaErrorUnknown.
The engine probes the path with a real allocation and re-sets the default allocator
when it fails. The allocator calls are monkeypatched so the logic runs without a GPU.
"""
import torch

from freetoken.engine.engine import _ensure_expandable_segments


def _clear_alloc_conf(monkeypatch):
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)


def test_user_allocator_config_is_respected(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", calls.append)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "expandable_segments:False")
    _ensure_expandable_segments(torch.device("cuda"))
    assert calls == []


def test_enables_and_keeps_when_the_probe_allocation_succeeds(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", calls.append)
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: None)
    _clear_alloc_conf(monkeypatch)
    _ensure_expandable_segments(torch.device("cuda"))
    assert calls == ["expandable_segments:True"]


def test_falls_back_to_the_default_allocator_when_the_probe_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", calls.append)

    def broken_empty(*a, **kw):
        raise RuntimeError("CUDA driver error: unknown error")

    monkeypatch.setattr(torch, "empty", broken_empty)
    _clear_alloc_conf(monkeypatch)
    _ensure_expandable_segments(torch.device("cuda"))
    assert calls == ["expandable_segments:True", "expandable_segments:False"]


def test_continues_without_probe_when_the_setting_api_is_missing(monkeypatch):
    def missing(*a, **kw):
        raise AttributeError("no such allocator API in this torch build")

    monkeypatch.setattr(torch.cuda.memory, "_set_allocator_settings", missing)
    _clear_alloc_conf(monkeypatch)
    _ensure_expandable_segments(torch.device("cuda"))  # must not raise
