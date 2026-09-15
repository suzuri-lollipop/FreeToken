"""Startup VRAM gate: refuse to materialize weights when the shared device is short.

On a multi-client GPU (WSL2), driving the device past its physical limit does not just
fail our allocation - it wedged the driver's GPU sync path (dxgkrnl) and took a vLLM
co-tenant down with us. The gate compares the exact dense footprint (meta state dict,
zero allocations) plus margin against the whole-device (NVML-clamped) free memory
before any weight is materialized. Pure functions, no GPU needed.
"""
import pytest
import torch

from freetoken.engine.engine import (
    _VRAM_MARGIN_ENV,
    _VRAM_OVERCOMMIT_ENV,
    _dense_weights_bytes,
    _vram_margin_gib,
    _vram_overcommit_allowed,
    _vram_startup_refusal,
)

GiB = 1 << 30


class _MetaModel:
    def __init__(self, tensors):
        self._tensors = tensors

    def state_dict(self):
        return {f"p{i}": t for i, t in enumerate(self._tensors)}


# --------------------------------------------------------------------------- refusal

def test_fits_returns_none(monkeypatch):
    monkeypatch.delenv(_VRAM_MARGIN_ENV, raising=False)
    assert _vram_startup_refusal(dense_bytes=9 * GiB, free_bytes=12 * GiB, margin_gib=2.0) is None


def test_exactly_at_the_boundary_still_starts(monkeypatch):
    assert _vram_startup_refusal(dense_bytes=9 * GiB, free_bytes=11 * GiB, margin_gib=2.0) is None


def test_short_free_memory_refuses_with_an_actionable_message(monkeypatch):
    msg = _vram_startup_refusal(dense_bytes=9 * GiB, free_bytes=10 * GiB, margin_gib=2.0)
    assert msg is not None
    assert "refusing to start" in msg
    # the three numbers a triage reader needs: dense, need, free
    assert "9.00 GiB" in msg
    assert "11.00 GiB" in msg
    assert "10.00 GiB" in msg
    assert _VRAM_OVERCOMMIT_ENV in msg
    assert _VRAM_MARGIN_ENV in msg


def test_zero_margin_only_refuses_when_the_weights_do_not_fit(monkeypatch):
    assert _vram_startup_refusal(dense_bytes=9 * GiB, free_bytes=9 * GiB + 1, margin_gib=0.0) is None
    assert _vram_startup_refusal(dense_bytes=9 * GiB, free_bytes=9 * GiB - 1, margin_gib=0.0) is not None


# ---------------------------------------------------------- dense footprint (meta)

def test_dense_weights_bytes_sums_mixed_dtypes_exactly(monkeypatch):
    model = _MetaModel(
        [
            torch.empty(4 * GiB // 2, dtype=torch.bfloat16, device="meta"),
            torch.empty(1 * GiB, dtype=torch.float8_e4m3fn, device="meta"),
            torch.empty(3, dtype=torch.float32, device="meta"),
        ]
    )
    assert _dense_weights_bytes(model) == 4 * GiB + 1 * GiB + 12


# ------------------------------------------------------------- env var handling

def test_margin_defaults_to_two_gib(monkeypatch):
    monkeypatch.delenv(_VRAM_MARGIN_ENV, raising=False)
    assert _vram_margin_gib() == 2.0


def test_margin_env_override(monkeypatch):
    monkeypatch.setenv(_VRAM_MARGIN_ENV, "0.5")
    assert _vram_margin_gib() == 0.5
    monkeypatch.setenv(_VRAM_MARGIN_ENV, " 3 ")
    assert _vram_margin_gib() == 3.0


def test_margin_env_rejects_garbage(monkeypatch):
    monkeypatch.setenv(_VRAM_MARGIN_ENV, "two")
    with pytest.raises(ValueError, match="FREETOKEN_VRAM_START_MARGIN_GIB"):
        _vram_margin_gib()


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", " ON ", "True"])
def test_overcommit_truthy(monkeypatch, value):
    monkeypatch.setenv(_VRAM_OVERCOMMIT_ENV, value)
    assert _vram_overcommit_allowed()


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off"])
def test_overcommit_falsy(monkeypatch, value):
    monkeypatch.setenv(_VRAM_OVERCOMMIT_ENV, value)
    assert not _vram_overcommit_allowed()


def test_overcommit_unset(monkeypatch):
    monkeypatch.delenv(_VRAM_OVERCOMMIT_ENV, raising=False)
    assert not _vram_overcommit_allowed()


# --------------------------------------------------- refused slot cache (post-weight OOM)

def test_slot_cache_oom_note_names_the_lying_reading_and_the_knobs():
    from types import SimpleNamespace

    from freetoken.engine.engine import _slot_cache_oom_note

    msg = _slot_cache_oom_note(
        SimpleNamespace(moe_cache_size=24576, memory_ratio=0.9),
        plan_bytes=66 * GiB,
        baseline_free=93 * GiB,
        cuda_free=84 * GiB,
        detail="CUDA out of memory. Tried to allocate 37.50 GiB.",
    )
    assert "66.00 GiB" in msg and "24576 slots" in msg
    assert "93.00 GiB" in msg  # what the sizing trusted
    assert "84.00 GiB" in msg  # what the device still claimed when it refused
    assert "WSL2" in msg
    for flag in ("--moe-cache-size", "--moe-cache-rate", "--memory-ratio", "--num-tokens"):
        assert flag in msg
    assert "Tried to allocate 37.50 GiB." in msg  # the driver's own words survive
