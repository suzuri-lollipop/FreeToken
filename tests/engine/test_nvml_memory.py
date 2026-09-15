"""NVML cross-check of the engine's free-memory view.

The CUDA allocator view (mem_get_info) is per-process-namespace: under WSL2 it misses
VRAM held by clients outside the namespace (e.g. a docker container on the Windows
host), so pool sizing is clamped to the whole-device NVML free. The clamp is a pure
function; the GPU assignment resolution is tested with the NVML layer mocked.
"""
import freetoken.gpu_select as gpu_select
from freetoken.engine.engine import _effective_free_bytes

GIB = 1 << 30


def test_effective_free_uses_nvml_when_cuda_sees_more():
    # the WSL2 case: CUDA sees 94 GiB free, the device physically has 11
    assert _effective_free_bytes(94 * GIB, 11 * GIB) == 11 * GIB


def test_effective_free_keeps_cuda_when_it_sees_less():
    assert _effective_free_bytes(8 * GIB, 90 * GIB) == 8 * GIB


def test_effective_free_falls_back_to_cuda_without_nvml():
    assert _effective_free_bytes(94 * GIB, None) == 94 * GIB


def _assign(monkeypatch, physical=None, visible=None):
    monkeypatch.setattr(gpu_select, "_assigned_physical", physical)
    monkeypatch.setattr(gpu_select, "_assigned_visible", visible)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def test_nvml_free_bytes_none_when_nvml_unavailable(monkeypatch):
    monkeypatch.setattr(gpu_select, "_nvml_memory", lambda: None)
    _assign(monkeypatch)
    assert gpu_select.nvml_free_bytes() is None


def test_nvml_free_bytes_resolves_by_uuid(monkeypatch):
    monkeypatch.setattr(gpu_select, "_nvml_memory", lambda: [(100, 40), (100, 70)])
    monkeypatch.setattr(gpu_select, "_nvml_uuids", lambda: ["GPU-aaa", "GPU-bbb"])
    _assign(monkeypatch, physical="GPU-bbb")
    assert gpu_select.nvml_free_bytes() == 70


def test_nvml_free_bytes_resolves_by_uuid_prefix(monkeypatch):
    monkeypatch.setattr(gpu_select, "_nvml_memory", lambda: [(100, 40), (100, 70)])
    monkeypatch.setattr(gpu_select, "_nvml_uuids", lambda: ["GPU-aaa", "GPU-bbb"])
    _assign(monkeypatch, physical="GPU-B")
    assert gpu_select.nvml_free_bytes() == 70


def test_nvml_free_bytes_unknown_uuid_is_none(monkeypatch):
    monkeypatch.setattr(gpu_select, "_nvml_memory", lambda: [(100, 40)])
    monkeypatch.setattr(gpu_select, "_nvml_uuids", lambda: ["GPU-aaa"])
    _assign(monkeypatch, physical="GPU-zzz")
    assert gpu_select.nvml_free_bytes() is None


def test_nvml_free_bytes_by_visible_ordinal(monkeypatch):
    monkeypatch.setattr(gpu_select, "_nvml_memory", lambda: [(100, 40), (100, 70)])
    _assign(monkeypatch, visible=1)
    assert gpu_select.nvml_free_bytes() == 70


def test_nvml_free_bytes_visible_ordinal_out_of_range_is_none(monkeypatch):
    monkeypatch.setattr(gpu_select, "_nvml_memory", lambda: [(100, 40)])
    _assign(monkeypatch, visible=1)
    assert gpu_select.nvml_free_bytes() is None


def test_nvml_free_bytes_defaults_to_physical_zero(monkeypatch):
    monkeypatch.setattr(gpu_select, "_nvml_memory", lambda: [(100, 40), (100, 70)])
    _assign(monkeypatch)
    assert gpu_select.nvml_free_bytes() == 40


def test_nvml_free_bytes_preset_index_reorders_visible_ordinal(monkeypatch):
    # a preset CUDA_VISIBLE_DEVICES reorders the visible ordinals: visible 0 = physical 1
    monkeypatch.setattr(gpu_select, "_nvml_memory", lambda: [(100, 40), (100, 70)])
    _assign(monkeypatch, visible=0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert gpu_select.nvml_free_bytes() == 70


def test_nvml_free_bytes_preset_uuid_entry(monkeypatch):
    monkeypatch.setattr(gpu_select, "_nvml_memory", lambda: [(100, 40), (100, 70)])
    monkeypatch.setattr(gpu_select, "_nvml_uuids", lambda: ["GPU-aaa", "GPU-bbb"])
    _assign(monkeypatch, visible=0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbb")
    assert gpu_select.nvml_free_bytes() == 70


def test_nvml_free_bytes_preset_out_of_range_is_none(monkeypatch):
    monkeypatch.setattr(gpu_select, "_nvml_memory", lambda: [(100, 40)])
    _assign(monkeypatch, visible=3)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert gpu_select.nvml_free_bytes() is None
