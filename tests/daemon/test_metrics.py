"""Engine footprint plumbing (daemon/metrics.py): VRAM attribution and the TTL cache.

Footprint numbers feed the daemon's status view; a wrong MiB->byte factor or a
pid filter that leaks another process's VRAM looks like a plausible number
forever. The NVML/SMI probes and PSS reads are faked at the module boundary so
the tests run anywhere and only pin the attribution arithmetic.
"""

from __future__ import annotations

import subprocess

import pytest

import freetoken.daemon.metrics as metrics


@pytest.fixture(autouse=True)
def _fresh_nvml(monkeypatch):
    metrics._NVML["ready"] = None
    monkeypatch.delitem(__import__("sys").modules, "pynvml", raising=False)


def _proc(pid, used):
    from types import SimpleNamespace

    return SimpleNamespace(pid=pid, usedGpuMemory=used)


def _install_pynvml(monkeypatch, n_devices, v3, fallback):
    """Register a fake pynvml: ``v3``/``fallback`` are callables(handle) ->
    [proc] that may raise; a None callable means the attribute is absent, as on
    pynvml versions that predate the v3 getter."""
    import sys
    from types import SimpleNamespace

    attrs = {
        "nvmlInit": lambda: None,
        "nvmlDeviceGetCount": lambda: n_devices,
        "nvmlDeviceGetHandleByIndex": lambda i: i,
    }
    if v3 is not None:
        attrs["nvmlDeviceGetComputeRunningProcesses_v3"] = v3
    if fallback is not None:
        attrs["nvmlDeviceGetComputeRunningProcesses"] = fallback
    sys.modules["pynvml"] = SimpleNamespace(**attrs)


def test_nvml_process_vram_sums_across_devices(monkeypatch):
    def v3(handle):
        return [_proc(100, 5_000_000)] if handle == 0 else [_proc(100, 3_000_000), _proc(200, 7_000_000)]

    _install_pynvml(monkeypatch, 2, v3, None)
    assert metrics._nvml_process_vram() == {100: 8_000_000, 200: 7_000_000}


def test_nvml_process_vram_falls_back_when_v3_raises(monkeypatch):
    def broken(handle):
        raise RuntimeError("MIG quirk")

    def v2(handle):
        return [_proc(9, 1_000_000)]

    _install_pynvml(monkeypatch, 1, broken, v2)
    assert metrics._nvml_process_vram() == {9: 1_000_000}


def test_nvml_process_vram_falls_back_when_v3_is_absent(monkeypatch):
    def v2(handle):
        return [_proc(9, 2_000_000)]

    _install_pynvml(monkeypatch, 1, None, v2)
    assert metrics._nvml_process_vram() == {9: 2_000_000}


def test_nvml_process_vram_returns_none_when_every_probe_fails(monkeypatch):
    def broken(handle):
        raise RuntimeError("driver mismatch")

    _install_pynvml(monkeypatch, 2, broken, broken)
    # None (not {}) tells the caller the smi fallback must still run
    assert metrics._nvml_process_vram() is None


def test_nvml_unavailable_leaves_ready_false(monkeypatch):
    def init():
        raise RuntimeError("no nvml")

    import sys
    from types import SimpleNamespace

    sys.modules["pynvml"] = SimpleNamespace(nvmlInit=init)
    assert metrics._nvml_process_vram() is None
    assert metrics._NVML["ready"] is False


def make_smi_run(stdout: str, returncode: int = 0):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout)

    return fake_run


def test_vram_bytes_only_counts_requested_pids(monkeypatch):
    monkeypatch.setattr(metrics, "_nvml_process_vram", lambda: None)
    monkeypatch.setattr(metrics, "_smi_process_vram", lambda: {100: 5, 200: 7, 300: 11})
    # the probe sees pids the caller does not care about
    assert metrics.vram_bytes_for_pids([200, 300]) == 18
    assert metrics.vram_bytes_for_pids([100]) == 5


def test_vram_is_zero_for_no_pids_without_probing(monkeypatch):
    def explode():
        raise AssertionError("probe must not run for an empty pid set")

    monkeypatch.setattr(metrics, "_nvml_process_vram", explode)
    assert metrics.vram_bytes_for_pids([]) == 0


def test_vram_falls_back_to_smi_when_nvml_gives_nothing(monkeypatch):
    monkeypatch.setattr(metrics, "_nvml_process_vram", lambda: None)
    monkeypatch.setattr(metrics, "_smi_process_vram", lambda: {7: 3})
    assert metrics.vram_bytes_for_pids([7]) == 3


def test_smi_parser_converts_mib_and_sums_duplicate_pids(monkeypatch):
    stdout = "101, 512\n102, 256\n101, 512\nbogus line\n"
    monkeypatch.setattr(metrics.subprocess, "run", make_smi_run(stdout))
    assert metrics._smi_process_vram() == {
        101: 512 * 1024 * 1024 * 2,
        102: 256 * 1024 * 1024,
    }


def test_smi_parser_returns_empty_on_failure(monkeypatch):
    monkeypatch.setattr(metrics.subprocess, "run", make_smi_run("", returncode=1))
    assert metrics._smi_process_vram() == {}


def test_smi_parser_returns_empty_when_smi_is_missing(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(metrics.subprocess, "run", missing)
    assert metrics._smi_process_vram() == {}


def test_footprint_cache_serves_ttl_and_refreshes(monkeypatch):
    calls = {"n": 0}

    def fake_footprint(pid):
        calls["n"] += 1
        return {"ramBytes": pid or 0, "vramBytes": 0, "pids": [pid] if pid else []}

    monkeypatch.setattr(metrics, "engine_footprint", fake_footprint)
    clock = {"t": 0.0}
    cache = metrics.FootprintCache(ttl_s=2.0, now=lambda: clock["t"])

    assert cache.get(42) == {"ramBytes": 42, "vramBytes": 0, "pids": [42]}
    clock["t"] = 1.0  # within TTL: cached
    assert cache.get(42)["ramBytes"] == 42
    assert calls["n"] == 1

    clock["t"] = 2.5  # past TTL: probes again
    cache.get(42)
    assert calls["n"] == 2


def test_footprint_cache_keeps_none_pid_separate(monkeypatch):
    calls = {"n": 0}

    def fake_footprint(pid):
        calls["n"] += 1
        return {"ramBytes": 0, "vramBytes": 0, "pids": []}

    monkeypatch.setattr(metrics, "engine_footprint", fake_footprint)
    cache = metrics.FootprintCache(ttl_s=2.0, now=lambda: 0.0)

    assert cache.get(None) == {"ramBytes": 0, "vramBytes": 0, "pids": []}
    assert cache.get(None)["ramBytes"] == 0  # cached
    assert calls["n"] == 1


def test_engine_footprint_with_none_pid_is_all_zero():
    assert metrics.engine_footprint(None) == {"ramBytes": 0, "vramBytes": 0, "pids": []}


def test_engine_footprint_sums_pss_across_the_process_tree(monkeypatch):
    monkeypatch.setattr(metrics.osproc, "tree_pids", lambda pid: [pid, pid + 1])
    monkeypatch.setattr(metrics.osproc, "read_pss_bytes", lambda p: p * 1000)
    monkeypatch.setattr(metrics, "vram_bytes_for_pids", lambda pids: 77)

    out = metrics.engine_footprint(10)
    assert out == {"ramBytes": 10_000 + 11_000, "vramBytes": 77, "pids": [10, 11]}