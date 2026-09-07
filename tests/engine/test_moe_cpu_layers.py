"""Resolver for the hybrid CPU/GPU MoE decode split (--moe-cpu-layers).

CPU-only: exercises _parse_cpu_layers_spec / _resolve_cpu_layers / _auto_cpu_layers without
a GPU. The auto split is sized against a pin allowance the caller measured, so it is pure
arithmetic here.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from freetoken.engine import engine
from freetoken.engine.engine import _auto_cpu_layers as auto
from freetoken.engine.engine import _parse_cpu_layers_spec as parse
from freetoken.engine.engine import _resolve_cpu_layers as resolve
from freetoken.moe import expert_banks

L = 40


def test_explicit_list():
    assert parse("3,7,11", L) == frozenset({3, 7, 11})
    assert parse("3, 7 ,11,", L) == frozenset({3, 7, 11})  # whitespace + trailing comma
    assert parse("5,5,5", L) == frozenset({5})  # dups collapse


def test_count_evenly_strided():
    assert parse("8", L) == frozenset({0, 5, 10, 15, 20, 25, 30, 35})
    assert parse("1", L) == frozenset({0})
    assert len(parse(str(L), L)) == L  # all layers
    assert parse("0", L) == frozenset()


def test_fraction():
    assert len(parse("0.5", L)) == L // 2
    assert len(parse("1.0", L)) == L
    assert parse("0.0", L) == frozenset()


def test_empty():
    assert parse("", L) == frozenset()
    assert parse("   ", L) == frozenset()


@pytest.mark.parametrize("spec", ["99", "40,1", "-1", "1.5"])
def test_out_of_range_raises(spec):
    with pytest.raises(ValueError):
        parse(spec, L)


def _cfg(backend, spec=None):
    return SimpleNamespace(moe_backend=backend, moe_cpu_layers=spec)


def test_resolve_backend_dispatch():
    # cpu backend -> every layer, ignoring any spec
    assert resolve(_cfg("cpu"), L) == frozenset(range(L))
    assert resolve(_cfg("cpu", "8"), L) == frozenset(range(L))
    # offload + spec -> parsed subset
    assert len(resolve(_cfg("offload", "8"), L)) == 8
    # offload, no spec -> none (plain offload)
    assert resolve(_cfg("offload", None), L) == frozenset()
    # non-offload backend ignores the spec (validation lives in _adjust_config)
    assert resolve(_cfg("fused", "8"), L) == frozenset()


# ---- _auto_cpu_layers: sizing the split against the measured pin allowance ----

GiB = 2**30
LAYERS = 48


def _stub_banks(monkeypatch, total_bytes):
    """_auto_cpu_layers reads the bank size through these two, then gates on the executor."""
    monkeypatch.setattr(expert_banks, "ftw_bank_bytes", lambda path: total_bytes)
    monkeypatch.setattr(expert_banks, "bank_bytes_estimate", lambda cfg: total_bytes)
    monkeypatch.setattr(engine, "_cpu_moe_executor_viable", lambda cfg: True)


def _auto_cfg():
    return SimpleNamespace(model_path="/ckpt", model_config=SimpleNamespace())


def test_auto_split_pins_only_what_the_allowance_fits(monkeypatch):
    _stub_banks(monkeypatch, 63 * GiB)
    ids = auto(_auto_cfg(), LAYERS, 35 * GiB)
    # head+tail: the U-shaped per-layer miss rate makes the ends cheapest to move off GPU
    assert ids == frozenset(range(11)) | frozenset(range(37, LAYERS))
    pinned = 63 * GiB - len(ids) * 63 * GiB // LAYERS
    assert pinned <= 35 * GiB


def test_auto_split_is_machine_wide_not_per_rank(monkeypatch):
    """63 GiB at tp=2 is 31.5 GiB/rank, which fits a 35 GiB allowance per rank -- and both
    ranks pin on the SAME host, so 63 GiB machine-wide is what has to fit."""
    _stub_banks(monkeypatch, 63 * GiB)
    assert len(auto(_auto_cfg(), LAYERS, 35 * GiB)) == 22
    assert auto(_auto_cfg(), LAYERS, 63 * GiB) == frozenset()


def test_auto_split_without_a_cap_keeps_every_layer_pinned(monkeypatch):
    _stub_banks(monkeypatch, 63 * GiB)
    assert auto(_auto_cfg(), LAYERS, None) == frozenset()


def test_auto_split_yields_when_the_cpu_executor_cannot_serve(monkeypatch):
    _stub_banks(monkeypatch, 63 * GiB)
    monkeypatch.setattr(engine, "_cpu_moe_executor_viable", lambda cfg: False)
    assert auto(_auto_cfg(), LAYERS, 35 * GiB) == frozenset()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
