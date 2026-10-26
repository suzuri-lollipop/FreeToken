"""Resolver for the hybrid CPU/GPU MoE decode split (--moe-cpu-layers).

CPU-only: exercises _parse_cpu_layers_spec / _resolve_cpu_layers without a GPU, and
the disk-tier split resolver (_resolve_disk_layers) with patched size/budget seams.
"""

from __future__ import annotations

from types import SimpleNamespace

import importlib

import pytest

import freetoken.engine.engine as engine_mod
from freetoken.engine.engine import _check_pin_budget as check_pin_budget
from freetoken.engine.engine import _parse_cpu_layers_spec as parse
from freetoken.engine.engine import _resolve_cpu_layers as resolve
from freetoken.engine.engine import _resolve_disk_layers as resolve_disk

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
    return SimpleNamespace(moe_strategy=backend, moe_cpu_layers=spec)


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


# --- disk tier (--moe-disk-tier) ----------------------------------------------------


def _disk_cfg(mode):
    return SimpleNamespace(
        moe_strategy="offload", moe_cpu_layers=None, moe_disk_tier=mode,
        model_path="/nonexistent",
    )


def _sizes(monkeypatch, bank_bytes, budget):
    monkeypatch.setattr(engine_mod, "_bank_bytes", lambda config, method=None: bank_bytes)
    monkeypatch.setattr(
        engine_mod, "_disk_ram_budget", lambda reserved=0, *, config=None, size=1: budget
    )


def test_disk_tier_off_never_spills(monkeypatch):
    _sizes(monkeypatch, 100, 10)
    assert resolve_disk(_disk_cfg("off"), L) == frozenset()


def test_disk_tier_all(monkeypatch):
    _sizes(monkeypatch, 100, 90)
    assert resolve_disk(_disk_cfg("all"), L) == frozenset(range(L))


def test_disk_tier_auto_fits_ram(monkeypatch):
    _sizes(monkeypatch, 100, 200)
    assert resolve_disk(_disk_cfg("auto"), L) == frozenset()


def test_disk_tier_auto_unmeasurable_never_spills(monkeypatch):
    # no readable RAM budget (MemAvailable hidden, no pin quota): stay all-pinned
    _sizes(monkeypatch, 100, None)
    assert resolve_disk(_disk_cfg("auto"), L) == frozenset()


def test_disk_tier_auto_spills_head_and_tail(monkeypatch):
    _sizes(monkeypatch, 100, 50)  # ceil(40 * 0.5) = 20 layers, head 10 tail 10
    ids = resolve_disk(_disk_cfg("auto"), L)
    assert len(ids) == 20
    assert ids == frozenset(range(10)) | frozenset(range(30, 40))  # contiguous head+tail
    assert 10 not in ids and 29 not in ids  # the pinned block in the middle


def test_disk_tier_auto_spill_capped_at_layers(monkeypatch):
    _sizes(monkeypatch, 100, 0)  # budget 0: everything spills but stays in range
    ids = resolve_disk(_disk_cfg("auto"), L)
    assert ids == frozenset(range(L))


def test_disk_tier_no_bank_bytes_never_spills(monkeypatch):
    _sizes(monkeypatch, None, 10)
    assert resolve_disk(_disk_cfg("auto"), L) == frozenset()


def test_check_pin_budget_skips_when_disk_absorbs(monkeypatch):
    """A plain offload boot whose banks exceed the pin budget normally stops before
    reading anything; with the disk tier engaged it must be allowed to start."""
    monkeypatch.setattr(engine_mod, "_bank_bytes", lambda config, method=None, layer_methods=None: 100)
    monkeypatch.setattr(engine_mod, "_pin_budget_bytes", lambda reserved=0: 50)
    cfg = _disk_cfg("auto")
    with pytest.raises(ValueError, match="pin budget"):
        check_pin_budget(cfg, reserved=0)
    check_pin_budget(cfg, reserved=0, disk_layer_ids=frozenset(range(4)))  # no raise


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))


def test_resolve_hot_experts_from_profile(tmp_path, monkeypatch):
    """With --moe-bank-profile, the hot resolver takes the DISK layers' pairs in
    rank order while the hot budget lasts, skips pinned layers, and rejects a
    profile whose dimensions do not match the model."""
    from freetoken.moe.expert_profile import rank_by_counts, write_profile

    engine_mod = importlib.import_module("freetoken.engine.engine")
    # 0.5 MiB per expert (16 MiB / 32 pairs) -> 2.5 MiB hot budget = 5 experts
    _sizes(monkeypatch, bank_bytes=16 << 20, budget=10 << 20)
    monkeypatch.setattr(
        engine_mod, "_disk_ram_budget", lambda reserved=0, *, config=None, size=1: 10 << 20
    )

    counts = [[0] * 8 for _ in range(4)]
    counts[3][5] = 9   # hottest, but layer 3 stays pinned -> skipped
    counts[0][6] = 7   # disk layer 0, taken first
    counts[1][2] = 5   # disk layer 1, taken second
    counts[1][4] = 1   # disk layer 1, third by count (budget allows)
    ranked = rank_by_counts(counts)
    path = tmp_path / "p.bin"
    write_profile(path, ranked, num_layers=4, num_experts=8)

    cfg = SimpleNamespace(
        moe_bank_profile=str(path),
        moe_hot_frac=0.25,
        model_config=SimpleNamespace(num_experts=8),
        moe_disk_tier="auto",
        moe_strategy="offload",
    )
    disk = frozenset({0, 1})
    hot = engine_mod._resolve_hot_experts(cfg, disk, 4, method=None)
    # counted pairs first (pinned layer 3 skipped), then the round-robin tail
    assert hot == {0: [6, 0, 1], 1: [2, 4]}

    # no disk layers -> no hot; no profile -> None
    assert engine_mod._resolve_hot_experts(cfg, frozenset(), 4, method=None) is None
    cfg_none = SimpleNamespace(
        moe_bank_profile=None, moe_hot_frac=0.25,
        model_config=SimpleNamespace(num_experts=8),
    )
    assert engine_mod._resolve_hot_experts(cfg_none, disk, 4, method=None) is None

    # mismatched profile dims are a hard error (wrong experts would pin silently)
    bad = tmp_path / "bad.bin"
    write_profile(bad, ranked, num_layers=5, num_experts=8)
    cfg_bad = SimpleNamespace(
        moe_bank_profile=str(bad), moe_hot_frac=0.25,
        model_config=SimpleNamespace(num_experts=8),
    )
    with pytest.raises(ValueError, match="ranks 5 layers"):
        engine_mod._resolve_hot_experts(cfg_bad, disk, 4, method=None)


def test_resolve_disk_layers_budget_shrinks_with_profile(tmp_path, monkeypatch):
    """A profile spends hot_frac of the RAM budget on hot experts, so fewer whole
    layers pin than without one (the disk side grows by that slice)."""
    from freetoken.moe.expert_profile import round_robin, write_profile

    engine_mod = importlib.import_module("freetoken.engine.engine")
    L = 10
    _sizes(monkeypatch, bank_bytes=100 << 20, budget=50 << 20)
    base = SimpleNamespace(
        moe_disk_tier="auto", moe_strategy="offload", moe_bank_profile=None,
        model_config=SimpleNamespace(num_experts=8), moe_hot_frac=0.25,
    )
    plain = engine_mod._resolve_disk_layers(base, L, method=None)
    assert len(plain) == 5  # 100 MiB banks, 50 MiB budget: half the layers spill

    path = tmp_path / "p.bin"
    write_profile(path, round_robin(L, 8), num_layers=L, num_experts=8)
    prof = SimpleNamespace(
        moe_disk_tier="auto", moe_strategy="offload", moe_bank_profile=str(path),
        model_config=SimpleNamespace(num_experts=8), moe_hot_frac=0.25,
    )
    hot = engine_mod._resolve_disk_layers(prof, L, method=None)
    # hot budget 12.5 MiB leaves 37.5 for whole layers -> 7 layers spill
    assert len(hot) == 7
    assert hot > plain


def test_ram_budget_cap_replaces_heuristic(monkeypatch):
    """An explicit --moe-ram-budget is the pin ceiling verbatim (no headroom re-subtracted)."""
    from freetoken.moe import host_banks as _hb

    monkeypatch.setattr(_hb, "mem_available_bytes", lambda: 100 << 30)
    monkeypatch.setattr(engine_mod, "_host_ram_bytes", lambda: 121 << 30)
    monkeypatch.setattr(engine_mod, "_pin_budget_bytes", lambda reserved=0: None)
    cfg = SimpleNamespace(moe_ram_budget=64.0)
    assert engine_mod._disk_ram_budget(config=cfg) == 64 << 30


def test_ram_budget_auto_stays_headroom_sized(monkeypatch):
    """The auto path keeps the MemAvailable-minus-headroom heuristic at TP=1."""
    from freetoken.moe import host_banks as _hb

    monkeypatch.setattr(_hb, "mem_available_bytes", lambda: 100 << 30)
    monkeypatch.setattr(engine_mod, "_host_ram_bytes", lambda: 121 << 30)
    monkeypatch.setattr(engine_mod, "_pin_budget_bytes", lambda reserved=0: None)
    assert engine_mod._disk_ram_budget() == (100 << 30) - (121 << 30) // 10


def test_ram_budget_divides_across_tp_ranks(monkeypatch):
    """MemAvailable and the explicit cap are machine-wide: each rank takes 1/size.
    The auto path also reserves the headroom PER RANK (not once machine-wide)."""
    from freetoken.moe import host_banks as _hb

    monkeypatch.setattr(_hb, "mem_available_bytes", lambda: 100 << 30)
    monkeypatch.setattr(engine_mod, "_host_ram_bytes", lambda: 121 << 30)
    monkeypatch.setattr(engine_mod, "_pin_budget_bytes", lambda reserved=0: None)
    assert engine_mod._disk_ram_budget(size=2) == (100 << 30) // 2 - (121 << 30) // 10
    cfg = SimpleNamespace(moe_ram_budget=80.0)
    assert engine_mod._disk_ram_budget(config=cfg, size=2) == 40 << 30
