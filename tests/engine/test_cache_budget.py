from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import os

from freetoken.engine.cache_budget import (
    cpu_layer_count,
    expert_bytes_per_slot,
    host_memory_budget_bytes,
    host_pin_allowance,
    plan_cache_budget,
    plan_pageout,
    resolve_moe_cache_auto,
    watchdog_nudge_bytes,
)
from freetoken.engine.engine import _pin_budget_bytes


def test_moe_priority_fills_experts_up_to_total():
    # budget large enough to cache every expert; KV gets the remainder.
    # per_expert=100, cache_per_page=10, total=8 experts (L*E), E=4.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=2000, per_expert_bytes=100, cache_per_page=10,
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_pages=5, max_slots=8,
    )
    assert size == 8  # capped at full residency
    assert pages == (2000 - 8 * 100) // 10  # == 120, remainder to KV
    assert overlap is True


def test_offload_case_experts_take_most_kv_gets_reserve_floor():
    # budget too small for full residency: experts take what they can, KV keeps its floor.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=1000, per_expert_bytes=100, cache_per_page=10,
        num_experts=2, total_experts=50, prefill_overlap=True,
        kv_reserve_pages=10, max_slots=50,
    )
    # raw = (1000 - 10*10) // 100 = 9 ; clamped to [4, 50] -> 9
    assert size == 9
    assert pages == max((1000 - 9 * 100) // 10, 10)  # remainder 10 pages, == floor
    assert overlap is True


def test_marlin_cap_clamps_count_and_rolls_bytes_to_kv():
    # budget would fund 1500 experts, but marlin caps at 992; freed bytes become KV pages.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=200_000, per_expert_bytes=100, cache_per_page=10,
        num_experts=128, total_experts=4000, prefill_overlap=True,
        kv_reserve_pages=0, max_slots=992,
    )
    assert size == 992
    assert pages == (200_000 - 992 * 100) // 10


def test_small_cache_disables_prefill_overlap():
    # cap below 2*num_experts -> overlap impossible, falls back to num_experts floor.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=10_000, per_expert_bytes=100, cache_per_page=10,
        num_experts=8, total_experts=12, prefill_overlap=True,
        kv_reserve_pages=0, max_slots=12,
    )
    assert overlap is False
    # raw = 10000//100 = 100, clamped to hi = min(12, 12) = 12.
    assert size == 12


def test_insufficient_kv_memory_raises():
    with pytest.raises(AssertionError, match="not enough memory"):
        plan_cache_budget(
            budget_bytes=410, per_expert_bytes=100, cache_per_page=10,
            num_experts=4, total_experts=4, prefill_overlap=False,
            kv_reserve_pages=0, max_slots=4,
        )  # experts eat 400, KV gets 1 page -> not > 1


def test_budget_too_small_for_min_moe_plus_reserve_raises():
    # Budget cannot fund even the minimum MoE slots + the KV reserve, so the floored plan
    # would exceed budget_bytes. Reject in arithmetic rather than OOM in a later CUDA alloc.
    with pytest.raises(AssertionError, match="budget too small"):
        plan_cache_budget(
            budget_bytes=300, per_expert_bytes=100, cache_per_page=10,
            num_experts=4, total_experts=4, prefill_overlap=False,
            kv_reserve_pages=10, max_slots=4,
        )  # min moe = 4 slots (400 B) + reserve (10 pages = 100 B) = 500 B > 300 B budget


def test_prefill_overlap_false_is_honored():
    # Even when the cache could fit 2*num_experts, an explicit False stays False.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=2000, per_expert_bytes=100, cache_per_page=10,
        num_experts=4, total_experts=8, prefill_overlap=False,
        kv_reserve_pages=0, max_slots=8,
    )
    assert size == 8
    assert overlap is False
    assert pages == (2000 - 8 * 100) // 10


def test_expert_bytes_per_slot_sums_row_bytes_over_banks():
    sources = {
        "gate_up": [torch.zeros(4, 32, 8, dtype=torch.float16)],  # row = 32*8*2 = 512
        "down": [torch.zeros(4, 8, 16, dtype=torch.float16)],     # row = 8*16*2 = 256
    }
    assert expert_bytes_per_slot(sources) == 512 + 256


def test_resolve_auto_applies_ratio_once_and_marlin_cap():
    # baseline 1000, weights 100, ratio 0.9 -> budget = 900 - 100 - 0(fixed) = 800
    size, pages, overlap = resolve_moe_cache_auto(
        baseline_free=1000, weights_bytes=100, memory_ratio=0.9,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=50,
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_tokens=0, page_size=1, quant_format="bf16",
    )
    # budget 800: experts cap at 8 -> 400 bytes; KV = 400//10 = 40 pages
    assert size == 8 and pages == 40 and overlap is True


def test_resolve_auto_marlin_caps_slots():
    size, _, _ = resolve_moe_cache_auto(
        baseline_free=10_000_000, weights_bytes=0, memory_ratio=1.0,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=100,
        num_experts=128, total_experts=4000, prefill_overlap=False,
        kv_reserve_tokens=0, page_size=1, quant_format="nvfp4_marlin",
    )
    assert size == 992


def _dsv4_adjust_cfg(**over):
    # A DSV4 _adjust_config stub mirroring the real checkpoint (ds_fp4 experts, dsv4_sparse
    # attention, offload MoE backend).
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        single_stream_only=False, dsv4_args=SimpleNamespace(window_size=128), is_moe=True,
        expert_quant="ds_fp4", has_swa_attention=False, has_linear_attention=False,
    )

    class Cfg:
        moe_cache_auto = True
        moe_cache_size = 0
        moe_cache_rate = None
        moe_backend = "offload"
        max_running_req = 1
        cuda_graph_max_bs = 1
        cuda_graph_bs = [1]
        max_seq_len = 1024
        max_extend_tokens = 4096
        page_size = 1
        attention_backend = "dsv4_sparse"
        moe_cpu_layers = None
        nvfp4_backend = "auto"
        num_page_override = None
        num_token_override = None

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    for k, v in over.items():
        object.__setattr__(cfg, k, v)
    return cfg


def test_adjust_config_allows_auto_for_dsv4():
    # DSV4 now supports --moe-cache-auto via the affine KV cost bridge (dsv4_auto_cost_model);
    # _adjust_config must NOT reject it.
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg()
    _adjust_config(cfg)  # must not raise
    assert cfg.moe_backend == "offload"
    assert cfg.moe_cache_auto is True  # resolved later at engine init, not here
    assert cfg.page_size == 128  # DSV4's KV page is the P-token window page


def test_adjust_config_resolves_num_tokens_for_dsv4():
    # --num-tokens resolves AFTER every page_size override, so DSV4's P=128 page divides it.
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131072)
    _adjust_config(cfg)
    assert cfg.page_size == 128
    assert cfg.num_page_override == 1024


def test_adjust_config_rejects_num_tokens_not_multiple_of_page():
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131000)  # not a multiple of 128
    with pytest.raises(ValueError, match="not a multiple"):
        _adjust_config(cfg)


def test_adjust_config_rejects_num_tokens_with_num_pages():
    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(num_token_override=131072, num_page_override=1024)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _adjust_config(cfg)


def test_adjust_config_resolves_num_tokens_generic():
    # Generic model keeps its page_size (1 here): tokens map 1:1 onto pages.
    from types import SimpleNamespace

    from freetoken.engine.engine import _adjust_config

    model_config = SimpleNamespace(
        single_stream_only=False, is_moe=False, expert_quant="none",
        has_swa_attention=False, has_linear_attention=False,
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
        moe_backend = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "fi"
        nvfp4_backend = "auto"
        num_page_override = None
        num_token_override = 5000

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    _adjust_config(cfg)
    assert cfg.num_page_override == 5000


def test_mha_kv_cost_simple_full_attention():
    import torch

    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.models.config import KVCacheGroupSpec
    from freetoken.utils import div_even

    class StubModelConfig:
        has_swa_attention = False

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=tuple(range(3)),
                num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.bfloat16
        page_size = 16
        max_running_req = 4
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    cache_per_page, fixed, _, _ = MHAKVCache.kv_cost(StubConfig())
    per_token = 2 * 64 * div_even(8, 1, allow_replicate=True) * 2 * 3
    assert cache_per_page == per_token * 16
    assert fixed == 0


def test_engine_resolve_auto_moe_cache_size_maps_kwargs():
    import torch

    from freetoken.engine.engine import Engine
    from freetoken.models.config import KVCacheGroupSpec

    class StubModelConfig:
        has_swa_attention = False
        num_experts = 4
        num_moe_layers = 2  # total_experts = 8

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=(0, 1, 2), num_kv_heads=8, head_dim=64, sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.float16
        page_size = 16
        max_running_req = 4
        hybrid_swa_cache_mode = "auto"
        memory_ratio = 0.9
        moe_prefill_overlap = True
        kv_reserve_tokens = 0
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    class StubBanks:
        quant_format = "bf16"
        # 2 layers (num_moe_layers above) x 4 experts each -- per-layer host bank contract.
        sources = {
            "gate_up": [torch.zeros(4, 32, 8, dtype=torch.float16)] * 2,  # row = 32*8*2 = 512
            "down": [torch.zeros(4, 8, 16, dtype=torch.float16)] * 2,     # row = 8*16*2 = 256
        }

    from freetoken.kvcache.mha_pool import MHAKVCache

    engine = Engine.__new__(Engine)  # bypass __init__/GPU
    engine._baseline_free = 10_000_000
    engine._weights_bytes = 1_000_000
    engine._pool_cls = MHAKVCache  # __init__ skipped -> install the generic pool family

    size, pages, overlap = engine._resolve_auto_moe_cache_size(StubConfig(), StubBanks())

    # cross-check against the same pure functions, proving the kwarg mapping is faithful
    from freetoken.engine.cache_budget import expert_bytes_per_slot, resolve_moe_cache_auto
    from freetoken.kvcache.mha_pool import MHAKVCache

    cache_per_page, fixed, _, _ = MHAKVCache.kv_cost(StubConfig())
    expected = resolve_moe_cache_auto(
        baseline_free=10_000_000, weights_bytes=1_000_000, memory_ratio=0.9,
        cache_per_page=cache_per_page, fixed_cache_size=fixed,
        per_expert_bytes=expert_bytes_per_slot(StubBanks.sources),
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_tokens=0, page_size=16, quant_format="bf16",
    )
    assert (size, pages, overlap) == expected


# ---------------------------------------------------------------------------
# offload-cache sizing guard + auto-resolution (_require_offload_cache_size / _adjust_config),
# the floor rule compute_cache_floors documents above.
# ---------------------------------------------------------------------------


def _offload_engine_config(**overrides):
    """A frozen EngineConfig for a quantized-experts MoE checkpoint in the bare-invocation state
    (moe_backend="auto") unless overridden — the shared fixture for the _adjust_config tests."""
    from freetoken.distributed import DistributedInfo
    from freetoken.engine.config import EngineConfig

    config = EngineConfig(
        model_path="/tmp/freetoken-test-model",
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        attention_backend="fi",
        **overrides,
    )
    object.__setattr__(
        config,
        "model_config",
        SimpleNamespace(
            has_swa_attention=False,
            has_linear_attention=False,
            is_moe=True,
            num_layers=10,
            num_moe_layers=10,
            num_experts=8,
            expert_quant="nvfp4",  # quantized experts -> must resolve to an offload backend
            moe_backend="auto",
        ),
    )
    return config


def test_guard_passes_when_size_covers_one_expert_per_layer():
    from freetoken.engine.engine import _require_offload_cache_size

    _require_offload_cache_size(cache_size=128, num_experts=128)  # no raise


def test_guard_raises_actionable_error_when_too_small():
    from freetoken.engine.engine import _require_offload_cache_size

    with pytest.raises(ValueError) as exc:
        _require_offload_cache_size(cache_size=0, num_experts=128)
    msg = str(exc.value)
    assert "128" in msg and "moe-cache" in msg


def test_adjust_config_defaults_moe_cache_auto_for_auto_resolved_offload_backend():
    """Bare `ft serve <FTW MoE checkpoint>`: no --moe-backend, no --moe-cache-* flags at all.

    args.py's parse-time default only fires when the backend is *already*
    offload-family at parse time -- but a bare invocation leaves moe_backend="auto" at parse
    time, and the "auto" -> offload/cpu/hybrid resolution only happens later, in _adjust_config,
    once the model_config (and its expert_quant) is known. This proves the engine-level
    resolution: a quantized-experts model auto-resolving to an offload-family backend also gets
    moe_cache_auto=True, so _init_offload_moe_cache's _require_offload_cache_size guard is never
    reached with moe_cache_size still 0.
    """
    from freetoken.engine.engine import _adjust_config
    from freetoken.moe import is_offload_moe_backend

    config = _offload_engine_config()
    _adjust_config(config)

    # Which member of the family gets picked is not this test's claim, and is not ours to
    # decide: a bare "auto" consults ~/.cache/freetoken/benchbw.json, so a box that has run
    # `ft bench bw` resolves nvfp4 experts to hybrid instead. Assert the family, not the member.
    assert is_offload_moe_backend(config.moe_backend)
    assert config.moe_cache_auto is True
    assert config.moe_cache_size == 0  # still unresolved -- the scheduler sizes it from VRAM


def test_page_table_width_covers_whole_trailing_pages():
    # _write_page_table writes WHOLE trailing pages, so the width must reach the last
    # page's end, not just the next multiple of 32 (DSV4's P=128 exposed the gap).
    from freetoken.engine.engine import _page_table_width

    assert _page_table_width(4001, 128) == 4096   # align32 alone gave 4032 -> OOB
    assert _page_table_width(4096, 128) == 4096   # page-aligned length unchanged
    assert _page_table_width(100, 1) == 128       # page_size 1 degenerates to align32
    assert _page_table_width(33, 128) == 128
    for max_seq_len in (1, 31, 33, 4001, 4095, 4096):
        for page_size in (1, 32, 64, 128):
            w = _page_table_width(max_seq_len, page_size)
            last_col = -(-max_seq_len // page_size) * page_size - 1
            assert w > last_col and w % 32 == 0


def _generic_rotary_cfg(max_position, override):
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        single_stream_only=False, is_moe=False, expert_quant="none",
        has_swa_attention=False, has_linear_attention=False,
        rotary_config=SimpleNamespace(max_position=max_position),
    )

    class Cfg:
        moe_cache_auto = False
        moe_cache_size = 0
        moe_cache_rate = None
        moe_backend = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "triton"
        nvfp4_backend = "auto"
        num_page_override = None
        num_token_override = None
        max_seq_len_override = None

        @property
        def model_config(self):
            return model_config

    cfg = Cfg()
    object.__setattr__(cfg, "max_seq_len_override", override)
    return cfg


def test_adjust_config_rejects_override_past_rope_table():
    from freetoken.engine.engine import _adjust_config

    with pytest.raises(ValueError, match="rope table"):
        _adjust_config(_generic_rotary_cfg(max_position=1024, override=2048))


def test_adjust_config_allows_override_at_rope_table_boundary():
    from freetoken.engine.engine import _adjust_config

    _adjust_config(_generic_rotary_cfg(max_position=1024, override=1024))  # must not raise


def test_adjust_config_rope_gate_exempts_dsv4():
    # DSV4 sizes its own rope table from the resolved max_seq_len (_adjust_dsv4_config),
    # so the generic gate must not fire even when the override dwarfs max_position.
    from types import SimpleNamespace

    from freetoken.engine.engine import _adjust_config

    cfg = _dsv4_adjust_cfg(max_seq_len_override=10_000_000)
    cfg.model_config.rotary_config = SimpleNamespace(max_position=1024)
    _adjust_config(cfg)  # must not raise


# ---- _pin_budget_bytes: host bytes already pinned outside the expert banks ----


def test_reserved_subtracts_from_the_cap(monkeypatch):
    monkeypatch.setenv("FREETOKEN_PIN_BUDGET_GB", "2")
    assert _pin_budget_bytes() == 2 * 2**30
    assert _pin_budget_bytes(reserved=2**30) == 2**30
    assert _pin_budget_bytes(reserved=4 * 2**30) == 0


def test_linux_pin_budget_uses_host_memory_ratio(monkeypatch):
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)
    if hasattr(os, "uname") and "microsoft" in os.uname().release.lower():
        pytest.skip("WSL caps pinning differently")
    # default host_memory_ratio=0.9 -> returns a positive budget, not None
    budget = _pin_budget_bytes(reserved=0, host_memory_ratio=0.8)
    assert budget is not None
    assert budget > 0
    # host_memory_ratio=1.0 with no reserved -> large budget (total RAM)
    full = _pin_budget_bytes(reserved=0, host_memory_ratio=1.0)
    assert full is not None
    assert full >= budget


# ---- host_memory_budget_bytes ----


def test_host_memory_budget_basic(monkeypatch):
    monkeypatch.setattr(os, "sysconf", lambda name: {
        "SC_PHYS_PAGES": 16 << 10,  # pages
        "SC_PAGE_SIZE": 1 << 20,    # 1 MiB pages -> 16 GiB total
    }.get(name, 0))
    # 80% of 16 GiB = 12.8 GiB
    assert host_memory_budget_bytes(0.8) == int(0.8 * 16 * 2**30)


def test_host_memory_budget_reserved_subtracts(monkeypatch):
    monkeypatch.setattr(os, "sysconf", lambda name: {
        "SC_PHYS_PAGES": 16 << 10,
        "SC_PAGE_SIZE": 1 << 20,
    }.get(name, 0))
    reserved = 4 * 2**30  # 4 GiB
    expected = int(0.9 * 16 * 2**30) - reserved
    assert host_memory_budget_bytes(0.9, reserved=reserved) == expected


def test_host_memory_budget_clamps_at_zero(monkeypatch):
    monkeypatch.setattr(os, "sysconf", lambda name: {
        "SC_PHYS_PAGES": 4 << 10,   # 4 GiB total
        "SC_PAGE_SIZE": 1 << 20,
    }.get(name, 0))
    # reserved larger than the ratio-scaled total -> 0
    assert host_memory_budget_bytes(0.5, reserved=8 * 2**30) == 0


# ---- host_pin_allowance / cpu_layer_count: the free-RAM floor ----

GiB = 2**30


def _stub_ram(monkeypatch, total_gib: int):
    """1 MiB pages, so ``total_gib << 10`` pages is exactly total_gib GiB."""
    monkeypatch.setattr(os, "sysconf", lambda name: {
        "SC_PHYS_PAGES": total_gib << 10,
        "SC_PAGE_SIZE": 1 << 20,
    }.get(name, 0))


def test_pin_allowance_keeps_the_free_floor(monkeypatch):
    _stub_ram(monkeypatch, 60)
    # ratio 0.8 -> 12 GiB must stay free; 47 GiB is available now, so 35 GiB may be pinned.
    assert host_pin_allowance(47 * GiB, 0.8) == 47 * GiB - 12 * GiB


def test_pin_allowance_shrinks_on_a_busy_host(monkeypatch):
    """The regression: a MemTotal-derived budget ignores what is already resident.

    Same ratio, same machine, but the desktop is holding most of the RAM. Sizing against
    ``ratio * MemTotal`` still hands the banks 48 GiB and leaves nothing free; the measured
    allowance has to collapse to what is actually available minus the floor.
    """
    _stub_ram(monkeypatch, 60)
    assert host_memory_budget_bytes(0.8) == 48 * GiB  # unchanged, and now wrong
    assert host_pin_allowance(20 * GiB, 0.8) == 8 * GiB


def test_pin_allowance_clamps_at_zero(monkeypatch):
    _stub_ram(monkeypatch, 60)
    assert host_pin_allowance(5 * GiB, 0.8) == 0  # already below the 12 GiB floor


def _engine_with_tables(monkeypatch, avail_gib: int, tables_gib: int):
    """An Engine shell (no __init__, no GPU) with a stubbed MemAvailable and RAM size."""
    import freetoken.engine.engine as eng
    from freetoken.engine.engine import Engine

    _stub_ram(monkeypatch, 60)  # 60 GiB total -> free_floor 12 GiB, ratio budget 48 GiB
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)
    monkeypatch.setattr(eng, "_host_mem_available_bytes", lambda: avail_gib * GiB)
    engine = Engine.__new__(Engine)
    engine._host_tables_bytes = tables_gib * GiB
    return engine, SimpleNamespace(host_memory_ratio=0.8)


def test_pin_allowance_is_clamped_by_the_ratio_of_total_cap(monkeypatch):
    """Tables pinned OUTSIDE the banks are already out of MemAvailable, so the measured
    allowance on its own lets tables + banks overshoot ratio * MemTotal."""
    engine, config = _engine_with_tables(monkeypatch, avail_gib=45, tables_gib=20)
    # measured: 45 - 12 (free floor) = 33 GiB; cap: 48 - 20 (tables) = 28 GiB
    assert engine._sync_host_pin_allowance(config) == 28 * GiB
    assert engine._host_tables_bytes + 28 * GiB == int(0.8 * 60 * GiB)


def test_pin_allowance_keeps_the_measured_figure_when_it_is_tighter(monkeypatch):
    """The clamp must not raise the allowance: on a busy host the measured figure still wins."""
    engine, config = _engine_with_tables(monkeypatch, avail_gib=20, tables_gib=0)
    assert engine._sync_host_pin_allowance(config) == 8 * GiB  # 20 - 12, below the 48 GiB cap


def test_cpu_layer_count_zero_when_the_banks_fit():
    assert cpu_layer_count(30 * GiB, 48, 35 * GiB) == 0


def test_cpu_layer_count_is_machine_wide_not_per_rank():
    """TP ranks each pin their own shard on the SAME host, so the total has to fit.

    63 GiB of banks over 2 ranks is 31.5 GiB/rank, which fits a 35 GiB allowance per rank --
    and 63 GiB machine-wide, which does not. Dividing by tp here would move no layers.
    """
    assert cpu_layer_count(63 * GiB, 48, 35 * GiB) == 22  # ceil(48 * 28/63)
    assert cpu_layer_count(64 * GiB, 48, 32 * GiB) == 24  # exact: half the layers


def test_cpu_layer_count_caps_at_every_layer():
    assert cpu_layer_count(63 * GiB, 48, 0) == 48
    assert cpu_layer_count(63 * GiB, 0, 35 * GiB) == 0


# ---- watchdog_nudge_bytes / plan_pageout: the runtime reclaim policy ----

MiB = 2**20


def test_nudge_is_capped_by_the_ram_that_is_actually_free():
    """The regression: 1 GiB available with a 10.9 GiB deficit.

    ``min(deficit + 256 MiB, 2 GiB)`` asked for 2 GiB -- more than existed -- so the nudge
    that exists to relieve pressure became the pressure: direct reclaim in the allocating
    thread plus 2 GiB of new anonymous pages that themselves need swap.
    """
    assert watchdog_nudge_bytes(11 * GiB, 1 * GiB) == GiB // 2
    assert watchdog_nudge_bytes(100 * MiB, 1 * GiB) == 356 * MiB  # deficit + 256 MiB wins
    assert watchdog_nudge_bytes(10 * GiB, 100 * GiB) == 2 * GiB   # the ceiling wins
    assert watchdog_nudge_bytes(10 * GiB, 0) == 0                 # nothing free -> no nudge


def test_nudge_never_exceeds_the_ram_it_would_have_to_come_out_of():
    for avail in (MiB, 64 * MiB, GiB, 8 * GiB):
        assert watchdog_nudge_bytes(64 * GiB, avail) <= avail


def test_plan_pageout_stops_once_the_budget_is_covered():
    sizes = [10, 20, 30, 40]
    assert plan_pageout(sizes, 25, 0) == ([0, 1], 2)  # 10 then 10+20 >= 25
    assert plan_pageout(sizes, 25, 2) == ([2], 3)     # 30 alone covers it


def test_plan_pageout_sweeps_from_the_cursor_and_wraps():
    sizes = [10, 20, 30, 40]
    assert plan_pageout(sizes, 25, 3) == ([3], 0)     # wraps the cursor to the head
    assert plan_pageout(sizes, 5, 7) == ([3], 0)      # cursor is normalized mod n


def test_plan_pageout_never_advises_one_bank_twice_in_a_sweep():
    sizes = [5, 5, 5]
    indices, cursor = plan_pageout(sizes, 1000, 1)    # budget above the total
    assert sorted(indices) == [0, 1, 2]
    assert cursor == 1                                # a full sweep returns to the start


def test_plan_pageout_rotation_covers_every_bank_over_successive_ticks():
    """Always restarting at 0 would re-evict the banks the CPU executor just faulted in."""
    sizes = [10] * 5
    cursor, seen = 0, []
    for _ in range(5):
        indices, cursor = plan_pageout(sizes, 10, cursor)
        seen.extend(indices)
    assert seen == [0, 1, 2, 3, 4]


def test_plan_pageout_edge_cases():
    assert plan_pageout([], 100, 5) == ([], 0)   # no banks at all
    assert plan_pageout([10, 20], 0, 1) == ([], 1)   # no deficit -> advise nothing
    assert plan_pageout([10, 20], -5, 1) == ([], 1)
