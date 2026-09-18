from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import os

from freetoken.engine.cache_budget import (
    expert_bytes_per_slot,
    has_explicit_cache_sizing,
    plan_cache_budget,
    resolve_moe_cache_auto,
    slot_cache_expert_cap,
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


def test_resolve_auto_applies_ratio_once():
    # baseline 1000, weights 100, ratio 0.9 -> budget = 900 - 100 - 0(fixed) = 800
    size, pages, overlap = resolve_moe_cache_auto(
        baseline_free=1000, weights_bytes=100, memory_ratio=0.9,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=50,
        num_experts=4, total_experts=8, prefill_overlap=True,
        kv_reserve_tokens=0, page_size=1,
    )
    # budget 800: experts cap at 8 -> 400 bytes; KV = 400//10 = 40 pages
    assert size == 8 and pages == 40 and overlap is True


def test_resolve_auto_caps_slots_at_the_kernel_limit():
    size, _, _ = resolve_moe_cache_auto(
        baseline_free=10_000_000, weights_bytes=0, memory_ratio=1.0,
        cache_per_page=10, fixed_cache_size=0, per_expert_bytes=100,
        num_experts=128, total_experts=4000, prefill_overlap=False,
        kv_reserve_tokens=0, page_size=1, max_slots=992,
    )
    assert size == 992


# --------------------------------------------------- residency-aware slot ceiling

def _residency(pinned, num_layers):
    from freetoken.moe.host_banks import HostResidency

    return [
        HostResidency.PINNED.value if i in pinned else HostResidency.PAGEABLE.value
        for i in range(num_layers)
    ]


def test_slot_cache_expert_cap_counts_only_reachable_layers():
    # 1 of 4 layers keeps a device address: its experts plus the one-window pageable
    # prefill staging. The other 3 layers' experts can never occupy a GPU slot.
    assert slot_cache_expert_cap(4, 4, _residency({0}, 4), prefill_overlap=False) == 8


def test_slot_cache_expert_cap_adds_the_overlap_double_buffer():
    assert slot_cache_expert_cap(4, 4, _residency({0}, 4), prefill_overlap=True) == 12


def test_slot_cache_expert_cap_of_an_all_cpu_model_is_the_two_layer_buffer():
    # Every bank non-pinned: the cache is nothing but prefill staging, the geometry
    # --moe-strategy cpu fixes at 2 * num_experts.
    assert slot_cache_expert_cap(4, 4, _residency(set(), 4), prefill_overlap=True) == 8


def test_slot_cache_expert_cap_allows_full_residency_when_every_layer_is_pinned():
    assert slot_cache_expert_cap(4, 4, _residency({0, 1, 2, 3}, 4), prefill_overlap=True) == 16


@pytest.mark.parametrize("residency", [None, [], ["pinned"] * 3, ["pageable"] * 5])
def test_slot_cache_expert_cap_trusts_an_unlabeled_loader(residency):
    # A loader that does not echo per-layer residency (or echoes a partial list) must not
    # be read as "nothing is reachable": report the whole-model ceiling instead.
    assert slot_cache_expert_cap(4, 4, residency, prefill_overlap=False) == 16


def test_capped_plan_fits_the_floor_and_gives_the_rest_to_kv():
    # The cap must stay a legal plan_cache_budget argument: 8 slots >= the num_experts
    # floor, and the freed bytes land in the KV page count.
    size, pages, overlap = plan_cache_budget(
        budget_bytes=8_000_000, per_expert_bytes=480_000, cache_per_page=131_072,
        num_experts=4, total_experts=16, prefill_overlap=False,
        kv_reserve_pages=0, max_slots=slot_cache_expert_cap(4, 4, _residency({0}, 4),
                                                            prefill_overlap=False),
    )
    assert size == 8 and overlap is False
    assert pages == (8_000_000 - 8 * 480_000) // 131_072


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
        moe_strategy = "offload"
        max_running_req = 1
        cuda_graph_max_bs = 1
        cuda_graph_bs = [1]
        max_seq_len = 1024
        max_extend_tokens = 4096
        page_size = 1
        attention_backend = "dsv4_sparse"
        moe_cpu_layers = None
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
    assert cfg.moe_strategy == "offload"
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
        moe_strategy = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "fi"
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
        # 2 layers (num_moe_layers above) x 4 experts each -- per-layer host bank contract.
        sources = {
            "gate_up": [torch.zeros(4, 32, 8, dtype=torch.float16)] * 2,  # row = 32*8*2 = 512
            "down": [torch.zeros(4, 8, 16, dtype=torch.float16)] * 2,     # row = 8*16*2 = 256
        }

    from freetoken.kvcache.mha_pool import MHAKVCache

    engine = Engine.__new__(Engine)  # bypass __init__/GPU
    engine._baseline_free = 10_000_000
    engine._weights_bytes = 1_000_000
    # No device: memory_ratio keeps its historical free-baseline reading.
    engine._device_total = 0
    engine._nonpool_overhead_floor = 0
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
        kv_reserve_tokens=0, page_size=16,
    )
    assert (size, pages, overlap) == expected

    class StubMethod:
        def slot_limit(self):
            return 5

    size, _, _ = engine._resolve_auto_moe_cache_size(StubConfig(), StubBanks(), StubMethod())
    assert size == 5


# ------------------------------------------------- auto plan vs host residency / pinned KV
#
# Stub geometry: budget = 0.9 * 10_000_000 - 1_000_000 = 8_000_000 B, one expert slot
# 320_000 + 160_000 = 480_000 B, one KV page 8192 B/token * 16 = 131_072 B.


def _auto_plan_stub(num_layers=4, *, layer_residency="unset", num_page_override=None,
                    moe_prefill_overlap=False):
    from freetoken.engine.engine import Engine
    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.models.config import KVCacheGroupSpec

    class ModelConfig:
        has_swa_attention = False
        num_experts = 4
        num_moe_layers = num_layers

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=tuple(range(num_layers)), num_kv_heads=8, head_dim=64,
                sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class Config:
        dtype = torch.float16
        page_size = 16
        max_running_req = 4
        hybrid_swa_cache_mode = "auto"
        memory_ratio = 0.9
        kv_reserve_tokens = 0
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = ModelConfig()

        class tp_info:
            size = 1

    Config.moe_prefill_overlap = moe_prefill_overlap
    Config.num_page_override = num_page_override

    class Banks:
        sources = {
            "gate_up": [torch.zeros(4, 200, 800, dtype=torch.float16)] * num_layers,  # 320_000
            "down": [torch.zeros(4, 400, 200, dtype=torch.float16)] * num_layers,  # 160_000
        }

    if layer_residency != "unset":
        Banks.layer_residency = layer_residency

    engine = Engine.__new__(Engine)  # bypass __init__/GPU
    engine._baseline_free = 10_000_000
    engine._weights_bytes = 1_000_000
    # No device: memory_ratio keeps its historical free-baseline reading.
    engine._device_total = 0
    engine._nonpool_overhead_floor = 0
    engine._pool_cls = MHAKVCache
    return engine, Config(), Banks()


def test_auto_plan_spends_unreachable_expert_bytes_on_kv():
    # The WSL2 shape: --moe-cpu-layers leaves 1 of 4 MoE layers with a device address. The
    # greedy fill must not plan all 16 experts on GPU (7.7 MB of the 8 MB budget, in one
    # bank-sized contiguous allocation); the unreachable bytes belong to the KV pool.
    engine, config, banks = _auto_plan_stub(layer_residency=None)
    uncapped_size, uncapped_pages, _ = engine._resolve_auto_moe_cache_size(config, banks)
    assert (uncapped_size, uncapped_pages) == (16, 2)  # every expert of every layer planned

    engine, config, banks = _auto_plan_stub(layer_residency=_residency({0}, 4))
    size, pages, _ = engine._resolve_auto_moe_cache_size(config, banks)
    assert size == 8  # the pinned layer's experts + the pageable-prefill staging window
    assert pages == (8_000_000 - size * 480_000) // 131_072 > uncapped_pages
    assert size * 480_000 + pages * 131_072 <= 8_000_000


def test_auto_plan_reserves_a_pinned_kv_geometry():
    # --num-tokens pins the KV pool; solve_num_pages spends it whatever the plan says, so
    # the expert fill must leave room for it, not only for kv_reserve_tokens.
    engine, config, banks = _auto_plan_stub(layer_residency=None, num_page_override=3)
    size, pages, _ = engine._resolve_auto_moe_cache_size(config, banks)
    assert size == 15  # one layer shorter than the unpinned plan
    assert pages >= 3  # the pinned geometry survives the MoE-first split


def test_auto_plan_of_the_reported_wsl2_run_caches_one_layer_not_the_model():
    # RadixArk-Qwen3.8-Flash-Next-NVFP4: 48 MoE layers x 512 experts, 47 of them settled
    # pageable after the mlock failure, prefill overlap off -> the plan asked for all
    # 24576 slots (a 37.5 GiB gate_up bank); only one layer + staging can ever be filled.
    assert slot_cache_expert_cap(
        48, 512, _residency({24}, 48), prefill_overlap=False
    ) == 1024


# ------------------------------------------------- memory_ratio as a total-footprint cap

_GIB = 1 << 30


def test_footprint_cap_reads_ratio_off_the_whole_device():
    # The historical reading scales whatever FREE memory happened to be available before the
    # weights loaded; with a device total, --memory-ratio instead caps the engine's whole
    # footprint, so bytes already committed outside the pools are charged against it.
    from freetoken.engine.cache_budget import net_cache_budget_bytes

    # 24 GiB card, 6 GiB of weights, and 3 GiB of CUDA context / NCCL / co-tenant that the
    # legacy view never accounts for anywhere. Only 8 GiB was free at the baseline, so the
    # legacy figure is the smaller one and the cap must not enlarge the plan.
    total, baseline, weights, overhead = 24 * _GIB, 8 * _GIB, 6 * _GIB, 3 * _GIB
    legacy = net_cache_budget_bytes(0.85, baseline, weights, 0)
    capped = net_cache_budget_bytes(
        0.85, baseline, weights, 0, device_total=total, nonpool_overhead_bytes=overhead
    )
    assert legacy == int(0.85 * baseline) - weights
    assert capped == legacy < int(0.85 * total) - weights - overhead
    # What the engine then ends up resident on the device honors ratio x total.
    assert capped + weights + overhead <= int(0.85 * total)


def test_footprint_cap_holds_when_free_memory_is_not_the_binding_limit():
    # The case the legacy reading gets wrong: a nearly empty device. ratio x free is then
    # bigger than ratio x device minus the unaccounted overhead, and only the cap keeps the
    # footprint inside what --memory-ratio promises.
    from freetoken.engine.cache_budget import net_cache_budget_bytes

    total, weights, overhead = 24 * _GIB, 6 * _GIB, 2 * _GIB
    legacy = net_cache_budget_bytes(0.85, total, weights, 0)
    capped = net_cache_budget_bytes(
        0.85, total, weights, 0, device_total=total, nonpool_overhead_bytes=overhead
    )
    assert legacy > capped
    assert capped + weights + overhead == int(0.85 * total)


def test_footprint_cap_never_enlarges_the_plan():
    # A co-tenant can make ratio x device smaller than the legacy figure. The cap guards
    # against exceeding it, so it must not silently hand out MORE bytes than before.
    from freetoken.engine.cache_budget import net_cache_budget_bytes

    legacy = net_cache_budget_bytes(0.9, 10_000, 1_000, 0)
    with_room = net_cache_budget_bytes(0.9, 10_000, 1_000, 0, device_total=100_000)
    assert with_room == legacy
    assert net_cache_budget_bytes(0.9, 10_000, 1_000, 0, device_total=0) == legacy


def test_nonpool_overhead_is_clamped_by_the_cross_rank_min_free():
    # Under TP every rank plans off the same cross-rank MIN free figure; a rank whose local
    # context is bigger must not spend bytes the tightest rank does not have.
    from freetoken.engine.engine import _nonpool_overhead_bytes

    assert _nonpool_overhead_bytes(24 * _GIB, 20 * _GIB, 20 * _GIB) == 4 * _GIB
    assert _nonpool_overhead_bytes(24 * _GIB, 20 * _GIB, 1 * _GIB) == 1 * _GIB
    # No usable device total -> no cap, and the callers fall back to the legacy reading.
    assert _nonpool_overhead_bytes(0, 20 * _GIB, 20 * _GIB) == 0
    assert _nonpool_overhead_bytes(24 * _GIB, -1, 20 * _GIB) == 0


def test_growth_cap_respects_the_footprint_and_reserve():
    from freetoken.engine.cache_budget import growth_cap_bytes

    total = 24 * _GIB
    # ratio x total = 20.4 GiB is the ceiling on the WHOLE footprint, so a 20 GiB footprint
    # may only grow by the 0.4 GiB still under it -- the cap binds, not the 1.5 GiB reserve.
    assert growth_cap_bytes(0.85, total, 20 * _GIB, 1536 * 1024 * 1024) == int(0.85 * total) - 20 * _GIB
    # A ratio near 1 leaves the ceiling loose, so the graph/activation reserve is what binds.
    assert growth_cap_bytes(0.99, total, 20 * _GIB, 1536 * 1024 * 1024) == total - 20 * _GIB - 1536 * 1024 * 1024
    # Nothing left below the reserve -> no growth at all, never a negative room.
    assert growth_cap_bytes(0.85, total, 23 * _GIB, 2 * _GIB) == 0


def test_growth_never_spends_the_memory_ratio_remainder():
    """--memory-ratio is the top-level footprint authority: the (1 - ratio) remainder is what
    it reserves for graph capture and prefill activations, so the post-init growth may only
    reclaim rounding slack BELOW the ceiling. Regression for a cap term that bounded the room
    by ``ratio x total`` instead of ``ratio x total - footprint`` and so never bound: on the
    measured rig (24467 MiB device, ratio 0.85, 20608 MiB resident) it grew the expert cache
    by 1.92 GiB and landed the footprint at 92.8% of the device."""
    from freetoken.engine.cache_budget import growth_cap_bytes

    mib = 1024 * 1024
    total, footprint, reserve = 24467 * mib, 20608 * mib, 1536 * 1024 * 1024
    room = growth_cap_bytes(0.85, total, footprint, reserve)
    assert room == int(0.85 * total) - footprint          # ~189 MiB of rounding slack
    assert footprint + room <= int(0.85 * total)          # the ceiling holds
    assert room < 2 * 1024 * mib                          # ... not the 1.92 GiB it used to take
    # A footprint already at or above the ceiling must not grow at all.
    assert growth_cap_bytes(0.85, total, int(0.85 * total), reserve) == 0
    assert growth_cap_bytes(0.85, total, int(0.85 * total) + mib, reserve) == 0


def test_explicit_cache_sizing_suppresses_auto_fallback():
    from freetoken.engine.cache_budget import has_explicit_cache_sizing

    assert has_explicit_cache_sizing(4000, None) is True
    assert has_explicit_cache_sizing(0, 0.25) is True
    assert has_explicit_cache_sizing(0, None) is False
    assert has_explicit_cache_sizing(0, 0) is False


def test_pinned_slot_cache_skips_the_headroom_growth():
    # --moe-cache-size pins the cache; growing it into the (1 - ratio) remainder afterwards
    # would put bytes outside the pool budget, so --memory-ratio stops being the footprint cap.
    # The gate itself lives in Engine.__init__, which needs a GPU, so assert on its terms:
    # an explicit size is detected, and _adjust_config never turns auto back on for it.
    config = _offload_engine_config(moe_strategy="offload", moe_cache_size=12)
    assert has_explicit_cache_sizing(config.moe_cache_size, config.moe_cache_rate)


def test_headroom_growth_survives_the_auto_size_write_back():
    """--moe-cache-auto resolves a size and writes it back into config.moe_cache_size, so the
    growth gate must read the PRE-resolution latch: reading the config at the gate sees the
    resolved 8969 as user-pinned and silently skips the growth on every auto-sized run."""
    from freetoken.engine.cache_budget import headroom_growth_eligible

    # the latch, taken before the resolution: auto-sizing was not asked for explicitly
    latched = has_explicit_cache_sizing(0, None)
    assert headroom_growth_eligible(moe_cache_auto=True, sizing_explicit=latched) is True
    # what the same config looks like after _init_offload_moe_cache wrote the size back
    resolved = has_explicit_cache_sizing(8969, None)
    assert headroom_growth_eligible(moe_cache_auto=True, sizing_explicit=resolved) is False
    # a user-pinned cache still declines, whichever reading is used
    assert headroom_growth_eligible(moe_cache_auto=False, sizing_explicit=True) is False


def test_growth_cap_only_tightens_the_reserve():
    # A rank that skips the growth on its own leaves the others blocked in the go/no-go
    # all-reduce, so the cap may only ever tighten `reserve` -- room == 0 must fall through
    # to extra <= 0 (go stays False) instead of returning early.
    from freetoken.engine.cache_budget import growth_cap_bytes

    total, reserve, post_free = 24 * _GIB, 2 * _GIB, 1 * _GIB
    room = growth_cap_bytes(0.85, total, total - post_free, reserve)
    assert room == 0
    tightened = max(reserve, post_free - room)
    assert tightened >= post_free  # nothing spendable -> extra == 0 on this rank


def test_adjust_config_keeps_an_explicit_slot_cache_pinned():
    """--moe-cache-size must not be re-widened by the auto fallback it suppresses."""
    from freetoken.engine.engine import _adjust_config
    from freetoken.moe import is_offload_moe_strategy

    config = _offload_engine_config(moe_cache_size=4000)
    _adjust_config(config)
    assert is_offload_moe_strategy(config.moe_strategy)
    assert config.moe_cache_auto is False
    assert config.moe_cache_size == 4000


# ---------------------------------------------------------------------------
# offload-cache sizing guard + auto-resolution (_require_offload_cache_size / _adjust_config),
# the floor rule compute_cache_floors documents above.
# ---------------------------------------------------------------------------


def _offload_engine_config(**overrides):
    """A frozen EngineConfig for a quantized-experts MoE checkpoint in the bare-invocation state
    (moe_strategy="auto") unless overridden — the shared fixture for the _adjust_config tests."""
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
            moe_strategy="auto",
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
    offload-family at parse time -- but a bare invocation leaves moe_strategy="auto" at parse
    time, and the "auto" -> offload/cpu/hybrid resolution only happens later, in _adjust_config,
    once the model_config (and its expert_quant) is known. This proves the engine-level
    resolution: a quantized-experts model auto-resolving to an offload-family backend also gets
    moe_cache_auto=True, so _init_offload_moe_cache's _require_offload_cache_size guard is never
    reached with moe_cache_size still 0.
    """
    from freetoken.engine.engine import _adjust_config
    from freetoken.moe import is_offload_moe_strategy

    config = _offload_engine_config()
    _adjust_config(config)

    # Which member of the family gets picked is not this test's claim, and is not ours to
    # decide: a bare "auto" consults ~/.cache/freetoken/benchbw.json, so a box that has run
    # `ft bench bw` resolves nvfp4 experts to hybrid instead. Assert the family, not the member.
    assert is_offload_moe_strategy(config.moe_strategy)
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
        moe_strategy = "auto"
        max_running_req = 4
        cuda_graph_max_bs = 2
        cuda_graph_bs = [1, 2]
        max_seq_len = 1024
        page_size = 1
        attention_backend = "triton"
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


def test_uncapped_platform_stays_uncapped(monkeypatch):
    monkeypatch.delenv("FREETOKEN_PIN_BUDGET_GB", raising=False)
    if hasattr(os, "uname") and "microsoft" in os.uname().release.lower():
        pytest.skip("WSL caps pinning")
    assert _pin_budget_bytes(reserved=2**30) is None
