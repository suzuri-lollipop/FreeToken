"""Hybrid decode's bandwidth-matched fetch split.

Covers the profile reader of --moe-hybrid-max-fetch auto, which turns the `ft bench bw`
row of each MoE layer kind into a fetch fraction.
"""

import json
import os

import pytest
import torch

from freetoken.moe.bench_profile import (
    _CPU_HIDEABILITY,
    _HIDEABILITY_MAX_SHIFT,
    PROFILE_VERSION,
    default_profile_path,
    fetch_fraction,
    layer_kind_key,
    load_layer_kinds,
    locked_fetch_fraction,
)
from freetoken.moe.offload_cache import OffloadMoeCache

Q = 1 << 16


def test_layer_kind_rows_give_fetch_fractions(tmp_path):
    bf16 = layer_kind_key("bf16", {}, 2048, 768, 8, 12)
    gguf = layer_kind_key("gguf", {"gate_up_type": 12, "down_type": 14}, 2048, 768, 8, 12)
    assert gguf == "gguf[down_type=14,gate_up_type=12]:2048x768:top8:t12"
    prof = {
        "version": PROFILE_VERSION,
        "gpu": {"name": "FAKE GPU"},
        "layer_kinds": {
            bf16: {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            # overlapped (contended) pair wins over the standalone numbers when present
            gguf: {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0, "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    rows = load_layer_kinds(path=str(path))
    # standalone fallback: full-contention assumption -> pcie / cpu
    assert fetch_fraction(rows[bf16]) == pytest.approx(0.4)
    # overlapped pair preferred: pcie_ov / (pcie_ov + cpu_ov)
    assert fetch_fraction(rows[gguf]) == pytest.approx(0.25)
    # a locked layer's fetch adds a memcpy at the CPU MoE rate before the gather: p / (2p + c), from the same pair
    assert locked_fetch_fraction(rows[bf16]) == pytest.approx(40 / 180)
    assert locked_fetch_fraction(rows[gguf]) == pytest.approx(30 / 150)
    # another thread count is another kind
    assert layer_kind_key("bf16", {}, 2048, 768, 8, 6) not in rows
    # a profile from different hardware is ignored
    assert load_layer_kinds(gpu_name="OTHER", path=str(path)) is None
    # so is one of another version
    path.write_text(json.dumps({**prof, "version": PROFILE_VERSION - 1}))
    assert load_layer_kinds(path=str(path)) is None


def _balanced_fetch(num_missing: int, frac_q16: int) -> int:
    """Reference split: F ~ frac * misses, rounded to whichever integer neighbor
    minimizes the slower overlapped side (fetch ~ F*(1-frac), CPU ~ (M-F)*frac)."""
    lo = (num_missing * frac_q16) >> 16
    cost = lambda f: max(f * (Q - frac_q16), (num_missing - f) * frac_q16)  # noqa: E731
    return min(num_missing, lo if cost(lo) <= cost(lo + 1) else lo + 1)


def test_balanced_fetch_tracks_fraction():
    # The split follows fetched : cpu = pcie : (cpu - pcie) up to integer rounding, and
    # never over/under-shoots by more than one expert.
    for frac in (0.1, 0.415, 0.454, 0.7, 1.0):
        q = round(frac * Q)
        for m in range(0, 65):
            f = _balanced_fetch(m, q)
            assert 0 <= f <= m
            assert abs(f - frac * m) <= 1.0
    # ceil would over-fetch here (the regression this rule fixed): 41.5% of 3 misses is
    # 1.24 -> fetching 2 makes the PCIe side ~1.6x slower than balance; keep it at 1.
    assert _balanced_fetch(3, round(0.415 * Q)) == 1
    assert _balanced_fetch(4, round(0.415 * Q)) == 2


def test_prefill_cpu_split_off_unless_a_cpu_executor_and_fraction_are_in_play():
    """The prefill CPU split is the decode executor, so it must stay off when any one
    ingredient is missing: no executor, a fetch fraction of 1 (pure-PCIe plan), no CPU
    mask, or a chunk longer than the executor's pinned-buffer ``max_tokens``. A slip that
    routed such a chunk to the CPU side would silently drop the H2D staging and corrupt
    the prefill."""
    from types import SimpleNamespace

    from freetoken.layers.moe import _prefill_cpu_split

    off = SimpleNamespace(
        cpu_executor=None, prefill_fetch_fraction=0.5, prefill_cpu_mask=object(),
        prefill_pcie_frac_q16=lambda: 999,  # must never be read when off
    )
    assert _prefill_cpu_split(off, 10) == (None, Q)
    # fraction 1.0 -> the whole prefill stays on the PCIe plan
    staged = SimpleNamespace(
        cpu_executor=SimpleNamespace(max_tokens=64), prefill_fetch_fraction=1.0,
        prefill_cpu_mask=object(), prefill_pcie_frac_q16=lambda: Q,
    )
    assert _prefill_cpu_split(staged, 10) == (None, Q)
    # the CPU mask is only built when the on-demand buffers exist
    no_mask = SimpleNamespace(
        cpu_executor=SimpleNamespace(max_tokens=64), prefill_fetch_fraction=0.5,
        prefill_cpu_mask=None, prefill_pcie_frac_q16=lambda: 999,
    )
    assert _prefill_cpu_split(no_mask, 10) == (None, Q)
    # a chunk longer than the executor buffers keeps the old behaviour, not a resize
    too_long = SimpleNamespace(
        cpu_executor=SimpleNamespace(max_tokens=4), prefill_fetch_fraction=0.5,
        prefill_cpu_mask=object(), prefill_pcie_frac_q16=lambda: 999,
    )
    assert _prefill_cpu_split(too_long, 10) == (None, Q)
    # all present: the chunk rides the CPU executor at the fraction's Q16 share
    executor = SimpleNamespace(max_tokens=64)
    on = SimpleNamespace(
        cpu_executor=executor, prefill_fetch_fraction=0.4, prefill_cpu_mask=object(),
        prefill_pcie_frac_q16=lambda: round(0.4 * Q),
    )
    assert _prefill_cpu_split(on, 10) == (executor, round(0.4 * Q))


def test_prefill_cpu_split_gates_big_chunks_off_the_cpu_side(monkeypatch):
    # The auto fraction is calibrated on decode-step bandwidth; an 8192-token chunk
    # hands ~82% of its misses to the CPU where each miss carries ~344 routes --
    # measured ~79 s/chunk vs ~7.3 s pure-PCIe (65k prompt: 630 s vs 58 s). The gate
    # keeps full-size chunks on the PCIe plan and the measured win for short prefills.
    from types import SimpleNamespace

    import freetoken.layers.moe as moe_layers

    executor = SimpleNamespace(max_tokens=8192)
    cache = SimpleNamespace(
        cpu_executor=executor, prefill_fetch_fraction=0.176, prefill_cpu_mask=object(),
        prefill_pcie_frac_q16=lambda: round(0.176 * Q),
    )
    frac_q16 = round(0.176 * Q)
    monkeypatch.setattr(moe_layers, "_HYBRID_PREFILL_MAX_TOKENS", 512)
    assert moe_layers._prefill_cpu_split(cache, 150) == (executor, frac_q16)
    assert moe_layers._prefill_cpu_split(cache, 512) == (executor, frac_q16)
    assert moe_layers._prefill_cpu_split(cache, 8192) == (None, Q)
    monkeypatch.setattr(moe_layers, "_HYBRID_PREFILL_MAX_TOKENS", 0)
    assert moe_layers._prefill_cpu_split(cache, 1) == (None, Q)
    monkeypatch.setattr(moe_layers, "_HYBRID_PREFILL_MAX_TOKENS", 1 << 30)
    assert moe_layers._prefill_cpu_split(cache, 8192) == (executor, frac_q16)


def test_fetch_fraction_prefers_the_overlapped_pair_and_tp_weights_the_cpu_term(tmp_path):
    """Per-layer rows: the contended pair wins, and under TP the CPU term is the rank's slice of the shared pool, weighted by hideability."""
    bf16 = layer_kind_key("bf16", {}, 2048, 768, 8, 12)
    nvfp4 = layer_kind_key("nvfp4", {}, 2048, 768, 8, 12)
    prof = {
        "version": PROFILE_VERSION,
        "gpu": {"name": "FAKE GPU"},
        "layer_kinds": {
            bf16: {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            nvfp4: {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                    "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    rows = load_layer_kinds(path=str(path))
    # tp=1 keeps the plain bandwidth balance -- the hideability weighting is calibrated on
    # a 2-rank rig and is deliberately not extrapolated to a single rank.
    assert fetch_fraction(rows[bf16]) == pytest.approx(0.4)
    assert fetch_fraction(rows[nvfp4]) == pytest.approx(0.25)
    # under TP the shared CPU pool is split (tp^0.75) and the CPU term weighted
    share, hide = 2 ** 0.75, _CPU_HIDEABILITY
    assert fetch_fraction(rows[bf16], cpu_share=share, hide=hide) == pytest.approx(
        40.0 / (100.0 / share * hide))
    assert fetch_fraction(rows[nvfp4], cpu_share=share, hide=hide) == pytest.approx(
        30.0 / (30.0 + 90.0 / share * hide))
    # a format with no row is left to the fixed cap of 1
    assert fetch_fraction(rows.get(layer_kind_key("mxfp4_triton", {}, 2048, 768, 8, 12))) is None
    # a profile from different hardware is ignored
    assert load_layer_kinds(gpu_name="OTHER", path=str(path)) is None


def test_hideability_weighting_fetches_less_than_a_pure_bandwidth_balance(tmp_path):
    """Equal measured bandwidths are not equal cost: the gather holds SMs on the critical
    path while the CPU overflow hides under the GPU's dense work, so the split must lean
    further onto the CPU than a raw pcie:cpu balance would."""
    bf16 = layer_kind_key("bf16", {}, 2048, 768, 8, 12)
    prof = {
        "version": PROFILE_VERSION,
        "gpu": {"name": "FAKE GPU"},
        # deliberately equal rates: an unweighted balance would fetch exactly half
        "layer_kinds": {bf16: {"cpu_moe_gbs": 50.0, "pcie_gather_gbs": 50.0}},
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    row = load_layer_kinds(path=str(path))[bf16]
    c = 50.0 / 2 ** 0.75
    # the production path: a live per-rank link measurement, so pcie/(pcie + cpu_eff)
    frac = fetch_fraction(row, cpu_share=2 ** 0.75, hide=_CPU_HIDEABILITY, pcie_gbs=50.0)
    assert frac == pytest.approx(50.0 / (50.0 + c * _CPU_HIDEABILITY))
    assert frac < 0.5
    assert _CPU_HIDEABILITY > 1.0
    # a single rank is NOT extrapolated to: it keeps the unweighted balance, and its CPU
    # slice is the whole pool (cpu_share == 1), so equal rates balance at exactly half
    assert fetch_fraction(row, pcie_gbs=50.0) == pytest.approx(0.5)


def test_the_hideability_shift_is_bounded_relative_to_the_measured_balance(tmp_path):
    """The factor is calibrated on ONE rig, so it may not drag the split arbitrarily far from
    what this machine's own measured bandwidths say. The bound is relative on purpose: an
    absolute floor would clip a slow link's legitimate operating point (a gen4 x4 rank
    resolves to ~0.14, and a 0.10 floor starts biting once the probe reads 6 GB/s)."""
    from freetoken.moe.bench_profile import _bound_hideability

    assert 0.0 < _HIDEABILITY_MAX_SHIFT < 1.0
    # a weight of H shifts the split by at most 1/H, so today's 2.2 (-> 0.4545) stays slack
    assert 1.0 / _CPU_HIDEABILITY > _HIDEABILITY_MAX_SHIFT
    # slack: the weighted value passes through untouched
    assert _bound_hideability(0.145, 0.272) == pytest.approx(0.145)
    # a runaway weight is clamped to the relative floor
    assert _bound_hideability(0.05, 0.5) == pytest.approx(_HIDEABILITY_MAX_SHIFT * 0.5)
    # never above 1, and never raises the fetched share above the unweighted balance
    assert _bound_hideability(1.4, 1.8) == 1.0
    assert _bound_hideability(0.3, 0.2) == pytest.approx(0.3)

    # ... and through the public function the calibration rig's slow rank stays slack:
    # cpu_eff/pcie ~ 2.7 is what a gen4 x4 link beside a 32 GB/s CPU pool actually measures.
    bf16 = layer_kind_key("bf16", {}, 2048, 768, 8, 12)
    prof = {"version": PROFILE_VERSION, "gpu": {"name": "FAKE GPU"},
            "layer_kinds": {bf16: {"cpu_moe_gbs": 19.02 * 2 ** 0.75,
                                   "pcie_gather_gbs": 19.02 * 2 ** 0.75}}}
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    row = load_layer_kinds(path=str(path))[bf16]
    assert fetch_fraction(row, cpu_share=2 ** 0.75, hide=_CPU_HIDEABILITY, pcie_gbs=7.1) == pytest.approx(
        7.1 / (7.1 + 19.02 * _CPU_HIDEABILITY))


def test_fetch_fraction_splits_the_cpu_pool_under_tp(tmp_path):
    """Under TP every rank keeps its private PCIe link but shares ONE RAM pool, so the
    per-rank CPU term shrinks (sublinearly: disjoint core slices of a RAM-bound pool each
    keep most of the stream rate -> tp^0.75) and the balanced split fetches MORE over PCIe."""
    bf16 = layer_kind_key("bf16", {}, 2048, 768, 8, 12)
    nvfp4 = layer_kind_key("nvfp4", {}, 2048, 768, 8, 12)
    prof = {
        "version": PROFILE_VERSION,
        "gpu": {"name": "FAKE GPU"},
        "layer_kinds": {
            bf16: {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            nvfp4: {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                    "cpu_moe_overlap_gbs": 80.0, "pcie_gather_overlap_gbs": 20.0},
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    rows = load_layer_kinds(path=str(path))
    hide, share = _CPU_HIDEABILITY, 2 ** 0.75
    # tp=1: no peer rank waits at an all-reduce, so the hideability weighting does not
    # apply and the split is the plain bandwidth balance.
    assert fetch_fraction(rows[bf16]) == pytest.approx(0.4)
    assert fetch_fraction(rows[nvfp4]) == pytest.approx(0.2)
    # tp=2: standalone -> pcie / (cpu / 2^0.75 x hide); overlapped -> pcie_ov / (pcie_ov + cpu_ov / 2^0.75 x hide)
    assert fetch_fraction(rows[bf16], cpu_share=share, hide=hide) == pytest.approx(
        40.0 / (100.0 / share * hide))
    assert fetch_fraction(rows[nvfp4], cpu_share=share, hide=hide) == pytest.approx(
        20.0 / (20.0 + 80.0 / share * hide))
    # The tp^0.75 CPU-slice term still pushes tp=2 toward PCIe on its own: the same
    # weighting against an UNSHARED pool (tp=1 arithmetic) would fetch less, since each
    # rank's slice of the one machine-wide RAM pool is smaller.
    unshared = 40.0 / (100.0 * hide)
    assert fetch_fraction(rows[bf16], cpu_share=share, hide=hide) > unshared


def test_profile_lookup_prefers_the_gpu_uuid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BENCHBW_PATH", raising=False)
    uuid = "GPU-2f3a9b1c-0000-1111-2222-333344445555"

    def write(path, name, row):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"version": PROFILE_VERSION, "gpu": {"name": name}, "layer_kinds": {"k": row}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "legacy")
    assert load_layer_kinds(gpu_name="FAKE GPU", gpu_uuid=uuid) == {"k": "legacy"}
    assert load_layer_kinds(gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "card")
    assert load_layer_kinds(gpu_name="FAKE GPU", gpu_uuid=uuid) == {"k": "card"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fraction_gpu_matches_cpu_reference():
    torch.manual_seed(0)
    num_experts, cache_size, top_k, frac = 32, 40, 8, 0.415

    def make():
        # one entry so every layer's fraction is the same,
        # i.e. the case the legacy single-float field used to express
        return OffloadMoeCache(
            num_layers=2, num_experts=num_experts, cache_size=cache_size,
            device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=num_experts, hybrid_fetch_fraction=[frac, frac],
        )

    gpu, ref = make(), make()
    frac_q16 = round(frac * Q)
    for step in range(64):
        ids = torch.randperm(num_experts)[:top_k].to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()  # a CPU ids tensor drives the reference path
        gpu.ensure_experts_hybrid(0, g, torch.ones_like(g, dtype=torch.float32))
        ref.ensure_experts_hybrid(0, c, torch.ones_like(c, dtype=torch.float32))
        missing = int(gpu.num_missing_full.item())
        fetched = int(gpu.num_indices.item())
        assert missing == int(ref.num_missing_full.item())
        assert fetched == int(ref.num_indices.item()) == _balanced_fetch(missing, frac_q16)
        # slot rewrites (hit/fetched -> slot, overflow -> -1) and LRU state stay identical
        assert torch.equal(g.cpu(), c)
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        assert torch.equal(gpu.id_of_slot.cpu(), ref.id_of_slot.cpu())
        assert (g >= 0).sum().item() == len(set(ids.tolist())) - (missing - fetched)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fixed_cap_unchanged():
    # fraction 0 (no profile / explicit --moe-hybrid-max-fetch) keeps the fixed cap.
    cache = OffloadMoeCache(
        num_layers=1, num_experts=32, cache_size=40, device=torch.device("cuda"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=1,
        hybrid_fetch_fraction=[0.0],
    )
    ids = torch.arange(8, dtype=torch.int32).cuda()
    cache.ensure_experts_hybrid(0, ids, torch.ones_like(ids, dtype=torch.float32))
    assert int(cache.num_missing_full.item()) == 8
    assert int(cache.num_indices.item()) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_min_bs_routes_small_batches_to_the_gpu_path():
    """hybrid_min_bs keeps small decode batches on the GPU slot-cache path.

    A warm bs=1 step misses ~1 expert/layer: the per-layer CPU submit/sync handshake
    costs more than the PCIe it saves, so the engine raises the floor to 2 under TP and
    the layer dispatch (captured per batch size into the decode graphs) must honor it.
    """
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer

    if try_get_tp_info() is None:
        set_tp_info(0, 1)
    cache = OffloadMoeCache(
        num_layers=1, num_experts=8, cache_size=16, device=torch.device("cuda"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=1, hybrid_min_bs=2,
    )
    layer = OffloadMoELayer(
        layer_id=0, num_experts=8, top_k=2, hidden_size=4, intermediate_size=8,
        strategy="hybrid", decode_target="hybrid",
    )
    layer.offload_cache = cache
    took = []
    layer._decode_hybrid = lambda c, h, w, i: took.append("hybrid")
    cache.ensure_experts = lambda lid, ids: took.append("ensure")
    cache.copy_missing = lambda: took.append("copy")
    cache.pool_bank_views = lambda lid: ()
    cache.alphas_for_slots = lambda lid: None
    layer._expert_gemm = lambda *a, **k: took.append("gemm")

    dev = torch.device("cuda")
    w = torch.zeros(1, 2, device=dev)
    ids = torch.zeros(1, 2, dtype=torch.int32, device=dev)
    h1 = torch.zeros(1, 4, device=dev, dtype=torch.bfloat16)
    layer._decode_routed(h1, w, ids)
    assert took == ["ensure", "copy", "gemm"]  # bs=1 < hybrid_min_bs -> GPU slot path

    took.clear()
    cache.hybrid_min_bs = 1  # historical default: hybrid serves every batch size
    layer._decode_routed(h1, w, ids)
    assert took == ["hybrid"]

    took.clear()
    cache.hybrid_min_bs = 2
    h2 = torch.zeros(2, 4, device=dev, dtype=torch.bfloat16)
    layer._decode_routed(h2, torch.zeros(2, 2, device=dev),
                         torch.zeros(2, 2, dtype=torch.int32, device=dev))
    assert took == ["hybrid"]  # bs=2 >= hybrid_min_bs -> CPU overflow path


def test_a_measured_link_replaces_the_profiled_pcie_term(tmp_path):
    """A live per-rank link measurement beats the cached PCIe term.

    The profile's number is one machine-wide sample: it cannot say that rank 0 sits in a
    gen3 x16 slot and rank 1 in a gen4 x4 one, and its contended-overlap variant measures
    the gather starved by a 24-thread CPU pool, which understates the decode-time rate.
    """
    nvfp4 = layer_kind_key("nvfp4", {}, 2048, 768, 8, 12)
    prof = {
        "version": PROFILE_VERSION,
        "gpu": {"name": "FAKE GPU"},
        "layer_kinds": {nvfp4: {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                                "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0}},
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    row = load_layer_kinds(path=str(path))[nvfp4]
    hide, share = _CPU_HIDEABILITY, 2 ** 0.75
    # no measurement: the contended pair still decides (tp=1, so unweighted)
    assert fetch_fraction(row) == pytest.approx(0.25)
    # measured link vs the standalone CPU rate, split per rank by tp_size
    assert fetch_fraction(row, cpu_share=share, hide=hide, pcie_gbs=12.5) == pytest.approx(
        12.5 / (12.5 + 100.0 / share * hide))
    # a slower link in the other slot fetches less over PCIe, from the same profile
    assert fetch_fraction(row, cpu_share=share, hide=hide, pcie_gbs=7.0) < fetch_fraction(
        row, cpu_share=share, hide=hide, pcie_gbs=12.5)
    # a non-positive measurement is ignored, not trusted as "no PCIe"
    assert fetch_fraction(row, pcie_gbs=0.0) == pytest.approx(0.25)
