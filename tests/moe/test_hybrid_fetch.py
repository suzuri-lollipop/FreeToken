"""Hybrid decode's bandwidth-matched fetch split.

Covers the two halves of --moe-hybrid-max-fetch auto: the profile reader that turns
`ft bench bw` kernel bandwidths into a fetch fraction, and the ensure kernel's
per-step integer split (GPU kernel vs CPU reference mirror, and the balance rule).
"""

import json
import os

import pytest
import torch

from freetoken.moe.bench_profile import default_profile_path, load_backend_recommendation, load_hybrid_fetch_fraction
from freetoken.moe.offload_cache import OffloadMoeCache

Q = 1 << 16


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


def test_load_hybrid_fetch_fraction(tmp_path):
    prof = {
        "gpu": {"name": "FAKE GPU"},
        "dtype_kernels": {
            "bf16": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            # overlapped (contended) pair wins over the standalone numbers when present
            "nvfp4_x": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                        "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
        "workloads": {
            "m": {"kernels": {"ds_fp4": {"cpu_moe_gbs": 80.0, "pcie_gather_gbs": 50.0}}}
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    # standalone fallback: full-contention assumption -> pcie / cpu
    assert load_hybrid_fetch_fraction("bf16", path=str(path)) == pytest.approx(0.4)
    # overlapped pair preferred: pcie_ov / (pcie_ov + cpu_ov)
    assert load_hybrid_fetch_fraction("nvfp4_x", path=str(path)) == pytest.approx(0.25)
    # per-model fallback when there is no per-dtype entry for the format
    assert load_hybrid_fetch_fraction("ds_fp4", path=str(path)) == pytest.approx(0.625)
    assert load_hybrid_fetch_fraction("nvfp4", path=str(path)) is None
    # a profile from different hardware is ignored
    assert load_hybrid_fetch_fraction("bf16", gpu_name="OTHER", path=str(path)) is None


def test_load_hybrid_fetch_fraction_splits_the_cpu_pool_under_tp(tmp_path):
    """Under TP every rank keeps its private PCIe link but shares ONE RAM pool, so the
    per-rank CPU term shrinks (sublinearly: disjoint core slices of a RAM-bound pool each
    keep most of the stream rate -> tp^0.75) and the balanced split fetches MORE over PCIe."""
    prof = {
        "gpu": {"name": "FAKE GPU"},
        "dtype_kernels": {
            "bf16": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            "nvfp4_x": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                        "cpu_moe_overlap_gbs": 80.0, "pcie_gather_overlap_gbs": 20.0},
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    # tp=1 keeps the historical single-rank values
    assert load_hybrid_fetch_fraction("bf16", path=str(path), tp_size=1) == pytest.approx(0.4)
    assert load_hybrid_fetch_fraction("nvfp4_x", path=str(path), tp_size=1) == pytest.approx(0.2)
    # tp=2: standalone -> pcie / (cpu / 2^0.75); overlapped -> pcie_ov / (pcie_ov + cpu_ov / 2^0.75)
    share = 2 ** 0.75
    assert load_hybrid_fetch_fraction("bf16", path=str(path), tp_size=2) == pytest.approx(
        40.0 / (100.0 / share))
    assert load_hybrid_fetch_fraction("nvfp4_x", path=str(path), tp_size=2) == pytest.approx(
        20.0 / (20.0 + 80.0 / share))
    # ... and always fetches at least as much as tp=1 (the CPU slice only got smaller)
    for fmt in ("bf16", "nvfp4_x"):
        assert load_hybrid_fetch_fraction(fmt, path=str(path), tp_size=2) > \
            load_hybrid_fetch_fraction(fmt, path=str(path), tp_size=1)


def test_profile_lookup_prefers_the_gpu_uuid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BENCHBW_PATH", raising=False)
    uuid = "GPU-2f3a9b1c-0000-1111-2222-333344445555"

    def write(path, name, verdict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"gpu": {"name": name}, "dtypes": {"bf16": verdict}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "hybrid")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "hybrid"
    assert load_backend_recommendation("bf16", gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "offload")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "offload"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fraction_gpu_matches_cpu_reference():
    torch.manual_seed(0)
    num_experts, cache_size, top_k, frac = 32, 40, 8, 0.415

    def make():
        return OffloadMoeCache(
            num_layers=2, num_experts=num_experts, cache_size=cache_size,
            device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=num_experts, hybrid_fetch_fraction=frac,
        )

    gpu, ref = make(), make()
    frac_q16 = round(frac * Q)
    for step in range(64):
        ids = torch.randperm(num_experts)[:top_k].to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()  # a CPU ids tensor drives the reference path
        gpu.ensure_experts_hybrid(0, g)
        ref.ensure_experts_hybrid(0, c)
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
    )
    ids = torch.arange(8, dtype=torch.int32).cuda()
    cache.ensure_experts_hybrid(0, ids)
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
    cache.bank_views = lambda n=None: ()
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
    prof = {
        "gpu": {"name": "FAKE GPU"},
        "dtype_kernels": {
            "nvfp4": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                      "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    # no measurement: the contended pair still decides
    assert load_hybrid_fetch_fraction("nvfp4", path=str(path)) == pytest.approx(0.25)
    # measured link vs the standalone CPU rate, split per rank by tp_size
    share = 2 ** 0.75
    assert load_hybrid_fetch_fraction("nvfp4", path=str(path), tp_size=2,
                                      pcie_gbs=12.5) == pytest.approx(12.5 / (12.5 + 100.0 / share))
    # a slower link in the other slot fetches less over PCIe, from the same profile
    assert load_hybrid_fetch_fraction("nvfp4", path=str(path), tp_size=2,
                                      pcie_gbs=7.0) < load_hybrid_fetch_fraction(
        "nvfp4", path=str(path), tp_size=2, pcie_gbs=12.5)
    # a non-positive measurement is ignored, not trusted as "no PCIe"
    assert load_hybrid_fetch_fraction("nvfp4", path=str(path), pcie_gbs=0.0) == pytest.approx(0.25)
