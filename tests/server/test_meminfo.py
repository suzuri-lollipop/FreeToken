"""Tests for the /v1/meminfo VRAM breakdown (build_meminfo) and the /meminfo dashboard.

build_meminfo is a pure read over a FrontendManager-like state: the per-rank readiness
("meta", ...) payloads (device, weights, pool allocation, unit costs) summed into fleet
totals, with the rank0 stats tracker supplying the live gauges. These pin the per-rank and
fleet arithmetic, the rank0-only live/residual scoping, the never-raises contract on a bare
state and the flat single-rank fallback, and that both routes answer through the real wiring.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from freetoken.server.control_api import register_control_routes
from freetoken.server.meminfo import MEMINFO_HTML, build_meminfo

MIB = 1024**2
GIB = 1024**3

_KV_TOK = 512          # kv bytes per token, per rank
_MOE_SLOT = 3 * MIB    # moe bytes per slot, per rank
_MAMBA_SLOT = 8 * MIB  # mamba bytes per slot, per rank
_PAGE = 64


def _meta(rank, gpu_index, pages, moe, mamba):
    """One rank's raw readiness payload (as _on_meta stores it, pre-pop). Asymmetric by
    design: rank1 gets fewer KV pages and MORE moe slots, so sums must be real sums."""
    return {
        "rank": rank, "tp_size": 2,
        "gpus": [{"index": gpu_index, "name": f"RTX-{gpu_index}", "uuid": f"u{gpu_index}",
                  "total_bytes": 24 * GIB}],
        "weights_bytes": 4 * GIB,
        "device_total_bytes": 24 * GIB,
        "nonpool_overhead_bytes": 512 * MIB,
        "cache_budget_bytes": 16 * GIB,
        "free_vram_bytes": 20 * GIB,
        "pools": {"num_pages": pages, "page_size": _PAGE, "moe_cache_size": moe,
                  "num_mamba_slots": mamba, "num_swa_pages": 0, "swa_page_size": 1},
        "kv_bytes_per_token": _KV_TOK, "moe_bytes_per_expert": _MOE_SLOT,
        "mamba_bytes_per_slot": _MAMBA_SLOT, "swa_bytes_per_token": 0,
    }


_KV0, _MOE0, _MAN0 = 100, 200, 8   # rank0 unit counts
_KV1, _MOE1, _MAN1 = 90, 240, 8    # rank1 unit counts (uneven shards)


def _full_state():
    stats = SimpleNamespace(vram_bytes=10 * GIB, kv_used_pages=10, kv_total_pages=100,
                            kv_cached_pages=25, mamba_used_slots=2, mamba_total_slots=8,
                            mamba_cached_slots=1, moe_used_slots=50, moe_active_slots=10,
                            moe_total_slots=200)
    return SimpleNamespace(
        config=SimpleNamespace(served_model_name="m", page_size=_PAGE, tp_size=2),
        stats=stats,
        rank_metas={0: _meta(0, 0, _KV0, _MOE0, _MAN0), 1: _meta(1, 1, _KV1, _MOE1, _MAN1)},
        ready_at=None,
    )


def _pool(doc, key):
    return next(p for p in doc["pools"] if p["key"] == key)


def test_totals_sum_every_rank():
    doc = build_meminfo(_full_state())
    t = doc["totals"]
    kv = (_KV0 + _KV1) * _PAGE * _KV_TOK
    moe = (_MOE0 + _MOE1) * _MOE_SLOT
    mamba = (_MAN0 + _MAN1) * _MAMBA_SLOT
    assert doc["tp_size"] == 2
    assert len(doc["ranks"]) == 2
    assert t["device_total_bytes"] == 48 * GIB
    assert t["weights_bytes"] == 8 * GIB
    assert t["overhead_bytes"] == 1 * GIB
    assert t["cache_budget_bytes"] == 32 * GIB
    assert t["free_after_weights_bytes"] == 40 * GIB
    assert t["pools_bytes"] == kv + moe + mamba
    assert t["allocated_bytes"] == t["weights_bytes"] + t["pools_bytes"] + t["overhead_bytes"]
    # Fleet donut closes: residual + free + allocated == device total, exactly.
    assert t["used_est_bytes"] == t["allocated_bytes"] + t["residual_bytes"]
    assert t["used_est_bytes"] + t["free_est_bytes"] == t["device_total_bytes"]


def test_pools_report_allocated_and_used():
    doc = build_meminfo(_full_state())
    kv, moe, mamba = _pool(doc, "kv"), _pool(doc, "moe"), _pool(doc, "mamba")
    assert kv["bytes"] == (_KV0 + _KV1) * _PAGE * _KV_TOK
    # Units stay the per-replica logical count (max over ranks); unit cost sums the shards.
    assert kv["units"] == _KV0 * _PAGE and kv["unit_bytes"] == 2 * _KV_TOK
    # Rank0 gauge prices occupancy (kept + evictable cache) at the summed allocation. KV kept
    # 10/100 pages and warm cache 25/100 pages -> 35% occupancy, split into pinned + cached.
    assert kv["used_bytes"] == round(kv["bytes"] * 35 / 100)
    assert kv["pinned_bytes"] == round(kv["bytes"] * 10 / 100)
    assert kv["cached_bytes"] == round(kv["bytes"] * 25 / 100)
    assert kv["used_bytes"] == kv["pinned_bytes"] + kv["cached_bytes"]
    assert kv["used_mode"] == "rank0"
    # Same fraction logic for mamba (kept 2/8 + cache 1/8 -> 37.5%).
    assert mamba["used_bytes"] == round(mamba["bytes"] * 3 / 8)
    assert mamba["cached_bytes"] == round(mamba["bytes"] * 1 / 8)
    assert mamba["det_mode"] == "split"
    # MoE: rank0 filled 50/200 slots and the last forward read 10 of them -> the fill splits
    # like the others (working set + warm rest), never as an exclusively held bar.
    assert moe["det_mode"] == "working_set" and moe["used_mode"] == "rank0"
    assert moe["pinned_bytes"] == round(moe["bytes"] * 10 / 200)
    assert moe["cached_bytes"] == round(moe["bytes"] * 40 / 200)
    assert moe["used_bytes"] == moe["pinned_bytes"] + moe["cached_bytes"]
    # Absent pools (swa here) are omitted, not printed as a fake 0.
    assert all(p["key"] != "swa" for p in doc["pools"])


def test_occupancy_clamps_to_allocation():
    # A full cache on top of kept content cannot report more than the pool physically holds.
    st = _full_state()
    st.stats = SimpleNamespace(vram_bytes=10 * GIB, kv_used_pages=80, kv_total_pages=100,
                               kv_cached_pages=90, mamba_used_slots=0, mamba_total_slots=8)
    kv = _pool(build_meminfo(st), "kv")
    assert kv["used_bytes"] == kv["bytes"]
    assert kv["pinned_bytes"] + kv["cached_bytes"] > kv["bytes"]


def test_live_and_residual_are_rank0_scoped():
    doc = build_meminfo(_full_state())
    r0 = next(r for r in doc["ranks"] if r["rank"] == 0)
    r1 = next(r for r in doc["ranks"] if r["rank"] == 1)
    assert r0["live_bytes"] == 10 * GIB
    assert doc["live"]["rank0_vram_bytes"] == 10 * GIB
    assert r1["live_bytes"] is None  # no live channel exists for other ranks
    assert doc["live"]["kv_cached_pages"] == 25 and doc["live"]["mamba_cached_slots"] == 1
    rank0_alloc = 4 * GIB + 512 * MIB + _KV0 * _PAGE * _KV_TOK + _MOE0 * _MOE_SLOT + _MAN0 * _MAMBA_SLOT
    assert r0["residual_bytes"] == 10 * GIB - rank0_alloc
    assert r1["residual_bytes"] == 0  # unmeasured, never a fabricated remainder
    assert r0["pools"]["moe"] == _MOE0 * _MOE_SLOT  # the per-GPU table row stays rank-local
    assert r0["gpus"][0]["name"] == "RTX-0" and r1["gpus"][0]["name"] == "RTX-1"


def test_usage_gauges_none_when_never_sampled():
    st = _full_state()
    st.stats = SimpleNamespace(vram_bytes=10 * GIB)  # tracker never fed a pool reply
    doc = build_meminfo(st)
    assert _pool(doc, "kv")["used_bytes"] is None
    assert _pool(doc, "mamba")["used_bytes"] is None
    assert _pool(doc, "moe")["used_bytes"] is None  # dense model / cache disabled
    # A tracker predating the cache gauges: occupancy degrades to kept-only, never to 0.
    st.stats = SimpleNamespace(vram_bytes=10 * GIB, kv_used_pages=10, kv_total_pages=100)
    kv = _pool(build_meminfo(st), "kv")
    assert kv["cached_bytes"] == 0 and kv["used_bytes"] == kv["pinned_bytes"]
    # A backend that reports no MoE working set: the whole fill shows as warm cache.
    st.stats = SimpleNamespace(vram_bytes=10 * GIB, moe_used_slots=50, moe_total_slots=200)
    moe = _pool(build_meminfo(st), "moe")
    assert moe["pinned_bytes"] == 0 and moe["cached_bytes"] == round(moe["bytes"] * 50 / 200)
    # An oversized active gauge (stale sample) can never out-report the fill inside it.
    st.stats = SimpleNamespace(vram_bytes=10 * GIB, moe_used_slots=50, moe_active_slots=90,
                               moe_total_slots=200)
    moe = _pool(build_meminfo(st), "moe")
    assert moe["pinned_bytes"] == round(moe["bytes"] * 50 / 200) and moe["cached_bytes"] == 0


def test_live_gauges_carry_the_moe_split():
    live = build_meminfo(_full_state())["live"]
    assert (live["moe_used_slots"], live["moe_active_slots"], live["moe_total_slots"]) == (
        50, 10, 200)


def test_flat_fallback_and_bare_state_never_raise():
    flat = SimpleNamespace(
        config=SimpleNamespace(served_model_name="m", page_size=_PAGE, tp_size=1),
        stats=SimpleNamespace(vram_bytes=6 * GIB),
        unit_bytes={"kv_bytes_per_token": _KV_TOK, "moe_bytes_per_expert": _MOE_SLOT,
                    "mamba_bytes_per_slot": _MAMBA_SLOT, "swa_bytes_per_token": 0},
        cache_pools={"num_pages": _KV0, "page_size": _PAGE, "moe_cache_size": _MOE0,
                     "num_mamba_slots": _MAN0, "num_swa_pages": 0, "swa_page_size": 1},
        weights_bytes=4 * GIB, device_total_bytes=24 * GIB, nonpool_overhead_bytes=512 * MIB,
        cache_budget_bytes=16 * GIB, free_vram_bytes=20 * GIB,
        gpus=[{"index": 0, "name": "GPU", "uuid": "u", "total_bytes": 24 * GIB}],
        ready_at=None,
    )
    doc = build_meminfo(flat)
    assert len(doc["ranks"]) == 1 and doc["ranks"][0]["rank"] == 0
    assert doc["totals"]["device_total_bytes"] == 24 * GIB
    assert doc["totals"]["weights_bytes"] == 4 * GIB

    bare = build_meminfo(SimpleNamespace())
    assert bare["totals"]["device_total_bytes"] == 0
    assert bare["pools"] == []
    assert bare["ranks"][0]["live_bytes"] is None
    assert bare["tp_size"] == 1


def test_routes_serve_json_and_html():
    app = FastAPI()
    register_control_routes(app, lambda: _full_state())
    client = TestClient(app)

    r = client.get("/v1/meminfo")
    assert r.status_code == 200
    doc = r.json()
    assert doc["model"] == "m"
    assert doc["totals"]["device_total_bytes"] == 48 * GIB
    assert any(p["key"] == "moe" for p in doc["pools"])

    r = client.get("/meminfo")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "FreeToken" in r.text and "/v1/meminfo" in MEMINFO_HTML
