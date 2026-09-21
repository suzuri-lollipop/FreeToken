"""Env-gated MoE diagnostics (`FREETOKEN_MOE_STATS=1`).

The one computation in ``_debug_stats`` that is worth pinning is the decode duplicate-route
histogram: it feeds the batch-dedup tuning, so a wrong pair count is a wrong number that
still looks like a number. It is off by default, so each probe is driven directly (a
duck-typed cache + a set ``_dec``) instead of going through the env gate, and the env toggles
that gate at runtime are exercised too.
"""

from types import SimpleNamespace

import torch

import freetoken.moe._debug_stats as ds


def _probe(num_experts: int = 8) -> ds.MoEProbe:
    p = ds.MoEProbe.__new__(ds.MoEProbe)  # __init__ gates on the env; build the internals
    p.cache = SimpleNamespace(num_experts=num_experts)
    p._dec = torch.zeros(8, dtype=torch.int64)
    p._host = {}
    p._host_slow = []
    return p


# ------------------------------------------------------------- decode route accounting

def test_dup_route_histogram_counts_pairs_and_masks_out_gpu_assigned(monkeypatch):
    """Five routes landing on one expert are C(5,2)=10 duplicate pairs; a ``-1`` (GPU-assigned)
    route clamps onto expert 0 in the histogram but is zeroed by its mask, so it must neither
    add a pair nor count as a CPU route."""
    monkeypatch.setattr(ds, "ENABLED", True)
    monkeypatch.setenv("FREETOKEN_MOE_STATS_HYBRID_PROBE", "1")
    p = _probe(8)
    ids = torch.tensor([[2], [2], [2], [2], [2], [-1]])
    p.decode_step(ids, ids.clone())
    assert p._dec[3].item() == 10  # duplicate pairs
    assert p._dec[4].item() == 5  # CPU-assigned routes (the -1 is masked)
    assert p._dec[5].item() == 1  # layer count
    # a second layer whose routes are all distinct adds no pairs
    p.decode_step(torch.tensor([[0], [1], [3]]), torch.tensor([[0], [1], [3]]))
    assert p._dec[3].item() == 10
    assert p._dec[4].item() == 8
    assert p._dec[5].item() == 2


def test_gpu_path_counts_routes_only_without_a_histogram(monkeypatch):
    """cpu_ids None (pure GPU path) never touches the dup-pair counter, only the route count
    of valid (>= 0) routes; a regression that histogrammed here would count pairs that the
    kernel cannot dedup."""
    monkeypatch.setattr(ds, "ENABLED", True)
    p = _probe(8)
    p.decode_step(torch.tensor([[0], [2], [2], [-1]]), None)
    assert p._dec[4].item() == 3  # >= 0 routes
    assert p._dec[5].item() == 1
    assert p._dec[3].item() == 0  # no histogram on the GPU path


def test_decode_is_a_noop_when_stats_are_disabled(monkeypatch):
    monkeypatch.setattr(ds, "ENABLED", False)
    p = _probe(8)
    p.decode_step(torch.tensor([[2], [2]]), torch.tensor([[2], [2]]))
    assert p._dec[3].item() == 0
    assert p._dec[5].item() == 0


# ------------------------------------------------------------- host-side phase timings

def test_host_phase_averages_per_call_and_trims_the_slow_tail():
    p = _probe()
    for _ in range(3):
        p.host_phase("prefill", 0.002)  # 6ms total -> 2ms/call, not slow
    for _ in range(2):
        p.host_phase("ensure", 0.012)  # > 8ms -> the slow-capture tail
    out = p.host_dump()
    assert "prefill=2.00ms x3" in out
    assert "ensure=12.00ms x2" in out
    assert "slow:" in out and "ensure@12ms" in out
    # the dump clears the accumulators so a periodic dump never double-counts
    assert p._host == {} and p._host_slow == []


def test_host_dump_omits_the_slow_tail_when_no_phase_runs_long():
    p = _probe()
    p.host_phase("decode", 0.001)
    out = p.host_dump()
    assert out == "decode=1.00ms x1"
    assert "slow:" not in out