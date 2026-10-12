"""The expert profile: format round trip, ranking, and budget prefixes."""

from __future__ import annotations

import pytest

from freetoken.moe.expert_profile import (
    load_profile,
    rank_by_counts,
    round_robin,
    take_prefix_until_budget,
    write_profile,
)


def test_round_robin_prefix_spreads_across_layers():
    ranked = round_robin(4, 3, first_layer=1)
    # expert-major, layer-ascending: holding k pairs gives every layer ceil(k/3)
    assert ranked == [(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)]
    layers = {l for l, _ in ranked[:5]}
    assert layers == {1, 2, 3}


def test_rank_by_counts_orders_by_descending_usage():
    counts = [[0, 5, 2], [7, 0, 1]]
    ranked = rank_by_counts(counts)
    assert ranked[:3] == [(1, 0), (0, 1), (0, 2)]
    # count 1 ranks above the never-routed tail, which keeps (layer, expert) order
    assert ranked[3:] == [(1, 2), (0, 0), (1, 1)]


def test_profile_round_trip_and_validation(tmp_path):
    ranked = [(0, 3), (1, 1), (2, 2)]
    path = tmp_path / "p.bin"
    write_profile(path, ranked, num_layers=3, num_experts=4)
    loaded, num_layers, num_experts = load_profile(path)
    assert loaded == ranked
    assert (num_layers, num_experts) == (3, 4)


def test_write_profile_rejects_out_of_range_pairs(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        write_profile(tmp_path / "p.bin", [(3, 0)], num_layers=3, num_experts=4)


def test_load_profile_rejects_bad_magic_and_duplicates(tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"XXXX" + b"\x00" * 24)
    with pytest.raises(ValueError, match="magic"):
        load_profile(bad)
    import struct

    dup = tmp_path / "dup.bin"
    dup.write_bytes(
        struct.pack("<4sIIII", b"FTXP", 1, 1, 4, 2) + struct.pack("<II", 0, 0) + struct.pack("<II", 0, 0)
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_profile(dup)


def test_take_prefix_until_budget_is_exact():
    ranked = round_robin(2, 3)
    taken = take_prefix_until_budget(ranked, bytes_per_expert=10, budget=25)
    assert len(taken) == 2
    assert take_prefix_until_budget(ranked, bytes_per_expert=10, budget=0) == []
    with pytest.raises(ValueError):
        take_prefix_until_budget(ranked, bytes_per_expert=0, budget=10)
