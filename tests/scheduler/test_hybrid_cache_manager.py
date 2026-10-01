"""P2b integration: CacheManager hybrid path (match_req -> cache_req donate -> prefix hit).
CPU, real LinearStatePool + page_table, hand-built Reqs. Exercises the two-currency wiring
without the full scheduler/engine."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.kvcache.linear_state_pool import LinearStatePool
from freetoken.models.config import LinearGatedDeltaGroupConfig
from freetoken.scheduler.cache import CacheManager


def _pool(num_slots=16):
    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    return LinearStatePool(group=g, num_slots=num_slots, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1)


def _pend(ids):
    # int32 to match production Req.input_ids dtype (fast_compare_key needs consistent dtype)
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids))


def test_hybrid_cache_manager_donate_then_hit():
    from freetoken.scheduler.prefill import ChunkedReq

    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    assert cm.is_hybrid

    # cold match on an empty tree
    mr = cm.match_req(_pend([1, 2, 3, 4, 5]))
    assert mr.cuda_handle.cached_len == 0 and mr.mamba_value is None

    # admit req A (MID-prefill: a ChunkedReq keeps its pair for the next chunk's track):
    # allocate live + ping-pong, stage KV pages, mark a ×N snapshot at boundary 4
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[0, :4] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    reqA = ChunkedReq(input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32), table_idx=0,
                      cached_len=4, output_len=1, uid=0, sampling_params=SamplingParams(),
                      cache_handle=mr.cuda_handle)
    reqA.linear_slot_idx, reqA.mamba_ping_pong = live, pp
    reqA.mamba_next_track_idx = 1            # flipped from 0 in build_fla_metadata; frozen = pp[0]
    reqA.mamba_last_track_seqlen = 4
    cm.lock(mr.cuda_handle)

    free_before = pool.num_free_slots
    cm.cache_req(reqA, finished=False)       # donate pp[0] at boundary 4; replace it in the pair
    # pp[0] donated to the tree; a fresh replacement was alloc'd -> net free-slot count unchanged
    assert pool.num_free_slots == free_before - 1  # one replacement alloc'd (donated slot now tree-owned)
    assert reqA.mamba_ping_pong[0] != pp[0]        # slot 0 replaced; pp[0] now lives in the tree

    # req B shares the [1,2,3,4] prefix -> HIT: returns the donated snapshot + reused KV
    mrB = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert mrB.cuda_handle.cached_len == 4
    assert mrB.mamba_value == pp[0]
    assert mrB.cuda_handle.get_matched_indices().tolist() == [100, 101, 102, 103]


def test_final_prefill_commit_returns_the_pair_to_the_reserve():
    """The FINAL chunk (plain Req) is the decode transition: donate the frozen track,
    then return the rest of the pair -- no replacement alloc, no parked slots."""
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    mr = cm.match_req(_pend([1, 2, 3, 4, 5]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[0, :4] = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32), table_idx=0,
              cached_len=4, output_len=1, uid=0, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    req.mamba_next_track_idx = 1             # frozen = pp[0]
    req.mamba_last_track_seqlen = 4
    cm.lock(mr.cuda_handle)

    free_before = pool.num_free_slots
    cm.cache_req(req, finished=False)        # final commit: donate + return the pair
    assert req.mamba_ping_pong is None       # nothing parked through decode
    assert pool.num_free_slots == free_before + 1   # pp[1] back; pp[0] tree-owned; no alloc

    # the donated snapshot is a tree hit for a prefix-sharing request
    mrB = cm.match_req(_pend([1, 2, 3, 4, 9]))
    assert mrB.mamba_value == pp[0]

    # an untracked final commit (no ×64 boundary crossed) returns the whole pair
    mr2 = cm.match_req(_pend([6, 7, 8, 9]))
    live2, pp2 = pool.alloc(1)[0], tuple(pool.alloc(2))
    req2 = Req(input_ids=torch.tensor([6, 7, 8, 9], dtype=torch.int32), table_idx=2,
               cached_len=3, output_len=1, uid=2, sampling_params=SamplingParams(),
               cache_handle=mr2.cuda_handle)
    req2.linear_slot_idx, req2.mamba_ping_pong = live2, pp2
    cm.lock(mr2.cuda_handle)
    free2 = pool.num_free_slots
    cm.cache_req(req2, finished=False)
    assert req2.mamba_ping_pong is None
    assert pool.num_free_slots == free2 + 2  # both slots back


def test_toolcall_anchor_borrows_a_single_slot_and_skips_when_tight():
    """Decode-time anchors borrow one slot from the shared reserve (a pending freeze
    is single-slot by design); a tight pool skips the borrow like the old None path."""
    pool = _pool(num_slots=12)
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    live = pool.alloc(1)[0]
    req = Req(input_ids=torch.arange(8, dtype=torch.int32), table_idx=0, cached_len=6,
              output_len=2, uid=0, sampling_params=SamplingParams(), cache_handle=None)
    req.linear_slot_idx = live
    req.mamba_ping_pong = None               # pair returned at the decode transition
    req.toolcall_anchor_len = 6              # anchor == cached_len: freeze now
    pool.conv_states[:, live] = 7.0          # distinctive live state to verify the copy

    cm.snapshot_toolcall_anchor([req])
    assert req.mamba_ping_pong is not None and len(req.mamba_ping_pong) == 1
    borrowed = req.mamba_ping_pong[0]
    assert req.mamba_last_track_seqlen == 6
    assert torch.all(pool.conv_states[:, borrowed] == 7.0)   # state frozen into the borrow

    # a second anchor while one is pending is a no-op (single pending freeze)
    before = pool.num_free_slots
    cm.snapshot_toolcall_anchor([req])
    assert pool.num_free_slots == before

    # tight pool: the borrow must not starve the admission gate's 3-slot set
    req2 = Req(input_ids=torch.arange(8, dtype=torch.int32), table_idx=1, cached_len=6,
               output_len=2, uid=1, sampling_params=SamplingParams(), cache_handle=None)
    req2.linear_slot_idx = pool.alloc(1)[0]
    req2.toolcall_anchor_len = 6
    held = []
    while pool.num_free_slots > 3:
        held.append(pool.alloc(1)[0])
    cm.snapshot_toolcall_anchor([req2])
    assert req2.mamba_ping_pong is None      # skipped gracefully
    pool.free(held)


def test_hybrid_finish_donates_live_slot():
    pool = _pool()
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    page_table[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1,
              cached_len=3, output_len=1, uid=1, sampling_params=SamplingParams(),
              cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)

    cm.cache_req(req, finished=True)         # donate the live slot directly (final state)
    # ping-pong pair freed; live slot kept (now owned by the tree)
    mr2 = cm.match_req(_pend([7, 8, 9, 10]))
    assert mr2.cuda_handle.cached_len == 3 and mr2.mamba_value == live


def test_free_req_slots_idempotent():
    """C2: a finish/abort double-free of the same request must NOT push its GDN slots twice."""
    pool = _pool()
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    req = Req(input_ids=torch.tensor([1, 2, 3], dtype=torch.int32), table_idx=0, cached_len=2,
              output_len=1, uid=0, sampling_params=SamplingParams(), cache_handle=None)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    base = pool.num_free_slots
    cm._free_req_slots(req)
    assert pool.num_free_slots == base + 3        # live + 2 ping-pong returned once
    cm._free_req_slots(req)                        # second free (abort/finish race)
    assert pool.num_free_slots == base + 3         # idempotent: nothing pushed twice


def test_rebuild_reclaims_donated_gdn_slots():
    """C5: a runtime cache rebuild must return the discarded tree's GDN snapshot slots (idle)."""
    pool = _pool(num_slots=16)
    pt = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, pt, "hybrid_radix", linear_state_pool=pool)
    mr = cm.match_req(_pend([7, 8, 9, 10]))
    live, pp = pool.alloc(1)[0], tuple(pool.alloc(2))
    pt[1, :3] = torch.tensor([200, 201, 202], dtype=torch.int32)
    req = Req(input_ids=torch.tensor([7, 8, 9, 10], dtype=torch.int32), table_idx=1, cached_len=3,
              output_len=1, uid=1, sampling_params=SamplingParams(), cache_handle=mr.cuda_handle)
    req.linear_slot_idx, req.mamba_ping_pong = live, pp
    cm.lock(mr.cuda_handle)
    cm.cache_req(req, finished=True)              # donates `live` to the tree, frees ping-pong
    assert pool.num_free_slots < pool.num_slots - 1   # a slot is now tree-owned
    cm.rebuild(64, pt)                            # idle rebuild discards the tree
    assert pool.num_free_slots == pool.num_slots - 1  # all GDN slots reclaimed (no leak)


def test_prefill_chunk_ends_on_a_page_boundary():
    """A hybrid chunk must end page-aligned: the snapshot commit skips any other boundary."""
    from freetoken.scheduler.prefill import ChunkedReq, PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pool = _pool()
    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "hybrid_radix", linear_state_pool=pool)
    assert cm.prefill_chunk_align == 64
    tm = TableManager(max_running_reqs=4, page_table=pt)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))

    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    req = adder.try_add_one(pending)
    assert isinstance(req, ChunkedReq) and req.extend_len == 64

    # a budget below one page keeps the unaligned chunk rather than stalling the request
    adder = PrefillAdder(token_budget=40, reserved_size=0, cache_manager=cm, table_manager=tm)
    assert adder.try_add_one(pending).extend_len == 40


def test_naive_cache_does_not_align_prefill_chunks():
    """The alignment hook is hybrid-only; every other cache keeps the raw budget chunk."""
    from freetoken.scheduler.prefill import PrefillAdder
    from freetoken.scheduler.table import TableManager
    from freetoken.scheduler.utils import PendingReq

    pt = torch.zeros(4, 512, dtype=torch.int32)
    cm = CacheManager(64, 64, pt, "radix")
    assert cm.prefill_chunk_align == 1
    tm = TableManager(max_running_reqs=4, page_table=pt)
    adder = PrefillAdder(token_budget=100, reserved_size=0, cache_manager=cm, table_manager=tm)
    pending = PendingReq(0, torch.arange(300, dtype=torch.int32), SamplingParams(max_tokens=1))
    assert adder.try_add_one(pending).extend_len == 100


def test_pool_sizing_covers_the_admission_floor():
    """C6: pool must reserve the deadlock floor (2 slots per running request for
    live+committed, one 3-slot admission set, the padding sink) even at a tiny
    ratio -- the chunk-track pair rides a shared reserve now, returned at the
    decode transition."""
    from types import SimpleNamespace
    from freetoken.kvcache.linear_state_pool import (
        _linear_pool_min_slots,
        _linear_pool_num_slots,
    )
    for mr in (1, 8, 64):
        c = SimpleNamespace(max_running_req=mr, cache_type="hybrid_radix",
                            linear_state_cache_ratio=0.1)
        assert _linear_pool_min_slots(c) == 2 * mr + 4
        assert _linear_pool_num_slots(c) >= _linear_pool_min_slots(c), (
            mr, _linear_pool_num_slots(c))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")


def test_copy_from_roundtrips_a_spec_snapshot_including_slot_states():
    """The spec-verify rollback contract: on a reject copy_from(scratch -> live) restores
    the post-row-0 state the GDN/PLE ops saved mid-forward, and the sibling slot_states
    (the PLE n-gram window rides them) roll back with the same call."""
    from freetoken.models.config import SlotStateSpec

    g = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 1), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    specs = (SlotStateSpec(name="ple_ngram_ctx", shape=(2,), dtype=torch.int32,
                           fill_value=-1),)
    pool = LinearStatePool(group=g, num_slots=8, dtype=torch.bfloat16,
                           device=torch.device("cpu"), tp_size=1, slot_states=specs)
    live, scratch = pool.alloc(2)
    pool.conv_states[:, live].normal_(generator=torch.Generator().manual_seed(1))
    pool.recurrent_states[:, live].normal_(generator=torch.Generator().manual_seed(2))
    pool.slot_state("ple_ngram_ctx")[live] = torch.tensor([7, 9], dtype=torch.int32)
    snap_conv = pool.conv_states[:, live].clone()
    snap_rec = pool.recurrent_states[:, live].clone()

    pool.copy_from(live, scratch)
    # the verify step advances the live slot by two tokens; the draft row pollutes it
    pool.conv_states[:, live].add_(1.0)
    pool.recurrent_states[:, live].mul_(2.0)
    pool.slot_state("ple_ngram_ctx")[live] = torch.tensor([9, 424242], dtype=torch.int32)

    pool.copy_from(scratch, live)  # reject -> restore
    assert torch.equal(pool.conv_states[:, live], snap_conv)
    assert torch.equal(pool.recurrent_states[:, live], snap_rec)
    assert torch.equal(pool.slot_state("ple_ngram_ctx")[live],
                       torch.tensor([7, 9], dtype=torch.int32))


def test_req_spec_fields_default_to_disabled():
    req = Req(input_ids=torch.tensor([1, 2, 3], dtype=torch.int32), table_idx=0,
              cached_len=0, output_len=1, uid=0, sampling_params=SamplingParams(),
              cache_handle=None)
    assert req.spec_slot_idx is None
    assert req.spec_residual is None
    assert req.spec_draft is None


def test_batch_spec_fields_default_to_regular():
    from freetoken.core import Batch

    b = Batch(reqs=[], phase="decode")
    assert b.spec_mode is None
    assert b.spec_draft_id == -1
    assert b.spec_pages is None
    assert b.spec_prologue is None


def test_allocate_paged_record_and_release_roundtrip():
    """Spec-batch page accounting: a recorded charge is exactly reversible (the reject
    path releases, the replay step re-charges the same position span)."""
    pool = _pool()
    page_table = torch.zeros(4, 128, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)

    # the default (record=False) call stays None-returning and unchanged
    idle = Req(input_ids=torch.arange(8, dtype=torch.int32), table_idx=1, cached_len=7,
               output_len=4, uid=1, sampling_params=SamplingParams(), cache_handle=None)
    idle.device_len = 7  # nothing new to charge
    assert cm.allocate_paged([idle]) is None

    req = Req(input_ids=torch.arange(65, dtype=torch.int32), table_idx=0, cached_len=64,
              output_len=10, uid=0, sampling_params=SamplingParams(), cache_handle=None)
    free_before = len(cm.free_slots)

    # simulate the spec draft bump: two rows at positions [64, 65]; this manager runs
    # page_size=1, so the span charges exactly two one-slot pages
    req.device_len = 66
    info = cm.allocate_paged([req], record=True)
    assert info == [(0, 64, 66)]
    assert len(cm.free_slots) == free_before - 2
    assert page_table[0, 64:66].abs().sum() > 0

    cm.release_paged(info)
    assert len(cm.free_slots) == free_before

    # the replay step re-charges the same span and gets fresh slots
    info2 = cm.allocate_paged([req], record=True)
    assert info2 == info and len(cm.free_slots) == free_before - 2
    cm.release_paged(info2)
    assert len(cm.free_slots) == free_before


import pytest

# ---------------------------------------------------------------------------
# Host-tiered GDN snapshots (insurance backups): a tombstoned snapshot keeps a
# resumable boundary via the pinned host cache instead of dying.
# ---------------------------------------------------------------------------

def _host_cm(pool, budget_mb=64, backend="cpp"):
    from freetoken.kvcache.linear_state_host import LinearStateHostCache

    import os
    os.environ["FREETOKEN_RADIX_BACKEND"] = backend
    page_table = torch.zeros(4, 64, dtype=torch.int32)
    cm = CacheManager(64, 1, page_table, "hybrid_radix", linear_state_pool=pool)
    cm.host_cache = LinearStateHostCache(pool, torch.device("cpu"), budget_mb)
    return cm


@pytest.mark.parametrize("backend", ["cpp", "py"])
def test_host_backup_survives_tombstone_and_restores(backend):
    pool = _pool(num_slots=8)
    cm = _host_cm(pool, backend=backend)

    ids4 = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    kv4 = torch.tensor([100, 101, 102, 103], dtype=torch.int32)
    S = pool.alloc(1)[0]
    pool.conv_states[:, S] = 3.0
    pool.recurrent_states[:, S] = 5.0
    conv_ref, rec_ref = pool.conv_states[:, S].clone(), pool.recurrent_states[:, S].clone()

    _, exist = cm.prefix_cache.insert(ids4, kv4, S)
    assert not exist
    cm._stash_snapshot_insurance(ids4, S)
    m = cm.prefix_cache.match_prefix(ids4)
    assert m.mamba_value == S
    assert m.node.mamba_host_id is not None, "backup buffer parked on the node"

    # a longer shared path makes the boundary node INTERNAL (the tombstone case)
    ids5 = torch.tensor([1, 2, 3, 4, 9], dtype=torch.int32)
    kv5 = torch.tensor([100, 101, 102, 103, 200], dtype=torch.int32)
    T = pool.alloc(1)[0]
    _, exist = cm.prefix_cache.insert(ids5, kv5, T)
    assert not exist

    # pool pressure tombstones the OLDER boundary snapshot; the host backup stays
    free_before = pool.num_free_slots
    cm.ensure_mamba_slots(free_before + 1)
    assert pool.num_free_slots == free_before + 1
    m2 = cm.prefix_cache.match_prefix(ids4)
    assert m2.mamba_value is None
    assert m2.mamba_host_id is not None, "tombstone must keep the host backup"

    # match_req plumbing carries the host id to the admission path
    mr = cm.match_req(_pend([1, 2, 3, 4, 7]))
    assert mr.cuda_handle.cached_len == 4
    assert mr.mamba_value is None and mr.mamba_host_id == m2.mamba_host_id

    # the backup restores the exact bytes the donate captured
    D = pool.alloc(1)[0]
    cm.host_cache.restore_from(m2.mamba_host_id, D)
    assert torch.equal(pool.conv_states[:, D], conv_ref)
    assert torch.equal(pool.recurrent_states[:, D], rec_ref)


@pytest.mark.parametrize("backend", ["cpp", "py"])
def test_leaf_with_backup_tombstones_keeps_kv_and_full_evict_releases(backend):
    pool = _pool(num_slots=8)
    cm = _host_cm(pool, backend=backend)

    ids3 = torch.tensor([1, 2, 3], dtype=torch.int32)
    kv3 = torch.tensor([10, 11, 12], dtype=torch.int32)
    S = pool.alloc(1)[0]
    _, exist = cm.prefix_cache.insert(ids3, kv3, S)
    assert not exist
    cm._stash_snapshot_insurance(ids3, S)
    first_buf = cm.prefix_cache.match_prefix(ids3).node.mamba_host_id
    assert first_buf is not None

    # GDN-slot pressure: a backup-carrying LEAF tombstones (slot freed, KV + backup kept)
    free_before = pool.num_free_slots
    cm.ensure_mamba_slots(free_before + 1)
    assert pool.num_free_slots == free_before + 1
    m = cm.prefix_cache.match_prefix(ids3)
    assert m.cached_len == 3 and m.mamba_value is None
    assert m.mamba_host_id == first_buf, "tombstoned leaf resumes from its backup"

    # KV-pressure eviction deletes the node: the backup buffer returns to the free list.
    # Drive it through the real CM `_allocate` path (empty the KV free-list first) so the
    # release handling is what production runs.
    cm.free_slots = cm.free_slots[:0]
    cm._allocate(1)
    assert cm.prefix_cache.match_prefix(ids3).cached_len == 0

    # the released buffer is reusable by a later stash
    ids2 = torch.tensor([7, 8], dtype=torch.int32)
    kv2 = torch.tensor([20, 21], dtype=torch.int32)
    S2 = pool.alloc(1)[0]
    _, exist = cm.prefix_cache.insert(ids2, kv2, S2)
    assert not exist
    cm._stash_snapshot_insurance(ids2, S2)
    assert cm.prefix_cache.match_prefix(ids2).node.mamba_host_id == first_buf


@pytest.mark.parametrize("backend", ["cpp", "py"])
def test_leaf_without_backup_still_deletes_on_eviction(backend):
    pool = _pool(num_slots=8)
    cm = _host_cm(pool, backend=backend)

    ids3 = torch.tensor([1, 2, 3], dtype=torch.int32)
    kv3 = torch.tensor([10, 11, 12], dtype=torch.int32)
    S = pool.alloc(1)[0]
    _, exist = cm.prefix_cache.insert(ids3, kv3, S)
    assert not exist
    # no stash: the historical leaf-deletion path must stay intact
    free_before = pool.num_free_slots
    cm.ensure_mamba_slots(free_before + 1)
    assert pool.num_free_slots == free_before + 1
    assert cm.prefix_cache.match_prefix(ids3).cached_len == 0


@pytest.mark.parametrize("backend", ["cpp", "py"])
def test_stash_skips_gracefully_when_host_full(backend):
    pool = _pool(num_slots=8)
    # 1-MiB budget against a ~16.8-MiB slot: exactly one buffer
    cm = _host_cm(pool, budget_mb=1, backend=backend)

    ids_a = torch.tensor([1, 2], dtype=torch.int32)
    kv_a = torch.tensor([10, 11], dtype=torch.int32)
    Sa = pool.alloc(1)[0]
    assert not cm.prefix_cache.insert(ids_a, kv_a, Sa)[1]
    cm._stash_snapshot_insurance(ids_a, Sa)
    assert cm.prefix_cache.match_prefix(ids_a).node.mamba_host_id is not None

    ids_b = torch.tensor([3, 4], dtype=torch.int32)
    kv_b = torch.tensor([20, 21], dtype=torch.int32)
    Sb = pool.alloc(1)[0]
    assert not cm.prefix_cache.insert(ids_b, kv_b, Sb)[1]
    cm._stash_snapshot_insurance(ids_b, Sb)
    # buffer pool exhausted: B's snapshot skips the backup and stays slot-only
    mb = cm.prefix_cache.match_prefix(ids_b)
    assert mb.mamba_host_id is None and mb.mamba_value == Sb
