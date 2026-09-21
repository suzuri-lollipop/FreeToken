"""C++-backed radix tree (_radix_tree) unit tests.

The scenario battery (test_plain_radix / test_swa_radix / test_hybrid_radix)
already runs every spec through BOTH backends against the reference model; this
file covers what belongs to the C++ implementation alone:

- node-identity lifecycle: generation-tagged ids, stale-id loudness after evict,
  split identity (the original node stays the suffix), registry `is` stability;
- production-shape inputs: int32 ids (token pool dtype), GPU value tensors
  (page-table rows) staying on-device through match/evict/path concat;
- cross-backend differential fuzz: the same random op stream through the Python
  and C++ trees must produce identical observable answers;
- the counter-mirror tamper tripwire (the py-backend self-test's cpp sibling).
"""
from __future__ import annotations

import random

import pytest
import torch

from freetoken.kvcache.cpp_radix_tree import (
    CppHybridRadixCache,
    CppRadixCacheHandle,
    CppRadixPrefixCache,
    CppSWARadixCache,
    cpp_radix_available,
    radix_backend,
)

pytestmark = pytest.mark.skipif(
    not cpp_radix_available(), reason="_radix_tree extension is not built"
)

DEV = torch.device("cpu")


def _ids(xs, dtype=torch.int64):
    return torch.tensor(list(xs), dtype=dtype)


def _slots(xs):
    return torch.tensor(list(xs), dtype=torch.int32)


# --------------------------------------------------------------------------- identity
def test_node_identity_is_stable_and_split_keeps_the_suffix():
    c = CppRadixPrefixCache(DEV, page_size=2)
    ids = _ids([1, 2, 3, 4, 5, 6])
    c.insert_prefix(ids, _slots(range(100, 106)))

    m1 = c.match_prefix(ids).cuda_handle
    m2 = c.match_prefix(ids).cuda_handle
    assert m1.node is m2.node                     # registry identity
    assert m1.node.is_leaf()

    # a diverging prefix splits: the ORIGINAL node must stay the suffix
    before_key = m1.node._key.tolist()
    partial = _ids([1, 2, 3, 4, 9, 9])
    m3 = c.match_prefix(partial).cuda_handle
    assert m3.node is not m1.node
    assert m1.node._key.tolist() == before_key[4:]        # suffix kept its tail
    assert m3.node._key.tolist() == before_key[:4]        # new prefix node
    assert m1.node.parent is m3.node
    assert m1.node.path_slots() == [100, 101, 102, 103, 104, 105]


def test_stale_node_id_fails_loudly_after_eviction():
    c = CppRadixPrefixCache(DEV, page_size=1)
    c.insert_prefix(_ids([7, 8, 9]), _slots([10, 11, 12]))
    node = c.match_prefix(_ids([7, 8, 9])).cuda_handle.node
    stale_id = node.id64
    c.evict(3)                                    # unlinks the leaf
    with pytest.raises(RuntimeError, match="stale"):
        c._tree.path_kv(stale_id)
    with pytest.raises(RuntimeError, match="stale"):
        node.length


def test_children_dict_is_keyed_like_key_fn():
    for page in (1, 4):
        c = CppRadixPrefixCache(DEV, page_size=page)
        # first tokens differ -> two distinct root buckets for any page size
        c.insert_prefix(_ids([1] * page + [2] * page), _slots(range(2 * page)))
        c.insert_prefix(_ids([3] * page + [4] * page), _slots(range(100, 100 + 2 * page)))
        root = c.root_node
        for key, child in root.children.items():
            # the battery's closure check: registration under key_fn(child._key)
            assert root.children.get(c.key_fn(child._key)) is child
            assert key == c.key_fn(child._key)
        assert len(root.children) == 2


# --------------------------------------------------------------------------- inputs
def test_int32_ids_and_int64_ids_agree():
    a = CppRadixPrefixCache(DEV, page_size=2)
    b = CppRadixPrefixCache(DEV, page_size=2)
    seq = [5, 6, 7, 8]
    a.insert_prefix(_ids(seq, torch.int32), _slots(range(4)))
    b.insert_prefix(_ids(seq, torch.int64), _slots(range(4)))
    ma = a.match_prefix(_ids(seq, torch.int32)).cuda_handle
    mb = b.match_prefix(_ids(seq, torch.int64)).cuda_handle
    assert ma.cached_len == mb.cached_len == 4
    assert ma.get_matched_indices().tolist() == mb.get_matched_indices().tolist()


def test_rejects_bad_ids():
    c = CppRadixPrefixCache(DEV, page_size=1)
    with pytest.raises(RuntimeError):
        c.match_prefix(torch.zeros(2, 2, dtype=torch.int64))       # not 1-D
    with pytest.raises(RuntimeError):
        c.match_prefix(_ids([1.5, 2.5], torch.float32))            # not int
    with pytest.raises(RuntimeError):
        c.match_prefix(_ids([1, 2, 3])[::2])                       # non-contiguous


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_gpu_value_tensors_stay_on_device():
    c = CppRadixPrefixCache(torch.device("cuda"), page_size=4)
    ids = _ids([1, 2, 3, 4, 5, 6, 7, 8])
    slots = torch.arange(1000, 1008, dtype=torch.int32, device="cuda")
    c.insert_prefix(ids, slots)
    m = c.match_prefix(ids).cuda_handle
    idx = m.get_matched_indices()
    assert idx.is_cuda and idx.dtype == torch.int32
    assert idx.tolist() == slots.tolist()
    ev = c.evict(4)
    assert ev.is_cuda


# --------------------------------------------------------------------------- handles
def test_cold_miss_handle_and_evict_zero_parity():
    c = CppRadixPrefixCache(DEV, page_size=1)
    m = c.match_prefix(_ids([1, 2, 3]))
    assert m.cuda_handle.cached_len == 0
    assert m.cuda_handle.node.is_root()
    assert c.evict(0).numel() == 0
    assert isinstance(m.cuda_handle, CppRadixCacheHandle)


def test_counter_mirror_tamper_is_visible():
    """The py-backend _ForgetfulInsert self-test's cpp sibling: the mirror attrs
    are plain writable Python state, so the battery's recomputation can catch a
    drifted counter exactly as it does on the Python tree."""
    c = CppRadixPrefixCache(DEV, page_size=1)
    c.insert_prefix(_ids([1, 2]), _slots([10, 11]))
    assert c.evictable_size == 2
    c.evictable_size = 0                          # tamper
    assert c.size_info.evictable_size == 0        # what the battery would read
    c.insert_prefix(_ids([3]), _slots([12]))      # next op re-syncs from C++
    assert c.evictable_size == 3


# --------------------------------------------------------------------------- hybrid/swa
def test_hybrid_snapshot_dedup_and_root_guard():
    c = CppHybridRadixCache(DEV, page_size=1)
    # root can't hold a snapshot: a zero-length insert reports exist
    matched, exist = c.insert(_ids([]), _slots([]), mamba_value=55)
    assert (matched, exist) == (0, True)
    matched, exist = c.insert(_ids([1, 2]), _slots([10, 11]), mamba_value=77)
    assert (matched, exist) == (0, False)
    matched, exist = c.insert(_ids([1, 2]), _slots([10, 11]), mamba_value=88)
    assert (matched, exist) == (2, True)          # dedup: caller frees 88
    m = c.match_prefix(_ids([1, 2, 3]))
    assert m.cached_len == 2 and m.mamba_value == 77
    assert c.mamba_evictable == 1
    c.inc_lock(m.node)
    assert c.mamba_evictable == 0 and c.mamba_protected == 1
    c.dec_lock(m.node)
    assert c.mamba_evictable == 1 and c.mamba_protected == 0
    c.check_integrity()


def test_swa_uuid_stable_per_boundary_node_and_trim():
    c = CppSWARadixCache(DEV, page_size=1, sliding_window_size=2)
    ids = _ids([1, 2, 3, 4, 5, 6])
    c.insert(ids, _slots(range(6)))
    m = c.match_prefix(ids)
    u1 = c.inc_lock(m.node)
    c.dec_lock(m.node, u1)
    m2 = c.match_prefix(ids)
    u2 = c.inc_lock(m2.node)
    # the window boundary is the SAME node -> its uuid is reused, not re-minted
    assert u1 is not None and u2 is not None and u2 == u1
    c.dec_lock(m2.node, u2)
    c.check_integrity()
    # trim tombstones unlocked internal heads below keep_from (the re-match
    # splits a boundary at keep_from first, so the head becomes internal)
    freed = c.trim_head_swa(ids, 4)
    assert freed.numel() > 0
    c.check_integrity()


# --------------------------------------------------------------------------- differential fuzz
def _fuzz_ops(rng, n_ops, vocab, max_len):
    ops = []
    for _ in range(n_ops):
        kind = rng.choice(["match", "insert", "lock", "unlock", "evict_full", "evict_second"])
        ids = [rng.randrange(1, vocab) for _ in range(rng.randrange(1, max_len))]
        ops.append((kind, ids))
    return ops


@pytest.mark.parametrize("kind,page,window", [
    ("plain", 1, 0), ("plain", 4, 0),
    ("swa", 1, 4), ("swa", 4, 8),
    ("hybrid", 1, 0), ("hybrid", 4, 0),
])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cross_backend_differential_fuzz(kind, page, window, seed):
    """The same random op stream through both backends: every observable answer
    (matched lengths, indices, freed slots, counters, uuid parity, integrity)
    must agree. Independent of the reference model -- a direct py<->cpp check."""
    rng = random.Random(seed * 100 + page + window)
    ops = _fuzz_ops(rng, 120, vocab=3 * page + 2, max_len=10 * page)

    def build(backend):
        from freetoken.kvcache.hybrid_radix_cache import HybridRadixCache
        from freetoken.kvcache.radix_cache import RadixPrefixCache
        from freetoken.kvcache.swa_radix_cache import SWARadixCache

        cpp = backend == "cpp"
        if kind == "plain":
            cls = CppRadixPrefixCache if cpp else RadixPrefixCache
            return cls(DEV, page_size=page)
        if kind == "swa":
            if cpp:
                return CppSWARadixCache(DEV, page, window)
            return SWARadixCache(DEV, page_size=page, sliding_window_size=window)
        cls = CppHybridRadixCache if cpp else HybridRadixCache
        return cls(DEV, page)

    caches = {"py": build("py"), "cpp": build("cpp")}
    held = {"py": [], "cpp": []}          # (node, uuid) pairs currently locked
    slot_box = [10_000]                   # shared slot allocator (same ids both backends)
    mamba_box = [900_000]

    def counters(c):
        if kind == "plain":
            si = c.size_info
            return (si.evictable_size, si.protected_size)
        if kind == "swa":
            return (c.full_evictable, c.full_protected, c.swa_evictable, c.swa_protected)
        return (c.full_evictable, c.full_protected, c.mamba_evictable, c.mamba_protected)

    for step, (op, ids) in enumerate(ops):
        t_ids = _ids(ids)
        n = len(ids)
        # Payloads are allocated ONCE per op so both backends see identical ids.
        slots = _slots(range(slot_box[0], slot_box[0] + n))
        slot_box[0] += n
        mamba = mamba_box[0]
        mamba_box[0] += 1
        results = {}
        skip = False
        for backend in ("py", "cpp"):
            c = caches[backend]
            if op == "match":
                if kind == "plain":
                    h = c.match_prefix(t_ids).cuda_handle
                    out = (h.cached_len,
                           [] if h.cached_len == 0 else h.get_matched_indices().tolist())
                else:
                    m = c.match_prefix(t_ids)
                    out = (m.cached_len, m.kv_indices.tolist(),
                           m.mamba_value if kind == "hybrid" else None)
            elif op == "insert":
                if kind == "plain":
                    r = c.insert_prefix(t_ids, slots)
                    out = (r.cached_len, r.handle.cached_len)
                elif kind == "hybrid":
                    out = tuple(c.insert(t_ids, slots, mamba))
                else:
                    matched, freed = c.insert(t_ids, slots)
                    out = (matched, freed.tolist())
            elif op == "lock":
                if kind == "plain":
                    h = c.match_prefix(t_ids).cuda_handle
                    if h.cached_len == 0:
                        skip = True            # cold: nothing to lock (both backends)
                        break
                    c.lock_handle(h)
                    held[backend].append((h.node, None))
                    out = counters(c)
                elif kind == "hybrid":
                    m = c.match_prefix(t_ids)
                    if m.cached_len == 0:
                        skip = True
                        break
                    c.inc_lock(m.node)
                    held[backend].append((m.node, None))
                    out = counters(c)
                else:
                    m = c.match_prefix(t_ids)
                    if m.cached_len == 0:
                        skip = True
                        break
                    u = c.inc_lock(m.node)
                    held[backend].append((m.node, u))
                    out = (u is not None, counters(c))
            elif op == "unlock":
                if not held[backend]:
                    skip = True
                    break
                node, u = held[backend].pop()
                if kind == "plain":
                    from freetoken.kvcache.radix_cache import RadixCacheHandle
                    c.lock_handle(RadixCacheHandle(0, node), unlock=True)
                elif kind == "hybrid":
                    c.dec_lock(node)
                else:
                    c.dec_lock(node, u)
                out = counters(c)
            elif op == "evict_full":
                if kind == "plain":
                    size = min(n, c.size_info.evictable_size)
                    # Eviction victim ORDER among equal-stamp candidates is a heap
                    # implementation detail (both orders are model-legal, and the
                    # freed set lands in an unordered free list) -> compare sets.
                    out = sorted(c.evict(size).tolist())
                else:
                    ev = c.evict_full(n)
                    out = (sorted(ev.kv_indices.tolist()),
                           sorted(ev.swa_indices.tolist()) if kind == "swa"
                           else sorted(ev.mamba_slots))
            else:  # evict_second
                if kind == "plain":
                    skip = True
                    break
                ev = (c.evict_swa(n) if kind == "swa" else c.evict_mamba(max(1, n // 4)))
                out = (sorted(ev.kv_indices.tolist()),
                       sorted(ev.swa_indices.tolist()) if kind == "swa"
                       else sorted(ev.mamba_slots))
            results[backend] = out
        if skip or len(results) < 2:
            continue
        assert results["py"] == results["cpp"], (
            f"step {step} op={op} ids={ids}: py={results['py']} cpp={results['cpp']}")
        assert counters(caches["py"]) == counters(caches["cpp"]), f"counter drift at step {step}"
    for backend in ("py", "cpp"):
        caches[backend].check_integrity()


def test_backend_env_selection(monkeypatch):
    monkeypatch.setenv("FREETOKEN_RADIX_BACKEND", "py")
    assert radix_backend() == "py"
    monkeypatch.setenv("FREETOKEN_RADIX_BACKEND", "cpp")
    assert radix_backend() == "cpp"
    monkeypatch.setenv("FREETOKEN_RADIX_BACKEND", "bogus")
    with pytest.raises(ValueError):
        radix_backend()
