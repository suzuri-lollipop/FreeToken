"""Prefill hit-D2D split: resident experts must be gathered device-side, misses
H2D'd via cudaMemcpyBatchAsync, and the buffer must end up byte-identical to the
full-layer copy for every mix -- including experts resident in the volatile
buffer slots (< 2 * num_experts), which must be re-fetched over PCIe."""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.moe.offload_cache import OffloadMoeCache

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
JIT = pytest.mark.skipif(
    os.getenv("FREETOKEN_DISABLE_JIT", "").strip().lower() in {"1", "true", "yes", "on"},
    reason="batch_memcpy has no AOT prebuild; needs runtime JIT",
)


def _cuda_at_least(major: int, minor: int) -> bool:
    cuda = torch.version.cuda
    if cuda is None:
        return False
    return tuple(int(x) for x in cuda.split(".")[:2]) >= (major, minor)

BATCH_API = pytest.mark.skipif(
    not _cuda_at_least(13, 0), reason="the cudaMemcpyBatchAsync binding needs CUDA >= 13.0"
)

NUM_LAYERS, E, CACHE_SIZE = 3, 8, 24  # hit region = slots [16, 24)


def _make_cache() -> tuple[OffloadMoeCache, dict[str, list[torch.Tensor]]]:
    dev = torch.device("cuda")
    sources = {
        "gate_up": [torch.randn(E, 32, 8, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
        "down": [torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
    }
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS,
        num_experts=E,
        cache_size=CACHE_SIZE,
        device=dev,
        prefill_overlap=True,
        prefill_hit_d2d=True,
    )
    cache.set_bank_sources(sources)
    return cache, sources


def _seed_resident(cache, sources, layer_id: int, expert_id: int, slot: int) -> None:
    cache.slot_for_id[layer_id, expert_id] = slot
    cache.id_of_slot[slot] = layer_id * E + expert_id
    for name, per_layer in sources.items():
        cache.bank_caches[name][slot].copy_(per_layer[layer_id][expert_id])


@CUDA
@JIT
@BATCH_API
def test_batch_memcpy_roundtrip():
    from freetoken.kernel.batch_memcpy import batch_memcpy_jit

    rows, feat = 16, 1024
    src = torch.randint(0, 256, (rows, feat), dtype=torch.uint8).pin_memory()
    dst = torch.zeros(rows, feat, dtype=torch.uint8, device="cuda")
    perm = torch.randperm(rows)
    dst_ptrs = torch.tensor([dst[i].data_ptr() for i in range(rows)], dtype=torch.int64)
    src_ptrs = torch.tensor([src[p].data_ptr() for p in perm.tolist()], dtype=torch.int64)
    sizes = torch.full((rows,), feat, dtype=torch.int64)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        batch_memcpy_jit(dst_ptrs, src_ptrs, sizes, stream.cuda_stream)
    stream.synchronize()
    assert torch.equal(dst.cpu(), src[perm])

@CUDA
@JIT
@BATCH_API
def test_probe_waits_for_the_calling_stream():
    """The probe must order its copy after work queued on the CALLER's stream: a
    caller inside ``torch.cuda.stream()`` has pending writes there, and an
    unordered copy lets them overtake it and read back as wrong bytes."""
    from freetoken.kernel import batch_memcpy

    fn = batch_memcpy.load_batch_memcpy()
    for _ in range(4):
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):
            torch.cuda._sleep(10_000_000)  # park the stream so the race is not a coin flip
            batch_memcpy._probe(fn)
        side.synchronize()


@CUDA
@JIT
@BATCH_API
def test_probe_runs_once_per_process():
    """The probe is a real H2D plus a host sync, so a loader call must not pay for
    it again -- ``batch_memcpy_jit`` goes through the loader on every call."""
    from freetoken.kernel import batch_memcpy

    calls = []
    real_probe = batch_memcpy._probe
    batch_memcpy._probe = lambda fn: calls.append(fn)
    batch_memcpy.load_batch_memcpy.cache_clear()
    try:
        for _ in range(3):
            batch_memcpy.load_batch_memcpy()
    finally:
        batch_memcpy._probe = real_probe
        batch_memcpy.load_batch_memcpy.cache_clear()
    assert len(calls) == 1


@CUDA
@JIT
@BATCH_API
def test_prefill_hit_d2d_matches_sources():
    cache, sources = _make_cache()
    # layer 1 residents: two real hits in the hit region, one expert stuck in a
    # buffer slot (< 2E) whose cache bytes are poisoned -- it must come back via
    # H2D, so poison must never surface in the buffer.
    _seed_resident(cache, sources, layer_id=1, expert_id=1, slot=17)
    _seed_resident(cache, sources, layer_id=1, expert_id=3, slot=20)
    _seed_resident(cache, sources, layer_id=1, expert_id=0, slot=2)
    for name in sources:
        cache.bank_caches[name][2].fill_(float("nan"))

    cache.begin_prefill()
    assert cache._prefill_hit_d2d_active
    cache.prefetch_prefill_layer(0)
    cache.prefetch_prefill_layer(1)
    for layer_id in (0, 1):
        views = cache.wait_prefill_layer(layer_id)
        torch.cuda.synchronize()
        for view, (name, per_layer) in zip(views, sources.items()):
            assert torch.equal(view.cpu(), per_layer[layer_id]), (layer_id, name)
        cache.release_prefill_layer(layer_id)
    # prefetch(0) had no hits; prefetch(1) served experts 1 and 3 from the cache.
    assert cache.prefill_hit_rows == 2
    assert cache.prefill_total_rows == 2 * E
    # the buffer-slot resident was invalidated (its slot belongs to buffer 0)...
    assert int(cache.slot_for_id[1, 0].item()) == -1
    # ...while true hits keep their cache residency.
    assert int(cache.slot_for_id[1, 1].item()) == 17
    assert int(cache.slot_for_id[1, 3].item()) == 20


@CUDA
@JIT
@BATCH_API
@pytest.mark.parametrize("nhit", [0, E])
def test_prefill_hit_d2d_pure_extremes(nhit):
    cache, sources = _make_cache()
    for e in range(nhit):
        _seed_resident(cache, sources, layer_id=2, expert_id=e, slot=2 * E + e)
    cache.begin_prefill()
    cache.prefetch_prefill_layer(2)
    views = cache.wait_prefill_layer(2)
    torch.cuda.synchronize()
    for view, (name, per_layer) in zip(views, sources.items()):
        assert torch.equal(view.cpu(), per_layer[2]), name
    assert cache.prefill_hit_rows == nhit


@CUDA
def test_prefill_hit_d2d_noop_without_spare_slots():
    dev = torch.device("cuda")
    sources = {
        "gate_up": [torch.randn(E, 32, 8, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
        "down": [torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
    }
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS,
        num_experts=E,
        cache_size=2 * E,  # the cpu backend's pinned geometry: no hit region
        device=dev,
        prefill_overlap=True,
        prefill_hit_d2d=True,
    )
    cache.set_bank_sources(sources)
    cache.begin_prefill()
    assert not cache._prefill_hit_d2d_active
    cache.prefetch_prefill_layer(0)
    views = cache.wait_prefill_layer(0)
    torch.cuda.synchronize()
    for view, (name, per_layer) in zip(views, sources.items()):
        assert torch.equal(view.cpu(), per_layer[0]), name


# ---------------------------------------------------------------- on-demand


@CUDA
@JIT
def test_prefill_ondemand_stages_touched_rows_only():
    """Small banks (no D2D-eligible rows): every touched expert is H2D-gathered
    from the host banks; untouched buffer rows stay stale and unmapped ids of the
    staged buffer are invalidated."""
    cache, sources = _make_cache()
    # layer 0 stages into buffer 0 (slots [0, E)): seed one true hit region slot,
    # one stale buffer-slot mapping with poisoned bytes.
    _seed_resident(cache, sources, layer_id=0, expert_id=1, slot=17)
    cache.slot_for_id[0, 0] = 2
    cache.id_of_slot[2] = 0
    cache.usage[2] = 999
    for name in sources:
        cache.bank_caches[name][2].fill_(float("nan"))
    topk = torch.tensor([[1, 3, 0, 5], [5, 7, 1, 6]], dtype=torch.int32, device="cuda")
    views = cache.prefetch_prefill_layer_ondemand(0, topk)
    torch.cuda.synchronize()
    touched = (0, 1, 3, 5, 6, 7)
    for view, (name, per_layer) in zip(views, sources.items()):
        for e in touched:
            assert torch.equal(view[e].cpu(), per_layer[0][e]), (name, e)
    # the poisoned slot-2 bytes never surface in the TOUCHED rows (expert 0 was
    # re-fetched from the host); untouched rows legitimately keep stale bytes
    for v in views:
        assert not bool(torch.isnan(v.float()[list(touched)]).any())
    # hit residency is untouched; the staged buffer's stale mapping is invalidated
    assert int(cache.slot_for_id[0, 1].item()) == 17
    assert int(cache.slot_for_id[0, 0].item()) == -1
    assert int(cache.id_of_slot[2].item()) == -1


@CUDA
@JIT
def test_prefill_ondemand_hits_gather_from_cache():
    """Big banks: touched hits are served D2D from their cache slots (NOT the
    host), touched misses H2D from the host banks."""
    dev = torch.device("cuda")
    # rows >= 256 KiB so both banks are D2D-eligible (_SMALL_BANK_FEAT_BYTES)
    sources = {
        "gate_up": [torch.randn(E, 128, 1024, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
        "down": [torch.randn(E, 64, 128, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
    }
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS, num_experts=E, cache_size=CACHE_SIZE,
        device=dev, prefill_overlap=True, prefill_hit_d2d=True,
    )
    cache.set_bank_sources(sources)
    assert cache._gather_dst_ptrs is not None
    # expert 1 resident at slot 17 with CACHE bytes that differ from the host bank
    distinct = {name: torch.randn_like(per_layer[2][1]) for name, per_layer in sources.items()}
    cache.slot_for_id[2, 1] = 19
    cache.id_of_slot[19] = 2 * E + 1
    for name, rows in distinct.items():
        cache.bank_caches[name][19].copy_(rows)
    topk = torch.tensor([[1, 2], [2, 4]], dtype=torch.int32, device=dev)
    views = cache.prefetch_prefill_layer_ondemand(2, topk)
    torch.cuda.synchronize()
    for view, (name, per_layer) in zip(views, sources.items()):
        # hit row came from the CACHE slot, not the host
        assert torch.equal(view[1].cpu(), distinct[name].cpu()), name
        # miss rows came from the host banks
        for e in (2, 4):
            assert torch.equal(view[e].cpu(), per_layer[2][e]), (name, e)
    assert int(cache.slot_for_id[2, 1].item()) == 19


# ---------------------------------------------------------------- promotion


def _make_big_cache(promote: float) -> tuple[OffloadMoeCache, dict[str, list[torch.Tensor]]]:
    """D2D-eligible banks (rows >= 256 KiB) so promoted slots serve as plan hits."""
    dev = torch.device("cuda")
    sources = {
        "gate_up": [torch.randn(E, 128, 1024, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
        "down": [torch.randn(E, 64, 128, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
    }
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS, num_experts=E, cache_size=CACHE_SIZE,
        device=dev, prefill_overlap=True, prefill_hit_d2d=True,
    )
    cache.set_bank_sources(sources)
    cache.prefill_promote_fraction = promote
    return cache, sources


@CUDA
@JIT
def test_promote_slots_survive_and_later_staging_hits():
    """Full promotion: every touched miss gets a cache-region slot and its host
    bytes; the following on-demand plan finds NO misses (all D2D hits) and the
    buffer still ends up byte-identical to the host banks."""
    cache, sources = _make_big_cache(promote=1.0)
    topk = torch.tensor([[1, 3], [3, 5]], dtype=torch.int32, device="cuda")
    raw_before = topk.clone()

    cache.promote_prefill_misses(2, topk)
    torch.cuda.synchronize()
    # raw ids untouched (promote clones before the in-place slot rewrite)
    assert torch.equal(topk, raw_before)
    # every touched expert now sits in the CACHE region (>= 2E), never a buffer slot
    for e in (1, 3, 5):
        slot = int(cache.slot_for_id[2, e].item())
        assert slot >= 2 * E, f"expert {e} promoted into the buffer region ({slot})"
        assert int(cache.id_of_slot[slot].item()) == 2 * E + e

    views = cache.prefetch_prefill_layer_ondemand(2, topk)
    torch.cuda.synchronize()
    assert int(cache._prefill_miss_num.item()) == 0     # nothing streamed from host
    assert int(cache._prefill_hit_num.item()) == 3      # all touched gathered D2D
    for view, (name, per_layer) in zip(views, sources.items()):
        for e in (1, 3, 5):
            assert torch.equal(view[e].cpu(), per_layer[2][e]), (name, e)

    # a second staging of the same routing is still all-hits (rows survived)
    cache.prefetch_prefill_layer_ondemand(2, topk)
    torch.cuda.synchronize()
    assert int(cache._prefill_hit_num.item()) == 3
    assert int(cache._prefill_miss_num.item()) == 0


@CUDA
@JIT
def test_promote_fraction_caps_the_promoted_set():
    """fraction 0.5 promotes the bandwidth-balanced half of the miss set; the rest
    stays missing and streams through the plan as before."""
    cache, sources = _make_big_cache(promote=0.5)
    topk = torch.tensor([[1, 3], [5, 7]], dtype=torch.int32, device="cuda")
    cache.promote_prefill_misses(2, topk)
    torch.cuda.synchronize()
    promoted = [e for e in (1, 3, 5, 7) if int(cache.slot_for_id[2, e].item()) >= 2 * E]
    assert len(promoted) == 2
    views = cache.prefetch_prefill_layer_ondemand(2, topk)
    torch.cuda.synchronize()
    assert int(cache._prefill_hit_num.item()) == 2
    assert int(cache._prefill_miss_num.item()) == 2
    for view, (name, per_layer) in zip(views, sources.items()):
        for e in (1, 3, 5, 7):
            assert torch.equal(view[e].cpu(), per_layer[2][e]), (name, e)


@CUDA
@JIT
def test_promote_zero_fraction_is_a_noop():
    cache, _ = _make_big_cache(promote=0.0)
    topk = torch.tensor([[1, 3], [3, 5]], dtype=torch.int32, device="cuda")
    cache.promote_prefill_misses(2, topk)
    torch.cuda.synchronize()
    for e in (1, 3, 5):
        assert int(cache.slot_for_id[2, e].item()) == -1
    cache.prefetch_prefill_layer_ondemand(2, topk)
    torch.cuda.synchronize()
    assert int(cache._prefill_hit_num.item()) == 0
    assert int(cache._prefill_miss_num.item()) == 3


@CUDA
@JIT
def test_promote_guard_keeps_fresh_cache_out_of_buffer_slots():
    """On a FRESH cache every usage is 0 and the LRU tie-break is the lowest slot
    index -- without the step+1 guard, promotion would assign exactly the buffer
    slots [0, 2E) that this chunk is about to trample."""
    cache, _ = _make_big_cache(promote=1.0)
    topk = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda")
    cache.promote_prefill_misses(0, topk)
    torch.cuda.synchronize()
    for e in (0, 1, 2, 3):
        assert int(cache.slot_for_id[0, e].item()) >= 2 * E


@CUDA
def test_prefill_ondemand_stages_file_tier_rows_pageably():
    """A disk-tier layer (no device alias, overlap off) still gets touched-only
    staging: the plan runs device-side, the move gathers the routed rows from the
    host banks pageably into the double buffer, and untouched rows stay zero."""
    dev = torch.device("cuda")
    # layer 0's banks are pageable host tensors (an mmap'd shard row in the real
    # loader); layers 1-2 are pinned like a normal resident share
    sources = {
        "gate_up": [
            torch.randn(E, 32, 8, dtype=torch.bfloat16),
            torch.randn(E, 32, 8, dtype=torch.bfloat16).pin_memory(),
            torch.randn(E, 32, 8, dtype=torch.bfloat16).pin_memory(),
        ],
        "down": [
            torch.randn(E, 8, 16, dtype=torch.bfloat16),
            torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory(),
            torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory(),
        ],
    }
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS, num_experts=E, cache_size=CACHE_SIZE,
        device=dev, prefill_overlap=False,
    )
    cache.cpu_layer_ids = {0}
    cache.set_bank_sources(sources, layer_residency=["disk", "pinned", "pinned"])
    # overlap is off, but the on-demand staging buffers exist for the file tier
    assert not cache.prefill_overlap
    assert cache.prefill_bank_buffers

    topk = torch.tensor([[1, 3, 0, 5], [5, 7, 1, 6]], dtype=torch.int32, device=dev)
    views = cache.plan_prefill_layer_ondemand(0, topk)
    cache.move_prefill_layer_ondemand(0)
    torch.cuda.synchronize()
    touched = (0, 1, 3, 5, 6, 7)
    for view, (name, per_layer) in zip(views, sources.items()):
        for e in touched:
            assert torch.equal(view[e].cpu(), per_layer[0][e]), (name, e)
    # untouched buffer rows keep their init-zero bytes
    for view in views:
        assert not bool(torch.count_nonzero(view[list(set(range(E)) - set(touched))]))
    # buffer alternation: layer 2 (pinned, buffer 0 again after 0->2 % 2 == 0? no:
    # layer 2 % 2 == 0) -- plan for a PINNED layer still uses the fused path
    views2 = cache.plan_prefill_layer_ondemand(2, topk)
    cache.move_prefill_layer_ondemand(2)
    torch.cuda.synchronize()
    for view, (name, per_layer) in zip(views2, sources.items()):
        for e in touched:
            assert torch.equal(view[e].cpu(), per_layer[2][e]), (name, e)


@CUDA
def test_prefill_ondemand_reads_file_backed_rows_by_pread(tmp_path):
    """The file-tier move reads the touched rows straight from the backing file
    with parallel preads into pinned staging (no mmap faults): the routed buffer
    rows must byte-match the file, untouched rows stay zero, and the bank's mmap
    pages are never touched (a poisoned mapping proves the read bypassed it)."""
    import os

    from freetoken.moe.host_banks import HostBank

    dev = torch.device("cuda")
    spec = (32, 8)  # bf16 rows
    row_bytes = spec[0] * spec[1] * 2
    # one aligned file holding layer 0's rows; a different byte pattern per expert
    layer_rows = torch.randn(E, *spec, dtype=torch.bfloat16)
    off = 4096
    path = tmp_path / "shard.bin"
    with open(path, "wb") as f:
        f.write(b"\x00" * off)
        f.write(layer_rows.contiguous().view(torch.uint8).numpy().tobytes())
    hb = HostBank((E, *spec), torch.bfloat16, backing="file", file=str(path), offset=off)
    assert hb.tensor.shape == (E, *spec)
    # poison the mapping: any gather that reads through it sees wrong bytes
    hb.tensor.fill_(float("nan"))
    down_spec = (8, 16)
    down_rows = torch.randn(E, *down_spec, dtype=torch.bfloat16)
    down_path = tmp_path / "shard_down.bin"
    with open(down_path, "wb") as f:
        f.write(b"\x00" * off)
        f.write(down_rows.contiguous().view(torch.uint8).numpy().tobytes())
        f.write(b"\x00" * ((-(off + down_rows.numel() * 2)) % 4096))  # block-pad the window
    hb_down = HostBank((E, *down_spec), torch.bfloat16, backing="file", file=str(down_path), offset=off)
    hb_down.tensor.fill_(float("nan"))

    sources = {
        "gate_up": [hb.tensor, torch.randn(E, *spec, dtype=torch.bfloat16).pin_memory(),
                    torch.randn(E, *spec, dtype=torch.bfloat16).pin_memory()],
        "down": [hb_down.tensor, torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory(),
                 torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory()],
    }
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS, num_experts=E, cache_size=CACHE_SIZE,
        device=dev, prefill_overlap=False,
    )
    cache.cpu_layer_ids = {0}
    cache.set_bank_sources(
        sources, layer_residency=["disk", "pinned", "pinned"],
        file_banks={"gate_up": [hb, None, None], "down": [hb_down, None, None]},
    )
    topk = torch.tensor([[1, 3, 0, 5], [5, 7, 1, 6]], dtype=torch.int32, device=dev)
    views = cache.plan_prefill_layer_ondemand(0, topk)
    cache.move_prefill_layer_ondemand(0)
    torch.cuda.synchronize()
    touched = (0, 1, 3, 5, 6, 7)
    for view, ref in ((views[0], layer_rows), (views[1], down_rows)):
        for e in touched:
            # the rows came from the FILE bytes, not the poisoned mmap mapping
            assert torch.equal(view[e].cpu(), ref[e].cpu()), e
        assert not bool(torch.isnan(view.float()[list(touched)]).any())
    # staging exists and is pinned; the second chunk reuses it and re-reads the file
    assert cache._prefill_pinned_staging["gate_up"].is_pinned()
    views = cache.plan_prefill_layer_ondemand(0, topk)
    cache.move_prefill_layer_ondemand(0)
    torch.cuda.synchronize()
    for view, ref in ((views[0], layer_rows), (views[1], down_rows)):
        for e in touched:
            assert torch.equal(view[e].cpu(), ref[e].cpu()), e
    hb.release()
    hb_down.release()


@CUDA
def test_prefill_ondemand_single_buffer_below_double_floor():
    """The on-demand path without overlap borrows ONE expert-layer buffer, so a
    slot cache sized between num_experts and 2 * num_experts (the auto sizer's
    plain floor -- measured on the NVFP4 rig as 575 of 576) still stages; the
    hit threshold tracks the actual buffer region, not a hardcoded 2 * E."""
    dev = torch.device("cuda")
    sources = {
        "gate_up": [torch.randn(E, 32, 8, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
        "down": [torch.randn(E, 8, 16, dtype=torch.bfloat16).pin_memory() for _ in range(NUM_LAYERS)],
    }
    cache = OffloadMoeCache(
        num_layers=NUM_LAYERS, num_experts=E, cache_size=3 * E // 2,  # 12: E..2E
        device=dev, prefill_overlap=False,
    )
    cache.set_bank_sources(sources)
    assert cache.prefill_bank_buffers  # 1 buffer despite 12 < 16
    assert cache._prefill_num_buffers == 1
    topk = torch.tensor([[1, 3, 0, 5], [5, 7, 1, 6]], dtype=torch.int32, device=dev)
    for layer in range(3):  # odd layer ids must not index a second buffer
        views = cache.plan_prefill_layer_ondemand(layer, topk)
        cache.move_prefill_layer_ondemand(layer)
        torch.cuda.synchronize()
        touched = (0, 1, 3, 5, 6, 7)
        for view, (name, per_layer) in zip(views, sources.items()):
            for e in touched:
                assert torch.equal(view[e].cpu(), per_layer[layer][e]), (layer, name, e)
