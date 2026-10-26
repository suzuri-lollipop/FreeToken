from __future__ import annotations

import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from flashlib.kernels.slot_cache import lru_ensure

# Hybrid backend: which of a step's missing experts to fetch (when capped below the miss
# count). "recency" (default) fetches the experts most-recently active before this step
# (LRU on the expert -> prioritizes recurring misses, lowering the steady miss rate);
# "lowest_id" fetches the smallest expert ids (the original, routing-blind heuristic).
_HYBRID_FETCH_BY_RECENCY = (
    os.getenv("FREETOKEN_HYBRID_FETCH", "recency").strip().lower() != "lowest_id"
)
# How fetch_fraction * misses becomes a fetch count: "min-wait" (default) picks the integer
# that makes the PCIe fetch and the CPU compute finish closest together, "stochastic" rounds
# up with probability equal to the fractional part, "prefer-fetch" always rounds up.
_HYBRID_FETCH_ROUNDING = {"min-wait": 0, "stochastic": 1, "prefer-fetch": 2}
_hybrid_fetch_round = (os.getenv("FREETOKEN_HYBRID_FETCH_ROUND") or "min-wait").strip().lower()
if _hybrid_fetch_round not in _HYBRID_FETCH_ROUNDING:
    raise ValueError(f"FREETOKEN_HYBRID_FETCH_ROUND={_hybrid_fetch_round!r}: expected one of {sorted(_HYBRID_FETCH_ROUNDING)}")
_HYBRID_FETCH_ROUND = _HYBRID_FETCH_ROUNDING[_hybrid_fetch_round]
_HYBRID_FETCH_SEED = 0
# Phase 2 of the hybrid ensure kernel scans a pool wider than this in tiles of this width.
_HYBRID_EVICT_CHUNK = 8192


def ensure_experts(cache, layer_id: int, expert_ids: torch.Tensor, plan: int = 0) -> None:
    """Make this layer's routed experts resident; rewrite ``expert_ids`` to slot ids.

    Delegates to flashlib's slot cache. ``id_base`` maps this layer's expert ids into the
    flat ``layer * num_experts + expert`` space the cache indexes by, and maps
    ``src_indices`` back, so ``copy_missing`` still resolves against this layer's own host
    tensor. ``out_indices`` aliases the input, preserving the in-place rewrite every
    downstream GEMM depends on.

    ``plan`` selects the dual-microbatch decode's second staging plan (the two halves'
    ensure/copy pairs run on separate streams and must not alias plan buffers; the LRU
    state itself stays shared and is serialized by per-layer events)."""
    src_indices, evict_slots, num_indices = cache._plan_buffers(plan)
    id_of_slot, usage = cache.pool_state(layer_id)
    lru_ensure(
        expert_ids,
        cache.slot_for_id.view(-1),
        id_of_slot,
        usage,
        cache.step,
        expert_ids,
        src_indices,
        evict_slots,
        num_indices,
        stats=cache.lru_stats[layer_id] if cache.collect_stats else None,
        id_base=layer_id * cache.num_experts,
    )


class HybridRoutes(NamedTuple):
    """One layer's routes split between the GPU and the CPU, each [bs, top_k]."""

    gpu_slots: torch.Tensor  # cache slot of each route; a CPU route is -1, or 0 with clamp
    gpu_w: torch.Tensor  # router weights, 0 for a CPU route
    cpu_ids: torch.Tensor  # expert ids, -1 for a GPU route


def ensure_experts_hybrid(
    cache,
    layer_id: int,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    max_fetch: int,
    fetch_fraction: float = 0.0,
    clamp: bool = False,
    fetch_cap: int | None = None,
) -> HybridRoutes:
    """Capped-fetch variant of ``ensure_experts`` (hybrid backend).

    Identical LRU bookkeeping, but only the first ``max_fetch`` of this step's missing
    experts are given a slot and scheduled for copy; the overflow misses stay
    non-resident and their ``expert_ids`` positions are rewritten to ``-1`` (compute on
    the CPU). ``fetch_fraction`` > 0 replaces the fixed cap with the bandwidth-matched
    split (fraction = pcie_bw / cpu_bw): fetch ~fraction of the step's misses, rounded as
    ``FREETOKEN_HYBRID_FETCH_ROUND`` says (default: the integer that makes the PCIe fetch and the CPU overflow compute finish closest together).
    ``num_indices`` = capped fetch count (copy_missing); ``num_missing_full`` =
    pre-cap miss count (stats).

    The same launch splits the routes: ``expert_ids`` becomes the GPU slots in place, and
    ``clamp`` gives CPU routes slot 0 instead of -1 for GEMM kernels that read every slot.
    ``fetch_cap`` bounds the fetch count after ``max_fetch`` / ``fetch_fraction`` (a fixed-size staging buffer)."""
    # Q16 fixed point: the kernel caps in integers.
    frac_q16 = min(1 << 16, max(0, round(fetch_fraction * (1 << 16))))
    # A disk layer's fetchable rows live in its pinned hot compact at remapped row
    # indices (None = every layer-local row is fetchable, the whole-bank case).
    hot_rows = getattr(cache, "_hot_row_of", None)
    hot_row = hot_rows[layer_id] if hot_rows else None
    weights = weights.contiguous()
    routes = HybridRoutes(expert_ids, torch.empty_like(weights), torch.empty_like(expert_ids))
    cap = cache.num_experts if fetch_cap is None else fetch_cap
    if not expert_ids.is_cuda:
        _ensure_experts_hybrid_cpu(cache, layer_id, expert_ids, max_fetch, frac_q16, hot_row)
    else:
        _ensure_experts_hybrid_gpu(
            cache, layer_id, expert_ids, weights, routes, clamp, max_fetch, frac_q16, cap, hot_row
        )
    return routes


def promote_touched(
    cache, layer_id: int, touched: torch.Tensor, sub_k: int = 128,
    miss_acc: torch.Tensor | None = None,
) -> None:
    """Mask-driven slot promotion for one on-demand prefill layer (auto-promote policy).

    Input is the [num_experts] touched mask, NOT the [T, top_k] route list: the fixed
    E-length query (touched ids + one repeated dummy id that lru_ensure's dedup collapses
    to a single entry) is processed in ``sub_k`` slices so flashlib's phase-1 dedup matrix
    stays at sub_k^2 per launch. The route-sized alternative (ensure_experts_hybrid's
    kernel loops T*top_k routes serially on ONE SM) costs ~seconds per chunk at prefill
    T -- measured: chunk cadence 1.0s -> 2.0s. Slices run sequentially and each sees the
    previous slice's assignments, so an id is never double-assigned; ``num_indices`` is
    per-call, so each slice's misses are copied right after its ensure. ``miss_acc``
    (a [1] device int64) accumulates the fresh-miss count across the slices for the
    policy feedback (the slot-direct path runs no plan, so nothing else counts them).
    """
    num_experts = cache.num_experts
    query = torch.where(touched != 0, cache._promote_arange, cache._promote_dummy)
    buffer_slots = 2 * num_experts
    n_slices = -(-num_experts // sub_k)
    # Future-stamp the double-buffer guard ONCE for the whole slice loop. lru_ensure
    # bumps its step per call and excludes only usage == (own call's step) from the
    # victim argmin, so a stamp of step + n_slices keeps the buffers strictly NEWEST
    # across every slice (no other slot's usage can exceed the current step; the
    # final slice's own equality-exclusion covers the last one). The earlier
    # per-slice freshen cost 2 launches x n_slices per layer in the host-launch-bound
    # chunk cadence; this needs 2 total. A once-per-loop step+1 stamp was the STALE
    # variant (slices 2+ could victim the buffers: promotions below 2E are re-missed
    # by the plan and invalidated next chunk -- zero reuse, double traffic); the
    # regression test pins the assigned-slots >= 2E invariant. Buffers age back into
    # eviction candidacy once the step counter passes the stamp.
    torch.maximum(
        cache.usage[:buffer_slots], cache.step + n_slices, out=cache.usage[:buffer_slots]
    )
    cache._pending_src_layer = layer_id
    cache._pending_whole_layer = False
    for lo in range(0, num_experts, sub_k):
        sub = query[lo : lo + sub_k]
        lru_ensure(
            sub,
            cache.slot_for_id.view(-1),
            cache.id_of_slot,
            cache.usage,
            cache.step,
            sub,
            cache.src_indices,
            cache.evict_slots,
            cache.num_indices,
            stats=None,
            id_base=layer_id * num_experts,
        )
        if miss_acc is not None:
            miss_acc += cache.num_indices
        cache.copy_missing()


def prefill_hit_compact(cache, layer_id: int) -> None:
    """Compact this layer's cache-resident experts into gather indices, device-side.

    hit = slot_for_id[layer_id][e] >= the pool's staging threshold (the double buffer owns the
    slots below, so those bytes are volatile within a prefill chunk and classify
    as miss). Writes fixed-shape ``_prefill_hit_dst``/``_prefill_hit_src`` (buffer
    row / pool-local slot) and the count into ``_prefill_hit_num``; one launch on the
    current stream, no host sync. Safe against the concurrent buffer invalidation
    on the copy stream: that only rewrites entries already below the threshold."""
    num_experts = cache.num_experts
    _prefill_hit_compact_kernel[(1,)](
        cache.slot_for_id[layer_id],
        cache._prefill_hit_dst,
        cache._prefill_hit_src,
        cache._prefill_hit_num,
        cache._pool_stable_from[cache.pool_of_layer[layer_id]],
        num_experts,
        BLOCK=triton.next_power_of_2(num_experts),
    )


def prefill_ondemand_compact(
    cache, layer_id: int, buffer_id: int, touched: torch.Tensor, threshold: int | None = None,
    pcie_frac_q16: int = 1 << 16,
) -> None:
    """Touched-only split of one layer's experts for the on-demand prefill path.

    ``touched`` is a [num_experts] 0/1 mask of the experts this chunk's routing
    actually selected. Emits two fixed-shape gather plans, device-side, no host
    sync: hits (mask & slot >= 2E) as (buffer row, cache slot) pairs for the D2D
    gather, and misses (mask & slot < 2E) as (buffer row, host row) pairs -- both
    indexed position == expert id, so the GEMM keeps the full-layer bank contract.
    ``threshold`` overrides the hit cut (a huge value routes every touched expert
    to the miss/H2D plan). The caller passes a threshold no slot can reach when the
    cache has no D2D-eligible bank at all: a hit it cannot gather would leave that
    buffer row as the stale bytes of an earlier chunk.

    ``pcie_frac_q16`` is the Q16 share of this layer's misses that the H2D plan
    carries; the remainder are marked in ``cache._prefill_cpu_mask`` for the CPU
    executor, which reads them straight from the host banks. 1 << 16 (the default)
    sends every miss over PCIe and leaves the mask all zero.
    """
    num_experts = cache.num_experts
    _prefill_ondemand_compact_kernel[(1,)](
        cache.slot_for_id[layer_id],
        touched,
        cache._prefill_hit_dst,
        cache._prefill_hit_src,
        cache._prefill_hit_num,
        cache._prefill_miss_dst,
        cache._prefill_miss_src,
        cache._prefill_miss_num,
        cache._prefill_cpu_mask,
        buffer_id * num_experts,
        2 * num_experts if threshold is None else threshold,
        num_experts,
        int(pcie_frac_q16),
        BLOCK=triton.next_power_of_2(num_experts),
    )


def materialize_layer(cache, layer_id: int) -> None:
    if len(cache.pool_sizes) == 1:
        _materialize_layer_gpu(cache, layer_id)
        return
    # staging buffer 0 may overlap several pools, so it only holds the layer for this forward and registers nothing
    E = cache.num_experts
    cache._invalidate_prefill_buffer(0, layer_id)
    rows = torch.arange(E, dtype=torch.int32, device=cache.device)
    cache.evict_slots[:E].copy_(rows)
    cache.src_indices[:E].copy_(rows)
    cache.num_indices.fill_(E)


def reset_cache(cache) -> None:
    _reset_cache_gpu(cache)






def _ensure_experts_hybrid_gpu(
    cache,
    layer_id: int,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    routes: HybridRoutes,
    clamp: bool,
    max_fetch: int,
    frac_q16: int,
    fetch_cap: int,
    hot_row: "torch.Tensor | None" = None,
) -> None:
    id_of_slot, usage = cache.pool_state(layer_id)
    num_slots = id_of_slot.numel()
    block_e = triton.next_power_of_2(cache.num_experts)
    chunk = min(_HYBRID_EVICT_CHUNK, triton.next_power_of_2(num_slots))
    # with 4 warps a 4096+ chunk spills on sm80-sm89
    num_warps = 8 if chunk >= 4096 else 4
    _ensure_experts_hybrid_kernel[(1,)](
        expert_ids,
        hot_row,
        cache.slot_for_id,
        id_of_slot,
        usage,
        cache.step,
        cache.active_mask,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        cache.num_missing_full,
        cache.expert_recency,
        weights,
        routes.cpu_ids,
        routes.gpu_w,
        *(cache.hybrid_stats if cache.collect_stats else (expert_ids,) * 2),
        layer_id,
        expert_ids.numel(),
        int(max_fetch),
        int(frac_q16),
        int(fetch_cap),
        cache.num_experts,
        num_slots,
        _HYBRID_FETCH_SEED,
        BLOCK_E=block_e,
        CHUNK=chunk,
        SMALL=num_slots <= chunk,
        BY_RECENCY=_HYBRID_FETCH_BY_RECENCY,
        HAS_HOT=hot_row is not None,
        ROUND=_HYBRID_FETCH_ROUND,
        CLAMP=clamp,
        STATS=cache.collect_stats,
        num_warps=num_warps,
    )


def _ensure_experts_hybrid_cpu(
    cache, layer_id: int, expert_ids: torch.Tensor, max_fetch: int, frac_q16: int,
    hot_row: "torch.Tensor | None" = None,
) -> None:
    """CPU reference mirror of the hybrid kernel (eviction/fetch decisions bit-identical to
    the GPU path; see tests/test_offload_lru_kernels.py). Fetches at most ``max_fetch`` (or
    the bandwidth-matched ``~frac_q16/2^16 * misses`` when ``frac_q16`` > 0) of the missing
    experts; overflow misses are rewritten to -1. With ``BY_RECENCY`` the fetch set is the
    most-recently-active misses (ties -> lower id); else the lowest ids."""
    seen = []
    for expert in expert_ids.view(-1).tolist():
        if expert not in seen:
            seen.append(expert)

    cache.active_mask.zero_()
    step = int(cache.step.item()) + 1
    cache.step.fill_(step)
    for expert in seen:
        cache.active_mask[expert] = 1

    for expert in seen:
        slot = int(cache.slot_for_id[layer_id, expert].item())
        if slot != -1:
            cache.usage[slot] = step

    missing = [e for e in seen if int(cache.slot_for_id[layer_id, e].item()) == -1]
    if _HYBRID_FETCH_BY_RECENCY:
        rec = cache.expert_recency[layer_id].tolist()
        missing.sort(key=lambda e: (-rec[e], e))
    else:
        missing.sort()
    # num_missing_full is the PRE-filter count (the stat's meaning is unchanged: every
    # miss, fetchable or not)
    num_full = len(missing)
    # a compact-backed layer cannot fetch cold experts (no registered row); see the kernel
    if hot_row is not None:
        row_map = hot_row.tolist()
        missing = [e for e in missing if row_map[e] >= 0]
    if frac_q16 > 0:
        m, q = len(missing), 1 << 16
        lo = (m * frac_q16) >> 16
        cost = lambda f: max(f * (q - frac_q16), (m - f) * frac_q16)  # noqa: E731
        max_fetch = lo if cost(lo) <= cost(lo + 1) else lo + 1
    num_fetch = min(len(missing), int(max_fetch))
    cache.num_missing_full.fill_(num_full)
    cache.num_indices.fill_(num_fetch)

    usage = cache.usage.tolist()
    for idx in range(num_fetch):
        expert = missing[idx]
        victim = min(range(cache.cache_size), key=lambda s: (usage[s], s))
        old_id = int(cache.id_of_slot[victim].item())
        if old_id >= 0:
            cache.slot_for_id.view(-1)[old_id] = -1
        cache.id_of_slot[victim] = layer_id * cache.num_experts + expert
        cache.slot_for_id[layer_id, expert] = victim
        cache.usage[victim] = step
        usage[victim] = step
        cache.evict_slots[idx] = victim
        if hot_row is not None:
            cache.src_indices[idx] = int(row_map[expert])  # compact row
        else:
            cache.src_indices[idx] = expert  # layer-local row

    if _HYBRID_FETCH_BY_RECENCY:
        for expert in seen:
            cache.expert_recency[layer_id, expert] = step

    # Overflow misses keep slot_for_id == -1, so the rewrite below yields -1 for them.
    flat = expert_ids.view(-1)
    for i in range(flat.numel()):
        flat[i] = int(cache.slot_for_id[layer_id, int(flat[i].item())].item())


def _materialize_layer_gpu(cache, layer_id: int) -> None:
    block = triton.next_power_of_2(max(cache.num_experts, cache.cache_size))
    _materialize_layer_kernel[(1,)](
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        layer_id,
        cache.num_experts,
        cache.cache_size,
        BLOCK=block,
    )


def _reset_cache_gpu(cache) -> None:
    block = 256
    total_ids = cache.num_layers * cache.num_experts
    grid = (triton.cdiv(max(total_ids, cache.cache_size), block),)
    _reset_cache_kernel[grid](
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.active_mask,
        cache.num_indices,
        total_ids,
        cache.num_experts,
        cache.cache_size,
        BLOCK=block,
    )


@triton.jit
def _reset_cache_kernel(
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    active_mask_ptr,
    num_indices_ptr,
    total_ids: tl.constexpr,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(slot_for_id_ptr + off, -1, mask=off < total_ids)
    tl.store(id_of_slot_ptr + off, -1, mask=off < cache_size)
    tl.store(usage_ptr + off, 0, mask=off < cache_size)
    tl.store(active_mask_ptr + off, 0, mask=off < num_experts)
    if tl.program_id(0) == 0:
        tl.store(step_ptr, 0)
        tl.store(num_indices_ptr, 0)


@triton.jit
def _materialize_layer_kernel(
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    evict_slots_ptr,
    src_indices_ptr,
    num_indices_ptr,
    layer_id: tl.constexpr,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.arange(0, BLOCK)
    expert_mask = off < num_experts
    slot_mask = off < cache_size
    slot = off

    base = layer_id * num_experts
    old_id = tl.load(id_of_slot_ptr + slot, mask=slot_mask, other=-1)
    # Flat ids make "belongs to this layer" a range check instead of a field compare.
    same_layer = slot_mask & (old_id >= base) & (old_id < base + num_experts)
    tl.store(id_of_slot_ptr + slot, -1, mask=same_layer)
    tl.store(usage_ptr + slot, 0, mask=same_layer)

    old_valid = expert_mask & (old_id >= 0) & (~same_layer)
    tl.store(slot_for_id_ptr + old_id, -1, mask=old_valid)

    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    tl.store(id_of_slot_ptr + slot, base + off, mask=expert_mask)
    tl.store(slot_for_id_ptr + base + off, slot, mask=expert_mask)
    tl.store(usage_ptr + slot, step, mask=expert_mask)
    tl.store(evict_slots_ptr + off, slot, mask=expert_mask)
    tl.store(src_indices_ptr + off, off, mask=expert_mask)  # layer-local row
    tl.store(num_indices_ptr, num_experts)




@triton.jit(do_not_specialize=["layer_id", "num_active", "max_fetch", "fetch_frac_q16", "fetch_cap", "cache_size", "id_of_slot_ptr", "usage_ptr"])
def _ensure_experts_hybrid_kernel(
    expert_ids_ptr,
    hot_row_of_ptr,  # [num_experts] int32 compact-row map; -1 = CPU-only (no fetch)
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    active_mask_ptr,
    evict_slots_ptr,
    src_indices_ptr,
    num_indices_ptr,
    num_missing_full_ptr,
    expert_recency_ptr,
    weights_ptr,
    cpu_ids_ptr,
    gpu_w_ptr,
    stats_ptr,
    stats_layer_ptr,
    layer_id,
    num_active,
    max_fetch,
    fetch_frac_q16,
    fetch_cap,
    num_experts: tl.constexpr,
    cache_size,
    seed,
    BLOCK_E: tl.constexpr,
    CHUNK: tl.constexpr,
    SMALL: tl.constexpr,
    BY_RECENCY: tl.constexpr,
    HAS_HOT: tl.constexpr,
    ROUND: tl.constexpr,
    CLAMP: tl.constexpr,
    STATS: tl.constexpr,
):
    """Capped-fetch timestamp-LRU (hybrid backend).

    Same as ``_ensure_experts_lru_v2_kernel`` but only ``min(num_missing, max_fetch)``
    missing experts are evicted-into / scheduled for copy; the overflow misses stay
    non-resident, so Phase 3 rewrites their positions to -1 (the layer computes those on
    the CPU). ``fetch_frac_q16`` > 0 (Q16 fixed point) replaces the fixed cap with the
    bandwidth-matched split ``~frac * num_missing`` (see the Phase-1 comment), computed
    in-kernel because ``num_missing`` only exists device-side (CUDA graph). ``num_indices``
    = the capped fetch count (copy_missing), ``num_missing_full`` = the pre-cap miss count
    (stats).

    Which misses to fetch is the cap policy. ``BY_RECENCY`` (default) fetches the experts
    most-recently active before this step (LRU on the expert, via ``expert_recency``),
    breaking ties toward the lower expert id -- this prioritizes *recurring* misses for
    caching, lowering the steady miss rate. Otherwise the lowest expert ids are fetched
    (``missing_rank``), the original routing-blind heuristic.

    Victims are the least-used slots, ties to the lowest slot. Every usage is at most the
    step before this launch, so the hit slots stamped with ``step`` in Phase 1 are never
    picked. A pool of at most CHUNK slots stays in registers (SMALL); a larger one is scanned
    from memory in CHUNK tiles per victim."""
    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    base = layer_id * num_experts

    # ---- Phase 1: active + missing over experts ----
    off_e = tl.arange(0, BLOCK_E)
    e_mask = off_e < num_experts
    is_active = tl.zeros((BLOCK_E,), dtype=tl.int1)
    for i in tl.range(num_active):
        e = tl.load(expert_ids_ptr + i)
        is_active = is_active | (off_e == e)
    tl.store(active_mask_ptr + off_e, is_active.to(tl.int32), mask=e_mask)
    slot = tl.load(slot_for_id_ptr + base + off_e, mask=e_mask, other=-1)
    is_missing = is_active & (slot == -1) & e_mask
    num_missing = tl.sum(is_missing.to(tl.int32))
    # A compact-backed layer can only fetch experts whose hot row exists (>= 0 in the
    # map); its cold experts have no registered row to DMA from, so they always take
    # the CPU route regardless of the fetch budget.
    if HAS_HOT:
        hrow = tl.load(hot_row_of_ptr + off_e, mask=e_mask, other=-1)
        is_fetchable = is_missing & (hrow >= 0)
    else:
        is_fetchable = is_missing
    n_cand = tl.sum(is_fetchable.to(tl.int32))
    # Cap the fetches; the overflow misses are computed on the CPU (left non-resident).
    if fetch_frac_q16 > 0:
        # Bandwidth-matched split (fetch_frac = pcie_bw / cpu_bw): fetch time scales with
        # F * (1 - frac), CPU time with (M - F) * frac; they balance at F = frac * M. Pick
        # the integer neighbor that minimizes the slower (max) side of the overlap.
        lo = (n_cand * fetch_frac_q16) >> 16
        if ROUND == 1:
            # step is unique per launch, so each (layer, step) draws its own number
            rem = (n_cand * fetch_frac_q16) & 0xFFFF
            u = tl.rand(seed, (step & 0x7FFFFFFF).to(tl.int32))
            max_fetch = lo + ((u * 65536.0) < rem.to(tl.float32)).to(tl.int32)
        elif ROUND == 2:
            max_fetch = (n_cand * fetch_frac_q16 + 0xFFFF) >> 16
        else:
            cost_lo = tl.maximum(lo * ((1 << 16) - fetch_frac_q16), (n_cand - lo) * fetch_frac_q16)
            cost_hi = tl.maximum(
                (lo + 1) * ((1 << 16) - fetch_frac_q16), (n_cand - lo - 1) * fetch_frac_q16
            )
            max_fetch = tl.where(cost_lo <= cost_hi, lo, lo + 1)
    is_hit = is_active & (slot >= 0)
    # a compact-backed layer has fewer fetchable candidates, and a pool smaller than
    # num_experts may hold fewer free slots than misses
    num_fetch = tl.minimum(
        tl.minimum(n_cand, tl.minimum(max_fetch, fetch_cap)), cache_size - tl.sum(is_hit.to(tl.int32))
    )
    tl.store(num_missing_full_ptr, num_missing.to(tl.int64))
    tl.store(num_indices_ptr, num_fetch.to(tl.int64))
    tl.store(usage_ptr + slot, step, mask=is_hit)

    # Fetch-selection priority: encode (recency desc, id asc) into one strictly-ordered
    # score so argmax has no ties (rec deltas are multiples of num_experts; the id term
    # spans only [0, num_experts), so it can only break exact-recency ties).
    if BY_RECENCY:
        rec = tl.load(expert_recency_ptr + base + off_e, mask=e_mask, other=-1).to(tl.int64)
        score = tl.where(
            is_fetchable, rec * num_experts + (num_experts - 1 - off_e), -1152921504606846976
        ).to(tl.int64)
    else:
        missing_rank = tl.cumsum(is_fetchable.to(tl.int32)) - 1

    # ---- Phase 2: evict victims by argmin(usage), only for the capped fetches ----
    if num_fetch > 0:
        off_k = tl.arange(0, CHUNK)
        tl.debug_barrier()  # the scan below must see the hit stamps
        if SMALL:
            u_reg = tl.load(usage_ptr + off_k, mask=off_k < cache_size, other=9223372036854775807).to(tl.int64)
        for i in tl.range(num_fetch):
            if SMALL:
                victim = tl.argmin(u_reg, axis=0).to(tl.int32)
                u_reg = tl.where(off_k == victim, step, u_reg)
            else:
                best_u = tl.full((), 9223372036854775807, tl.int64)
                best_s = tl.full((), 0, tl.int32)
                for c0 in tl.range(0, cache_size, CHUNK):
                    offs = c0 + off_k
                    uu = tl.load(usage_ptr + offs, mask=offs < cache_size, other=9223372036854775807).to(tl.int64)
                    cmin = tl.min(uu, axis=0)
                    carg = (tl.argmin(uu, axis=0) + c0).to(tl.int32)
                    take = cmin < best_u
                    best_s = tl.where(take, carg, best_s)
                    best_u = tl.where(take, cmin, best_u)
                victim = best_s
            old_id = tl.load(id_of_slot_ptr + victim)
            if old_id >= 0:
                tl.store(slot_for_id_ptr + old_id, -1)
            if BY_RECENCY:
                e = tl.argmax(score, axis=0).to(tl.int32)
                score = tl.where(off_e == e, -1152921504606846976, score)
            else:
                e = tl.sum(tl.where((missing_rank == i) & is_fetchable, off_e, 0))
            tl.store(id_of_slot_ptr + victim, base + e)
            tl.store(slot_for_id_ptr + base + e, victim)
            tl.store(usage_ptr + victim, step)
            tl.store(evict_slots_ptr + i, victim)
            if HAS_HOT:
                tl.store(src_indices_ptr + i, tl.load(hot_row_of_ptr + e))  # compact row
            else:
                tl.store(src_indices_ptr + i, e)  # layer-local row
            tl.debug_barrier()  # the next scan must see this victim's stamp
    tl.debug_barrier()  # Phase 3 reads slot_for_id entries written above

    # ---- Phase 3: rewrite expert_ids -> slot id (hit/fetched) or -1 (overflow -> CPU) ----
    for i in tl.range(num_active):
        e = tl.load(expert_ids_ptr + i)
        s = tl.load(slot_for_id_ptr + base + e)
        # every warp loads e, one thread stores below; a warp that loads after that store indexes with a slot id
        tl.debug_barrier()
        on_gpu = s >= 0
        tl.store(cpu_ids_ptr + i, tl.where(on_gpu, -1, e))
        tl.store(gpu_w_ptr + i, tl.where(on_gpu, tl.load(weights_ptr + i), 0.0))
        if CLAMP:
            s = tl.maximum(s, 0)
        tl.store(expert_ids_ptr + i, s)

    # stats_ptr: (missing, fetched, active, calls); stats_layer_ptr: the same per layer, [num_layers, 4]
    if STATS:
        n_active = tl.sum((is_active & e_mask).to(tl.int64))
        tl.store(stats_ptr + 0, tl.load(stats_ptr + 0) + num_missing)
        tl.store(stats_ptr + 1, tl.load(stats_ptr + 1) + num_fetch)
        tl.store(stats_ptr + 2, tl.load(stats_ptr + 2) + n_active)
        tl.store(stats_ptr + 3, tl.load(stats_ptr + 3) + 1)
        row = stats_layer_ptr + layer_id * 4
        tl.store(row + 0, tl.load(row + 0) + num_missing)
        tl.store(row + 1, tl.load(row + 1) + num_fetch)
        tl.store(row + 2, tl.load(row + 2) + n_active)
        tl.store(row + 3, tl.load(row + 3) + 1)

    # Bump every active expert's recency to this step (LRU on the expert): an overflow miss
    # computed on the CPU now ranks high if it recurs, so it gets fetched next time.
    if BY_RECENCY:
        step_vec = tl.zeros((BLOCK_E,), dtype=tl.int64) + step
        tl.store(expert_recency_ptr + base + off_e, step_vec, mask=is_active & e_mask)


@triton.jit(do_not_specialize=["threshold"])
def _prefill_hit_compact_kernel(
    slot_ptr,     # [num_experts] int32: this layer's slot_for_id row
    dst_ptr,      # [num_experts] int32 out: buffer rows (expert ids), compacted
    src_ptr,      # [num_experts] int32 out: pool-local slots, compacted
    num_ptr,      # [1] int64 out: hit count
    threshold,    # first pool-local slot outside the staging buffers
    num_experts,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    lane = offs < num_experts
    slots = tl.load(slot_ptr + offs, mask=lane, other=-1)
    is_hit = lane & (slots >= threshold)
    pos = tl.cumsum(is_hit.to(tl.int32)) - 1
    tl.store(dst_ptr + pos, offs.to(tl.int32), mask=is_hit)
    tl.store(src_ptr + pos, slots, mask=is_hit)
    tl.store(num_ptr, tl.sum(is_hit.to(tl.int64)))


@triton.jit(do_not_specialize=["buffer_base"])
def _prefill_ondemand_compact_kernel(
    slot_ptr,      # [num_experts] int32: this layer's LIVE slot_for_id row
    touched_ptr,   # [num_experts] int32: 1 for experts this chunk's routing selected
    hit_dst,       # [num_experts] int32 out: buffer rows of touched hits
    hit_src,       # [num_experts] int32 out: slots of touched hits, relative to their pool
    hit_num,       # [1] int64 out
    miss_dst,      # [num_experts] int32 out: buffer rows of the PCIe-assigned misses
    miss_src,      # [num_experts] int32 out: host bank rows (== expert id)
    miss_num,      # [1] int64 out: PCIe-assigned miss count
    cpu_mask,      # [num_experts] int32 out: 1 for misses assigned to the CPU executor
    buffer_base,   # buffer_id * num_experts
    threshold,     # 2 * num_experts
    num_experts,
    frac_q16,      # Q16 share of this layer's misses that stream over PCIe
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    lane = offs < num_experts
    slots = tl.load(slot_ptr + offs, mask=lane, other=-1)
    touched = tl.load(touched_ptr + offs, mask=lane, other=0) != 0
    is_hit = lane & touched & (slots >= threshold)
    is_miss = lane & touched & (slots < threshold)
    pos_h = tl.cumsum(is_hit.to(tl.int32)) - 1
    tl.store(hit_dst + pos_h, (buffer_base + offs).to(tl.int32), mask=is_hit)
    tl.store(hit_src + pos_h, slots, mask=is_hit)  # pool-local: pool offsets start at each slot range
    tl.store(hit_num, tl.sum(is_hit.to(tl.int64)))
    # Split the misses in expert-id order: the first frac_q16 share streams over PCIe
    # into the double buffer, the rest are computed on the CPU straight from the host
    # banks. frac_q16 == 1<<16 puts every miss on the PCIe plan and leaves cpu_mask
    # all zero, which is the pre-split behaviour bit for bit.
    pos_m = tl.cumsum(is_miss.to(tl.int32)) - 1
    n_pcie = (tl.sum(is_miss.to(tl.int32)) * frac_q16) >> 16
    to_pcie = is_miss & (pos_m < n_pcie)
    to_cpu = is_miss & (pos_m >= n_pcie)
    pos_p = tl.cumsum(to_pcie.to(tl.int32)) - 1
    tl.store(miss_dst + pos_p, (buffer_base + offs).to(tl.int32), mask=to_pcie)
    tl.store(miss_src + pos_p, offs.to(tl.int32), mask=to_pcie)
    tl.store(miss_num, tl.sum(to_pcie.to(tl.int64)))
    tl.store(cpu_mask + offs, to_cpu.to(tl.int32), mask=lane)
