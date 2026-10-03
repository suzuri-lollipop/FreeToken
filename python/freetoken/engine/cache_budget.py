"""Pure GPU-memory budget policy shared by startup auto-sizing and runtime rebuild.

No torch/GPU side effects: every function here is integer/byte arithmetic over already-
measured quantities, so it is unit-testable without a device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from freetoken.utils import div_ceil

if TYPE_CHECKING:
    import torch


def expert_bytes_per_slot(sources: dict[str, "list[torch.Tensor]"]) -> int:
    """Bytes one expert slot occupies on GPU: summed row bytes over all banks.

    Each bank source is per-layer ``[num_experts, *row_shape]`` tensors and is
    already TP-sharded upstream, so the per-row byte count is the per-rank slot
    size.
    """
    # marlin/b12x gate_up/down alpha scales are fixed [L*E] residency (do not scale
    # with cache_size), so they are intentionally excluded from the per-slot growth term.
    # tensor[0].numel() is the per-row element count (one expert slot); see the matching
    # slot-byte idiom in kvcache/linear_state_pool.py and kvcache/dsv4_paged_pool.py.
    return sum(t[0][0].numel() * t[0].element_size() for t in sources.values())


def slot_cache_expert_cap(
    num_layers: int,
    num_experts: int,
    layer_residency: "list[str] | None",
    *,
    prefill_overlap: bool,
) -> int:
    """Slots the GPU expert cache can ever fill, given each host bank layer's residency.

    Only PINNED banks carry a device address, so only their layers take LRU slots. A
    LOCKED/PAGEABLE layer touches the cache exactly once per prefill chunk -- the
    whole-layer pageable staging in ``copy_missing`` -- which occupies
    ``[0, num_experts)`` (two such buffers with prefill overlap). Sizing for their
    experts buys dead bytes that the KV pool could use, and one huge contiguous bank.

    ``layer_residency is None`` is a loader that settles banks without per-layer labels
    (see ``moe.expert_banks._echo_residency``): report the whole-model ceiling rather
    than shrink a residency we cannot see.
    """
    from freetoken.moe.host_banks import HostResidency

    total_experts = num_layers * num_experts
    if not layer_residency or len(layer_residency) != num_layers:
        return total_experts
    staging = (2 if prefill_overlap else 1) * num_experts
    pinned = sum(1 for r in layer_residency if r == HostResidency.PINNED.value)
    if pinned == num_layers:
        return total_experts
    return (pinned * num_experts) + staging


def has_explicit_cache_sizing(moe_cache_size: int, moe_cache_rate: "float | None") -> bool:
    """Whether the user pinned the expert slot cache with --moe-cache-size / --moe-cache-rate.

    An explicit size suppresses the --moe-cache-auto fallback, and with it the post-init
    headroom growth, so --memory-ratio's pool budget then fully determines the footprint.

    Read this BEFORE --moe-cache-auto resolves: the resolution writes its size back into
    ``config.moe_cache_size``, after which an auto-sized cache is indistinguishable from a
    user-pinned one. ``headroom_growth_eligible`` takes the latched answer."""
    return moe_cache_size > 0 or (moe_cache_rate is not None and moe_cache_rate > 0)


def headroom_growth_eligible(moe_cache_auto: bool, sizing_explicit: bool) -> bool:
    """Whether the post-init expert-cache growth may spend the ``(1 - ratio)`` headroom.

    ``sizing_explicit`` is the latched pre-resolution reading of
    ``has_explicit_cache_sizing``; passing the post-resolution config instead silently
    disables the growth on every auto-sized run."""
    return moe_cache_auto and not sizing_explicit


def net_cache_budget_bytes(
    memory_ratio: float,
    baseline_free: int,
    weights_bytes: int,
    fixed_cache_size: int,
    *,
    device_total: int = 0,
    nonpool_overhead_bytes: int = 0,
) -> int:
    """Net GPU bytes available for the MoE + KV pools under --memory-ratio.

    With a device total (whole-VRAM view at startup), the ratio caps the engine's TOTAL
    per-rank footprint: pools get ``ratio x device_total`` minus what is already committed
    outside them (resident weights + the measured CUDA-context/NCCL/allocator overhead).
    Without one, the historical reading applies: ``ratio`` of the pre-model free baseline
    minus weights. The ``(1-ratio)`` remainder is the graph/activation headroom. Single
    source of truth for startup auto-sizing and the runtime-rebuild fit check."""
    legacy = int(memory_ratio * baseline_free) - weights_bytes - fixed_cache_size
    if device_total <= 0:
        return legacy
    capped = (
        int(memory_ratio * device_total)
        - weights_bytes
        - nonpool_overhead_bytes
        - fixed_cache_size
    )
    # A co-tenant already holding VRAM can push the capped figure below the legacy one;
    # the cap guards against exceeding ratio x device, it must not silently enlarge the plan.
    return min(legacy, capped)


def growth_cap_bytes(
    memory_ratio: float,
    device_total: int,
    current_footprint_bytes: int,
    reserve_bytes: int,
) -> int:
    """Bytes the post-init expert-cache growth may take without breaking --memory-ratio.

    ``memory_ratio x device_total`` is the ceiling on the engine's WHOLE footprint, so the
    growth may only close the gap between what the startup plan actually allocated and that
    ceiling. The ``(1 - ratio)`` remainder stays reserved for CUDA-graph capture and the
    largest prefill chunk's activations -- it is what the ratio promises to leave free, so
    spending it would make the flag stop being a footprint cap. ``reserve_bytes`` is a
    second, independent floor for those same consumers; it is what binds when the ratio
    itself is set near 1. Returns 0 when the plan already sits at or above the ceiling."""
    room = min(
        device_total - current_footprint_bytes - reserve_bytes,
        int(memory_ratio * device_total) - current_footprint_bytes,
    )
    return max(0, room)


def required_bytes(
    moe_cache_size: int, num_pages: int, per_expert_bytes: int, cache_per_page: int
) -> int:
    """GPU bytes a ``(moe_cache_size, num_pages)`` geometry occupies (MoE slots + KV pages)."""
    return moe_cache_size * per_expert_bytes + num_pages * cache_per_page


def padded_graph_rows(bs: int) -> int:
    """The decode-graph size a ``bs``-row batch pads up to.

    Mirrors engine/graph.py's default capture candidates (1..8 exact, then stride 4
    to 32 and stride 8 above) so budget planning charges the dummy rows' expert
    demand. An explicit --cuda-graph-bs list can differ; this is a planning
    estimate, not a promise."""
    if bs <= 8:
        return max(1, bs)
    if bs <= 32:
        return -(-bs // 4) * 4
    return -(-bs // 8) * 8


def expert_working_set_slots(
    num_layers: int,
    num_experts: int,
    top_k: int,
    batch_rows: int,
    factor: float,
) -> int:
    """Expert slots the decode LRU wants at ``batch_rows`` concurrent rows.

    One step touches ``num_experts * (1 - ((E-k)/E)^rows)`` distinct experts per
    layer (uniform-routing upper bound; measured routing is more concentrated).
    LRU's miss-rate knee sits near 1.5-2x that per-step working set -- at ~1.0x
    the reuse distance equals the capacity and every step re-misses most of it --
    so ``factor`` scales the claim the expert cache makes on the shared budget."""
    if batch_rows <= 0 or num_experts <= 0:
        return 0
    untouched = ((num_experts - min(top_k, num_experts)) / num_experts) ** batch_rows
    distinct_per_layer = num_experts * (1.0 - untouched)
    return int(factor * num_layers * distinct_per_layer)


def plan_linear_state_slots(
    *,
    slots_desired: int,
    slots_floor: int,
    bytes_per_slot: int,
    poolable_bytes: int,
    expert_slots_target: int,
    per_expert_bytes: int,
) -> int:
    """GDN state-pool slot count that protects the decode expert working set.

    ``poolable_bytes`` is what remains for the GDN pool + the expert slot cache
    after the (fixed) KV geometry. The expert cache claims up to
    ``expert_slots_target`` slots -- the high-batch decode working-set knee --
    and the pool's cross-request SNAPSHOT cache yields its ratio-sized wish
    before the per-request working-set floor (live + ping-pong + committed
    slots, below which admission deadlocks) is touched. Returns a slot count in
    ``[slots_floor, slots_desired]``; callers must agree on it cross-rank."""
    assert bytes_per_slot > 0 and per_expert_bytes > 0
    assert slots_floor <= slots_desired, "floor above desired: sizing formulas disagree"
    gdn_bytes = poolable_bytes - expert_slots_target * per_expert_bytes
    slots = gdn_bytes // bytes_per_slot
    return max(slots_floor, min(slots_desired, slots))


def plan_cache_budget(
    budget_bytes: int,
    per_expert_bytes: int,
    cache_per_page: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_pages: int,
    max_slots: int,
) -> tuple[int, int, bool]:
    """Split ``budget_bytes`` MoE-first into (moe_cache_size, num_pages, prefill_overlap).

    ``budget_bytes`` is the net pool for MoE cache + KV cache (caller already subtracted
    weights + fixed_cache_size; the (1-memory_ratio) remainder is the graph headroom).
    Experts greedily fill the budget after reserving ``kv_reserve_pages`` for KV, clamped
    to ``[floor, min(total_experts, max_slots)]`` (floor is ``2*num_experts`` when prefill
    overlap is feasible else ``num_experts``); KV pages take whatever remains.
    """
    assert per_expert_bytes > 0, "per_expert_bytes must be positive"
    assert cache_per_page > 0, "cache_per_page must be positive (owned-KV models unsupported here)"

    hi = min(total_experts, max_slots)
    # Prefill overlap borrows two full expert-layer buffers, so it needs >= 2*num_experts
    # slots; disable it (and lower the floor) if the cap cannot fit that.
    overlap = prefill_overlap and hi >= 2 * num_experts
    lo = 2 * num_experts if overlap else num_experts
    assert hi >= lo, f"slot cap {hi} below the minimum {lo} slots"

    kv_reserve_bytes = kv_reserve_pages * cache_per_page
    # MoE-priority: reserve KV first, then experts greedily take the remaining budget.
    raw = (budget_bytes - kv_reserve_bytes) // per_expert_bytes
    moe_cache_size = max(lo, min(raw, hi))
    # A tiny budget may have forced moe_cache_size below 2*num_experts even with overlap on.
    overlap = overlap and moe_cache_size >= 2 * num_experts

    remaining = budget_bytes - moe_cache_size * per_expert_bytes
    num_pages = max(remaining // cache_per_page, kv_reserve_pages)
    # A tiny budget can floor num_pages at kv_reserve_pages even when ``remaining`` is below
    # the reserve (or negative), yielding a plan that exceeds budget_bytes. Reject here so
    # --moe-cache-auto fails in arithmetic instead of OOMing in a later CUDA allocation.
    total = moe_cache_size * per_expert_bytes + num_pages * cache_per_page
    assert total <= budget_bytes, (
        f"cache budget too small: minimum plan (moe={moe_cache_size} slots, "
        f"kv={num_pages} pages) needs {total} B > budget {budget_bytes} B "
        "(raise memory_ratio, lower kv_reserve_tokens, or free GPU memory)"
    )
    assert num_pages > 1, "not enough memory for KV cache after MoE allocation"
    return moe_cache_size, num_pages, overlap


def replan_local_slots(
    *,
    budget_bytes: int,
    num_pages: int,
    cache_per_page: int,
    per_expert_bytes: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    max_slots: int,
    skew_bytes: int = 0,
) -> int:
    """This rank's expert slot count for a FIXED KV geometry under an uneven TP shard.

    ``plan_cache_budget`` solved the shared byte budget with the cross-rank MAX
    per-expert bytes, so ``num_pages`` is identical on every rank (the scheduler on
    rank 0 allocates page indices that every rank's pool must honor). This re-splits
    the same bytes into this rank's own slots: a rank whose shard of each expert is
    thinner gets MORE slots (a bigger resident working set, fewer PCIe misses), which
    is exactly the capacity feedback the bandwidth-weighted split is buying.
    """
    hi = min(total_experts, max_slots)
    lo = 2 * num_experts if prefill_overlap else num_experts
    # skew_bytes: per-rank slot-budget tilt (FREETOKEN_SLOT_BUDGET_SKEW_MIB). A rank
    # whose fetch path is the straggler (fatter shard bytes per miss) banks extra slots
    # from the (1 - memory_ratio) headroom; the peer gives the same bytes back. The KV
    # geometry is untouched (num_pages stays cross-rank identical).
    budget_bytes += skew_bytes
    slot_bytes = budget_bytes - num_pages * cache_per_page
    size = max(lo, min(slot_bytes // per_expert_bytes, hi))
    total = size * per_expert_bytes + num_pages * cache_per_page
    assert total <= budget_bytes, (
        f"local slot replan exceeds the budget: {size} slots x {per_expert_bytes} B "
        f"+ {num_pages} pages x {cache_per_page} B = {total} B > {budget_bytes} B"
    )
    return size


def resolve_moe_cache_auto(
    *,
    baseline_free: int,
    weights_bytes: int,
    memory_ratio: float,
    cache_per_page: int,
    fixed_cache_size: int,
    per_expert_bytes: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_tokens: int,
    page_size: int,
    max_slots: int | None = None,
    device_total: int = 0,
    nonpool_overhead_bytes: int = 0,
    per_expert_bytes_ref: int | None = None,
    slot_budget_skew_bytes: int = 0,
    fixed_cache_size_ref: int | None = None,
) -> tuple[int, int, bool]:
    """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

    ``max_slots`` is the expert kernel's addressable slot limit; the plan never exceeds it.

    ``per_expert_bytes_ref`` is the cross-rank MAX per-slot bytes under an uneven
    (bandwidth-weighted) expert shard: the KV geometry is planned from it so every rank
    agrees on num_pages, then this rank's slot count is re-split from its own bytes.

    Applies memory_ratio exactly once via net_cache_budget_bytes (the usage-cap reading
    when a device total is given), then defers the MoE-vs-KV split to plan_cache_budget.
    The (1-memory_ratio) remainder is the CUDA-graph/activation headroom.
    """
    budget_bytes = net_cache_budget_bytes(
        memory_ratio,
        baseline_free,
        weights_bytes,
        fixed_cache_size,
        device_total=device_total,
        nonpool_overhead_bytes=nonpool_overhead_bytes,
    )
    max_slots = total_experts if max_slots is None else min(max_slots, total_experts)
    kv_reserve_pages = div_ceil(kv_reserve_tokens, page_size)
    ref_bytes = per_expert_bytes if per_expert_bytes_ref is None else per_expert_bytes_ref
    # Uneven dense head split: the GDN state pool (part of fixed_cache_size) is rank-
    # local, but the KV geometry must stay cross-rank identical -- plan the shared
    # geometry from the MAX fixed cost, then re-split this rank's own (cheaper) bytes
    # into slots, exactly like the uneven expert shard does for per-expert bytes.
    plan_budget = budget_bytes
    if fixed_cache_size_ref is not None and fixed_cache_size_ref != fixed_cache_size:
        plan_budget = net_cache_budget_bytes(
            memory_ratio,
            baseline_free,
            weights_bytes,
            fixed_cache_size_ref,
            device_total=device_total,
            nonpool_overhead_bytes=nonpool_overhead_bytes,
        )
    size, num_pages, overlap = plan_cache_budget(
        budget_bytes=plan_budget,
        per_expert_bytes=ref_bytes,
        cache_per_page=cache_per_page,
        num_experts=num_experts,
        total_experts=total_experts,
        prefill_overlap=prefill_overlap,
        kv_reserve_pages=kv_reserve_pages,
        max_slots=max_slots,
    )
    if ref_bytes != per_expert_bytes or slot_budget_skew_bytes or plan_budget != budget_bytes:
        size = replan_local_slots(
            budget_bytes=budget_bytes,
            num_pages=num_pages,
            cache_per_page=cache_per_page,
            per_expert_bytes=per_expert_bytes,
            num_experts=num_experts,
            total_experts=total_experts,
            prefill_overlap=overlap,
            max_slots=max_slots,
            skew_bytes=slot_budget_skew_bytes,
        )
    return size, num_pages, overlap
