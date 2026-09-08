"""Pure GPU-memory budget policy shared by startup auto-sizing and runtime rebuild.

No torch/GPU side effects: every function here is integer/byte arithmetic over already-
measured quantities, so it is unit-testable without a device.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Sequence

from freetoken.utils import div_ceil

if TYPE_CHECKING:
    import torch


def _cgroup_mem_limit() -> int | None:
    """Cgroup v2 memory.max, or None when unlimited / unavailable."""
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            raw = f.read().strip()
        if raw == "max":
            return None
        return int(raw)
    except (OSError, ValueError):
        return None


def _total_ram_bytes() -> int:
    """Total physical RAM, falling back to the cgroup limit when it is tighter."""
    try:
        total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        total = 8 << 30
    cg = _cgroup_mem_limit()
    return min(total, cg) if cg is not None else total


def host_memory_budget_bytes(host_memory_ratio: float, reserved: int = 0) -> int:
    """Total host RAM the engine may consume for expert banks + pinned tables.

    ``host_memory_ratio`` (0, 1] scales total physical RAM (or the cgroup limit,
    whichever is tighter); ``reserved`` subtracts bytes already committed outside
    the expert banks (e.g. the Qwen3.8 PLE n-gram table).
    """
    return max(0, int(host_memory_ratio * _total_ram_bytes()) - reserved)


def host_pin_allowance(mem_available: int, host_memory_ratio: float) -> int:
    """Machine-wide host bytes the expert banks may hold NON-RECLAIMABLE (pinned or mlocked).

    ``host_memory_ratio`` is a floor on free RAM, not a cap on engine usage. Sizing against
    ``ratio * total_ram`` alone ignores whatever is already resident, so on a busy desktop
    the engine can honour the cap and still leave no free RAM. ``mem_available`` is measured
    once the weights and host tables are resident, so every rank's own footprint is already
    deducted from it. Anything the banks hold above this allowance must stay pageable so the
    kernel can push it to swap and keep ``(1 - ratio)`` of RAM free.
    """
    total = _total_ram_bytes()
    # Same int(ratio * total) as host_memory_budget_bytes, so the floor is exactly the
    # complement of what that budget allows -- (1 - ratio) * total would drift a byte on
    # ratios float cannot represent.
    free_floor = total - int(host_memory_ratio * total)
    return max(0, mem_available - free_floor)


def cpu_layer_count(bank_bytes: int, num_layers: int, pin_allowance: int) -> int:
    """Layers that must leave the pinned set for the banks to fit ``pin_allowance``.

    ``bank_bytes`` is the machine-wide total, not a per-rank shard: at TP>1 every rank pins
    its own copy, so the aggregate is what has to fit. 0 when everything already fits.
    """
    if num_layers <= 0 or bank_bytes <= pin_allowance:
        return 0
    return min(num_layers, div_ceil(num_layers * (bank_bytes - pin_allowance), bank_bytes))


def watchdog_nudge_bytes(deficit: int, mem_available: int, ceiling: int = 2 << 30) -> int:
    """Anonymous bytes the host-mem watchdog may allocate to force one reclaim pass.

    Bounded by what is actually free: allocating more than ``mem_available`` turns the nudge
    into the very pressure it exists to relieve -- the allocating thread enters direct reclaim,
    and the new anonymous pages themselves need swap. Half of free RAM is the ceiling because
    the nudge is freed immediately after, so it only has to be large enough to be noticed.
    """
    return max(0, min(deficit + (256 << 20), ceiling, mem_available // 2))


def plan_pageout(sizes: Sequence[int], budget: int, cursor: int) -> tuple[list[int], int]:
    """Indices to evict this tick, sweeping from ``cursor`` until ``budget`` bytes are covered.

    Rotating rather than always restarting at zero: the pageable banks are the CPU-decode
    layers' experts, so re-advising the same head of the list every tick evicts exactly the
    ranges the executor most recently faulted back in. Returns (indices, next_cursor); one
    sweep never advises an index twice, so a budget above the total covers everything once.
    """
    n = len(sizes)
    if n == 0:
        return [], 0
    start = cursor % n
    if budget <= 0:
        return [], start
    chosen: list[int] = []
    spent = 0
    for offset in range(n):
        idx = (start + offset) % n
        chosen.append(idx)
        spent += sizes[idx]
        if spent >= budget:
            break
    return chosen, (start + len(chosen)) % n


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


def net_cache_budget_bytes(
    memory_ratio: float, baseline_free: int, weights_bytes: int, fixed_cache_size: int
) -> int:
    """Net GPU bytes available for the MoE + KV pools: ``memory_ratio`` of the pre-model
    baseline minus weights and fixed (non-paged) cache. The ``(1-memory_ratio)`` remainder
    is the CUDA-graph/activation headroom. Single source of truth for startup auto-sizing
    and the runtime-rebuild fit check."""
    return int(memory_ratio * baseline_free) - weights_bytes - fixed_cache_size


def required_bytes(
    moe_cache_size: int, num_pages: int, per_expert_bytes: int, cache_per_page: int
) -> int:
    """GPU bytes a ``(moe_cache_size, num_pages)`` geometry occupies (MoE slots + KV pages)."""
    return moe_cache_size * per_expert_bytes + num_pages * cache_per_page


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
    quant_format: str,
) -> tuple[int, int, bool]:
    """Resolve --moe-cache-auto into (moe_cache_size, num_pages, prefill_overlap).

    Applies memory_ratio to the persisted pre-model baseline exactly once, then defers
    the MoE-vs-KV split to plan_cache_budget. The (1-memory_ratio) remainder is the
    CUDA-graph/activation headroom (not subtracted here).
    """
    budget_bytes = net_cache_budget_bytes(memory_ratio, baseline_free, weights_bytes, fixed_cache_size)
    max_slots = 992 if quant_format == "nvfp4_marlin" else total_experts
    kv_reserve_pages = div_ceil(kv_reserve_tokens, page_size)
    return plan_cache_budget(
        budget_bytes=budget_bytes,
        per_expert_bytes=per_expert_bytes,
        cache_per_page=cache_per_page,
        num_experts=num_experts,
        total_experts=total_experts,
        prefill_overlap=prefill_overlap,
        kv_reserve_pages=kv_reserve_pages,
        max_slots=max_slots,
    )
