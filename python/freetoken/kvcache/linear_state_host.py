"""Host-tiered GDN snapshot store (hybrid models).

One pinned host buffer per snapshot capacity: when evict_mamba would LOSE an internal
node's GDN snapshot (tombstone), the CacheManager stashes the slot's bytes here, frees
the VRAM slot, and the node keeps ``mamba_host_id`` as a resumable boundary. A later
prefix hit restores straight into the request's live slot (H2D) instead of re-prefilling
the GDN recurrence. Snapshot bytes are TP-local; the per-rank store mirrors the pool it
serves, and both ranks drive it through the same replicated eviction sequence, so buffer
allocation stays deterministic across ranks.

Buffer count is capped: on exhaustion ``try_alloc`` returns None and the save is skipped
(graceful degradation to today's lose-the-snapshot behavior). All D2H/H2D copies run on
the engine stream (the caller issues them from the scheduler's stream context), so buffer
reuse and slot reuse are stream-ordered and never need host synchronization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.kvcache.linear_state_pool import LinearStatePool

logger = init_logger(__name__)


class LinearStateHostCache:
    def __init__(self, pool: "LinearStatePool", device: torch.device, budget_mb: int) -> None:
        slot_bytes = pool.bytes_per_slot()
        num_buffers = max(1, (budget_mb << 20) // slot_bytes) if slot_bytes else 0
        self._pool = pool
        self._slot_bytes = slot_bytes
        self._num_buffers = num_buffers
        self._buf: torch.Tensor | None = None
        self._free: list[int] = []
        if num_buffers:
            # Pinned on CUDA so the D2H/H2D copies are async DMA; plain host on CPU (unit tests).
            self._buf = torch.empty(
                num_buffers * slot_bytes, dtype=torch.uint8,
                pin_memory=(device.type == "cuda"),
            )
            self._free = list(range(num_buffers))
            logger.info_rank0(
                f"GDN snapshot host tier: {num_buffers} buffers x {slot_bytes >> 20} MiB "
                f"({budget_mb} MiB budget)"
            )

    @property
    def enabled(self) -> bool:
        return self._buf is not None

    def try_alloc(self) -> int | None:
        if not self._free:
            return None
        return self._free.pop()

    def stash_to(self, buf: int, slot: int) -> None:
        self._pool.stash_to(self._buf, buf * self._slot_bytes, slot)

    def restore_from(self, buf: int, slot: int) -> None:
        self._pool.restore_from(self._buf, buf * self._slot_bytes, slot)

    def free(self, buf: int | None) -> None:
        if buf is not None:
            self._free.append(buf)