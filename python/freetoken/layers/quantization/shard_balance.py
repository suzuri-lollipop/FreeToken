"""Process-wide registry of bandwidth-weighted expert TP shard weights.

Two cards in one box routinely sit in different PCIe slots (measured on a 2-GPU
rig: gen5 x16 = ~21 GB/s next to gen4 x4 = ~6.6 GB/s). With the historical even
expert split both ranks pull the same miss bytes per decode step, so the slow
link sets the step time while the fast one idles. The engine probes each rank's
H2D bandwidth once at startup, all-gathers the result so every rank resolves the
SAME split, and records the per-rank weights here BEFORE the model is built:
``MoEConfig.local_intermediate_range`` turns them into whole-group row ranges,
and the loader's ``pack()``, the host banks and the slot cache all derive their
shapes from that config. ``None`` (the default) keeps the even split.

Only the offload strategy consumes the weights: resident (fused) experts never
cross PCIe, so an uneven split there would only unbalance compute.
"""

from __future__ import annotations

_WEIGHTS: "tuple[float, ...] | None" = None


def set_expert_shard_weights(weights: "tuple[float, ...] | None") -> None:
    global _WEIGHTS
    _WEIGHTS = None if weights is None else tuple(float(w) for w in weights)


def get_expert_shard_weights() -> "tuple[float, ...] | None":
    return _WEIGHTS
