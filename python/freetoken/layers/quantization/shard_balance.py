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
_DENSE_WEIGHTS: "tuple[float, ...] | None" = None


def set_expert_shard_weights(weights: "tuple[float, ...] | None") -> None:
    global _WEIGHTS
    _WEIGHTS = None if weights is None else tuple(float(w) for w in weights)


def get_expert_shard_weights() -> "tuple[float, ...] | None":
    return _WEIGHTS


def set_dense_shard_weights(weights: "tuple[float, ...] | None") -> None:
    """Per-rank share of the DENSE projections (attention/GDN head splits).

    The decode step's critical path is the slow rank's serial chain; when one rank
    also carries the fatter expert shard (more PCIe fetch + expert GEMM bytes), the
    peer's K2 all-reduce spin shows it idles ms per step. Tilting the head split
    moves dense FLOPs/bytes onto the idling rank. ``None`` keeps the even split;
    consumers snap the fractions to whole heads."""
    global _DENSE_WEIGHTS
    _DENSE_WEIGHTS = None if weights is None else tuple(float(w) for w in weights)


def get_dense_shard_weights() -> "tuple[float, ...] | None":
    return _DENSE_WEIGHTS


def dense_head_shares(total_heads: int, size: int) -> "list[int] | None":
    """Per-rank head counts under the dense weights, or None (even-split caller path).

    Head-snapped cumulative rounding: identical on every rank for the same weights,
    so all ranks agree without another collective. Every rank must get >= 1 head."""
    w = _DENSE_WEIGHTS
    if w is None or size < 2 or len(w) != size:
        return None
    shares: list[int] = []
    prev = 0
    cum = 0.0
    for r in range(size):
        cum += w[r] * total_heads
        bound = total_heads if r == size - 1 else int(round(cum))
        shares.append(bound - prev)
        prev = bound
    if any(sh < 1 for sh in shares) or sum(shares) != total_heads:
        raise ValueError(
            f"dense shard weights {_DENSE_WEIGHTS} cannot split {total_heads} heads "
            f"across {size} ranks with >= 1 head each"
        )
    # Measured (qwen4_exp GDN, 2 ranks): an EVEN head split (6/10 of 16, 18/30 of 48)
    # reproduces det4-stable text, while an odd split (7/9, 20/28 -- e.g. weights
    # 0.42/0.58) makes every prompt collapse to one identical garbage completion
    # (suspected head-packing assumption in a GDN kernel path). Fail loudly at
    # startup until the root cause is pinned; even splits only.
    if any(sh % 2 for sh in shares):
        raise ValueError(
            f"dense shard weights {_DENSE_WEIGHTS} produce odd head shares {shares} "
            f"for {total_heads} heads; only even per-rank head counts are supported"
        )
    return shares


def dense_local_heads(total_heads: int, rank: int, size: int) -> "int | None":
    shares = dense_head_shares(total_heads, size)
    return None if shares is None else shares[rank]


def dense_head_range(total_heads: int, rank: int, size: int) -> "tuple[int, int] | None":
    """This rank's [lo, hi) HEAD index range, for loader row slicing (or None)."""
    shares = dense_head_shares(total_heads, size)
    if shares is None:
        return None
    lo = sum(shares[:rank])
    return lo, lo + shares[rank]
