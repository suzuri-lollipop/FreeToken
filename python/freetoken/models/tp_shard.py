"""Rank-local weight slicing for tensor parallelism, shared by the model families' readers.

A reader fuses checkpoint leaves into the buffers the model declares, so every rank reads the
same stream and keeps only its own heads / intermediate columns. ``TpShard`` knows which axis
of which leaf splits, how many global units that axis holds (so a block scale narrows by its
own ratio), and which leaves carry several head runs inside one tensor (GDN's ``in_proj_qkv``
and ``conv1d`` store q | k | v back to back). Leaves it does not list stay replicated: the
router must send every rank's tokens to the same experts, and a sparse indexer must pick the
same blocks on every rank.
"""

from __future__ import annotations

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_ceil

ROWS, COLS = 0, 1
# the vocab shard is rounded up and clamped, exactly like VocabParallelEmbedding sizes it
CEIL_LEAVES = frozenset({"lm_head", "embed_tokens"})

# checkpoint leaf -> the axis of its weight this rank splits
LEAF_AXIS = {
    "q_proj": ROWS, "k_proj": ROWS, "v_proj": ROWS,
    "gate_proj": ROWS, "up_proj": ROWS,
    "in_proj_qkv": ROWS, "in_proj_z": ROWS, "in_proj_b": ROWS, "in_proj_a": ROWS,
    "linear_attn.conv1d": ROWS, "A_log": ROWS, "dt_bias": ROWS,
    "o_proj": COLS, "down_proj": COLS, "out_proj": COLS,
    "lm_head": ROWS, "embed_tokens": ROWS,
}
# state-dict roles, i.e. the trailing key component that names a tensor of a module
ROLE_SUFFIXES = frozenset({"weight", "weight_scale", "weight_scale_inv", "weight_global", "input_scale", "bias"})
# leaves a replicated module reuses (qwen4's ple.conv1d is a depthwise conv over hc*hidden,
# not the GDN conv), so they only shard under the owner-qualified leaf
CONTEXT_LEAVES = frozenset({"conv1d"})


def module_leaf(name: str) -> str:
    """The module a state-dict key belongs to: ``...o_proj.weight`` -> ``o_proj``.

    A context leaf carries its owner module too: ``...linear_attn.conv1d.weight`` ->
    ``linear_attn.conv1d``, ``...ple.conv1d.weight`` -> ``ple.conv1d`` (replicated).

    Leaves outside ``LEAF_AXIS`` (routers, indexer projections, the HC and PLE mixers, every
    norm) are replicated, so passing them through the slicer is a no-op by design.
    """
    module, _, last = name.rpartition(".")
    if last in ROLE_SUFFIXES:
        parent, _, leaf = module.rpartition(".")
    else:
        parent, leaf = module, last
    if leaf in CONTEXT_LEAVES and parent:
        leaf = f"{parent.rpartition('.')[2]}.{leaf}"
    return leaf


def _leaf_units(config) -> dict[str, int]:
    """Global units along each sharded leaf's split axis, from the model config."""
    head_dim = config.head_dim
    intermediate = (
        config.shared_expert_intermediate_size if getattr(config, "moe_enabled", False)
        else config.intermediate_size
    )
    units = {
        "q_proj": config.num_qo_heads * head_dim * 2,  # the per-head output gate doubles it
        "k_proj": config.num_kv_heads * head_dim,
        "v_proj": config.num_kv_heads * head_dim,
        "o_proj": config.num_qo_heads * head_dim,
        "gate_proj": intermediate,
        "up_proj": intermediate,
        "down_proj": intermediate,
        "lm_head": config.vocab_size,
        "embed_tokens": config.vocab_size,
    }
    group = config.linear_attention_group()
    if group is not None:
        key, value = group.num_key_heads * group.key_head_dim, group.num_value_heads * group.value_head_dim
        units.update({
            "in_proj_qkv": 2 * key + value, "in_proj_z": value,
            "in_proj_b": group.num_value_heads, "in_proj_a": group.num_value_heads,
            "linear_attn.conv1d": 2 * key + value, "A_log": group.num_value_heads,
            "dt_bias": group.num_value_heads, "out_proj": value,
        })
    return units


def _leaf_runs(config) -> dict[str, list[tuple[int, int]]]:
    """``(offset, length)`` global row runs of one leaf, when it holds several."""
    group = config.linear_attention_group()
    if group is None:
        return {}
    key = group.num_key_heads * group.key_head_dim
    value = group.num_value_heads * group.value_head_dim
    runs = [(0, key), (key, key), (2 * key, value)]
    return {"in_proj_qkv": runs, "linear_attn.conv1d": runs}


class TpShard:
    """Cuts one checkpoint tensor to the slice this rank's model buffers hold.

    A reader builds one instance per weight namespace: ``tp_shard`` for the text tower, and a
    second one where a tower declares its buffers elsewhere (the Qwen VL vision stack, whose
    leaf table lives with the tower in ``models/qwen3_vl/vision.py``).
    """

    def __init__(self, rank: int, world: int, axes, units, runs=None) -> None:
        self.rank, self.world = rank, world
        self.axes, self.units = axes, units
        self.runs = runs or {}

    def part(self, leaf: str, part: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        total = self.units.get(leaf)
        axis = self.axes.get(leaf)
        if axis is None or total is None:
            return part
        return {role: self._cut(tensor, axis, total, leaf) for role, tensor in part.items()}

    def tensor(self, leaf: str, tensor: torch.Tensor) -> torch.Tensor:
        axis, total = self.axes.get(leaf), self.units.get(leaf)
        if axis is None or total is None:
            return tensor
        return self._cut(tensor, axis, total, leaf)

    def _cut(self, tensor: torch.Tensor, axis: int, total: int, leaf: str) -> torch.Tensor:
        """This rank's runs of ``total`` units, where ``tensor`` tiles that axis in blocks."""
        if tensor.numel() == 1 or axis >= tensor.dim():
            return tensor  # a scalar, or a per-row vector for a column-sharded weight
        blocks = tensor.shape[axis]
        if not blocks or total % blocks:
            return tensor  # this axis is not the split one (a [rows, 1] scale column)
        per_block = total // blocks
        pieces = []
        for offset, length in self.runs.get(leaf, ((0, total),)):
            padded = leaf in CEIL_LEAVES and per_block == 1
            local = div_ceil(length, self.world) if padded else length // self.world
            start = self.rank * local
            take = max(0, min(local, length - start))
            if local < per_block or local % per_block:
                return tensor  # the run is smaller than one scale block: not sharded this way
            if take != local and not padded:
                return tensor  # the run is shorter than the rank count: keep it whole
            piece = tensor.narrow(axis, (offset + start) // per_block, take // per_block)
            if take < local:
                # the vocab shard is rounded up, so the last rank pads the rows no id selects
                pad = torch.zeros(local - take, *tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
                piece = pad if piece.numel() == 0 else torch.cat([piece, pad], dim=axis)
            pieces.append(piece)
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=axis)


def tp_shard_for(config, rank: int, world: int) -> TpShard:
    """The text tower's slicer for an explicit rank, so a test can pin one without a process group."""
    return TpShard(rank, world, LEAF_AXIS, _leaf_units(config), _leaf_runs(config))


def tp_shard(config) -> TpShard | None:
    """This rank's slicer for ``config``'s model, or None when running single-process."""
    tp = get_tp_info()
    return None if tp.size == 1 else tp_shard_for(config, tp.rank, tp.size)


__all__ = ["COLS", "ROWS", "ROLE_SUFFIXES", "TpShard", "module_leaf", "tp_shard", "tp_shard_for"]