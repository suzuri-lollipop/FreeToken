"""Inkling-2 vision tower: CLIP-style ViT with LayerScale, axial 2-D rope and a learned
position table bilinearly resampled to each image's native patch grid.

One 40x40 image token is four native 20x20 patches (merge 2); each image prepends a class
token. The encoder attends inside each image (non-causal), then the patch features of
every token are concatenated and projected (gelu MLP, no bias) into the text width.
Transcribed from the transformers inkling2 vision reference (Inkling2VisionModel).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, LayerNorm, LinearColParallelMerged, LinearReplicated, LinearRowParallel, OPList
from freetoken.models.qwen3_vl.vision import vision_module_leaf
from freetoken.models.tp_shard import COLS, ROWS, TpShard
from freetoken.models.weight_stream import BlockWeightStreamer
from freetoken.utils import div_even

from .config import VisionConfig


class _Table(BaseOP):
    def __init__(self, rows: int, cols: int):
        self.weight = torch.empty(rows, cols)


class _Embeddings(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.patch_embedding = LinearReplicated(vc.patch_dim, vc.hidden_size, has_bias=False, prefix="visual.backbone.embeddings.patch_embedding")
        self.position_embedding = _Table(vc.posemb_grid_size**2, vc.hidden_size)
        self.class_embedding = torch.empty(vc.hidden_size)
        self.cls_position_embedding = torch.empty(vc.hidden_size)


class _VisionMLP(BaseOP):
    def __init__(self, vc: VisionConfig, prefix: str):
        self.c_fc = LinearColParallelMerged(vc.hidden_size, [vc.intermediate_size], has_bias=True, prefix=f"{prefix}.c_fc")
        self.c_proj = LinearRowParallel(vc.intermediate_size, vc.hidden_size, has_bias=True, prefix=f"{prefix}.c_proj")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj.forward(F.gelu(self.c_fc.forward(x), approximate="none"))


class _VisionAttention(BaseOP):
    def __init__(self, vc: VisionConfig, prefix: str):
        self.num_heads = div_even(vc.num_heads, get_tp_info().size)
        self.head_dim = vc.head_dim
        self.q_proj = LinearColParallelMerged(vc.hidden_size, [vc.hidden_size], has_bias=True, prefix=f"{prefix}.q_proj")
        self.k_proj = LinearColParallelMerged(vc.hidden_size, [vc.hidden_size], has_bias=True, prefix=f"{prefix}.k_proj")
        self.v_proj = LinearColParallelMerged(vc.hidden_size, [vc.hidden_size], has_bias=True, prefix=f"{prefix}.v_proj")
        self.out_proj = LinearRowParallel(vc.hidden_size, vc.hidden_size, has_bias=True, prefix=f"{prefix}.out_proj")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        seq = x.shape[0]
        q = self.q_proj.forward(x).view(seq, self.num_heads, self.head_dim)
        k = self.k_proj.forward(x).view(seq, self.num_heads, self.head_dim)
        v = self.v_proj.forward(x).view(seq, self.num_heads, self.head_dim)
        q, k = _apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0),
            scale=self.head_dim**-0.5,
        )
        return self.out_proj.forward(out.squeeze(0).transpose(0, 1).reshape(seq, -1))


class _EncoderLayer(BaseOP):
    def __init__(self, vc: VisionConfig, index: int):
        prefix = f"visual.backbone.layers.{index}"
        self.ln_1 = LayerNorm(vc.hidden_size, eps=vc.layer_norm_eps)
        self.ln_2 = LayerNorm(vc.hidden_size, eps=vc.layer_norm_eps)
        self.ls_1_gamma = torch.empty(vc.hidden_size)
        self.ls_2_gamma = torch.empty(vc.hidden_size)
        self.self_attn = _VisionAttention(vc, f"{prefix}.self_attn")
        self.mlp = _VisionMLP(vc, f"{prefix}.mlp")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.ls_1_gamma * self.self_attn.forward(self.ln_1.forward(x), cos, sin)
        return x + self.ls_2_gamma * self.mlp.forward(self.ln_2.forward(x))


class _Backbone(BaseOP):
    def __init__(self, vc: VisionConfig):
        self.embeddings = _Embeddings(vc)
        self.ln_pre = LayerNorm(vc.hidden_size, eps=vc.layer_norm_eps)
        self.layers = OPList([_EncoderLayer(vc, i) for i in range(vc.num_layers)])


def _rotate_pairwise(x: torch.Tensor) -> torch.Tensor:
    x = x.view(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    dtype = q.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q = (q * cos + _rotate_pairwise(q) * sin).to(dtype)
    k = (k * cos + _rotate_pairwise(k) * sin).to(dtype)
    return q, k


def _axial_rope(position_ids: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Inkling2 axial rope: width frequencies then height, each interleaved pair repeated."""
    spatial = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, spatial, 2, dtype=torch.float32, device=position_ids.device) / spatial))
    freqs = position_ids.float().unsqueeze(-1) * inv_freq  # [S, 2, spatial//2]
    freq_h, freq_w = freqs[:, 0], freqs[:, 1]
    angles = torch.cat([freq_w, freq_h], dim=-1).repeat_interleave(2, dim=-1)
    return angles.cos(), angles.sin()


def _interp_pos(grid_h: int, grid_w: int, side: int, merge: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear (align_corners=False) resample indices/weights of the side x side position
    table onto the native patch grid, in merge-block order (the reference helper's layout)."""
    device = torch.device("cpu")
    br = torch.arange(grid_h // merge, device=device)
    bc = torch.arange(grid_w // merge, device=device)
    ir = torch.arange(merge, device=device)
    ic = torch.arange(merge, device=device)
    shape = (grid_h // merge, grid_w // merge, merge, merge)
    r = (br[:, None, None, None] * merge + ir[None, None, :, None]).expand(shape).reshape(-1)
    c = (bc[None, :, None, None] * merge + ic[None, None, None, :]).expand(shape).reshape(-1)
    # F.interpolate clamps negative sample coordinates to 0 (align_corners=False, bilinear)
    src_r = ((r.float() + 0.5) * side / grid_h - 0.5).clamp(min=0)
    src_c = ((c.float() + 0.5) * side / grid_w - 0.5).clamp(min=0)
    r0f, c0f = torch.floor(src_r), torch.floor(src_c)
    r0, c0 = r0f.long().clamp(0, side - 1), c0f.long().clamp(0, side - 1)
    r1, c1 = (r0 + 1).clamp(max=side - 1), (c0 + 1).clamp(max=side - 1)
    wr1, wc1 = (src_r - r0f).clamp(0, 1), (src_c - c0f).clamp(0, 1)
    wr0, wc0 = 1.0 - wr1, 1.0 - wc1
    indices = torch.stack([r0 * side + c0, r0 * side + c1, r1 * side + c0, r1 * side + c1], dim=-1)
    weights = torch.stack([wr0 * wc0, wr0 * wc1, wr1 * wc0, wr1 * wc1], dim=-1)
    return indices, weights


def _patch_position_ids(grid_h: int, grid_w: int, merge: int) -> torch.Tensor:
    """1-indexed (row, col) of every native patch in merge-block order (the vision_utils helper)."""
    hpos, wpos = torch.meshgrid(torch.arange(grid_h), torch.arange(grid_w), indexing="ij")
    shape = (grid_h // merge, merge, grid_w // merge, merge)
    hpos = hpos.reshape(shape).transpose(1, 2).flatten()
    wpos = wpos.reshape(shape).transpose(1, 2).flatten()
    return torch.stack([hpos, wpos], dim=-1) + 1


class Inkling2VisionModel(BaseOP):
    def __init__(self, vc: VisionConfig):
        self._vc = vc
        self.backbone = _Backbone(vc)
        self.proj_in = LinearColParallelMerged(
            vc.merge_size**2 * vc.hidden_size, [vc.projection_hidden_size],
            has_bias=vc.projector_bias, prefix="visual.proj_in",
        )
        self.proj_out = LinearRowParallel(
            vc.projection_hidden_size, vc.text_hidden_size,
            has_bias=vc.projector_bias, prefix="visual.proj_out",
        )
        self._streamer: BlockWeightStreamer | None = None

    def place_weights(self, mode: str) -> None:
        # the linear weights are still unallocated at placement time; the bare embeddings exist
        device = self.backbone.embeddings.class_embedding.device
        if mode == "host" and self._streamer is None:
            self._streamer = BlockWeightStreamer(self.backbone.layers.op_list, device)
        elif mode == "gpu" and self._streamer is not None:
            self._streamer.unstream()
            self._streamer = None
        elif mode not in ("gpu", "host"):
            raise ValueError(f"unknown vision weight placement {mode!r}")

    def _layers(self):
        if self._streamer is None:
            return enumerate(self.backbone.layers.op_list)
        return self._streamer.blocks(self.backbone.layers.op_list)

    def forward(self, pixel_values: torch.Tensor, grid_thw: List[int]) -> torch.Tensor:
        vc = self._vc
        merge, patch = vc.merge_size, vc.patch_size
        t, rows, cols = grid_thw
        if t != 1:
            raise NotImplementedError("inkling2 vision: video/temporal input is not served")
        # the processor hands CPU pixels; the tower's resident weights set the device
        device = self.backbone.embeddings.patch_embedding.weight.device
        pixel_values = pixel_values.to(device)
        grid_h, grid_w = rows * merge, cols * merge
        num_tokens = rows * cols
        num_patches = num_tokens * merge**2

        emb = self.backbone.embeddings
        patches = pixel_values.reshape(-1, merge, patch, merge, patch, pixel_values.shape[-1]).transpose(2, 3)
        patch_emb = emb.patch_embedding.forward(patches.reshape(num_patches, vc.patch_dim))
        indices, weights = _interp_pos(grid_h, grid_w, vc.posemb_grid_size, merge)
        indices = indices.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        pos = (emb.position_embedding.weight[indices] * weights[:, :, None]).sum(1)
        patch_emb = patch_emb + pos.to(patch_emb.dtype)
        cls = (emb.class_embedding + emb.cls_position_embedding).to(patch_emb.dtype)
        hidden = cls.repeat(num_patches + 1, 1)
        hidden[1:] = patch_emb
        hidden = self.backbone.ln_pre.forward(hidden)

        pos_ids = torch.zeros(num_patches + 1, 2, dtype=torch.long)
        pos_ids[1:] = _patch_position_ids(grid_h, grid_w, merge)
        cos, sin = _axial_rope(pos_ids.to(device), vc.head_dim, vc.rope_theta)
        for _, layer in self._layers():
            hidden = layer.forward(hidden, cos, sin)

        merged = hidden[1:].reshape(num_tokens, merge**2 * vc.hidden_size)
        return self.proj_out.forward(F.gelu(self.proj_in.forward(merged), approximate="none"))


# The tower's rank split, keyed by the module that owns each leaf: the projector cuts by its
# own hidden width, not the block's. Kept next to the Linear classes above because it has to
# agree with them. Embeddings, norms and LayerScale stay replicated; the row-parallel
# all-reduce hands every rank the full-width hidden state they multiply.
_TP_LEAF_AXIS = {
    "self_attn.q_proj": ROWS,
    "self_attn.k_proj": ROWS,
    "self_attn.v_proj": ROWS,
    "self_attn.out_proj": COLS,
    "mlp.c_fc": ROWS,
    "mlp.c_proj": COLS,
    "visual.proj_in": ROWS,
    "visual.proj_out": COLS,
}


def _tp_leaf_units(vc: VisionConfig) -> dict[str, int]:
    """Global units along each sharded leaf's split axis."""
    return {
        "self_attn.q_proj": vc.hidden_size,
        "self_attn.k_proj": vc.hidden_size,
        "self_attn.v_proj": vc.hidden_size,
        "self_attn.out_proj": vc.hidden_size,
        "mlp.c_fc": vc.intermediate_size,
        "mlp.c_proj": vc.intermediate_size,
        "visual.proj_in": vc.projection_hidden_size,
        "visual.proj_out": vc.projection_hidden_size,
    }


def vision_tp_shard_for(vc: VisionConfig, rank: int, world: int) -> TpShard | None:
    """The tower's slicer for an explicit rank, so a reader with its own TP view can use it."""
    return None if world == 1 else TpShard(rank, world, _TP_LEAF_AXIS, _tp_leaf_units(vc))


def vision_tp_shard(vc: VisionConfig) -> TpShard | None:
    """This rank's slicer for the tower's own keys, or None when running single-process."""
    tp = get_tp_info()
    return vision_tp_shard_for(vc, tp.rank, tp.size)


__all__ = ["Inkling2VisionModel", "vision_module_leaf", "vision_tp_shard", "vision_tp_shard_for"]
