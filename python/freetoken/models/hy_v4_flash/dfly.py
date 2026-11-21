"""DFly2 speculative-decoding draft for Hy4-flash (AngelSpec, arxiv 2607.25852).

Port of vLLM PR #68296 ``vllm/models/hy_v4/nvidia/dfly.py`` onto the engine-native
hy4-flash modules. The checkpoint ships ``model.dfly_layers.*``: 5 plain pre-norm
decoder layers (gated MLA + dense SwiGLU, NO iHC / engram / indexer), an
``mtp_fusion`` (gated embed+target-hidden), a ``hidden_correction`` (SwiGLU delta)
and a ``final_norm``. The draft reuses the target embedding / LM head and attends
over the TARGET KV pool at the paired sparse layers [25,29,33,37,41] with external
top-k indices (Phase 2 backend); this module owns the projections, the fusion /
correction math and the weight loading.

Phase 1: modules + weight loading (verified by loading parity). The draft attention
backend (target-KV reuse) and the multi-step proposer land in Phases 2-3; the
forward methods here are not yet wired into the spec runtime.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch
from freetoken.layers import BaseOP, LinearOProj, LinearReplicated, LinearRowParallel, OPList, RMSNorm
from freetoken.layers.activation import silu_and_mul
from freetoken.layers.base import iter_ops
from freetoken.layers.quantization import finalize_quant

from .args import HyV4FlashArgs

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# one sparse target layer per draft layer (1-based in the checkpoint config)
DFLY_TARGET_LAYERS = (25, 29, 33, 37, 41)
DFLY_NUM_LAYERS = 5
DFLY_MLP_INTER = 9216          # draft dense SwiGLU width (from gate_proj shape)
DFLY_CORRECTION_INTER = 3072   # hidden_correction SwiGLU width
DFLY_MASK_TOKEN_ID = 120009


def draft_model_config(config: "ModelConfig", draft_quant) -> "ModelConfig":
    """Rebuild the target ModelConfig for the 5-layer draft: indexed-geometry
    attention group over layers 0..4, no MoE (dense SwiGLU), block-fp8 draft quant.
    Mirrors ``parse_config``'s single FullAttentionGroupConfig (mla, index_ratio=4)
    so the draft backend's kv_cache_group_specs / index slots match 5 layers."""
    import dataclasses as _dc

    group = config.attention_groups[0]
    new_group = _dc.replace(
        group,
        layer_ids=tuple(range(DFLY_NUM_LAYERS)),
        num_index_layers=DFLY_NUM_LAYERS,
    )
    return _dc.replace(
        config,
        num_layers=DFLY_NUM_LAYERS,
        hy4_args=draft_args(config.hy4_args),
        attention_groups=(new_group,),
        quant=draft_quant,
        moe_enabled=False,
        num_experts=0,
        num_experts_per_tok=0,
        first_k_dense_replace=DFLY_NUM_LAYERS,
    )


def draft_args(target: HyV4FlashArgs) -> HyV4FlashArgs:
    """5 indexed-geometry draft layers, dense MLP, no engram / iHC / MoE.

    The draft layers reuse the target sparse-layer RoPE and head geometry; the
    compressed indexer is absent (the backend supplies external target indices).
    """
    return dataclasses.replace(
        target,
        n_layers=DFLY_NUM_LAYERS,
        swa_layers=(False,) * DFLY_NUM_LAYERS,
        engram_layer_ids=(),
        engram_head_vocab_sizes={},
        first_k_dense_replace=DFLY_NUM_LAYERS,
        intermediate_size=DFLY_MLP_INTER,
        hc_mult=1,
    )


class DraftAttention(BaseOP):
    """Gated MLA projections identical to the target indexed layers, minus the
    compressed indexer (the draft attends over reused target KV / indices)."""

    def __init__(self, config: "ModelConfig", layer_id: int, *, prefix: str = "") -> None:
        from freetoken.distributed import get_tp_info

        args = config.hy4_args
        self.layer_id = layer_id
        # TP shards the query heads (latent KV / selection stay replicated), like Hy4Attention
        self.num_heads = args.n_heads // get_tp_info().size
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        # q_a / kv_a / kv_b ship bf16 in the checkpoint (no block scales); only
        # q_b / linear_gate / o_proj are block-fp8. Pass None for the bf16 leaves
        # rather than relying on the target's modules_to_not_convert name map.
        self.q_a_proj = LinearReplicated(args.dim, args.q_lora_rank, has_bias=False, quant_config=None, prefix=f"{prefix}.q_a_proj")
        self.q_a_layernorm = RMSNorm(args.q_lora_rank, eps=args.norm_eps)
        self.q_b_proj = LinearReplicated(args.q_lora_rank, self.num_heads * self.qk_head_dim, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.q_b_proj")
        self.kv_a_proj_with_mqa = LinearReplicated(args.dim, args.kv_lora_rank + args.qk_rope_head_dim, has_bias=False, quant_config=None, prefix=f"{prefix}.kv_a_proj_with_mqa")
        self.kv_a_layernorm = RMSNorm(args.kv_lora_rank, eps=args.norm_eps)
        self.kv_b_proj = LinearReplicated(args.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), has_bias=False, quant_config=None, prefix=f"{prefix}.kv_b_proj")
        self.linear_gate = LinearReplicated(args.dim, self.num_heads * self.v_head_dim, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.linear_gate")
        # LinearOProj shards its input columns internally: pass the FULL head width
        self.o_proj = LinearOProj(args.n_heads * self.v_head_dim, args.dim, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.o_proj")
        self.post_proj_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.learnable_sink_param = torch.empty(self.num_heads, dtype=torch.float32)
        for proj in (self.q_b_proj, self.o_proj, self.linear_gate):
            proj.fp8_w8a16 = True
        from freetoken.layers.rotary import get_rope

        self.rope = get_rope(
            head_dim=args.qk_rope_head_dim, rotary_dim=args.qk_rope_head_dim,
            max_position=args.max_position, base=args.rope_theta_indexed, is_neox=False,
        )
        self._w_uk: torch.Tensor | None = None
        self._w_uv: torch.Tensor | None = None

    def _kv_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._w_uk is None:
            w = self.kv_b_proj.weight.view(self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
            self._w_uk = w[:, : self.qk_nope_head_dim, :].contiguous()
            self._w_uv = w[:, self.qk_nope_head_dim :, :].transpose(1, 2).contiguous()
        return self._w_uk, self._w_uv

    def prepare_for_runtime(self) -> None:
        self._kv_b()
        self.kv_b_proj.weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gated MLA projections (identical math to the target indexed layers),
        then the backend attends over the paired TARGET layer's KV with reused
        indices instead of a private draft KV chain."""
        from freetoken.core import get_global_ctx

        ctx = get_global_ctx()
        t = x.shape[0]
        w_uk, w_uv = self._kv_b()
        q_resid = self.q_a_layernorm.forward(self.q_a_proj.forward(x))
        q = self.q_b_proj.forward(q_resid).view(t, self.num_heads, self.qk_head_dim)
        q_nope, q_rope = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_proj_with_mqa.forward(x)
        c_kv, k_rope = kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_layernorm.forward(c_kv)
        rope_dim = self.qk_rope_head_dim
        q_rope = q_rope.reshape(t, self.num_heads * rope_dim)
        k_rope = k_rope.reshape(t, rope_dim)
        q_rope, k_rope = self.rope.forward(ctx.batch.positions, q_rope, k_rope)
        q_rope = q_rope.view(t, self.num_heads, rope_dim)
        q_absorbed = torch.bmm(q_nope.transpose(0, 1).contiguous(), w_uk).transpose(0, 1)
        o_latent = ctx.attn_backend.dfly_mla_forward(
            q_absorbed.contiguous(), q_rope.contiguous(), c_kv.contiguous(), k_rope.contiguous(),
            self.layer_id, ctx.batch, sinks=self.learnable_sink_param,
        )
        o = torch.bmm(o_latent.transpose(0, 1).contiguous(), w_uv).transpose(0, 1)
        gate = torch.sigmoid(self.linear_gate.forward(x).view(t, self.num_heads, self.v_head_dim))
        o = o * gate
        return self.post_proj_norm.forward(self.o_proj.forward(o.reshape(t, self.num_heads * self.v_head_dim)))


class DraftMLP(BaseOP):
    """Plain dense SwiGLU (gate / up / down), no post norm — the draft FFN."""

    def __init__(self, config: "ModelConfig", *, prefix: str = "") -> None:
        from freetoken.distributed import get_tp_info

        args = config.hy4_args
        inter = args.intermediate_size
        local_inter = inter // get_tp_info().size
        self.gate_proj = LinearReplicated(args.dim, local_inter, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.gate_proj")
        self.up_proj = LinearReplicated(args.dim, local_inter, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.up_proj")
        self.down_proj = LinearRowParallel(inter, args.dim, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.down_proj")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(torch.cat([self.gate_proj.forward(x), self.up_proj.forward(x)], dim=-1)))


class DFlyBlock(BaseOP):
    """Plain pre-norm block: norm -> gated MLA -> residual; norm -> SwiGLU -> residual."""

    def __init__(self, config: "ModelConfig", layer_id: int, *, prefix: str = "") -> None:
        args = config.hy4_args
        self.self_attn = DraftAttention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp = DraftMLP(config, prefix=f"{prefix}.mlp")
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn.forward(self.input_layernorm.forward(x))
        return x + self.mlp.forward(self.post_attention_layernorm.forward(x))


class DFlyFusion(BaseOP):
    """``anchor + gate * eh_proj(cat(enorm(anchor), hnorm(target_hidden)))``."""

    def __init__(self, config: "ModelConfig", *, prefix: str = "") -> None:
        args = config.hy4_args
        self.enorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.hnorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.eh_proj = LinearReplicated(2 * args.dim, args.dim, has_bias=False, quant_config=None, prefix=f"{prefix}.eh_proj")
        self.gate = torch.empty(args.dim)

    def forward(self, anchors: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
        fused = self.eh_proj.forward(torch.cat([self.enorm.forward(anchors), self.hnorm.forward(target_hidden)], dim=-1))
        return anchors + self.gate * fused


class DFlyHiddenCorrection(BaseOP):
    """``h + down_proj(silu_and_mul(gate_up_proj(cat(hidden_norm(h), embed_norm(prev_embed)))))``."""

    def __init__(self, config: "ModelConfig", *, prefix: str = "") -> None:
        from freetoken.distributed import get_tp_info

        args = config.hy4_args
        inter = DFLY_CORRECTION_INTER
        local_inter = inter // get_tp_info().size
        self.hidden_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.embed_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.gate_proj = LinearReplicated(2 * args.dim, local_inter, has_bias=False, quant_config=None, prefix=f"{prefix}.gate_proj")
        self.up_proj = LinearReplicated(2 * args.dim, local_inter, has_bias=False, quant_config=None, prefix=f"{prefix}.up_proj")
        self.down_proj = LinearRowParallel(inter, args.dim, has_bias=False, quant_config=None, prefix=f"{prefix}.down_proj")

    def forward(self, hidden: torch.Tensor, previous_embed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([self.hidden_norm.forward(hidden), self.embed_norm.forward(previous_embed)], dim=-1)
        delta = self.down_proj.forward(silu_and_mul(torch.cat([self.gate_proj.forward(combined), self.up_proj.forward(combined)], dim=-1)))
        return hidden + delta


class Hy4DFlyDraft(BaseOP):
    """The DFly2 draft model: 5 blocks + final_norm + fusion + hidden_correction.

    Embedding and LM head are the target's (bound at load). ``forward_hidden`` /
    ``head_input`` (the spec proposer contract) are wired in Phase 3.
    """

    attention_backend = None

    def __init__(self, config: "ModelConfig") -> None:
        self.config = config
        args = config.hy4_args
        self.layers = OPList([DFlyBlock(config, i, prefix=f"layers.{i}") for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.mtp_fusion = DFlyFusion(config, prefix="mtp_fusion")
        self.hidden_correction = DFlyHiddenCorrection(config, prefix="hidden_correction")
        self._embed_tokens = None
        self._lm_head = None
        for op in iter_ops(self):
            if isinstance(op, (LinearReplicated, LinearOProj, LinearRowParallel)):
                op.w8a16_decode_ok = False

    @staticmethod
    def relay_width(config: "ModelConfig") -> int:
        return config.hy4_args.dim

    def head_input(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden

    def forward_block(self, embeds: torch.Tensor, target_hidden: torch.Tensor, query_width: int) -> torch.Tensor:
        """One draft forward over the query block (vLLM ``HYV4DFlyQueryModel``):
        fuse each request's bonus (anchor) embedding with the target hidden, run
        the 5 draft layers (each attends the paired target KV via the backend's
        ``dfly_mla_forward`` with the proposer-supplied reuse indices), final norm.
        ``embeds``/return are ``[batch * query_width, dim]``; ``target_hidden`` is
        ``[batch, dim]`` (the target's last accepted hidden)."""
        x = embeds.clone()
        x[::query_width] = self.mtp_fusion.forward(x[::query_width], target_hidden)
        for block in self.layers.op_list:
            x = block.forward(x)
        return self.norm.forward(x)

    def apply_hidden_correction(self, hidden: torch.Tensor, previous_ids: torch.Tensor) -> torch.Tensor:
        return self.hidden_correction.forward(hidden, self._embed_tokens.forward(previous_ids))

    def draft_tokens(self, hidden: torch.Tensor, bonus: torch.Tensor, sample_fn, active: torch.Tensor, query_width: int) -> torch.Tensor:
        """Autoregressive hidden-correction + sampling over the block's first
        ``query_width - 1`` positions (vLLM ``HYV4DFlyQueryModel.forward`` tail).
        ``hidden`` is ``[batch, query_width, dim]``; ``sample_fn(step, corrected)``
        returns the draft token ids via the gathered target head. Returns
        ``[batch, query_width - 1]`` draft tokens (-1 on inactive rows)."""
        batch = hidden.shape[0]
        previous = bonus.long()
        drafts = hidden.new_full((batch, query_width - 1), -1, dtype=torch.long)
        for step in range(query_width - 1):
            corrected = self.apply_hidden_correction(hidden[:, step], previous)
            previous = sample_fn(step, corrected)
            drafts[:, step] = torch.where(active, previous, torch.full_like(previous, -1))
        return drafts


def _rename_dfly(name: str) -> str | None:
    """model.dfly_layers.* -> draft state-dict key; None for ignored tensors."""
    prefix = "model.dfly_layers."
    if not name.startswith(prefix):
        return None
    rest = name[len(prefix):]
    if rest.startswith("confidence_head."):
        return None
    if rest.startswith("final_norm."):
        return "norm." + rest[len("final_norm."):]
    head, _, tail = rest.partition(".")
    if head.isdigit():
        return f"layers.{head}.{tail}"
    return rest  # mtp_fusion.* / hidden_correction.*


def iter_dfly_weights(model_path: str, device: torch.device, config: "ModelConfig"):
    """Stream the draft tensors under their draft state-dict names, TP-sharded."""
    from freetoken.models.loader import ShardReader, safetensors_weight_map
    from freetoken.utils import download_hf_weight

    folder = download_hf_weight(model_path)
    weight_map = safetensors_weight_map(folder)
    reader = ShardReader(model_path, device)
    try:
        for name in weight_map:
            key = _rename_dfly(name)
            if key is None:
                continue
            tensor = reader.get_tensor(name)
            yield key, _tp_cut_dfly(key, tensor, config)
    finally:
        reader.close()


def _tp_cut_dfly(name: str, tensor: torch.Tensor, config: "ModelConfig") -> torch.Tensor:
    from freetoken.distributed import get_tp_info
    from freetoken.models.tp_shard import COLS, ROWS, TpShard

    tp = get_tp_info()
    if tp.size == 1:
        return tensor
    args = config.hy4_args
    heads = args.n_heads
    parts = name.split(".")
    leaf = axis = units = None
    if parts[0] == "layers":
        mod = ".".join(parts[2:-1]) if parts[-1] in ("weight", "weight_scale_inv") else ".".join(parts[2:])
        if mod == "self_attn.q_b_proj":
            leaf, axis, units = mod, ROWS, heads * args.qk_head_dim
        elif mod == "self_attn.kv_b_proj":
            leaf, axis, units = mod, ROWS, heads * (args.qk_nope_head_dim + args.v_head_dim)
        elif mod == "self_attn.linear_gate":
            leaf, axis, units = mod, ROWS, heads * args.v_head_dim
        elif mod == "self_attn.o_proj":
            leaf, axis, units = mod, COLS, heads * args.v_head_dim
        elif mod == "self_attn.learnable_sink_param":
            leaf, axis, units = mod, ROWS, heads
        elif mod in ("mlp.gate_proj", "mlp.up_proj"):
            leaf, axis, units = mod, ROWS, args.intermediate_size
        elif mod == "mlp.down_proj":
            leaf, axis, units = mod, COLS, args.intermediate_size
    elif name.startswith("hidden_correction."):
        mod = ".".join(parts[1:-1]) if parts[-1] in ("weight", "weight_scale_inv") else ".".join(parts[1:])
        if mod in ("gate_proj", "up_proj"):
            leaf, axis, units = mod, ROWS, DFLY_CORRECTION_INTER
        elif mod == "down_proj":
            leaf, axis, units = mod, COLS, DFLY_CORRECTION_INTER
    if leaf is None:
        return tensor
    return TpShard(tp.rank, tp.size, {leaf: axis}, {leaf: units}).tensor(leaf, tensor)


def load_dfly_draft(config: "ModelConfig", target_model, model_path: str, device: torch.device, dtype: torch.dtype) -> Hy4DFlyDraft:
    """Build the draft on meta, load every dfly tensor, bind target embed/head.

    The draft layers are block-fp8 / bf16 regardless of the target's quant (vLLM:
    "draft layers must be BF16 or FP8"), so the draft quant is always Fp8BlockConfig
    even when the target serves NVFP4. The dfly tensors live in the FP8 release;
    an NVFP4 target must point ``--speculative-draft-model-path`` at the FP8
    checkpoint (the NVFP4 conversion drops ``model.dfly_layers.*``)."""
    from freetoken.layers.quantization import Fp8BlockConfig
    from freetoken.utils import torch_dtype

    draft_quant = Fp8BlockConfig({"quant_method": "fp8", "weight_block_size": [128, 128], "activation_scheme": "dynamic", "fmt": "e4m3"})
    dconfig = draft_model_config(config, draft_quant)
    with torch.device("meta"), torch_dtype(dtype):
        draft = Hy4DFlyDraft(dconfig)
    expected = draft.state_dict()
    loaded = {}
    for name, tensor in iter_dfly_weights(model_path, device, dconfig):
        if name not in expected:
            raise ValueError(f"{model_path}: unexpected DFly tensor {name!r}")
        loaded[name] = tensor.to(device=device, dtype=expected[name].dtype)
    if missing := sorted(set(expected) - set(loaded)):
        raise ValueError(f"{model_path}: DFly tensors missing: {missing}")
    draft.load_state_dict(loaded)
    draft._embed_tokens = target_model.model.embed_tokens
    draft._lm_head = target_model.lm_head
    finalize_quant(draft.layers)
    finalize_quant(draft.mtp_fusion)
    finalize_quant(draft.hidden_correction)
    return draft


__all__ = [
    "Hy4DFlyDraft", "DFlyBlock", "DraftAttention", "DraftMLP", "DFlyFusion",
    "DFlyHiddenCorrection", "draft_args", "iter_dfly_weights", "load_dfly_draft",
    "DFLY_TARGET_LAYERS", "DFLY_NUM_LAYERS",
]
