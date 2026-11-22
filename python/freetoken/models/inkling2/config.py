"""Engine-facing config for Inkling-2-Mini (``inkling2_loop``).

The checkpoint is a multimodal wrapper: the text tower lives in ``text_config`` and the
weights under ``model.llm.``; the CLIP-style vision tower under ``model.visual.``. The
decoder's 24 physical layers run as 48 visits (the middle 12 loop three times), so
``num_layers`` counts VISITS: every visit owns a KV cache slot and a short-conv state
row, while the MoE expert banks stay at 24 (``moe_layer_count``).

Attention is NoPE GQA with a learned relative-position bias, a per-query sink logit and
L2-clipped q/k; the full-attention layers add log-scaled queries and dynamic K/V convs.
The conv history rides the linear-state pool as a slot state on a zero-layer linear
group (the PLE precedent, qwen4_exp); the SWA + conv-state combination runs on the naive
prefix cache (the engine has no radix cache serving both yet).

Only the routed experts are quantized: layers 0-19 NVFP4, layers 20-23 MXFP8 (served
dequantized to bf16); every projection, the router, the controllers and the vision tower
are bf16. The registry's ``unquantized_modules`` keeps the global ModelOpt NVFP4 algo
off those modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from freetoken.models.config import (
    FullAttentionGroupConfig,
    LinearGatedDeltaGroupConfig,
    ModelConfig,
    RotaryConfig,
    SlotStateSpec,
    SWAAttentionGroupConfig,
    detect_expert_quant,
)

from .args import CONV_STATE, Inkling2Args, build_args

# the chat template's image placeholder <|unused_200054|>
IMAGE_TOKEN_ID = 200054


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    patch_size: int
    merge_size: int
    posemb_grid_size: int
    projection_hidden_size: int
    projector_bias: bool
    text_hidden_size: int
    layer_norm_eps: float
    rope_theta: float

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        return 3 * self.patch_size**2

    @property
    def token_patch_size(self) -> int:
        return self.patch_size * self.merge_size


def parse_vision_config(hf: Any) -> VisionConfig | None:
    vc = getattr(hf, "vision_config", None)
    if vc is None:
        return None
    hidden = int(vc.hidden_size)
    mlp_ratio = float(getattr(vc, "mlp_ratio", 4.0))
    intermediate = int(getattr(vc, "intermediate_size", None) or hidden * mlp_ratio)
    rope = getattr(vc, "rope_parameters", None) or {}
    return VisionConfig(
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_layers=int(vc.num_hidden_layers),
        num_heads=int(vc.num_attention_heads),
        patch_size=int(getattr(vc, "native_patch_size", None) or vc.patch_size),
        merge_size=int(getattr(vc, "adapter_merge_size", None) or vc.spatial_merge_size),
        posemb_grid_size=int(vc.posemb_grid_size),
        projection_hidden_size=int(vc.projection_hidden_dim),
        projector_bias=bool(getattr(vc, "projector_bias", False)),
        text_hidden_size=int(vc.text_hidden_size),
        layer_norm_eps=float(vc.layer_norm_eps),
        rope_theta=float(rope.get("rope_theta", 10000.0)),
    )


def _text_config(hf: Any) -> Any:
    return getattr(hf, "text_config", None) or hf


def parse_config(hf: Any) -> ModelConfig:
    text = _text_config(hf)
    loop = getattr(text, "loop_config", None)
    if loop is None:
        raise ValueError("inkling2_loop requires text_config.loop_config (the non-loop Inkling-2 arch is not served)")
    if isinstance(loop, dict):
        loop = dict(loop)
    else:
        loop = {k: getattr(loop, k) for k in ("core_start", "core_end", "num_passes", "gate_rank", "is_varloop")}
    local = list(getattr(text, "local_layer_ids", None) or [i for i in range(int(text.num_hidden_layers)) if (i + 1) % 6])
    args = build_args(
        hidden_size=int(text.hidden_size),
        num_hidden_layers=int(text.num_hidden_layers),
        loop_config=loop,
        local_layer_ids=local,
        n_layers_per_attn_res_block=int(text.n_layers_per_attn_res_block),
        num_attention_heads=int(text.num_attention_heads),
        num_key_value_heads=int(text.num_key_value_heads),
        head_dim=int(text.head_dim),
        d_rel=int(text.d_rel),
        rel_extent=int(text.rel_extent),
        sliding_window_size=int(text.sliding_window_size),
        qk_l2_clip_threshold=float(text.qk_l2_clip_threshold),
        rel_l2_clip_threshold=float(text.rel_l2_clip_threshold),
        log_scaling_n_floor=int(text.log_scaling_n_floor),
        log_scaling_alpha=float(text.log_scaling_alpha),
        sconv_kernel_size=int(text.sconv_kernel_size),
        mlp_gate_softcap=float(text.mlp_gate_softcap),
        mlp_up_softcap=float(text.mlp_up_softcap),
        route_scale=float(text.route_scale),
        rms_norm_eps=float(text.rms_norm_eps),
        final_logit_softcapping=float(text.final_logit_softcapping),
        vocab_size=int(text.vocab_size),
        unpadded_vocab_size=int(text.unpadded_vocab_size),
        image_token_id=int(getattr(hf, "image_token_id", IMAGE_TOKEN_ID)),
    )
    if int(text.swa_num_attention_heads) != args.num_qo_heads or int(text.swa_num_key_value_heads) != args.num_kv_heads:
        raise NotImplementedError("inkling2_loop: SWA and full attention must share head counts")
    if int(text.n_routed_experts) <= 0 or int(text.n_shared_experts) != 0:
        raise NotImplementedError("inkling2_loop: only the routed-only MoE (no shared experts) is served")
    if bool(getattr(text, "use_gate_bias", True)) is False:
        raise NotImplementedError("inkling2_loop: the router bias is required")

    # NoPE: the relative-position bias carries position; rotary_dim 0 skips the rope path
    rope = RotaryConfig(head_dim=args.head_dim, rotary_dim=0, max_position=1 << 20, base=1e6, scaling=None)
    swa_visits, full_visits = args.swa_visits(), args.full_visits()
    vision = parse_vision_config(hf)
    expert_quant = detect_expert_quant(hf)
    return ModelConfig(
        num_layers=args.num_visits,
        num_qo_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        hidden_size=args.hidden_size,
        # the sampler and tokenizer see the real vocabulary; the head slices its padded rows
        vocab_size=args.unpadded_vocab_size,
        intermediate_size=int(text.intermediate_size),
        rms_norm_eps=args.rms_norm_eps,
        rotary_config=rope,
        hidden_act="situ",
        tie_word_embeddings=False,
        num_experts=int(text.n_routed_experts),
        num_experts_per_tok=int(text.num_experts_per_tok),
        moe_intermediate_size=int(text.intermediate_size),
        norm_topk_prob=True,
        model_type="inkling2_loop",
        architectures=list(getattr(hf, "architectures", None) or ["Inkling2LoopForConditionalGeneration"]),
        moe_enabled=True,
        moe_layer_count=args.num_phys_layers,
        expert_quant=expert_quant,
        attn_quant="none",
        dense_quant="none",
        lm_head_quant="none",
        has_router_bias=True,
        routed_scaling_factor=args.route_scale,
        final_logit_softcapping=args.final_logit_softcapping,
        vision_config=vision,
        image_token_id=args.image_token_id,
        attention_groups=(
            SWAAttentionGroupConfig(
                name="swa", layer_ids=swa_visits,
                num_kv_heads=args.num_kv_heads, head_dim=args.head_dim,
                rotary_config=rope, sliding_window=args.sliding_window,
            ),
            FullAttentionGroupConfig(
                name="full", layer_ids=full_visits,
                num_kv_heads=args.num_kv_heads, head_dim=args.head_dim,
                rotary_config=rope,
            ),
            # zero GDN layers: the group exists only to carry the conv-history slot states
            # on the linear-state pool's per-request slots (the qwen4_exp PLE precedent)
            LinearGatedDeltaGroupConfig(
                name="conv_state", layer_ids=(),
                num_key_heads=args.num_kv_heads, num_value_heads=args.num_kv_heads,
                key_head_dim=args.head_dim, value_head_dim=args.head_dim,
                conv_kernel_dim=args.conv_kernel, output_gate="silu",
            ),
        ),
        slot_states=(
            SlotStateSpec(
                name=CONV_STATE,
                shape=(args.conv_channels, args.conv_state_len),
                layer_ids=tuple(range(args.num_visits)),
                dtype=torch.float32,
            ),
        ),
        inkling2_args=args,
    )


__all__ = ["IMAGE_TOKEN_ID", "VisionConfig", "parse_config", "parse_vision_config"]
