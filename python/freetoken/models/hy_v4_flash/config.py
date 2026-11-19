"""Engine-facing config for Hy4-flash (``hy_v4_flash``).

All 42 layers store the same MLA latent (kv_lora_rank + qk_rope_head_dim) in one
paged pool; the 12 indexed layers additionally own a 1/4-shadow compressed index
slab (the kpool layout: pooled group keys + per-request tail rings). The SWA
layers read a 512-token window of the same latent rows -- the backend owns that
selection, so the pool stays one FullAttentionGroupConfig with mla=True and
index_ratio=4 (the engine coerces page_size to 64 for it).

Routed experts run in the 1536-wide latent space (LatentMoE); the NVFP4 release
stores them per-expert in the ModelOpt layout and serves them from the offload
cache. Everything else ships bf16 (the FP8 source's block-fp8 projections are
dequantized by the conversion).
"""

from __future__ import annotations

from typing import Any

from freetoken.layers.quantization import NameMap, QuantConfig, QuantKind
from freetoken.models.config import (
    FullAttentionGroupConfig,
    ModelConfig,
    RotaryConfig,
)

from .args import HyV4FlashArgs


def _expert_quant(hf_config: Any, args: HyV4FlashArgs) -> tuple[str, tuple[int, int] | None]:
    """The routed experts' quant kind as the engine's format tag, with the scale
    block of block-fp8 (the Qwen3.5 MoE precedent): probe the experts container
    under the family's checkpoint name map, so the FP8 release's wrapper-named
    modules_to_not_convert and the NVFP4 ignore list both resolve."""
    from freetoken.models.register import get_model_spec

    arch = (getattr(hf_config, "architectures", None) or ["HYV4FlashForCausalLM"])[0]
    spec = get_model_spec(arch)
    quant = QuantConfig.from_hf(hf_config, name_map=NameMap(roots=spec.checkpoint_roots))
    scheme = quant.scheme_for(f"model.layers.{args.first_k_dense_replace}.mlp.experts")
    if scheme is None:
        return "none", None
    return str(scheme.kind), scheme.weight.group if scheme.kind is QuantKind.FP8_BLOCK else None


def parse_config(hf_config: Any) -> ModelConfig:
    args = HyV4FlashArgs.from_hf(hf_config)
    text = getattr(hf_config, "text_config", None) or hf_config
    g = text.get if isinstance(text, dict) else (lambda k, d=None: getattr(text, k, d))
    n_indexed = sum(1 for s in args.swa_layers[: args.n_layers] if not s)
    expert_quant, weight_block_size = _expert_quant(hf_config, args)
    rotary_config = RotaryConfig(
        head_dim=args.qk_head_dim,
        rotary_dim=args.qk_rope_head_dim,
        max_position=args.max_position,
        base=args.rope_theta_indexed,
        scaling=None,  # rope_type "default"; the SWA table is built inside the model
    )
    return ModelConfig(
        num_layers=args.n_layers,
        num_qo_heads=args.n_heads,
        num_kv_heads=1,  # single shared MLA latent
        head_dim=args.latent_kv_dim,
        hidden_size=args.dim,
        vocab_size=args.vocab_size,
        intermediate_size=args.intermediate_size,
        hidden_act=g("hidden_act", "silu"),
        rms_norm_eps=args.norm_eps,
        tie_word_embeddings=bool(g("tie_word_embeddings", False)),
        rotary_config=rotary_config,
        attention_groups=(
            FullAttentionGroupConfig(
                name="full",
                layer_ids=tuple(range(args.n_layers)),
                num_kv_heads=1,
                head_dim=args.latent_kv_dim,
                rotary_config=rotary_config,
                mla=True,
                index_head_dim=args.index_head_dim,
                num_index_layers=n_indexed,
                index_ratio=args.index_compress_ratio,
            ),
        ),
        num_experts=args.n_routed_experts,
        num_experts_per_tok=args.n_activated_experts,
        moe_intermediate_size=args.moe_inter_dim,
        norm_topk_prob=args.norm_topk_prob,
        model_type="hy_v4_flash",
        architectures=getattr(hf_config, "architectures", ["HYV4FlashForCausalLM"]),
        moe_enabled=True,
        expert_quant=expert_quant,
        weight_block_size=weight_block_size,
        first_k_dense_replace=args.first_k_dense_replace,
        n_shared_experts=args.n_shared_experts,
        routed_scaling_factor=args.route_scale,
        n_group=1,
        topk_group=1,
        attn_sm_scale=args.qk_head_dim**-0.5,
        swiglu_limit=args.swiglu_limit,
        hy4_args=args,
    )


__all__ = ["parse_config"]
