"""Engine-facing config for ERNIE 5.5 Flash (``ernie5_5_moe``).

``parse_config`` maps the standard transformer fields into :class:`ModelConfig`
and carries the full :class:`Ernie55Args` in ``ModelConfig.dsv4_args``: ERNIE's
HCA layers ARE the DSV4 window+compressed attention (same compressor, same
paged pools), so the family rides the DSV4 engine path; the six DSA layers add
token-level latent + index slabs to the same pool (see
``kvcache/dsv4/ernie55_pool.py``). The attention group declares the ``ernie55``
variant, which resolves to :class:`AttnType.ERNIE55` (the hybrid pool/backend).

Routed experts are MXFP4 (``ds_fp4`` banks) served from the offload cache; the
dense projections are 128x128 block-fp8 with F32 scales (the checkpoint's
``scale_fmt: ue8m0`` describes the expert scales, not the dense ones -- the
ModelSpec overrides it before the fp8 dialect reads it).
"""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig

from freetoken.models.config import DSV4AttentionGroupConfig, ModelConfig, RotaryConfig

from .args import Ernie55Args, load_args


class Ernie55HFConfig(PretrainedConfig):
    """transformers-side config so AutoTokenizer (the checkpoint ships a custom
    ErnieChatTokenizer via auto_map) resolves the family. max_position_embeddings
    must be set BEFORE super().__init__: transformers' RoPE standardization reads
    it during construction and a bare PretrainedConfig lacks the attribute."""

    model_type = "ernie5_5_moe"

    def __init__(self, **kwargs: Any) -> None:
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 1048576)
        super().__init__(**kwargs)


AutoConfig.register("ernie5_5_moe", Ernie55HFConfig)


def model_config_from_args(args: Ernie55Args, *, max_position: int) -> ModelConfig:
    rope_scaling = {
        "rope_type": "yarn",
        "factor": args.rope_factor,
        "beta_fast": args.beta_fast,
        "beta_slow": args.beta_slow,
        "original_max_position_embeddings": args.original_seq_len,
    }
    return ModelConfig(
        num_layers=args.n_layers,
        num_qo_heads=args.n_heads,
        num_kv_heads=1,  # HCA: a single shared latent KV head (K == V)
        head_dim=args.head_dim,
        hidden_size=args.dim,
        vocab_size=args.vocab_size,
        intermediate_size=args.intermediate_size,
        hidden_act="situ",
        rms_norm_eps=args.norm_eps,
        tie_word_embeddings=False,
        rotary_config=RotaryConfig(
            head_dim=args.head_dim,
            rotary_dim=args.rope_head_dim,
            max_position=max_position,
            base=args.compress_rope_theta,
            scaling=rope_scaling,
        ),
        num_experts=args.n_routed_experts,
        num_experts_per_tok=args.n_activated_experts,
        moe_intermediate_size=args.moe_inter_dim,
        norm_topk_prob=args.norm_topk_prob,
        model_type="ernie5_5_moe",
        architectures=["Ernie5_5_MoeForCausalLM"],
        moe_enabled=True,
        expert_quant="ds_fp4",
        first_k_dense_replace=args.first_k_dense_replace,
        n_shared_experts=args.n_shared_experts,
        routed_scaling_factor=args.routed_scaling_factor,
        attn_sm_scale=args.head_dim**-0.5,
        dsv4_args=args,
        attention_groups=(
            DSV4AttentionGroupConfig(
                name="ernie55",
                layer_ids=tuple(range(args.n_layers)),
                num_kv_heads=1,
                head_dim=args.head_dim,
                sliding_window=args.window_size,
                variant="ernie55",
            ),
        ),
    )


def parse_config(hf_config: Any) -> ModelConfig:
    args = load_args(hf_config, max_batch_size=1)
    max_position = int(getattr(hf_config, "max_position_embeddings", 0) or 0)
    if max_position <= 0:
        max_position = int(args.original_seq_len * args.rope_factor)
    if args.multimax != "lm_head":
        raise NotImplementedError(
            f"ERNIE 5.5 multimax={args.multimax!r} is not supported (only 'lm_head')"
        )
    return model_config_from_args(args, max_position=max_position)


__all__ = ["model_config_from_args", "parse_config"]
