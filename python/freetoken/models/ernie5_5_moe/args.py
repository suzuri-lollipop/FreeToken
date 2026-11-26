"""ERNIE 5.5 Flash (``ernie5_5_moe``) hyperparameters.

Built from the HF ``config.json`` (the checkpoint ships no separate inference
config). The field names the DSV4 pool / cost model / compressor machinery
reads are kept duck-compatible with :class:`~freetoken.models.deepseek_v4.args.DeepseekV4Args`
(``n_layers``, ``dim``, ``head_dim``, ``window_size``, ``compress_ratios``,
``index_n_heads``, ``index_head_dim``), so ERNIE rides the DSV4 window-tier
engine path; ``compress_ratios`` maps the checkpoint's DSA sentinel (-2) to 0
(no compression tier) while ``layer_compress_ratios`` keeps the raw values the
model branches on. ERNIE-only geometry (DSA MLA dims, latent MoE, situ,
multimax, DSpark) lives alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Tuple


def _get(cfg: Any, name: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


@dataclass
class Ernie55Args:
    # ----- runtime (overlaid by the engine at config resolution) -----
    max_batch_size: int = 1
    max_seq_len: int = 1048576

    # ----- shape -----
    vocab_size: int = 201216
    dim: int = 4096
    n_layers: int = 43
    n_heads: int = 64
    head_dim: int = 512
    qk_nope_head_dim: int = 448
    rope_head_dim: int = 64
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    norm_eps: float = 1e-6
    intermediate_size: int = 14336
    first_k_dense_replace: int = 1

    # Raw per-layer compression ratios: 128 = HCA, -2 = DSA (token-level MLA).
    layer_compress_ratios: Tuple[int, ...] = ()
    # DSV4-machinery view: DSA layers carry no compression tier (ratio 0).
    compress_ratios: Tuple[int, ...] = ()

    # ----- DSA (hybrid MLA) layers -----
    mla_n_heads: int = 64
    mla_q_lora_rank: int = 1024
    mla_kv_lora_rank: int = 512
    mla_qk_nope_head_dim: int = 192
    mla_qk_rope_head_dim: int = 64
    mla_v_head_dim: int = 256
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048

    # ----- rope -----
    # YaRN is HCA-only (at compress_rope_theta); DSA, its indexer and the draft
    # layers use plain RoPE at rope_theta.
    rope_theta: float = 8000000.0
    compress_rope_theta: float = 160000.0
    original_seq_len: int = 65536
    rope_factor: float = 16.0
    beta_fast: int = 32
    beta_slow: int = 1

    # ----- mHC -----
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # ----- MoE -----
    n_routed_experts: int = 512
    n_shared_experts: int = 1
    n_activated_experts: int = 10
    moe_inter_dim: int = 2048
    moe_latent_size: int = 2048
    routed_scaling_factor: float = 3.0
    norm_topk_prob: bool = True

    # ----- activation / head -----
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0
    multimax: str = "lm_head"

    # ----- DSpark draft (bundled under model.mtp.*) -----
    dspark_block_size: int = 7
    dspark_markov_rank: int = 256
    dspark_noise_token_id: int = 1024
    dspark_target_layer_ids: Tuple[int, ...] = ()
    dspark_num_layers: int = 3
    dspark_head_dim: int = 128
    dspark_num_qo_heads: int = 2
    dspark_kv_heads: int = 1
    dspark_rope_theta: float = 1e6
    dspark_window: int = 1024
    dspark_ring: int = 2048

    # ----- flashindexer head routing (optional speed path) -----
    enable_flashindexer: bool = False
    flashindexer_topk_heads: int = 8

    def __post_init__(self) -> None:
        if isinstance(self.layer_compress_ratios, list):
            self.layer_compress_ratios = tuple(self.layer_compress_ratios)
        if isinstance(self.dspark_target_layer_ids, list):
            self.dspark_target_layer_ids = tuple(self.dspark_target_layer_ids)
        if not self.layer_compress_ratios:
            self.layer_compress_ratios = (128,) * self.n_layers
        if not self.compress_ratios:
            self.compress_ratios = tuple(
                max(r, 0) for r in self.layer_compress_ratios
            )

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.rope_head_dim

    @property
    def mla_qk_head_dim(self) -> int:
        return self.mla_qk_nope_head_dim + self.mla_qk_rope_head_dim

    @property
    def mla_latent_dim(self) -> int:
        return self.mla_kv_lora_rank + self.mla_qk_rope_head_dim

    def is_dsa_layer(self, layer_id: int) -> bool:
        return (
            layer_id < len(self.layer_compress_ratios)
            and self.layer_compress_ratios[layer_id] <= 0
        )

    @property
    def dsa_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i in range(self.n_layers) if self.is_dsa_layer(i))

    @property
    def hca_layer_ids(self) -> Tuple[int, ...]:
        return tuple(i for i in range(self.n_layers) if not self.is_dsa_layer(i))


def load_args(hf_config: Any, **overrides: Any) -> Ernie55Args:
    """Build :class:`Ernie55Args` from the checkpoint's HF config."""
    scaling = _get(hf_config, "rope_scaling", None) or {}
    kwargs = dict(
        vocab_size=int(_get(hf_config, "vocab_size", 201216)),
        dim=int(_get(hf_config, "hidden_size", 4096)),
        n_layers=int(_get(hf_config, "num_hidden_layers", 43)),
        n_heads=int(_get(hf_config, "num_attention_heads", 64)),
        head_dim=int(_get(hf_config, "head_dim", 512)),
        qk_nope_head_dim=int(_get(hf_config, "qk_nope_head_dim", 448)),
        rope_head_dim=int(_get(hf_config, "qk_rope_head_dim", 64)),
        q_lora_rank=int(_get(hf_config, "q_lora_rank", 1024)),
        kv_lora_rank=int(_get(hf_config, "kv_lora_rank", 512)),
        o_groups=int(_get(hf_config, "o_groups", 8)),
        o_lora_rank=int(_get(hf_config, "o_lora_rank", 1024)),
        window_size=int(_get(hf_config, "sliding_window", 128)),
        norm_eps=float(_get(hf_config, "rms_norm_eps", 1e-6)),
        intermediate_size=int(_get(hf_config, "intermediate_size", 14336)),
        first_k_dense_replace=int(_get(hf_config, "first_k_dense_replace", 1)),
        layer_compress_ratios=tuple(_get(hf_config, "compress_ratios", ()) or ()),
        mla_n_heads=int(_get(hf_config, "hybrid_mla_num_attention_heads", 64)),
        mla_q_lora_rank=int(_get(hf_config, "hybrid_mla_q_lora_rank", 1024)),
        mla_kv_lora_rank=int(_get(hf_config, "hybrid_mla_kv_lora_rank", 512)),
        mla_qk_nope_head_dim=int(_get(hf_config, "hybrid_mla_qk_nope_head_dim", 192)),
        mla_qk_rope_head_dim=int(_get(hf_config, "hybrid_mla_qk_rope_head_dim", 64)),
        mla_v_head_dim=int(_get(hf_config, "hybrid_mla_v_head_dim", 256)),
        index_n_heads=int(_get(hf_config, "index_n_heads", 64)),
        index_head_dim=int(_get(hf_config, "index_head_dim", 128)),
        index_topk=int(_get(hf_config, "index_topk", 2048)),
        rope_theta=float(_get(hf_config, "rope_theta", 8000000.0)),
        compress_rope_theta=float(_get(hf_config, "compress_rope_theta", 160000.0)),
        original_seq_len=int(scaling.get("original_max_position_embeddings", 65536)),
        rope_factor=float(scaling.get("factor", 16.0)),
        beta_fast=int(scaling.get("beta_fast", 32)),
        beta_slow=int(scaling.get("beta_slow", 1)),
        hc_mult=int(_get(hf_config, "hc_mult", 4)),
        hc_sinkhorn_iters=int(_get(hf_config, "hc_sinkhorn_iters", 20)),
        hc_eps=float(_get(hf_config, "hc_eps", 1e-6)),
        n_routed_experts=int(_get(hf_config, "n_routed_experts", 512)),
        n_shared_experts=int(_get(hf_config, "n_shared_experts", 1)),
        n_activated_experts=int(_get(hf_config, "num_experts_per_tok", 10)),
        moe_inter_dim=int(_get(hf_config, "moe_intermediate_size", 2048)),
        moe_latent_size=int(_get(hf_config, "moe_latent_size", 2048)),
        routed_scaling_factor=float(_get(hf_config, "routed_scaling_factor", 3.0)),
        norm_topk_prob=bool(_get(hf_config, "norm_topk_prob", True)),
        situ_beta=float(_get(hf_config, "activation_situ_beta", 4.0)),
        situ_linear_beta=float(_get(hf_config, "activation_situ_linear_beta", 25.0)),
        multimax=str(_get(hf_config, "multimax", "lm_head") or "lm_head"),
        dspark_block_size=int(_get(hf_config, "dspark_block_size", 7)),
        dspark_markov_rank=int(_get(hf_config, "dspark_markov_rank", 256)),
        dspark_noise_token_id=int(_get(hf_config, "dspark_noise_token_id", 1024)),
        dspark_target_layer_ids=tuple(_get(hf_config, "dspark_target_layer_ids", ()) or ()),
        enable_flashindexer=bool(_get(hf_config, "enable_flashindexer", False)),
        max_seq_len=int(_get(hf_config, "max_position_embeddings", 1048576)),
    )
    kwargs.update(overrides)
    return Ernie55Args(**kwargs)


__all__ = ["Ernie55Args", "load_args"]
