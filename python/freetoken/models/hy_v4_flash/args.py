"""Hy4-flash (tencent/Hy4-flash, ``hy_v4_flash``) hyperparameters.

Read from the HF ``config.json``'s ``text_config``. The checkpoint dialect spells
the indexed layers ``deepseek_sparse_attention``; the transformers PR calls them
``indexed_attention`` -- both map to the same role here. Engram layer ids are
ONE-based in the config (the reference builds the memory on ``layer_idx + 1``);
args keep them zero-based like every other layer list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HyV4FlashArgs:
    vocab_size: int = 120832
    dim: int = 3072
    n_layers: int = 42
    n_heads: int = 32
    swa_n_heads: int = 64
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    norm_eps: float = 1e-5
    max_position: int = 1048576
    # per-layer attention role: True = sliding window, False = indexed (sparse)
    swa_layers: tuple[bool, ...] = ()
    sliding_window: int = 512
    rope_theta_indexed: float = 1e7
    rope_theta_swa: float = 1e4
    mrope_section: tuple[int, ...] = (12, 10, 10)
    # dense / MoE layout
    first_k_dense_replace: int = 1
    intermediate_size: int = 13824
    moe_inter_dim: int = 1536
    latent_dim: int = 1536  # routed_expert_hidden_size: the width the experts run in
    n_routed_experts: int = 512
    n_activated_experts: int = 16
    n_shared_experts: int = 1
    route_scale: float = 3.998
    norm_topk_prob: bool = True
    swiglu_limit: float = 10.0
    # compressed lightning indexer (indexed layers only)
    index_compress_ratio: int = 4
    index_n_heads: int = 16
    index_head_dim: int = 128
    index_group_topk: int = 512
    # identity hyper-connections
    hc_mult: int = 4
    hc_magnitude: float = 2.0
    hc_eps: float = 1e-6
    # engram (zero-based layer ids)
    engram_layer_ids: tuple[int, ...] = ()
    engram_max_ngram: int = 3
    engram_n_heads: int = 8  # hash heads per n-gram order
    engram_head_dim: int = 96  # n_embed_per_ngram // n_heads
    engram_pad_id: int = 120002
    engram_seed: int = 0
    engram_lookup_size: int = 120818
    engram_fp8_embedding: bool = True
    # per engram layer: the 16 prime hash moduli, n-gram-major (order 2 heads, order 3 heads)
    engram_head_vocab_sizes: dict[int, tuple[int, ...]] = field(default_factory=dict)
    learnable_sink_init: float = 0.0
    image_token_id: int = 120120
    video_token_id: int = 120124

    def __post_init__(self) -> None:
        self.swa_layers = tuple(bool(v) for v in self.swa_layers)
        self.engram_layer_ids = tuple(int(v) for v in self.engram_layer_ids)
        if len(self.swa_layers) < self.n_layers:
            raise ValueError(f"swa_layers has {len(self.swa_layers)} entries for {self.n_layers} layers")
        if self.n_shared_experts != 1:
            raise ValueError("Hy4-flash has exactly one shared expert")
        if self.hc_mult < 1:
            raise ValueError("hc_mult must be positive")

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def latent_kv_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def candidate_topk(self) -> int:
        """Token budget of the indexed selection: group_topk groups of compress_ratio tokens."""
        return self.index_group_topk * self.index_compress_ratio

    @property
    def index_topk(self) -> int:
        """Duck-typed field the DSA backend family reads (the token recall budget)."""
        return self.candidate_topk

    @property
    def engram_width(self) -> int:
        """Concatenated lookup width: (orders - 1) * heads * head_dim."""
        return (self.engram_max_ngram - 1) * self.engram_n_heads * self.engram_head_dim

    @property
    def engram_orders(self) -> int:
        return self.engram_max_ngram - 1

    def heads_of(self, layer: int) -> int:
        return self.swa_n_heads if self.swa_layers[layer] else self.n_heads

    def theta_of(self, layer: int) -> float:
        return self.rope_theta_swa if self.swa_layers[layer] else self.rope_theta_indexed

    @classmethod
    def from_hf(cls, hf_config: Any) -> "HyV4FlashArgs":
        text = getattr(hf_config, "text_config", None) or hf_config
        raw = text if isinstance(text, dict) else (text.to_dict() if hasattr(text, "to_dict") else vars(text))
        g = raw.get
        layer_types = g("layer_types") or []
        swa = tuple(t == "sliding_attention" for t in layer_types)
        rope = g("rope_parameters") or {}
        indexed_rope = rope.get("deepseek_sparse_attention") or rope.get("indexed_attention") or {}
        swa_rope = rope.get("sliding_attention") or {}
        n_layers = int(g("num_hidden_layers", 42))
        # dev/testing cap (FREETOKEN_HY4_MAX_LAYERS): exercise the forward / KV / offload
        # paths without the full 220B checkpoint; unset in normal use.
        cap = os.environ.get("FREETOKEN_HY4_MAX_LAYERS")
        if cap:
            n_layers = min(n_layers, int(cap))
        engram_ids_1based = g("engram_layer_ids") or []
        head_sizes = {
            int(k): tuple(int(s) for order in v for s in order)
            for k, v in (g("engram_layer_head_vocab_sizes") or {}).items()
        }
        n_embed = int(g("engram_n_embed_per_ngram", 768))
        n_heads = int(g("engram_n_head_per_ngram", 8))
        return cls(
            vocab_size=int(g("vocab_size", 120832)),
            dim=int(g("hidden_size", 3072)),
            n_layers=n_layers,
            n_heads=int(g("num_attention_heads", 32)),
            swa_n_heads=int(g("swa_num_attention_heads", 64)),
            q_lora_rank=int(g("q_lora_rank", 1024)),
            kv_lora_rank=int(g("kv_lora_rank", 512)),
            qk_nope_head_dim=int(g("qk_nope_head_dim", 192)),
            qk_rope_head_dim=int(g("qk_rope_head_dim", 64)),
            v_head_dim=int(g("v_head_dim", 256)),
            norm_eps=float(g("rms_norm_eps", 1e-5)),
            max_position=int(g("max_position_embeddings", 1048576)),
            swa_layers=swa[:n_layers],
            sliding_window=int(g("sliding_window", 512)),
            rope_theta_indexed=float(indexed_rope.get("rope_theta", 1e7)),
            rope_theta_swa=float(swa_rope.get("rope_theta", 1e4)),
            mrope_section=tuple(indexed_rope.get("mrope_section", (12, 10, 10))),
            first_k_dense_replace=int(g("first_k_dense_replace", 1)),
            intermediate_size=int(g("intermediate_size", 13824)),
            moe_inter_dim=int(g("moe_intermediate_size", 1536)),
            latent_dim=int(g("routed_expert_hidden_size") or g("hidden_size", 3072)),
            n_routed_experts=int(g("n_routed_experts", 512)),
            n_activated_experts=int(g("num_experts_per_tok", 16)),
            n_shared_experts=int(g("n_shared_experts", 1)),
            route_scale=float(g("routed_scaling_factor", 3.998)),
            norm_topk_prob=bool(g("norm_topk_prob", True)),
            swiglu_limit=float(g("swiglu_limit", 10.0)),
            index_compress_ratio=int(g("compressed_indexer_compress_ratio", 4)),
            index_n_heads=int(g("compressed_indexer_num_attention_heads", 16)),
            index_head_dim=int(g("compressed_indexer_head_dim", 128)),
            index_group_topk=int(g("compressed_indexer_group_topk", 512)),
            hc_mult=int(g("hc_mult", 4)),
            hc_magnitude=float(g("hc_magnitude", 2.0)),
            hc_eps=float(g("hc_eps", 1e-6)),
            engram_layer_ids=tuple(int(i) - 1 for i in engram_ids_1based if int(i) - 1 < n_layers),
            engram_max_ngram=int(g("engram_max_ngram_size", 3)),
            engram_n_heads=n_heads,
            engram_head_dim=n_embed // n_heads,
            engram_pad_id=int(g("engram_pad_id", 120002)),
            engram_seed=int(g("engram_seed", 0)),
            engram_lookup_size=int(g("engram_lookup_table_size", 120818)),
            engram_fp8_embedding=bool(g("engram_fp8_embedding", True)),
            engram_head_vocab_sizes={int(k) - 1: v for k, v in head_sizes.items()},
            learnable_sink_init=float(g("learnable_sink_init", 0.0)),
        )


__all__ = ["HyV4FlashArgs"]
