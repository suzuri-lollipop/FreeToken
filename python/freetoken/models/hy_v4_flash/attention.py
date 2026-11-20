"""Hy4-flash gated MLA with the compressed lightning indexer.

Latent-KV MLA with weight absorption (the glm_moe_dsa pattern): the pool stores
one ``ckv (512) | kpe (64)`` row per token; ``kv_b`` is absorbed into Q and onto
the output. On top of GLM's form Hy4 adds a per-head sigmoid output gate
(``linear_gate``), learned attention sinks (the backend's sparse kernel seeds the
softmax denominator with them) and a sandwich ``post_proj_norm``.

Two layer shapes share the module: indexed layers (32 heads, theta 1e7, the
compressed indexer selects up to 2048 history tokens) and SWA layers (64 heads,
theta 1e4, a 512-token window the backend selects). Both store the same latent.

The indexer pools every 4 consecutive token keys into one group key (a learned
gated convex combination, fp32, biased by ``intra_group_ape``), norms it and
carries the RoPE phase of the group's first token. The projections live here;
the pooling / storage / selection addressing lives in the hy4 DSA backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, LinearOProj, LinearReplicated, RMSNorm
from freetoken.layers.quantization import QuantKind, without_online
from freetoken.layers.rotary import get_rope
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Fp32Linear(BaseOP):
    """A bias-free linear whose weight stays fp32 (the indexer's pooling projections)."""

    def __init__(self, in_size: int, out_size: int) -> None:
        self.weight = torch.empty(out_size, in_size, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x.float(), self.weight)


class Hy4CompressedIndexer(BaseOP):
    """Per-token indexer projections; the backend pools / scores / selects.

    ``compute`` returns the value/gate streams the kpool store pools (bf16 on the
    wire, fp32 math inside the store kernel), the rotated scoring queries, the
    head weights, and the norm weight + rope cache the store applies to each
    closing group key.
    """

    def __init__(self, config: ModelConfig, *, prefix: str = "") -> None:
        args = config.hy4_args
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.rope_dim = args.qk_rope_head_dim
        self.ratio = args.index_compress_ratio
        self.query_projection = LinearReplicated(
            args.q_lora_rank, self.n_heads * self.head_dim, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.query_projection",
        )
        self.head_weight_projection = LinearReplicated(
            args.dim, self.n_heads, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.head_weight_projection",
        )
        self.token_key_projection = LinearReplicated(
            args.dim, self.head_dim, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.token_key_projection",
        )
        self.token_key_norm = RMSNorm(self.head_dim, eps=args.norm_eps)
        self.key_value_projection = Fp32Linear(self.head_dim, self.head_dim)
        self.key_gate_projection = Fp32Linear(self.head_dim, self.head_dim)
        self.key_norm = RMSNorm(self.head_dim, eps=args.norm_eps)
        self.intra_group_ape = torch.empty(self.ratio, self.head_dim, dtype=torch.float32)
        # same frequencies as the indexed layers' main rope; Hy4-flash rotates
        # adjacent (even, odd) pairs (GLM interleaved), not the NeoX half-split
        self.rope = get_rope(
            head_dim=self.rope_dim, rotary_dim=self.rope_dim,
            max_position=args.max_position, base=args.rope_theta_indexed, is_neox=False,
        )

    def compute(self, x: torch.Tensor, q_resid: torch.Tensor, positions: torch.Tensor):
        from freetoken.attention.dsa import DSAIndexerInputs

        t = x.shape[0]
        tk = self.token_key_norm.forward(self.token_key_projection.forward(x))
        # fp32 streams: the reference projects the bf16 token keys in fp32 at pooling
        # time; rounding the projections to bf16 first flips near-tied group scores.
        # (Tail-ring members still round to the pool's bf16 storage when a group
        # straddles two forwards -- the incremental-cache tradeoff.)
        k = self.key_value_projection.forward(tk)
        gate = self.key_gate_projection.forward(tk)
        q = self.query_projection.forward(q_resid).view(t, self.n_heads, self.head_dim)
        # partial interleaved (GLM) rope on the TRAILING rope_dim channels: adjacent
        # (even, odd) pairs rotate by one frequency (the PR's apply_rotary_pos_emb),
        # the last 64 of the 128-dim index heads; queries at their token position
        half = self.rope_dim // 2
        cs = self.rope._cos_sin_cache[positions.long()]
        cos, sin = cs[:, :half], cs[:, half:]
        q_pass, q_rot = q.split([self.head_dim - self.rope_dim, self.rope_dim], dim=-1)
        q_even, q_odd = q_rot[..., 0::2], q_rot[..., 1::2]
        c, s = cos[:, None, :], sin[:, None, :]
        q_rot = torch.stack([q_even * c - q_odd * s, q_odd * c + q_even * s], dim=-1).flatten(-2)
        q = torch.cat([q_pass, q_rot], dim=-1).to(torch.bfloat16)
        # the backend's index_scale folds in head_dim**-0.5 (dsa_decode_scores / dsa_prefill_logits)
        w = self.head_weight_projection.forward(x).float() * (self.n_heads**-0.5)
        return DSAIndexerInputs(
            q=q, k=k, w=w, gate=gate, ape=self.intra_group_ape,
            norm_weight=self.key_norm.weight, rope_cache=self.rope._cos_sin_cache,
        )


class Hy4Attention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = "") -> None:
        args = config.hy4_args
        self.layer_id = layer_id
        self.is_swa = args.swa_layers[layer_id]
        tp = get_tp_info().size
        global_heads = args.heads_of(layer_id)
        if global_heads % tp:
            raise ValueError(f"Hy4-flash layer {layer_id} has {global_heads} heads; pick a TP size that divides them")
        # TP shards the query heads; the latent KV, indexer and sinks' selection stay
        # rank-replicated (every rank attends the same tokens with its head subset)
        self.num_heads = global_heads // tp
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.qk_head_dim = args.qk_head_dim
        self.v_head_dim = args.v_head_dim
        self.kv_lora_rank = args.kv_lora_rank
        self.rope = get_rope(
            head_dim=args.qk_rope_head_dim, rotary_dim=args.qk_rope_head_dim,
            max_position=args.max_position, base=args.theta_of(layer_id), is_neox=False,
        )
        self.q_a_proj = LinearReplicated(
            args.dim, args.q_lora_rank, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.q_a_proj",
        )
        self.q_a_layernorm = RMSNorm(args.q_lora_rank, eps=args.norm_eps)
        # column-parallel over heads: the output stays rank-local, no collective
        self.q_b_proj = LinearReplicated(
            args.q_lora_rank, self.num_heads * self.qk_head_dim, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.q_b_proj",
        )
        self.kv_a_proj_with_mqa = LinearReplicated(
            args.dim, args.kv_lora_rank + args.qk_rope_head_dim, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.kv_a_proj_with_mqa",
        )
        self.kv_a_layernorm = RMSNorm(args.kv_lora_rank, eps=args.norm_eps)
        # MLA absorption reads kv_b_proj.weight as bmm operands: unquantized only
        self.kv_b_proj = LinearReplicated(
            args.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            has_bias=False, quant_config=without_online(config.quant), prefix=f"{prefix}.kv_b_proj",
        )
        assert self.kv_b_proj.quant_method.kind is QuantKind.NONE, (
            f"{prefix}.kv_b_proj: a quantized kv_b_proj needs a dequantized copy for the MLA absorption"
        )
        self.kv_b_proj.w8a16_decode_ok = False
        self.linear_gate = LinearReplicated(
            args.dim, self.num_heads * self.v_head_dim, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.linear_gate",
        )
        # row-parallel over heads: reduces the per-head partials across ranks
        self.o_proj = LinearOProj(
            global_heads * self.v_head_dim, args.dim, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.o_proj",
        )
        self.post_proj_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.learnable_sink_param = torch.empty(self.num_heads, dtype=torch.float32)
        # Hy4-flash is bf16-trained: its activations do not survive the fp8 kernels'
        # dynamic per-128-group W8A8 quantization (generation degenerates). Serve the
        # block-fp8 projections weight-only (W8A16) at every batch size.
        for proj in (self.q_b_proj, self.o_proj, self.linear_gate):
            proj.fp8_w8a16 = True
        self.compressed_indexer = (
            Hy4CompressedIndexer(config, prefix=f"{prefix}.compressed_indexer")
            if not self.is_swa else None
        )
        self._w_uk: torch.Tensor | None = None
        self._w_uv: torch.Tensor | None = None

    def _kv_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._w_uk is None:
            w = self.kv_b_proj.weight.view(
                self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
            )
            self._w_uk = w[:, : self.qk_nope_head_dim, :].contiguous()
            self._w_uv = w[:, self.qk_nope_head_dim :, :].transpose(1, 2).contiguous()
        return self._w_uk, self._w_uv

    def prepare_for_runtime(self) -> None:
        self._kv_b()
        self.kv_b_proj.weight = None

    @nvtx_annotate("Hy4MLA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        indexer_inputs = (
            self.compressed_indexer.compute(x, q_resid, ctx.batch.positions)
            if self.compressed_indexer is not None and getattr(ctx.attn_backend, "dsa_enabled", False)
            else None
        )
        o_latent = ctx.attn_backend.mla_forward(
            q_absorbed.contiguous(), q_rope.contiguous(), c_kv.contiguous(),
            k_rope.contiguous(), self.layer_id, ctx.batch,
            indexer_inputs=indexer_inputs, sinks=self.learnable_sink_param,
        )  # [T, H, kv_lora_rank]

        o = torch.bmm(o_latent.transpose(0, 1).contiguous(), w_uv).transpose(0, 1)
        gate = torch.sigmoid(self.linear_gate.forward(x).view(t, self.num_heads, self.v_head_dim))
        o = o * gate
        return self.post_proj_norm.forward(self.o_proj.forward(o.reshape(t, self.num_heads * self.v_head_dim)))


__all__ = ["Hy4Attention", "Hy4CompressedIndexer", "Fp32Linear"]
