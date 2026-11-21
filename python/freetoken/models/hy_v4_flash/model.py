"""Hy4-flash model (engine-native port of the transformers PR's modeling).

    embed -> 4 iHC residual streams -> 42 blocks (Engram at 1/11/21/31, iHC around
    gated MLA and LatentMoE) -> hc_head collapse -> norm -> head

Each block: the engram residual (when present) lands on the streams first; then
per sublayer the iHC PRE gates mix the streams into the sublayer input, and the
POST gates scale the sublayer output back into every stream (fp32, ``ihc_inject``).

Attention is latent-KV MLA over the paged pool: indexed layers attend the tokens
the compressed indexer recalls (up to 2048), SWA layers the last 512; both carry
learned sinks. The hy4 DSA backend owns selection and storage; this module owns
the projections and the residual chain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNorm, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel, embed_input_ids

from .attention import Hy4Attention
from .engram import EngramLayer, ZeroEngramTable
from .hc import IHCHead, IHCPre, ihc_inject
from .moe import Hy4DenseMLP, Hy4LatentMoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _HCWrap(BaseOP):
    """Checkpoint nesting: ``hc_attn_layer.hc_pre.*`` / ``hc_mlp_layer.hc_pre.*``."""

    def __init__(self, pre: IHCPre) -> None:
        self.hc_pre = pre

    def forward(self, streams: torch.Tensor):
        return self.hc_pre.forward(streams)


class Block(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = "") -> None:
        args = config.hy4_args
        self.layer_id = layer_id
        self.self_attn = Hy4Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp: BaseOP = (
            Hy4DenseMLP(config, prefix=f"{prefix}.mlp")
            if layer_id < args.first_k_dense_replace
            else Hy4LatentMoE(config, layer_id, prefix=f"{prefix}.mlp")
        )
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)
        self.hc_attn_layer = _HCWrap(IHCPre(args.hc_mult, args.dim, args.norm_eps, args.hc_eps, args.hc_magnitude))
        self.hc_mlp_layer = _HCWrap(IHCPre(args.hc_mult, args.dim, args.norm_eps, args.hc_eps, args.hc_magnitude))
        self.engram_layer = (
            EngramLayer(args, layer_id, config.quant, prefix=f"{prefix}.engram_layer")
            if layer_id in args.engram_layer_ids else None
        )

    def forward(self, streams: torch.Tensor, engram_mask: torch.Tensor | None) -> torch.Tensor:
        if self.engram_layer is not None and engram_mask is not None:
            streams = streams + self.engram_layer.forward(streams, engram_mask)
        post, x = self.hc_attn_layer.forward(streams)
        y = self.self_attn.forward(self.input_layernorm.forward(x))
        streams = ihc_inject(streams, y, post)
        post, x = self.hc_mlp_layer.forward(streams)
        y = self.mlp.forward(self.post_attention_layernorm.forward(x))
        return ihc_inject(streams, y, post)


class Transformer(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model") -> None:
        args = config.hy4_args
        self.args = args
        self.embed_tokens = VocabParallelEmbedding(num_embeddings=args.vocab_size, embedding_dim=args.dim)
        self.layers = OPList([Block(config, i, prefix=f"{prefix}.layers.{i}") for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.hc_head = IHCHead(args.hc_mult, args.dim, args.norm_eps, args.hc_eps)

    def forward(self, input_ids: torch.Tensor, engram_mask: torch.Tensor | None) -> torch.Tensor:
        h = embed_input_ids(self.embed_tokens, input_ids, get_global_ctx().batch)
        streams = h.unsqueeze(1).expand(-1, self.args.hc_mult, -1).contiguous()
        for block in self.layers.op_list:
            streams = block.forward(streams, engram_mask)
        return self.norm.forward(self.hc_head.forward(streams))


class HyV4FlashForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._args = config.hy4_args
        self.model = Transformer(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=self._args.vocab_size, embedding_dim=self._args.dim,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant, prefix="lm_head",
        )
        self._engram_host = None
        # bf16-trained model: the per-channel fp8 rewrite of the bf16 projections
        # (W8A16 decode linear) accumulates enough weight noise across 42 layers to
        # break generation; keep every projection bf16 (the block-fp8 modules are
        # served weight-only via fp8_w8a16 instead).
        from freetoken.layers import LinearOProj, LinearReplicated, LinearRowParallel
        from freetoken.layers.base import iter_ops

        for op in iter_ops(self):
            if isinstance(op, (LinearReplicated, LinearOProj, LinearRowParallel)):
                op.w8a16_decode_ok = False
        self.lm_head.w8a16_decode_ok = False
        # same reasoning for the per-row fp8 rewrite of the embedding table
        self.model.embed_tokens.w8a16_embed_ok = False

    def engram_layers(self) -> list[EngramLayer]:
        return [b.engram_layer for b in self.model.layers.op_list if b.engram_layer is not None]

    def _engram_mask(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        if not self.engram_layers():
            return None
        a = self._args
        return (input_ids != a.engram_pad_id) & (input_ids != a.image_token_id) & (input_ids != a.video_token_id)

    def prepare_for_runtime(self) -> None:
        for block in self.model.layers.op_list:
            block.self_attn.prepare_for_runtime()
        for layer in self.engram_layers():
            if layer._hash is None:
                layer.init_hash(self._args)
        torch.cuda.empty_cache()

    def load_host_tables(self, engine_config) -> int:
        """Attach the engram tables (streamed from the checkpoint shards; zeros for dummy
        weights) and take over ``forward_host_ctx`` so every dispatch stages its rows."""
        import os

        layers = self.engram_layers()
        if not layers:
            return 0
        device = torch.device("cuda", torch.cuda.current_device())
        if getattr(engine_config, "use_dummy_weight", False):
            for layer in layers:
                layer.attach_table(ZeroEngramTable(layer.width, device))
            return 0
        from freetoken.utils import download_hf_weight

        from .engram_table import EngramDiskTable, EngramHost, engram_row_source

        folder = download_hf_weight(engine_config.model_path)
        # spec decode (DFly): every decode step is a verify of decode_rows_per_req
        # rows/request, so the engram graph buffers must hold bs * rows_per_req rows.
        rows_per_req = getattr(engine_config, "decode_rows_per_req", 1) or 1
        graph_rows = max(engine_config.max_running_req, engine_config.cuda_graph_max_bs or 0, 1) * rows_per_req
        tables, hashes = [], []
        for layer in layers:
            hashes.append(layer.init_hash(self._args))
            tables.append(EngramDiskTable(
                engram_row_source(folder, layer.layer_id), layer.n_cols, device,
                max_graph_rows=graph_rows,
                max_extend_tokens=engine_config.max_extend_tokens,
                use_io_uring=os.getenv("FREETOKEN_ENGRAM_IO_URING", "1") != "0",
            ))
        for layer, table in zip(layers, tables):
            layer.attach_table(table)
        self._engram_host = EngramHost(hashes, tables, device, sync_mode=os.getenv("FREETOKEN_ENGRAM_SYNC", "auto"))
        self.forward_host_ctx = self._engram_host.forward_host_ctx
        return sum(
            t._graph_pinned.numel() + sum(b.numel() for b in t._eager_pinned)
            for t in tables if hasattr(t, "_graph_pinned")
        )

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        hidden = self.model.forward(batch.input_ids, self._engram_mask(batch.input_ids))
        return self.lm_head.forward(hidden)


__all__ = ["HyV4FlashForCausalLM", "Transformer", "Block"]
