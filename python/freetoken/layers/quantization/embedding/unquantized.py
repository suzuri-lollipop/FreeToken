"""Dense tables: the indexing kernel looks rows up; a tied head runs the unquantized linear kernel on the same table."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from ..linear.unquantized import TorchLinearKernel
from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import EmbeddingKernel, EmbeddingMethod

_EMBED_LOGGED = [False]  # one-shot latch for the fp8-embed startup line


class IndexingEmbeddingKernel(EmbeddingKernel):
    name = "indexing"

    def finalize(self, layer: Any) -> None:
        """Quantize the table to per-row fp8-e4m3 and drop the bf16 master (--no-embed-fp8 keeps it).

        Reuses the linear W8A16 quantizer: for a [vocab, hidden] table the per-row axis is
        the hidden dim, exactly matching its per-output-channel semantics. Prefill gathers
        also dequantize on the fly (the whole table is rewritten only once at load)."""
        if not getattr(layer, "w8a16_embed_ok", False) or layer.weight is None or not layer.weight.is_cuda:
            return
        from freetoken.kernel.triton.w8a16_linear import quantize_weight_w8a16
        from freetoken.utils import init_logger

        if not _EMBED_LOGGED[0]:
            _EMBED_LOGGED[0] = True
            init_logger(__name__).info_rank0("W8A16 embed: input-embedding table quantized to fp8")
        layer._w8a16_dtype = layer.weight.dtype
        w8, scale = quantize_weight_w8a16(layer.weight)
        layer._w8a16_weight = w8
        layer._w8a16_scale = scale  # [V] fp32, gathered 1-D then unsqueezed in embed
        layer.weight = None

    def embed(self, layer: Any, ids: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel import indexing

        vocab_range = layer.vocab_range if layer.tp_size > 1 else None
        if layer._w8a16_weight is None:
            return indexing(weights=layer.weight, indices=ids, vocab_range=vocab_range)
        # Gather the fp8 rows, then dequantize with the per-row scale. The scale is
        # gathered 1-D with plain torch indexing (the JIT `indexing` kernel rejects
        # 4-byte rows); its vocab_range mask mirrors index.cu's, then it is unsqueezed
        # to broadcast against the row gather.
        w8 = indexing(weights=layer._w8a16_weight, indices=ids, vocab_range=vocab_range)
        scale = layer._w8a16_scale
        if vocab_range is not None:
            start, length = vocab_range
            local = ids.long() - start
            valid = (local >= 0) & (local < length)
            s = scale[local.clamp(0, scale.shape[0] - 1)].masked_fill(~valid, 0.0)
        else:
            s = scale[ids.long()]
        return (w8.float() * s[:, None]).to(layer._w8a16_dtype)

    def logits(self, layer: Any, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        return TorchLinearKernel().apply(SimpleNamespace(weight=layer.weight, bias=bias), x)


@register_method(QuantKind.NONE, LayerKind.EMBEDDING)
class UnquantizedEmbeddingMethod(EmbeddingMethod):
    candidates = (IndexingEmbeddingKernel,)

    def create_weights(self, layer: Any) -> None:
        layer.weight = torch.empty(self.cfg.num_embeddings, self.cfg.embedding_dim)
