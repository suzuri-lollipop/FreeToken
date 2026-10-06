from __future__ import annotations

import os
from typing import Dict

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.utils import div_ceil, nvtx_annotate

from .base import BaseOP
from .quantization import LayerKind, QuantConfig, quant_method_for

_EMBED_LOGGED = [False]  # one-shot latch for the fp8-embed startup line


def set_w8a16_embed_default(enabled: bool) -> None:
    """Resolve --no-embed-fp8 into the default new embeddings read at construction."""
    VocabParallelEmbedding._W8A16_EMBED_DEFAULT = bool(enabled)


class VocabParallelEmbedding(BaseOP):
    # Module-level W8A16 default, resolved from --no-embed-fp8 by the engine before the
    # model is built (embeddings have no config handle to consult).
    _W8A16_EMBED_DEFAULT = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embed_scale: float | None = None,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        # Gemma scales embeddings by sqrt(hidden_size). The scale is materialized in
        # the weight dtype (bf16) to match HF, which downcasts the scalar. The GPU
        # scalar is built lazily (model __init__ runs on the meta device) and cached
        # so it is not reallocated inside a captured CUDA graph.
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None
        self._comm = DistributedCommunicator()
        # W8A16-in for the INPUT embedding (opt-in, default OFF pending quality A/B):
        # the bf16 table (vocab/tp x hidden, 1.27 GiB total here) is REPLACED by its
        # per-row fp8-e4m3 form at finalize, freeing half its VRAM straight into the
        # expert slot cache. The decode gather then reads one fp8 row per token and
        # dequantizes on the fly -- the absolute byte saving per token is tiny (2 rows'
        # worth), so this is a MISS-byte / slot-capacity lever, not an embed-read lever.
        self.w8a16_embed_ok = VocabParallelEmbedding._W8A16_EMBED_DEFAULT
        self._w8a16_weight: torch.Tensor | None = None
        self._w8a16_scale: torch.Tensor | None = None
        self._w8a16_dtype: torch.dtype = torch.bfloat16

    def finalize(self) -> None:
        """Quantize the input-embedding table to per-row fp8-e4m3 and drop the bf16 master.

        Reuses the linear W8A16 quantizer: for a [vocab, hidden] table the per-row axis is
        the hidden dim, exactly matching its per-output-channel semantics. Prefill gathers
        also dequantize on the fly (the whole table is rewritten only once at load)."""
        if not self.w8a16_embed_ok or self.weight is None or not self.weight.is_cuda:
            return
        from freetoken.kernel.triton.w8a16_linear import quantize_weight_w8a16
        from freetoken.utils import init_logger

        if not _EMBED_LOGGED[0]:
            _EMBED_LOGGED[0] = True
            init_logger(__name__).info_rank0("W8A16 embed: input-embedding table quantized to fp8")
        self._w8a16_dtype = self.weight.dtype
        w8, scale = quantize_weight_w8a16(self.weight)
        self._w8a16_weight = w8
        self._w8a16_scale = scale  # [V] fp32, gathered 1-D then unsqueezed in forward
        self.weight = None

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel import indexing

        if self._w8a16_weight is not None:
            # Gather the fp8 rows, then dequantize with the per-row scale. The scale is
            # gathered 1-D with plain torch indexing (the JIT `indexing` kernel rejects
            # 4-byte rows); its vocab_range mask mirrors index.cu's, then it is unsqueezed
            # to broadcast against the row gather.
            w8 = indexing(
                weights=self._w8a16_weight,
                indices=x,
                vocab_range=self.vocab_range if self.tp_size > 1 else None,
            )
            if self.tp_size > 1 and self.vocab_range is not None:
                start, length = self.vocab_range
                local = x.long() - start
                valid = (local >= 0) & (local < length)
                s = self._w8a16_scale[local.clamp(0, self._w8a16_scale.shape[0] - 1)]
                s = s.masked_fill(~valid, 0.0)
            else:
                s = self._w8a16_scale[x.long()]
            y = (w8.float() * s[:, None]).to(self._w8a16_dtype)
        else:
            y = indexing(
                weights=self.weight,
                indices=x,
                vocab_range=self.vocab_range if self.tp_size > 1 else None,
            )

        if self.tp_size > 1:
            y = self._comm.all_reduce(y)
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(
                    self._embed_scale, dtype=y.dtype, device=y.device
                )
            y = y * self._embed_scale_t
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """The head is a linear layer over the vocab shard: its weights come from ``quant_method``
    unless they are tied to the input embedding."""

    quant_layer_kind = LayerKind.LINEAR

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.has_bias = bias
        self.prefix = prefix
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings
        self.in_features = embedding_dim
        self.out_features = self.num_embeddings_tp
        self.output_sizes = (self.num_embeddings_tp,)
        self.quant_method = None
        # W8A16 opt-in for the UNTIED head: it is the largest bf16 decode read on the
        # rank (vocab/tp x hidden), and its fp8 replacement frees ~0.3 GiB straight
        # into the expert slot cache. Sampled forward only ever sees M <= running
        # requests (prefill logits are gathered to last positions), so the decode
        # kernel's M <= 32 window covers every captured graph; the speculative head
        # pass exceeds it and takes the method's dequantized branch. A tied head shares
        # the input embedding's weight and keeps bf16 (quant_method is None there anyway).
        self.w8a16_decode_ok = (
            tied_embedding is None
            and os.environ.get("FREETOKEN_W8A16_LM_HEAD", "1") == "1"
        )
        if tied_embedding is None:
            self.quant_method = quant_method_for(quant_config, self, prefix)
            self.quant_method.create_weights(self)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None

    def finalize(self) -> None:
        if self.quant_method is not None:
            self.quant_method.finalize(self)

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    def shard_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Logit columns of this rank's vocab shard, for any number of rows.

        Goes through the head's own projection: an untied head's weight is replaced by
        its fp8 form in :meth:`finalize` (leaving ``weight`` as None), so a caller must
        never multiply against ``weight`` directly. Speculative decoding reaches for
        this with more rows than the running batch, which is why it sits outside
        :meth:`forward`'s last-token selection.
        """
        if self.tied_embedding is not None:
            return F.linear(x, self.tied_embedding.weight, self.bias)
        return self.quant_method.apply(self, x)

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        logits = self.shard_logits(x)
        if self.tp_size == 1:
            return logits
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)

        if bs == 1:
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]