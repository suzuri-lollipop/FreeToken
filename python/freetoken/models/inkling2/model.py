"""Inkling-2-Mini decoder: 24 physical layers executed as 48 visits with cross-pass messages.

Every pass through the looped core (layers 6-17) starts from the prelude output. Passes
2 and 3 read two "messages" banks from the previous pass -- the input of its second core
block and its output -- through four rank-64 controllers per layer (q, k, v, mlp):

    gate = down([rms_norm(cur); rms_norm(msg)])
    cur  = (1 - tanh(gain(gate))) * cur + tanh(forget(gate)) * msg

The block-residual mix is attention over the depth: earlier block inputs plus the current
stream, unweighted-RMS-normed, combined by learned queries (scale 1/sqrt(H)). At each
6-layer boundary the layer input joins the blocks and the attention output starts a new
stream; the FFN input mixes the blocks again. Reference: llama.cpp's inkling2_loop graph
(ggml-org/llama.cpp#32694) and the transformers inkling2 decoder layers.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNorm, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel, embed_input_ids
from freetoken.models.config import ModelConfig
from freetoken.utils import nvtx_annotate

from .args import Inkling2Args
from .attention import Inkling2Attention, unweighted_rms_norm
from .conv import ConvMeta, Inkling2Conv, build_conv_meta, conv_state_slab
from .moe import Inkling2MoE


def mix(blocks: List[torch.Tensor], cur: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Attention over the depth streams: ``[T, Q, H]`` fp32 (the BlockAttentionResidual)."""
    states = torch.stack([b.float() for b in blocks] + [cur.float()], dim=1)
    states = unweighted_rms_norm(states, 1e-6)
    hidden = states.shape[-1]
    scores = torch.einsum("tsh,qh->tsq", states, query.float()) * hidden**-0.5
    probs = torch.softmax(scores, dim=1)
    return torch.einsum("tsq,tsh->tqh", probs, states)


class LoopController(BaseOP):
    """One rank-64 message controller (down / gain / forget), bf16 compute as the ggml graph."""

    def __init__(self, hidden: int, rank: int):
        self.down_weight = torch.empty(rank, 2 * hidden)
        self.gain_weight = torch.empty(hidden, rank)
        self.forget_weight = torch.empty(hidden, rank)

    def forward(self, cur: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        dtype = self.down_weight.dtype
        cur = cur.to(dtype)
        msg = msg.to(dtype)
        gate = torch.cat([unweighted_rms_norm(cur, 1e-6), unweighted_rms_norm(msg, 1e-6)], dim=-1)
        h = F.linear(gate, self.down_weight)
        gain = torch.tanh(F.linear(h, self.gain_weight))
        forget = torch.tanh(F.linear(h, self.forget_weight))
        return (1.0 - forget) * cur + gain * msg


class Inkling2DecoderLayer(BaseOP):
    """One physical layer; ``forward`` takes the VISIT id (its KV slot and conv state row)."""

    def __init__(self, config: ModelConfig, phys_layer: int, *, prefix: str = ""):
        args: Inkling2Args = config.inkling2_args
        self._phys_layer = phys_layer
        self.args = args
        self.swa = phys_layer in args.swa_physical
        self.is_boundary = phys_layer % args.block_size == 0
        self.is_core = args.core_start <= phys_layer < args.core_end
        H = config.hidden_size
        self.self_attn = Inkling2Attention(config, phys_layer, prefix=f"{prefix}.self_attn")
        self.mlp = Inkling2MoE(config, phys_layer, prefix=f"{prefix}.mlp")
        self.mlp_norm = RMSNorm(H, config.rms_norm_eps)
        self.ffn_post_conv = Inkling2Conv(config, H, prefix=f"{prefix}.mlp_sconv")
        if phys_layer > 0:
            self.attn_res_query = torch.empty(1 if self.swa else 3, H, dtype=torch.float32)
        self.ffn_res_query = torch.empty(1, H, dtype=torch.float32)
        if self.is_core:
            self.loop_msg_query = torch.empty(4, H, dtype=torch.float32)
            self.controllers = OPList([LoopController(H, args.gate_rank) for _ in range(4)])

    @nvtx_annotate("Layer_{}", layer_id_field="_phys_layer")
    def forward(
        self,
        x: torch.Tensor,
        blocks: List[torch.Tensor],
        visit: int,
        conv_meta: ConvMeta,
        positions: torch.Tensor,
        messages: torch.Tensor | None = None,
    ) -> torch.Tensor:
        args = self.args
        inpSA = x
        if self._phys_layer > 0:
            mixed = mix(blocks, x, self.attn_res_query)
            if self.swa:
                inputs = [mixed[:, 0]] * 3
            else:
                inputs = [mixed[:, j] for j in range(3)]
        else:
            inputs = [x, x, x]
        if messages is not None:
            inputs = [self.controllers.op_list[j].forward(inputs[j], messages[:, j]) for j in range(3)]
        normed = self.self_attn.norm_inputs(inputs, separate=messages is not None)
        attn_out = self.self_attn.forward(normed, visit, conv_meta, positions)

        if self.is_boundary:
            blocks.append(inpSA)
            residual = attn_out.float()
        else:
            residual = inpSA.float() + attn_out.float()

        ffn_in = mix(blocks, residual, self.ffn_res_query)[:, 0]
        if messages is not None:
            ffn_in = self.controllers.op_list[3].forward(ffn_in, messages[:, 3])
        h = self.mlp_norm.forward(ffn_in.to(x.dtype))
        h = self.mlp.forward(h)
        h = self.ffn_post_conv.forward(h, conv_meta, conv_state_slab(visit, args.hidden_size, 2 * args.kv_dim + args.hidden_size))
        return (residual + h.float()).to(x.dtype)


class _NormedEmbedding:
    """embed_tokens + embed_norm as one lookup, so image rows scatter in AFTER the norm (the reference never norms encoded image embeddings)."""

    def __init__(self, embed_tokens: VocabParallelEmbedding, embed_norm: RMSNorm):
        self.embed_tokens = embed_tokens
        self.embed_norm = embed_norm

    @property
    def num_embeddings(self) -> int:
        return self.embed_tokens.num_embeddings

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm.forward(self.embed_tokens.forward(input_ids))


class Inkling2TextModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        args: Inkling2Args = config.inkling2_args
        self.args = args
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=args.padded_vocab_size,
            embedding_dim=config.hidden_size,
            quant_config=config.quant,
            prefix=f"{prefix}.embed_tokens",
        )
        self.embed_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self._normed_embedding = _NormedEmbedding(self.embed_tokens, self.embed_norm)
        self.layers = OPList(
            [Inkling2DecoderLayer(config, p, prefix=f"{prefix}.layers.{p}") for p in range(args.num_phys_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.output_res_query = torch.empty(1, config.hidden_size, dtype=torch.float32)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = embed_input_ids(self._normed_embedding, input_ids, get_global_ctx().batch)
        return self.forward_embeds(x)

    def forward_embeds(self, x: torch.Tensor) -> torch.Tensor:
        args = self.args
        ctx = get_global_ctx()
        conv_meta = build_conv_meta(ctx.batch, x.device)
        positions = ctx.batch.get_attn_positions()
        layers = self.layers.op_list
        blocks: List[torch.Tensor] = []

        for il in range(args.core_start):
            x = layers[il].forward(x, blocks, il, conv_meta, positions)

        prelude = x
        prelude_blocks = list(blocks)
        core = args.core_end - args.core_start
        for p in range(args.num_passes):
            if p > 0:
                # the previous pass's second-core-block input and its output
                bank = blocks[len(prelude_blocks) + 1 :]
                bank_out = x
                x = prelude
                blocks = list(prelude_blocks)
            for ip in range(args.core_start, args.core_end):
                il = ip if p == 0 else args.num_phys_layers + (p - 1) * core + (ip - args.core_start)
                messages = None
                if p > 0:
                    messages = mix(bank, bank_out, layers[ip].loop_msg_query)
                x = layers[ip].forward(x, blocks, il, conv_meta, positions, messages)

        for il in range(args.core_end, args.num_phys_layers):
            x = layers[il].forward(x, blocks, il, conv_meta, positions)

        x = mix(blocks, x, self.output_res_query)[:, 0]
        return self.norm.forward(x.to(self.norm.weight.dtype))


class Inkling2LoopForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        if get_tp_info().size > 1:
            raise NotImplementedError("inkling2_loop supports TP=1 only")
        args: Inkling2Args = config.inkling2_args
        self.model = Inkling2TextModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=args.padded_vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=False,
            quant_config=config.quant,
            prefix="lm_head",
        )
        self._softcap = args.final_logit_softcapping
        self._unpadded = args.unpadded_vocab_size
        super().__init__()

    def forward(self) -> torch.Tensor:
        hidden = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(hidden).float()
        if self._softcap:
            logits = torch.tanh(logits / self._softcap) * self._softcap
        # contiguous: the padded-vocab slice is a view, and the sampler rejects strided logits
        return logits[:, : self._unpadded].contiguous()


class Inkling2LoopForConditionalGeneration(Inkling2LoopForCausalLM):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.is_multimodal:
            from .vision import Inkling2VisionModel

            self.visual = Inkling2VisionModel(config.vision_config)

    def place_encoder_weights(self, mode: str) -> None:
        self.visual.place_weights(mode)

    def encode(self, item) -> torch.Tensor:
        return self.visual.forward(item.feature, item.grid_thw)


__all__ = [
    "Inkling2DecoderLayer",
    "Inkling2LoopForCausalLM",
    "Inkling2LoopForConditionalGeneration",
    "Inkling2TextModel",
    "LoopController",
    "mix",
]
