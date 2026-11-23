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
import torch.distributed as dist
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNorm, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel, embed_input_ids
from freetoken.models.config import ModelConfig
from freetoken.utils import nvtx_annotate

from .args import Inkling2Args

import os as _os
import time as _time

_IK2_PROF = _os.environ.get("FREETOKEN_IK2_PROFILE") is not None


class _Prof:
    def __init__(self):
        self.t: dict[str, float] = {}
        self.n: dict[str, int] = {}
        self.total_t0 = 0.0
        self.total_n = 0

    def add(self, name, dt):
        self.t[name] = self.t.get(name, 0.0) + dt
        self.n[name] = self.n.get(name, 0) + 1

    def report(self):
        if self.total_n == 0:
            return
        import sys
        n = self.total_n
        lines = [f"inkling2 profile ({n} tokens):"]
        for k in sorted(self.t, key=lambda k: -self.t[k]):
            lines.append(f"  {k:24s} {self.t[k]/n*1e3:8.2f} ms/tok  ({self.n[k]//n:3d} calls, {self.t[k]/max(1,self.n[k])*1e3:.3f} ms/call)")
        print("\n".join(lines), file=sys.stderr, flush=True)
        self.t.clear()
        self.n.clear()
        self.total_n = 0


_prof = _Prof()
from .attention import Inkling2Attention, unweighted_rms_norm
from .conv import ConvMeta, Inkling2Conv, build_conv_meta, conv_state_slab
from .moe import Inkling2MoE


def mix(blocks: List[torch.Tensor], cur: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Attention over the depth streams: ``[T, Q, H]`` fp32 (the BlockAttentionResidual)."""
    states = torch.stack(blocks + [cur], dim=1)
    if states.is_cuda:
        from freetoken.kernel.triton.ik2_mix import mix_fused

        return mix_fused(states, query)
    from freetoken.kernel.triton.ik2_mix import mix_ref

    return mix_ref(blocks, cur, query)


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
        if cur.is_cuda:
            from freetoken.kernel.triton.ik2_ctrl import ctrl_gate_out, ctrl_rms_cat

            gate = ctrl_rms_cat(cur, msg)
            h = F.linear(gate, self.down_weight)
            gain_raw = F.linear(h, self.gain_weight)
            forget_raw = F.linear(h, self.forget_weight)
            return ctrl_gate_out(gain_raw, forget_raw, cur, msg)
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
        if _IK2_PROF:
            torch.cuda.synchronize()
            _t0 = _time.perf_counter()
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

        if _IK2_PROF:
            torch.cuda.synchronize()
            _t1 = _time.perf_counter()
            _prof.add("attn_phase", _t1 - _t0)

        ffn_in = mix(blocks, residual, self.ffn_res_query)[:, 0]
        if messages is not None:
            ffn_in = self.controllers.op_list[3].forward(ffn_in, messages[:, 3])
        h = self.mlp_norm.forward(ffn_in.to(x.dtype))
        h = self.mlp.forward(h)

        if _IK2_PROF:
            torch.cuda.synchronize()
            _t2 = _time.perf_counter()
            _prof.add("moe_phase", _t2 - _t1)

        # split mode builds the conv slab at TP1 width (args.conv_channels); the offset
        # must match it, not the real TP size, or the FFN conv overwrites the K/V history
        _tp = 1 if _os.environ.get("FREETOKEN_LAYER_SPLIT") == "1" else get_tp_info().size
        _ffn_off = 2 * (args.kv_dim // _tp) + args.hidden_size
        h = self.ffn_post_conv.forward(h, conv_meta, conv_state_slab(visit, args.hidden_size, _ffn_off), verify_key=(visit, _ffn_off))
        out = (residual + h.float()).to(x.dtype)
        if _IK2_PROF:
            torch.cuda.synchronize()
            _t3 = _time.perf_counter()
            _prof.add("post_conv_res", _t3 - _t2)
        return out


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


def _split_send(t: torch.Tensor, dst: int) -> None:
    from freetoken.distributed import split_p2p_group

    grp = split_p2p_group()
    if grp is not None:
        dist.send(t.contiguous(), dst=dst, group=grp)  # NCCL: GPU-direct, no host staging
        return
    # gloo fallback: its TCP transport writev()s the buffer pointer, so a CUDA address is
    # EFAULT ("Bad address"); stage through host memory.
    host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
    host.copy_(t)
    dist.send(host, dst=dst)


def _split_recv(shape, dtype, device, src: int) -> torch.Tensor:
    from freetoken.distributed import split_p2p_group

    grp = split_p2p_group()
    if grp is not None:
        buf = torch.empty(shape, dtype=dtype, device=device)
        dist.recv(buf, src=src, group=grp)
        return buf
    host = torch.empty(shape, dtype=dtype, pin_memory=True)
    dist.recv(host, src=src)
    return host.to(device)


class Inkling2TextModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        args: Inkling2Args = config.inkling2_args
        self.args = args
        # Split mode: build the model as TP1 (full-width layers, replicated weights)
        from freetoken.distributed import info as _di
        _saved_tp = _di._TP_INFO
        if _os.environ.get("FREETOKEN_LAYER_SPLIT") == "1":
            _di._TP_INFO = _di.DistributedInfo(rank=0, size=1)
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
        if _os.environ.get("FREETOKEN_LAYER_SPLIT") == "1":
            _di._TP_INFO = _saved_tp

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = embed_input_ids(self._normed_embedding, input_ids, get_global_ctx().batch)
        if _os.environ.get("FREETOKEN_LAYER_SPLIT") == "1" and get_tp_info().size > 1:
            return self.forward_split(x)
        return self.forward_embeds(x)

    def forward_embeds(self, x: torch.Tensor) -> torch.Tensor:
        args = self.args
        ctx = get_global_ctx()
        if _IK2_PROF:
            torch.cuda.synchronize()
            _prof.total_t0 = _time.perf_counter()
        conv_meta = build_conv_meta(ctx.batch, x.device)
        positions = ctx.batch.get_attn_positions()
        layers = self.layers.op_list
        blocks: List[torch.Tensor] = []
        # the DSpark drafter's taps: the stream at the start of a tap visit (sequential visit
        # id, not the slot id) and the final normed hidden; the sink copies into its own buffers
        features = ctx.batch.features
        wanted = features.wanted_layers if features is not None else ()
        vis = 0

        for il in range(args.core_start):
            if vis in wanted:
                features.layer_output(vis, x)
            x = layers[il].forward(x, blocks, il, conv_meta, positions)
            vis += 1

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
                if vis in wanted:
                    features.layer_output(vis, x)
                x = layers[ip].forward(x, blocks, il, conv_meta, positions, messages)
                vis += 1

        for il in range(args.core_end, args.num_phys_layers):
            if vis in wanted:
                features.layer_output(vis, x)
            x = layers[il].forward(x, blocks, il, conv_meta, positions)
            vis += 1

        x = mix(blocks, x, self.output_res_query)[:, 0]
        out = self.norm.forward(x.to(self.norm.weight.dtype))
        if features is not None:
            features.final_hidden(out)
        if _IK2_PROF:
            torch.cuda.synchronize()
            _prof.add("total_forward", _time.perf_counter() - _prof.total_t0)
            _prof.total_n += int(x.shape[0])
            if _prof.total_n >= 10:
                _prof.report()
        return out



    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        """Layer-split forward: each rank runs its local layers, transfers at the boundary.

        Enabled by FREETOKEN_LAYER_SPLIT=1 with tp_size=2. Rank 0 owns layers 0..split-1,
        rank 1 owns layers split..N-1. Weights are replicated (both ranks hold all layers);
        only the COMPUTE is split. No allreduce — just point-to-point at the boundary.
        """
        args = self.args
        ctx = get_global_ctx()
        tp = get_tp_info()
        split = args.num_phys_layers // tp.size  # 12 for 24 layers / 2
        rank = tp.rank
        peer = 1 - rank
        device = x.device
        # every message is the model dtype: rank 1 enters with an empty x, so the recv
        # buffers take it from the weights (blocks ride the residual stream's dtype too)
        dtype = self.norm.weight.dtype
        conv_meta = build_conv_meta(ctx.batch, device)
        positions = ctx.batch.get_attn_positions()
        layers = self.layers.op_list
        blocks: List[torch.Tensor] = []
        H = args.hidden_size
        T = x.shape[0] if rank == 0 else int(ctx.batch.input_ids.shape[0])
        # block counts at each transfer are fixed by the loop geometry (a boundary layer
        # appends its input); static counts keep the forward CUDA-graph capturable (no
        # data-dependent recv loop, no device->host sync)
        bsz = args.block_size
        n_prelude = sum(1 for il in range(args.core_start) if il % bsz == 0)
        n_first = n_prelude + sum(1 for ip in range(args.core_start, split) if ip % bsz == 0)
        n_second = n_first + sum(1 for ip in range(split, args.core_end) if ip % bsz == 0)

        def send_x(t):
            _split_send(t.to(dtype), peer)

        def recv_x():
            return _split_recv((T, H), dtype, device, peer)

        def send_blocks(bl):
            for b in bl:
                _split_send(b.to(dtype), peer)

        def recv_blocks(n):
            return [_split_recv((T, H), dtype, device, peer) for _ in range(n)]

        # --- Prelude (layers 0..core_start-1): all on rank 0 ---
        if rank == 0:
            for il in range(args.core_start):
                x = layers[il].forward(x, blocks, il, conv_meta, positions)
            prelude = x.clone()
            prelude_blocks = list(blocks)
            # send prelude output + blocks to rank 1 for core
            send_x(x)
            send_blocks(blocks)
        else:
            x = recv_x()
            blocks = recv_blocks(n_prelude)
            prelude = x.clone()
            prelude_blocks = list(blocks)

        # --- Core passes ---
        core = args.core_end - args.core_start
        for p in range(args.num_passes):
            if p > 0:
                bank = blocks[len(prelude_blocks) + 1:]
                bank_out = x
                x = prelude.clone()
                blocks = list(prelude_blocks)

            # First half (core_start..split-1): rank 0
            if rank == 0:
                for ip in range(args.core_start, split):
                    il = ip if p == 0 else args.num_phys_layers + (p - 1) * core + (ip - args.core_start)
                    messages = None
                    if p > 0:
                        messages = mix(bank, bank_out, layers[ip].loop_msg_query)
                    x = layers[ip].forward(x, blocks, il, conv_meta, positions, messages)
                send_x(x)
                send_blocks(blocks)
            else:
                x = recv_x()
                blocks = recv_blocks(n_first)

            # Second half (split..core_end-1): rank 1
            if rank == 1:
                for ip in range(split, args.core_end):
                    il = ip if p == 0 else args.num_phys_layers + (p - 1) * core + (ip - args.core_start)
                    messages = None
                    if p > 0:
                        messages = mix(bank, bank_out, layers[ip].loop_msg_query)
                    x = layers[ip].forward(x, blocks, il, conv_meta, positions, messages)
                # send back for next pass (or coda)
                send_x(x)
                send_blocks(blocks)
            else:
                x = recv_x()
                blocks = recv_blocks(n_second)

        # --- Coda (core_end..N-1): rank 1 ---
        if rank == 1:
            for il in range(args.core_end, args.num_phys_layers):
                x = layers[il].forward(x, blocks, il, conv_meta, positions)
            x = mix(blocks, x, self.output_res_query)[:, 0]
            out = self.norm.forward(x.to(self.norm.weight.dtype))
            send_x(out)
        else:
            out = recv_x()

        return out


class Inkling2LoopForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        # TP>=2: attention heads split column/row parallel, experts EP-sharded
        args: Inkling2Args = config.inkling2_args
        self.model = Inkling2TextModel(config)
        from freetoken.distributed import info as _di2
        _saved_tp2 = _di2._TP_INFO
        if _os.environ.get("FREETOKEN_LAYER_SPLIT") == "1":
            _di2._TP_INFO = _di2.DistributedInfo(rank=0, size=1)
        self.lm_head = ParallelLMHead(
            num_embeddings=args.padded_vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=False,
            quant_config=config.quant,
            prefix="lm_head",
        )
        if _os.environ.get("FREETOKEN_LAYER_SPLIT") == "1":
            _di2._TP_INFO = _saved_tp2
        self._softcap = args.final_logit_softcapping
        self._unpadded = args.unpadded_vocab_size
        super().__init__()

    def forward(self) -> torch.Tensor:
        if _os.environ.get("FREETOKEN_LAYER_SPLIT") == "1" and get_tp_info().size > 1:
            return self._forward_split()
        hidden = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(hidden).float()
        if self._softcap:
            logits = torch.tanh(logits / self._softcap) * self._softcap
        # contiguous: the padded-vocab slice is a view, and the sampler rejects strided logits
        return logits[:, : self._unpadded].contiguous()

    def _forward_split(self) -> torch.Tensor:
        tp = get_tp_info()
        device = f"cuda:{tp.rank}"
        if tp.rank == 1:
            # rank 1 runs its layers inside forward_split, which sends the normed hidden
            # to rank 0; the head lives there, so receive the sampled logits back: both
            # ranks must sample the SAME logits or their schedulers diverge (eos, batches)
            self.model.forward_split(torch.empty(0, dtype=self.model.norm.weight.dtype, device=device))
            bs = get_global_ctx().batch.size
            return _split_recv((bs, self._unpadded), torch.float32, device, src=0)
        hidden = self.model.forward(get_global_ctx().batch.input_ids)
        logits = self.lm_head.forward(hidden).float()
        if self._softcap:
            logits = torch.tanh(logits / self._softcap) * self._softcap
        logits = logits[:, : self._unpadded].contiguous()
        _split_send(logits, dst=1)
        return logits


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
