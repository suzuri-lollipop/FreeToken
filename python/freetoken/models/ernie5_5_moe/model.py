"""ERNIE 5.5 Flash model: 43-layer mHC decoder, HCA/DSA hybrid attention, latent MoE.

Engine adapter over the shared paged engine (DSV4 precedent): KV lives in
:class:`~freetoken.kvcache.dsv4.ernie55_pool.Ernie55PagedKVCache` (HCA window +
compressed tiers, DSA full-history latent/index slabs), addressed through
:class:`~freetoken.attention.ernie55_sparse.Ernie55SparseAttnBackend`. The
residual stream is ``hc_mult`` parallel copies mixed by manifold-constrained
Hyper-Connections; ERNIE's mHC differs from the shared GLM/DSV4 one ONLY in the
mixer RMS factor (``1 / (sqrt(mean) + eps)``, eps outside the sqrt -- the head
contract keeps the upstream ``rsqrt(mean + eps)``). The lm_head logits carry the
learned multimax correction (fp32).

The bundled DSpark draft (``model.mtp.*``) is not served yet (deepseek_v41
precedent); the reader skips it.
"""

from __future__ import annotations

import torch

from freetoken.core import get_global_ctx
from freetoken.kernel.triton.ernie55 import multimax_logits
from freetoken.layers import BaseOP, OPList, ParallelLMHead, RMSNorm, VocabParallelEmbedding
from freetoken.layers.mhc import hc_contract, mhc_post, mhc_pre
from freetoken.models.blocks import BaseLLMModel

from .args import Ernie55Args
from .attention import Ernie55DSAAttention, Ernie55HCAAttention
from .moe import Ernie55MLP, Ernie55MoE


class Block(BaseOP):
    """Decoder block with mHC (4 residual streams) around attention and FFN."""

    def __init__(
        self, layer_id: int, args: Ernie55Args, *, strategy: str = "offload",
        decode_target: str = "gpu", quant_config=None, prefix: str = "",
    ):
        self.layer_id = layer_id
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.hc_eps = args.hc_eps
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.norm_eps = args.norm_eps
        self.is_dsa = args.is_dsa_layer(layer_id)
        if self.is_dsa:
            self.attn = Ernie55DSAAttention(layer_id, args, quant_config=quant_config, prefix=f"{prefix}.attn")
        else:
            self.attn = Ernie55HCAAttention(layer_id, args, quant_config=quant_config, prefix=f"{prefix}.attn")
        if layer_id < args.first_k_dense_replace:
            self.ffn = Ernie55MLP(args.dim, args.intermediate_size, args, quant_config=quant_config, prefix=f"{prefix}.ffn")
        else:
            self.ffn = Ernie55MoE(layer_id, args, strategy=strategy, decode_target=decode_target, quant_config=quant_config, prefix=f"{prefix}.ffn")
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        mix_hc = (2 + args.hc_mult) * args.hc_mult
        hc_dim = args.hc_mult * args.dim
        self.hc_attn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_ffn_fn = torch.empty(mix_hc, hc_dim, dtype=torch.float32)
        self.hc_attn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_ffn_base = torch.empty(mix_hc, dtype=torch.float32)
        self.hc_attn_scale = torch.empty(3, dtype=torch.float32)
        self.hc_ffn_scale = torch.empty(3, dtype=torch.float32)

    def hc_pre(self, x, fn, scale, base):
        # x [..., hc, dim] -> shared mhc with the ERNIE RMS factor (eps outside).
        shape = x.shape
        flat = x.reshape(-1, self.hc_mult, self.dim)
        post, comb, y = mhc_pre(
            flat, fn, scale, base, self.norm_eps, self.hc_eps, 2.0,
            self.hc_sinkhorn_iters, rms_eps_outside=True,
        )
        return y.view(*shape[:-2], self.dim), post, comb

    def hc_post(self, x, residual, post, comb):
        shape = residual.shape
        y = mhc_post(
            x.reshape(-1, self.dim), residual.reshape(-1, self.hc_mult, self.dim), post, comb
        )
        return y.view(shape)

    def prefill_batched(self, x, input_ids, segments, flat_positions):
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm.forward(x)
        if self.is_dsa:
            x = self.attn.forward(x.view(-1, self.dim)).view(x.shape)
        else:
            x = self.attn.forward_ragged(x, segments, flat_positions)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm.forward(x)
        x = self.ffn.forward(x)
        x = self.hc_post(x, residual, post, comb)
        return x

    def decode_step(self, x, pos, rows, cmp_stage_cap, input_ids, wctx=None):
        residual = x
        x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.attn_norm.forward(x)
        if self.is_dsa:
            x = self.attn.forward(x.view(-1, self.dim)).view(x.shape)
        else:
            x = self.attn.decode_step(x, pos, rows, cmp_stage_cap, wctx)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.ffn_norm.forward(x)
        x = self.ffn.forward(x)
        x = self.hc_post(x, residual, post, comb)
        return x


class Transformer(BaseOP):
    def __init__(self, args: Ernie55Args, quant_config=None, *, strategy: str = "offload", decode_target: str = "gpu", prefix: str = ""):
        self.args = args
        self.hc_mult = args.hc_mult
        self.hc_eps = args.hc_eps
        self.norm_eps = args.norm_eps
        self.embed = VocabParallelEmbedding(args.vocab_size, args.dim, quant_config=quant_config, prefix=f"{prefix}.embed")
        self.layers = OPList([
            Block(i, args, strategy=strategy, decode_target=decode_target, quant_config=quant_config, prefix=f"{prefix}.layers.{i}")
            for i in range(args.n_layers)
        ])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = ParallelLMHead(args.vocab_size, args.dim, quant_config=quant_config, prefix=f"{prefix}.head")
        hc_dim = args.hc_mult * args.dim
        self.hc_head_fn = torch.empty(args.hc_mult, hc_dim, dtype=torch.float32)
        self.hc_head_base = torch.empty(args.hc_mult, dtype=torch.float32)
        self.hc_head_scale = torch.empty(1, dtype=torch.float32)
        self.multimax_ranges = torch.empty(4, dtype=torch.float32)
        self.multimax_ts = torch.empty(4, dtype=torch.float32)

    def bind(self, pool, device: torch.device) -> None:
        for layer in self.layers.op_list:
            if not layer.is_dsa:
                layer.attn.bind(pool, device)

    def prepare_for_runtime(self) -> None:
        for layer in self.layers.op_list:
            layer.attn.prepare_for_runtime()

    def hc_head(self, x):
        # The head contract keeps the upstream RMS factor (rsqrt(mean + eps));
        # only the per-layer mixers use the ERNIE eps-outside variant.
        shape = x.shape
        dim = self.args.dim
        xf = x.reshape(-1, self.hc_mult * dim).float()
        rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = xf @ self.hc_head_fn.t() * rsqrt
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        y = (pre.unsqueeze(-1) * xf.view(-1, self.hc_mult, dim)).sum(dim=1)
        return y.view(*shape[:-2], dim).to(x.dtype)

    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.head.forward(h).float()
        return multimax_logits(logits, self.multimax_ranges, self.multimax_ts)

    def prefill_batched(self, input_ids, segments, flat_positions):
        h = self.embed.forward(input_ids.view(-1)).view(1, -1, self.args.dim)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        features = get_global_ctx().batch.features
        wanted = features.wanted_layers if features is not None else ()
        for layer in self.layers.op_list:
            h = layer.prefill_batched(h, input_ids, segments, flat_positions)
            if layer.layer_id in wanted:
                features.layer_output(layer.layer_id, hc_contract(h[0]))
        h = self.hc_head(h)
        h = self.norm.forward(h)
        if features is not None:
            features.final_hidden(h[0])
        return self._logits(h[0])

    def decode(self, input_ids, pos, cmp_stage_cap):
        B = input_ids.size(0)
        rows = torch.arange(B, device=input_ids.device)
        h = self.embed.forward(input_ids.view(-1)).view(B, 1, self.args.dim)
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        wctx = get_global_ctx().batch.attn_metadata.window_ctx(pos, rows)
        features = get_global_ctx().batch.features
        wanted = features.wanted_layers if features is not None else ()
        for layer in self.layers.op_list:
            h = layer.decode_step(h, pos, rows, cmp_stage_cap, input_ids, wctx)
            if layer.layer_id in wanted:
                features.layer_output(layer.layer_id, hc_contract(h[:, -1]))
        h = self.hc_head(h)
        h = self.norm.forward(h)
        if features is not None:
            features.final_hidden(h[:, -1])
        return self._logits(h[:, -1])


class Ernie55ForCausalLM(BaseLLMModel):
    """Registered engine adapter; KV pools / rope bind on the first forward."""

    def __init__(self, config):
        self._config = config
        self._args: Ernie55Args = config.dsv4_args
        self.model = Transformer(
            self._args, config.quant, strategy=config.moe_strategy,
            decode_target=config.decode_target, prefix="model",
        )
        self._bound = False

    def _ensure_bound(self) -> None:
        if self._bound:
            return
        pool = get_global_ctx().kv_cache
        self.model.bind(pool, pool.device)
        self._bound = True

    def mark_for_rebind(self) -> None:
        self._bound = False

    def prepare_for_runtime(self) -> None:
        self.model.prepare_for_runtime()

    def forward(self) -> torch.Tensor:
        self._ensure_bound()
        batch = get_global_ctx().batch
        input_ids = batch.input_ids.long()
        md = batch.attn_metadata
        if batch.is_prefill:
            return self.model.prefill_batched(
                input_ids.view(1, -1), md.segments, batch.positions.long()
            )
        # a spec verify forwards rows_per_req query rows per request, so size off the
        # actual row count, not the request count (padded_size)
        B = input_ids.size(0)
        pos = batch.positions.long().view(-1)[:B]
        if torch.cuda.is_current_stream_capturing():
            cmp_stage_cap = md.stage_width - 1
        else:
            cmp_stage_cap = int(pos.max().item())
        return self.model.decode(input_ids.view(B, 1), pos, cmp_stage_cap)


__all__ = ["Transformer", "Ernie55ForCausalLM"]
