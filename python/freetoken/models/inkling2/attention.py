"""Inkling-2 attention: NoPE GQA with a learned relative-position bias and a per-query sink.

    q, k, v, r = projections(normed inputs)          # full layers norm q/k/v separately
    k = short_conv(k, static [+ per-token taps])     # full layers: dynamic, head-shared
    v = short_conv(v, ...)
    q = l2_clip(q * q_scale);  k = l2_clip(k)        # per-head L2 norm capped at qk_clip
    v = unweighted_rms_norm(v)                       # no gamma
    r = l2_clip(r * rel_scale)
    full layers: q *= tau; r *= tau                  # tau = 1 + alpha*log(max(1, (pos+1)/floor))
    bias[q, k] = (r @ rel_table)[clamp(q_pos - k_pos, 0, extent)]   # zero beyond the extent
    out = softmax(qk/sqrt(hd) + bias, sink) @ v      # the sink is one extra logit per query
    out = o_proj(out * sigmoid(gate))

The rel bias and the sink ride the triton backend's AttentionSpec (rel_table / per-query
sinks); the KV cache slot is the VISIT id, so every loop pass keeps its own K/V.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.attention import AttentionSpec
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, LinearReplicated, RMSNorm
from freetoken.models.config import ModelConfig
from freetoken.utils import nvtx_annotate

from .args import Inkling2Args
from .conv import ConvMeta, Inkling2Conv, conv_state_slab


def l2_clip(x: torch.Tensor, threshold: float, scale: torch.Tensor | None = None) -> torch.Tensor:
    """Per-head L2 cap in fp32 (the reference's Inkling2L2Clip / ScaledL2Clip)."""
    xf = x.float()
    if scale is not None:
        xf = xf * scale.float()
    norm = torch.linalg.vector_norm(xf, dim=-1, keepdim=True)
    return (xf * (threshold / (norm + 1e-6)).clamp(max=1.0)).to(x.dtype)


def unweighted_rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)


class Inkling2Attention(BaseOP):
    def __init__(self, config: ModelConfig, phys_layer: int, *, prefix: str = ""):
        args: Inkling2Args = config.inkling2_args
        self.args = args
        self.phys_layer = phys_layer
        self.swa = phys_layer in args.swa_physical
        self.nh, self.nk, self.hd = args.num_qo_heads, args.num_kv_heads, args.head_dim
        self.d_rel = args.d_rel
        H = config.hidden_size
        lin = lambda out_dim, name: LinearReplicated(
            H, out_dim, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.{name}"
        )
        self.q_proj = lin(args.q_dim, "q_proj")
        self.k_proj = lin(args.kv_dim, "k_proj")
        self.v_proj = lin(args.kv_dim, "v_proj")
        self.r_proj = lin(self.nh * self.d_rel, "r_proj")
        self.gate_proj = lin(args.q_dim, "gate_proj")
        self.sink_proj = lin(self.nh, "sink_proj")
        self.o_proj = LinearReplicated(
            args.q_dim, H, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.o_proj"
        )
        self.q_clip_scale = torch.empty(self.nh, self.hd)
        self.rel_clip_scale = torch.empty(self.nh, self.d_rel)
        extent = args.sliding_window if self.swa else args.rel_extent
        self.rel_bias = torch.empty(self.d_rel, extent)
        if self.swa:
            self.input_layernorm = RMSNorm(H, config.rms_norm_eps)
            self.k_conv = Inkling2Conv(config, args.kv_dim, prefix=f"{prefix}.k_sconv")
            self.v_conv = Inkling2Conv(config, args.kv_dim, prefix=f"{prefix}.v_sconv")
        else:
            self.q_input_norm = RMSNorm(H, config.rms_norm_eps)
            self.k_input_norm = RMSNorm(H, config.rms_norm_eps)
            self.v_input_norm = RMSNorm(H, config.rms_norm_eps)
            self.k_conv = Inkling2Conv(config, args.kv_dim, num_heads=self.nk, prefix=f"{prefix}.k_sconv")
            self.v_conv = Inkling2Conv(config, args.kv_dim, num_heads=self.nk, prefix=f"{prefix}.v_sconv")
        self.attn_post_conv = Inkling2Conv(config, H, prefix=f"{prefix}.attn_sconv")
        self.attn_spec = AttentionSpec(
            sliding_window=args.sliding_window if self.swa else None,
            sm_scale=self.hd**-0.5,
        )

    def norm_inputs(self, inputs: list[torch.Tensor], separate: bool) -> list[torch.Tensor]:
        # the mix hands fp32 streams; the fused norm kernels read the weight's dtype
        dtype = self.input_layernorm.weight.dtype if self.swa else self.q_input_norm.weight.dtype
        inputs = [i.to(dtype) for i in inputs]
        if self.swa:
            if separate:
                return [self.input_layernorm.forward(i) for i in inputs]
            n = self.input_layernorm.forward(inputs[0])
            return [n, n, n]
        return [
            self.q_input_norm.forward(inputs[0]),
            self.k_input_norm.forward(inputs[1]),
            self.v_input_norm.forward(inputs[2]),
        ]

    @nvtx_annotate("MHA")
    def forward(self, inputs: list[torch.Tensor], visit: int, conv_meta: ConvMeta, positions: torch.Tensor) -> torch.Tensor:
        args = self.args
        ctx = get_global_ctx()
        xq, xk, xv = inputs
        q = self.q_proj.forward(xq)
        k = self.k_conv.forward(self.k_proj.forward(xk), conv_meta, conv_state_slab(visit, args.kv_dim, 0), dyn_input=xk if not self.swa else None)
        v = self.v_conv.forward(self.v_proj.forward(xv), conv_meta, conv_state_slab(visit, args.kv_dim, args.kv_dim), dyn_input=xv if not self.swa else None)
        r = self.r_proj.forward(xq)
        gate = self.gate_proj.forward(xq)
        sinks = self.sink_proj.forward(xq)

        tokens = q.shape[0]
        q = l2_clip(q.view(tokens, self.nh, self.hd), args.qk_clip, self.q_clip_scale)
        k = l2_clip(k.view(tokens, self.nk, self.hd), args.qk_clip)
        v = unweighted_rms_norm(v.view(tokens, self.nk, self.hd), args.rms_norm_eps)
        r = l2_clip(r.view(tokens, self.nh, self.d_rel), args.rel_clip, self.rel_clip_scale)
        if not self.swa:
            tau = 1.0 + args.log_scale_alpha * torch.log(
                ((positions + 1).float() / args.log_scale_floor).clamp(min=1.0)
            )
            tau = tau[:, None, None]
            q = (q.float() * tau).to(q.dtype)
            r = (r.float() * tau).to(r.dtype)

        # project r on the bias table, then the kernel gathers by distance; the padded
        # last column holds the zero bias of keys beyond the extent (bf16 matmul as HF)
        rel_table = F.pad(r @ self.rel_bias, (0, 1)).float().contiguous()
        spec = AttentionSpec(
            sliding_window=self.attn_spec.sliding_window,
            sm_scale=self.attn_spec.sm_scale,
            sinks=sinks.float(),
            rel_table=rel_table,
        )
        out = ctx.attn_backend.forward(
            q.view(tokens, self.nh, self.hd), k.view(tokens, self.nk * self.hd), v.view(tokens, self.nk * self.hd),
            visit, ctx.batch, attn_spec=spec,
        )
        out = out.reshape(tokens, args.q_dim)
        out = (out.float() * torch.sigmoid(gate.float())).to(out.dtype)
        out = self.o_proj.forward(out.reshape(tokens, args.q_dim))
        return self.attn_post_conv.forward(out, conv_meta, conv_state_slab(visit, args.hidden_size, 2 * args.kv_dim))


__all__ = ["Inkling2Attention", "l2_clip", "unweighted_rms_norm"]
