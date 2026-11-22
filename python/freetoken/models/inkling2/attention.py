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
from freetoken.layers import BaseOP, LinearReplicated, LinearRowParallel, RMSNorm
from freetoken.layers.linear import LinearColParallelMerged
from freetoken.distributed import get_tp_info
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
        tp = get_tp_info().size
        self.nh, self.nk, self.hd = args.num_qo_heads // tp, args.num_kv_heads // tp, args.head_dim
        self.d_rel = args.d_rel
        H = config.hidden_size
        local_q = self.nh * self.hd
        local_kv = self.nk * self.hd
        col = lambda out_dim, name: LinearColParallelMerged(
            H, [out_dim], has_bias=False, quant_config=config.quant, prefix=f"{prefix}.{name}"
        )
        self.q_proj = col(args.q_dim, "q_proj")
        self.k_proj = col(args.kv_dim, "k_proj")
        self.v_proj = col(args.kv_dim, "v_proj")
        self.r_proj = col(args.num_qo_heads * self.d_rel, "r_proj")
        self.gate_proj = col(args.q_dim, "gate_proj")
        self.sink_proj = col(args.num_qo_heads, "sink_proj")
        self.o_proj = LinearRowParallel(
            args.q_dim, H, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.o_proj"
        )
        self.q_clip_scale = torch.empty(self.nh, self.hd)
        self.rel_clip_scale = torch.empty(self.nh, self.d_rel)
        extent = args.sliding_window if self.swa else args.rel_extent
        self.rel_bias = torch.empty(self.d_rel, extent)
        if self.swa:
            self.input_layernorm = RMSNorm(H, config.rms_norm_eps)
            self.k_conv = Inkling2Conv(config, local_kv, prefix=f"{prefix}.k_sconv")
            self.v_conv = Inkling2Conv(config, local_kv, prefix=f"{prefix}.v_sconv")
        else:
            self.q_input_norm = RMSNorm(H, config.rms_norm_eps)
            self.k_input_norm = RMSNorm(H, config.rms_norm_eps)
            self.v_input_norm = RMSNorm(H, config.rms_norm_eps)
            self.k_conv = Inkling2Conv(config, local_kv, num_heads=self.nk, prefix=f"{prefix}.k_sconv")
            self.v_conv = Inkling2Conv(config, local_kv, num_heads=self.nk, prefix=f"{prefix}.v_sconv")
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

    @staticmethod
    def _vk_ggml_single(proj: LinearBase):
        """Return (weight_slice, ggml_type) if `proj` is a single-segment quantized ggml linear supported by ik2_vk."""
        segs = getattr(proj, "_ggml_segments", None)
        if not segs or len(segs) != 1 or proj.bias is not None or getattr(proj, "ggml_scale", None) is not None:
            return None
        from freetoken.kernel import ik2_vk
        from freetoken.layers.quantization.ggml_types import is_quantized, segment_param

        s = segs[0]
        if not is_quantized(s.ggml_type) or not ik2_vk.supports_type(s.ggml_type):
            return None
        return getattr(proj, segment_param(0))[: s.rows], s.ggml_type

    @nvtx_annotate("MHA")
    def forward(self, inputs: list[torch.Tensor], visit: int, conv_meta: ConvMeta, positions: torch.Tensor) -> torch.Tensor:
        args = self.args
        ctx = get_global_ctx()
        xq, xk, xv = inputs
        from freetoken.kernel import ggml
        from .model import _IK2_PROF, _prof
        import time as _time

        if _IK2_PROF:
            torch.cuda.synchronize()
            _a0 = _time.perf_counter()

        if ggml.backend_name() == "vulkan":
            # Batch projections sharing xq into ONE Vulkan command buffer & ONE handoff.
            shared_projs = [self.q_proj, self.r_proj, self.gate_proj]
            if xk is xq and xv is xq:
                shared_projs.extend((self.k_proj, self.v_proj))
            vk_specs = [self._vk_ggml_single(p) for p in shared_projs]
            if all(sp is not None for sp in vk_specs):
                from freetoken.kernel import ik2_vk

                m = xq.shape[0]
                ws = [sp[0] for sp in vk_specs]
                tys = [sp[1] for sp in vk_specs]
                outs = [torch.empty(m, w.shape[0], dtype=xq.dtype, device=xq.device) for w in ws]
                ik2_vk.mul_mat_batch(xq, ws, tys, outs)
                q, r, gate = outs[0], outs[1], outs[2]
                if xk is xq and xv is xq:
                    k_raw, v_raw = outs[3], outs[4]
                else:
                    k_raw = self.k_proj.forward(xk)
                    v_raw = self.v_proj.forward(xv)
                sinks = self.sink_proj.forward(xq)
                k = self.k_conv.forward(k_raw, conv_meta, conv_state_slab(visit, self.nk * self.hd, 0), dyn_input=xk if not self.swa else None)
                v = self.v_conv.forward(v_raw, conv_meta, conv_state_slab(visit, self.nk * self.hd, self.nk * self.hd), dyn_input=xv if not self.swa else None)
            else:
                q = self.q_proj.forward(xq)
                k = self.k_conv.forward(self.k_proj.forward(xk), conv_meta, conv_state_slab(visit, self.nk * self.hd, 0), dyn_input=xk if not self.swa else None)
                v = self.v_conv.forward(self.v_proj.forward(xv), conv_meta, conv_state_slab(visit, self.nk * self.hd, self.nk * self.hd), dyn_input=xv if not self.swa else None)
                r = self.r_proj.forward(xq)
                gate = self.gate_proj.forward(xq)
                sinks = self.sink_proj.forward(xq)
        else:
            q = self.q_proj.forward(xq)
            k = self.k_conv.forward(self.k_proj.forward(xk), conv_meta, conv_state_slab(visit, self.nk * self.hd, 0), dyn_input=xk if not self.swa else None)
            v = self.v_conv.forward(self.v_proj.forward(xv), conv_meta, conv_state_slab(visit, self.nk * self.hd, self.nk * self.hd), dyn_input=xv if not self.swa else None)
            r = self.r_proj.forward(xq)
            gate = self.gate_proj.forward(xq)
            sinks = self.sink_proj.forward(xq)

        tokens = q.shape[0]
        if _IK2_PROF:
            torch.cuda.synchronize()
            _a1 = _time.perf_counter()
            _prof.add("attn_projs", _a1 - _a0)
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
        if _IK2_PROF:
            torch.cuda.synchronize()
            _a2 = _time.perf_counter()
            _prof.add("attn_clips_rel", _a2 - _a1)
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
        if _IK2_PROF:
            torch.cuda.synchronize()
            _a3 = _time.perf_counter()
            _prof.add("attn_triton", _a3 - _a2)
        out = out.reshape(tokens, self.nh * self.hd)
        out = (out.float() * torch.sigmoid(gate.float())).to(out.dtype)
        out = self.o_proj.forward(out.reshape(tokens, self.nh * self.hd))
        out = self.attn_post_conv.forward(out, conv_meta, conv_state_slab(visit, args.hidden_size, 2 * args.kv_dim))
        if _IK2_PROF:
            torch.cuda.synchronize()
            _a4 = _time.perf_counter()
            _prof.add("attn_gate_o_conv", _a4 - _a3)
        return out


__all__ = ["Inkling2Attention", "l2_clip", "unweighted_rms_norm"]
