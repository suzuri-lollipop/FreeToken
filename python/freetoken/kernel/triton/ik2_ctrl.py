"""Fused Inkling-2 loop-controller elementwise stages.

The rank-64 message controller (down / gain / forget) runs 96x per decode step
(4 controllers x 12 core layers x 2 loop passes). Its three GEMVs stay on cuBLAS;
the surrounding elementwise -- two RMS norms + concat, and two tanh + the gated
residual -- fused here into two kernels instead of ~8 tiny ones. fp32 compute as
the reference; CPU falls back to torch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _rms_cat_kernel(GATE, CUR, MSG, s_cur, s_msg, H: tl.constexpr, eps: tl.constexpr, BLOCK_H: tl.constexpr):
    t = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    m = h < H
    cur = tl.load(CUR + t * s_cur + h, mask=m, other=0.0).to(tl.float32)
    msg = tl.load(MSG + t * s_msg + h, mask=m, other=0.0).to(tl.float32)
    rc = cur * tl.math.rsqrt(tl.sum(cur * cur, axis=0) / H + eps)
    rm = msg * tl.math.rsqrt(tl.sum(msg * msg, axis=0) / H + eps)
    tl.store(GATE + t * 2 * H + h, rc.to(GATE.dtype.element_ty), mask=m)
    tl.store(GATE + t * 2 * H + H + h, rm.to(GATE.dtype.element_ty), mask=m)


@triton.jit
def _gate_out_kernel(OUT, GAIN, FORGET, CUR, MSG, s_cur, s_msg, H: tl.constexpr, BLOCK_H: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    m = h < H
    g = tl.load(GAIN + t * H + h, mask=m, other=0.0).to(tl.float32)
    f = tl.load(FORGET + t * H + h, mask=m, other=0.0).to(tl.float32)
    cur = tl.load(CUR + t * s_cur + h, mask=m, other=0.0).to(tl.float32)
    msg = tl.load(MSG + t * s_msg + h, mask=m, other=0.0).to(tl.float32)
    gain = libdevice.tanh(g)
    forget = libdevice.tanh(f)
    out = (1.0 - forget) * cur + gain * msg
    tl.store(OUT + t * H + h, out.to(OUT.dtype.element_ty), mask=m)


def ctrl_rms_cat(cur: torch.Tensor, msg: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``cur, msg [T, H]`` (bf16) -> ``[T, 2H]`` bf16 = cat(rms(cur), rms(msg))."""
    T, H = cur.shape
    gate = torch.empty(T, 2 * H, dtype=cur.dtype, device=cur.device)
    if T == 0:
        return gate
    _rms_cat_kernel[(T,)](gate, cur, msg, cur.stride(0), msg.stride(0), H, eps,
                          BLOCK_H=triton.next_power_of_2(H), num_warps=8, num_stages=1)
    return gate


def ctrl_gate_out(gain_raw, forget_raw, cur, msg) -> torch.Tensor:
    """``(1 - tanh(forget_raw)) * cur + tanh(gain_raw) * msg``, all [T, H] bf16."""
    T, H = cur.shape
    out = torch.empty(T, H, dtype=cur.dtype, device=cur.device)
    if T == 0:
        return out
    BLOCK_H = 1024 if H >= 1024 else triton.next_power_of_2(H)
    _gate_out_kernel[(T, triton.cdiv(H, BLOCK_H))](
        out, gain_raw, forget_raw, cur, msg, cur.stride(0), msg.stride(0), H,
        BLOCK_H=BLOCK_H, num_warps=4, num_stages=1,
    )
    return out


__all__ = ["ctrl_rms_cat", "ctrl_gate_out"]
