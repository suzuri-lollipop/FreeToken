"""Fused per-head norms for Inkling-2 attention (decode elementwise-tail reduction).

``l2_clip`` and ``unweighted_rms_norm`` run on every visit's q/k/v/r and on the
depth-mix streams; as plain torch each spawns ~5 tiny kernels (fp32 cast, pow/sum,
rsqrt or sqrt, scale/clamp, cast back). At bs1 decode that is hundreds of launches
and memory passes per step. One single-pass Triton kernel per op reads the row once
and writes the result, reducing in fp32 exactly as the reference. Pattern follows
kernel/triton/norm.py (BLOCK == next_power_of_2(H), strided per-head path, PDL on
the contiguous decode-replay path). CPU tensors fall back to the torch reference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

from freetoken.utils.arch import is_rocm, is_sm90_supported

_HEUR = {"BLOCK": lambda a: triton.next_power_of_2(a["H"])}


def _num_warps(M: int) -> int:
    return 4 if M >= 1024 else 8


def _leading(t):
    # (A, B, strideA, strideB) for the 2D program grid; last dim is the norm axis.
    if t.ndim == 2:
        return t.shape[0], 1, t.stride(0), 0
    return t.shape[0], t.shape[1], t.stride(0), t.stride(1)


@triton.heuristics(_HEUR)
@triton.jit
def _rms_noweight_kernel(
    OUT, X,
    eps: tl.constexpr, H: tl.constexpr,
    sxa, sxb, soa, sob,
    CONTIG: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0)
    b = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    if CONTIG:
        xoff = a.to(tl.int64) * H
        ooff = xoff
    else:
        xoff = a.to(tl.int64) * sxa + b * sxb
        ooff = a.to(tl.int64) * soa + b * sob
    if ENABLE_PDL:
        gdc_wait()
    x = tl.load(X + xoff + cols, mask=mask, other=0.0).to(tl.float32)
    if ENABLE_PDL:
        gdc_launch_dependents()
    inv = tl.math.rsqrt(tl.sum(x * x, axis=0) / H + eps)
    tl.store(OUT + ooff + cols, (x * inv).to(OUT.dtype.element_ty), mask=mask)


@triton.heuristics(_HEUR)
@triton.jit
def _l2_clip_kernel(
    OUT, X, SCALE,
    threshold: tl.constexpr, H: tl.constexpr,
    sxa, sxb, soa, sob, ss,
    HAS_SCALE: tl.constexpr,
    CONTIG: tl.constexpr,
    ENABLE_PDL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    a = tl.program_id(0)
    b = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    mask = cols < H
    if CONTIG:
        xoff = a.to(tl.int64) * H
        ooff = xoff
    else:
        xoff = a.to(tl.int64) * sxa + b * sxb
        ooff = a.to(tl.int64) * soa + b * sob
    if ENABLE_PDL:
        gdc_wait()
    x = tl.load(X + xoff + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_SCALE:
        # scale is [B, H] (per-head); row b is this head's scale
        s = tl.load(SCALE + b * ss + cols, mask=mask, other=0.0).to(tl.float32)
        x = x * s
    if ENABLE_PDL:
        gdc_launch_dependents()
    norm = tl.sqrt(tl.sum(x * x, axis=0))
    factor = tl.minimum(threshold / (norm + 1e-6), 1.0)
    tl.store(OUT + ooff + cols, (x * factor).to(OUT.dtype.element_ty), mask=mask)


def rms_noweight(x: torch.Tensor, eps: float = 1e-6, out: torch.Tensor | None = None) -> torch.Tensor:
    """``x * rsqrt(mean(x^2, -1) + eps)`` over the last axis, fp32 reduction."""
    if not x.is_cuda:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).to(x.dtype)
    H = x.shape[-1]
    if x.ndim < 2 or x.ndim > 3:
        return rms_noweight(x.reshape(-1, H), eps).reshape(x.shape)
    assert x.stride(-1) == 1
    if out is None:
        out = torch.empty_like(x)
    A, B, sxa, sxb = _leading(x)
    _, _, soa, sob = _leading(out)
    contig = x.ndim == 2 and x.is_contiguous() and out.is_contiguous()
    pdl = contig and is_sm90_supported()
    _rms_noweight_kernel[(A, B)](
        out, x, eps, H, sxa, sxb, soa, sob,
        CONTIG=contig, ENABLE_PDL=pdl,
        **({} if is_rocm() else {"launch_pdl": pdl}),
        num_warps=_num_warps(A * B), num_stages=1,
    )
    return out


def l2_clip(x: torch.Tensor, threshold: float, scale: torch.Tensor | None = None,
            out: torch.Tensor | None = None) -> torch.Tensor:
    """Per-head L2 cap: ``xf = x.float() [* scale]; xf * min(1, threshold/(||xf||+1e-6))``."""
    if not x.is_cuda:
        xf = x.float()
        if scale is not None:
            xf = xf * scale.float()
        norm = torch.linalg.vector_norm(xf, dim=-1, keepdim=True)
        return (xf * (threshold / (norm + 1e-6)).clamp(max=1.0)).to(x.dtype)
    H = x.shape[-1]
    if x.ndim < 2 or x.ndim > 3:
        return l2_clip(x.reshape(-1, H), threshold, scale).reshape(x.shape)
    assert x.stride(-1) == 1
    if out is None:
        out = torch.empty_like(x)
    A, B, sxa, sxb = _leading(x)
    _, _, soa, sob = _leading(out)
    contig = x.ndim == 2 and x.is_contiguous() and out.is_contiguous()
    pdl = contig and is_sm90_supported()
    has_scale = scale is not None
    ss = scale.stride(0) if has_scale and scale.ndim == 2 else H
    sc = scale if has_scale else x  # unused pointer when HAS_SCALE=False
    _l2_clip_kernel[(A, B)](
        out, x, sc, threshold, H, sxa, sxb, soa, sob, ss,
        HAS_SCALE=has_scale, CONTIG=contig, ENABLE_PDL=pdl,
        **({} if is_rocm() else {"launch_pdl": pdl}),
        num_warps=_num_warps(A * B), num_stages=1,
    )
    return out


@triton.jit
def _rel_table_kernel(
    OUT, R, BIAS,
    D: tl.constexpr, E: tl.constexpr,
    D_PAD: tl.constexpr, BLOCK_E: tl.constexpr,
):
    row = tl.program_id(0)
    d = tl.arange(0, D_PAD)
    dm = d < D
    r = tl.load(R + row * D + d, mask=dm, other=0.0).to(tl.float32)
    e = tl.arange(0, BLOCK_E)
    for e0 in range(0, E, BLOCK_E):
        ee = e0 + e
        em = ee < E
        bias = tl.load(BIAS + d[:, None] * E + ee[None, :], mask=dm[:, None] & em[None, :], other=0.0).to(tl.float32)
        acc = tl.sum(r[:, None] * bias, axis=0)
        # the reference runs a bf16 matmul (fp32 accumulate, bf16 output) then casts to fp32
        acc = acc.to(tl.bfloat16).to(tl.float32)
        tl.store(OUT + row * (E + 1) + ee, acc, mask=em)
    tl.store(OUT + row * (E + 1) + E, 0.0)  # the padded zero bias column


def rel_table_fused(r: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """``r [T, nh, d_rel]`` bf16, ``bias [d_rel, extent]`` bf16 -> ``[T, nh, extent+1]`` fp32.

    Fuses the bf16 rel-bias matmul, the zero pad column and the fp32 cast (the reference's
    ``F.pad(r @ bias, (0, 1)).float().contiguous()``) into one kernel.
    """
    T, nh, D = r.shape
    E = bias.shape[1]
    out = torch.empty(T * nh, E + 1, dtype=torch.float32, device=r.device)
    if T * nh == 0:
        return out.view(T, nh, E + 1)
    r2 = r.reshape(T * nh, D)
    BLOCK_E = 256 if E >= 256 else triton.next_power_of_2(E)
    _rel_table_kernel[(T * nh,)](
        out, r2, bias, D, E, D_PAD=triton.next_power_of_2(D), BLOCK_E=BLOCK_E,
        num_warps=4, num_stages=1,
    )
    return out.view(T, nh, E + 1)


__all__ = ["l2_clip", "rms_noweight", "rel_table_fused"]
