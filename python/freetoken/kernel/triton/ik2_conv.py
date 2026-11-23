"""Fused Inkling-2 short-conv decode step.

The decode conv ran as index_select + flip + two cats + mul + sum + index_copy +
casts (~10 kernels) for each of the 4 convs per visit, 48 visits per step. One
kernel reads the strided history slab, dots the causal window against the static
(+ dynamic head-shared) taps, shifts the history and appends the current input,
and applies the residual + out_scale -- fp32 compute as the reference. The verify
and prefill paths keep their torch implementations. CPU falls back to torch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_decode_kernel(
    OUT, X, STATES, SLOTS, WEIGHT, DYN, FRESH,
    C: tl.constexpr, K: tl.constexpr, out_scale,
    HAS_DYN: tl.constexpr, HAS_FRESH: tl.constexpr,
    s_slot, s_chan, BLOCK_C: tl.constexpr,
):
    t = tl.program_id(0)
    c = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = c < C
    slot = tl.load(SLOTS + t)
    x = tl.load(X + t * C + c, mask=cm, other=0.0).to(tl.float32)
    base = STATES + slot * s_slot + c * s_chan
    fresh = tl.load(FRESH + t) if HAS_FRESH else 0
    # lag 0 is the current token; lag L (1..K-1) reads history column K-1-L
    w0 = tl.load(WEIGHT + c * K, mask=cm, other=0.0)
    if HAS_DYN:
        w0 += tl.load(DYN + t * C * K + c * K, mask=cm, other=0.0)
    acc = x * w0
    for L in tl.static_range(1, K):
        h = tl.load(base + (K - 1 - L), mask=cm, other=0.0)
        if HAS_FRESH:
            h = tl.where(fresh != 0, 0.0, h)
        w = tl.load(WEIGHT + c * K + L, mask=cm, other=0.0)
        if HAS_DYN:
            w += tl.load(DYN + t * C * K + c * K + L, mask=cm, other=0.0)
        acc += h * w
    out = (acc + x) * out_scale
    tl.store(OUT + t * C + c, out.to(OUT.dtype.element_ty), mask=cm)
    # shift history left one column and append the current input
    for j in tl.static_range(0, K - 1):
        if j < K - 2:
            val = tl.load(base + (j + 1), mask=cm, other=0.0)
            if HAS_FRESH:
                val = tl.where(fresh != 0, 0.0, val)
        else:
            val = x
        tl.store(base + j, val, mask=cm)


def conv_decode_fused(x, states, slots, weight, dyn, fresh, out_scale):
    """``x [T, C]``, ``states [slots, C, K-1]`` fp32 (lag axis unit-stride), ``weight [C, K]`` fp32,
    ``dyn [T, C, K]`` fp32 or None, ``fresh [T]`` bool or None -> ``(conv(x) + x) * out_scale`` in x's dtype."""
    T, C = x.shape
    K = weight.shape[1]
    out = torch.empty(T, C, dtype=x.dtype, device=x.device)
    if T == 0:
        return out
    assert states.stride(2) == 1
    BLOCK_C = 256 if C >= 256 else triton.next_power_of_2(C)
    _conv_decode_kernel[(T, triton.cdiv(C, BLOCK_C))](
        out, x, states, slots, weight, dyn if dyn is not None else x,
        fresh if fresh is not None else slots,
        C, K, out_scale, dyn is not None, fresh is not None,
        states.stride(0), states.stride(1), BLOCK_C=BLOCK_C, num_warps=4, num_stages=1,
    )
    return out


__all__ = ["conv_decode_fused"]
