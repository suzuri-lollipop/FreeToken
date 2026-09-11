"""Quantized KV-cache store: model-dtype activations into an e4m3 pool row.

Mirror of the JIT ``store_cache`` scatter (row ``i`` of ``k``/``v`` goes to slot
``indices[i]``) with a static per-tensor quantize in front of the write, so a
quantized pool keeps the exact addressing semantics the schedulers, radix caches and
CUDA-graph replay rely on. Descaling happens on the read side, in the attention
kernels (``kernel/triton/attention.py``) or the backend libraries' scale arguments.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Largest finite e4m3 magnitude; the store clamps to it because a cast of an
# out-of-range value yields NaN, and one NaN row poisons every later read of the page.
_E4M3_MAX = tl.constexpr(448.0)


@triton.jit
def _store_kv_e4m3_kernel(
    k_ptr,
    v_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    stride_kt,
    stride_vt,
    stride_kc,
    stride_vc,
    k_scale,
    v_scale,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(indices_ptr + row)
    for start in tl.range(0, N, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N
        k = tl.load(k_ptr + row * stride_kt + offs, mask=mask, other=0.0).to(tl.float32)
        v = tl.load(v_ptr + row * stride_vt + offs, mask=mask, other=0.0).to(tl.float32)
        k = tl.minimum(tl.maximum(k / k_scale, -_E4M3_MAX), _E4M3_MAX)
        v = tl.minimum(tl.maximum(v / v_scale, -_E4M3_MAX), _E4M3_MAX)
        tl.store(
            k_cache_ptr + slot * stride_kc + offs,
            k.to(k_cache_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            v_cache_ptr + slot * stride_vc + offs,
            v.to(v_cache_ptr.dtype.element_ty),
            mask=mask,
        )


def store_kv_e4m3(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: float,
    v_scale: float,
) -> None:
    """Quantize ``k``/``v`` and scatter them into the e4m3 cache views at ``indices`` --
    the same call shape as :func:`cache.store_cache`. The caches are the pool's
    ``[num_slots, num_kv_heads, head_dim]`` views; a row is one slot's whole head set."""
    num_tokens = indices.numel()
    if num_tokens == 0:
        return
    n = k_cache.numel() // k_cache.shape[0]
    assert n == v_cache.numel() // v_cache.shape[0], "K and V cache rows differ"
    BLOCK_N = min(triton.next_power_of_2(n), 1024)
    _store_kv_e4m3_kernel[(num_tokens,)](
        k,
        v,
        k_cache,
        v_cache,
        indices,
        k.stride(0),
        v.stride(0),
        k_cache.stride(0),
        v_cache.stride(0),
        k_scale,
        v_scale,
        n,
        BLOCK_N=BLOCK_N,
    )
