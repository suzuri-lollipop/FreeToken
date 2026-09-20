# SPDX-License-Identifier: Apache-2.0
"""Optimized GDN decode kernel with fp16/bf16 recurrent state support.

This kernel reduces state memory by 50% compared to the fp32 baseline,
enabling larger batch sizes or longer contexts within the same memory budget.

Key optimizations:
1. fp16/bf16 state storage with fp32 accumulation in registers
2. Same algorithm as fused_sigmoid_gating_delta_rule_update_kernel
3. Type conversion only at load/store boundaries

Memory savings:
- Per-request state: 1.57 MB (fp32) -> 0.79 MB (bf16) = 50% reduction
- At batch=256: 402 MB -> 201 MB

Numerical stability:
- State is loaded as bf16 and immediately upcast to fp32 for computation
- All accumulation happens in fp32 registers
- Final state is downcast back to bf16 before storing
- The delta-rule update is numerically stable at reduced precision because
  the gating factor exp(g) keeps the state bounded
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def gdn_decode_bf16_state_kernel(
    A_log_ptr,
    a_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    b_ptr,
    o_ptr,
    h0_source_ptr,  # bf16 state [slots, HV, K, V]
    h0_indices_ptr,
    cu_seqlens_ptr,
    softplus_beta,
    softplus_threshold,
    scale,
    T,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    NP2_T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """GDN decode kernel with bf16 state support.
    
    Identical to fused_sigmoid_gating_delta_rule_update_kernel except:
    - State is loaded as bf16 and upcast to fp32
    - State is stored as bf16 after downcast from fp32
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int64)
        seq_len = eos - bos
    else:
        bos = i_n * T
        seq_len = T

    if seq_len == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q_ptr + bos * stride_q + i_h * K + o_k
    p_k = k_ptr + bos * stride_k + i_h * K + o_k
    p_v = v_ptr + bos * stride_v + i_hv * V + o_v
    p_b = b_ptr + bos * stride_b + i_hv
    p_o = o_ptr + ((i_k * seq_len + bos) * HV + i_hv) * V + o_v

    p_A_log = A_log_ptr + i_hv
    p_a = a_ptr + bos * stride_a + i_hv
    p_dt_bias = dt_bias_ptr + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # Load state in bf16 and upcast to fp32 for computation
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    idx = tl.load(h0_indices_ptr + i_n)
    if idx >= 0:
        # State layout: [slots, HV, K, V]
        # Element stride is the same regardless of dtype
        p_h0 = (h0_source_ptr + idx * HV * K * V + i_hv * K * V 
                + o_v[None, :] * K + o_k[:, None])
        # Load as bf16 and upcast to fp32
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.bfloat16).to(tl.float32)

    # Load inputs
    b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
    b_b = tl.load(p_b).to(tl.float32)
    b_A_log = tl.load(p_A_log).to(tl.float32)
    b_a = tl.load(p_a).to(tl.float32)
    b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

    # Compute gating: g = -exp(A_log) * softplus(a + dt_bias)
    x = b_a + b_dt_bias
    beta_x = softplus_beta * x
    softplus_x = tl.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
        x,
    )
    b_g = -tl.exp(b_A_log) * softplus_x
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))

    # L2 norm if enabled
    if USE_QK_L2NORM:
        b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
        b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

    b_q = b_q * scale

    # Delta rule update (all in fp32)
    b_h *= tl.exp(b_g)
    b_v -= tl.sum(b_h * b_k[:, None], 0)
    b_v *= b_beta
    b_h += b_k[:, None] * b_v[None, :]

    # Compute output: o = sum(h * q, dim=0)
    b_o = tl.sum(b_h * b_q[:, None], 0)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    # Store state back in bf16
    if idx >= 0:
        p_h0 = (h0_source_ptr + idx * HV * K * V + i_hv * K * V 
                + o_v[None, :] * K + o_k[:, None])
        tl.store(p_h0, b_h.to(tl.bfloat16), mask=mask_h)


def gdn_decode_bf16_state(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,  # bf16
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    """GDN decode with bf16 state support.
    
    Args:
        A_log: [HV] fp32, log decay rates
        a: [B, HV] raw gating input
        dt_bias: [HV] fp32, bias for softplus
        q, k: [1, B, H, K] query/key projections
        v: [1, B, HV, V] value projection
        b: [B, HV] raw beta gating input
        initial_state_source: [slots, HV, K, V] bf16 recurrent states
        initial_state_indices: [B] int32 slot indices
        scale: attention scale (default: K^-0.5)
        use_qk_l2norm_in_kernel: apply L2 normalization to q/k
        cu_seqlens: [B+1] cumulative sequence lengths for varlen
        softplus_beta: beta parameter for softplus
        softplus_threshold: threshold for numerical stability
        
    Returns:
        Output tensor [B, HV, V]
    """
    assert initial_state_source.dtype == torch.bfloat16, \
        f"State must be bf16, got {initial_state_source.dtype}"
    
    B, T, H, K = k.shape
    HV = v.shape[2]
    V = v.shape[-1]
    
    if scale is None:
        scale = K ** -0.5
    
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    
    assert NK == 1, "NK > 1 not supported"
    
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    o = q.new_empty(NK, B, HV, V)
    
    NP2_T = triton.next_power_of_2(T)
    IS_VARLEN = cu_seqlens is not None
    
    grid = (NK, NV, N * HV)
    
    gdn_decode_bf16_state_kernel[grid](
        A_log, a, dt_bias, q, k, v, b, o,
        initial_state_source, initial_state_indices, cu_seqlens,
        softplus_beta, softplus_threshold, scale,
        T, a.stride(-2), q.stride(1), k.stride(1), v.stride(1), b.stride(-2),
        NP2_T=NP2_T, B=B, H=H, HV=HV, K=K, V=V,
        BK=BK, BV=BV,
        USE_QK_L2NORM=use_qk_l2norm_in_kernel,
        IS_VARLEN=IS_VARLEN,
        num_warps=1,
        num_stages=3,
    )
    
    return o.squeeze(0)


__all__ = ["gdn_decode_bf16_state"]
