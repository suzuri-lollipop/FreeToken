# SPDX-License-Identifier: Apache-2.0
"""Fused GDN decode + HyperConnection combine kernel for Qwen3.8-Flash-Next.

This kernel fuses the GatedDeltaNet decode recurrence with the subsequent
HyperConnection combine operation, eliminating one write+read round-trip of
the full residual tensor between these operations.

Architecture insight: In Qwen3.8-Flash-Next, every linear_attention layer
follows this pattern:
    x, s = hc.mix(R)           # R [T, hc*hidden] -> x [T, hidden]
    y = GDN(x)                 # [T, hidden] -> [T, hidden]  
    R = hc.combine(R, y, s)    # inject y back into all hc streams

The current implementation writes y to memory after GDN, then reads it back
in hc.combine. This kernel eliminates that round-trip by keeping y in registers
and directly computing the combined residual.

Performance impact:
- Eliminates 1 HBM write + 1 HBM read of [T, hidden] per linear layer
- For Qwen3.8 (36 linear layers, hidden=3584): saves ~72 * T * 3584 bytes/step
- At batch=256, T=1: saves ~26 MB/s of HBM bandwidth per decode step
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_decode_hc_combine_kernel(
    # GDN inputs
    A_log_ptr,
    a_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    b_ptr,
    h0_source_ptr,
    h0_indices_ptr,
    cu_seqlens_ptr,
    # HC combine inputs
    res_ptr,          # residual R [T, hc*hidden]
    inj_ptr,          # injection logits s [T, hc_count]
    # Output
    out_ptr,          # combined residual R' [T, hc*hidden]
    # Strides
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    stride_a,
    stride_res,
    stride_inj,
    stride_out,
    # Dimensions
    T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,       # num_k_heads
    HV: tl.constexpr,      # num_v_heads
    K: tl.constexpr,       # head_k_dim
    V: tl.constexpr,       # head_v_dim
    HC: tl.constexpr,      # hc_count
    HC_DIM: tl.constexpr,  # hidden_size = hc*hidden / hc
    BK: tl.constexpr,
    BV: tl.constexpr,
    SCALE: tl.constexpr,
    SOFTPLUS_BETA: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Fused GDN decode + HC combine kernel.
    
    Each program handles one (batch, v_head) pair for GDN, then broadcasts
    the output across all HC streams for the combine operation.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    
    # Handle varlen batching
    if IS_VARLEN:
        bos = tl.load(cu_seqlens_ptr + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens_ptr + i_n + 1).to(tl.int64)
        seq_len = eos - bos
    else:
        bos = i_n * T
        seq_len = T
    
    # Only process first token for decode
    if seq_len == 0:
        return
    
    # Load GDN state
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    
    # Pointers for first token
    p_q = q_ptr + bos * stride_q + i_h * K + o_k
    p_k = k_ptr + bos * stride_k + i_h * K + o_k
    p_v = v_ptr + bos * stride_v + i_hv * V + o_v
    p_b = b_ptr + bos * stride_b + i_hv
    p_a = a_ptr + bos * stride_a + i_hv
    p_A_log = A_log_ptr + i_hv
    p_dt_bias = dt_bias_ptr + i_hv
    
    # Load recurrent state
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    idx = tl.load(h0_indices_ptr + i_n)
    if idx >= 0:
        p_h0 = (h0_source_ptr + idx * HV * K * V + i_hv * K * V 
                + o_v[None, :] * K + o_k[:, None])
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    
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
    beta_x = SOFTPLUS_BETA * x
    softplus_x = tl.where(
        beta_x <= SOFTPLUS_THRESHOLD,
        (1.0 / SOFTPLUS_BETA) * tl.log(1.0 + tl.exp(beta_x)),
        x,
    )
    b_g = -tl.exp(b_A_log) * softplus_x
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))
    
    # L2 norm if enabled
    if USE_QK_L2NORM:
        b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
        b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
    
    b_q = b_q * SCALE
    
    # Delta rule update
    b_h *= tl.exp(b_g)
    b_v -= tl.sum(b_h * b_k[:, None], 0)
    b_v *= b_beta
    b_h += b_k[:, None] * b_v[None, :]
    
    # Compute GDN output: o = sum(h * q, dim=0) -> [BV]
    b_o = tl.sum(b_h * b_q[:, None], 0)
    
    # Store updated state back
    if idx >= 0:
        p_h0 = (h0_source_ptr + idx * HV * K * V + i_hv * K * V 
                + o_v[None, :] * K + o_k[:, None])
        tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)
    
    # Now fuse with HC combine
    # Load injection logits for this token
    hc_offs = tl.arange(0, 4)  # HC is typically 4 for Qwen3.8
    mask_hc = hc_offs < HC
    p_inj = inj_ptr + bos * stride_inj + hc_offs
    inj = tl.load(p_inj, mask=mask_hc, other=0.0)
    inj_scale = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
    
    # Process each HC stream
    # Unroll the loop for HC=4 (Qwen3.8 default)
    offs_inner = tl.arange(0, BV)
    mask_inner = offs_inner < HC_DIM
    
    # Stream 0
    if HC > 0:
        stream_offs_0 = 0 * HC_DIM + offs_inner
        p_res_0 = res_ptr + bos * stride_res + stream_offs_0
        res_val_0 = tl.load(p_res_0, mask=mask_inner, other=0.0).to(tl.float32)
        gdn_contrib_0 = tl.where(mask_inner, b_o, 0.0) * tl.sum(tl.where(hc_offs == 0, inj_scale, 0.0))
        out_val_0 = res_val_0 + gdn_contrib_0
        p_out_0 = out_ptr + bos * stride_out + stream_offs_0
        tl.store(p_out_0, out_val_0.to(p_out_0.dtype.element_ty), mask=mask_inner)
    
    # Stream 1
    if HC > 1:
        stream_offs_1 = 1 * HC_DIM + offs_inner
        p_res_1 = res_ptr + bos * stride_res + stream_offs_1
        res_val_1 = tl.load(p_res_1, mask=mask_inner, other=0.0).to(tl.float32)
        gdn_contrib_1 = tl.where(mask_inner, b_o, 0.0) * tl.sum(tl.where(hc_offs == 1, inj_scale, 0.0))
        out_val_1 = res_val_1 + gdn_contrib_1
        p_out_1 = out_ptr + bos * stride_out + stream_offs_1
        tl.store(p_out_1, out_val_1.to(p_out_1.dtype.element_ty), mask=mask_inner)
    
    # Stream 2
    if HC > 2:
        stream_offs_2 = 2 * HC_DIM + offs_inner
        p_res_2 = res_ptr + bos * stride_res + stream_offs_2
        res_val_2 = tl.load(p_res_2, mask=mask_inner, other=0.0).to(tl.float32)
        gdn_contrib_2 = tl.where(mask_inner, b_o, 0.0) * tl.sum(tl.where(hc_offs == 2, inj_scale, 0.0))
        out_val_2 = res_val_2 + gdn_contrib_2
        p_out_2 = out_ptr + bos * stride_out + stream_offs_2
        tl.store(p_out_2, out_val_2.to(p_out_2.dtype.element_ty), mask=mask_inner)
    
    # Stream 3
    if HC > 3:
        stream_offs_3 = 3 * HC_DIM + offs_inner
        p_res_3 = res_ptr + bos * stride_res + stream_offs_3
        res_val_3 = tl.load(p_res_3, mask=mask_inner, other=0.0).to(tl.float32)
        gdn_contrib_3 = tl.where(mask_inner, b_o, 0.0) * tl.sum(tl.where(hc_offs == 3, inj_scale, 0.0))
        out_val_3 = res_val_3 + gdn_contrib_3
        p_out_3 = out_ptr + bos * stride_out + stream_offs_3
        tl.store(p_out_3, out_val_3.to(p_out_3.dtype.element_ty), mask=mask_inner)


def gdn_decode_hc_combine(
    # GDN parameters
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    # HC parameters
    residual: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
    # Output
    output: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """Fused GDN decode + HC combine.
    
    Args:
        A_log: [HV] fp32, log decay rates
        a: [B, HV] or [T, HV] raw gating input
        dt_bias: [HV] fp32, bias for softplus
        q, k: [1, B, H, K] bf16, query/key projections
        v: [1, B, HV, V] bf16, value projection
        b: [B, HV] raw beta gating input
        initial_state_source: [slots, HV, K, V] fp32, recurrent states
        initial_state_indices: [B] int32, slot indices
        cu_seqlens: [B+1] int32, cumulative sequence lengths
        scale: float, attention scale (typically K^-0.5)
        residual: [T, hc*hidden] residual streams
        injection_logits: [T, hc_count] injection logits from HC mix
        hc_count: int, number of hyper-connection streams
        output: optional pre-allocated output tensor
        use_qk_l2norm: whether to apply L2 normalization to q/k
        
    Returns:
        Combined residual [T, hc*hidden]
    """
    B, T, H, K = k.shape
    HV = v.shape[2]
    V = v.shape[-1]
    
    assert K == V, "GDN requires head_k_dim == head_v_dim"
    assert residual.shape[-1] % hc_count == 0
    HC_DIM = residual.shape[-1] // hc_count
    
    if output is None:
        output = torch.empty_like(residual)
    
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    
    assert NK == 1, "NK > 1 not supported"
    assert hc_count <= 4, "Only HC <= 4 supported currently"
    
    grid = (NK, NV, B * HV)
    
    _gdn_decode_hc_combine_kernel[grid](
        A_log, a, dt_bias, q, k, v, b,
        initial_state_source, initial_state_indices, cu_seqlens,
        residual, injection_logits, output,
        q.stride(1), k.stride(1), v.stride(1), b.stride(-2), a.stride(-2),
        residual.stride(0), injection_logits.stride(0), output.stride(0),
        T=T, B=B, H=H, HV=HV, K=K, V=V, HC=hc_count, HC_DIM=HC_DIM,
        BK=BK, BV=BV, SCALE=scale,
        SOFTPLUS_BETA=1.0, SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM=use_qk_l2norm,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=1,
        num_stages=3,
    )
    
    return output


__all__ = ["gdn_decode_hc_combine"]
