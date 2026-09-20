#!/usr/bin/env python3
"""Benchmark for Qwen3.8-Flash-Next fused kernel optimizations.

Compares baseline vs fused kernel performance for:
1. GDN decode + HC combine fusion
2. QSA score + top-k fusion

Usage:
    python benchmarks/benchmark_qwen4_exp_fused.py [--batch-size 256] [--seq-len 1024]
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import torch


def benchmark_gdn_decode_baseline(
    batch_size: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    hc_count: int,
    hidden_size: int,
    warmup: int = 10,
    repeats: int = 100,
) -> Tuple[float, float]:
    """Benchmark baseline GDN decode + HC combine (separate kernels).
    
    Note: GDN output is [batch, num_v_heads * head_dim] which may differ from hidden_size.
    In the actual model, out_proj projects GDN output to hidden_size before HC combine.
    For this benchmark, we measure just the GDN kernel + HC combine overhead,
    using the GDN output dimension directly.
    """
    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update
    
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    # GDN output dimension (before out_proj)
    gdn_output_dim = num_v_heads * head_dim
    
    # Allocate tensors
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    a = torch.randn(batch_size, num_v_heads, dtype=dtype, device=device)
    b = torch.randn(batch_size, num_v_heads, dtype=dtype, device=device)
    q = torch.randn(1, batch_size, num_k_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, batch_size, num_k_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(1, batch_size, num_v_heads, head_dim, dtype=dtype, device=device)
    
    # State pool
    num_slots = batch_size
    state_source = torch.zeros(num_slots, num_v_heads, head_dim, head_dim, 
                               dtype=torch.float32, device=device)
    indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    
    # HC tensors - use gdn_output_dim as the per-stream dimension for this benchmark
    # (In real model, out_proj projects to hidden_size first)
    residual = torch.randn(batch_size, hc_count * gdn_output_dim, dtype=dtype, device=device)
    injection_logits = torch.randn(batch_size, hc_count, dtype=dtype, device=device)
    
    scale = head_dim ** -0.5
    
    # Warmup
    for _ in range(warmup):
        o = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, dt_bias=dt_bias,
            softplus_beta=1.0, softplus_threshold=20.0,
            q=q, k=k, v=v, b=b,
            initial_state_source=state_source,
            initial_state_indices=indices,
            scale=scale, use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        # Simulate HC combine (simplified)
        gdn_out = o[0].reshape(batch_size, -1)
        inject_scale = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
        combined = residual.float().unflatten(-1, (hc_count, gdn_output_dim))
        combined = combined + gdn_out.float().unsqueeze(-2) * inject_scale.unsqueeze(-1)
        combined = combined.flatten(-2).to(dtype)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(repeats):
        o = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, dt_bias=dt_bias,
            softplus_beta=1.0, softplus_threshold=20.0,
            q=q, k=k, v=v, b=b,
            initial_state_source=state_source,
            initial_state_indices=indices,
            scale=scale, use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
        gdn_out = o[0].reshape(batch_size, -1)
        inject_scale = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
        combined = residual.float().unflatten(-1, (hc_count, gdn_output_dim))
        combined = combined + gdn_out.float().unsqueeze(-2) * inject_scale.unsqueeze(-1)
        combined = combined.flatten(-2).to(dtype)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / repeats) * 1000
    throughput = batch_size * repeats / elapsed
    return avg_ms, throughput


def benchmark_gdn_decode_fused(
    batch_size: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    hc_count: int,
    hidden_size: int,
    warmup: int = 10,
    repeats: int = 100,
) -> Tuple[float, float]:
    """Benchmark fused GDN decode + HC combine kernel."""
    try:
        from freetoken.kernel.triton.gdn_hc_fused import gdn_decode_hc_combine
    except ImportError:
        print("Fused kernel not available, skipping")
        return float('inf'), 0.0
    
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    # GDN output dimension (before out_proj)
    gdn_output_dim = num_v_heads * head_dim
    
    # Allocate tensors (same as baseline)
    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    a = torch.randn(batch_size, num_v_heads, dtype=dtype, device=device)
    b = torch.randn(batch_size, num_v_heads, dtype=dtype, device=device)
    q = torch.randn(1, batch_size, num_k_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, batch_size, num_k_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(1, batch_size, num_v_heads, head_dim, dtype=dtype, device=device)
    
    num_slots = batch_size
    state_source = torch.zeros(num_slots, num_v_heads, head_dim, head_dim,
                               dtype=torch.float32, device=device)
    indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    
    # Use gdn_output_dim for consistency with baseline
    residual = torch.randn(batch_size, hc_count * gdn_output_dim, dtype=dtype, device=device)
    injection_logits = torch.randn(batch_size, hc_count, dtype=dtype, device=device)
    output = torch.empty_like(residual)
    
    scale = head_dim ** -0.5
    
    # Warmup
    for _ in range(warmup):
        gdn_decode_hc_combine(
            A_log=A_log, a=a, dt_bias=dt_bias,
            q=q, k=k, v=v, b=b,
            initial_state_source=state_source,
            initial_state_indices=indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
            residual=residual,
            injection_logits=injection_logits,
            hc_count=hc_count,
            output=output,
        )
    
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(repeats):
        gdn_decode_hc_combine(
            A_log=A_log, a=a, dt_bias=dt_bias,
            q=q, k=k, v=v, b=b,
            initial_state_source=state_source,
            initial_state_indices=indices,
            cu_seqlens=cu_seqlens,
            scale=scale,
            residual=residual,
            injection_logits=injection_logits,
            hc_count=hc_count,
            output=output,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / repeats) * 1000
    throughput = batch_size * repeats / elapsed
    return avg_ms, throughput


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.8-Flash-Next fused kernels")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for benchmark")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length (for context)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()
    
    # Qwen3.8-Flash-Next dimensions
    num_k_heads = 24  # linear_num_key_heads
    num_v_heads = 24  # linear_num_value_heads  
    head_dim = 128    # linear_key_head_dim == linear_value_head_dim
    hc_count = 4      # hc_count
    hidden_size = 3584  # hidden_size
    
    print(f"Benchmarking Qwen3.8-Flash-Next GDN Decode")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Heads: {num_k_heads}k / {num_v_heads}v, dim={head_dim}")
    print(f"  HC: {hc_count} streams, hidden={hidden_size}")
    print()
    
    # Baseline
    print("Running baseline (separate GDN + HC)...")
    baseline_ms, baseline_tps = benchmark_gdn_decode_baseline(
        args.batch_size, num_k_heads, num_v_heads, head_dim,
        hc_count, hidden_size, args.warmup, args.repeats,
    )
    print(f"  Baseline: {baseline_ms:.3f} ms/step, {baseline_tps:.0f} tokens/s")
    
    # Fused
    print("Running fused kernel...")
    fused_ms, fused_tps = benchmark_gdn_decode_fused(
        args.batch_size, num_k_heads, num_v_heads, head_dim,
        hc_count, hidden_size, args.warmup, args.repeats,
    )
    if fused_ms != float('inf'):
        print(f"  Fused:    {fused_ms:.3f} ms/step, {fused_tps:.0f} tokens/s")
        
        speedup = baseline_ms / fused_ms
        gdn_output_dim = num_v_heads * head_dim
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"HBM savings: ~{args.batch_size * gdn_output_dim * 2 * 36 / 1e6:.1f} MB/step (36 linear layers)")
    else:
        print("  Fused kernel not available")


if __name__ == "__main__":
    main()
