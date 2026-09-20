#!/usr/bin/env python3
"""Benchmark GDN decode kernel with TP=1 vs TP=2."""

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def benchmark_tp2(rank, world_size, results_dict):
    """Run GDN decode benchmark on one rank of a TP group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    from freetoken.distributed import set_tp_info
    set_tp_info(rank=rank, size=world_size)

    device = torch.device(f'cuda:{rank}')
    dtype = torch.bfloat16

    # Qwen3.8-Flash-Next configuration
    num_k_heads_full = 24
    num_v_heads_full = 24
    head_dim = 128
    batch_size = 64

    # TP sharding: each GPU gets heads / world_size
    num_k_heads = num_k_heads_full // world_size
    num_v_heads = num_v_heads_full // world_size

    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    a = torch.randn(batch_size, num_v_heads, dtype=dtype, device=device)
    b = torch.randn(batch_size, num_v_heads, dtype=dtype, device=device)
    q = torch.randn(1, batch_size, num_k_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, batch_size, num_k_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(1, batch_size, num_v_heads, head_dim, dtype=dtype, device=device)
    state_fp32 = torch.zeros(batch_size, num_v_heads, head_dim, head_dim,
                             dtype=torch.float32, device=device)
    indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    scale = head_dim ** -0.5

    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

    # Warmup
    for _ in range(10):
        o = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, dt_bias=dt_bias, softplus_beta=1.0, softplus_threshold=20.0,
            q=q, k=k, v=v, b=b, initial_state_source=state_fp32,
            initial_state_indices=indices, scale=scale, use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
    torch.cuda.synchronize()
    dist.barrier()

    repeats = 100
    start = time.perf_counter()
    for _ in range(repeats):
        o = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, dt_bias=dt_bias, softplus_beta=1.0, softplus_threshold=20.0,
            q=q, k=k, v=v, b=b, initial_state_source=state_fp32,
            initial_state_indices=indices, scale=scale, use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
    torch.cuda.synchronize()
    dist.barrier()

    elapsed = time.perf_counter() - start
    ms = elapsed / repeats * 1000
    tps = batch_size * repeats / elapsed
    state_mb = batch_size * num_v_heads * head_dim * head_dim * 4 / 1e6

    if rank == 0:
        results_dict['tp2_ms'] = ms
        results_dict['tp2_tps'] = tps
        results_dict['heads_per_gpu'] = (num_k_heads, num_v_heads)
        results_dict['state_per_gpu_mb'] = state_mb

    dist.destroy_process_group()


def main():
    print('=' * 70)
    print('GDN Decode Benchmark: TP=1 vs TP=2')
    print('=' * 70)
    print()

    # --- TP=1 baseline ---
    print('--- TP=1 (Single GPU) ---')
    torch.cuda.set_device(0)
    from freetoken.distributed import set_tp_info, get_tp_info

    # Reset TP info if previously set
    import freetoken.distributed.info as tp_mod
    tp_mod._TP_INFO = None
    set_tp_info(rank=0, size=1)

    device = torch.device('cuda:0')
    dtype = torch.bfloat16
    num_k_heads = 24
    num_v_heads = 24
    head_dim = 128
    batch_size = 64

    A_log = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    dt_bias = torch.randn(num_v_heads, dtype=torch.float32, device=device)
    a = torch.randn(batch_size, num_v_heads, dtype=dtype, device=device)
    b = torch.randn(batch_size, num_v_heads, dtype=dtype, device=device)
    q = torch.randn(1, batch_size, num_k_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(1, batch_size, num_k_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(1, batch_size, num_v_heads, head_dim, dtype=dtype, device=device)
    state_fp32 = torch.zeros(batch_size, num_v_heads, head_dim, head_dim,
                             dtype=torch.float32, device=device)
    indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    scale = head_dim ** -0.5

    from freetoken.kernel.fla import fused_sigmoid_gating_delta_rule_update

    for _ in range(10):
        o = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, dt_bias=dt_bias, softplus_beta=1.0, softplus_threshold=20.0,
            q=q, k=k, v=v, b=b, initial_state_source=state_fp32,
            initial_state_indices=indices, scale=scale, use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
    torch.cuda.synchronize()

    repeats = 100
    start = time.perf_counter()
    for _ in range(repeats):
        o = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, dt_bias=dt_bias, softplus_beta=1.0, softplus_threshold=20.0,
            q=q, k=k, v=v, b=b, initial_state_source=state_fp32,
            initial_state_indices=indices, scale=scale, use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
        )
    torch.cuda.synchronize()
    tp1_elapsed = time.perf_counter() - start
    tp1_ms = tp1_elapsed / repeats * 1000
    tp1_tps = batch_size * repeats / tp1_elapsed
    tp1_state_mb = batch_size * num_v_heads * head_dim * head_dim * 4 / 1e6

    print(f'TP=1: {tp1_ms:.3f} ms | {tp1_tps:,.0f} tok/s')
    print(f'  Heads: {num_k_heads}k / {num_v_heads}v')
    print(f'  State memory: {tp1_state_mb:.1f} MB')
    print()

    # --- TP=2 ---
    print('--- TP=2 (2 GPUs) ---')
    tp_mod._TP_INFO = None  # Reset for TP=2

    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(benchmark_tp2, args=(2, results), nprocs=2, join=True)

    tp2_ms = results['tp2_ms']
    tp2_tps = results['tp2_tps']
    heads_per_gpu = results['heads_per_gpu']
    state_per_gpu = results['state_per_gpu_mb']

    print(f'TP=2: {tp2_ms:.3f} ms | {tp2_tps:,.0f} tok/s')
    print(f'  Heads per GPU: {heads_per_gpu[0]}k / {heads_per_gpu[1]}v')
    print(f'  State memory per GPU: {state_per_gpu:.1f} MB')
    print(f'  Total state memory: {state_per_gpu * 2:.1f} MB')
    print()

    # --- Comparison ---
    print('--- Comparison ---')
    print(f'{"":>8} | {"Latency":>10} | {"Throughput":>12} | {"Heads":>10} | {"State":>10}')
    print('-' * 60)
    print(f'{"TP=1":>8} | {tp1_ms:>8.3f} ms | {tp1_tps:>10,.0f} | {num_k_heads:>10} | {tp1_state_mb:>8.1f} MB')
    print(f'{"TP=2":>8} | {tp2_ms:>8.3f} ms | {tp2_tps:>10,.0f} | {heads_per_gpu[0]:>10}/GPU | {state_per_gpu*2:>8.1f} MB')
    print()

    speedup_latency = tp1_ms / tp2_ms
    speedup_throughput = tp2_tps / tp1_tps
    print(f'Latency speedup:   {speedup_latency:.2f}x')
    print(f'Throughput ratio:  {speedup_throughput:.2f}x')
    print(f'State memory:      {state_per_gpu*2/tp1_state_mb:.2f}x ({state_per_gpu*2:.1f} vs {tp1_state_mb:.1f} MB)')


if __name__ == '__main__':
    main()
