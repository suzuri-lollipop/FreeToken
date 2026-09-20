#!/usr/bin/env python3
"""Benchmark Qwen3.8-Flash-Next with fp8 KV cache, TP=2, context=131072.

Usage:
    # With actual model (requires download):
    python benchmarks/benchmark_qwen38_fp8_tp2.py --model-path NVIDIA/Qwen3.8-flash-next-nvfp4

    # With dummy weights (for memory/performance testing):
    python benchmarks/benchmark_qwen38_fp8_tp2.py --dummy-weight

Options:
    --context-len     Context length (default: 131072)
    --memory-ratio    VRAM usage limit (default: 0.85)
    --tp-size         Tensor parallel size (default: 2)
    --kv-cache-dtype  KV cache dtype: fp8 or bf16 (default: fp8)
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.8-Flash-Next with fp8 KV")
    parser.add_argument("--model-path", type=str, default="NVIDIA/Qwen3.8-flash-next-nvfp4",
                        help="Model path or HF repo ID")
    parser.add_argument("--dummy-weight", action="store_true",
                        help="Use dummy weights for testing")
    parser.add_argument("--context-len", type=int, default=131072,
                        help="Context length")
    parser.add_argument("--memory-ratio", type=float, default=0.85,
                        help="VRAM usage ratio limit")
    parser.add_argument("--tp-size", type=int, default=2,
                        help="Tensor parallel size")
    parser.add_argument("--kv-cache-dtype", type=str, default="fp8",
                        choices=["fp8", "bf16"],
                        help="KV cache dtype")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for benchmark")
    parser.add_argument("--num-tokens", type=int, default=128,
                        help="Number of tokens to generate")
    args = parser.parse_args()

    print("=" * 70)
    print("Qwen3.8-Flash-Next Benchmark Configuration")
    print("=" * 70)
    print(f"  Model: {args.model_path}")
    print(f"  Context length: {args.context_len:,}")
    print(f"  TP size: {args.tp_size}")
    print(f"  KV cache dtype: {args.kv_cache_dtype}")
    print(f"  Memory ratio: {args.memory_ratio:.0%}")
    print(f"  Batch size: {args.batch_size}")
    print()

    # Memory estimation
    print("-" * 70)
    print("Memory Estimation (per GPU)")
    print("-" * 70)

    # Qwen3.8-Flash-Next specs
    num_layers = 48
    full_attn_layers = 12
    linear_attn_layers = 36
    num_kv_heads = 2
    head_dim = 256
    compress_ratio = 4
    index_head_dim = 128

    kv_heads_per_gpu = num_kv_heads // args.tp_size
    bytes_per_elem = 1 if args.kv_cache_dtype == "fp8" else 2

    # QSA KV cache
    qsa_kv = full_attn_layers * args.context_len * kv_heads_per_gpu * head_dim * 2 * bytes_per_elem
    
    # QSA index slab (replicated)
    slab_entries = args.context_len // compress_ratio
    index_slab = full_attn_layers * slab_entries * index_head_dim * bytes_per_elem
    
    # GDN state (fp32, per request)
    num_v_heads = 24
    v_heads_per_gpu = num_v_heads // args.tp_size
    gdn_state_per_req = linear_attn_layers * v_heads_per_gpu * 128 * 128 * 4

    total_kv_gb = (qsa_kv + index_slab) / 1e9
    
    print(f"  QSA KV cache: {qsa_kv / 1e9:.2f} GB")
    print(f"  QSA index slab: {index_slab / 1e9:.2f} GB")
    print(f"  Total KV/Index: {total_kv_gb:.2f} GB")
    print(f"  GDN state/request: {gdn_state_per_req / 1e6:.2f} MB")
    print()

    # Check VRAM
    import torch
    if torch.cuda.is_available():
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        limit_gb = gpu_mem_gb * args.memory_ratio
        print(f"  GPU VRAM: {gpu_mem_gb:.1f} GB")
        print(f"  Limit ({args.memory_ratio:.0%}): {limit_gb:.1f} GB")
        print(f"  KV fits: {'YES' if total_kv_gb < limit_gb else 'NO'}")
        max_batch = int((limit_gb - total_kv_gb) * 1e9 / gdn_state_per_req)
        print(f"  Max batch (GDN state limited): ~{max_batch}")
    print()

    # Build ft serve command
    cmd = [
        "ft", "serve",
        "--model-path", args.model_path,
        "--tensor-parallel-size", str(args.tp_size),
        "--memory-ratio", str(args.memory_ratio),
        "--kv-cache-dtype", args.kv_cache_dtype,
        "--max-seq-len-override", str(args.context_len),
        "--gpu", "0,1",
    ]
    
    if args.dummy_weight:
        cmd.append("--dummy-weight")

    print("-" * 70)
    print("Command to run server:")
    print("-" * 70)
    print(" ".join(cmd))
    print()
    print("After server starts, use 'ft bench' or curl to benchmark.")
    print()
    
    # Ask if user wants to start the server
    if not args.dummy_weight:
        print("Note: This requires downloading the model from HuggingFace.")
        print("Add --dummy-weight to test without downloading.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
