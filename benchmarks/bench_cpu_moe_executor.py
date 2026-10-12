"""Per-layer CPU MoE executor latency at decode batch sizes.

Isolates the executor pool (no server, no offload cache): synthetic pinned bf16
banks at GLM MoE scale, one CpuMoeExecutor.decode per timed iteration. Reports
per-layer wall time, per-expert latency and the pool's effective bandwidth, so a
machine's ISA ceiling (avx2 vs avx512/vnni) shows up directly in the GB/s number.

Usage (from the repo root, CUDA GPU visible for the D2H/H2D round trip):
    uv run python benchmarks/bench_cpu_moe_executor.py [--experts 8] [--bs 1]
"""

import argparse
import time
from types import SimpleNamespace

import torch

from freetoken.kernel.pinned import alloc_pinned_tensor


def make_cache(L: int, E: int, H: int, I: int, scale: float) -> SimpleNamespace:
    gate_up = alloc_pinned_tensor(L * E, 2 * I, H, dtype=torch.bfloat16)
    down = alloc_pinned_tensor(L * E, H, I, dtype=torch.bfloat16)
    gate_up.copy_(torch.randn(L * E, 2 * I, H) * scale)
    down.copy_(torch.randn(L * E, H, I) * scale)
    return SimpleNamespace(
        quant_format="bf16",
        bank_sources={"gate_up": list(gate_up.split(E)), "down": list(down.split(E))},
        num_layers=L,
        num_experts=E,
        decode_target="cpu",
        cpu_executor=None,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hidden", type=int, default=3072, help="H")
    p.add_argument("--inter", type=int, default=3584, help="I")
    p.add_argument("--num-experts", type=int, default=32, help="experts per layer")
    p.add_argument("--top-k", type=int, default=8, help="routed experts per token")
    p.add_argument("--bs", type=int, default=1, help="decode batch size")
    p.add_argument("--iters", type=int, default=60, help="timed decode iterations")
    args = p.parse_args()

    from freetoken.moe.cpu_executor import CpuMoeExecutor

    torch.manual_seed(0)
    cache = make_cache(1, args.num_experts, args.hidden, args.inter, 0.02)
    ex = CpuMoeExecutor(
        cache,
        top_k=args.top_k,
        activation="silu",
        apply_router_weight_on_input=False,
        num_threads=0,
        max_tokens=args.bs,
        device=torch.device("cuda"),
    )
    hidden = torch.randn(args.bs, args.hidden, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(
        0, args.num_experts, (args.bs, args.top_k), device="cuda", dtype=torch.int32
    )
    w = torch.rand(args.bs, args.top_k, device="cuda", dtype=torch.float32)

    for _ in range(5):
        ex.decode(0, hidden, w, ids)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        ex.decode(0, hidden, w, ids)
    torch.cuda.synchronize()
    per_ms = (time.perf_counter() - t0) / args.iters * 1e3

    mb = args.bs * args.top_k * (2 * args.inter * args.hidden + args.hidden * args.inter) * 2 / 1e6
    print(f"layer decode (bs={args.bs}, {args.top_k} experts): {per_ms:.2f} ms")
    print(f"per-expert: {per_ms / (args.bs * args.top_k) * 1e3:.0f} us")
    print(f"pool bandwidth: {mb / 1e3 / (per_ms / 1e3):.1f} GB/s over {ex.num_threads} threads ({ex.isa})")


if __name__ == "__main__":
    main()
