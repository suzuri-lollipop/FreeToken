"""Env-gated torch.profiler hook for the scheduler loop (FREETOKEN_PROFILE=<dir>).

Decomposes a window of decode steps into CPU/CUDA kernel time so the speculative-decoding
bottleneck (attention vs MoE vs draft vs TP comm) can be read off a real run. Diagnostics
only: with the env unset, ``from_env`` returns None and the loop carries a single ``is None``
check per iteration.

Run with CUDA graphs off (``--cuda-graph-max-bs 0``) so every kernel is visible instead of
hidden inside a graph replay. Each TP rank writes its own files (pid-suffixed); rank 0 is the
one to read. The profile is exported after WAIT+ACTIVE steps and serving then continues
unprofiled.
"""

from __future__ import annotations

import os
from pathlib import Path

# kernel-name -> bucket, first match wins (lowercased substring)
_BUCKETS = (
    ("tp_comm", ("nccl", "allreduce", "all_reduce", "all_gather", "reduce_scatter", "alltoall", "broadcast")),
    ("moe_fetch", ("fast_index_copy", "index_copy_multi", "copy_multi")),
    ("attention", ("attention", "attn", "flash", "fmha", "ring_attn", "qsa", "paged", "rope", "norm_rope")),
    ("moe_gemm", ("moe", "expert", "grouped_gemm", "w4a4", "nvfp4", "marlin", "gemv", "gemm", "cutlass", "s2_gemv", "cublas")),
    ("conv_gdn", ("conv", "gdn", "recurrent", "sigmoid_gating", "causal_conv", "chunk_", "gated_delta")),
    ("norm_act", ("norm", "silu", "gelu", "activation", "softmax", "elementwise", "vectorized", "reduce_kernel")),
    ("sample", ("sample", "topk", "top_k", "multinomial", "argmax", "sort", "cumsum", "selector")),
    ("copy", ("memcpy", "memset", "copy_", "cat", "fill", "index_select")),
)


def _bucket(name: str) -> str:
    low = name.lower()
    for bucket, keys in _BUCKETS:
        if any(k in low for k in keys):
            return bucket
    return "other"


class _LoopProfiler:
    def __init__(self, out_dir: str, wait: int, active: int, warmup: int = 2) -> None:
        import torch
        from torch.profiler import ProfilerActivity, profile, schedule

        self._torch = torch
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.remaining = wait + warmup + active
        self.done = False
        self.prof = profile(
            schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=1),
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
            profile_memory=False,
        )

    def start(self) -> None:
        self.prof.start()

    def step(self) -> None:
        if self.done:
            return
        self.prof.step()
        self.remaining -= 1
        if self.remaining <= 0:
            try:
                self._export()
            finally:
                self.prof.stop()
                self.done = True

    def _export(self) -> None:
        pid = os.getpid()
        ka = self.prof.key_averages()
        cuda_table = ka.table(sort_by="cuda_time_total", row_limit=60)
        cpu_table = ka.table(sort_by="cpu_time_total", row_limit=60)
        (self.out_dir / f"cuda-{pid}.txt").write_text(cuda_table)
        (self.out_dir / f"cpu-{pid}.txt").write_text(cpu_table)
        self.prof.export_chrome_trace(str(self.out_dir / f"trace-{pid}.json"))
        # bucketed CUDA-time summary (the bottleneck decomposition)
        totals: dict[str, float] = {}
        for ev in ka:
            t = getattr(ev, "device_time_total", 0.0) or getattr(ev, "cuda_time_total", 0.0)
            if t:
                totals[_bucket(ev.key)] = totals.get(_bucket(ev.key), 0.0) + t
        grand = sum(totals.values()) or 1.0
        lines = [f"{'bucket':12s} {'cuda_ms':>12s} {'share':>8s}"]
        for bucket, t in sorted(totals.items(), key=lambda kv: -kv[1]):
            lines.append(f"{bucket:12s} {t / 1e3:12.2f} {100 * t / grand:7.1f}%")
        summary = "\n".join(lines)
        (self.out_dir / f"buckets-{pid}.txt").write_text(summary + "\n")
        from freetoken.utils import init_logger

        init_logger(__name__).info_rank0(f"[profiler] exported to {self.out_dir} (pid {pid})\n{summary}")


def from_env() -> "_LoopProfiler | None":
    out = os.getenv("FREETOKEN_PROFILE", "").strip()
    if not out:
        return None
    wait = int(os.getenv("FREETOKEN_PROFILE_WAIT", "8") or 8)
    active = int(os.getenv("FREETOKEN_PROFILE_ACTIVE", "16") or 16)
    return _LoopProfiler(out, wait, active)
