"""Env-gated MoE instrumentation (FREETOKEN_MOE_STATS=1). Diagnostics only.

Adds:
- per-layer prefill probes: unique experts touched, resident hits, streamed miss
  rows, and CUDA-event split of (copy stall) vs (GEMM) time;
- per-step decode accumulation of the on-device miss/fetch counters plus a
  fixed-shape duplicate-route histogram (batch dedup potential);
- host-side scheduler-loop phase timings (pure perf_counter, never a device
  sync: the decode replay parks at the PLE lookup WAIT until the post-drain
  fill signals, so any sync instrumentation here would deadlock the loop);
- periodic dumps from the overlap loop's post-fill safe point.

The only ops injected into the decode hot path are fixed-shape (CUDA-graph
safe) and gated behind the env flag; production runs carry a ``probe() is
None`` check per call site.
"""

from __future__ import annotations

import os
import time

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

ENABLED = os.getenv("FREETOKEN_MOE_STATS", "0").strip() not in ("", "0", "false", "no")
# torch.profiler capture of N decode batches after W batches (diag only; REQUIRES
# PLE_FILL_AT_EXIT=1 -- the engine refuses to profile without it, see forward_batch)
PROFILE_STEPS = int(os.getenv("FREETOKEN_PROFILE_STEPS", "0") or 0)
PROFILE_SKIP = int(os.getenv("FREETOKEN_PROFILE_SKIP", "100") or 0)
PROFILE_SHAPES = os.getenv("FREETOKEN_PROFILE_SHAPES", "0").strip() not in ("", "0", "false", "no")

# how many prefill chunks to probe with events/unique counts before going quiet
_PREFILL_PROBE_LIMIT = int(os.getenv("FREETOKEN_MOE_STATS_PREFILLS", "12"))


class MoEProbe:
    def __init__(self) -> None:
        self.cache = None
        self._prefill_events: list = []
        self._prefill_rows: dict[str, list] = {
            "unique": [], "hits": [], "miss": [], "stall_ms": [], "gemm_ms": [],
        }
        self._prefills_probed = 0
        self._prefill_t0 = 0.0
        # decode accumulators (device, fixed shape; graph safe)
        self._dec = None
        self._dec_steps = 0
        self._host: dict[str, list] = {}
        self._host_slow: list = []
        self._decode_batches = 0
        self._profile_done = False
        self._last_dump = time.monotonic()

    # ------------------------------------------------------------ lifecycle

    def attach(self, cache) -> None:
        self.cache = cache
        dev = cache.device
        # [misses, fetched, active_calls, dup_routes, total_routes, layer_calls]
        self._dec = torch.zeros(8, dtype=torch.int64, device=dev)

    # -------------------------------------------------------------- prefill

    def prefill_layer_begin(self, layer_id: int) -> None:
        if not ENABLED or self.cache is None:
            return
        if self._prefills_probed >= _PREFILL_PROBE_LIMIT:
            return
        if layer_id == 0:
            self._prefill_events = []
            self._prefill_t0 = time.perf_counter()
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._prefill_events.append(("start", layer_id, ev))

    def prefill_layer_waited(self, layer_id: int) -> None:
        if not ENABLED or self._prefills_probed >= _PREFILL_PROBE_LIMIT:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._prefill_events.append(("waited", layer_id, ev))

    def prefill_layer_done(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        if not ENABLED or self.cache is None:
            return
        if self._prefills_probed >= _PREFILL_PROBE_LIMIT:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._prefill_events.append(("done", layer_id, ev))
        cache = self.cache
        E = cache.num_experts
        # routing concentration + residency of the touched set (host sync: prefill
        # is seconds long, one .cpu() per layer is noise)
        u = topk_ids.reshape(-1).unique()
        snap = cache._prefill_snapshot_np[layer_id] if cache._prefill_snapshot_np is not None else None
        un = u.cpu().numpy()
        if snap is not None:
            hits = int((snap[un] >= 2 * E).sum())
        else:
            hits = -1
        self._prefill_rows["unique"].append(len(un))
        self._prefill_rows["hits"].append(hits)
        self._prefill_rows["miss"].append(len(un) - hits)
        if layer_id == cache.num_layers - 1:
            torch.cuda.synchronize(cache.device)
            evs = {}
            for kind, lid, e in self._prefill_events:
                evs.setdefault(lid, {})[kind] = e
            stall = gemm = 0.0
            for lid, d in sorted(evs.items()):
                if {"start", "waited", "done"} <= set(d):
                    stall += d["start"].elapsed_time(d["waited"])
                    gemm += d["waited"].elapsed_time(d["done"])
            self._prefill_rows["stall_ms"].append(stall)
            self._prefill_rows["gemm_ms"].append(gemm)
            self._prefills_probed += 1
            wall = (time.perf_counter() - self._prefill_t0) * 1e3
            n = len(self._prefill_rows["unique"]) // self._prefills_probed
            uq = self._prefill_rows["unique"][-n:]
            ht = self._prefill_rows["hits"][-n:]
            ms = self._prefill_rows["miss"][-n:]
            logger.info(
                f"[moestat] prefill #{self._prefills_probed}: wall={wall:.0f}ms "
                f"stall={stall:.0f}ms gemm={gemm:.0f}ms | per-layer avg "
                f"unique={sum(uq)/n:.0f} hits={sum(ht)/n:.0f} miss={sum(ms)/n:.0f} "
                f"(touched {sum(uq)/n/512*100:.0f}% of 512)"
            )

    # --------------------------------------------------------------- decode

    def should_profile(self) -> bool:
        if not (ENABLED and PROFILE_STEPS) or self._profile_done:
            return False
        self._decode_batches += 1
        if self._decode_batches < PROFILE_SKIP:
            return False
        if self._decode_batches >= PROFILE_SKIP + PROFILE_STEPS:
            self._profile_done = True
            return False
        return True

    def profile_dump(self, prof) -> None:
        table = (
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="self_device_time_total", row_limit=40
            )
            if PROFILE_SHAPES
            else prof.key_averages().table(sort_by="self_device_time_total", row_limit=30)
        )
        logger.info("[moestat] decode profile (self CUDA time):\n" + table)

    def host_phase(self, name: str, dt: float) -> None:
        """Accumulate host-side loop phase timings (pure perf_counter, no syncs)."""
        acc = self._host.setdefault(name, [0.0, 0])
        acc[0] += dt
        acc[1] += 1
        if dt > 0.008:
            slow = self._host_slow
            slow.append((dt, name))
            if len(slow) > 400:
                del slow[:200]

    def host_dump(self) -> str:
        parts = []
        for name, (t, n) in self._host.items():
            parts.append(f"{name}={1e3*t/max(1,n):.2f}ms x{n}")
        slow = sorted(self._host_slow, reverse=True)[:8]
        self._host_slow.clear()
        if slow:
            parts.append("slow:" + ",".join(f"{n}@{1e3*d:.0f}ms" for d, n in slow))
        self._host.clear()
        return " ".join(parts)

    def decode_step(self, topk_ids: torch.Tensor, cpu_ids: torch.Tensor | None) -> None:
        """Accumulate one decode layer's routing counters (fixed shape, graph safe).

        ``topk_ids`` is the post-ensure slot/-1 view; ``cpu_ids`` the raw ids for
        the CPU-assigned routes (hybrid) or None (pure GPU path)."""
        if not ENABLED or self._dec is None:
            return
        if os.getenv("FREETOKEN_MOE_STATS_HYBRID_PROBE", "0") == "0" and cpu_ids is not None:
            return  # hybrid-path probe wedges bs>=2 graph replays; off by default
        d = self._dec
        if cpu_ids is not None:
            # fixed-shape duplicate count (CUDA-graph safe): histogram over experts,
            # -1 (GPU-assigned) routes clamp onto expert 0 but contribute 0 via mask
            mask = (cpu_ids >= 0).reshape(-1)
            ids = cpu_ids.reshape(-1).clamp_min(0).long()
            hist = torch.zeros(self.cache.num_experts, dtype=torch.int32, device=d.device)
            hist.scatter_add_(0, ids, mask.to(torch.int32))
            d[3] += (hist * (hist - 1) // 2).sum().to(torch.int64)  # duplicate pairs
            d[4] += mask.sum().to(torch.int64)
        else:
            d[4] += (topk_ids >= 0).sum().to(torch.int64)
        d[5] += 1

    def dump(self, kind: str) -> None:
        if not ENABLED or self.cache is None:
            return
        now = time.monotonic()
        if now - self._last_dump < 5.0:
            return
        self._last_dump = now
        cache = self.cache
        try:
            stats = cache.decode_miss_stats()
            d = self._dec.cpu().tolist()
            pairs, routes, layers = d[3], d[4], max(1, d[5])
            per_layer = f"routes/layer={routes/layers:.1f} dup_pairs/layer={pairs/layers:.2f}"
            # flashlib lru counters (GPU slot-cache path; the hybrid stat_* counters
            # only see bs>=hybrid_min_bs steps plus the capture warmups)
            from flashlib.kernels.slot_cache import Stat

            lru = cache.lru_stats.sum(0).tolist()
            calls = max(1.0, lru[Stat.CALLS])
            lru_s = (
                f"lru: active={lru[Stat.ACTIVE]/calls:.2f} miss={lru[Stat.MISS]/calls:.2f} "
                f"rate={100*lru[Stat.MISS]/max(1.0,lru[Stat.ACTIVE]):.1f}% calls={int(calls)}"
            )
            logger.info(
                f"[moestat] decode({kind}): "
                f"active={stats['active_per_layer']:.2f}/layer "
                f"miss={stats['missing_per_layer']:.2f}/layer "
                f"rate={stats['miss_rate']*100:.1f}% "
                f"fetched={stats['fetched_per_layer']:.2f} cpu={stats['cpu_per_layer']:.2f} "
                f"calls={stats['layer_calls']} | {per_layer} | {lru_s} | "
                f"prefill_rows hit={stats['prefill_hit_rows']}/{stats['prefill_rows']}"
            )
            if self._host:
                logger.info(f"[moestat] host: {self.host_dump()}")
        except Exception as exc:  # noqa: BLE001 -- diagnostics must never kill serving
            logger.warning(f"[moestat] dump failed: {exc}")


_PROBE: MoEProbe | None = MoEProbe() if ENABLED else None


def probe() -> MoEProbe | None:
    return _PROBE
