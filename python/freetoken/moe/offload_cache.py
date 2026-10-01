from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterator

import torch
from flashlib.kernels.slot_cache import N_STATS, Stat

# Fuse the per-bank expert copies into a single multi-bank launch (one per copy_missing
# instead of one per bank). Set FREETOKEN_FUSED_COPY=0 to force the legacy per-bank path
# (kept for A/B profiling). Falls back to per-bank automatically if a bank's row bytes or
# base address are not 16-byte aligned.
_FUSED_COPY = os.getenv("FREETOKEN_FUSED_COPY", "1").strip().lower() not in {"0", "false", "no", "off"}

# cudaMemcpyBatchAsync silently degrades to a SYNCHRONOUS copy when a batch mixes
# large entries with sub-~256KB entries on registered host memory (H100 + CUDA 13.0,
# empirically bisected: a single 5-22KB entry beside one large entry blocks the
# calling thread for the full transfer; >=253KB entries never do). A synchronous
# call still moves bytes at full PCIe rate but stalls the host, which un-hides the
# GEMM under the copy in transition-zone workloads (gpt-oss 2048tok: -22% e2e).
# Banks whose rows are smaller than this ship as ONE whole-layer entry (their
# whole layer is tiny) and are excluded from the hit gather, so every per-run
# entry the batch sees is >= this size.
_SMALL_BANK_FEAT_BYTES = 256 * 1024

from freetoken.utils import init_logger

logger = init_logger(__name__)

# quant_format -> bank names, in registration order: the single place a format's bank
# layout is declared. The cache machinery (copy_missing, the prefill double buffers,
# bank_views) iterates banks in this order, the layers' kernel dispatch unpacks views
# in this order, and set_bank_sources validates against it.
_BANK_SCHEMAS: dict[str, tuple[str, ...]] = {
    # dense bf16 expert weights
    "bf16": ("gate_up", "down"),
    # DeepSeek-V3-style 128x128 block-fp8 experts (Qwen3.5-FP8): fp8-e4m3 weights +
    # bf16 per-block weight_scale_inv. gate_up [L*E, 2I, H] fp8 + gate_up_scale
    # [L*E, 2I//128, H//128] bf16; down [L*E, H, I] fp8 + down_scale [L*E, H//128, I//128].
    # Half the host/cache footprint of bf16; the grouped GEMM (kernel/triton/fp8_blockscale_moe)
    # reads the routed fp8 rows directly and dequantizes in the K-loop (no bf16 materialization).
    "fp8_block": ("gate_up", "gate_up_scale", "down", "down_scale"),
    # native GGUF Q4_0 experts: packed block bytes per output row, dequantized inside
    # the borrowed ggml MoE kernels. gate_up [L*E, 2I, H//32*18], down [L*E, H, I//32*18].
    "q4_0": ("gate_up", "down"),
    # native ModelOpt rows for the Triton inline-dequant kernels: packed e2m1 codes +
    # fp8-e4m3 per-16 block scales + per-output-row fp16 globals (w1/w3 carry distinct
    # globals, and folding them into the e4m3 block scales would underflow)
    "nvfp4": (
        "gate_up_packed",
        "gate_up_scale",
        "gate_up_global",
        "down_packed",
        "down_scale",
        "down_global",
    ),
    # pre-tiled layouts for the borrowed kernels; the globals are folded into the
    # block scales at repack time and collapse to [L*E] GPU-resident alpha vectors
    # (set_alphas), so they are not banks
    "nvfp4_marlin": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    "nvfp4_b12x": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    # gpt-oss mxfp4, transposed split-K layout (N innermost): per-expert blocks_t
    # [K//2, N] (uint8), scales_t [K//32, N] (uint8 e8m0), bias [N]. No folded alphas
    # (scales are a bank); split-K GEMV decode + transposed _t grouped prefill.
    "mxfp4_triton": (
        "gate_up_blocks",
        "gate_up_scales",
        "gate_up_bias",
        "down_blocks",
        "down_scales",
        "down_bias",
    ),
    # DeepSeek-V4 FP4: packed e2m1 codes + e8m0 per-32 block scales, no global scale
    # (4 banks). Read by DeepSeek-V4's own DS-FP4 grouped GEMV kernels via bank_views().
    "ds_fp4": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
}

# lives in kernel/aot_models.py: the AOT row table shares it and must stay importable in the torch-only kernel-cache build env, which cannot import freetoken.moe
from freetoken.kernel.aot_models import fp8_block_scale_pad


# bytes per (expert, layer) as f(hidden, moe_intermediate), from the bank shapes above; keep in sync with _BANK_SCHEMAS
# keyed by the config-time format tag (expert_quant / moe_weight_format), not quant_format: "mxfp4" sizes the mxfp4_triton banks, "nvfp4" also covers its repacked variants
_BANK_BYTES_PER_EXPERT = {
    "bf16": lambda H, I: 3 * I * H * 2,
    "fp8_block": lambda H, I: 3 * I * H + (
        (2 * I // 128) * fp8_block_scale_pad(2 * I // 128, H // 128)
        + (H // 128) * fp8_block_scale_pad(H // 128, I // 128)
    ) * 2,
    "q4_0": lambda H, I: 2 * I * (H // 32) * 18 + H * (I // 32) * 18,
    "nvfp4": lambda H, I: 2 * I * (H // 2 + H // 16 + 2) + H * (I // 2 + I // 16 + 2),
    "mxfp4": lambda H, I: 2 * I * (H // 2 + H // 32 + 2) + H * (I // 2 + I // 32 + 2),
    "ds_fp4": lambda H, I: 2 * I * (H // 2 + H // 32) + H * (I // 2 + I // 32),
}

# vLLM's marlin grouped-GEMM hands the full [cache_size] slot cache as its expert
# dimension; moe_align_block_size requires round_up(experts, 32) < 1024, i.e. <= 992.
MARLIN_MAX_CACHE_SIZE = 992


@dataclass
class OffloadMoeCache:
    num_layers: int
    num_experts: int
    cache_size: int
    device: torch.device
    cache_policy: str = "lru"
    prefill_overlap: bool = False
    # Prefill hit/miss split: experts already resident in the slot cache (slots
    # >= 2 * num_experts) are gathered device-side into the double buffer instead
    # of re-crossing PCIe; only the misses are H2D'd (one cudaMemcpyBatchAsync of
    # coalesced runs). Requires prefill_overlap, cache_size > 2 * num_experts and
    # the fused copy plan; silently falls back to the full-layer copy otherwise.
    prefill_hit_d2d: bool = False
    # Touched-only prefill staging: instead of streaming the whole layer (every
    # non-resident expert row) into the double buffer, gather exactly the experts the
    # chunk's routing selected -- hits D2D from the slot cache, misses H2D via the SM
    # gather. Skewed routing means a ~150-token chunk touches ~30% of a layer, so this
    # cuts the prefill's PCIe bytes ~3x; the LRU is left untouched. It also stages the
    # SMALL banks (scales/globals) per row, which the streaming path must copy
    # whole-layer to keep every batch-memcpy entry above the driver's async floor.
    # That whole-layer waste is a per-chunk constant, while the double-buffered overlap
    # touched-only gives up is bounded by the GEMM, so measured over 256..7800-token
    # chunks touched-only is never slower (2.5x faster at 256, 1.1x at 4096, parity at
    # 6144 -- table at the engine's default). ``ondemand_max_tokens`` therefore only
    # guards chunk sizes nobody has measured. Requires prefill_overlap buffers + the
    # fused copy plan + all-pinned layers; set by the engine after construction.
    ondemand_prefill: bool = False
    ondemand_max_tokens: int = 8192
    # Hybrid prefill: share of each layer's touched-and-missing experts that streams
    # over PCIe into the double buffer; the rest are computed on the CPU executor
    # straight from the host banks and their partial is added to the GPU's. 1.0 (the
    # default) keeps every miss on the PCIe plan, i.e. the pre-split behaviour. Only
    # the on-demand path splits -- it stages per row, so a CPU-assigned expert simply
    # leaves the gather plan; the streaming path copies whole layers and cannot.
    # Worth it when the rank's link is slower than its CPU slice: measured on a gen4
    # x4 rank (7.1 GB/s) at T=150, 82 missing experts cost 16.0 ms over PCIe and the
    # balanced split 10.1 ms.
    prefill_fetch_fraction: float = 1.0
    # Hybrid prefill promotion: share of each layer's touched-and-missing experts to
    # LRU-assign a SLOT to and H2D once (copy_missing) BEFORE the on-demand plan runs,
    # so the plan classifies them as hits and gathers them D2D into the double buffer.
    # The promoted rows SURVIVE the chunk: a later prefill/decode routing the same
    # experts reads them at HBM rate instead of re-streaming from host. Note the
    # promoted share rides PCIe IN ADDITION to the plan's own fetch share (plan's
    # pcie_frac then applies to the remaining misses); lower prefill_fetch_fraction
    # to hold total bytes flat. Only pays off while the routing working set fits the
    # slot cache -- measured on a 2x24GB rig (8969 slots), a 600-token prompt touches
    # ~15k experts and thrashes the LRU (no accumulation), and promoting everything
    # binds to the slower rank's link. Default 0.0 = historical stream-and-discard.
    prefill_promote_fraction: float = 0.0
    # Flat residency: every expert of every layer owns a permanent slot
    # (``layer * num_experts + expert``) instead of an LRU one. Needs one cache slot per
    # expert and GPU decode; drops the prefill double buffers, so after the single load in
    # :meth:`materialize_flat` neither prefill nor decode moves an expert byte.
    flat_residency: bool = False
    # "bf16" (default, dense expert weights) or one of the NVFP4 bank layouts:
    # "nvfp4" (native ModelOpt rows, FreeToken Triton kernels), "nvfp4_marlin"
    # (Marlin-tiled, vLLM W4A16 GEMM, sm_80-99) or "nvfp4_b12x" (flashinfer SM12x
    # W4A16); or "mxfp4_triton" (gpt-oss transposed split-K GEMV decode + _t grouped
    # prefill). The format names its bank layout (_BANK_SCHEMAS) and which kernels
    # may read the banks; the cache machinery itself is layout-agnostic.
    quant_format: str = "bf16"
    # Decode mode + bank layout; per-layer CPU routing is cpu_layer_ids. "gpu":
    # GPU-tiled banks, all decode on GPU (stream misses over PCIe into the slot
    # cache, GEMM on GPU). "cpu": native (CPU-readable) banks + a CPU executor;
    # decode computes experts on the CPU (the slot cache only backs the prefill
    # double buffer). "hybrid": native banks + a CPU executor + a full slot cache;
    # each layer fetches a capped subset of its misses over PCIe (``hybrid_max_fetch``
    # / ``hybrid_fetch_fraction`` below; the GPU computes those plus the hits) and the
    # CPU absorbs the overflow misses, then the partials merge. The CPU executor is
    # attached (set_cpu_executor) for cpu/hybrid, set whenever >=1 layer decodes on the CPU.
    decode_target: str = "gpu"
    # Pure-GPU decode fetch overlap (--no-decode-fetch-overlap): stream the misses' H2D
    # on a side stream while the hit routes' GEMV runs, then the miss routes' GEMV after
    # the join (see layers/moe.py _decode_fetch_overlap). Measured on the 2x24GB NVFP4
    # rig at bs16: aggregate +25% (greedy, paired), det16 bit-identical to the serial path.
    decode_fetch_overlap: bool = True
    # hybrid only: max experts fetched over PCIe per (layer, decode step); the rest
    # of that step's misses are computed on the CPU. 0 -> never fetch (CPU does every
    # miss, the GPU cache stays cold); large -> behaves like pure offload.
    hybrid_max_fetch: int = 1
    # hybrid only: when > 0, replaces the fixed cap with a per-step fraction -- fetch
    # ~fraction * misses experts over PCIe (rounded to whichever integer balances the
    # overlap best), the CPU computes the rest. The engine sets it to the benched
    # pcie_bw / cpu_bw ratio so the PCIe fetch and the CPU overflow GEMV take equal
    # time (perfect overlap): fetched : cpu = pcie : cpu - pcie.
    hybrid_fetch_fraction: float = 0.0
    # hybrid only: smallest decode batch that routes misses through the CPU executor.
    # Below it the per-layer submit/sync handshake costs more than the PCIe it saves --
    # a warm bs=1 step misses only ~1 expert/layer, so the GPU slot cache serves it
    # faster alone. The engine raises this to 2 under TP (measured: bs=1 hybrid decodes
    # slower than pure offload there); CUDA graphs bake the branch per captured batch size.
    hybrid_min_bs: int = 1
    # bank layout from the expert kernel (a BankSpec per role); when given it replaces the _BANK_SCHEMAS lookup and the slot cap comes from max_slots
    layout: dict | None = None
    max_slots: int | None = None

    def __post_init__(self) -> None:
        policy_ids = {"lru": 0}
        assert self.cache_policy in policy_ids
        assert self.decode_target in ("gpu", "cpu", "hybrid"), self.decode_target
        if self.layout is None:
            assert self.quant_format in _BANK_SCHEMAS, f"unknown quant_format {self.quant_format!r}"
        # Attached by the engine for decode_target == "cpu" (CpuMoeExecutor); None
        # for the GPU decode path.
        self.cpu_executor = None
        # MoE layer ids whose decode runs on the CPU executor; the rest use the GPU
        # offload/PCIe path. Set by the engine after construction (empty = all-GPU,
        # all layers = the plain --moe-strategy cpu case).
        self.cpu_layer_ids: frozenset = frozenset()
        # num_experts floor + nvfp4_marlin slot cap, shared with the runtime-rebuild path.
        self.validate_rebuild(self.cache_size)
        assert not self.prefill_overlap or self.cache_size >= 2 * self.num_experts, (
            "Prefill overlap borrows two full expert-layer buffers from the unified MoE "
            "cache, so cache_size must be at least 2 * num_experts "
            "(raise moe_cache_size or disable moe_prefill_overlap)"
        )
        if self.flat_residency:
            # Nothing may borrow slots (the double buffers alias slots < 2 * num_experts,
            # which are permanent here) and the CPU executor reads host banks, not slots.
            if self.decode_target != "gpu":
                raise ValueError(
                    f"flat residency serves experts from GPU slots; decode_target="
                    f"{self.decode_target!r} computes them on the CPU executor instead"
                )
            self.prefill_overlap = False
            self.prefill_hit_d2d = False
        self.cache_policy_id = policy_ids[self.cache_policy]
        self.slot_for_id = torch.full(
            (self.num_layers, self.num_experts),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        # Reverse map, in the flat id space flashlib's slot_cache works in:
        # id == layer_id * num_experts + expert, so one array replaces the (layer,
        # expert) pair and evicting a slot needs no decode.
        self.id_of_slot = torch.full(
            (self.cache_size,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.usage = torch.zeros((self.cache_size,), dtype=torch.int64, device=self.device)
        self.step = torch.zeros((), dtype=torch.int64, device=self.device)
        self.active_mask = torch.zeros((self.num_experts,), dtype=torch.int32, device=self.device)
        # lru_ensure validates these against plan = min(batch * top_k, cache_size), so num_experts elements would under-size them
        plan_slots = max(self.num_experts, self.cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.num_indices = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # Dual-microbatch decode (FREETOKEN_DUAL_STREAM_DECODE): the trailing half runs
        # its own ensure/copy plan so the two streams' fetch staging never aliases, and
        # per-layer events serialize the two halves' LRU mutations against each other's
        # in-flight slot reads (see layers/moe.py _decode_routed). Allocated on demand
        # by ensure_dual_plans() before the dual graph warms up.
        self.evict_slots2: torch.Tensor | None = None
        self.src_indices2: torch.Tensor | None = None
        self.num_indices2: torch.Tensor | None = None
        self.dual_events: list | None = None
        self._pending_src_layer2 = None
        # hybrid only: full missing count BEFORE the per-step fetch cap (num_indices holds
        # the capped count that copy_missing actually fetches). The difference is what the
        # CPU computes this step. Written by the hybrid ensure kernel.
        self.num_missing_full = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # hybrid only: per-(layer, expert) last-active decode step (LRU on the expert), -1
        # if never active. The hybrid ensure kernel reads it to pick which capped misses to
        # fetch (most-recently active first) and bumps it for every active expert.
        self.expert_recency = torch.full(
            (self.num_layers, self.num_experts), -1, dtype=torch.int64, device=self.device
        )
        # Pure-GPU decode fetch overlap (see layers/moe.py _decode_fetch_overlap): stream
        # the misses' H2D on a side stream while the hit routes' GEMV runs, then run the
        # miss routes' GEMV after the join. The flag is the --no-decode-fetch-overlap
        # CLI knob (dataclass field, resolved by the engine).
        self.decode_copy_stream: torch.cuda.Stream | None = None
        # Narrow side-stream gather: 1-2 blocks/bank still saturate this rig's PCIe
        # links (measured 6.62/20.4 GB/s vs the wide default's 6.63/20.7) while leaving
        # ~90% of the SMs to the concurrent hit GEMV; the wide 8-blocks/bank launch is
        # what made the first A/B of this split a wash at bs>=2 (SM contention).
        try:
            self.decode_overlap_gather_bpb = int(
                os.environ.get("FREETOKEN_OVERLAP_GATHER_BPB", "2")
            )
        except ValueError:
            self.decode_overlap_gather_bpb = 2
        self._alloc_miss_scratch()
        # Measurement probe (FREETOKEN_TOUCH_OVERLAP_PROBE=1): per-layer cross-chunk
        # touched-expert overlap -- the go/no-go number for ANY cross-chunk expert reuse
        # scheme (promote family). Pure measurement: staging behavior is untouched.
        self._touch_probe = os.environ.get("FREETOKEN_TOUCH_OVERLAP_PROBE", "0") == "1"
        self._probe_prev_touched: torch.Tensor | None = None
        self._probe_acc: torch.Tensor | None = None
        self._probe_chunk = 0
        if self._touch_probe:
            self._probe_prev_touched = torch.zeros(
                (self.num_layers, self.num_experts), dtype=torch.bool, device=self.device
            )
            self._probe_acc = torch.zeros(2, dtype=torch.int64, device=self.device)
        # Auto-promote policy (see ondemand_chunk_begin): per-chunk decision to slot-
        # promote the touched-and-missing experts during on-demand prefill staging, so
        # consecutive chunks of one document gather them D2D instead of re-streaming
        # (measured cross-chunk touched overlap: ~83-90% within a document, and the
        # overlap the fixed-fraction promote path could never exploit because its
        # route-sized ensure kernel costs ~seconds per chunk at prefill T). Env-gated
        # for A/B; an explicit prefill_promote_fraction > 0 takes precedence.
        self.auto_promote = os.environ.get("FREETOKEN_PREFILL_AUTO_PROMOTE", "0") == "1"
        self.promote_auto_frac = 0.0  # policy output for the chunk in flight
        self._promote_on = False
        self._promote_off_streak = 0  # consecutive thrashing chunks while on
        self._promote_frozen = False  # set during graph capture: no policy flips mid-capture
        self._promote_host_ms = 0.0  # chunk-timing attribution (see layers/moe.py)
        self._promote_ema_touched: float | None = None  # per-layer touched EMA
        self._promote_ema_hitrate: float | None = None
        self._promote_harvests = 0
        self._promote_acc = torch.zeros(2, dtype=torch.int64, device=self.device)
        self._promote_miss_tmp = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._promote_pin: torch.Tensor | None = None
        self._promote_ev = None
        self._promote_ev_graph = None  # dedicated event for in-capture rotations
        self._promote_arange = torch.arange(
            self.num_experts, dtype=torch.int32, device=self.device
        )
        self._promote_dummy = torch.zeros((), dtype=torch.int32, device=self.device)
        if self.auto_promote and self.device.type == "cuda":
            self._promote_pin = torch.zeros(2, dtype=torch.int64, pin_memory=True)
        # Host source banks (one [num_experts, ...] tensor per layer, so layers can
        # carry independent host attributes -- see layer_residency) and their GPU
        # slot caches, keyed by the format's bank schema (attached by
        # set_bank_sources). The GPU slot cache stays one unified pool per bank.
        if self.layout is not None:
            self.bank_schema = tuple(role for role, spec in self.layout.items() if not spec.resident)
        else:
            self.bank_schema = _BANK_SCHEMAS[self.quant_format]
        self.bank_sources: dict[str, list[torch.Tensor]] = {}
        self.bank_caches: dict[str, torch.Tensor] = {}
        # per-layer host residency: the GPU movement paths require "pinned"; LOCKED/PAGEABLE layers decode on the CPU executor and prefill via copy_missing's pageable branch
        # _unpinned_layers is the derived id set the hot paths test against
        self.layer_residency: list[str] = []
        self._unpinned_layers: frozenset = frozenset()
        # marlin/b12x per-expert global scales ([L*E], GPU resident, see set_alphas).
        self.gate_up_alpha: torch.Tensor | None = None
        self.down_alpha: torch.Tensor | None = None
        # Opt-in decode miss-rate instrumentation. Accumulated on-device (no per-step host
        # sync); read via ``decode_miss_stats``. Graph-safe: the ``+=`` is captured into the
        # decode graph and re-executes with each replay's REAL routing (record_decode_stats
        # must be enabled before capture — see engine graph setup). The only graph artifact
        # is a one-off warm-up increment at capture time (<0.1% over a session).
        self.collect_stats = False
        # [num_layers, N_STATS] -- ensure_experts passes lru_stats[layer_id] straight to
        # the kernel, which accumulates in the same launch. The stat_* tensors below stay
        # for the hybrid path, whose kernel is still ours.
        self.lru_stats = torch.zeros(
            (self.num_layers, N_STATS), dtype=torch.int64, device=self.device
        )
        self.stat_missing = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_active = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_calls = torch.zeros((), dtype=torch.int64, device=self.device)
        # hybrid only: experts actually fetched over PCIe (<= stat_missing). The CPU
        # computes stat_missing - stat_fetched of them.
        self.stat_fetched = torch.zeros((), dtype=torch.int64, device=self.device)
        # Per-layer counterparts of the scalars above (indexed by MoE-layer id). Same
        # device-side accumulation (graph-safe: layer_id is a static index per graph node),
        # so one req's per-layer miss rate is readable via decode_miss_stats_per_layer().
        self.stat_missing_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_active_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_fetched_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_steps_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        # Opt-in decode routing histogram (per layer, per expert) for cache-skew
        # analysis. Accumulated in ``ensure_experts`` from the raw expert ids before the
        # kernel rewrites them to slots. Only accurate with CUDA graphs disabled (the
        # captured graph would not re-run this host-side scatter on replay).
        self.collect_decode_freq = False
        self.decode_freq = torch.zeros(
            (self.num_layers, self.num_experts), dtype=torch.int64, device=self.device
        )
        # (per-layer sources, cache) per bank, in schema order. Every piece of cache
        # machinery that moves bank bytes (copy_missing, the prefill double buffers,
        # bank_views) iterates this list, so the slot cache is bank-count agnostic.
        self.banks: list[tuple[list[torch.Tensor], torch.Tensor]] = []
        # Fused multi-bank copy descriptor (built by set_bank_sources/_build_copy_plan).
        # Source pointers are per layer (_copy_src_ptrs[layer_id] -> [num_banks] device
        # tensor); dst/feat are layer-invariant.
        self._copy_fused_ok = False
        self._copy_dst_ptrs: torch.Tensor | None = None
        self._copy_src_ptrs: list[torch.Tensor] | None = None
        self._copy_feat_bytes: torch.Tensor | None = None
        # The layer whose misses ensure_experts/materialize_layer staged last; consumed
        # by copy_missing to pick the per-layer source (part of the same pending-copy
        # state as evict_slots/src_indices/num_indices).
        # _pending_whole_layer records WHICH staged it: the pageable branch is only sound after materialize_layer
        self._pending_src_layer: int | None = None
        self._pending_whole_layer = False
        # Per-bank [2, num_experts, ...] double-buffer views over the slot cache's
        # first 2 * num_experts slots (set up when prefill_overlap is enabled).
        self.prefill_bank_buffers: list[torch.Tensor] = []
        self.prefill_copy_stream: torch.cuda.Stream | None = None
        self.prefill_begin_event: torch.cuda.Event | None = None
        self.prefill_ready_events: list[torch.cuda.Event] = []
        self.prefill_release_events: list[torch.cuda.Event] = []
        self._prefill_buffer_layer: list[int | None] = [None, None]
        self._prefill_buffer_released: list[bool] = [True, True]
        self._prefill_buffer_has_release_event: list[bool] = [False, False]
        # hit-D2D split state: pinned begin-of-chunk snapshot of slot_for_id (the
        # classification input; frozen for the chunk -- no decode runs inside one,
        # and buffer invalidation only clears slot < 2E entries, which classify as
        # miss regardless), the lazily resolved batch-memcpy entry point (False =
        # unavailable), and row counters for cache reports.
        self._prefill_slot_snapshot: torch.Tensor | None = None
        self._prefill_snapshot_np = None
        self._prefill_hit_d2d_active = False
        self._hit_d2d_fallback_logged = False
        self._batch_memcpy = None
        # On-demand (touched-only) prefill staging state: the miss gather plan and the
        # per-chunk touched mask (the hit plan reuses the _prefill_hit_* tensors).
        self._prefill_miss_dst: torch.Tensor | None = None
        self._prefill_miss_src: torch.Tensor | None = None
        self._prefill_miss_num: torch.Tensor | None = None
        self._prefill_touched: torch.Tensor | None = None
        self._prefill_cpu_mask: torch.Tensor | None = None
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0

    def set_bank_sources(
        self,
        sources: dict[str, list[torch.Tensor]],
        layer_residency: list[str] | None = None,
    ) -> None:
        """Attach the host (CPU pinned) expert source banks and allocate a GPU slot
        cache per bank, following the format's bank schema.

        Every bank is a list of ``num_layers`` tensors, one ``[num_experts, ...]``
        per layer (independent allocations, so each layer can carry its own host
        attributes); each slot cache mirrors the bank's row shape and dtype as one
        unified GPU pool. The row layouts are produced by the weight loaders /
        repackers (see ``_BANK_SCHEMAS`` and :mod:`freetoken.layers.quantization.moe.nvfp4`)
        -- the cache machinery is layout-agnostic and just moves rows.

        ``layer_residency`` labels each layer with a ``HostResidency`` value (default: all pinned).
        Non-pinned (LOCKED/PAGEABLE) layers have no device address: they must already be routed to the CPU executor (``cpu_layer_ids``, set BEFORE this call), the copy plan skips their rows, and their only movement is ``copy_missing``'s whole-layer pageable prefill branch -- which is why prefill overlap is incompatible with them.
        """
        from freetoken.moe.legacy_format import canonical_role
        from freetoken.moe.host_banks import HostResidency

        # loaders and FTW files may still name the banks the old way (gate_up_packed, ...)
        by_role = {canonical_role(name): per_layer for name, per_layer in sources.items()}
        if set(by_role) != {canonical_role(n) for n in self.bank_schema}:
            raise AssertionError(
                f"banks {sorted(sources)} do not match the {self.quant_format!r} schema {self.bank_schema}"
            )
        sources = {name: by_role[canonical_role(name)] for name in self.bank_schema}
        residency = layer_residency or [HostResidency.PINNED.value] * self.num_layers
        assert len(residency) == self.num_layers, (len(residency), self.num_layers)
        unpinned = frozenset(
            i for i, r in enumerate(residency) if r != HostResidency.PINNED.value
        )
        if unpinned:
            if not unpinned <= self.cpu_layer_ids:
                raise ValueError(
                    f"non-pinned layers {sorted(unpinned - self.cpu_layer_ids)} are not in "
                    f"cpu_layer_ids: a layer without a device address can only decode on "
                    f"the CPU executor (set cache.cpu_layer_ids before set_bank_sources)"
                )
            if self.prefill_overlap:
                raise ValueError(
                    "prefill overlap DMAs from registered banks; it must be disabled "
                    "when any layer is LOCKED/PAGEABLE (the engine does this)"
                )
        self._unpinned_layers = unpinned
        self.layer_residency = list(residency)
        for name in self.bank_schema:
            per_layer = sources[name]
            assert len(per_layer) == self.num_layers, (name, len(per_layer))
            head = per_layer[0]
            if self.layout is not None:
                spec = self.layout[name]
                if tuple(head.shape[1:]) != tuple(spec.shape) or head.dtype != spec.dtype:
                    raise ValueError(
                        f"bank {name!r} rows are {tuple(head.shape[1:])} {head.dtype} but the expert kernel's layout "
                        f"wants {tuple(spec.shape)} {spec.dtype}; the banks were packed for another kernel"
                    )
            for layer_id, source in enumerate(per_layer):
                assert source.is_contiguous(), f"bank {name!r} layer {layer_id} must be contiguous"
                assert source.size(0) == self.num_experts, (name, layer_id, source.shape)
                assert source.shape == head.shape and source.dtype == head.dtype, (
                    name, layer_id, source.shape, source.dtype,
                )
            self.bank_sources[name] = list(per_layer)
            self.bank_caches[name] = torch.empty(
                (self.cache_size, *head.shape[1:]),
                dtype=head.dtype,
                device=self.device,
            )
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]
        self._build_copy_plan()
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def _build_copy_plan(self) -> None:
        self._build_fused_copy_plan()
        if self._copy_fused_ok or self.device.type != "cuda" or not self.banks:
            if self.device.type == "cuda" and self.banks:
                # Per-bank copies silently serialize the in-stream batch gather; log the
                # fallback so a degraded copy is visible in the startup log.
                logger.warning_rank0(
                    "MoE fused multi-bank copy disabled; copy_missing falls back to the "
                    "per-bank path (alignment or FREETOKEN_FUSED_COPY=0)"
                )
            return
        for name in self.bank_schema:
            cache = self.bank_caches[name]
            feat = math.prod(cache.shape[1:]) * cache.element_size()
            if feat % 128:
                raise RuntimeError(
                    f"MoE bank {name!r} rows are {feat} bytes (not a multiple of 128): "
                    f"only the fused multi-bank copy can move them, but it is disabled"
                )

    def _build_fused_copy_plan(self) -> None:
        """Precompute the fused multi-bank copy descriptor (base addrs + per-row bytes).

        Built once here (and on :meth:`rebuild`, which reallocates the slot caches);
        the addresses are fixed for the cache's lifetime so the descriptor tensors are
        CUDA-graph safe. Disabled (-> per-bank fallback) if any bank's row bytes or base
        address is not 16-byte aligned, or via FREETOKEN_FUSED_COPY=0.
        """
        self._copy_fused_ok = False
        self._copy_dst_ptrs = None
        self._copy_src_ptrs = None
        self._copy_feat_bytes = None
        self._copy_dst_ptrs_host: list[int] = []
        self._copy_src_ptrs_host: list[list[int]] = []
        self._copy_feat_bytes_host: list[int] = []
        self._gather_bank_ids: list[int] = []
        self._gather_dst_ptrs: torch.Tensor | None = None
        self._gather_feat_bytes: torch.Tensor | None = None
        if not _FUSED_COPY or self.device.type != "cuda" or not self.banks:
            return
        from freetoken.kernel.pinned import device_ptr

        dst_ptrs, feats = [], []
        layer_src_ptrs = [[] for _ in range(self.num_layers)]
        for per_layer, cache in self.banks:
            feat = math.prod(per_layer[0].shape[1:]) * per_layer[0].element_size()
            if feat % 16 != 0 or cache.data_ptr() % 16 != 0:
                return  # leave fused disabled; copy_missing uses the per-bank path
            for layer_id, source in enumerate(per_layer):
                if layer_id in self._unpinned_layers:
                    # unregistered layer: no device alias exists, and the row is never consumed (CPU decode; pageable prefill)
                    # a 0 placeholder keeps the descriptor shape
                    layer_src_ptrs[layer_id].append(0)
                    continue
                # The kernel dereferences these on the GPU, so store each host bank's
                # device alias (== data_ptr() under UVA identity; differs on
                # Windows/WDDM).
                src_dev = device_ptr(source)
                if src_dev % 16 != 0:
                    return
                layer_src_ptrs[layer_id].append(src_dev)
            dst_ptrs.append(cache.data_ptr())
            feats.append(feat)
        self._copy_dst_ptrs = torch.tensor(dst_ptrs, dtype=torch.int64, device=self.device)
        self._copy_src_ptrs = [
            torch.tensor(ptrs, dtype=torch.int64, device=self.device)
            for ptrs in layer_src_ptrs
        ]
        self._copy_feat_bytes = torch.tensor(feats, dtype=torch.int64, device=self.device)
        self._copy_dst_ptrs_host = dst_ptrs
        self._copy_src_ptrs_host = layer_src_ptrs
        self._copy_feat_bytes_host = feats
        # hit-D2D gather serves only the big banks; small banks are whole-layer
        # H2D entries (see _SMALL_BANK_FEAT_BYTES), so their rows never need D2D.
        self._gather_bank_ids = [i for i, f in enumerate(feats) if f >= _SMALL_BANK_FEAT_BYTES]
        if len(self._gather_bank_ids) == len(feats):
            self._gather_dst_ptrs = self._copy_dst_ptrs
            self._gather_feat_bytes = self._copy_feat_bytes
        elif self._gather_bank_ids:
            self._gather_dst_ptrs = self._copy_dst_ptrs[self._gather_bank_ids].contiguous()
            self._gather_feat_bytes = self._copy_feat_bytes[self._gather_bank_ids].contiguous()
        self._copy_fused_ok = True

    def validate_rebuild(self, cache_size: int) -> None:
        """Pure geometry validation of a rebuild target (no GPU side effects).

        Raises ``ValueError`` if ``cache_size`` is below the ``num_experts`` floor or
        above the marlin slot cap. Called by :meth:`rebuild` and by the engine's
        pre-teardown check, so an invalid target rejects with the old cache intact
        (no destructive free first).
        """
        if self.flat_residency and cache_size < self.total_experts:
            raise ValueError(
                f"flat residency needs one slot per expert: cache_size={cache_size} < "
                f"{self.num_layers} layers * {self.num_experts} experts = {self.total_experts} "
                "(raise moe_cache_size, or drop --moe-flat-residency to cache experts)"
            )
        if cache_size < self.num_experts:
            raise ValueError(f"cache_size {cache_size} < num_experts {self.num_experts}")
        if self.max_slots is not None and cache_size > self.max_slots:
            raise ValueError(
                f"moe_cache_size={cache_size} exceeds the expert kernel's slot limit of {self.max_slots}; "
                f"pass --moe-cache-size {self.max_slots} or less, or let the default kernel serve the experts"
            )
        if self.layout is None and self.quant_format == "nvfp4_marlin" and cache_size > MARLIN_MAX_CACHE_SIZE:
            raise ValueError(
                f"moe_cache_size={cache_size} exceeds the marlin backend's slot limit of "
                f"{MARLIN_MAX_CACHE_SIZE} (vLLM moe_align_block_size caps padded experts at "
                "1024); reduce moe_cache_size or force --quant-backend moe.nvfp4=triton"
            )

    def rebuild(self, cache_size: int) -> None:
        """Resize the GPU slot cache + bookkeeping to ``cache_size`` IN PLACE.

        Keeps the CPU/pinned ``bank_sources`` and the GPU-resident alphas; never
        reloads banks. Tears down prefill-overlap buffers first (their views alias
        the old ``bank_caches``), frees the old GPU tensors, then reallocates. Slots
        cold-start after rebuild. Object identity is preserved so attached layers and
        ``ctx.moe_offload_cache`` stay valid.
        """
        assert self.bank_sources, "set_bank_sources must run before rebuild"
        self.validate_rebuild(cache_size)
        # 1. Tear down prefill-overlap (its buffer views alias the old bank_caches).
        self.prefill_bank_buffers = []
        self.prefill_copy_stream = None
        self.prefill_begin_event = None
        self.prefill_ready_events = []
        self.prefill_release_events = []
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # 2. Drop old GPU tensors (free-before-alloc).
        self.banks = []
        self.bank_caches = {}
        self.cache_size = cache_size
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        # 3. Reallocate the slot cache from the retained host sources.
        for name in self.bank_schema:
            head = self.bank_sources[name][0]
            self.bank_caches[name] = torch.empty(
                (cache_size, *head.shape[1:]), dtype=head.dtype, device=self.device
            )
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]
        self._build_copy_plan()  # slot caches were reallocated -> refresh fused-copy addrs
        # 4. Reallocate cache_size-shaped bookkeeping; reset the slot map (cold start).
        self.slot_for_id.fill_(-1)
        self.id_of_slot = torch.full((cache_size,), -1, dtype=torch.int32, device=self.device)
        self.usage = torch.zeros((cache_size,), dtype=torch.int64, device=self.device)
        plan_slots = max(self.num_experts, cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        if self.evict_slots2 is not None:
            self.evict_slots2 = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
            self.src_indices2 = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self._alloc_miss_scratch()  # cache_size/plan-shaped fetch-overlap scratch
        self.step.zero_()
        self.active_mask.zero_()
        self.num_indices.zero_()
        self.num_missing_full.zero_()
        self.expert_recency.fill_(-1)
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        # a rebuild is a cold start for the cache; carrying pre-rebuild hit/miss counts over would skew every post-rebuild stats report
        self.lru_stats.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()
        self.decode_freq.zero_()
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self._hit_d2d_fallback_logged = False  # geometry changed; re-log if still unusable
        # 5. Flat residency is a permanent mapping, not cache state: the reallocated
        # slots are empty, so reload every expert and restore the identity slot map.
        if self.flat_residency:
            self.materialize_flat()
        # 6. Re-evaluate prefill overlap against the new size.
        if self.prefill_overlap and cache_size < 2 * self.num_experts:
            logger.warning(
                f"Disabling MoE prefill overlap on rebuild: cache_size {cache_size} "
                f"< 2*num_experts {2 * self.num_experts}."
            )
            self.prefill_overlap = False
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def set_alphas(
        self, gate_up_alpha: torch.Tensor | None, down_alpha: torch.Tensor | None
    ) -> None:
        """Attach the marlin/b12x per-expert global scales (``[L*E]``, GPU resident).

        These are kernel-preprocessed scalars, far too small to bother offloading;
        the forward path looks them up per slot with :meth:`alphas_for_slots` /
        :meth:`alphas_for_layer` (pure device-side lookups, CUDA-graph safe).
        ``(None, None)`` is a no-op so callers can pass a format's (possibly
        absent) alphas through unconditionally.
        """
        if gate_up_alpha is None and down_alpha is None:
            return
        assert gate_up_alpha is not None and down_alpha is not None
        total = self.num_layers * self.num_experts
        assert gate_up_alpha.shape == down_alpha.shape == (total,)
        self.gate_up_alpha = gate_up_alpha.to(self.device)
        self.down_alpha = down_alpha.to(self.device)

    def set_cpu_executor(self, executor) -> None:
        """Attach the CPU MoE executor (``decode_target`` in {"cpu", "hybrid"}).

        The executor owns the persistent worker pool, the pinned activation/result
        IO buffers, and the ``cudaLaunchHostFunc`` submit/sync plumbing. It reads
        experts straight from this cache's host ``bank_sources`` (no extra copy).
        """
        assert self.decode_target in ("cpu", "hybrid"), (
            "set_cpu_executor requires decode_target in {'cpu','hybrid'}"
        )
        self.cpu_executor = executor

    def is_cpu_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id`` decodes on the CPU executor (vs the GPU offload path)."""
        return layer_id in self.cpu_layer_ids

    def is_unpinned_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id``'s host banks have no device address (LOCKED/PAGEABLE): the GPU slot-gather paths cannot serve it.
        ``copy_missing`` takes the whole-layer pageable branch, which presumes materialize's position == expert id (never ``ensure_experts``'s LRU slot remap)."""
        return layer_id in self._unpinned_layers

    def alphas_for_slots(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-slot global scales for a decode call, or ``None`` when the format
        keeps no GPU-resident alphas (bf16 / triton-nvfp4). Slots of other layers
        yield garbage values, but only slots routed to -- and those belong to
        ``layer_id`` -- are ever read by the grouped GEMM."""
        if self.gate_up_alpha is None:
            return None
        idx = layer_id * self.num_experts + (
            self.id_of_slot.clamp(min=0).long() % self.num_experts
        )
        return self.gate_up_alpha[idx], self.down_alpha[idx]

    def alphas_for_layer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Global scales for a full-layer prefill (overlap or materialize), where
        position == expert id (contiguous slices, no gather); ``None`` when the
        format keeps no GPU-resident alphas."""
        if self.gate_up_alpha is None:
            return None
        lo = layer_id * self.num_experts
        hi = lo + self.num_experts
        return self.gate_up_alpha[lo:hi], self.down_alpha[lo:hi]

    def bank_views(self, n: int | None = None) -> tuple[torch.Tensor, ...]:
        """Per-bank cache views in registration order: the full ``[S]`` slot cache
        (decode), or its first ``n`` slots (materialized layer)."""
        assert self.banks, "set_bank_sources must register the banks first"
        if n is None:
            return tuple(cache for _, cache in self.banks)
        return tuple(cache[:n] for _, cache in self.banks)

    @property
    def total_experts(self) -> int:
        """Slots flat residency needs: one per (layer, expert)."""
        return self.num_layers * self.num_experts

    def flat_slot_base(self, layer_id: int) -> int:
        """First slot of ``layer_id``'s permanent block (flat residency). Adding it to a
        raw expert id yields the slot id the decode kernels index the cache with."""
        return layer_id * self.num_experts

    def bank_views_flat(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        """Per-bank views of one layer's permanent slot block, rows in expert-id order
        (flat residency) -- the same contract the prefill double buffer hands the GEMM,
        minus the copy."""
        assert self.banks, "set_bank_sources must register the banks first"
        lo = self.flat_slot_base(layer_id)
        return tuple(cache[lo : lo + self.num_experts] for _, cache in self.banks)

    def materialize_flat(self) -> int:
        """Flat residency setup: stream EVERY expert into its permanent slot.

        Slot ``layer * num_experts + expert`` holds that expert for the life of the
        process, so ``slot_for_id`` / ``id_of_slot`` become identity maps and the forward
        paths never reach ``ensure_experts`` / ``copy_missing`` again. Runs once at
        startup and again after a rebuild; returns the bytes moved.
        """
        assert self.flat_residency, "materialize_flat requires flat_residency"
        assert self.banks, "set_bank_sources must register the banks first"
        moved = 0
        for per_layer, cache in self.banks:
            for layer_id, source in enumerate(per_layer):
                lo = self.flat_slot_base(layer_id)
                cache[lo : lo + self.num_experts].copy_(source)
                moved += source.numel() * source.element_size()
        identity = torch.arange(self.total_experts, dtype=torch.int32, device=self.device)
        self.slot_for_id.view(-1).copy_(identity)
        self.id_of_slot.fill_(-1)
        self.id_of_slot[: self.total_experts].copy_(identity)
        return moved

    def _init_prefill_overlap_buffers(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # The double buffers borrow the slot cache's first 2 * num_experts slots
        # (one full expert layer per buffer), one view per registered bank.
        self.prefill_bank_buffers = [
            cache[: 2 * self.num_experts].view(2, self.num_experts, *cache.shape[1:])
            for _, cache in self.banks
        ]
        # Zero the borrowed region once. A row no gather ever staged is read by the
        # hybrid prefill split, which redirects a CPU-assigned route to row 0 with a
        # zeroed weight: bank_caches come from torch.empty, and an uninitialized nvfp4
        # scale byte can encode NaN, which 0 x NaN would propagate into the output.
        # All-zero bytes dequantize to a zero expert, so the redirect contributes 0.
        for buffer in self.prefill_bank_buffers:
            buffer.zero_()
        if self.device.type == "cuda":
            self.prefill_copy_stream = torch.cuda.Stream(device=self.device)
            self.prefill_ready_events = [torch.cuda.Event() for _ in range(2)]
            self.prefill_release_events = [torch.cuda.Event() for _ in range(2)]
            self.prefill_begin_event = torch.cuda.Event()
            # Gather-plan tensors for the hit-D2D split AND the on-demand path (the
            # engine decides which runs after construction; both are tiny).
            self._prefill_hit_dst = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_src = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_num = torch.zeros((1,), dtype=torch.int64, device=self.device)
            self._prefill_miss_dst = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_miss_src = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_miss_num = torch.zeros((1,), dtype=torch.int64, device=self.device)
            self._prefill_touched = torch.zeros(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            # Per-layer verdict of the hybrid prefill split: 1 for a touched miss the
            # CPU executor computes instead of the H2D plan staging. All zero while
            # prefill_fetch_fraction is 1.
            self._prefill_cpu_mask = torch.zeros(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
        if self.prefill_hit_d2d and self.device.type == "cuda":
            self._prefill_slot_snapshot = torch.empty(
                (self.num_layers, self.num_experts), dtype=torch.int32, pin_memory=True
            )
            self._prefill_snapshot_np = self._prefill_slot_snapshot.numpy()

    def _invalidate_prefill_buffer(self, buffer_id: int) -> None:
        slot_start = buffer_id * self.num_experts
        slot_end = slot_start + self.num_experts
        old_ids = self.id_of_slot[slot_start:slot_end]
        self.slot_for_id.view(-1)[old_ids[old_ids >= 0].long()] = -1
        old_ids.fill_(-1)
        # Stamp step+1, not zero. Zero made the borrowed slots the first victim of every
        # decode ensure, so interleaved decode rows landed here and the next chunk's
        # invalidation wiped them (decode hit rate ~0 under mixed load, every interleaved
        # step re-fetched in full). step+1 outranks every LRU-touched row, so decode
        # evicts its own coldest regular slot and keeps residency the chunk plans never
        # touch; the kernel's usage==step mask still falls back to these when every
        # regular slot was used this step. Same device-side stamp as the promotion guard.
        self.usage[slot_start:slot_end] = self.step + 1

    def begin_prefill(self) -> None:
        if not self.prefill_overlap:
            return
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        if self.prefill_copy_stream is not None:
            # Fence this prefill's copy-stream work behind everything already enqueued
            # on the compute stream. The release/ready events only order against the
            # *previous prefill*; under overlap scheduling a new prefill can be enqueued
            # while the preceding decode batch is still running, and that decode may
            # have loaded experts into the slots the buffers borrow -- without this
            # fence the first prefetch would stomp bytes a running GEMM is reading.
            self.prefill_begin_event.record(torch.cuda.current_stream(self.device))
            self.prefill_copy_stream.wait_event(self.prefill_begin_event)
        self._prefill_hit_d2d_active = self.prefill_hit_d2d and self._hit_d2d_usable()
        if self._prefill_hit_d2d_active:
            # The copy stream is fenced behind the previous decode, so the snapshot
            # observes its final slot map; one host sync per chunk, then per-layer
            # classification is pure host math.
            with torch.cuda.stream(self.prefill_copy_stream):
                self._prefill_slot_snapshot.copy_(self.slot_for_id, non_blocking=True)
            self.prefill_copy_stream.synchronize()

    def prefetch_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap or layer_id >= self.num_layers:
            return
        if layer_id < 0:
            raise ValueError(f"Invalid prefill layer id: {layer_id}")

        assert self.banks and self.prefill_bank_buffers

        buffer_id = layer_id % 2
        if self._prefill_buffer_layer[buffer_id] == layer_id:
            return
        if self._prefill_buffer_layer[buffer_id] is not None:
            assert self._prefill_buffer_released[buffer_id], (
                "Prefill overlap buffer is being reused before release"
            )

        def copy() -> None:
            self._invalidate_prefill_buffer(buffer_id)
            for (per_layer, _), buffer in zip(self.banks, self.prefill_bank_buffers):
                buffer[buffer_id].copy_(per_layer[layer_id], non_blocking=True)

        if self._prefill_hit_d2d_active:
            self._prefetch_split(layer_id, buffer_id)
        elif self.prefill_copy_stream is None:
            copy()
        else:
            with torch.cuda.stream(self.prefill_copy_stream):
                if self._prefill_buffer_has_release_event[buffer_id]:
                    self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
                copy()
                self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

        self._prefill_buffer_layer[buffer_id] = layer_id
        self._prefill_buffer_released[buffer_id] = False

    def _hit_d2d_usable(self) -> bool:
        """Whether the hit-D2D split can serve this prefill; logs the first fallback.

        The flag is an auto-fallback optional: any unusable condition must degrade
        to the legacy full-layer copy AND say so once in the server log, so a
        configuration that silently runs the legacy path is visible.
        """
        from freetoken.kernel.fast_index_copy import _skip_fast_index_copy_enabled

        if self._prefill_slot_snapshot is None or self.prefill_copy_stream is None:
            reason = "prefill overlap buffers are not initialized for this device"
        elif _skip_fast_index_copy_enabled():
            reason = "FREETOKEN_SKIP_FAST_INDEX_COPY is set (the hit gather would be a no-op)"
        elif not self._copy_fused_ok:
            reason = "the fused copy plan is unavailable (bank alignment or FREETOKEN_FUSED_COPY=0)"
        elif self.cache_size <= 2 * self.num_experts:
            reason = (
                f"cache_size {self.cache_size} leaves no hit region "
                f"(needs > {2 * self.num_experts} slots)"
            )
        elif not self._resolve_batch_memcpy():
            reason = "cudaMemcpyBatchAsync is unavailable"  # resolve logged the specifics
        else:
            return True
        if not self._hit_d2d_fallback_logged:
            logger.warning(
                f"MoE prefill hit-D2D requested but unavailable ({reason}); "
                "falling back to full-layer copies"
            )
            self._hit_d2d_fallback_logged = True
        return False

    def _resolve_batch_memcpy(self) -> bool:
        if self._batch_memcpy is None:
            try:
                from freetoken.kernel.batch_memcpy import load_batch_memcpy

                self._batch_memcpy = load_batch_memcpy()
            except Exception as exc:  # noqa: BLE001 -- any build/runtime gap => legacy path
                logger.warning(f"MoE prefill hit-D2D disabled ({exc}); using full-layer copies")
                self._batch_memcpy = False
        return self._batch_memcpy is not False

    def _prefetch_split(self, layer_id: int, buffer_id: int) -> None:
        """Hit/miss-split prefetch of one expert layer into the double buffer.

        Resident experts are gathered cache -> buffer on the CURRENT stream, fully
        device-side: a one-launch compaction reads the LIVE slot_for_id row into
        fixed-shape gather indices (no host round trip), then fast_index_copy_multi
        moves the rows. Serializing the gather before this layer's GEMMs costs its
        plain duration instead of nondeterministic SM contention. Misses cross
        PCIe as ONE cudaMemcpyBatchAsync of coalesced expert-id runs on the copy
        stream, under the existing release/ready event discipline; its host-built
        run list comes from the begin-of-chunk snapshot because the batch API
        takes HOST pointer arrays. Live-vs-snapshot cannot disagree: the only
        chunk-internal writer (buffer invalidation) rewrites slots already below
        the 2E threshold, and slots < 2E (including -1) are misses on both sides
        -- the buffers own those slots, so their bytes are volatile within the
        chunk. Hit and miss row sets are disjoint, so the streams need no
        ordering against each other.
        """
        import numpy as np

        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
        from freetoken.moe.offload_kernels import prefill_hit_compact

        E = self.num_experts
        snap = self._prefill_snapshot_np[layer_id]
        hit_mask = snap >= 2 * E
        self.prefill_hit_rows += int(hit_mask.sum())
        self.prefill_total_rows += E
        if self._gather_dst_ptrs is not None:
            prefill_hit_compact(self, layer_id, buffer_id)
            # blocks_per_bank=64 vs the PCIe-tuned default of 8: HBM D2D needs the
            # wider grid (~22 GB/s per 1024-thread block on H100).
            fast_index_copy_multi_jit(
                self._gather_dst_ptrs,
                self._gather_dst_ptrs,
                self._gather_feat_bytes,
                self._prefill_hit_dst,
                self._prefill_hit_src,
                self._prefill_hit_num,
                blocks_per_bank=64,
            )
        miss = np.nonzero(~hit_mask)[0]
        with torch.cuda.stream(self.prefill_copy_stream):
            if self._prefill_buffer_has_release_event[buffer_id]:
                self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
            self._invalidate_prefill_buffer(buffer_id)
            if miss.size:
                run_starts = np.concatenate(([0], np.nonzero(np.diff(miss) != 1)[0] + 1))
                starts = miss[run_starts]
                lengths = np.diff(np.concatenate((run_starts, [miss.size])))
            dst, src, nbytes = [], [], []
            for b, feat in enumerate(self._copy_feat_bytes_host):
                if feat < _SMALL_BANK_FEAT_BYTES:
                    # Whole layer as one entry, EVEN with zero misses: it keeps every
                    # batch entry above the driver's async floor and covers the hit
                    # rows the gather skips for these banks.
                    dst.append(self._copy_dst_ptrs_host[b] + buffer_id * E * feat)
                    src.append(self._copy_src_ptrs_host[layer_id][b])
                    nbytes.append(E * feat)
                elif miss.size:
                    dst.extend(self._copy_dst_ptrs_host[b] + (buffer_id * E + starts) * feat)
                    src.extend(self._copy_src_ptrs_host[layer_id][b] + starts * feat)
                    nbytes.extend(lengths * feat)
            if dst:
                self._batch_memcpy(
                    torch.tensor(dst, dtype=torch.int64),
                    torch.tensor(src, dtype=torch.int64),
                    torch.tensor(nbytes, dtype=torch.int64),
                    torch.cuda.current_stream(self.device).cuda_stream,
                )
            self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

    def prefetch_prefill_layer_ondemand(
        self, layer_id: int, topk_ids: torch.Tensor, pcie_frac_q16: int = 1 << 16
    ) -> tuple[torch.Tensor, ...]:
        """Plan AND move one layer's touched-only staging; see ``plan_``/``move_`` below.

        Callers that also hand part of the miss set to the CPU executor must use the two
        halves directly so the submit lands between them.
        """
        views = self.plan_prefill_layer_ondemand(layer_id, topk_ids, pcie_frac_q16)
        self.move_prefill_layer_ondemand(layer_id)
        return views

    def promote_prefill_misses(self, layer_id: int, topk_ids: torch.Tensor) -> None:
        """LRU-assign slots to ~``prefill_promote_fraction`` of this layer's misses and
        H2D them ONCE into the slot cache, before ``plan_prefill_layer_ondemand`` runs.

        The plan then classifies the promoted experts as hits (slot >= 2E) and gathers
        them D2D into the double buffer: same PCIe bytes as streaming into the buffer,
        but the rows survive the chunk and a later prefill/decode that routes the same
        experts reads them at HBM rate. Recency-ordered like the decode fetch (recurring
        misses promote first).

        The double buffer ALIASES slots [0, 2E): a promotion landing there would be
        trampled by this very chunk's staging and then invalidated, wasting its H2D.
        Guard the round's victim set by freshening those slots' usage to step+1 (the
        stamp this round's own assigns will get, device-side max, no sync): the LRU
        min-(usage, index) pick then prefers every older slot, while later decode steps
        age the buffer slots back into candidacy as usual.
        """
        if self.prefill_promote_fraction <= 0:
            return
        if layer_id in self._unpinned_layers:
            return  # no device alias to slot-map against (see copy_missing's guard)
        from freetoken.moe.offload_kernels import ensure_experts_hybrid

        buffer_slots = 2 * self.num_experts
        torch.maximum(
            self.usage[:buffer_slots], self.step + 1, out=self.usage[:buffer_slots]
        )
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        # clone: ensure rewrites ids to slot ids in place; the caller's plan and the
        # CPU-executor mask still need the raw expert ids.
        ensure_experts_hybrid(
            self, layer_id, topk_ids.clone(), self.num_experts, self.prefill_promote_fraction
        )
        self.copy_missing()

    def promote_touched(
        self, layer_id: int, topk_ids: torch.Tensor, stats: bool = False
    ) -> None:
        """Fast mask-driven promote for the auto policy (offload_kernels.promote_touched).

        Builds the touched mask exactly like ``plan_prefill_layer_ondemand`` does (the
        plan rebuilds it right after; the scatter is microseconds) and promotes ALL
        touched-and-missing rows -- the policy gate decides WHETHER this runs, not a
        per-call fraction, because the fast path's cost is per-launch, not per-route.

        ``stats=True`` (the slot-direct prefill path, which runs no plan) accumulates
        [hits, misses] into ``_promote_acc`` so the policy feedback keeps measuring
        reuse: a hit is a touched row that was already resident (no fresh H2D).
        """
        if layer_id in self._unpinned_layers:
            return
        from freetoken.moe.offload_kernels import promote_touched

        touched = self._prefill_touched
        touched.zero_()
        touched.scatter_(0, topk_ids.reshape(-1).long(), 1)
        count = stats and self.auto_promote and self._promote_pin is not None
        if count:
            self._promote_miss_tmp.zero_()
        promote_touched(
            self, layer_id, touched, miss_acc=self._promote_miss_tmp if count else None
        )
        if count:
            total = torch.count_nonzero(touched)
            # clamp: the promote query's dummy pad id counts as a miss when expert 0 is
            # cold, which would push hits negative and the hit-rate EMA below the demote
            # threshold on the very first (all-miss) chunks of a document.
            miss = torch.minimum(self._promote_miss_tmp, total)
            self._promote_acc[1:2] += miss
            self._promote_acc[0:1] += total - miss

    def ondemand_chunk_begin(self) -> None:
        """Auto-promote policy step at the first MoE layer of each on-demand prefill chunk.

        Harvests the previous chunk's plan hit/miss totals WITHOUT a host sync (they ride
        a pinned async copy whose event is polled with query(); the stats therefore lag one
        chunk, which is fine for a feedback policy), updates the EMAs, decides this chunk's
        promote gate, and re-arms the accumulator. All device work is stream-ordered on the
        compute stream after the previous chunk's plans.
        """
        if not self.auto_promote or self._promote_pin is None:
            return
        self.harvest_promote_stats()
        self.rotate_promote_acc()

    def harvest_promote_stats(self) -> None:
        """Host half of the policy feedback (see ondemand_chunk_begin). Split out so the
        prefill chunk graph can call it between replays: the in-graph layer-0 rotation
        (baked at capture) refills the pin, but host code does not run during a replay."""
        if not self.auto_promote or self._promote_pin is None or self._promote_frozen:
            return
        if self._promote_ev is not None and self._promote_ev.query():
            hits = int(self._promote_pin[0])
            misses = int(self._promote_pin[1])
            touched = hits + misses
            if touched > 0:
                per_layer = touched / self.num_layers
                ema = self._promote_ema_touched
                self._promote_ema_touched = per_layer if ema is None else 0.7 * ema + 0.3 * per_layer
                rate = hits / touched
                ema_r = self._promote_ema_hitrate
                self._promote_ema_hitrate = rate if ema_r is None else 0.7 * ema_r + 0.3 * rate
                self._promote_harvests += 1
                if self._promote_harvests % 32 == 0:
                    # rate-limited observability: the plan hit-rate the policy sees
                    # (~overlap when reuse works; ~incidental decode-row hits when the
                    # working set overflows the pool and the LRU thrashes cyclically).
                    logger.info_rank0(
                        f"auto-promote: chunk {self._promote_harvests} "
                        f"hitrate_ema={self._promote_ema_hitrate:.3f} "
                        f"touched/layer~{self._promote_ema_touched:.0f} on={self._promote_on}"
                    )
                self._update_promote_policy()

    def rotate_promote_acc(self) -> None:
        """Device half: async-copy the accumulator to the pin, mark the event, re-arm.
        Pure stream-ordered device ops, so it captures into the prefill chunk graph.

        Under capture the record must go to a DEDICATED event: recording a regular
        event on a capturing stream taints it for host queries (cudaEventQuery ->
        "invalid argument"), which would break the eager harvest after any graphed
        chunk. The graph event is a baked node nobody queries; the regular event
        keeps its last eager record (already complete), and the pin it guards holds
        the latest replay's totals, so a stale-but-signaled query still reads fresh
        stats."""
        capturing = (
            self.device.type == "cuda" and torch.cuda.is_current_stream_capturing()
        )
        if capturing:
            if self._promote_ev_graph is None:
                self._promote_ev_graph = torch.cuda.Event()
            ev = self._promote_ev_graph
        else:
            if self._promote_ev is None:
                self._promote_ev = torch.cuda.Event()
            ev = self._promote_ev
        self._promote_pin.copy_(self._promote_acc, non_blocking=True)
        ev.record(torch.cuda.current_stream(self.device))
        self._promote_acc.zero_()

    def _update_promote_policy(self) -> None:
        """Bang-bang policy with hysteresis over the measured EMAs.

        First arm is optimistic: ON while the chunk working set (touched/layer x layers)
        is within 2x the slot budget (cache_size minus the two borrowed double-buffer
        layers). The plan hit-rate is the real judge: OFF after two consecutive chunks of
        near-zero reuse (thrashing: promotion paid the ensure/copy overhead and bought
        nothing), and re-arming afterwards requires the estimate within 80% of the budget.
        """
        slots_avail = max(self.cache_size - 2 * self.num_experts, 0)
        ws = (self._promote_ema_touched or 0.0) * self.num_layers
        was_on = self._promote_on
        if self._promote_on:
            if (
                self._promote_ema_hitrate is not None
                and self._promote_ema_hitrate < 0.05
                and self._promote_harvests >= 4  # cold-start chunks must not trip demotion
            ):
                self._promote_off_streak += 1
                if self._promote_off_streak >= 2:
                    self._promote_on = False
            else:
                self._promote_off_streak = 0
        elif ws > 0 and ws <= slots_avail * (0.8 if self._promote_off_streak else 2.0):
            self._promote_on = True
            self._promote_off_streak = 0
        self.promote_auto_frac = (
            min(1.0, slots_avail / ws) if (self._promote_on and ws > 0) else 0.0
        )
        if self._promote_on != was_on:
            hr = self._promote_ema_hitrate
            logger.info_rank0(
                f"auto-promote {'ON' if self._promote_on else 'OFF'}: ws~{ws:.0f} "
                f"slots_avail={slots_avail} hitrate={'n/a' if hr is None else round(hr, 3)} "
                f"frac={self.promote_auto_frac:.2f}"
            )

    def plan_prefill_layer_ondemand(
        self, layer_id: int, topk_ids: torch.Tensor, pcie_frac_q16: int = 1 << 16
    ) -> tuple[torch.Tensor, ...]:
        """Touched-only staging PLAN for one expert layer of a prefill chunk; moves nothing.

        Fully device-side on the current stream (no host sync, no copy-stream
        choreography): mark the chunk's routed experts in ``_prefill_touched``,
        compact them into hit (D2D slot gather) and miss (H2D pinned gather, the
        same SM primitive decode's ``copy_missing`` uses -- it runs at DMA rate at
        these sizes) plans, invalidate this buffer's stale slot mappings, and return
        the full-layer views (position == expert id, so the caller's GEMM contract is
        unchanged). The next layer's routing does not exist yet, so there is nothing
        to prefetch ahead of.

        ``pcie_frac_q16`` < 1<<16 hands the tail of the miss set to the CPU executor
        instead of the H2D plan (see ``prefill_cpu_mask``); those rows stay unstaged,
        so the caller must zero their routing weight before the GEMM.

        The LRU slot cache is neither read for eviction nor written: only rows at
        slots >= 2E are gathered, and the buffer slots' stale ids are invalidated
        exactly like the streaming path does.
        """
        assert self.prefill_overlap and self.prefill_bank_buffers
        assert self._copy_fused_ok, "on-demand prefill needs the fused copy plan"
        from freetoken.moe.offload_kernels import prefill_ondemand_compact

        buffer_id = layer_id % 2
        touched = self._prefill_touched
        touched.zero_()
        touched.scatter_(0, topk_ids.reshape(-1).long(), 1)
        if self._touch_probe:
            # prev-chunk overlap of THIS layer's touched set; the per-chunk ratio is
            # logged at the last layer (one small sync per chunk, probe builds only).
            prev = self._probe_prev_touched[layer_id]
            cur = touched != 0
            self._probe_acc[0] += torch.count_nonzero(torch.logical_and(cur, prev))
            self._probe_acc[1] += torch.count_nonzero(cur)
            prev.copy_(cur)
            if layer_id == self.num_layers - 1:
                ov, tv = (int(x) for x in self._probe_acc.tolist())
                self._probe_chunk += 1
                if self._probe_chunk > 1:
                    logger.info_rank0(
                        f"touch-overlap: chunk {self._probe_chunk} T={topk_ids.shape[0]}: "
                        f"prev&cur/cur = {ov}/{tv} = {ov / max(tv, 1):.3f}"
                    )
                self._probe_acc.zero_()
        prefill_ondemand_compact(self, layer_id, buffer_id, touched, pcie_frac_q16=pcie_frac_q16)
        if self.auto_promote and self._promote_pin is not None:
            # chunk totals for the policy feedback (harvested at the next chunk begin);
            # stream-ordered after the compact kernel that wrote the per-layer counts.
            # [0:1] slices: += against the [1]-shaped counters must not reshape the view.
            self._promote_acc[0:1] += self._prefill_hit_num
            self._promote_acc[1:2] += self._prefill_miss_num
        self._invalidate_prefill_buffer(buffer_id)
        return tuple(buffer[buffer_id] for buffer in self.prefill_bank_buffers)

    def move_prefill_layer_ondemand(self, layer_id: int) -> None:
        """Issue the gathers planned by :meth:`plan_prefill_layer_ondemand`.

        Split out from the planning step so a caller can submit the CPU executor's partial
        BETWEEN the two: on one stream the D2H activation copy a submit needs cannot start
        until everything already enqueued has finished, so issuing the H2D expert gather
        first serializes the CPU work behind the whole transfer instead of overlapping it.
        """
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        # Hits gather D2D over ALL banks (the SM kernel has no small-row caveat,
        # unlike the streaming path's batch memcpy, so scales/globals come from
        # their cache slots too); misses gather H2D from the host banks. D2D
        # first: the HBM rate is ~40x the PCIe rate.
        fast_index_copy_multi_jit(
            self._copy_dst_ptrs,
            self._copy_dst_ptrs,
            self._copy_feat_bytes,
            self._prefill_hit_dst,
            self._prefill_hit_src,
            self._prefill_hit_num,
            blocks_per_bank=64,
        )
        fast_index_copy_multi_jit(
            self._copy_dst_ptrs,
            self._copy_src_ptrs[layer_id],
            self._copy_feat_bytes,
            self._prefill_miss_dst,
            self._prefill_miss_src,
            self._prefill_miss_num,
        )

    @property
    def prefill_cpu_mask(self) -> "torch.Tensor | None":
        """[num_experts] int32 verdict of the last ``prefetch_prefill_layer_ondemand``:
        1 for a touched miss the CPU executor owns instead of the H2D plan. None until
        the on-demand buffers exist; all zero while ``prefill_fetch_fraction`` is 1."""
        return self._prefill_cpu_mask

    def prefill_pcie_frac_q16(self) -> int:
        """Q16 share of a prefill layer's misses to stage over PCIe (see the kernel)."""
        return min(1 << 16, max(0, round(self.prefill_fetch_fraction * (1 << 16))))

    def wait_prefill_layer(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        """Full-layer ``[num_experts, ...]`` bank views for ``layer_id``, one per
        registered bank in registration order: bf16 ``(gate_up, down)``; nvfp4
        marlin/b12x ``(gate_up_packed, gate_up_scale, down_packed, down_scale)``;
        nvfp4 native adds the two global banks after each scale bank."""
        assert self.prefill_overlap
        assert self.prefill_bank_buffers
        self.prefetch_prefill_layer(layer_id)
        buffer_id = layer_id % 2
        assert self._prefill_buffer_layer[buffer_id] == layer_id
        if self.prefill_ready_events:
            torch.cuda.current_stream(self.device).wait_event(self.prefill_ready_events[buffer_id])
        return tuple(buffer[buffer_id] for buffer in self.prefill_bank_buffers)

    def release_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap:
            return
        buffer_id = layer_id % 2
        if self._prefill_buffer_layer[buffer_id] != layer_id:
            return
        if self.prefill_release_events:
            self.prefill_release_events[buffer_id].record(torch.cuda.current_stream(self.device))
            self._prefill_buffer_has_release_event[buffer_id] = True
        self._prefill_buffer_released[buffer_id] = True

    def ensure_dual_plans(self, num_layers: int) -> None:
        """Materialize the trailing half's fetch plan + the per-layer dual events.

        Idempotent; called by the graph runner before the dual decode graph warms up
        (never during capture: fresh device allocations must precede it)."""
        if self.evict_slots2 is not None:
            return
        plan_slots = max(self.num_experts, self.cache_size)
        self.evict_slots2 = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.src_indices2 = torch.empty((plan_slots,), dtype=torch.int32, device=self.device)
        self.num_indices2 = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # [layer][0]=A-done (recorded after the leading half's expert GEMV),
        # [layer][1]=B-done (trailing half's); waits are always forward edges in the
        # capture's issue order (A_l -> B_l -> A_{l+1}).
        self.dual_events = [
            (torch.cuda.Event(), torch.cuda.Event()) for _ in range(num_layers)
        ]

    def _plan_buffers(self, plan: int):
        if plan:
            return self.src_indices2, self.evict_slots2, self.num_indices2
        return self.src_indices, self.evict_slots, self.num_indices

    def _alloc_miss_scratch(self) -> None:
        """(Re)allocate the fetch-overlap miss-mask scratch for the current cache_size.

        The mask is a slot->flag scatter table (+1 sentinel row for the plan's unused
        tail, whose evict_slots bytes are uninitialized and must not flag real slots).
        """
        if not self.decode_fetch_overlap:
            self._miss_flags = None
            self._miss_arange = None
            self._miss_sentinel = None
            return
        self._miss_flags = torch.zeros(self.cache_size + 1, dtype=torch.int8, device=self.device)
        self._miss_arange = torch.arange(
            self.evict_slots.numel(), dtype=torch.int64, device=self.device
        )
        self._miss_sentinel = torch.full((1,), self.cache_size, dtype=torch.int64, device=self.device)

    def miss_route_mask(self, topk_ids: torch.Tensor, plans=(0,)) -> torch.Tensor:
        """[M, top_k] bool: True where the route's slot is one ``ensure_experts`` just
        staged for fetch (its bytes land via ``copy_missing``). Device-side fixed-shape
        (scatter through a sentinel row for each plan's unused tail), so it is CUDA-graph
        safe. Call BETWEEN the ensure(s) and the copies: it reads their shared plan state.
        ``plans`` covers the K-split admission's per-chunk plan buffers."""
        self._miss_flags.zero_()
        for plan in plans:
            _, evict_slots, num_indices = self._plan_buffers(plan)
            valid = self._miss_arange < num_indices
            idx = torch.where(valid, evict_slots.to(torch.int64), self._miss_sentinel)
            self._miss_flags.scatter_(0, idx, valid.to(torch.int8))
        flat = self._miss_flags[topk_ids.reshape(-1).to(torch.int64)]
        return flat.view(topk_ids.shape).bool()

    def ensure_experts(self, layer_id: int, expert_ids: torch.Tensor, plan: int = 0) -> None:
        from freetoken.moe.offload_kernels import ensure_experts

        if plan:
            self._pending_src_layer2 = layer_id
        else:
            self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        if self.collect_decode_freq:
            # ``expert_ids`` still holds raw expert ids here (the kernel rewrites them to
            # slot ids in place), so snapshot the routing histogram before that happens.
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        ensure_experts(self, layer_id, expert_ids, plan=plan)

    def ensure_experts_chunked(
        self, layer_id: int, expert_ids: torch.Tensor, max_k: int, plan: int = 0
    ) -> int:
        """ensure+copy per row-chunk so no lru_ensure call sees K > ``max_k``.

        The phase-1 dedup block is [K, K] in registers and spills hard above
        K~128 (flashlib cost model + measured ~198us/call at K=200 on sm120 =
        ~9.5ms of pure admission overhead per bs20 step). Chunking keeps every
        call in the flat cost region; correctness rides stream order (each
        chunk's gather consumes the plan buffer before the next ensure restages
        it) and LRU usage stamps (a chunk's admits and hits carry the current
        step's stamp, so the next chunk's eviction -- argmin over usage -- can
        never pick them while older slots exist). Returns the chunk count."""
        rows, top_k = expert_ids.shape
        rows_chunk = max(1, max_k // top_k)
        n = 0
        for a in range(0, rows, rows_chunk):
            self.ensure_experts(layer_id, expert_ids[a : a + rows_chunk], plan=plan)
            self.copy_missing(plan=plan)
            n += 1
        return n

    def ensure_experts_hybrid(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Capped-fetch LRU for the hybrid backend.

        Like :meth:`ensure_experts` but assigns slots to (and schedules copies for) at
        most ``hybrid_max_fetch`` -- or ``~hybrid_fetch_fraction * misses`` when the
        fraction is set -- of this step's missing experts; the overflow misses are
        left non-resident and ``expert_ids`` is rewritten to their cache slot (hit or
        freshly fetched) or ``-1`` (overflow -> compute on the CPU). ``num_indices`` holds
        the capped fetch count (for ``copy_missing``); ``num_missing_full`` the pre-cap
        miss count (for stats). All device-side / fixed-shape, so it is CUDA-graph safe."""
        from freetoken.moe.offload_kernels import ensure_experts_hybrid

        if self.collect_decode_freq:
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        ensure_experts_hybrid(
            self, layer_id, expert_ids, self.hybrid_max_fetch, self.hybrid_fetch_fraction
        )

    def materialize_layer(self, layer_id: int) -> None:
        from freetoken.moe.offload_kernels import materialize_layer

        self._pending_src_layer = layer_id
        self._pending_whole_layer = True
        materialize_layer(self, layer_id)

    def reset(self) -> None:
        from freetoken.moe.offload_kernels import reset_cache

        reset_cache(self)
        # Per-expert recency is not cache_size-shaped, so reset_cache leaves it alone; wipe
        # it here so a new sequence starts with cold hybrid fetch priorities.
        self.expert_recency.fill_(-1)

    def reset_stats(self) -> None:
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self.lru_stats.zero_()
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()

    def record_decode_stats(self, layer_id: int) -> None:
        """No-op: ``ensure_experts`` accumulates into ``lru_stats`` inside its own launch.

        Kept so the hybrid and non-hybrid call sites stay symmetric. The previous version
        was eight torch ops per layer per step, all captured into the decode graph.
        """

    def record_decode_stats_hybrid(self, layer_id: int) -> None:
        """Hybrid stats: full miss count (pre-cap), the PCIe-fetched count (capped), and
        the active count. The CPU computes (missing - fetched) experts. Device-side;
        accumulates both the scalar totals and the per-layer breakdown."""
        assert 0 <= layer_id < self.num_layers, f"layer_id {layer_id} out of range [0, {self.num_layers})"
        missing = self.num_missing_full.sum()
        fetched = self.num_indices.sum()
        active = self.active_mask.sum()
        self.stat_missing += missing
        self.stat_fetched += fetched
        self.stat_active += active
        self.stat_calls += 1
        self.stat_missing_layer[layer_id] += missing
        self.stat_fetched_layer[layer_id] += fetched
        self.stat_active_layer[layer_id] += active
        self.stat_steps_layer[layer_id] += 1

    def decode_miss_stats(self) -> dict:
        if self.decode_target == "hybrid":
            active = int(self.stat_active.item())
            missing = int(self.stat_missing.item())
            calls = int(self.stat_calls.item())
        else:
            active, missing, calls = (int(x) for x in self.lru_stats.sum(0))
        fetched = int(self.stat_fetched.item())
        return {
            "layer_calls": calls,
            "active_per_layer": (active / calls) if calls else 0.0,
            "missing_per_layer": (missing / calls) if calls else 0.0,
            "miss_rate": (missing / active) if active else 0.0,
            # hybrid: how the misses split between PCIe fetch (GPU) and CPU compute.
            "fetched_per_layer": (fetched / calls) if calls else 0.0,
            "cpu_per_layer": ((missing - fetched) / calls) if calls else 0.0,
            "fetch_rate": (fetched / missing) if missing else 0.0,
            # prefill hit-D2D split: expert rows served from the cache (D2D) vs all
            # rows prefetched into the double buffer since the last reset.
            "prefill_hit_rows": self.prefill_hit_rows,
            "prefill_rows": self.prefill_total_rows,
        }

    def decode_miss_stats_per_layer(self) -> dict:
        """Per-MoE-layer realized decode stats for one (reset_stats-delimited) window.

        Requires ``collect_stats`` and the call sites passing ``layer_id``. Returns python
        lists indexed by MoE-layer id: missing/active experts per step and the realized
        miss_rate (missing/active) -- i.e. how cacheable each layer's routing actually was
        under the running LRU. Reads device tensors once (no per-step host sync)."""
        if self.decode_target == "hybrid":
            steps = self.stat_steps_layer.tolist()
            missing = self.stat_missing_layer.tolist()
            active = self.stat_active_layer.tolist()
        else:
            cols = self.lru_stats.t().tolist()
            active, missing, steps = cols[Stat.ACTIVE], cols[Stat.MISS], cols[Stat.CALLS]
        fetched = self.stat_fetched_layer.tolist()
        per_layer = []
        for L in range(self.num_layers):
            s, m, a, f = steps[L], missing[L], active[L], fetched[L]
            per_layer.append({
                "layer": L,
                "steps": s,
                "active_per_step": (a / s) if s else 0.0,
                "missing_per_step": (m / s) if s else 0.0,
                "miss_rate": (m / a) if a else 0.0,
                "fetched_per_step": (f / s) if s else 0.0,
            })
        return {"per_layer": per_layer}

    def decode_routing_stats(self) -> dict:
        """Per-layer decode routing concentration, for cache-skew analysis.

        Uses the histogram from ``collect_decode_freq``. The ``oracle_hit`` is the best a
        per-layer LRU holding ``cache_size/num_layers`` slots could achieve on the observed
        (stationary) routing distribution -- i.e. an upper bound on hit rate that depends
        purely on how skewed routing is, independent of any LRU/LFU dynamics.
        """
        freq = self.decode_freq.float()
        total = freq.sum(dim=1)
        valid = total > 0
        if int(valid.sum()) == 0:
            return {}
        slots_per_layer = self.cache_size / self.num_layers
        C = max(1, int(round(slots_per_layer)))
        sorted_f, _ = torch.sort(freq, dim=1, descending=True)
        oracle_hit = (sorted_f[:, :C].sum(dim=1)[valid] / total[valid]).mean().item()
        ws = (freq > 0).sum(dim=1).float()
        cdf = torch.cumsum(sorted_f, dim=1) / total.clamp(min=1).unsqueeze(1)
        cover90 = ((cdf < 0.9).sum(dim=1).float() + 1)[valid]
        p = freq / total.clamp(min=1).unsqueeze(1)
        ent = -(p * p.clamp(min=1e-12).log()).sum(dim=1)[valid]
        norm_ent = (ent / torch.log(torch.tensor(float(self.num_experts)))).mean().item()
        return {
            "slots_per_layer": slots_per_layer,
            "working_set_mean": ws[valid].mean().item(),
            "working_set_max": int(ws[valid].max().item()),
            "experts_for_90pct": cover90.mean().item(),
            "oracle_hit_at_slots": oracle_hit,
            "norm_entropy": norm_ent,
        }

    def copy_missing(self, plan: int = 0, blocks_per_bank: int | None = None) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        layer_id = self._pending_src_layer2 if plan else self._pending_src_layer
        assert layer_id is not None, "no staged misses (ensure_experts/materialize_layer first)"
        if layer_id in self._unpinned_layers:
            if not self._pending_whole_layer:
                raise RuntimeError(
                    f"layer {layer_id} is unpinned: its only copy is the whole-layer "
                    f"pageable materialize (position == expert id); ensure_experts's "
                    f"LRU slot remap cannot be honored without a device alias"
                )
            # the only copy a non-pinned layer ever needs is the non-overlap prefill materialize, which schedules the whole layer into slots [0, num_experts) with position == expert id -- a plain synchronous pageable H2D copy
            # never CUDA-graph captured: prefill is not captured, and decode never reaches this branch (it routes to the CPU executor)
            for per_layer, cache in self.banks:
                cache[: self.num_experts].copy_(per_layer[layer_id])
            return
        if self._copy_fused_ok:
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

            # One launch copies the missing rows for every bank (instead of one launch per
            # bank). evict_slots/src_indices/num_indices are shared across banks;
            # src_indices holds layer-local expert rows, resolved against this layer's
            # source pointers (layer_id is a static int per captured graph node).
            src_indices, evict_slots, num_indices = self._plan_buffers(plan)
            kwargs = {} if blocks_per_bank is None else {"blocks_per_bank": blocks_per_bank}
            fast_index_copy_multi_jit(
                self._copy_dst_ptrs,
                self._copy_src_ptrs[layer_id],
                self._copy_feat_bytes,
                evict_slots,
                src_indices,
                num_indices,
                **kwargs,
            )
            return

        from freetoken.kernel import fast_index_copy_jit

        for per_layer, cache in self.banks:
            fast_index_copy_jit(
                cache,
                self.evict_slots,
                per_layer[layer_id],
                self.src_indices,
                self.num_indices,
            )


def iter_offload_moe_layers(model) -> Iterator:
    from freetoken.layers import BaseOP, OffloadMoELayer

    if isinstance(model, OffloadMoELayer):
        yield model

    if not isinstance(model, BaseOP):
        return

    for value in model.__dict__.values():
        if isinstance(value, BaseOP):
            yield from iter_offload_moe_layers(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from iter_offload_moe_layers(item)


def attach_offload_moe_cache(model, cache: OffloadMoeCache) -> list:
    layers = list(iter_offload_moe_layers(model))
    for layer in layers:
        layer.offload_cache = cache
    return layers
