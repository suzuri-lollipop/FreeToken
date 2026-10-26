from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

import torch
from flashlib.kernels.slot_cache import N_STATS, Stat

# Fuse the per-bank expert copies into a single multi-bank launch (one per copy_missing
# instead of one per bank). Set FREETOKEN_FUSED_COPY=0 to force the legacy per-bank path
# (kept for A/B profiling). Falls back to per-bank automatically if a bank's row bytes or
# base address are not 16-byte aligned.
_FUSED_COPY = os.getenv("FREETOKEN_FUSED_COPY", "1").strip().lower() not in {"0", "false", "no", "off"}
_PAGEABLE_TIMING = os.getenv("FREETOKEN_PAGEABLE_TIMING", "0") not in ("", "0", "false")
_PREFILL_STAGING_ROWS = int(os.getenv("FREETOKEN_PREFILL_STAGING_ROWS", "64") or 64)

# cudaMemcpyBatchAsync silently degrades to a SYNCHRONOUS copy when a batch mixes
# large entries with sub-~256KB entries on registered host memory (H100 + CUDA 13.0,
# empirically bisected: a single 5-22KB entry beside one large entry blocks the
# calling thread for the full transfer; >=253KB entries never do). A synchronous
# call still moves bytes at full PCIe rate but stalls the host, which un-hides the
# GEMM under the copy in transition-zone workloads (gpt-oss 2048tok: -22% e2e).
# Banks whose rows are smaller than this ship as ONE whole-layer entry (their
# whole layer is tiny) and are excluded from the streaming path's hit gather, so
# every per-run entry the batch sees is >= this size. The on-demand path's gathers
# are SM copies, not batch memcpys, so they take every bank regardless.
_SMALL_BANK_FEAT_BYTES = 256 * 1024

from freetoken.kernel.fast_index_copy import default_blocks_per_bank
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.moe.offload_kernels import HybridRoutes

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
    # GGUF experts in their own ggml type per layer: gate_up [L*E, 2I, row_bytes] (gate rows, then up rows), down [L*E, H, row_bytes]; float experts keep their dtype
    "gguf": ("gate_up", "down"),
    # GGUF experts whose NVFP4 tensors carry per-expert scales: global [L*E, 4] fp32 (gate, up, down, 1) weight_scale_2
    "gguf_scaled": ("gate_up", "down", "global"),
    # native ModelOpt rows for the Triton inline-dequant kernels: packed e2m1 codes +
    # fp8-e4m3 per-16 block scales + per-output-row fp32 globals (w1/w3 carry distinct
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
    # (4 banks). Read by DeepSeek-V4's own DS-FP4 grouped GEMV kernels via pool_bank_views().
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
    "nvfp4": lambda H, I: 2 * I * (H // 2 + H // 16 + 4) + H * (I // 2 + I // 16 + 4),
    "mxfp4": lambda H, I: 2 * I * (H // 2 + H // 32 + 2) + H * (I // 2 + I // 32 + 2),
    "ds_fp4": lambda H, I: 2 * I * (H // 2 + H // 32) + H * (I // 2 + I // 32),
}

# vLLM's marlin grouped-GEMM hands the full [cache_size] slot cache as its expert
# dimension; moe_align_block_size requires round_up(experts, 32) < 1024, i.e. <= 992.
MARLIN_MAX_CACHE_SIZE = 992


def placeholder_bank(num_experts: int) -> torch.Tensor:
    """The source of a bank whose role a layer's expert kernel does not read (experts of several formats): zero bytes per row."""
    return torch.empty((num_experts, 0), dtype=torch.uint8)


def bank_layouts(sources: dict[str, list[torch.Tensor]], layer_tags: list | None = None) -> list[tuple]:
    """Each layer's layout key: its quant tag and every bank's (dtype, row shape). A slot pool holds layers of one key."""
    num_layers = len(next(iter(sources.values())))
    tags = layer_tags if layer_tags is not None else [None] * num_layers
    assert len(tags) == num_layers, (len(tags), num_layers)
    return [(tags[l], tuple((t[l].dtype, tuple(t[l].shape[1:])) for t in sources.values())) for l in range(num_layers)]


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
    # of re-crossing PCIe; only the misses are H2D'd (one CUDA/HIP batch memcpy of
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
    # hybrid only, one per MoE layer: when > 0, replaces the fixed cap with a per-step fraction -- fetch
    # ~fraction * misses experts over PCIe (rounded to whichever integer balances the
    # overlap best), the CPU computes the rest. The engine sets it to the benched
    # pcie_bw / cpu_bw ratio so the PCIe fetch and the CPU overflow GEMV take equal
    # time (perfect overlap): fetched : cpu = pcie : cpu - pcie.
    hybrid_fetch_fraction: list[float] | None = None
    # hybrid only: smallest decode batch that routes misses through the CPU executor.
    # Below it the per-layer submit/sync handshake costs more than the PCIe it saves --
    # a warm bs=1 step misses only ~1 expert/layer, so the GPU slot cache serves it
    # faster alone. The engine raises this to 2 under TP (measured: bs=1 hybrid decodes
    # slower than pure offload there); CUDA graphs bake the branch per captured batch size.
    hybrid_min_bs: int = 1
    # bank layout from the expert kernel (a BankSpec per role); when given it replaces the _BANK_SCHEMAS lookup and the slot cap comes from max_slots
    layout: dict | None = None
    max_slots: int | None = None
    # Per-layer kernel layouts for experts of several formats (replaces ``layout``). The banks are the union of their roles;
    # a layer whose kernel reads no such role carries a zero-byte [num_experts, 0] uint8 placeholder (see placeholder_bank).
    layer_layouts: list[dict] | None = None
    # Each layer belongs to one pool: a contiguous slot range with its own LRU scope; pool p owns slots [sum(pool_sizes[:p]), +pool_sizes[p]). Slot ids in slot_for_id are local to the pool. None = one pool for all layers.
    pool_of_layer: list[int] | None = None
    pool_sizes: list[int] | None = None

    def __post_init__(self) -> None:
        policy_ids = {"lru": 0}
        assert self.cache_policy in policy_ids
        assert self.decode_target in ("gpu", "cpu", "hybrid"), self.decode_target
        if self.layer_layouts is not None:
            assert self.layout is None, "pass layout or layer_layouts, not both"
            assert len(self.layer_layouts) == self.num_layers, (len(self.layer_layouts), self.num_layers)
            if any(spec.resident for layout in self.layer_layouts for spec in layout.values()):
                raise ValueError("experts of several formats cannot use a kernel with GPU-resident alphas (marlin, b12x)")
        elif self.layout is None:
            assert self.quant_format in _BANK_SCHEMAS, f"unknown quant_format {self.quant_format!r}"
        if self.pool_sizes is None:
            self.pool_of_layer = [0] * self.num_layers
            self.pool_sizes = [self.cache_size]
        self.pool_of_layer = list(self.pool_of_layer)
        self.pool_sizes = list(self.pool_sizes)
        assert len(self.pool_of_layer) == self.num_layers, (len(self.pool_of_layer), self.num_layers)
        assert sorted(set(self.pool_of_layer)) == list(range(len(self.pool_sizes))), self.pool_of_layer
        assert sum(self.pool_sizes) == self.cache_size and min(self.pool_sizes) > 0, (self.pool_sizes, self.cache_size)
        # Attached by the engine for decode_target == "cpu" (CpuMoeExecutor); None
        # for the GPU decode path.
        self.cpu_executor = None
        # MoE layer ids whose decode runs on the CPU executor; the rest use the GPU
        # offload/PCIe path. Set by the engine after construction (empty = all-GPU,
        # all layers = the plain --moe-strategy cpu case).
        self.cpu_layer_ids: frozenset = frozenset()
        # hybrid: locked (unpinned) layers that keep slots of their own pools and fetch part of their misses through locked_fetch's pinned staging
        self.locked_hybrid: frozenset = frozenset()
        self.locked_fetch = None  # moe.locked_fetch.LockedLayerFetch, set by the engine with locked_hybrid
        # Bank row layouts, set by set_bank_sources: layouts[g][k] is (dtype, row shape) of bank k (layouts of two quant tags may be equal), layer_layout[l] is layer l's g.
        # With one (dtype, row shape) per bank a bank cache is typed [cache_size, *row]; with several, or with pads, it is a byte segment each pool views with its own layout.
        self.layouts: list[tuple[tuple[torch.dtype, tuple[int, ...]], ...]] = []
        self.layer_layout: list[int] = [0] * self.num_layers
        # _pad_bytes[g][k]: zeroed bytes after the last row of every pool and staging buffer of layout g in bank k, for kernels that read past it (16-aligned)
        self._pad_bytes: list[list[int]] = []
        self._raw_banks = False
        # num_experts floor + nvfp4_marlin slot cap, shared with the runtime-rebuild path.
        self.validate_rebuild(self.cache_size, self.pool_sizes)
        assert not self.prefill_overlap or self.cache_size >= 2 * self.num_experts, (
            "Prefill overlap borrows two full expert-layer buffers from the unified MoE "
            "cache, so cache_size must be at least 2 * num_experts "
            "(raise moe_cache_size or disable moe_prefill_overlap)"
        )
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
        if self.layer_layouts is not None:
            self.bank_schema = tuple(dict.fromkeys(
                role for layout in self.layer_layouts for role, spec in layout.items() if not spec.resident
            ))
        elif self.layout is not None:
            self.bank_schema = tuple(role for role, spec in self.layout.items() if not spec.resident)
        else:
            self.bank_schema = _BANK_SCHEMAS[self.quant_format]
        self.bank_sources: dict[str, list[torch.Tensor]] = {}
        self.bank_caches: dict[str, torch.Tensor] = {}
        # per-layer host residency: the GPU movement paths require "pinned"; LOCKED/PAGEABLE layers decode on the CPU executor (locked_hybrid ones also from their slots) and prefill via copy_missing's pageable branch
        # _unpinned_layers is the derived id set the hot paths test against
        self._file_banks: dict = {}  # disk-tier HostBanks by role (set_bank_sources)
        self._prefill_pinned_staging: dict = {}  # per-bank pinned pread blocks (lazy)
        # GPU-only decode (FREETOKEN_GPU_ONLY_DECODE): a dedicated VRAM buffer holding the
        # cold routes' staged rows, lazily allocated; the cold GEMM reads from it so the
        # CPU executor never runs. Capped at the host staging depth (bs*top_k cold routes).
        self._cold_stage_vram: dict | None = None
        self._cold_stage_cap = _PREFILL_STAGING_ROWS
        # hybrid-decode expert skip (0.0 = off; see _cpu_skip_mask)
        self.cpu_skip_minw: float = 0.0
        self.cpu_skip_keep: int = 2
        self._pageable_timing = {"item_ms": 0.0, "rows_ms": 0.0, "read_ms": 0.0, "h2d_ms": 0.0, "layers": 0, "rows": 0}
        # expert-profile hot copies by layer/role (set_bank_sources -> cpu executor)
        self.hot_bank_sources: dict[int, dict[str, tuple[torch.Tensor, dict[int, int]]]] | None = None
        # per layer, the hot compact's row bytes per bank (the compacts are checkpoint-dtype rows)
        self._hot_feat_bytes: dict[int, torch.Tensor] = {}
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
        # the kernel, which accumulates in the same launch. The stat_* tensors below are
        # the hybrid path's: views of one buffer its kernel accumulates into the same way.
        self.lru_stats = torch.zeros(
            (self.num_layers, N_STATS), dtype=torch.int64, device=self.device
        )
        # (missing, fetched, active, calls), then the same per layer
        stat_buf = torch.zeros(4 * (self.num_layers + 1), dtype=torch.int64, device=self.device)
        self.hybrid_stats = (stat_buf[:4], stat_buf[4:])
        # hybrid only: stat_fetched counts the experts actually fetched over PCIe (<= stat_missing).
        # The CPU computes stat_missing - stat_fetched of them.
        self.stat_missing, self.stat_fetched, self.stat_active, self.stat_calls = stat_buf[:4].unbind()
        # Per-layer counterparts of the scalars above (indexed by MoE-layer id). Same
        # device-side accumulation (graph-safe: layer_id is a static index per graph node),
        # so one req's per-layer miss rate is readable via decode_miss_stats_per_layer().
        per_layer = stat_buf[4:].view(self.num_layers, 4)
        self.stat_missing_layer, self.stat_fetched_layer = per_layer[:, 0], per_layer[:, 1]
        self.stat_active_layer, self.stat_steps_layer = per_layer[:, 2], per_layer[:, 3]
        # Opt-in decode routing histogram (per layer, per expert) for cache-skew
        # analysis. Accumulated in ``ensure_experts`` from the raw expert ids before the
        # kernel rewrites them to slots. Only accurate with CUDA graphs disabled (the
        # captured graph would not re-run this host-side scatter on replay).
        self.collect_decode_freq = False
        self.decode_freq = torch.zeros(
            (self.num_layers, self.num_experts), dtype=torch.int64, device=self.device
        )
        # learned-profile dump sink + step counter (see _profile_dump_step)
        self._profile_out: str | None = None
        self._profile_steps = 0
        # (per-layer sources, cache) per bank, in schema order. Every piece of cache
        # machinery that moves bank bytes (copy_missing, the prefill double buffers,
        # the pool and staging views) iterates this list, so the slot cache is bank-count agnostic.
        self.banks: list[tuple[list[torch.Tensor], torch.Tensor]] = []
        # Fused multi-bank copy descriptor (built by set_bank_sources/_build_copy_plan).
        # Source pointers are per layer (_copy_src_ptrs[layer_id] -> [num_banks] device
        # tensor); dst pointers are per pool and row bytes per layout.
        self._copy_fused_ok = False
        self._copy_blocks_per_bank = 8
        self._copy_dst_ptrs: torch.Tensor | None = None
        self._copy_dst_ptrs_pool: list[torch.Tensor] | None = None
        self._copy_src_ptrs: list[torch.Tensor] | None = None
        self._copy_feat_bytes: list[torch.Tensor] | None = None
        # The layer whose misses ensure_experts/materialize_layer staged last; consumed
        # by copy_missing to pick the per-layer source (part of the same pending-copy
        # state as evict_slots/src_indices/num_indices).
        # _pending_whole_layer records WHICH staged it: the pageable branch is only sound after materialize_layer
        self._pending_src_layer: int | None = None
        self._pending_whole_layer = False
        self._pending_fetch_cap: int | None = None  # the last hybrid ensure's cap on its fetch count
        # The two prefill staging buffers are the first 2 * _stage_bytes[k] bytes of every bank, on top of the first pools.
        # _stage_views[g][b] are buffer b's per-bank [num_experts, ...] views for layout g; _inval[b] are the global slots it overwrites.
        self._stage_bytes: list[int] = []
        self._stage_views: list[list[tuple[torch.Tensor, ...] | None]] = []
        # _stage_tails[g][b]: the pad after buffer b's rows in layout g
        self._stage_tails: list[list[list[torch.Tensor]]] = []
        self._inval: list[slice | torch.Tensor] = []
        # the split of _inval[b] by which of its slots may take the step+1 stamp (see _slice_pools)
        self._inval_deprio: list[slice | torch.Tensor] = []
        self._inval_evict: list[slice | torch.Tensor] = []
        self.prefill_copy_stream: torch.cuda.Stream | None = None
        self.prefill_begin_event: torch.cuda.Event | None = None
        self.prefill_ready_events: list[torch.cuda.Event] = []
        self.prefill_release_events: list[torch.cuda.Event] = []
        self._prefill_buffer_layer: list[int | None] = [None, None]
        self._prefill_buffer_released: list[bool] = [True, True]
        self._prefill_buffer_has_release_event: list[bool] = [False, False]
        # hit-D2D split state: pinned begin-of-chunk snapshot of slot_for_id (the
        # classification input; frozen for the chunk -- no decode runs inside one,
        # and buffer invalidation only clears slots under the staging buffers, which classify as
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
        self._slice_pools()

    def _pool_layout(self, pool: int) -> int:
        return self.layer_layout[self.pool_of_layer.index(pool)]

    def _bank_rows(self, k: int, layout: int, offset: int, nrows: int) -> torch.Tensor:
        """``nrows`` rows of bank ``k`` in ``layout``, starting ``offset`` bytes into the bank cache."""
        cache = self.banks[k][1]
        row = self._row_bytes[layout][k]
        if not self._raw_banks:
            first = offset // row
            return cache if first == 0 and nrows == cache.shape[0] else cache[first : first + nrows]
        dtype, shape = self.layouts[layout][k]
        return cache[offset : offset + nrows * row].view(dtype).view(nrows, *shape)

    def _bank_bytes(self, k: int, pool_sizes: list[int]) -> int:
        """Bytes of raw bank ``k``: each pool's slots and pad, plus the largest pad, so two staging buffers fit wherever two layers of slots do."""
        layouts = [self._pool_layout(p) for p in range(len(pool_sizes))]
        slots = sum(s * self._row_bytes[g][k] + self._pad_bytes[g][k] for g, s in zip(layouts, pool_sizes))
        return slots + max(pads[k] for pads in self._pad_bytes)

    def _stage_fits(self, buffers: int) -> bool:
        return all(c.numel() * c.element_size() >= buffers * s for (_, c), s in zip(self.banks, self._stage_bytes))

    def _slice_pools(self) -> None:
        E = self.num_experts
        bases = [0]
        for size in self.pool_sizes[:-1]:
            bases.append(bases[-1] + size)
        self.pool_bases = bases
        spans = list(zip(bases, self.pool_sizes))

        def cut(t: torch.Tensor, b: int, s: int) -> torch.Tensor:
            return t if s == t.shape[0] else t[b : b + s]

        self._pool_id_of_slot = [cut(self.id_of_slot, b, s) for b, s in spans]
        self._pool_usage = [cut(self.usage, b, s) for b, s in spans]
        # rows[k][p]: bytes of one slot of pool p in bank k; before the banks exist, one unit per slot with a stage of E
        if self.banks:
            rows = [[self._row_bytes[self._pool_layout(p)][k] for p in range(len(spans))] for k in range(len(self.banks))]
            pads = [[self._pad_bytes[self._pool_layout(p)][k] for p in range(len(spans))] for k in range(len(self.banks))]
            stage = self._stage_bytes
        else:
            rows, pads, stage = [[1] * len(spans)], [[0] * len(spans)], [E]
        offsets = []
        for row, pad in zip(rows, pads):
            offsets.append([0])
            for s, r, d in zip(self.pool_sizes[:-1], row, pad):
                offsets[-1].append(offsets[-1][-1] + s * r + d)
        self._pool_offsets = offsets
        # local slots below this index overlap the prefill staging area in some bank; a pool's zero-byte rows (a role its kernel does not read) overlap nothing
        self._pool_stable_from = [
            max((min(max(-(-(2 * stage[k] - offsets[k][p]) // rows[k][p]), 0), s) for k in range(len(rows)) if rows[k][p]), default=0)
            for p, s in enumerate(self.pool_sizes)
        ]

        def compact(mask: torch.Tensor) -> slice | torch.Tensor:
            idx = mask.nonzero().flatten()
            if idx.numel() == 0 or int(idx[-1] - idx[0]) + 1 == idx.numel():
                return slice(int(idx[0]), int(idx[-1]) + 1) if idx.numel() else slice(0, 0)
            return idx.to(self.device)

        # A borrowed slot is stamped step+1 so decode's LRU pick prefers the pool's own
        # regular slots (see _invalidate_prefill_buffer), but lru_ensure drops usage == step
        # from its victim argmin: stamp more of a pool than size - num_experts of them and an
        # ensure with that many misses has NO victim, and argmin over an all-max array returns
        # index 0 on every iteration, so all of its misses collapse onto one slot. One ensure
        # misses at most num_experts distinct ids, so capping the stamp there always leaves it
        # a victim per miss; the borrowed slots above the cap stay evict-first.
        depriable = torch.zeros(self.cache_size, dtype=torch.bool)
        for base, s in spans:
            depriable[base : base + max(s - E, 0)] = True
        self._inval, self._inval_deprio, self._inval_evict = [], [], []
        for b in range(2):
            mask = torch.zeros(self.cache_size, dtype=torch.bool)
            for k in range(len(rows)):
                lo, hi = b * stage[k], (b + 1) * stage[k]
                for p, (base, s) in enumerate(spans):
                    if not rows[k][p]:
                        continue
                    first = max((lo - offsets[k][p]) // rows[k][p], 0)
                    end = min(-(-(hi - offsets[k][p]) // rows[k][p]), s)
                    mask[base + first : base + max(first, end)] = True
            self._inval.append(compact(mask))
            self._inval_deprio.append(compact(mask & depriable))
            self._inval_evict.append(compact(mask & ~depriable))
        self._stage_dirty = [False, False]
        if not self.banks:
            self._pool_bank_views, self._stage_views, self._stage_tails = [], [], []
            return
        self._pool_bank_views = [
            tuple(self._bank_rows(k, self._pool_layout(p), offsets[k][p], s) for k in range(len(self.banks)))
            for p, s in enumerate(self.pool_sizes)
        ]
        self._stage_views = [
            [
                tuple(self._bank_rows(k, g, b * stage[k], E) for k in range(len(self.banks))) if self._stage_fits(b + 1) else None
                for b in range(2)
            ]
            for g in range(len(self.layouts))
        ]
        self._stage_tails = [
            [[self.banks[k][1][b * stage[k] + E * row[k] :][: pad[k]] for k in range(len(self.banks)) if pad[k] and self._stage_fits(b + 1)] for b in range(2)]
            for row, pad in zip(self._row_bytes, self._pad_bytes)
        ]

    @property
    def _gather_dst_ptrs(self):
        """Per-layout destination pointers of the banks the D2D hit gather serves (None: no gather).

        Empty list when the fused copy plan could not be built or no bank is wide enough.
        """
        return self._gather_stage_ptrs

    @property
    def prefill_bank_buffers(self) -> list[list[torch.Tensor]]:
        """Per bank, ``[buffer_id]`` -> that bank's ``[num_experts, ...]`` staging view of the cache's own layout.

        Empty when the slot cache cannot lend the staging slots, which is what the
        on-demand path checks before planning.
        """
        if not self._stage_views or not self._stage_fits(1):
            return []
        views = self._stage_views[0]
        # A cache between num_experts and 2 * num_experts slots lends ONE buffer, which is
        # all the on-demand path borrows (it has no second layer in flight); the second
        # entry is None there, so it must not be indexed.
        n = sum(v is not None for v in views)
        return [[views[b][k] for b in range(n)] for k in range(len(self.banks))]

    def stage_views(self, layer_id: int, buffer_id: int) -> tuple[torch.Tensor, ...]:
        """Per-bank ``[num_experts, ...]`` views of prefill staging buffer ``buffer_id`` in ``layer_id``'s layout; row == expert id."""
        views = self._stage_views[self.layer_layout[layer_id]][buffer_id]
        assert views is not None, f"the slot cache is too small for staging buffer {buffer_id}"
        return views

    def pool_state(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(id_of_slot, usage)`` views of the pool that serves ``layer_id``."""
        p = self.pool_of_layer[layer_id]
        return self._pool_id_of_slot[p], self._pool_usage[p]

    def pool_slots(self, layer_id: int) -> int:
        return self.pool_sizes[self.pool_of_layer[layer_id]]

    def pool_bank_views(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        """Per-bank slot views of the pool that serves ``layer_id``, indexed by its local slot ids."""
        assert self.banks, "set_bank_sources must register the banks first"
        return self._pool_bank_views[self.pool_of_layer[layer_id]]

    def bank_views(self, n: int | None = None) -> tuple[torch.Tensor, ...]:
        """Per-bank cache views in registration order: the whole slot cache, or its first ``n`` slots.

        The multi-pool counterpart of :meth:`pool_bank_views`: a caller that addresses rows
        by a pool-independent index (a materialized layer's expert id, the fused-copy
        descriptor tables) takes this one.
        """
        assert self.banks, "set_bank_sources must register the banks first"
        if n is None:
            return tuple(cache for _, cache in self.banks)
        return tuple(cache[:n] for _, cache in self.banks)

    def set_bank_sources(
        self,
        sources: dict[str, list[torch.Tensor]],
        layer_residency: list[str] | None = None,
        file_banks: dict[str, list] | None = None,
        hot: dict[int, dict[str, tuple[torch.Tensor, dict[int, int]]]] | None = None,
        layer_tags: list | None = None,
        row_tail_bytes: dict[str, list[int]] | None = None,
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
        DISK labels behave like PAGEABLE here (their banks are mmap'd shard files, no
        device address until something pins them); ``file_banks`` carries their
        ``HostBank`` objects for readahead prefetch (see :meth:`materialize_layer`).

        ``hot`` is the expert profile's compact pinned copies of DISK layers' hot
        experts ({layer: {role: (tensor, slot map)}}); stored on
        ``hot_bank_sources`` for the CPU executor's per-(layer, expert) table.
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
                    "when any layer is LOCKED/PAGEABLE/DISK (the engine does this)"
                )
        self._unpinned_layers = unpinned
        self.layer_residency = list(residency)
        # disk-tier banks by role -> [HostBank | None per layer]; the prefill path
        # fadvise-WILLNEEDs the next layer's banks while this one streams in
        self._file_banks = {canonical_role(n): per_layer for n, per_layer in (file_banks or {}).items()}
        self.hot_bank_sources = hot
        # per-layer [num_experts] int32 compact-row map (-1 = cold, CPU-only), or None
        # when every row is layer-local (pinned banks) -- the hybrid fetch kernel and its
        # CPU mirror read it to decide which misses can DMA at all
        self._hot_row_of: list[torch.Tensor | None] = self._build_hot_row_of(hot)
        for name in self.bank_schema:
            per_layer = sources[name]
            assert len(per_layer) == self.num_layers, (name, len(per_layer))
            for layer_id, source in enumerate(per_layer):
                assert source.is_contiguous(), f"bank {name!r} layer {layer_id} must be contiguous"
                assert source.size(0) == self.num_experts, (name, layer_id, source.shape)
                spec = self.layout[name] if self.layout is not None else None
                if self.layer_layouts is not None:
                    spec = self.layer_layouts[layer_id].get(name)
                    if spec is None and source.numel() != 0:
                        raise ValueError(f"layer {layer_id}'s expert kernel reads no {name!r} bank, so it takes a zero-byte placeholder")
                if spec is not None:
                    if tuple(source.shape[1:]) != tuple(spec.shape) or source.dtype != spec.dtype:
                        raise ValueError(
                            f"bank {name!r} rows are {tuple(source.shape[1:])} {source.dtype} but the expert kernel's layout "
                            f"wants {tuple(spec.shape)} {spec.dtype}; the banks were packed for another kernel"
                        )
            self.bank_sources[name] = list(per_layer)
        keys = bank_layouts(sources, layer_tags)
        unique = list(dict.fromkeys(keys))
        self.layouts = [banks for _, banks in unique]
        self.layer_layout = [unique.index(key) for key in keys]
        for p in range(len(self.pool_sizes)):
            mixed = {self.layer_layout[l] for l, q in enumerate(self.pool_of_layer) if q == p}
            assert len(mixed) == 1, f"slot pool {p} holds layers of {len(mixed)} layouts; a pool needs one"
        self._pad_bytes = [[0] * len(self.bank_schema) for _ in self.layouts]
        if row_tail_bytes is not None:
            tails = {canonical_role(name): per_layer for name, per_layer in row_tail_bytes.items()}
            for k, name in enumerate(self.bank_schema):
                for layer_id, tail in enumerate(tails[canonical_role(name)]):
                    g = self.layer_layout[layer_id]
                    self._pad_bytes[g][k] = max(self._pad_bytes[g][k], -(-tail // 16) * 16)
        self._raw_banks = len(set(self.layouts)) > 1 or any(map(any, self._pad_bytes))
        self._row_bytes = [[math.prod(shape) * dtype.itemsize for dtype, shape in layout] for layout in self.layouts]
        if self._raw_banks:
            # every pool view and staging buffer starts on a row boundary of its own layout
            assert all(r % 16 == 0 for rows in self._row_bytes for r in rows), self._row_bytes
        self._stage_bytes = [
            self.num_experts * max(rows[k] for rows in self._row_bytes) + max(pads[k] for pads in self._pad_bytes)
            for k in range(len(self.bank_schema))
        ]
        self._alloc_banks()
        if not self._stage_fits(1):
            raise ValueError(f"MoE cache of {self.cache_size} slots cannot stage one whole layer of experts")
        if self.prefill_overlap and not self._stage_fits(2):
            logger.warning("Disabling MoE prefill overlap: the slot cache cannot hold two whole layers of experts")
            self.prefill_overlap = False
        self._slice_pools()
        self._build_copy_plan()
        if self.prefill_overlap or (
            self.device.type == "cuda" and self.cache_size >= self.num_experts
        ):
            # The on-demand prefill path stages routed rows into the double buffers
            # even when the whole-layer overlap path is off (a disk-tier split has no
            # registered banks to DMA whole layers from, but its touched rows still
            # cross PCIe pageably); the buffers are slot-cache views, so building them
            # costs one zeroing pass. A cache too small to lend 2 * num_experts slots
            # (the cpu backend's pinned geometry) keeps the old no-buffer state.
            self._init_prefill_overlap_buffers()

    def _alloc_banks(self) -> None:
        for k, name in enumerate(self.bank_schema):
            if self._raw_banks:
                self.bank_caches[name] = torch.zeros(self._bank_bytes(k, self.pool_sizes), dtype=torch.uint8, device=self.device)
            else:
                dtype, shape = self.layouts[0][k]
                self.bank_caches[name] = torch.zeros((self.cache_size, *shape), dtype=dtype, device=self.device)
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]

    def _build_hot_row_of(self, hot) -> list:
        """One [num_experts] int32 row map per layer from the profile's compact slot
        maps. None for pinned layers (row == expert id) and for disk layers without a
        compact (nothing fetchable). The kernel consumes rows for the copy plan, so a
        disk layer with a map joins the GPU slot LRU through its registered compact."""
        maps: list[torch.Tensor | None] = []
        for layer_id in range(self.num_layers):
            roles = (hot or {}).get(layer_id)
            if roles is None or layer_id not in self._unpinned_layers:
                maps.append(None)
                continue
            slot = None
            for name in self.bank_schema:
                entry = roles.get(name)
                if entry is None:
                    continue
                this = entry[1]  # already {expert: compact row}
                if slot is None:
                    slot = this
                elif slot != this:
                    raise ValueError(
                        f"layer {layer_id}: hot compacts disagree on membership across "
                        f"banks ({name}); the fetch row map must be bank-independent"
                    )
            if not slot:
                maps.append(None)
                continue
            m = torch.full((self.num_experts,), -1, dtype=torch.int32)
            for e, i in slot.items():
                m[e] = i
            maps.append(m.to(self.device))
        return maps

    def _build_copy_plan(self) -> None:
        self._build_fused_copy_plan()
        if self._copy_fused_ok or self.device.type != "cuda" or not self.banks:
            if not self._copy_fused_ok and self.device.type == "cuda" and self.banks:
                # Per-bank copies silently serialize the in-stream batch gather; log the
                # fallback so a degraded copy is visible in the startup log.
                logger.warning_rank0(
                    "MoE fused multi-bank copy disabled; copy_missing falls back to the "
                    "per-bank path (alignment or FREETOKEN_FUSED_COPY=0)"
                )
            return
        for k, name in enumerate(self.bank_schema):
            for rows in self._row_bytes:
                if rows[k] % 128:
                    raise RuntimeError(
                        f"MoE bank {name!r} rows are {rows[k]} bytes (not a multiple of 128): "
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
        self._copy_dst_ptrs_pool = None
        self._copy_src_ptrs = None
        self._copy_feat_bytes = None
        self._copy_dst_ptrs_host: list[int] = []
        self._copy_src_ptrs_host: list[list[int]] = []
        self._gather_bank_ids: list[list[int]] = []
        self._gather_feat_bytes: list[torch.Tensor | None] = []
        self._gather_stage_ptrs: list[list[torch.Tensor] | None] = []
        self._gather_pool_ptrs: list[torch.Tensor | None] = []
        if not _FUSED_COPY or self.device.type != "cuda" or not self.banks:
            return
        from freetoken.kernel.pinned import device_ptr

        aligned = [self._stage_bytes] + self._row_bytes + self._pool_offsets
        if any(n % 16 for ns in aligned for n in ns) or any(c.data_ptr() % 16 for _, c in self.banks):
            return  # leave fused disabled; copy_missing uses the per-bank path
        dst_ptrs = [cache.data_ptr() for _, cache in self.banks]
        layer_src_ptrs = [[] for _ in range(self.num_layers)]
        for name, (per_layer, cache) in zip(self.bank_schema, self.banks):
            feat = math.prod(per_layer[0].shape[1:]) * per_layer[0].element_size()
            if feat % 16 != 0 or cache.data_ptr() % 16 != 0:
                return  # leave fused disabled; copy_missing uses the per-bank path
            for layer_id, source in enumerate(per_layer):
                if layer_id in self._unpinned_layers:
                    if self._hot_row_of[layer_id] is not None:
                        # a compact-backed disk layer: its hot rows ARE registered (the
                        # loader pinned the compact), so the fused copy DMAs them into
                        # slots exactly like a pinned layer's rows; src rows are compact
                        # rows (see _hot_row_of)
                        src_dev = device_ptr(self.hot_bank_sources[layer_id][name][0])
                        if src_dev % 16 != 0:
                            return
                        layer_src_ptrs[layer_id].append(src_dev)
                        continue
                    # unregistered layer: no device alias exists, and the row is never
                    # consumed (CPU decode; pageable prefill); a 0 placeholder keeps the
                    # descriptor shape
                    layer_src_ptrs[layer_id].append(0)
                    continue
                if source.numel() == 0:
                    layer_src_ptrs[layer_id].append(0)  # placeholder bank: zero bytes per row, never dereferenced
                    continue
                # The kernel dereferences these on the GPU, so store each host bank's
                # device alias (== data_ptr() under UVA identity; differs on
                # Windows/WDDM).
                src_dev = device_ptr(source)
                if src_dev % 16 != 0:
                    return
                layer_src_ptrs[layer_id].append(src_dev)

        def ptrs(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int64, device=self.device)

        self._copy_dst_ptrs = ptrs(dst_ptrs)
        self._copy_dst_ptrs_pool = [
            self._copy_dst_ptrs if all(o[p] == 0 for o in self._pool_offsets)
            else ptrs([d + o[p] for d, o in zip(dst_ptrs, self._pool_offsets)])
            for p in range(len(self.pool_sizes))
        ]
        self._copy_src_ptrs = [ptrs(values) for values in layer_src_ptrs]
        self._copy_feat_bytes = [ptrs(rows) for rows in self._row_bytes]
        self._copy_dst_ptrs_host = dst_ptrs
        self._copy_src_ptrs_host = layer_src_ptrs
        # The STREAMING path's hit-D2D gather serves only the big banks; small banks are
        # whole-layer H2D entries there (see _SMALL_BANK_FEAT_BYTES), so their rows never
        # need D2D. The gather copies cache rows of the layer's pool into its staging
        # buffer, row == expert id.
        self._gather_bank_ids = [[k for k, r in enumerate(rows) if r >= _SMALL_BANK_FEAT_BYTES] for rows in self._row_bytes]
        for g, ids in enumerate(self._gather_bank_ids):
            self._gather_feat_bytes.append(ptrs([self._row_bytes[g][k] for k in ids]) if ids else None)
            self._gather_stage_ptrs.append(
                [ptrs([dst_ptrs[k] + b * self._stage_bytes[k] for k in ids]) for b in range(2)] if ids else None
            )
        for p in range(len(self.pool_sizes)):
            ids = self._gather_bank_ids[self._pool_layout(p)]
            self._gather_pool_ptrs.append(ptrs([dst_ptrs[k] + self._pool_offsets[k][p] for k in ids]) if ids else None)
        self._copy_blocks_per_bank = default_blocks_per_bank(self.device)
        self._copy_fused_ok = True

    def validate_rebuild(self, cache_size: int, pool_sizes: list[int] | None = None) -> None:
        """Pure geometry validation of a rebuild target (no GPU side effects).

        Raises ``ValueError`` if ``cache_size`` is below the ``num_experts`` floor or
        a pool is above the expert kernel's slot cap, or the bank byte segments cannot stage one layer. Called by :meth:`rebuild` and by the engine's
        pre-teardown check, so an invalid target rejects with the old cache intact
        (no destructive free first).
        """
        if cache_size < self.num_experts:
            raise ValueError(f"cache_size {cache_size} < num_experts {self.num_experts}")
        # decode kernels see one pool's slots, so the cap applies per pool
        largest = cache_size if pool_sizes is None else max(pool_sizes)
        what = f"moe_cache_size={cache_size}" if largest == cache_size else f"a MoE slot pool of {largest} slots"
        if self.max_slots is not None and largest > self.max_slots:
            raise ValueError(
                f"{what} exceeds the expert kernel's slot limit of {self.max_slots}; "
                f"pass --moe-cache-size {self.max_slots} or less, or let the default kernel serve the experts"
            )
        if self.layout is None and self.quant_format == "nvfp4_marlin" and largest > MARLIN_MAX_CACHE_SIZE:
            raise ValueError(
                f"{what} exceeds the marlin backend's slot limit of "
                f"{MARLIN_MAX_CACHE_SIZE} (vLLM moe_align_block_size caps padded experts at "
                "1024); reduce moe_cache_size or force --quant-backend moe.nvfp4=triton"
            )
        if self._raw_banks and pool_sizes is not None:
            for k, stage in enumerate(self._stage_bytes):
                if self._bank_bytes(k, pool_sizes) < stage:
                    raise ValueError(f"moe_cache_size={cache_size} cannot stage one whole layer of experts")

    def rebuild(self, cache_size: int, pool_sizes: list[int] | None = None) -> None:
        """Resize the GPU slot cache + bookkeeping to ``cache_size`` IN PLACE.

        Keeps the CPU/pinned ``bank_sources`` and the GPU-resident alphas; never
        reloads banks. Tears down prefill-overlap buffers first (their views alias
        the old ``bank_caches``), frees the old GPU tensors, then reallocates. Slots
        cold-start after rebuild. Object identity is preserved so attached layers and
        ``ctx.moe_offload_cache`` stay valid.
        """
        assert self.bank_sources, "set_bank_sources must run before rebuild"
        sizes = [cache_size] if pool_sizes is None else list(pool_sizes)
        assert len(sizes) == len(self.pool_sizes) and sum(sizes) == cache_size and min(sizes) > 0, (sizes, cache_size)
        self.validate_rebuild(cache_size, sizes)
        # 1. Tear down prefill-overlap (its buffer views alias the old bank_caches).
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
        self._pool_bank_views = []
        self._stage_views = []
        self._stage_tails = []
        self.cache_size = cache_size
        self.pool_sizes = sizes
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        # 3. Reallocate the slot cache from the retained host sources.
        self._alloc_banks()
        # 4. Reallocate cache_size-shaped bookkeeping; reset the slot map (cold start).
        self.slot_for_id.fill_(-1)
        self.id_of_slot = torch.full((cache_size,), -1, dtype=torch.int32, device=self.device)
        self.usage = torch.zeros((cache_size,), dtype=torch.int64, device=self.device)
        self._slice_pools()
        self._build_copy_plan()  # slot caches were reallocated -> refresh fused-copy addrs
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
        # 5. Re-evaluate prefill overlap against the new size.
        if self.prefill_overlap and not self._stage_fits(2):
            logger.warning(
                f"Disabling MoE prefill overlap on rebuild: cache_size {cache_size} "
                f"cannot hold two whole layers of experts (2*num_experts {2 * self.num_experts} slots of one layout)."
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
        return layer_id in self.cpu_layer_ids and layer_id not in self.locked_hybrid

    def is_unpinned_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id``'s host banks have no device address (LOCKED/PAGEABLE): the GPU slot-gather paths cannot serve it.
        ``copy_missing`` takes the whole-layer pageable branch, which presumes materialize's position == expert id; a locked_hybrid layer's decode fetches go through ``locked_fetch`` instead."""
        return layer_id in self._unpinned_layers

    def alphas_for_slots(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-slot global scales for a decode call, or ``None`` when the format
        keeps no GPU-resident alphas (bf16 / triton-nvfp4). Slots of other layers
        yield garbage values, but only slots routed to -- and those belong to
        ``layer_id`` -- are ever read by the grouped GEMM."""
        if self.gate_up_alpha is None:
            return None
        id_of_slot, _ = self.pool_state(layer_id)
        idx = layer_id * self.num_experts + (
            id_of_slot.clamp(min=0).long() % self.num_experts
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

    def _init_prefill_overlap_buffers(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        # one staging buffer per layer in flight: 2 when the whole-layer overlap copy
        # runs ahead of the previous layer's GEMMs, 1 otherwise
        self._prefill_num_buffers = 2 if self.prefill_overlap else 1
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]
        # The double buffers are the staging views (_slice_pools), on top of the first pools'
        # slots; the bank caches are zero-filled, so a row no gather ever staged reads as a
        # zero expert: the hybrid prefill split redirects a CPU-assigned route there with
        # weight 0. ``prefill_bank_buffers`` exposes them to the on-demand planners.
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

    def _invalidate_prefill_buffer(self, buffer_id: int, layer_id: int) -> None:
        """Take buffer ``buffer_id``'s slots for staging ``layer_id``."""
        slots = self._inval[buffer_id]
        old_ids = self.id_of_slot[slots]
        self.slot_for_id.view(-1)[old_ids[old_ids >= 0].long()] = -1
        self.id_of_slot[slots] = -1
        self._stage_dirty[buffer_id] = True
        # Stamp step+1, not zero. Zero made the borrowed slots the first victim of every
        # decode ensure, so interleaved decode rows landed here and the next chunk's
        # invalidation wiped them (decode hit rate ~0 under mixed load, every interleaved
        # step re-fetched in full). step+1 outranks every LRU-touched row, so decode
        # evicts its own coldest regular slot and keeps residency the chunk plans never
        # touch. Their slot_for_id entries are -1 above, so a route that still resolves to
        # one is a miss and reloads.
        #
        # The stamp is capped per pool so one ensure always keeps a victim per miss; the
        # borrowed slots over the cap stay evict-first (see _slice_pools).
        self.usage[self._inval_deprio[buffer_id]] = self.step + 1
        self.usage[self._inval_evict[buffer_id]] = 0
        # the pad may hold rows of a wider layout or of a pool; a cache without banks
        # (a CPU-side bookkeeping cache) has no staging views and nothing to clear
        if self._stage_tails:
            for tail in self._stage_tails[self.layer_layout[layer_id]][buffer_id]:
                tail.zero_()

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
        # The on-demand planners read the LIVE slot map (their per-chunk promotion lands
        # there), so no snapshot of it is taken here; only the streaming path uses one.
        self._prefill_hit_d2d_active = self.prefill_hit_d2d and self._hit_d2d_usable()
        if self._prefill_hit_d2d_active and not self.ondemand_prefill:
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

        assert self.banks and self._stage_views

        buffer_id = layer_id % 2
        if self._prefill_buffer_layer[buffer_id] == layer_id:
            return
        if self._prefill_buffer_layer[buffer_id] is not None:
            assert self._prefill_buffer_released[buffer_id], (
                "Prefill overlap buffer is being reused before release"
            )

        def copy() -> None:
            self._invalidate_prefill_buffer(buffer_id, layer_id)
            for (per_layer, _), view in zip(self.banks, self.stage_views(layer_id, buffer_id)):
                view.copy_(per_layer[layer_id], non_blocking=True)

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
        elif all(stable >= s for stable, s in zip(self._pool_stable_from, self.pool_sizes)):
            reason = (
                f"cache_size {self.cache_size} leaves no hit region "
                f"(every slot lies under the prefill staging buffers of {2 * self.num_experts} experts)"
            )
        elif not self._resolve_batch_memcpy():
            reason = "batch memcpy is unavailable"  # resolve logged the specifics
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
        PCIe as ONE batch memcpy of coalesced expert-id runs on the copy
        stream, under the existing release/ready event discipline; its host-built
        run list comes from the begin-of-chunk snapshot because the batch API
        takes HOST pointer arrays. Live-vs-snapshot cannot disagree: the only
        chunk-internal writer (buffer invalidation) rewrites slots already below
        the pool's staging threshold, and those slots (and -1) are misses on both sides
        -- the buffers own those slots, so their bytes are volatile within the
        chunk. Hit and miss row sets are disjoint, so the streams need no
        ordering against each other.
        """
        import numpy as np

        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
        from freetoken.moe.offload_kernels import prefill_hit_compact

        E = self.num_experts
        layout = self.layer_layout[layer_id]
        pool = self.pool_of_layer[layer_id]
        snap = self._prefill_snapshot_np[layer_id]
        hit_mask = snap >= self._pool_stable_from[pool]
        self.prefill_hit_rows += int(hit_mask.sum())
        self.prefill_total_rows += E
        if self._gather_feat_bytes[layout] is not None:
            prefill_hit_compact(self, layer_id)
            # blocks_per_bank=64 vs the PCIe-tuned default of 8: HBM D2D needs the
            # wider grid (~22 GB/s per 1024-thread block on H100).
            fast_index_copy_multi_jit(
                self._gather_stage_ptrs[layout][buffer_id],
                self._gather_pool_ptrs[pool],
                self._gather_feat_bytes[layout],
                self._prefill_hit_dst,
                self._prefill_hit_src,
                self._prefill_hit_num,
                blocks_per_bank=64,
            )
        miss = np.nonzero(~hit_mask)[0]
        with torch.cuda.stream(self.prefill_copy_stream):
            if self._prefill_buffer_has_release_event[buffer_id]:
                self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
            self._invalidate_prefill_buffer(buffer_id, layer_id)
            if miss.size:
                run_starts = np.concatenate(([0], np.nonzero(np.diff(miss) != 1)[0] + 1))
                starts = miss[run_starts]
                lengths = np.diff(np.concatenate((run_starts, [miss.size])))
            dst, src, nbytes = [], [], []
            for b, feat in enumerate(self._row_bytes[layout]):
                stage = self._copy_dst_ptrs_host[b] + buffer_id * self._stage_bytes[b]
                if feat == 0:
                    continue
                if feat < _SMALL_BANK_FEAT_BYTES:
                    # Whole layer as one entry, EVEN with zero misses: it keeps every
                    # batch entry above the driver's async floor and covers the hit
                    # rows the gather skips for these banks.
                    dst.append(stage)
                    src.append(self._copy_src_ptrs_host[layer_id][b])
                    nbytes.append(E * feat)
                elif miss.size:
                    dst.extend(stage + starts * feat)
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
        ids = topk_ids.clone()  # ensure rewrites ids to slot ids in place
        ensure_experts_hybrid(
            self, layer_id, ids, torch.ones_like(ids, dtype=torch.float32),
            self.num_experts, self.prefill_promote_fraction,
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
        assert self.prefill_bank_buffers, "on-demand prefill needs the staging buffers"
        assert (
            self._copy_fused_ok or layer_id in self._unpinned_layers
        ), "on-demand prefill needs the fused copy plan (or a file-tier layer)"
        from freetoken.moe.offload_kernels import prefill_ondemand_compact

        buffer_id = layer_id % self._prefill_num_buffers
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
        # A hit is only a hit if the D2D gather will actually serve it. That gather runs
        # over EVERY bank (see move_), so the cut is the fused copy plan, not the batch
        # memcpy's small-bank floor; without it every touched expert takes the host plan.
        gatherable = self._copy_fused_ok
        prefill_ondemand_compact(
            self, layer_id, buffer_id, touched,
            # the pool-local slots the staging buffers own; above them a slot holds a real row
            threshold=self._pool_stable_from[self.pool_of_layer[layer_id]] if gatherable else 1 << 30,
            pcie_frac_q16=pcie_frac_q16,
        )
        if self.auto_promote and self._promote_pin is not None:
            # chunk totals for the policy feedback (harvested at the next chunk begin);
            # stream-ordered after the compact kernel that wrote the per-layer counts.
            # [0:1] slices: += against the [1]-shaped counters must not reshape the view.
            self._promote_acc[0:1] += self._prefill_hit_num
            self._promote_acc[1:2] += self._prefill_miss_num
        self._invalidate_prefill_buffer(buffer_id, layer_id)
        return self.stage_views(layer_id, buffer_id)

    def move_prefill_layer_ondemand(self, layer_id: int) -> None:
        """Issue the gathers planned by :meth:`plan_prefill_layer_ondemand`.

        Split out from the planning step so a caller can submit the CPU executor's partial
        BETWEEN the two: on one stream the D2H activation copy a submit needs cannot start
        until everything already enqueued has finished, so issuing the H2D expert gather
        first serializes the CPU work behind the whole transfer instead of overlapping it.
        """
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        if layer_id in self._unpinned_layers:
            # File-tier (unregistered) rows: the SM copy kernel dereferences device
            # aliases that mmap'd shard rows do not have, and promotion is skipped
            # for these layers so hits cannot exist either -- every touched row is a
            # miss that the pageable path below gathers.
            self._move_prefill_layer_ondemand_pageable(layer_id)
            return
        layout, pool = self.layer_layout[layer_id], self.pool_of_layer[layer_id]
        # Hits gather D2D from their cache slots, misses H2D from the host banks, and both
        # run over EVERY bank: the SM copy kernel has no small-row caveat (unlike the
        # streaming path's batch memcpy), so scales and globals come from their cache slots
        # too. Leaving the narrow banks to the miss plan would strand their hit rows on an
        # earlier chunk's bytes. D2D first: the HBM rate is ~40x PCIe.
        #
        # The plan's rows already carry the staging offset (buffer_id * num_experts + expert),
        # so both gathers aim at the bank base; hit sources are pool-local slots, so they aim
        # at the layer's pool.
        fast_index_copy_multi_jit(
            self._copy_dst_ptrs,
            self._copy_dst_ptrs_pool[pool],
            self._copy_feat_bytes[layout],
            self._prefill_hit_dst,
            self._prefill_hit_src,
            self._prefill_hit_num,
            blocks_per_bank=64,
        )
        fast_index_copy_multi_jit(
            self._copy_dst_ptrs,
            self._copy_src_ptrs[layer_id],
            self._copy_feat_bytes[layout],
            self._prefill_miss_dst,
            self._prefill_miss_src,
            self._prefill_miss_num,
        )

    def _move_prefill_layer_ondemand_pageable(self, layer_id: int) -> None:
        """Touched-only gather for an unpinned layer. File-backed rows come in by
        parallel pread into a pinned staging row block (no mmap fault storm, no
        pageable-H2D page locking) and land in the routed buffer rows with one async
        H2D plus a D2D scatter; OS-locked rows are already resident, so they keep the
        plain host gather. Either way the chunk only moves the bytes its routing
        touched, not the whole layer."""
        import time as _time

        t0 = _time.perf_counter()
        buffer_id = layer_id % self._prefill_num_buffers
        num = int(self._prefill_miss_num.item())
        t_item = _time.perf_counter()
        if num == 0:
            return
        src_rows = self._prefill_miss_src[:num].to("cpu").long()
        dst_rows = self._prefill_miss_dst[:num].to(self.device).long()
        rows = src_rows.tolist()
        t_rows = _time.perf_counter()
        read_ms = h2d_ms = 0.0
        views = self.stage_views(layer_id, buffer_id)
        for name, (per_layer, _cache), buffer in zip(self.bank_schema, self.banks, views):
            per_file = self._file_banks.get(name)
            file_bank = per_file[layer_id] if per_file is not None else None
            if file_bank is not None:
                staging = self._pageable_staging(name, per_layer[layer_id])
                cap = staging.shape[0]
                row_bytes = per_layer[layer_id][0].nbytes
                # batched stream: a fixed small pinned block serves any touched-set
                # size (a whole layer per bank would pin another ~2 GiB, which the
                # pin budget's headroom cannot absorb after the banks themselves)
                for i in range(0, num, cap):
                    n = min(cap, num - i)
                    out_rows = staging[:n].view(torch.uint8).numpy().reshape(n, -1)
                    _tr = _time.perf_counter()
                    file_bank.read_rows(rows[i:i + n], row_bytes, out_rows)
                    read_ms += _time.perf_counter() - _tr
                    _th = _time.perf_counter()
                    gpu_rows = staging[:n].to(self.device, non_blocking=True)
                    # uint8 views: fp8 scale banks have no index_copy kernel, the raw
                    # bytes scatter identically
                    buffer.view(torch.uint8).index_copy_(
                        0, dst_rows[i:i + n], gpu_rows.view(torch.uint8)
                    )
                    h2d_ms += _time.perf_counter() - _th
            else:
                _th = _time.perf_counter()
                staged = per_layer[layer_id][src_rows]
                # uint8 views: fp8 scale banks have no index_copy kernel, the raw
                # bytes scatter identically (same workaround as the file-bank branch)
                buffer.view(torch.uint8).index_copy_(
                    0, dst_rows, staged.to(self.device).view(torch.uint8)
                )
                h2d_ms += _time.perf_counter() - _th
        acc = self._pageable_timing
        acc["item_ms"] += (t_item - t0) * 1e3
        acc["rows_ms"] += (t_rows - t_item) * 1e3
        acc["read_ms"] += read_ms * 1e3
        acc["h2d_ms"] += h2d_ms * 1e3
        acc["layers"] += 1
        acc["rows"] += num
        if _PAGEABLE_TIMING and layer_id == self.num_layers - 1:
            logger.info_rank0(
                f"[pageable] chunk: layers={acc['layers']} rows={acc['rows']} "
                f"item={acc['item_ms']:.0f}ms rows={acc['rows_ms']:.0f}ms "
                f"read={acc['read_ms']:.0f}ms h2d={acc['h2d_ms']:.0f}ms"
            )
            for k in acc:
                acc[k] = 0

    def _pageable_staging(self, name: str, layer_tensor: "torch.Tensor") -> "torch.Tensor":
        """Pinned [cap_rows, *spec] staging block for one bank's file-tier rows,
        allocated once on first use and reused by every later chunk and batch: pread
        targets it directly, so its pages fault once and the H2D never re-locks them.
        The row cap (FREETOKEN_PREFILL_STAGING_ROWS) bounds the pinned commitment;
        the move streams any touched-set size through it in batches."""
        staging = self._prefill_pinned_staging.get(name)
        if staging is None:
            cap = min(self.num_experts, _PREFILL_STAGING_ROWS)
            staging = torch.empty(
                (cap, *layer_tensor.shape[1:]),
                dtype=layer_tensor.dtype, pin_memory=True,
            )
            self._prefill_pinned_staging[name] = staging
            logger.info_rank0(
                f"MoE file-tier prefill staging: {staging.nbytes / 2**20:.0f} MiB pinned "
                f"for bank {name!r} (batched pread target, {cap} rows)"
            )
        return staging

    def cold_stage_views(self) -> tuple:
        """Per-bank VRAM cold-staging views in registration order (the GPU-only decode
        GEMM reads the staged cold rows from here)."""
        assert self._cold_stage_vram is not None, "gpu_stage_cold must run first"
        return tuple(self._cold_stage_vram[n] for n in self.bank_schema)

    def gpu_stage_cold(self, layer_id: int, cpu_ids: "torch.Tensor"):
        """GPU-only decode: stage the cold routes' expert rows into a dedicated VRAM
        buffer so the grouped GEMM computes them on the GPU instead of the CPU executor.

        ``cpu_ids`` is [M, top_k] with the raw expert id at cold positions and -1
        elsewhere (the complement of the slot-resident routes). Each cold position gets
        its own staging row (no dedup; bs*top_k is small). The rows are pread from the
        layer's file-backed bank into the pinned host staging, then copied to VRAM.
        Eager-only (the host pread cannot be CUDA-graph captured). Returns [M, top_k]
        staging-row ids (-1 at non-cold positions), or None when there are no cold routes."""
        cold_mask = cpu_ids >= 0
        n_cold = int(cold_mask.sum().item())
        if n_cold == 0:
            return None
        cap = self._cold_stage_cap
        if n_cold > cap:
            raise RuntimeError(
                f"gpu-only decode: {n_cold} cold routes in one layer exceed the {cap}-row "
                f"staging cap (lower the batch size or raise FREETOKEN_PREFILL_STAGING_ROWS)"
            )
        cold_experts = cpu_ids[cold_mask].tolist()
        if self._cold_stage_vram is None:
            self._cold_stage_vram = {}
        idx_cpu = torch.tensor(cold_experts, dtype=torch.long)
        for name in self.bank_schema:
            src = self.bank_sources[name][layer_id]
            per_file = self._file_banks.get(name)
            file_bank = per_file[layer_id] if per_file is not None else None
            host = self._pageable_staging(name, src)
            vram = self._cold_stage_vram.get(name)
            if vram is None:
                vram = torch.empty((cap, *src.shape[1:]), dtype=src.dtype, device=self.device)
                self._cold_stage_vram[name] = vram
            row_bytes = src[0].nbytes
            out = host[:n_cold].view(torch.uint8).numpy().reshape(n_cold, -1)
            if file_bank is not None:
                file_bank.read_rows(cold_experts, row_bytes, out)
            else:
                # a pinned layer has no cold routes in practice, but stay correct: gather
                # the resident rows straight from the (registered) host bank.
                with torch.inference_mode():
                    host[:n_cold].copy_(src.index_select(0, idx_cpu.to(src.device)))
            vram[:n_cold].copy_(host[:n_cold], non_blocking=True)
        stage_ids = torch.full_like(cpu_ids, -1)
        stage_ids[cold_mask] = torch.arange(n_cold, device=cpu_ids.device, dtype=cpu_ids.dtype)
        return stage_ids

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
        assert self._stage_views
        self.prefetch_prefill_layer(layer_id)
        buffer_id = layer_id % 2
        assert self._prefill_buffer_layer[buffer_id] == layer_id
        if self.prefill_ready_events:
            torch.cuda.current_stream(self.device).wait_event(self.prefill_ready_events[buffer_id])
        return self.stage_views(layer_id, buffer_id)

    def release_prefill_layer(self, layer_id: int) -> None:
        if layer_id == self.num_layers - 1:
            # the staged slots were invalidated and nothing registers them within the forward, so their bytes are free to clear
            for b in range(2):
                if self._stage_dirty[b]:
                    # staged rows left there may be shifted or of another quant, and kernels still read them: MMQ past a slot's last row, weight-0 routes on slot 0
                    for (_, cache), stage in zip(self.banks, self._stage_bytes):
                        cache.view(-1).view(torch.uint8)[b * stage : (b + 1) * stage].zero_()
                    self._stage_dirty[b] = False
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

    def ensure_experts_hybrid(
        self, layer_id: int, expert_ids: torch.Tensor, weights: torch.Tensor, clamp: bool = False
    ) -> HybridRoutes:
        """Capped-fetch LRU for the hybrid backend.

        Like :meth:`ensure_experts` but assigns slots to (and schedules copies for) at
        most ``hybrid_max_fetch`` -- or ``~hybrid_fetch_fraction * misses`` when the
        fraction is set -- of this step's missing experts; the overflow misses are
        left non-resident and ``expert_ids`` is rewritten to their cache slot (hit or
        freshly fetched) or ``-1`` (overflow -> compute on the CPU). ``num_indices`` holds
        the capped fetch count (for ``copy_missing``); ``num_missing_full`` the pre-cap
        miss count (for stats). All device-side / fixed-shape, so it is CUDA-graph safe.
        Returns the routes split between GPU and CPU (:class:`freetoken.moe.offload_kernels.HybridRoutes`)."""
        from freetoken.moe.offload_kernels import ensure_experts_hybrid

        if self.collect_decode_freq:
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        fraction = self.hybrid_fetch_fraction[layer_id] if self.hybrid_fetch_fraction is not None else 0.0
        max_fetch, cap = self.hybrid_max_fetch, None
        if layer_id in self.locked_hybrid:
            # a locked layer fetches through the staging, so at most its rows per step
            (max_fetch, fraction), cap = self.locked_fetch.plan[layer_id], self.locked_fetch.max_fetch
        if os.environ.get("FREETOKEN_GPU_ONLY_DECODE", "").strip().lower() in ("1", "true", "yes", "on"):
            # Warehouse mode: fetch every compact-backed miss to a slot so only the truly
            # cold rows (no registered compact) fall through to gpu_stage_cold's pread.
            fraction = 1.0
        self._pending_fetch_cap = cap
        return ensure_experts_hybrid(self, layer_id, expert_ids, weights, max_fetch, fraction, clamp, cap)

    def materialize_layer(self, layer_id: int) -> None:
        from freetoken.moe.offload_kernels import materialize_layer

        self._pending_src_layer = layer_id
        self._pending_whole_layer = True
        materialize_layer(self, layer_id)
        # Disk tier: kick the kernel's readahead on the NEXT layer's shard pages while
        # this layer's whole-layer pageable copy runs -- the copy faults cold pages in
        # serially, so one layer of NVMe latency is already hidden behind the current
        # layer's copy + GEMM.
        nxt = layer_id + 1
        if nxt < self.num_layers and self._file_banks:
            for per_layer in self._file_banks.values():
                bank = per_layer[nxt] if nxt < len(per_layer) else None
                if bank is not None:
                    bank.willneed()

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
        if layer_id == 0:
            self._profile_dump_step()
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

    def residency_split(self) -> tuple[int, int]:
        """(#slots holding a valid expert, #of them read or fetched by the last forward),
        one host sync; the dashboard gauge path, never called inside forward. ``step``
        advances once per MoE-layer call, so one forward spans ``num_layers`` of them and
        the second count is that forward's working set. Everything filled but not in it is
        warm cache: the LRU may evict any of those slots on the next call."""
        filled = self.id_of_slot >= 0
        live = filled & (self.usage > self.step - self.num_layers)
        return int(filled.sum()), int(live.sum())

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

    def decode_stats_line(self) -> str | None:
        """Decode stats since the last call by layer group (pinned, locked with slots, CPU-only), as one log line; zeroes the counters."""
        if self.decode_target == "hybrid":
            rows = torch.stack([self.stat_steps_layer, self.stat_active_layer, self.stat_missing_layer, self.stat_fetched_layer], 1).tolist()
        else:
            # offload copies every miss
            cols = self.lru_stats.t()
            rows = torch.stack([cols[Stat.CALLS], cols[Stat.ACTIVE], cols[Stat.MISS], cols[Stat.MISS]], 1).tolist()
        self.reset_stats()
        groups = (("pinned", [l for l in range(self.num_layers) if l not in self.cpu_layer_ids]), ("locked", sorted(self.locked_hybrid)))
        parts = []
        for name, layers in groups:
            steps, active, missing, fetched = (sum(rows[l][k] for l in layers) for k in range(4))
            if steps and active:
                parts.append(
                    f"{name} {_layers(len(layers))}: hit {1 - missing / active:.1%}, per layer-step "
                    f"miss {missing / steps:.2f} fetch {fetched / steps:.2f} cpu {(missing - fetched) / steps:.2f}"
                )
        cpu_only = self.cpu_layer_ids - self.locked_hybrid
        if cpu_only:
            parts.append(f"cpu-only {_layers(len(cpu_only))}")
        return "MoE decode, " + "; ".join(parts) if parts else None

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
                "pool": self.pool_of_layer[L],
                "pool_slots": self.pool_slots(L),
                "steps": s,
                "active_per_step": (a / s) if s else 0.0,
                "missing_per_step": (m / s) if s else 0.0,
                "miss_rate": (m / a) if a else 0.0,
                "fetched_per_step": (f / s) if s else 0.0,
            })
        return {"per_layer": per_layer}

    def _profile_dump_step(self) -> None:
        """Periodic learned-profile checkpoint: refresh _profile_out every 256 decode
        steps so a hard-killed worker still leaves the histogram on disk (the atexit
        hook only runs on a graceful shutdown)."""
        if self._profile_out is None:
            return
        self._profile_steps += 1
        if self._profile_steps % 256 == 0:
            total = self.dump_bank_profile(self._profile_out)
            if total:
                logger.info_rank0(
                    f"expert profile checkpoint: {total} routed observations -> "
                    f"{self._profile_out}"
                )

    def dump_bank_profile(self, path: str) -> int:
        """Write the accumulated decode routing histogram as an FTXP expert profile
        (hottest (layer, expert) pairs first) for --moe-bank-profile.

        The histogram only fills with collect_decode_freq on and CUDA graphs disabled
        (a captured decode replays kernels, not this host-side accumulation). Returns
        the number of routed observations the ranking saw."""
        from freetoken.moe.expert_profile import rank_by_counts, write_profile

        freq = self.decode_freq.detach().to("cpu")
        total = int(freq.sum())
        if total == 0:
            return 0
        write_profile(
            path, rank_by_counts(freq.tolist()),
            num_layers=self.num_layers, num_experts=self.num_experts,
        )
        return total

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
                if layer_id in self.locked_hybrid:
                    self.locked_fetch.copy(self, layer_id)
                    return
                if self._hot_row_of[layer_id] is None:
                    # a disk layer without hot compacts has no registered row anywhere:
                    # the fetch plan's kernel would dereference the 0 placeholder
                    raise RuntimeError(
                        f"layer {layer_id} is unpinned without hot compacts: no device alias "
                        f"exists for a slot fetch (decode must route it to the CPU executor)"
                    )
                # a compact-backed disk layer: its hot rows ARE registered (the loader pinned
                # the compact), so the fused gather DMAs them into slots like a pinned
                # layer's rows -- fall through to it instead of staging the whole layer
            else:
                # the only whole-layer copy a non-pinned layer takes is the non-overlap
                # prefill materialize, which schedules it into staging buffer 0 with
                # position == expert id -- a plain synchronous pageable H2D copy (never
                # CUDA-graph captured; prefill is not captured)
                for (per_layer, _), view in zip(self.banks, self.stage_views(layer_id, 0)):
                    view.copy_(per_layer[layer_id])
                return
        # materialize stages rows of staging buffer 0; ensure stages slots local to the layer's pool
        whole = self._pending_whole_layer
        if self._copy_fused_ok:
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

            # One launch copies the missing rows for every bank (instead of one launch per
            # bank). evict_slots/src_indices/num_indices are shared across banks;
            # src_indices holds layer-local expert rows, resolved against this layer's
            # source pointers (layer_id is a static int per captured graph node).
            src_indices, evict_slots, num_indices = self._plan_buffers(plan)
            if os.environ.get("FREETOKEN_COPY_DIAG") == "1":
                # Diagnostic: turn the cryptic async CUDA illegal-access into a precise
                # Python error pinpointing the OOB layer/bank/index. Eager-only (uses
                # .item()); gated so production is unaffected. The src bound depends on
                # the layer's ACTUAL source: a compact-backed disk layer sources from the
                # compact (its row count), a pinned layer from the whole bank (num_experts).
                n = int(num_indices.item())
                if n > 0:
                    smax = int(src_indices[:n].max().item())
                    dmax = int(evict_slots[:n].max().item())
                    sptrs = self._copy_src_ptrs[layer_id].tolist()
                    is_disk = layer_id in self._unpinned_layers
                    has_map = self._hot_row_of[layer_id] is not None
                    hot = (self.hot_bank_sources or {}).get(layer_id)
                    if is_disk and has_map:
                        srows = [(hot[nm][0].shape[0] if (hot and nm in hot) else None) for nm in self.bank_schema]
                    elif is_disk:
                        srows = [0] * len(self.bank_schema)  # null src placeholder
                    else:
                        srows = [self.num_experts] * len(self.bank_schema)  # whole pinned bank
                    if not hasattr(self, "_copy_diag_logged"):
                        self._copy_diag_logged = set()
                    if layer_id not in self._copy_diag_logged:
                        self._copy_diag_logged.add(layer_id)
                        hrow = self._hot_row_of[layer_id]
                        hstat = (f"max={int(hrow.max().item())} nneg={int((hrow >= 0).sum().item())} "
                                 f"shape={tuple(hrow.shape)} dev={hrow.device}") if hrow is not None else "None"
                        logger.info_rank0(
                            f"[copy-diag] L{layer_id} n={n} src_max={smax} dst_max={dmax} cache_size={self.cache_size} "
                            f"is_disk={is_disk} has_map={has_map} src_rows={srows} feat={self._copy_feat_bytes.tolist()} "
                            f"src_indices={src_indices[:n].tolist()} hot_row_of[{hstat}] "
                            f"src_ptrs_zero={[self.bank_schema[i] for i, p in enumerate(sptrs) if p == 0]}"
                        )
                    if dmax >= self.cache_size:
                        raise RuntimeError(f"[copy-diag] OOB dst L{layer_id}: dst_max={dmax} >= cache_size={self.cache_size}")
                    for bi, nm in enumerate(self.bank_schema):
                        if srows[bi] is not None and smax >= srows[bi]:
                            raise RuntimeError(
                                f"[copy-diag] OOB src L{layer_id} bank={nm}: src_max={smax} >= src_rows={srows[bi]} "
                                f"(is_disk={is_disk} has_map={has_map})"
                            )
            if blocks_per_bank is None:
                # the grid width measured for this host's link, not the kernel's default
                blocks_per_bank = self._copy_blocks_per_bank
            # A compact-backed disk layer's rows come from its pinned hot compact, whose rows
            # are laid out at the checkpoint's element size, not the (possibly raw-byte) bank
            # layout: size the copy from the source it reads.
            hot = (self.hot_bank_sources or {}).get(layer_id) if layer_id in self._unpinned_layers else None
            if hot is not None and not whole:
                feat = self._hot_feat_bytes.get(layer_id)
                if feat is None:
                    feat = torch.tensor(
                        [
                            hot[name][0].stride(0) * hot[name][0].element_size() if name in hot else 0
                            for name in self.bank_schema
                        ],
                        dtype=torch.int64,
                        device=self.device,
                    )
                    self._hot_feat_bytes[layer_id] = feat
            else:
                feat = self._copy_feat_bytes[self.layer_layout[layer_id]]
            fast_index_copy_multi_jit(
                self._copy_dst_ptrs if whole else self._copy_dst_ptrs_pool[self.pool_of_layer[layer_id]],
                self._copy_src_ptrs[layer_id],
                feat,
                evict_slots,
                src_indices,
                num_indices,
                blocks_per_bank=blocks_per_bank,
            )
            return

        from freetoken.kernel import fast_index_copy_jit

        views = self.stage_views(layer_id, 0) if whole else self.pool_bank_views(layer_id)
        for name, (per_layer, _), cache in zip(self.bank_schema, self.banks, views):
            if cache.numel() == 0:
                continue
            if layer_id in self._unpinned_layers:
                if self._hot_row_of[layer_id] is None:
                    raise RuntimeError(
                        f"layer {layer_id} is unpinned without hot compacts: no device "
                        f"alias exists for a fetch copy"
                    )
                src = self.hot_bank_sources[layer_id][name][0]
            else:
                src = per_layer[layer_id]
            fast_index_copy_jit(
                cache,
                self.evict_slots,
                src,
                self.src_indices,
                self.num_indices,
            )


def _layers(n: int) -> str:
    return f"{n} layer" if n == 1 else f"{n} layers"


def iter_offload_moe_layers(model) -> Iterator:
    from freetoken.layers import OffloadMoELayer, iter_moe_layers

    return (layer for layer in iter_moe_layers(model) if isinstance(layer, OffloadMoELayer))


def attach_offload_moe_cache(model, cache: OffloadMoeCache) -> list:
    layers = list(iter_offload_moe_layers(model))
    for layer in layers:
        layer.offload_cache = cache
    return layers
