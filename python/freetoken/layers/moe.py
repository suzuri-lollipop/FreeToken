import os
from typing import TYPE_CHECKING, Tuple

import torch
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.moe import is_offload_moe_strategy
from freetoken.moe.fused import fused_topk
from freetoken.moe.offload_cache import OffloadMoeCache


from .base import BaseOP
from .quantization import ExpertView, LayerKind, QuantConfig, quant_method_for

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# Router decision (topk_weights[float32], topk_ids[int32]) for models whose router
# is computed outside the MoE layer. Such models call ``routed_forward`` with a
# precomputed routing instead of going through the generic softmax+top-k path.
TopK = Tuple[torch.Tensor, torch.Tensor]

# Hybrid decode overlaps the CPU overflow GEMV behind the GPU PCIe fetch + GEMM by
# default. Set FREETOKEN_HYBRID_OVERLAP=0 to force the serial path (CPU sync before the
# GPU work) -- a measurement-only escape hatch to A/B the overlap benefit.
_HYBRID_OVERLAP = os.getenv("FREETOKEN_HYBRID_OVERLAP", "1") != "0"

# Dual-microbatch decode: blocks/bank for each half's miss gather. Narrow grids
# saturate the PCIe link (measured: 1-2 blocks/bank reach link rate) while leaving
# the SMs to the concurrent half's dense/attention work.
_DUAL_GATHER_BPB = int(os.getenv("FREETOKEN_DUAL_GATHER_BPB", "2") or "2")

# lru_ensure's phase-1 dedup is a [K, K] register block over the query (K = rows *
# top_k): flat to K=128, then it spills hard (flashlib cost model: +55us at K=256;
# measured ~198us/call at K=200 on sm120 = ~9.5ms/step of pure admission overhead
# at bs20). Splitting the per-layer ensure into row chunks of <= this many query
# ids keeps every call in the flat region; the chunks chain ensure->copy on one
# stream (stream order protects the plan buffer, and same-step LRU usage stamps
# protect a chunk's admits from the next chunk's eviction). 0 keeps the
# single-call path. Measured on the 2x24GB NVFP4 rig: aggregate +6-8% at bs16-20
# (K=160-200), no change below the threshold.
_LRU_ENSURE_MAX_K = int(os.getenv("FREETOKEN_LRU_ENSURE_MAX_K", "128") or "128")

# Fetch-overlap decode (FREETOKEN_DECODE_FETCH_OVERLAP=1): minimum batch rows to run
# the hit/miss split. The two-pass GEMV's fixed overhead only pays once the hit
# routes' GEMV is wide enough to hide the miss gather under (measured a wash at
# bs4 on this rig's agent4 campaign; the window grows ~linearly with rows).
_FETCH_OVERLAP_MIN_ROWS = int(os.getenv("FREETOKEN_FETCH_OVERLAP_MIN_ROWS", "2") or "2")

# The CPU side of a prefill chunk pays routes-per-expert (~num_tokens * top_k / touched),
# while its PCIe side pays a flat per-expert row: the T=150 balance measurement behind
# prefill_fetch_fraction does NOT extrapolate to full chunks. Measured on a 2x24GB rig
# (gen5 x16 + gen4 x4, nvfp4, 512 experts top-10): an 8192-token chunk at the auto 17.6%
# PCIe / 82.4% CPU split ran ~79 s/chunk vs ~7.3 s for the pure-PCIe plan (a 65k-token
# prompt: 630 s vs 58 s). Chunks above this gate therefore stay wholly on the PCIe plan;
# small prefills (single short chunks) keep the measured split win. 0 disables the prefill
# split entirely, a large value restores the ungated behavior for A/B.
_HYBRID_PREFILL_MAX_TOKENS = int(os.getenv("FREETOKEN_HYBRID_PREFILL_MAX_TOKENS", "512"))

# Slot-direct prefill GEMM (A/B knob; requires the auto-promote policy ON so every
# routed row is LRU-resident): the grouped kernel reads expert rows from their slots
# through the id->slot map, skipping the per-chunk double-buffer D2D staging.
_PREFILL_SLOT_DIRECT = os.getenv("FREETOKEN_PREFILL_SLOT_DIRECT", "0") == "1"
# Per-chunk promote host-time attribution, reported on the scheduler's [chunktime]
# line (launch + any GPU backpressure that blocks the promote launches).
_CHUNK_TIMING = os.getenv("FREETOKEN_CHUNK_TIMING", "0") == "1"


def _is_spec_batch(batch) -> bool:
    """MTP spec steps keep phase="prefill" (sequential GDN/QSA/PLE) but must stage
    experts like decode does: device-driven on-demand fetch, no host readback."""
    return getattr(batch, "spec_mode", None) is not None


def _prefill_cpu_split(cache: OffloadMoeCache, num_tokens: int):
    """(CPU executor, Q16 PCIe share) for one prefill chunk, or (None, 1 << 16) when the
    chunk stays wholly on the PCIe plan.

    The CPU side reuses the decode executor, whose pinned IO buffers and registered tasks
    are sized by ``max_tokens``, so a chunk longer than that keeps the old behaviour
    instead of growing host buffers on the fly. Off unless the engine resolved a
    ``prefill_fetch_fraction`` below 1 -- see the measurement at that field.
    """
    executor = cache.cpu_executor
    if (
        executor is None
        or cache.prefill_fetch_fraction >= 1.0
        or cache.prefill_cpu_mask is None
        or num_tokens > executor.max_tokens
        or num_tokens > _HYBRID_PREFILL_MAX_TOKENS
    ):
        return None, 1 << 16
    return executor, cache.prefill_pcie_frac_q16()


class MoELayer(BaseOP):
    """Resident routed experts.

    The expert format comes from ``quant_method`` (declared by ``create_weights``, run by
    ``apply``); without a ``quant_config`` the experts are plain bf16. The gated activation is
    ``act(clamp(g, limit) * alpha) * (clamp(u) + beta)`` with ``interleaved`` gate|up rows
    for gpt-oss."""

    quant_layer_kind = LayerKind.MOE

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        allocate_experts: bool = True,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        limit: float | None = None,
        interleaved: bool = False,
        has_bias: bool = False,
        layer_id: int | None = None,
        strategy: str = "resident",
        decode_target: str = "gpu",
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self._comm = DistributedCommunicator()

        tp_info = get_tp_info()
        self.tp_rank = tp_info.rank
        self.tp_size = tp_size = tp_info.size
        self.renormalize = renormalize
        self.activation = activation
        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.alpha = alpha
        self.beta = beta
        self.limit = limit
        self.interleaved = interleaved
        self.has_bias = has_bias
        self.layer_id = layer_id
        self.strategy = strategy
        self.decode_target = decode_target
        self.prefix = prefix
        # offload layers without a quant config stay on the format-tag banks (GGUF q4_0)
        self.quant_method = None
        if quant_config is not None or allocate_experts:
            self.quant_method = quant_method_for(quant_config, self, prefix)
            if allocate_experts:
                self.quant_method.create_weights(self)

    def finalize(self) -> None:
        if self.quant_method is not None:
            self.quant_method.finalize(self)

    def _maybe_all_reduce(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            return self._comm.all_reduce(hidden_states)
        return hidden_states

    def _resident_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        assert self.quant_method is not None
        return self.quant_method.apply(
            hidden_states, topk_weights, topk_ids, self.quant_method.resident_view(self),
            layer=self, is_prefill=get_global_ctx().batch.is_prefill,
        )

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        Same name and shape as ``OffloadMoELayer.routed_forward`` so a model with
        its own router calls ``experts.routed_forward(...)`` without knowing whether
        the experts are resident or offloaded. The shared contract is the offload
        one: ``topk_ids`` must be safe to mutate in place (the offload decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.
        The resident path does not mutate it today, but callers must not rely on
        that.

        ``hidden_states`` may also be overwritten by the expert kernel. Compute
        shared branches that need the original input before calling this method.
        """
        out = self._resident_gemm(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        return self._maybe_all_reduce(self.routed_partial(hidden_states, router_logits))

    def routed_partial(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """This rank's expert compute WITHOUT the TP all-reduce.

        Lets a model fold the routed partial together with another row-parallel partial
        (the shared expert's) before ONE collective -- see Qwen4ExpMoE.forward. Same
        in-place contract as ``forward``: ``hidden_states`` may be overwritten.
        """
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._resident_gemm(hidden_states, topk_weights, topk_ids)

    def reduce_partial(self, partial: torch.Tensor) -> torch.Tensor:
        """The TP all-reduce ``forward`` would have applied to a ``routed_partial``."""
        return self._maybe_all_reduce(partial)


class OffloadMoELayer(MoELayer):
    def __init__(
        self,
        layer_id: int,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        *,
        alpha: float = 1.0,
        beta: float = 0.0,
        limit: float | None = None,
        interleaved: bool = False,
        has_bias: bool = False,
        strategy: str = "offload",
        decode_target: str = "gpu",
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            renormalize=renormalize,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            allocate_experts=False,
            alpha=alpha,
            beta=beta,
            limit=limit,
            interleaved=interleaved,
            has_bias=has_bias,
            layer_id=layer_id,
            strategy=strategy,
            decode_target=decode_target,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.offload_cache: OffloadMoeCache | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        return self._maybe_all_reduce(self.routed_partial(hidden_states, router_logits))

    def routed_partial(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """This rank's expert compute WITHOUT the TP all-reduce (prefill or decode by phase).

        OffloadMoELayer's ``forward`` all-reduces the phase output; this returns it raw so
        a model can fold it with another row-parallel partial into one collective.
        """
        ctx = get_global_ctx()
        if ctx.batch.is_prefill and not _is_spec_batch(ctx.batch):
            return self.prefill_forward(hidden_states, router_logits)
        # MTP spec batches ride phase="prefill" for the sequential GDN/QSA/PLE paths, but
        # their expert staging is the decode one: the two rows route independently and the
        # device-side on-demand fetch (no host readback) is both correct and far cheaper
        # at T=2 than the prefill streaming/on-demand machinery.
        return self.decode_forward(hidden_states, router_logits)

    def routed_forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Expert compute for an externally computed routing decision (``TopK``).

        The entry point for models whose router does not fit ``fused_topk`` (sigmoid
        scores, selection bias, group-limited top-k, ...); identical to ``forward``
        past the router. ``topk_ids`` must be safe to mutate in place (decode
        rewrites expert ids into cache slot ids); pass a fresh tensor or a clone.

        ``hidden_states`` may also be overwritten by the expert kernel. Compute
        shared branches that need the original input before calling this method.
        """
        ctx = get_global_ctx()
        if ctx.batch.is_prefill and not _is_spec_batch(ctx.batch):
            out = self._prefill_routed(hidden_states, topk_weights, topk_ids)
        else:
            out = self._decode_routed(hidden_states, topk_weights, topk_ids)
        return self._maybe_all_reduce(out)

    def decode_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._decode_routed(hidden_states, topk_weights, topk_ids)

    def prefill_forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor | None = None,
    ):
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
        )
        return self._prefill_routed(hidden_states, topk_weights, topk_ids)

    # ------------------------------------------------------------------
    # Data movement -- one decision tree for every quant format (the banks
    # registry makes the cache machinery bank-count agnostic). Decode loads
    # on demand; prefill streams whole layers, double-buffered when overlap
    # is enabled. The kernels only ever see bank views plus row indices;
    # which kernel runs is decided afterwards, in ``_expert_gemm``.
    # ------------------------------------------------------------------

    def _decode_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """On-demand load: ``ensure_experts`` rewrites ``topk_ids`` into cache slot
        ids in place (loading missing experts), then the GEMM reads the full slot
        cache. All device-side with fixed shapes, so the decode call is CUDA-graph
        capturable.

        For ``decode_target == "cpu"`` the experts are instead computed on the CPU
        (high RAM bandwidth) straight from the host banks: ship hidden/routing to
        pinned host memory, run the GEMV on the worker pool via host nodes, ship the
        result back. The GPU slot cache is untouched (topk_ids keep their raw expert
        ids), so no ``ensure_experts``/``copy_missing`` here."""
        cache = self.offload_cache
        assert cache is not None
        if cache.is_cpu_layer(self.layer_id):
            executor = cache.cpu_executor
            assert executor is not None, "CPU MoE executor was not initialized"
            return executor.decode(self.layer_id, hidden_states, topk_weights, topk_ids)
        if cache.decode_target == "hybrid" and hidden_states.shape[0] >= cache.hybrid_min_bs:
            return self._decode_hybrid(cache, hidden_states, topk_weights, topk_ids)
        # hybrid below hybrid_min_bs falls through to the GPU slot-cache path: a small
        # batch misses too few experts per layer for the CPU round trip to pay off.
        if cache.flat_residency:
            # This layer's experts own permanent slots, so the routing ids only need
            # shifting into the layer's block: no LRU lookup, no PCIe.
            topk_ids.add_(cache.flat_slot_base(self.layer_id))
            return self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=cache.bank_views(),
                n=None,
                alphas=cache.alphas_for_slots(self.layer_id),
                is_prefill=False,
            )
        try:
            ctx_batch = get_global_ctx().batch
            dual_slot = getattr(ctx_batch, "dual_slot", -1)
        except AssertionError:
            ctx_batch = None
            dual_slot = -1  # unit harnesses decode without a global ctx
        if dual_slot >= 0 and cache.dual_events is not None:
            return self._decode_dual_moe(
                cache, hidden_states, topk_weights, topk_ids, dual_slot
            )
        if (
            cache.decode_fetch_overlap
            and ctx_batch is not None
            and not _is_spec_batch(ctx_batch)
            and topk_ids.dim() == 2
            and hidden_states.shape[0] >= _FETCH_OVERLAP_MIN_ROWS
            and getattr(
                getattr(self.quant_method, "kernel", None), "supports_skip_w0", False
            )
        ):
            # The union miss-mask covers the two plan buffers the K-split ensure uses;
            # beyond two chunks fall back to the serial (chunked) path.
            k = topk_ids.numel()
            mk = _LRU_ENSURE_MAX_K if _LRU_ENSURE_MAX_K > 0 else k
            if mk <= 0 or -(-k // mk) <= 2:
                out = self._decode_fetch_overlap(
                    cache, hidden_states, topk_weights, topk_ids
                )
                if out is not None:
                    return out
        if (
            _LRU_ENSURE_MAX_K > 0
            and topk_ids.dim() == 2
            and topk_ids.numel() > _LRU_ENSURE_MAX_K
        ):
            cache.ensure_experts_chunked(self.layer_id, topk_ids, _LRU_ENSURE_MAX_K)
        else:
            cache.ensure_experts(self.layer_id, topk_ids)
            cache.copy_missing()
        from freetoken.moe import _debug_stats

        dbg = _debug_stats.probe()
        if dbg is not None:
            dbg.decode_step(topk_ids, None)
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=False,
        )

    def _decode_fetch_overlap(
        self,
        cache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | None:
        """Pure-GPU fetch overlap: run the hit routes' GEMV on the compute stream while
        the missing experts' H2D gather runs on a side stream, join, then run the
        complementary miss routes' GEMV and merge the partials.

        Both passes run the full decode pipeline over ALL routes with complementary
        zero-masked weights. The kernel's SKIP_W0 (gated by supports_skip_w0) makes each
        pass zero-store -- never READ -- the other half's slots, which is what makes the
        concurrency safe: a hit-pass read of a slot the copy is mid-write could pick up
        torn scale bytes, decode them as NaN, and 0 * NaN would poison the token's whole
        output row. Skip-W0 also keeps the total GEMV work ~1x (each pass computes only
        its own routes), and since each route's partial is computed independently and
        merged as x + 0.0 == x, the split is bit-identical to the serial single pass.

        The side-stream gather runs NARROW (few blocks per bank): measured on this rig's
        PCIe links, 1-2 blocks/bank still saturate both gen4 x4 (6.62 vs 6.63 GB/s) and
        gen5 x16 (20.4 vs 20.7), so the copy no longer occupies the SMs the concurrent
        hit GEMV needs. Admission stays K-split (the lru_ensure [K,K] dedup spill above
        K~128): the chunks stage into the two plan buffers, and the miss mask is their
        union.

        Mask construction, fork and join are all device-side / fixed-shape, so the split
        captures into the decode CUDA graph (multi-stream fork/join via events).
        Returns None when the second plan buffer is missing and the K-split needs it
        (the caller falls back to the serial path).
        """
        from freetoken.moe import _debug_stats

        rows, top_k = topk_ids.shape
        mk = _LRU_ENSURE_MAX_K if _LRU_ENSURE_MAX_K > 0 else rows * top_k
        rows_chunk = max(1, mk // top_k) if mk > 0 else rows
        starts = list(range(0, rows, rows_chunk))[:2]
        if len(starts) > 1 and cache.src_indices2 is None:
            # K-split needs the second plan buffer (the graph runner materializes it
            # before capture; a graphs-disabled eager run may not have it): fall back
            # to the caller's serial path rather than overwriting plan 0 mid-flight.
            return None
        for i, a in enumerate(starts):
            cache.ensure_experts(
                self.layer_id, topk_ids[a : a + rows_chunk], plan=i
            )
        miss = cache.miss_route_mask(topk_ids, tuple(range(len(starts))))
        w_hits = torch.where(miss, topk_weights.new_zeros(()), topk_weights)
        w_miss = torch.where(miss, topk_weights, topk_weights.new_zeros(()))
        dbg = _debug_stats.probe()
        if dbg is not None:
            dbg.decode_step(topk_ids, None)
        if cache.decode_copy_stream is None:
            cache.decode_copy_stream = torch.cuda.Stream(device=hidden_states.device)
        side = cache.decode_copy_stream
        main = torch.cuda.current_stream()
        fork = torch.cuda.Event()
        fork.record(main)  # orders the copies after this layer's ensure_experts
        side.wait_event(fork)
        with torch.cuda.stream(side):
            for i in range(len(starts)):
                cache.copy_missing(
                    plan=i, blocks_per_bank=cache.decode_overlap_gather_bpb
                )
        join = torch.cuda.Event()
        join.record(side)
        views = cache.bank_views()
        alphas = cache.alphas_for_slots(self.layer_id)
        # Per-call transient read by the nvfp4 kernel's apply (SKIP_W0 for both GEMVs).
        self._moe_skip_w0 = True
        try:
            out_hits = self._expert_gemm(
                cache, hidden_states, w_hits, topk_ids,
                views=views, n=None, alphas=alphas, is_prefill=False,
            )
            # The miss pass reads the fetched slot bytes; the next layer's ensure_experts
            # also re-stages the plan arrays the copy reads -- the join orders both.
            main.wait_event(join)
            out_miss = self._expert_gemm(
                cache, hidden_states, w_miss, topk_ids,
                views=views, n=None, alphas=alphas, is_prefill=False,
            )
        finally:
            self._moe_skip_w0 = False
        return out_hits + out_miss

    def _decode_dual_moe(
        self,
        cache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        slot: int,
    ) -> torch.Tensor:
        """Dual-microbatch decode: this half's MoE with cross-stream eviction safety.

        Both halves share the slot cache's LRU state, and an ``ensure_experts`` call
        evicts slots stamped before its own -- including slots the OTHER half's
        in-flight GEMV is still reading. Per-layer events serialize mutation against
        reads in the issue order A_l -> B_l -> A_{l+1}: the trailing half's ensure
        waits for the leading half's layer-l expert GEMV, and the leading half's
        layer-(l+1) ensure waits for the trailing half's layer-l GEMV. The waits sit
        inside the MoE (after this half's attention/gate), so one half's dense SM
        work overlaps the other half's PCIe fetch. Every edge points forward in the
        capture's issue order, so the captured graph is acyclic. The side-stream
        gather runs narrow (few blocks/bank: link-saturating, SMs left for the
        concurrent half)."""
        cur = torch.cuda.current_stream()
        ev_a, ev_b = cache.dual_events[self.layer_id]
        if slot == 1:
            cur.wait_event(ev_a)
        elif self.layer_id > 0:
            cur.wait_event(cache.dual_events[self.layer_id - 1][1])
        cache.ensure_experts(self.layer_id, topk_ids, plan=slot)
        cache.copy_missing(plan=slot, blocks_per_bank=_DUAL_GATHER_BPB)
        out = self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=False,
        )
        (ev_a if slot == 0 else ev_b).record(cur)
        return out

    def _decode_hybrid(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Hybrid decode: GPU computes cache hits + <=K freshly-fetched experts, the CPU
        computes the overflow misses, overlapped, then the partials merge.

        The CPU pool is kicked off (``decode_submit``) before the GPU PCIe fetch + GEMM so
        the CPU overflow GEMV runs concurrently with the GPU work. Capture-safe: the
        routing split is device-side elementwise and the CPU submit/sync are host nodes.
        Each route is computed exactly once -- the GPU weights are zeroed for CPU-assigned
        routes and the CPU ids are -1 for GPU-assigned routes (the C++ kernel skips id<0).
        """
        executor = cache.cpu_executor
        assert executor is not None, "CPU MoE executor was not initialized"
        raw = topk_ids.clone()  # raw expert ids for the CPU partial
        cache.ensure_experts_hybrid(self.layer_id, topk_ids)  # -> slot (hit/fetched) or -1
        if cache.collect_stats:
            cache.record_decode_stats_hybrid(self.layer_id)
        on_gpu = topk_ids >= 0

        cpu_ids = torch.where(on_gpu, raw.new_full((), -1), raw).contiguous()
        from freetoken.moe import _debug_stats

        dbg = _debug_stats.probe()
        if dbg is not None:
            dbg.decode_step(topk_ids, cpu_ids)
        pending = executor.decode_submit(self.layer_id, hidden_states, topk_weights, cpu_ids)

        # Measurement knob: FREETOKEN_HYBRID_OVERLAP=0 syncs the CPU pool *before* the
        # PCIe fetch + GPU GEMM, serializing the two so an A/B isolates the overlap win.
        cpu_routed_early = (
            executor.decode_sync(pending) if not _HYBRID_OVERLAP else None
        )

        cache.copy_missing()
        gpu_slots = topk_ids.clamp_min(0)  # -1 -> slot 0 (zero-weighted below)
        gpu_w = torch.where(on_gpu, topk_weights, topk_weights.new_zeros(())).contiguous()
        gpu_routed = self._expert_gemm(
            cache,
            hidden_states,
            gpu_w,
            gpu_slots,
            views=cache.bank_views(),
            n=None,
            alphas=cache.alphas_for_slots(self.layer_id),
            is_prefill=False,
        )
        cpu_routed = cpu_routed_early if not _HYBRID_OVERLAP else executor.decode_sync(pending)
        return gpu_routed + cpu_routed

    def _prefill_routed(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill movement: stream whole layers -- double-buffered behind the
        previous layer's GEMMs when ``prefill_overlap`` is on, else a synchronous
        ``materialize_layer``. In both, position == expert id, so the routing ids
        pass through unmapped."""
        cache = self.offload_cache
        assert cache is not None
        if cache.flat_residency:
            # The layer's experts already sit in their permanent slots in expert-id order,
            # so prefill is the plain full-layer GEMM: nothing to stage, nothing to copy.
            return self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=cache.bank_views_flat(self.layer_id),
                n=self.num_experts,
                alphas=cache.alphas_for_layer(self.layer_id),
                is_prefill=True,
            )
        if (
            cache.ondemand_prefill
            and cache.prefill_bank_buffers
            and hidden_states.shape[0] <= cache.ondemand_max_tokens
        ):
            # Short chunk: stage only the experts this chunk's routing touched (the
            # LRU stays untouched, the GEMM keeps the position == expert id contract).
            from freetoken.moe import _debug_stats

            dbg = _debug_stats.probe()
            if dbg is not None:
                dbg.prefill_layer_begin(self.layer_id)
            # Hybrid prefill: hand the tail of this layer's miss set to the CPU executor,
            # which reads the same rows straight out of the host banks over RAM, and stage
            # only the PCIe share. The two run concurrently and their partials sum, exactly
            # like _decode_hybrid -- worth it on a rank whose link is slower than its CPU
            # slice (a gen4 x4 rank measured 16.0 ms of PCIe against a 10.1 ms balanced
            # split for 82 missing experts at T=150).
            executor, frac_q16 = _prefill_cpu_split(cache, hidden_states.shape[0])
            if self.layer_id == 0:
                # Chunk-boundary policy step (async harvest of the previous chunk's plan
                # hit/miss totals + this chunk's promote gate); layer 0 runs once per
                # prefill forward, mirroring begin_prefill's hook on the streaming path.
                cache.ondemand_chunk_begin()
            if cache.prefill_promote_fraction > 0:
                # Slot-assign + H2D the recency-top share of this layer's misses BEFORE
                # the plan, so the plan gathers them D2D as hits and the rows survive
                # the chunk (repeated content stops re-streaming). See promote_prefill_misses.
                cache.promote_prefill_misses(self.layer_id, topk_ids)
            elif getattr(cache, "promote_auto_frac", 0.0) > 0:
                # Auto-promote policy: same contract, but the gate is decided per chunk
                # from measured reuse (ondemand_chunk_begin) and the promotion runs the
                # mask-driven fast path -- the route-sized ensure kernel behind
                # promote_prefill_misses costs ~seconds/chunk at prefill T (measured).
                if _CHUNK_TIMING:
                    import time as _time

                    _pt0 = _time.perf_counter()
                cache.promote_touched(
                    self.layer_id, topk_ids, stats=_PREFILL_SLOT_DIRECT
                )
                if _CHUNK_TIMING:
                    cache._promote_host_ms += (_time.perf_counter() - _pt0) * 1e3
                if (
                    _PREFILL_SLOT_DIRECT
                    and executor is None
                    and self.layer_id not in cache._unpinned_layers
                    and getattr(
                        getattr(self.quant_method, "kernel", None),
                        "supports_slot_direct_prefill", False,
                    )
                ):
                    # Slot-direct prefill: promote_touched just made every routed row
                    # resident in the LRU slot cache, so the grouped GEMM can read them
                    # IN PLACE through the id->slot map -- skipping the plan/move that
                    # D2D-stages the whole touched set into the double buffer every
                    # chunk (~110 rows/layer x 48 layers: the measured ~0.7-0.8s/chunk
                    # dominant cost at T=128 with promote reuse already at hitrate 1.0).
                    # stats=True on the promote above keeps the policy's hit/miss
                    # feedback alive (the plan-based accumulators never run here).
                    if dbg is not None:
                        dbg.prefill_layer_waited(self.layer_id)
                    out = self._expert_gemm(
                        cache,
                        hidden_states,
                        topk_weights,
                        topk_ids,
                        views=cache.bank_views(),
                        n=self.num_experts,
                        alphas=cache.alphas_for_slots(self.layer_id),
                        is_prefill=True,
                        slot_map=cache.slot_for_id[self.layer_id],
                    )
                    if dbg is not None:
                        dbg.prefill_layer_done(self.layer_id, topk_ids)
                    return out
            views = cache.plan_prefill_layer_ondemand(
                self.layer_id, topk_ids, pcie_frac_q16=frac_q16
            )
            pending = None
            gpu_ids, gpu_weights = topk_ids, topk_weights
            if executor is not None:
                on_cpu = cache.prefill_cpu_mask[topk_ids.reshape(-1).long()]
                on_cpu = on_cpu.view(topk_ids.shape) != 0
                # A CPU-owned route reads buffer row 0 at weight 0 -- zeroed once at
                # buffer init, so it contributes exactly nothing and its answer comes
                # from the partial. The CPU side gets the raw id and -1 for every route
                # the GPU owns (the kernel skips id < 0), so each route is computed once.
                gpu_ids = torch.where(on_cpu, topk_ids.new_zeros(()), topk_ids)
                gpu_weights = torch.where(
                    on_cpu, topk_weights.new_zeros(()), topk_weights
                ).contiguous()
                cpu_ids = torch.where(
                    on_cpu, topk_ids, topk_ids.new_full((), -1)
                ).contiguous()
                # Submit BEFORE the gather is issued: this enqueues the D2H activation
                # copy the CPU pool needs, so the experts stream over PCIe while the CPU
                # works instead of after it. Issuing the gather first serializes the two
                # on this stream and measures ~40-70% SLOWER than not splitting at all.
                pending = executor.decode_submit(
                    self.layer_id, hidden_states, topk_weights, cpu_ids
                )
            cache.move_prefill_layer_ondemand(self.layer_id)
            if dbg is not None:
                dbg.prefill_layer_waited(self.layer_id)
            out = self._expert_gemm(
                cache,
                hidden_states,
                gpu_weights,
                gpu_ids,
                views=views,
                n=self.num_experts,
                alphas=cache.alphas_for_layer(self.layer_id),
                is_prefill=True,
            )
            if pending is not None:
                out = out + executor.decode_sync(pending)
            if dbg is not None:
                dbg.prefill_layer_done(self.layer_id, topk_ids)
            return out
        if cache.prefill_overlap:
            from freetoken.moe import _debug_stats

            dbg = _debug_stats.probe()
            if dbg is not None:
                dbg.prefill_layer_begin(self.layer_id)
            views = self._wait_prefill_overlap(cache)
            if dbg is not None:
                dbg.prefill_layer_waited(self.layer_id)
            out = self._expert_gemm(
                cache,
                hidden_states,
                topk_weights,
                topk_ids,
                views=views,
                n=self.num_experts,
                alphas=cache.alphas_for_layer(self.layer_id),
                is_prefill=True,
            )
            if dbg is not None:
                dbg.prefill_layer_done(self.layer_id, topk_ids)
            cache.release_prefill_layer(self.layer_id)
            return out
        cache.materialize_layer(self.layer_id)
        cache.copy_missing()
        return self._expert_gemm(
            cache,
            hidden_states,
            topk_weights,
            topk_ids,
            views=cache.bank_views(self.num_experts),
            n=self.num_experts,
            alphas=cache.alphas_for_layer(self.layer_id),
            is_prefill=True,
        )

    def _wait_prefill_overlap(self, cache: OffloadMoeCache) -> tuple[torch.Tensor, ...]:
        """Double-buffer choreography for this layer's overlap prefill: kick off the
        next layer's full-layer H2D copy, then return this layer's bank views (in
        bank registration order; buffer position == expert id, so routing ids pass
        through unmapped). The caller runs ``release_prefill_layer`` after its GEMMs.
        """
        if self.layer_id == 0:
            cache.begin_prefill()
        cache.prefetch_prefill_layer(self.layer_id)
        cache.prefetch_prefill_layer(self.layer_id + 1)
        return cache.wait_prefill_layer(self.layer_id)

    # ------------------------------------------------------------------
    # Kernel dispatch: ``views`` are the bank tensors the movement step produced (in bank registration order) and ``topk_ids`` already index their rows.
    # GGUF q4_0 experts still dispatch on the cache's format tag until they get a method.
    # ------------------------------------------------------------------

    def _expert_gemm(
        self,
        cache: OffloadMoeCache,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        views: tuple[torch.Tensor, ...],
        n: int | None,
        alphas: tuple[torch.Tensor, torch.Tensor] | None,
        is_prefill: bool,
        slot_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.quant_method is not None:
            from freetoken.moe.legacy_format import canonical_role  # legacy_format imports this package

            view = ExpertView(
                {canonical_role(name): t for name, t in zip(cache.bank_schema, views)},
                # decode: per-route slot ids; slot-direct prefill: the [E] id->slot map;
                # double-buffer prefill: None (position == expert id)
                slots=topk_ids if n is None else slot_map, n=n, alphas=alphas,
            )
            return self.quant_method.apply(
                hidden_states, topk_weights, topk_ids, view, layer=self, is_prefill=is_prefill
            )
        fmt = cache.quant_format
        if fmt == "q4_0":
            # Native GGUF Q4_0 experts: dequant-in-kernel grouped GEMV (MMVQ) over the
            # streamed packed banks; topk_ids already index the cache slots / layer.
            from freetoken.moe.fused_q4_0 import fused_experts_gguf_q4_0

            gate_up, down = views
            return fused_experts_gguf_q4_0(
                hidden_states, gate_up, down, topk_weights, topk_ids, self.activation
            )
        raise AssertionError(f"offload experts without a quant method only serve q4_0 banks, got {fmt!r}")


def make_moe_layer(
    config: "ModelConfig",
    *,
    layer_id: int | None = None,
    activation: str = "silu",
    renormalize: bool | None = None,
    apply_router_weight_on_input: bool = False,
    num_experts: int | None = None,
    top_k: int | None = None,
    hidden_size: int | None = None,
    intermediate_size: int | None = None,
    resident_cls: type[MoELayer] | None = None,
    offload_cls: "type[OffloadMoELayer] | None" = None,
    alpha: float = 1.0,
    beta: float = 0.0,
    limit: float | None = None,
    interleaved: bool = False,
    has_bias: bool = False,
    quant_config: QuantConfig | None = None,
    prefix: str = "",
) -> MoELayer:
    """Build the experts layer for ``config.moe_strategy`` -- the one construction
    seam between a model and the MoE strategy.

    Picks ``OffloadMoELayer`` for the offload family (offload/cpu/hybrid) and
    ``MoELayer`` otherwise. Geometry defaults come from ``config``; pass overrides
    for models whose fields deviate. ``resident_cls``/``offload_cls`` keep model-specific
    subclasses constructible through the same seam.
    """
    offload = is_offload_moe_strategy(config.moe_strategy)
    layer_cls = (offload_cls or OffloadMoELayer) if offload else (resident_cls or MoELayer)
    kwargs = dict(
        num_experts=num_experts if num_experts is not None else config.num_experts,
        top_k=top_k if top_k is not None else config.num_experts_per_tok,
        hidden_size=hidden_size if hidden_size is not None else config.hidden_size,
        intermediate_size=(
            intermediate_size if intermediate_size is not None else config.moe_intermediate_size
        ),
        renormalize=renormalize if renormalize is not None else config.norm_topk_prob,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        alpha=alpha,
        beta=beta,
        limit=limit,
        interleaved=interleaved,
        has_bias=has_bias,
        quant_config=quant_config,
        prefix=prefix,
    )
    if offload:
        assert layer_id is not None, "offload MoE backends need the layer_id"
        kwargs["layer_id"] = layer_id
        kwargs["strategy"] = config.moe_strategy
        kwargs["decode_target"] = config.decode_target
    return layer_cls(**kwargs)
