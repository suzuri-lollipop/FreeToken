from __future__ import annotations

import os
import time

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from freetoken.core import Batch, Req
from freetoken.env import ENV
from freetoken.gpu_select import gpu_identity
from freetoken.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    FlushCacheBackendMsg,
    FlushCacheResultMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from freetoken.utils import (
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
    load_toolcall_anchor_id,
)

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .mm import cut_image_spans, plan_mm_batch
from .prefill import ChunkedReq, PrefillManager
from .status import SchedulerStatusReporter
from .table import TableManager

if TYPE_CHECKING:
    from freetoken.engine import BatchSamplingArgs, ForwardOutput
    from freetoken.engine.spec_protocol import SpecRuntime


logger = init_logger(__name__)

# FREETOKEN_PIPED_DEBUG also traces the scheduling state on EVERY rank: under TP the
# message-count broadcast stays aligned only while `blocking` and the pipe handoff are
# rank-identical, and a divergence is invisible unless both sides are logged.
_PIPED_TRACE = os.environ.get("FREETOKEN_PIPED_DEBUG", "0") == "1"

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.2f} GiB"


# Wall-clock bound on how long running decodes may wait behind consecutive prefill
# steps. The count-based prefill_decode_interval cannot bound ITL when chunk times
# swing by 10x (PLE device-latency spikes turn a 7s chunk into 55-155s); the debt
# hands a step to decode once this many prefill SECONDS accrued while a decode was
# runnable. 0 keeps the pure count-based behavior. Default 2.0: measured on the
# 2x RTX PRO 4000 rig (churn: 2 victims + a 30k-token aggressor) the victims' max
# inter-token gap dropped from the whole prefill window (91-159s) to one chunk
# (~7s) in both healthy and NVMe-pathological device states (_scratch/agent3).
# The chunk already launched but not yet drained counts toward those seconds (see
# _inflight_prefill): the drain books it after the decision, so accrued-seconds-only
# bounds a waiting decode at two chunks and lets one slow chunk buy two decode steps.
def _prefill_debt_budget() -> float:
    raw = os.environ.get("FREETOKEN_PREFILL_DEBT_S", "2.0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        raise ValueError(f"FREETOKEN_PREFILL_DEBT_S must be a number, got {raw!r}")


# Victim-aware adaptive chunking: while a decode is waiting, shrink the prefill
# chunk so ONE chunk's forward time tracks this target. The gap a waiting user
# perceives is bounded by one chunk (+ its concede step), so the target is the
# stall budget in seconds. 0 keeps the static configured chunk for everyone.
def _adaptive_chunk_target_s() -> float:
    raw = os.environ.get("FREETOKEN_ADAPTIVE_CHUNK_TARGET_S", "0")
    try:
        return max(0.0, float(raw))
    except ValueError:
        raise ValueError(f"FREETOKEN_ADAPTIVE_CHUNK_TARGET_S must be a number, got {raw!r}")


# FREETOKEN_MIXED_PREFILL_TOKENS=N (0 = off): while decodes run, the next prompt chunk
# is scheduled WITH every running request in one phase="prefill" batch (their
# extend_len==1 rows ride the chunk's forward) instead of as a dedicated prefill step.
# Decodes then advance on EVERY forward -- the scheduling effect of Strata's
# STRATA_PIPED_CONCURRENT -- and N caps the chunk so a mixed step stays near Strata's
# 1024-token concurrent prompt window. While a prompt queue exists and a decode runs,
# the prefill/decode interleave guard (prefill_decode_interval, FREETOKEN_PREFILL_DEBT_S)
# is inert: nobody waits on a dedicated chunk anymore. A mixed batch's ride-along
# decodes run as plain extend rows (no verify upgrade; the spec runtime proposes
# drafts only on dedicated decode steps), and the running requests' GDN rows take
# the chunked-scan kernel instead of the recurrent decode kernel for the duration
# -- same math, different rounding.
def _mixed_prefill_tokens() -> int:
    raw = os.environ.get("FREETOKEN_MIXED_PREFILL_TOKENS", "0")
    try:
        return max(0, int(raw))
    except ValueError:
        raise ValueError(
            f"FREETOKEN_MIXED_PREFILL_TOKENS must be an integer, got {raw!r}"
        )


# FREETOKEN_ADMIT_SHORT_FIRST=N (port of STRATA_ADMIT_SHORT_FIRST): a pending prompt of
# at most N tokens takes this step's token budget before longer ones queued ahead of it;
# a long one that waited FREETOKEN_ADMIT_MAX_WAIT_S counts as short (no starvation).
def _admit_short_first() -> int:
    raw = os.environ.get("FREETOKEN_ADMIT_SHORT_FIRST", "0")
    try:
        return max(0, int(raw))
    except ValueError:
        raise ValueError(f"FREETOKEN_ADMIT_SHORT_FIRST must be an integer, got {raw!r}")


def _admit_max_wait_s() -> float:
    raw = os.environ.get("FREETOKEN_ADMIT_MAX_WAIT_S", "30")
    try:
        return max(0.0, float(raw))
    except ValueError:
        raise ValueError(f"FREETOKEN_ADMIT_MAX_WAIT_S must be a number, got {raw!r}")


# Adaptive chunk shaping: multiples of 64 keep the hybrid GDN x64 snapshot
# boundaries (prefill_chunk_align's reuse points); the floor stops a slow outlier
# chunk from collapsing the prompt into per-chunk-overhead-dominated slivers.
_CHUNK_ALIGN = 64
# Floor 128: at 256-token chunks this model's per-layer touched set (~132) times 48
# layers (~6.3k rows) OVERFLOWS the ~6k-slot LRU budget, and global LRU thrashes
# cyclically (layer L's promotions evict layer L+1's rows) -- measured: cross-chunk
# reuse collapses despite ~83% touched overlap, cadence 2x. At 128 the working set
# (~4-5k) fits and the promote reuse survives (see _scratch/agent3 RESULTS).
_CHUNK_MIN = 128

# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D | None  # (req_mapping, seq_lens or -1); None on verify, whose drain writes the pending token


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from freetoken.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        torch.cuda.set_stream(self.stream)
        # sent on the readiness ack for /v1/stats gpus; a list so TP can add one entry per rank
        self.gpus = [gpu_identity(self.device.index)] if self.device.type == "cuda" else []

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        self.spec: SpecRuntime | None = self.engine.spec
        # ONE cache manager for every model (ShadowRadix layering): the shared page table is the
        # virtual full-token coordinate; model-specific tiers ride the plug-ins -- DSV4's
        # window/cmp/idx shadows via swa_pool, Gemma's swa via swa_pool, GDN state via
        # linear_state_pool. No model supplies its own manager.
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type,
            linear_state_pool=self.engine.linear_state_pool,
            swa_pool=self.engine.kv_cache,
            mamba_host_cache_mb=config.mamba_host_cache_mb,
            sliding_window_size=next(
                (g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa),
                None,
            ) or getattr(self.engine.kv_cache, "sliding_window_size", None),
            radix_lookahead=self.spec is not None and self.spec.radix_lookahead,
        )
        token_headroom = config.decode_rows_per_req - 1
        self.decode_manager = DecodeManager(config.page_size, token_headroom=token_headroom)
        self._bidirectional_mm = any(getattr(g, "bidirectional_mm_blocks", False) for g in config.model_config.attention_groups)
        self.prefill_manager = PrefillManager(
            self.cache_manager,
            self.table_manager,
            self.decode_manager,
            encoder_cache=self.engine.encoder_cache,
            keep_images_whole=self._bidirectional_mm,
            short_first=_admit_short_first(),
            admit_max_wait_s=_admit_max_wait_s(),
            token_headroom=token_headroom,
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        # Abort acknowledgements are a terminal accounting barrier. Queue them while processing
        # inbound control messages, then flush only AFTER _process_last_data publishes any
        # sampled replies from the prior overlapped forward.
        self._pending_abort_acks: Set[int] = set()
        # With multiple tokenizer workers, an AbortBackendMsg and its earlier UserMsg can arrive
        # through different PUSH producers and be observed out of order. Preserve a bounded
        # tombstone so an abort-before-admission request can never be resurrected after its
        # terminal accounting acknowledgement has already been published.
        self._abort_tombstones: dict[int, None] = {}
        self._forward_iter = 0  # global forward counter; drives the SWA proactive-eviction cadence
        # The launched-but-not-yet-drained batch (overlap): set at the top of each overlap_loop
        # iteration so the abort handler can tell whether a request's forward is still in flight
        # (mark it, defer the free to _process_last_data) or not (free immediately). Stays None
        # in normal_loop, where a batch launches and drains within one iteration.
        self._last_data: ForwardData | None = None
        # A received-but-not-yet-executed runtime cache rebuild (CacheRebuildBackendMsg),
        # run at the next idle safe point in overlap_loop. None when no rebuild is pending.
        self._pending_rebuild: CacheRebuildBackendMsg | None = None
        # A /v1/cache/flush waiting for the idle safe point, with its time.monotonic() deadline.
        self._pending_flush: Tuple[FlushCacheBackendMsg, float] | None = None
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
        from freetoken.engine.think_wrap import ThinkWrap

        self._think_wrap = ThinkWrap(self.tokenizer, config)
        self.engine._think_wrap = self._think_wrap
        self.toolcall_anchor_id = None
        if config.special_token_ckpt and (
            self.cache_manager.is_hybrid or self.cache_manager.is_swa
        ):
            from freetoken.server.function_call_parser import toolcall_opener_for

            self.toolcall_anchor_id = load_toolcall_anchor_id(
                self.tokenizer,
                toolcall_opener_for(getattr(config, "tool_call_parser", "")),
            )
        self.token_pool = self.table_manager.token_pool
        # Floor the prefill chunk by the cache manager's cap (DSV4: ~half the window pool) so a
        # sliding-window cache chunks long prompts and frees out-of-window pages between chunks
        # instead of OOMing _alloc_window on a prompt longer than the window pool.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(config.max_extend_tokens, _chunk_cap) if _chunk_cap else config.max_extend_tokens
        )
        self.config = config
        self._model_is_mrope = config.model_config.model_is_mrope
        self._warned_cut_image = False
        self._prefill_streak = 0
        self._prefill_debt = 0.0  # prefill seconds accrued while a decode was runnable
        # The in-flight chunk that already paid for the last concede, so its drain cannot
        # re-open the debt it bought (one chunk == at most one concede, see _inflight_prefill).
        self._debt_discharge = None
        self._prefill_debt_s = _prefill_debt_budget()
        self._mixed_prefill_tokens = _mixed_prefill_tokens()
        # Piped prefill (FREETOKEN_PIPED_PREFILL, Strata PIPED_CONCURRENT port): prefill
        # chunks run on the pipe thread's own default-priority stream while this loop
        # keeps issuing decode steps on the high-priority engine stream. It supersedes
        # mixed batching (decodes ride every REPLAY now, not every chunk), so enabling
        # it clears the mixed cap.
        self._pipe = None
        self._pipe_take = None
        if getattr(self.engine, "piped_prefill", False):
            self._pipe = self._make_pipe()
            if self._pipe is not None:
                self._mixed_prefill_tokens = 0
        self._chunk_target_s = _adaptive_chunk_target_s()
        self._chunk_ema_spt: float | None = None  # EMA of prefill seconds per token
        self._warned_adaptive_tp = False
        self._pipe_trace_n = 0  # one-time note: the EMA shrink is TP-unsafe
        # Opt-in per-chunk wall-clock breakdown ([chunktime] log lines). Independent of
        # the moe _debug_stats probe (which must not be enabled on PLE-graph rigs).
        self._chunk_timing = os.environ.get("FREETOKEN_CHUNK_TIMING", "0") == "1"
        self._ct_chunks = 0
        moe_cache = getattr(self.engine, "moe_offload_cache", None)
        self.status_reporter = SchedulerStatusReporter(
            log=logger.info_rank0,
            decode_log_interval=config.decode_log_interval,
            moe_stats=self._moe_stats_line if config.moe_collect_stats and moe_cache is not None else None,
        )

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def _moe_stats_line(self) -> str | None:
        # on the engine stream, so the read and the zeroing fall between decode replays
        with torch.cuda.stream(self.engine.stream):
            return self.engine.moe_offload_cache.decode_stats_line()

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    @torch.inference_mode()
    def rebuild_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
    ) -> None:
        """Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state
        pool, and/or the window pool (num_swa_pages), re-capture CUDA graphs, and re-thread the
        page managers (clearing the prefix cache on a KV/mamba/window resize). The caller MUST
        guarantee the scheduler is idle — no pending prefill, no running decode, no in-flight
        finished requests. All TP ranks must call this with identical arguments.
        """
        assert not self.prefill_manager.runnable, "rebuild requires no pending prefill"
        assert not self.decode_manager.runnable, "rebuild requires no running decode"
        torch.cuda.synchronize(self.device)
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()
        self.engine.rebuild_runtime_cache(
            moe_cache_size=moe_cache_size, num_pages=num_pages, num_mamba_slots=num_mamba_slots,
            num_swa_pages=num_swa_pages,
        )
        if num_pages is not None or num_mamba_slots is not None or num_swa_pages is not None:
            # Any of these resizes invalidates the prefix cache: a KV resize leaves stale page
            # indices, a mamba resize leaves stale GDN-snapshot slot ids, and a window-pool resize
            # (num_swa_pages) reallocates the SWA/window token pool, leaving stale slot ids in the
            # radix tree. Rebuild the prefix cache + reclaim the resized free-lists.
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
            if num_pages is not None:
                # token_pool is sized to the page table; only a KV-page resize reallocates it.
                # A mamba-only rebuild leaves the page table untouched, so skip this (else it
                # needlessly reallocates + zeros the whole GPU token_pool every mamba resize).
                self.table_manager.rebuild(self.engine.page_table)
                self.token_pool = self.table_manager.token_pool
            self.cache_manager.check_integrity()
            # A rebuild can reallocate the page/token/state pools whose addresses the
            # prefill chunk graph baked in (token_pool/page_table gathers): drop it so
            # lazy recapture follows.
            _pg_runner = getattr(self.engine, "_prefill_graph", None)
            if _pg_runner is not None:
                _pg_runner.invalidate()
        # The prefill chunk cap tracks the CURRENT window-pool size (DSV4); a rebuild that
        # shrank the pool must shrink the cap too, or the next long prompt is chunked against
        # the stale budget and crashes _alloc_window.
        _chunk_cap = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = (
            min(self.config.max_extend_tokens, _chunk_cap)
            if _chunk_cap else self.config.max_extend_tokens
        )
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # Expose the un-drained batch to _process_one_msg (abort in-flight check). Assigning
        # before the message loop is what makes the check airtight: the batch launched later
        # this iteration can only be probed by messages of the NEXT iteration, which sees it here.
        self._last_data = last_data
        from freetoken.moe import _debug_stats
        from freetoken.engine import latency

        _dbg = _debug_stats.probe()
        _lat = latency.probe()
        if _lat is not None:
            _lat.attach(self)
        if _dbg is not None or _lat is not None:
            import time as _time

            _t = _time.perf_counter()
        _ct = self._chunk_timing
        if _ct:
            _ct0 = time.perf_counter()
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to drain toward + execute
            or self._pending_flush is not None  # a deferred flush to run at idle or time out
            # a piped chunk holds pages/slots and owes a drain: the loop is NOT idle,
            # and blocking here would let run_when_idle's integrity check see the
            # chunk's own allocations as a leak (and stall the drain behind messages)
            or (self._pipe is not None and (self._pipe.inflight or self._pipe_take is not None))
        )
        if _PIPED_TRACE and self._pipe is not None:
            _pt_pre = (int(self._pipe.inflight), int(self._pipe.slot is not None),
                       int(self._pipe_take is not None))
        _msgs = self.receive_msg(blocking=blocking)
        # the admission aging / queue stamps below must read rank0's broadcast clock
        self.prefill_manager.sched_clock = getattr(self, "_tp_sched_clock", None)
        for msg in _msgs:
            self._process_one_msg(msg)
        if _PIPED_TRACE and self._pipe is not None:
            self._pipe_trace_n += 1
            logger.info(
                f"[pipetrace] it={self._pipe_trace_n} block={int(blocking)} "
                f"pre={_pt_pre} msgs={len(_msgs)} "
                f"bcast={int(getattr(self, '_pipe_ready_bcast', False))} "
                f"post=({int(self._pipe.inflight)},{int(self._pipe.slot is not None)},"
                f"{int(self._pipe_take is not None)}) "
                f"prun={int(self.prefill_manager.runnable)} "
                f"drun={int(self.decode_manager.runnable)} last={int(last_data is not None)}"
            )
        if _ct:
            _ct1 = time.perf_counter()
        if _dbg is not None or _lat is not None:
            _n = _time.perf_counter()
            if _dbg is not None:
                _dbg.host_phase("recv", _n - _t)
            if _lat is not None:
                _lat.host_phase("recv", _n - _t)
            _t = _n

        # Execute a queued cache rebuild once the scheduler is fully idle (the safe point):
        # no last batch to process, no pending prefill, no running decode. finished_reqs is
        # NOT a gate — those requests are already freed (no live GPU/page resources).
        if self._pending_rebuild is not None and last_data is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
            or (self._pipe is not None and self._pipe.inflight)
        ):
            self._execute_pending_rebuild()
        self._check_pending_flush()

        # Order this iteration's host->device token_pool copies (issued on ``self.stream``
        # during scheduling) after the previous batch's sampled-token writes (issued on the
        # engine stream in ``_forward``). Without this, a request that reuses a just-freed
        # table_idx can have its freshly copied prompt clobbered by the prior occupant's
        # still-pending output write -- corrupting tokens (e.g. dropping an image
        # placeholder, which the multimodal merge then rejects).
        self.stream.wait_stream(self.engine.stream)
        # Piped prefill: harvest a finished chunk BEFORE deciding this iteration's
        # batches (can_take() must see the slot as free) and order the staging stream
        # behind the chunk's pool write. A dead worker discards its chunk and the
        # loop reverts to inline prefill steps.
        piped_drain = None
        if self._pipe is not None:
            if self._pipe.stalled():
                # a chunk older than the hard ceiling means the worker is wedged
                # (a rendezvous bug would pin the GPU forever, invisible to the
                # stall watchdog); crash so the supervisor restarts cleanly.
                logger.warning_rank0(
                    f"piped prefill chunk in flight over {self._pipe.timeout_s:.0f}s: "
                    "the worker is wedged, restarting the engine"
                )
                raise RuntimeError("piped prefill stalled past the hard timeout")
            # the drain gate is the broadcast flag: rank0 fills it with its own
            # readiness, every other rank obeys it - one iteration, all ranks drain
            # (poll(force) waits our worker's publish and the drain's copy_done
            # synchronize absorbs GPU-side skew; strict also SUPPRESSES the rank-local
            # drain our own chunk_done would allow an iteration early)
            got = self._pipe.poll(
                force=bool(getattr(self, "_pipe_ready_bcast", False)),
                strict=self.config.tp_info.size > 1,
            )
            if got is not None:
                fin, fout, err = got
                if err is not None:
                    self._discard_piped_chunk(fin, err)
                    from freetoken.distributed import get_tp_info

                    if get_tp_info().size > 1:
                        # a rank-local fallback would run the next chunks' collectives
                        # on the wrong comm while the peer pipes on: crash and let the
                        # supervisor restart both ranks together.
                        raise err
                    self._pipe = None
                else:
                    self.stream.wait_event(self._pipe.chunk_done)
                    piped_drain = (fin, fout)
            elif self._pipe.inflight and last_data is None and not self.decode_manager.runnable:
                # the pipe is the only work: wait it out instead of spinning the loop
                # (message latency stays at one chunk, today's inline cadence). An
                # unrecorded event would answer instantly, so wait the cv for the
                # publish and only then the GPU point.
                if self._pipe.slot is None:
                    self._pipe.wait_slot()
                else:
                    self._pipe.chunk_done.synchronize()
        forward_input = self._schedule_next_batch()
        if _ct:
            _ct2 = time.perf_counter()
        if _dbg is not None or _lat is not None:
            _n = _time.perf_counter()
            if _dbg is not None:
                _dbg.host_phase("sched", _n - _t)
            if _lat is not None:
                _lat.host_phase("sched", _n - _t)
            _t = _n
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                # COW-restore GDN snapshots for prefix hits ON THE ENGINE STREAM, after the
                # cross-stream wait and before the forward reads the live slot (program order
                # vs the prior batch's snapshot writes). Doing this on self.stream would race.
                if _dbg is not None:
                    _n0 = _time.perf_counter()
                self._restore_linear_states(forward_input.batch)
                if _dbg is not None:
                    _n1 = _time.perf_counter()
                    _dbg.host_phase(
                        ("p." if forward_input.batch.is_prefill else "d.") + "fw.restore",
                        _n1 - _n0)
                ongoing_data = (forward_input, self._forward(forward_input))
                if self._pipe is not None:
                    # Strata's launch_gate: the chunk's kernel storm waits behind THIS
                    # launch, so prefill queueing can never delay the replay's launch.
                    self._pipe.record_gate()
                if _dbg is not None:
                    _n2 = _time.perf_counter()
                    _dbg.host_phase(
                        ("p." if forward_input.batch.is_prefill else "d.") + "fw.total",
                        _n2 - _n1)
        if self._pipe is not None and self._pipe_take is not None:
            take, self._pipe_take = self._pipe_take, None
            self._pipe.handoff(take)
        if _ct:
            _ct3 = time.perf_counter()
        if _dbg is not None or _lat is not None:
            _n = _time.perf_counter()
            if _dbg is not None:
                _dbg.host_phase("fwd_issue", _n - _t)
            if _lat is not None:
                _lat.host_phase("fwd_issue", _n - _t)
            _t = _n

        # The drain issues GPU-visible writes to state the batch just launched still reads: the
        # page-table re-point and, for the paged-SWA pools, the full->swa (DSV4: full->window)
        # sentinel scatter. DSV4 stages the page table at replay time and translates
        # full_to_window INSIDE the captured graph, so an unordered drain can redirect an
        # in-flight forward. copy_done only covers batch N; order against N+1 explicitly.
        self.stream.wait_stream(self.engine.stream)
        try:
            self._process_last_data(last_data)
            self._flush_abort_acks()
            if piped_drain is not None:
                # the chunk's sampled row joins running only now (its forward was not
                # gated by this loop); the drain delivered its first token first
                main_finished = self.finished_reqs
                self._process_last_data(piped_drain)
                # keep the main drain's finish set alive: _process_last_data replaces
                # it, and the next iteration's overlap-extra-step skip covers both
                self.finished_reqs = self.finished_reqs | main_finished
                self.decode_manager.filter_reqs(piped_drain[0].batch.reqs)
                if self._chunk_timing:
                    self._log_chunk_timing(piped_drain, (0.0, 0.0, 0.0, 0.0))
        finally:
            # Run the just-issued batch's deferred PLE fill AFTER the drain: its readback
            # event sits a hair past the drain's copy_done point in stream order (so the
            # wait is ~0 here), while the replay's captured memop WAIT keeps the ordering.
            # In a finally so a drain failure can never leave the WAIT unanswered.
            self.engine.run_pending_host_fill()
        if _ct:
            self._log_chunk_timing(last_data, (_ct1 - _ct0, _ct2 - _ct1,
                                               _ct3 - _ct2, time.perf_counter() - _ct3))
        if _dbg is not None or _lat is not None:
            _n = _time.perf_counter()
            if _dbg is not None:
                _dbg.host_phase("drain", _n - _t)
            if _lat is not None:
                _lat.host_phase("drain", _n - _t)
            # device-syncing stats dump ONLY here: past the deferred fill, so no
            # un-signaled PLE WAIT can be queued ahead of the sync
            if _dbg is not None:
                _dbg.dump("loop")
            # the latency probe is host-timing only, so the same safe point serves it
            if _lat is not None:
                _lat.maybe_dump()
        return ongoing_data

    def _log_chunk_timing(self, last_data, phases) -> None:
        """One [chunktime] line per drained prefill chunk: window = schedule->drain wall
        (the victim-perceived cadence), plus THIS iteration's host phases (which belong
        to the next batch's issue cycle) and 32-chunk PLE row-cache deltas."""
        batch = last_data[0].batch if last_data is not None else None
        if batch is None or not batch.is_prefill:
            return
        self._ct_chunks += 1
        window_ms = (time.perf_counter() - batch.scheduled_at) * 1e3 if batch.scheduled_at else -1.0
        recv, sched, issue, drain = (x * 1e3 for x in phases)
        mix = getattr(batch, "decoding_reqs", None)
        mix_s = f" mix={len(mix)}" if mix else (" pipe" if getattr(batch, "piped", False) else "")
        promote = ""
        moc = getattr(self.engine, "moe_offload_cache", None)
        if moc is not None:
            cur = getattr(moc, "_promote_host_ms", 0.0)
            base = getattr(self, "_ct_promote_base", 0.0)
            promote = f" promote={cur - base:.0f}ms"
            self._ct_promote_base = cur
        ple = ""
        if self._ct_chunks % 32 == 0:
            table = getattr(self.engine.model, "_ple_table", None)
            store = getattr(table, "_store", None)
            if store is not None:
                try:
                    st = store.cache_stats()
                    ple = (f" ple[hits={st.get('hits')} miss={st.get('misses')}"
                           f" evict={st.get('evicts')} slots={st.get('slots')}]")
                except Exception:  # noqa: BLE001 -- instrumentation must never break the loop
                    pass
        logger.info_rank0(
            f"[chunktime] #{self._ct_chunks} T={batch.log_new_tokens} "
            f"window={window_ms:.0f}ms recv={recv:.1f} sched={sched:.1f} "
            f"issue={issue:.1f} drain={drain:.1f}{mix_s}{promote}{ple}"
        )

    def normal_loop(self) -> None:
        _ct = self._chunk_timing
        if _ct:
            _ct0 = time.perf_counter()
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to execute at idle
            or self._pending_flush is not None  # a deferred flush to run at idle or time out
        )
        _msgs = self.receive_msg(blocking=blocking)
        self.prefill_manager.sched_clock = getattr(self, "_tp_sched_clock", None)
        for msg in _msgs:
            self._process_one_msg(msg)
        if _ct:
            _ct1 = time.perf_counter()

        # Non-overlap mode has no last_data to drain; execute a queued rebuild as soon as
        # the scheduler is idle (no pending prefill / running decode). Without this, a
        # rebuild in DISABLE_OVERLAP_SCHEDULING mode stays pending until the HTTP timeout.
        if self._pending_rebuild is not None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()
        self._check_pending_flush()

        forward_input = self._schedule_next_batch()
        if _ct:
            _ct2 = time.perf_counter()
        ongoing_data = None
        if forward_input is not None:
            # already inside engine_stream_ctx (run_forever); restore on the engine stream
            self._restore_linear_states(forward_input.batch)
            ongoing_data = (forward_input, self._forward(forward_input))
        if _ct:
            _ct3 = time.perf_counter()

        # Non-overlap drains the batch it just issued: the deferred PLE fill MUST run
        # first -- this batch's own replay is WAITing on its flag, so draining before
        # the fill would deadlock (drain waits replay, replay waits fill, fill waits drain).
        self.engine.run_pending_host_fill()
        self._process_last_data(ongoing_data)
        self._flush_abort_acks()
        if _ct:
            self._log_chunk_timing(ongoing_data, (_ct1 - _ct0, _ct2 - _ct1,
                                                  _ct3 - _ct2, time.perf_counter() - _ct3))

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # DSV4 (owned-KV) decode reads its per-token window/cmp/idx slot maps off the attention
        # backend's per-batch SNAPSHOT (staged in prepare_for_replay right before the replay, on
        # the same stream, like the generic out_loc copy_from), not the live slot maps -- so the
        # next batch's allocate_paged cannot corrupt the in-flight graph replay. DSV4 overlaps.
        if ENV.DISABLE_OVERLAP_SCHEDULING or self.spec is not None:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return
        from freetoken.moe import _debug_stats

        _dbg = _debug_stats.probe()
        if _dbg is not None:
            import time as _time

            _t0 = _time.perf_counter()

        batch, output = last_data[0].batch, last_data[1]
        # rows of running requests riding a mixed prefill batch (see _mix_decodes): they
        # drain as decode tokens, and only the prompt side is radix-committed below
        decoding = getattr(batch, "decoding_reqs", None) or ()
        _, next_tokens_cpu, copy_done = output[:3]
        copy_done.synchronize()
        if _dbg is not None:
            # split the GPU tail (the copy_done wait) from the host work that follows
            _n = _time.perf_counter()
            _dbg.host_phase("drain.sync", _n - _t0)
            _t0 = _n
        # Signal the in-flight batch's PLE rows at the earliest safe instant: its
        # readback event is a hair past this copy_done point, and every host us
        # spent here before the fill is a us the replay idles at its captured WAIT.
        # (normal_loop drains the batch it just issued and runs the fill BEFORE this
        # point, so this is a no-op there -- draining first would deadlock.)
        self.engine.run_pending_host_fill()
        self._charge_prefill_time(batch)
        # tokens are on the host from here: the per-request timings' token stamps
        now = time.perf_counter()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        generated = None
        with self.cache_manager.lazy_free_region():
            if batch.phase == "verify":
                generated = self._drain_verify(batch, output, reply, new_finished_reqs)
            else:
                for i, req in enumerate(batch.reqs):
                    if isinstance(req, ChunkedReq):
                        # Don't cache intermediate chunks; the full prompt is cached once when the
                        # final chunk is processed. Caching here snapshots a handle the next chunk
                        # already copied (overlap), so cache_req double-frees the prior chunk.
                        if req.aborted:
                            # Aborted mid-chunked-prefill while this chunk was in flight: the abort
                            # popped the pending continuation (no next chunk launches), and this
                            # drain point frees the chunk's pages/slots exactly once.
                            self._free_req_resources(req)
                        continue
                    if req.aborted:
                        # Aborted while this final-chunk prefill / decode step was in flight: free
                        # here (the forward is drained) and finish the request. No DetokenizeMsg --
                        # the abort ack flushed after this method stays the uid's terminal reply.
                        self.decode_manager.remove_req(req)
                        self._free_req_resources(req)
                        new_finished_reqs.add(req)
                        continue
                    if req in self.finished_reqs:
                        # Overlap scheduling launched one more decode step for a request that
                        # already terminated (filter_reqs keeps it while output budget remains,
                        # and the next batch is scheduled before this drain runs). Its resources
                        # are freed below/already; shipping this token would append past the
                        # client's terminal reply.
                        continue
                    msg, _ = self._commit_tokens(req, next_tokens_cpu[i : i + 1])
                    req.stats.stamp_token(now)
                    reply.append(msg)
                    # a timings_per_token request gets the snapshot on every reply, growing
                    if msg.finished or req.sampling_params.timings_per_token:
                        msg.timings = req.stats.finish(req.input_ids.numel() - req.prompt_len)
                    finished = msg.finished

                    # NOTE: overlap scheduling may make the request freed twice, skip second free
                    if finished and req not in self.finished_reqs:
                        self.decode_manager.remove_req(req)
                        self._free_req_resources(req)
                        new_finished_reqs.add(req)
                    elif batch.is_prefill and req not in decoding and req.table_idx != -1:
                        # for prefill, non-chunk req, cache the prefix. A mixed batch's ride-along
                        # decodes are NOT prompt completions: their prefix commit happens at finish.
                        # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
                        # the generic manager inserts the prefix into its radix/naive cache.
                        # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
                        # instead of freeing them (handled above), so a freed request should
                        # never reach this commit -- but if a future path frees one early, skip
                        # rather than re-read the freed page-table row (and on hybrid, deref the
                        # None'd GDN ping-pong slots).
                        self.cache_manager.cache_req(req, finished=False)

        self.finished_reqs = new_finished_reqs
        self._account_prefill_debt(batch)
        self._observe_chunk_time(batch)
        # Stamp each reply with the post-batch KV page occupancy so the frontend (shell
        # status bar) can show live KV usage without a separate query.
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            kv_cached = self.cache_manager.evictable_kv_pages
            mamba_cached = self.cache_manager.evictable_mamba_slots
            swa_cached = self.cache_manager.evictable_swa_tokens
            moe_used, moe_live, moe_total = self._moe_residency()
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.kv_cached_pages = kv_cached
                m.mamba_cached_slots = mamba_cached
                m.swa_cached_tokens = swa_cached
                m.moe_used_slots = moe_used
                m.moe_active_slots = moe_live
                m.moe_total_slots = moe_total
                m.gpu_mem_bytes = mem
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
            generated=generated,
        )
        self.send_result(reply)
        if _dbg is not None:
            _dbg.host_phase("drain.work", _time.perf_counter() - _t0)

    def _charge_prefill_time(self, batch: Batch) -> None:
        """Charge this prefill batch's schedule->drain window to the prompts it forwarded.

        The window is the forward's, not one prompt's, so a shared batch splits it by the
        tokens each prompt snapshotted when the batch was scheduled, and the per-request parts
        still add up to the wall time. Chunked prompts accumulate: one prompt's prefill spans
        several batches, and only its last chunk samples a token, which is the reply that
        carries the total.
        """
        if (
            not batch.is_prefill
            or not getattr(batch, "scheduled_at", 0.0)
        ):
            return
        weights = [req.prefill_chunk_tokens for req in batch.reqs]
        # A mixed batch's ride-along decodes still carry their own admission's snapshot;
        # they forwarded no prompt, so this window belongs to the prompt side alone.
        decoding = getattr(batch, "decoding_reqs", None)
        if decoding:
            weights = [
                0 if req in decoding else w for req, w in zip(batch.reqs, weights)
            ]
        total = sum(weights)
        if total <= 0:
            return
        window_ms = (time.perf_counter() - batch.scheduled_at) * 1e3
        for req, weight in zip(batch.reqs, weights):
            req.prefill_ms += window_ms * weight / total

    def _sched_now(self) -> float:
        """The clock scheduling thresholds compare. Under TP it is rank0's recv-time
        clock broadcast with the message count (io.py): a local read would let the
        admission-aging / prefill-debt thresholds flip on one rank an iteration before
        the other, composing different batches and wedging a collective. At tp1 the
        local clock stays exact."""
        clock = getattr(self, "_tp_sched_clock", None)
        if clock is not None and self.config.tp_info.size > 1:
            return clock
        return time.perf_counter()

    def _account_prefill_debt(self, batch) -> None:
        """Accrue the wall time a prefill batch held the engine while decodes waited.

        Schedule->drain is the window a victim perceives as its inter-token stall.
        Only counted while a decode is actually runnable, so a pure-prefill stretch
        with nobody waiting accrues nothing (and cannot suppress the next admission).
        """
        if (
            self._prefill_debt_s <= 0
            or not batch.is_prefill
            or not getattr(batch, "scheduled_at", 0.0)
            or not self.decode_manager.runnable
        ):
            return
        discharged = self._debt_discharge
        if discharged is not None:
            self._debt_discharge = None
            if batch is discharged:
                return  # this chunk bought the concede the guard just granted
        # a mixed batch ran the waiting decodes inside its own forward: it stalls nobody
        if getattr(batch, "decoding_reqs", None) or getattr(batch, "piped", False):
            # piped: the chunk ran on its own stream while decodes kept replaying
            return
        self._prefill_debt += self._sched_now() - batch.scheduled_at

    def _inflight_prefill(self) -> tuple[Batch | None, float]:
        """The launched-but-undrained prefill chunk and how long it has been running.

        ``overlap_loop`` drains ``_last_data`` a few lines AFTER deciding the next batch, so
        ``_prefill_debt`` is one chunk stale at the decision and one slow chunk stalls a
        waiting decode for TWO chunks. Counting the chunk in flight bounds the stall at one;
        ``_account_prefill_debt`` then discharges it, so one chunk == one concede. Where the
        already-booked debt alone also owed, that discharge skips a chunk which had not paid
        for the concede: at most one un-charged chunk each time, the price of a fresh view.
        Gate order is load-bearing -- stubs built with ``Scheduler.__new__`` have no
        ``_last_data`` and rely on ``_prefill_debt_s`` being checked first.
        """
        if self._prefill_debt_s <= 0 or not self.decode_manager.runnable:
            return None, 0.0
        data = self._last_data
        if data is None:
            return None, 0.0
        batch = data[0].batch
        if not batch.is_prefill:
            return None, 0.0
        if getattr(batch, "decoding_reqs", None):
            return None, 0.0  # mixed: the waiting decodes rode this one, they stalled for nobody
        if not getattr(batch, "scheduled_at", 0.0):
            return None, 0.0  # unstamped: now - 0 would concede unconditionally
        return batch, self._sched_now() - batch.scheduled_at

    def _observe_chunk_time(self, batch) -> None:
        """EMA of prefill seconds-per-token, from the same schedule->drain window the
        debt accounts. Feeds the adaptive chunk budget; one update per prefill batch."""
        if (
            self._chunk_target_s <= 0
            or not batch.is_prefill
            # a mixed window carries decode rows too: the chunk-size calibration stays
            # on pure chunks, which is what this budget will size again
            or getattr(batch, "decoding_reqs", None)
            or not getattr(batch, "scheduled_at", 0.0)
        ):
            return
        tokens = getattr(batch, "log_new_tokens", 0)
        if tokens <= 0:
            return
        spt = (self._sched_now() - batch.scheduled_at) / tokens
        ema = self._chunk_ema_spt
        self._chunk_ema_spt = spt if ema is None else 0.7 * ema + 0.3 * spt

    def _adaptive_prefill_budget(self) -> int:
        """Token budget for this scheduling pass.

        While a decode is waiting AND a chunk-time target is configured, shrink the
        chunk so its forward time tracks the target: the waiting user's perceived stall
        is one chunk (+ its concede step), so the target IS the stall budget. Nobody
        waiting -> the full configured budget (a solo huge prompt keeps its TTFT, and
        the shrink costs the aggressor expert re-stream per chunk boundary). Chunks are
        aligned to _CHUNK_ALIGN and floored at _CHUNK_MIN so per-chunk fixed overhead
        (PLE fill, staging, fence) cannot dominate.
        """
        if (
            self._chunk_target_s <= 0
            or self._chunk_ema_spt is None
            or self._chunk_ema_spt <= 0
            or not self.decode_manager.runnable
        ):
            return self.prefill_budget
        if self.config.tp_info.size > 1:
            # The EMA is THIS rank's own seconds-per-token, so the shrink would size the
            # same chunk differently on each rank: mismatched collective shapes, not a
            # slow step. Under TP every rank keeps the configured budget.
            if not getattr(self, "_warned_adaptive_tp", False):
                self._warned_adaptive_tp = True
                logger.warning_rank0(
                    "FREETOKEN_ADAPTIVE_CHUNK_TARGET_S is ignored under TP: a per-rank "
                    "chunk-size EMA would desynchronize the ranks' collectives"
                )
            return self.prefill_budget
        tok = int(self._chunk_target_s / self._chunk_ema_spt)
        tok = (tok // _CHUNK_ALIGN) * _CHUNK_ALIGN
        # the floor must never exceed the configured budget (small-budget configs/tests)
        return max(min(_CHUNK_MIN, self.prefill_budget), min(tok, self.prefill_budget))

    def _effective_debt_s(self) -> float:
        """Debt threshold for the interleave guard.

        In adaptive mode the threshold drops with the chunk target so a decode step is
        conceded after (nearly) every shrunken chunk -- gap ~= one chunk forward, which
        is the whole point of shrinking. Otherwise the configured static debt applies.
        """
        if self._prefill_debt_s <= 0:
            return self._prefill_debt_s
        if (
            self._chunk_target_s > 0
            and self._chunk_ema_spt is not None
            and self.decode_manager.runnable
        ):
            return min(self._prefill_debt_s, max(self._chunk_target_s * 0.5, 0.05))
        return self._prefill_debt_s

    def _drain_verify(self, batch: Batch, output, reply: List[DetokenizeMsg], new_finished_reqs: Set[Req]) -> int:
        """Commit each request's accepted tokens, return the slots its rejected rows took, then the target state; returns the tokens committed."""
        k = self.config.decode_rows_per_req - 1
        counts, finished = [], []
        now = time.perf_counter()
        for i, req in enumerate(batch.reqs):
            # spec forces normal_loop, which frees an aborted request between steps; overlap would need rollback_tail here and must not donate the GDN slot the verify rolled
            assert not req.aborted, "a verify batch drained an aborted request"
            cached = req.cached_len
            emitted = int(output.emitted_cpu[i])
            msg, n = self._commit_tokens(req, output.next_tokens_cpu[i, : emitted + 1])
            # k drafts were proposed; the step hands _commit_tokens the emitted accepted
            # drafts plus the target's own token, so when a stop lands on a draft every
            # committed token is an accepted draft: count min(n, emitted), with n already
            # post stop-truncation. Never exceeds predicted_n - 1: the request's first
            # committed token came from prefill, outside every verify step.
            req.stats.stamp_token(now)
            req.stats.draft_n += k
            req.stats.draft_n_accepted += min(n, emitted)
            reply.append(msg)
            # a timings_per_token request gets the snapshot on every reply, growing
            if msg.finished or req.sampling_params.timings_per_token:
                msg.timings = req.stats.finish(req.input_ids.numel() - req.prompt_len)
            req.commit_pending(n)
            self.cache_manager.rollback_tail(req, req.cached_len, cached + k + 1)
            counts.append(n)
            if msg.finished:
                finished.append(req)
        # before any finish free: the hybrid finish donates the live GDN slot the commit writes
        self.engine.commit_verify(batch.reqs, counts, output.next_tokens_gpu)
        # the last committed token is pending at cached_len, like a decode step's sampled token
        table, pos, last = torch.tensor(
            [(r.table_idx, r.cached_len, n - 1) for r, n in zip(batch.reqs, counts)],
            dtype=torch.int64, pin_memory=self.device.type == "cuda",
        ).to(self.device, non_blocking=True).unbind(1)
        self.token_pool[table, pos] = output.next_tokens_gpu.gather(1, last.unsqueeze(1)).squeeze(1).to(self.token_pool.dtype)
        for req in finished:
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)
            new_finished_reqs.add(req)
        return sum(counts)

    def _commit_tokens(self, req: Req, tokens: torch.Tensor) -> Tuple[DetokenizeMsg, int]:
        """Append sampled tokens in order until one finishes the request; returns the reply and how many tokens were committed."""
        params = req.sampling_params
        committed: List[int] = []
        finished, finish_reason, matched_stop = False, None, None
        _tw = getattr(self, "_think_wrap", None)
        for j in range(tokens.numel()):
            req.append_host(tokens[j : j + 1])
            token = int(tokens[j].item())
            committed.append(token)
            # thinking wrap-up: track the think block as tokens land (the force
            # decision itself runs at sample time in the engine)
            if _tw is not None:
                _tw.observe(req, token)
            # EOS / stop-string -> "stop", output budget exhausted -> "length";
            # EOS and stop strings win over length.
            # Overlap can advance device_len one token ahead of this host reply.
            hit_length = req.input_ids.numel() >= req.max_device_len
            hit_eos = not params.ignore_eos and token in self.eos_token_ids
            matched_stop = self._match_stop_str(req) if not hit_eos and params.stop_strs else None
            if hit_length or hit_eos or matched_stop is not None:
                finished = True
                finish_reason = "stop" if (hit_eos or matched_stop is not None) else "length"
                break
            if token == self.toolcall_anchor_id and req.toolcall_anchor_len is None:
                req.toolcall_anchor_len = req.input_ids.numel()
        msg = DetokenizeMsg(
            uid=req.uid,
            next_tokens=committed,
            finished=finished,
            finish_reason=finish_reason,
            matched_stop=matched_stop,
            stop_strs=params.stop_strs or None,
        )
        return msg, len(committed)

    def _match_stop_str(self, req: Req) -> str | None:
        """First stop string present in this request's generated tail, else None. Decodes
        only a short suffix (bounded by the longest stop string's char length, so a stop of
        N chars spans at most N tokens) to keep the per-step cost small."""
        stop_strs = req.sampling_params.stop_strs
        prompt_len = req.max_device_len - req.output_len
        if len(req.input_ids) <= prompt_len:
            return None
        max_chars = max(len(s) for s in stop_strs)
        tail_start = max(prompt_len, len(req.input_ids) - (max_chars + 1))
        tail = self.tokenizer.decode(req.input_ids[tail_start:].tolist())
        for s in stop_strs:
            if s in tail:
                return s
        return None

    def _kv_usage_pages(self) -> Tuple[int, int]:
        """(used_pages, total_pages) of the KV page pool.

        ``used`` follows SGLang's logging semantics: allocated pages that are not
        evictable (active requests + protected prefix cache). Evictable prefix-cache
        pages are available to future requests, so they are excluded from usage.
        Always the manager's own primary pool (for DSV4 the FULL cmp/idx tier); the
        window (swa) tier is reported separately by ``_swa_token_usage``.
        """
        return self.cache_manager.page_usage()

    def _mamba_slot_usage(self) -> Tuple[int, int] | None:
        """(used_slots, total_slots) of the GDN-state (mamba) pool for hybrid models, else None.

        Mirrors SGLang's mamba-pool semantics: ``total`` excludes the reserved padding
        sink (slot 0); ``used`` excludes free slots and evictable tree snapshots.
        """
        if not self.cache_manager.is_hybrid:
            return None
        total = self.cache_manager.linear_state_pool.num_slots - 1
        return total - self.cache_manager.mamba_available_size, total

    def _swa_token_usage(self) -> Tuple[int, int] | None:
        """(used_tokens, total_tokens) of the window (swa) pool for SWA models, else None.

        Mirrors the mamba accounting: ``total`` excludes the pool's reserved sentinel
        unit; ``used`` excludes free slots and evictable (unlocked) tree tokens.
        """
        cm = self.cache_manager
        if not cm.swa_paged:
            return None
        total = cm.swa_pool.swa_num_tokens - 1
        return total - cm.swa_available_size, total

    def _moe_residency(self) -> Tuple[int, int, int]:
        """(filled_slots, live_slots, total_slots) of the MoE expert slot cache, else 0s.
        ``live`` is the last forward's working set, the only part of an LRU expert cache
        anything is actually reading. The slot map lives on the GPU, so refresh at most every
        16th stamped batch and serve the cached value in between: the gauge feeds the 3s
        dashboard, never control."""
        moc = getattr(self.engine, "moe_offload_cache", None)
        if moc is None:
            return 0, 0, 0
        self._moe_res_step = getattr(self, "_moe_res_step", -1) + 1
        if self._moe_res_step % 16 == 0:
            self._moe_res = (*moc.residency_split(), moc.cache_size)
        filled, live, total = getattr(self, "_moe_res", (0, 0, moc.cache_size))
        if not (self.decode_manager.running_reqs or self.prefill_manager.pending_list):
            # Idle: no forward is reading experts, so every held slot is warm cache -- and the
            # cached tuple above would otherwise freeze the last batch's working set forever.
            live = 0
        return filled, live, total

    def _gpu_mem_bytes(self) -> int:
        """Bytes this engine process holds on the GPU (torch's reserved caching-allocator
        pool: weights + KV + MoE cache + graphs). 0 on CPU. Cheap, no device sync."""
        if self.device.type != "cuda":
            return 0
        return torch.cuda.memory_reserved(self.device)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is not None and msg.uid in tombstones:
                tombstones.pop(msg.uid, None)
                logger.debug_rank0(
                    "Dropping request %d because its abort arrived before admission", msg.uid
                )
                return
            if msg.mm_items and self.engine.encoder_cache is None:
                # no encoder runtime: fail loudly instead of decoding unexpanded placeholders
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            error="image input is not supported by this server",
                        )
                    ]
                )
                return
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
                # Tell the client instead of dropping silently — otherwise its wait_for_ack
                # never sees a `finished` reply and hangs until the request times out.
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            # "prompt is too long: N tokens > M" is the phrasing Claude Code and
                            # OpenClaw match on; the Anthropic wire has no error code to read.
                            error=(
                                f"prompt is too long: {input_len} tokens > {max_seq_len} maximum "
                                f"(prompt + generation); shorten the prompt or increase the KV "
                                f"cache budget"
                            ),
                            # OpenAI's standard class for this, for clients that read a code.
                            code="context_length_exceeded",
                        )
                    ]
                )
                return
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is None:
                tombstones = self._abort_tombstones = {}
            tombstones[msg.uid] = None
            # Unknown aborts normally consume their tombstone when the cross-worker UserMsg
            # catches up. Bound hostile/no-followup abort traffic without affecting realistic
            # in-flight concurrency.
            while len(tombstones) > 65_536:
                tombstones.pop(next(iter(tombstones)))
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if (
                req_to_free is not None
                and req_to_free.mm_items
                and self.engine.encoder_cache is not None
            ):
                # drop the aborted request's claims; entries it held alone die here
                self.engine.encoder_cache.release(
                    msg.uid, [item.hash for item in req_to_free.mm_items]
                )
            if req_to_free is not None:
                # SGLang-style abort: never free resources under an in-flight forward. If the
                # request is in the launched-but-not-drained batch (overlap), only mark it;
                # _process_last_data frees it this same iteration, after copy_done.synchronize()
                # -- so its KV pages / GDN slots are never recycled mid-write, and the
                # finished=False prefix-commit can't run on a freed request. A request with no
                # forward in flight (e.g. a decode req starved behind a long chunked prefill)
                # is freed immediately -- deferring would leak until its next batch, which
                # strict prefill-priority puts arbitrarily far away.
                inflight = (
                    self._last_data is not None
                    and req_to_free in self._last_data[0].batch.reqs
                )
                if not inflight and getattr(self, "_pipe", None) is not None:
                    # a piped chunk is the same hazard on the prefill stream: mark it,
                    # the pipe drain frees it after chunk_done
                    pb = self._pipe.inflight_batch()
                    inflight = pb is not None and req_to_free in pb.reqs
                if inflight:
                    req_to_free.aborted = True
                else:
                    self._free_req_resources(req_to_free)
            # Always acknowledge the abort, even when the request already left the manager,
            # but NOT yet: overlap_loop still has to publish the prior forward's sampled reply.
            # _flush_abort_acks runs after _process_last_data, making this a true terminal
            # accounting barrier for FrontendManager/prepare-stop.
            self._pending_abort_acks.add(msg.uid)
        elif isinstance(msg, CacheRebuildBackendMsg):
            # v1 scope: only if_idle, single-rank, non-owned-KV. drain mode and TP rebuild
            # need the drain-gate / all-rank failure-agreement machinery (deferred), so we
            # reject them cleanly rather than ship hang-prone half-wired paths.
            if not self.cache_manager.supports_runtime_rebuild:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "this model's cache does not support runtime rebuild"
                )
            elif msg.mode != "if_idle":
                self._reply_rebuild(
                    msg.request_id, "unsupported", f"mode {msg.mode!r} unsupported (use if_idle)"
                )
            elif self.config.tp_info.size > 1:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "runtime rebuild unsupported under TP > 1"
                )
            elif self.prefill_manager.runnable or self.decode_manager.runnable:
                # if_idle: refuse rather than wait. (finished_reqs hold no resources — they
                # are already freed — so they do not block a rebuild.)
                self._reply_rebuild(msg.request_id, "busy")
            else:
                self._pending_rebuild = msg
        elif isinstance(msg, FlushCacheBackendMsg):
            self._handle_flush(msg)
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _restore_linear_states(self, batch) -> None:
        """COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot
        (first chunk only). MUST run on the ENGINE stream so it is program-ordered after the
        prior batch's snapshot writes and before this forward reads the live slot.
        Host-tiered snapshots restore through a pinned H2D copy instead of the slot COW."""
        pool = self.engine.linear_state_pool
        if pool is None or not batch.is_prefill:
            return
        host_cache = self.cache_manager.host_cache
        for req in batch.reqs:
            if req.mamba_restore_host is not None:
                assert host_cache is not None
                host_cache.restore_from(req.mamba_restore_host, req.linear_slot_idx)
                logger.info_rank0(
                    f"GDN host restore: buffer {req.mamba_restore_host} -> slot "
                    f"{req.linear_slot_idx}"
                )
                req.mamba_restore_host = None  # consumed: restore exactly once
            elif req.mamba_restore_src is not None:
                pool.copy_from(req.mamba_restore_src, req.linear_slot_idx)
                req.mamba_restore_src = None  # consumed: restore exactly once

    def _free_req_resources(self, req: Req) -> None:
        # Idempotent: an EOS-finished request can stay in running_reqs (output budget left), so an
        # abort in the same overlap iteration races _process_last_data and would free it twice --
        # double-freeing its table_idx and (hybrid) GDN slots onto the free-list, handing the same
        # slots to two later requests. table_idx == -1 marks an already-freed request.
        if req.table_idx == -1:
            return
        if self.spec is not None:
            self.spec.release(req)
        # Polymorphic free: the DSV4 manager returns the request's window pages + cmp/idx blocks
        # to their tier free-lists; the generic manager frees its KV pages (it reads
        # page_table[req.table_idx], so free the table entry after).
        self.cache_manager.cache_req(req, finished=True)
        self.table_manager.free(req.table_idx)
        req.table_idx = -1

    def _reply_rebuild(self, request_id: str, status: str, error: str | None = None) -> None:
        # Single source of truth with the rollback snapshot (_current_cache_geometry): mamba is
        # usable slots (padding sink excluded, matching the status-bar gauge), and num_swa_pages
        # reports 0 unless the model actually has a window pool.
        geo = self._current_cache_geometry()
        self.send_result(
            [
                CacheRebuildResultMsg(
                    request_id=request_id,
                    status=status,
                    moe_cache_size=geo["moe_cache_size"] or 0,
                    num_pages=geo["num_pages"],
                    mamba_slots=geo["num_mamba_slots"] or 0,
                    num_swa_pages=geo["num_swa_pages"] or 0,
                    error=error,
                )
            ]
        )

    def _execute_pending_rebuild(self) -> None:
        from freetoken.engine.engine import CacheRebuildRejected

        msg = self._pending_rebuild
        assert msg is not None
        self._pending_rebuild = None
        requested = {
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "num_mamba_slots": msg.num_mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
        }
        # Rollback target: the CURRENT (serving) sizes of ONLY the pools this request touches.
        # Passing the untouched pools too would trip rebuild_cache's KV/mamba/SWA gate and wipe
        # the prefix cache that a successful resize of just the requested pool preserves.
        snapshot = self._current_cache_geometry()
        prior = {k: snapshot[k] for k, v in requested.items() if v is not None}
        # Cleared here, set by engine.rebuild_runtime_cache at its point of no return — lets the
        # except below tell a pre-teardown failure (engine untouched) from a mid-teardown one.
        self.engine.rebuild_teardown_started = False
        try:
            self.rebuild_cache(**requested)
        except CacheRebuildRejected as e:
            # Rejected before any destructive free — old cache intact, keep serving.
            logger.warning(f"cache rebuild rejected: {e}")
            self._reply_rebuild(msg.request_id, "rejected", error=str(e))
            return
        except Exception as e:  # noqa: BLE001
            if not getattr(self.engine, "rebuild_teardown_started", True):
                # Failed before the destructive phase began: graphs and pools are untouched and
                # the engine is still serving. A destructive rollback would only add risk.
                logger.error(f"cache rebuild failed before teardown: {e!r} — old cache intact")
                self._reply_rebuild(msg.request_id, "rejected", error=repr(e))
                return
            if self.config.tp_info.size > 1:
                # A lone-rank failure cannot be rolled back symmetrically: rebuild_cache runs TP
                # barriers, and ranks that succeeded will not re-enter them — a solo rollback
                # would desync the group. Keep the latch-failed behavior for tp>1.
                logger.error(f"cache rebuild failed: {e!r} — tp>1, latching failed")
                self._reply_rebuild(msg.request_id, "failed", error=repr(e))
                return
            # The destructive phase failed — typically a CUDA OOM while reallocating a pool or
            # recapturing graphs. The graphs/pools are already torn down, so the engine cannot
            # serve as-is. Rather than latch "failed" (which forces a full process restart),
            # rebuild the touched pools back to the sizes that were serving a moment ago: they
            # fit before, so shrinking back frees the just-attempted allocation and restores
            # service. Only if the rollback ALSO fails is the engine genuinely wedged. (Post-OOM
            # CUDA state is not guaranteed sane — a rollback that succeeds here may still surface
            # a deferred fault on a later request; that residual risk is accepted over always
            # forcing a restart.)
            logger.error(f"cache rebuild failed: {e!r} — rolling back to the previous geometry")
            try:
                self.rebuild_cache(**prior)
            except Exception as e2:  # noqa: BLE001 — rollback failed too; genuinely unrecoverable
                logger.error(f"cache rebuild rollback failed: {e2!r} — server latched failed")
                self._reply_rebuild(
                    msg.request_id,
                    "failed",
                    error=f"{e!r}; rollback to the prior geometry also failed: {e2!r}",
                )
                return
            logger.warning("cache rebuild rolled back to the previous geometry — still serving")
            self._log_cache_geometry("Cache rolled back")
            self._reply_rebuild(
                msg.request_id, "rejected", error=f"rebuild failed and was rolled back: {e!r}"
            )
            return
        # Outside the try: an ack/send failure after a fully-applied rebuild must not be
        # mistaken for a rebuild failure and roll back the geometry the engine now serves.
        self._log_cache_geometry("Cache rebuilt")
        self._reply_rebuild(msg.request_id, "ok")

    def _handle_flush(self, msg: FlushCacheBackendMsg) -> None:
        # Same idle/timeout contract as sglang's /flush_cache. Even timeout=0 waits for this iteration's safe point.
        if self.config.tp_info.size > 1:
            # each rank checks the deadline on its own clock, so a deferred flush could run on only some ranks
            self._reply_flush(msg.request_id, "unsupported", "cache flush unsupported under TP > 1")
        elif self._pending_flush is not None:
            self._reply_flush(msg.request_id, "busy", "another cache flush is already waiting")
        else:
            self._pending_flush = (msg, time.monotonic() + msg.timeout)

    def _check_pending_flush(self) -> None:
        """Run the pending flush once the scheduler is fully idle, or fail it at its deadline."""
        if self._pending_flush is None:
            return
        msg, deadline = self._pending_flush
        if self._last_data is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
            # a piped chunk holds pages too: flushing under it would leak its
            # allocations out from under the in-flight forward
            or (self._pipe is not None and (self._pipe.inflight or self._pipe_take is not None))
        ):
            self._pending_flush = None
            freed = self.cache_manager.flush()
            self.cache_manager.check_integrity()
            logger.info_rank0(f"Cache flushed: {freed} prefix-cache tokens freed")
            self._reply_flush(msg.request_id, "ok")
        elif time.monotonic() >= deadline:
            self._pending_flush = None
            error = (
                "scheduler is not idle: "
                f"{len(self.prefill_manager.pending_list)} queued, "
                f"{len(self.decode_manager.running_reqs)} running"
                + (", last batch still in flight" if self._last_data is not None else "")
            )
            logger.warning_rank0(f"Cache not flushed, {error}")
            self._reply_flush(msg.request_id, "timeout" if msg.timeout > 0 else "busy", error)

    def _reply_flush(self, request_id: str, status: str, error: str | None = None) -> None:
        self.send_result([FlushCacheResultMsg(request_id=request_id, status=status, error=error)])

    def _current_cache_geometry(self) -> dict:
        """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
        the single source for _reply_rebuild's readout. None for a pool this model lacks
        (rebuild_cache skips those; the reply maps them to the wire format's 0). num_swa_pages is
        the CONCRETE current window (usable pages) so a rollback restores it byte-for-byte,
        whether it was pinned or ratio-derived."""
        eng = self.engine
        from freetoken.kvcache.cache_status import window_pool_spec

        spec = window_pool_spec(self.config)
        num_swa_pages = eng.kv_cache.window_pages if spec is not None else None
        return dict(
            num_pages=eng.num_pages,
            moe_cache_size=eng.moe_offload_cache.cache_size if eng.moe_offload_cache is not None else None,
            num_mamba_slots=(eng.linear_state_pool.num_slots - 1) if eng.linear_state_pool is not None else None,
            num_swa_pages=num_swa_pages,
        )

    def _log_cache_geometry(self, event: str) -> None:
        """One-line readout of every pool's new size + VRAM after a rebuild changed them:
        full KV always; swa/mamba/MoE only for models with the pool. Byte figures are
        best-effort (0 when a unit cost cannot be measured) and must never block the reply."""
        from freetoken.kvcache.cache_status import compute_cache_pools, compute_cache_unit_bytes

        try:
            pools = compute_cache_pools(self.engine)
            unit = compute_cache_unit_bytes(self.engine)
            kv_tokens = pools["num_pages"] * pools["page_size"]
            parts = [
                f"KV {pools['num_pages']} pages"
                f" ({kv_tokens} tokens, {_gib(kv_tokens * unit['kv_bytes_per_token'])})"
            ]
            if pools["num_swa_pages"]:
                swa_tokens = pools["num_swa_pages"] * pools["swa_page_size"]
                parts.append(
                    f"swa {pools['num_swa_pages']} pages"
                    f" ({swa_tokens} tokens, {_gib(swa_tokens * unit['swa_bytes_per_token'])})"
                )
            if pools["num_mamba_slots"]:
                parts.append(
                    f"mamba {pools['num_mamba_slots']} slots"
                    f" ({_gib(pools['num_mamba_slots'] * unit['mamba_bytes_per_slot'])})"
                )
            moe = self.engine.moe_offload_cache
            if moe is not None:
                parts.append(
                    f"MoE cache {moe.cache_size}/{moe.num_layers * moe.num_experts}"
                    f" ({_gib(moe.cache_size * unit['moe_bytes_per_expert'])})"
                )
            logger.info_rank0(f"{event}: " + ", ".join(parts))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not log cache geometry: {e!r}")

    def _free_window_slots(self, batch: Batch) -> None:
        """Run the two SWA eviction drivers over the requests each one owns, BEFORE the
        batch's pages are allocated. The decode driver bounds a running request's swa
        footprint at an interval with the window+page margin and the first-step guard;
        the extend driver runs every prefill pass with the pre-chunk cached_len
        frontier, so a chunked prompt longer than the swa pool never accumulates its
        whole swa footprint (which would exhaust alloc_swa). Both no-op unless the
        model is SWA / paged swa pool. A mixed batch runs each driver on its own side:
        the ride-along decodes must not take the extend frontier (it ignores the anchor
        retention) and the prompt chunks must not skip the eviction cadence."""
        decoding = getattr(batch, "decoding_reqs", None)
        if not batch.is_prefill or decoding:
            dec = decoding or batch.reqs
            self.cache_manager.maybe_free_swa_out_of_window(
                dec, forward_iter=self._forward_iter)
            for req in dec:
                req.decode_batch_idx += 1
        if batch.is_prefill:
            prompt_reqs = (
                [r for r in batch.reqs if r not in decoding] if decoding else batch.reqs
            )
            self.cache_manager.free_swa_out_of_window_extend(prompt_reqs)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        if self.spec is not None and batch.is_decode:
            batch.phase, batch.rows_per_req = "verify", self.config.decode_rows_per_req
        self.engine.graph_runner.pad_batch(batch)
        self._forward_iter += 1
        # mixed-prefill-aware SWA eviction (ride-along decodes vs prompt chunks); a
        # verify batch runs the decode driver like any other decode step
        self._free_window_slots(batch)
        # before the widening below, so the penalty check reads each request's real remain_len
        sample_args = self.engine.sampler.prepare(batch)
        # widened after the window free above, whose threshold reads device_len
        if batch.phase == "verify":
            for req in batch.reqs:
                req.widen_pending(batch.rows_per_req - 1)
        # Polymorphic page allocation: DSV4 allocates window pages + cmp/idx blocks into its
        # slot maps; the generic manager allocates KV pages into the page table.
        self.cache_manager.allocate_paged(batch.reqs)
        if batch.phase == "verify":
            # the drafts run on the GPU while the host builds this batch's metadata below
            self.spec.propose_into(batch.reqs, self.token_pool)
        if batch.is_prefill:
            self._gather_multimodal(batch)
        batch.positions = _make_positions(batch, self.device)
        if self._model_is_mrope:
            batch.mrope_positions = _make_mrope_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = None if batch.phase == "verify" else _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_backend is not None:
            self.engine.linear_backend.prepare_metadata(batch)
        if not batch.is_prefill:
            # This batch's padded per-row page-table rows. Backends that snapshot the table for
            # a captured replay (DSV4) read them in prepare_metadata / prepare_for_replay.
            batch.active_table_idx = input_mapping[0][:: batch.rows_per_req].view(-1)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=sample_args,
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _gather_multimodal(self, batch: Batch) -> None:
        """Plan the chunk's encoder jobs, gather rows and scatter rows over the batch; the engine runs them before the LM forward."""
        jobs, plan, rows, block_ends = plan_mm_batch(batch.padded_reqs, self.engine.encoder_cache)
        if plan:
            batch.mm_encoder_jobs = jobs
            batch.mm_gather_plan = plan
            batch.mm_rows = torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
            batch.mm_block_ends = torch.tensor(block_ends, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        if self._bidirectional_mm and not self._warned_cut_image and (cut := cut_image_spans(batch.padded_reqs)):
            # only a bidirectional image span loses context when cut, and only an image longer than the chunk still gets cut
            lo, hi = cut[0]
            self._warned_cut_image = True
            logger.warning_rank0(
                f"an image of {hi - lo} tokens does not fit one prefill chunk (--max-extend-tokens {self.prefill_budget}, or the sliding-window pool's share of it): its earlier rows attend within the first part only"
            )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        from freetoken.moe import _debug_stats

        _dbg = _debug_stats.probe()
        if getattr(self, "_pipe", None) is not None:
            # getattr: Scheduler.__new__ stubs predate the pipe (same rule as mixed)
            return self._schedule_next_batch_piped()
        interval = self.config.prefill_decode_interval
        batch = None
        # schedule_next_batch consumes the admission, so the interleave guard must run
        # BEFORE scheduling prefill: after N consecutive prefill steps -- or after
        # _prefill_debt_s prefill SECONDS accrued while a decode was runnable (wall-clock
        # bound: chunk times swing ~10x under PLE device-latency spikes, which the count
        # cannot see) -- hand one step to running decodes so a long chunked prompt cannot
        # stall everyone's ITL. The chunk still in flight counts too: the drain that would
        # book it runs after this decision, so the accrued debt alone is one chunk late and
        # bounds a waiting decode at two chunks instead of one.
        inflight_batch, inflight_s = self._inflight_prefill()
        threshold = self._effective_debt_s()
        owed = (interval > 0 and self._prefill_streak >= interval) or (
            threshold > 0 and self._prefill_debt + inflight_s >= threshold
        )
        # Mixed mode: while a decode runs, the prefill step carries it -- the concede the
        # guard below would buy is already paid for, so prefill is scheduled even when
        # the streak/debt says owed. (getattr first: Scheduler.__new__ stubs predate it.)
        mixed = getattr(self, "_mixed_prefill_tokens", 0) > 0 and self.decode_manager.runnable
        budget = self._adaptive_prefill_budget()
        if mixed:
            budget = min(budget, self._mixed_prefill_tokens)
        if mixed or not (owed and self.decode_manager.runnable):
            batch = self.prefill_manager.schedule_next_batch(budget)
        if batch is None:
            batch = self.decode_manager.schedule_next_batch()
        elif mixed:
            batch = self._mix_decodes(batch)
        if batch is None:
            return None
        if batch.is_prefill and not getattr(batch, "decoding_reqs", None):
            self._prefill_streak += 1
        else:
            # A mixed step ran the decodes too: like a concede it ends the streak and the
            # debt, and the chunk that bought it (if one was in flight) is discharged.
            self._prefill_streak = 0
            self._debt_discharge = inflight_batch if inflight_s > 0 else None
            self._prefill_debt = 0.0
        if _dbg is not None:
            _tp = time.perf_counter()
        forward_input = self._prepare_batch(batch)
        if _dbg is not None:
            _dbg.host_phase("sched.prepare", time.perf_counter() - _tp)
        self._report_prompt_admissions(batch)
        return forward_input

    def _mix_decodes(self, batch: Batch) -> Batch:
        """Let every running request ride this prefill chunk's forward (the ported effect
        of Strata's STRATA_PIPED_CONCURRENT: the decode loop never waits for the prompt).

        The batch keeps phase="prefill" so the extend machinery runs: attention and GDN see
        the decode rows as length-1 varlen sequences -- the chunk kernels' smallest partial
        chunk, the same case a chunked continuation is, just shorter. They sample, write and
        commit like the rows of a decode step (the drain branches on ``decoding_reqs``),
        while only the prompt side is radix-committed. What they give up for the ride: no
        CUDA graph, no verify upgrade (the batch stays phase="prefill", so the spec
        runtime proposes no drafts for these rows), and the GDN chunked-scan kernel
        instead of the recurrent decode kernel -- same math, different rounding.
        """
        reqs = sorted(self.decode_manager.running_reqs, key=lambda req: req.uid)
        batch.reqs.extend(reqs)
        batch.decoding_reqs = reqs
        return batch

    def _make_pipe(self):
        from freetoken.distributed import get_tp_info, pipe_distributed_ready

        if ENV.DISABLE_OVERLAP_SCHEDULING:
            logger.warning_rank0(
                "FREETOKEN_PIPED_PREFILL needs overlap scheduling (FREETOKEN_DISABLE_"
                "OVERLAP_SCHEDULING is set); piping disabled"
            )
            return None
        if self.spec is not None:
            # v1 scope: a spec verify shares the PLE staging slots / plan 0 with the
            # pipe's chunk; until the spec runtime has its own staging, piping stays
            # off while speculation is armed.
            logger.info_rank0(
                "FREETOKEN_PIPED_PREFILL: speculative decoding armed; piping disabled "
                "(spec verify shares the PLE/plan staging with the pipe)"
            )
            return None
        if get_tp_info().size > 1 and not pipe_distributed_ready():
            # The engine builds the pipe's own NCCL group at init when piping is on; if
            # it is missing the chunk's collectives would share the decode comm and
            # interleave two call sequences across ranks, so TP stays on the inline
            # interleave (FREETOKEN_MIXED_PREFILL_TOKENS bounds its stall).
            logger.warning_rank0(
                "FREETOKEN_PIPED_PREFILL: this rank built no dedicated pipe comm; "
                "piping disabled (use FREETOKEN_MIXED_PREFILL_TOKENS under TP)"
            )
            return None
        if os.environ.get("FREETOKEN_PIPED_DEBUG", "0") == "1":
            # crash/freeze diagnostic: SIGABRT/SIGSEGV (glibc heap aborts included) dump
            # every Python thread's stack, and a 30 s repeating dump catches a deadlock
            # as two stacks parked in cuda* calls that never return.
            import faulthandler

            faulthandler.enable()
            faulthandler.dump_traceback_later(30, repeat=True, exit=False)
            logger.info_rank0("piped prefill: faulthandler enabled (abort + 30s stack dumps)")
        from .prefill_pipe import PrefillPiper

        logger.info_rank0(
            "piped prefill enabled: chunks run on the prefill stream, decode replays "
            "keep the high-priority engine stream"
        )
        return PrefillPiper(self)

    def _schedule_next_batch_piped(self) -> ForwardInput | None:
        """Piped mode: the pipe thread owns prefill issue, this loop owns decode.

        The streak/debt/mixed guards are inert by construction -- a decode is never
        behind a chunk any more, so the main issue is simply the decode batch, and a
        chunk is prepared the moment the pipe's single slot is free. The chunk's
        FORWARD launches on the pipe thread (see _issue_piped_forward); everything
        that mutates scheduler state (allocation, radix, admissions) ran here.
        """
        self._pipe_take = None
        budget = self._adaptive_prefill_budget()
        _can, _prun = self._pipe.can_take(), self.prefill_manager.runnable
        if _can and _prun:
            pbatch = self.prefill_manager.schedule_next_batch(budget)
            if pbatch is not None:
                pbatch.piped = True
                self._prefill_streak = 0
                self._prefill_debt = 0.0
                self._pipe_take = self._prepare_batch(pbatch)
                # GDN snapshot restore runs HERE, on the main thread's staging stream,
                # not on the pipe thread: the snapshot host tier's buffer bookkeeping is
                # single-thread code, and a prefix-hit restore under it raced the main
                # loop's eviction saves (glibc "double free" at the first piped prefix
                # hit). self.stream already waited chunk_done, so the source snapshot
                # and the prior chunk's writes are ordered; prefill_stream waits
                # prepare_done before the forward reads the live slot.
                self._restore_linear_states(pbatch)
                self._report_prompt_admissions(pbatch)
        if _PIPED_TRACE:
            logger.info(
                f"[pipetrace] take can={int(_can)} prun={int(_prun)} budget={budget} "
                f"chunk={0 if self._pipe_take is None else int(getattr(self._pipe_take.batch, 'log_new_tokens', 0))} "
                f"drun={int(self.decode_manager.runnable)}"
            )
        if not self.decode_manager.runnable:
            return None
        batch = self.decode_manager.schedule_next_batch()
        if batch is None:
            return None
        return self._prepare_batch(batch)

    def _issue_piped_forward(self, forward_input: ForwardInput):
        """The pipe thread's half of a chunk: gather inputs, run the forward, write the
        sampled row. Only this chunk's Req device counters advance here (complete_one
        inside forward_batch) -- the running decodes the main loop reads are disjoint.
        No filter_reqs (the drain runs it on the main thread), no toolcall anchor (a
        prompt chunk carries none)."""
        batch, sample_args, input_mapping, output_mapping = forward_input
        if input_mapping is not None:
            batch.input_ids = self.token_pool[input_mapping]
        # (_restore_linear_states already ran on the main thread at handoff)
        forward_output = self.engine.forward_batch(batch, sample_args)
        if output_mapping is not None:
            self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        return forward_output

    def _discard_piped_chunk(self, forward_input: ForwardInput, err: BaseException) -> None:
        """Release a chunk whose worker died: wait its stream point (its kernels may
        have launched before the raise), free its resources, drop the pending
        continuation it owned, and end each request with an error reply."""
        from freetoken.message import ErrorReplyMsg

        from .prefill import ChunkedReq

        batch = forward_input.batch
        logger.warning_rank0(f"piped prefill chunk failed: {err!r}")
        self._pipe.chunk_done.synchronize()
        dead = set(batch.reqs)
        self.prefill_manager.pending_list = [
            p for p in self.prefill_manager.pending_list if p.chunked_req not in dead
        ]
        # the drain's lazy_free_region is the normal flusher; a discard outside any
        # region would strand the deferred page frees (integrity sees them as leaks)
        with self.cache_manager.lazy_free_region():
            for req in dead:
                self.decode_manager.remove_req(req)
                self._free_req_resources(req)
        for req in dead:
            if not isinstance(req, ChunkedReq):
                self.send_result([ErrorReplyMsg(
                    uid=req.uid, error=f"prefill forward failed: {err}"
                )])

    def _report_prompt_admissions(self, batch: Batch) -> None:
        """Publish first-prefill accounting only after batch preparation succeeded.

        ``send_result`` is rank-aware: TP rank 0 forwards the signal, other ranks are
        no-ops. The offline handler explicitly ignores this online-accounting message.
        """
        if not batch.is_prefill or not batch.prompt_admissions:
            return
        self.send_result(
            [
                PromptAdmittedMsg(uid=uid, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)
                for uid, prompt_tokens, cached_tokens in batch.prompt_admissions
            ]
        )

    def _flush_abort_acks(self) -> None:
        pending = getattr(self, "_pending_abort_acks", None)
        if not pending:
            return
        uids = sorted(pending)
        pending.clear()
        self.send_result([ErrorReplyMsg(uid=uid, error="request aborted") for uid in uids])

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        # Real work is running: suppress the idle clock-keeper pulse.
        self.engine.clock_keeper.notify()
        from freetoken.moe import _debug_stats

        _dbg = _debug_stats.probe()
        _pk = "p." if batch.is_prefill else "d."
        if _dbg is not None:
            import time as _time

            _q0 = _time.perf_counter()
        batch.input_ids = self.token_pool[input_mapping]
        if _dbg is not None:
            _q1 = _time.perf_counter()
            _dbg.host_phase(_pk + "fw.inputids", _q1 - _q0)
        if self.toolcall_anchor_id is not None:
            # prompt chunks cannot carry a generated anchor; a mixed batch snapshots on its
            # ride-along decodes only
            anchor_reqs = (
                batch.reqs if not batch.is_prefill
                else getattr(batch, "decoding_reqs", None)
            )
            if anchor_reqs:
                self.cache_manager.snapshot_toolcall_anchor(anchor_reqs)
        if _dbg is not None:
            _qa = _time.perf_counter()
            _dbg.host_phase(_pk + "fw.anchor", _qa - _q1)
        if batch.phase == "verify":
            return self.spec.verify_step(batch, sample_args)
        forward_output = self.engine.forward_batch(batch, sample_args)
        if _dbg is not None:
            _q2 = _time.perf_counter()
            _dbg.host_phase(_pk + "fw.fbcall", _q2 - _qa)
        if output_mapping is not None:
            # None on verify: its drain writes the pending token itself (_drain_verify)
            self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        if _dbg is not None:
            _q3 = _time.perf_counter()
            _dbg.host_phase(_pk + "fw.tail", _q3 - _q2)
        return forward_output


def _make_mrope_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    """[3, N] rope rows: an image request's prompt tokens use their precomputed columns, everything else is sequence index + per-request delta."""
    needed = sum(r.extend_len for r in batch.padded_reqs)
    host = torch.empty((3, needed), dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        out = host[:, offset : offset + length]
        full = req.mrope_positions_full
        if full is not None and req.device_len <= full.shape[1]:
            out.copy_(full[:, req.cached_len : req.device_len])
        else:
            row = torch.arange(
                req.cached_len + req.mrope_delta,
                req.device_len + req.mrope_delta,
                dtype=torch.int32,
            )
            out.copy_(row.unsqueeze(0).expand(3, -1))
        offset += length
    return host.to(device, non_blocking=True)


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
