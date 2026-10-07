from __future__ import annotations

import os
import time

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
from freetoken.attention.linear import build_fla_metadata
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
    PromptAdmittedMsg,
    UserMsg,
)
from freetoken.utils import (
    div_ceil,
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
    load_toolcall_anchor_id,
)

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from freetoken.engine.config import mtp_cold_start, mtp_stash_budget
from freetoken.engine.spec_sample import debug_traced, sampling_path_enabled, spec_supported
from .io import SchedulerIOMixin
from .mm import cut_image_spans, plan_mm_batch
from .prefill import ChunkedReq, PrefillManager
from .status import SchedulerStatusReporter
from .table import TableManager

if TYPE_CHECKING:
    from freetoken.engine import BatchSamplingArgs, ForwardOutput


logger = init_logger(__name__)

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

# MTP low-acceptance auto-off. Every verify pays a second row through the target model, so
# a request whose draft keeps missing stops paying for it: below this acceptance over a
# window of verifies it drafts less than the extra row costs. 0 disables the check.
_MTP_ACCEPT_WINDOW = max(8, int(os.environ.get("FREETOKEN_MTP_ACCEPT_WINDOW", "64")))
_MTP_MIN_ACCEPTANCE = float(os.environ.get("FREETOKEN_MTP_MIN_ACCEPTANCE", "0.35"))
# Decode steps a request stands down for after a window that missed the floor, then re-probes.
# A window is a small sample: at 0.45 true acceptance a 64-window misses a 0.35 floor ~6% of the
# time (a 16-window, ~20%), and ending the request on one trip switched good heads off
# mid-answer. One window is the default so a false trip costs a window of drafts, not the answer;
# 0 restores the terminal behaviour for an A/B.
_RESUME_AFTER_ENV = os.environ.get("FREETOKEN_MTP_RESUME_AFTER")
_MTP_RESUME_AFTER = (
    max(0, int(_RESUME_AFTER_ENV)) if _RESUME_AFTER_ENV else _MTP_ACCEPT_WINDOW
)
# The fast half of the same decision, on the host and per step: measured on live traffic, the
# next draft lands 0.51 of the time after an accept, 0.33 after two misses and ~0.15 after
# three -- while a verify costs about 2.1 decode steps. So after a short miss streak the next
# few steps are worth more spent decoding. 0 disables the stand-down.
_MTP_REJECT_STREAK = max(0, int(os.environ.get("FREETOKEN_MTP_REJECT_STREAK", "3")))
_MTP_SKIP_STEPS = max(1, int(os.environ.get("FREETOKEN_MTP_SKIP_STEPS", "8")))
# Rows a request may decode without the head seeing them (because the batch grew beyond
# one request) before its prompt residual stash is written off as too stale to catch up.
# Bounded so a request that never runs alone cannot pin its stash for its whole life.
_MTP_RESYNC_LAG = max(1, int(os.environ.get("FREETOKEN_MTP_RESYNC_LAG", "8")))
# Decline lines per reason. A reason is terminal for a request so the counters stay small, but
# a workload of thousands of declined requests still needs a ceiling on the log it writes.
_MTP_DECLINE_LOG_CAP = max(0, int(os.environ.get("FREETOKEN_MTP_DECLINE_LOG", "5")))


def _mtp_report_interval_s() -> float:
    """How often the MTP tally may repeat when verifies are too sparse to drive it.

    The line used to ride the verify window alone, so a workload that never became eligible
    printed nothing and read exactly like MTP being switched off. 0 keeps it window-only.
    """
    raw = os.environ.get("FREETOKEN_MTP_REPORT_INTERVAL_S", "60")
    try:
        return max(0.0, float(raw))
    except ValueError:
        raise ValueError(f"FREETOKEN_MTP_REPORT_INTERVAL_S must be a number, got {raw!r}")


# For overlap scheduling, we also need to cache some other data to avoid IMA
class ForwardInput(NamedTuple):
    batch: Batch
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)


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
        )
        self.decode_manager = DecodeManager(config.page_size)
        self._bidirectional_mm = any(getattr(g, "bidirectional_mm_blocks", False) for g in config.model_config.attention_groups)
        self.prefill_manager = PrefillManager(
            self.cache_manager,
            self.table_manager,
            self.decode_manager,
            encoder_cache=self.engine.encoder_cache,
            keep_images_whole=self._bidirectional_mm,
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
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
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
        # the MTP spec graph gathers its input rows from the pool inside the capture;
        # hand the engine the live reference (refreshed on every cache rebuild)
        self.engine.spec_token_pool = self.token_pool
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
        self._chunk_target_s = _adaptive_chunk_target_s()
        self._chunk_ema_spt: float | None = None  # EMA of prefill seconds per token
        # Opt-in per-chunk wall-clock breakdown ([chunktime] log lines). Independent of
        # the moe _debug_stats probe (which must not be enabled on PLE-graph rigs).
        self._chunk_timing = os.environ.get("FREETOKEN_CHUNK_TIMING", "0") == "1"
        self._ct_chunks = 0
        # MTP bookkeeping: why requests stopped drafting (reason -> count) and the running
        # verify/accept tally. Both the lifetime tally and the tally since the line was last
        # printed are kept: the lifetime numbers alone read as if they were the window's.
        self._spec_rejections: dict[str, int] = {}
        self._spec_verifies = 0
        self._spec_accepted = 0
        self._spec_win_verifies = 0
        self._spec_win_accepted = 0
        # coverage: decode steps that ran as spec steps vs regular ones. Acceptance alone
        # cannot tell "drafted and lost" from "switched off sixteen verifies into a 256-token
        # answer", which is the difference between a bad head and a bad policy.
        self._spec_steps = 0
        self._spec_decode_steps = 0
        # decline lines already printed per reason, so a burst of declined requests cannot
        # bury the log while the first few still explain the workload
        self._spec_decline_lines: dict[str, int] = {}
        self._spec_report_s = _mtp_report_interval_s()
        self._spec_last_report = time.monotonic()
        self._spec_reported: tuple[int, dict[str, int]] = (0, {})
        self.status_reporter = SchedulerStatusReporter(
            log=logger.info_rank0,
            decode_log_interval=config.decode_log_interval,
        )
        if getattr(config, "speculative", "none") == "mtp":
            self._report_spec_config()

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        # A short burst never reaches the verify window, so drain it here instead of leaving its
        # numbers stranded until some later request crosses the window.
        if getattr(self.config, "speculative", "none") == "mtp":
            self._report_spec_tally(force=True)
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
            # A rebuild can reallocate the page/token/state pools whose addresses the MTP
            # spec graph baked in: drop it (lazy recapture follows) and refresh the ref.
            self.engine.spec_token_pool = self.token_pool
            _spec_runner = getattr(self.engine, "_spec_graph", None)
            if _spec_runner is not None:
                _spec_runner.invalidate()
            # the prefill chunk graph bakes the same pools (token_pool/page_table gathers)
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

        _dbg = _debug_stats.probe()
        if _dbg is not None:
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
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)
        if _ct:
            _ct1 = time.perf_counter()
        if _dbg is not None:
            _n = _time.perf_counter()
            _dbg.host_phase("recv", _n - _t)
            _t = _n

        # Execute a queued cache rebuild once the scheduler is fully idle (the safe point):
        # no last batch to process, no pending prefill, no running decode. finished_reqs is
        # NOT a gate — those requests are already freed (no live GPU/page resources).
        if self._pending_rebuild is not None and last_data is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        # Order this iteration's host->device token_pool copies (issued on ``self.stream``
        # during scheduling) after the previous batch's sampled-token writes (issued on the
        # engine stream in ``_forward``). Without this, a request that reuses a just-freed
        # table_idx can have its freshly copied prompt clobbered by the prior occupant's
        # still-pending output write -- corrupting tokens (e.g. dropping an image
        # placeholder, which the multimodal merge then rejects).
        self.stream.wait_stream(self.engine.stream)
        # The accept of an in-flight spec batch is host state the next batch is built from, so
        # this iteration cannot issue first: drain it (in the order normal_loop uses -- the
        # deferred PLE fill runs inside _process_last_data, before that batch's copy_done wait,
        # and the launch that would otherwise need ordering has not happened yet).
        if self._needs_serial_drain(last_data):
            self._process_last_data(last_data)
            self._flush_abort_acks()
            last_data = None
            self._last_data = None
        forward_input = self._schedule_next_batch()
        if _ct:
            _ct2 = time.perf_counter()
        if _dbg is not None:
            _n = _time.perf_counter()
            _dbg.host_phase("sched", _n - _t)
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
                if _dbg is not None:
                    _n2 = _time.perf_counter()
                    _dbg.host_phase(
                        ("p." if forward_input.batch.is_prefill else "d.") + "fw.total",
                        _n2 - _n1)
        if _ct:
            _ct3 = time.perf_counter()
        if _dbg is not None:
            _n = _time.perf_counter()
            _dbg.host_phase("fwd_issue", _n - _t)
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
        finally:
            # Run the just-issued batch's deferred PLE fill AFTER the drain: its readback
            # event sits a hair past the drain's copy_done point in stream order (so the
            # wait is ~0 here), while the replay's captured memop WAIT keeps the ordering.
            # In a finally so a drain failure can never leave the WAIT unanswered.
            self.engine.run_pending_host_fill()
        if _ct:
            self._log_chunk_timing(last_data, (_ct1 - _ct0, _ct2 - _ct1,
                                               _ct3 - _ct2, time.perf_counter() - _ct3))
        if _dbg is not None:
            _n = _time.perf_counter()
            _dbg.host_phase("drain", _n - _t)
            # device-syncing stats dump ONLY here: past the deferred fill, so no
            # un-signaled PLE WAIT can be queued ahead of the sync
            _dbg.dump("loop")
        return ongoing_data

    def _log_chunk_timing(self, last_data, phases) -> None:
        """One [chunktime] line per drained prefill chunk: window = schedule->drain wall
        (the victim-perceived cadence), plus THIS iteration's host phases (which belong
        to the next batch's issue cycle) and 32-chunk PLE row-cache deltas."""
        batch = last_data[0].batch if last_data is not None else None
        if batch is None or not batch.is_prefill or getattr(batch, "spec_mode", None):
            return
        self._ct_chunks += 1
        window_ms = (time.perf_counter() - batch.scheduled_at) * 1e3 if batch.scheduled_at else -1.0
        recv, sched, issue, drain = (x * 1e3 for x in phases)
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
            f"issue={issue:.1f} drain={drain:.1f}{promote}{ple}"
        )

    def normal_loop(self) -> None:
        _ct = self._chunk_timing
        if _ct:
            _ct0 = time.perf_counter()
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to execute at idle
        )
        for msg in self.receive_msg(blocking=blocking):
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
        # MTP keeps the overlap loop too: an MTP spec batch is drained before the next batch is
        # scheduled (see _needs_serial_drain), which is the only place its accept is needed.
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    @staticmethod
    def _needs_serial_drain(last_data: ForwardData | None) -> bool:
        """Whether the in-flight batch must be drained before the next one may be scheduled.

        Only an MTP spec batch qualifies: how far its tokens committed (one token on a reject,
        two on an accept) is what the next batch's positions, page spans and token_pool rows
        are computed from, and the overlap loop would build those from the un-read reply. Every
        other batch leaves the pipeline intact -- this dependency is the only reason
        ``--speculative mtp`` used to run the whole scheduler without overlap, taxing the
        requests that never drafted a token."""
        return (
            last_data is not None
            and getattr(last_data[0].batch, "spec_mode", None) is not None
        )

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            return

        batch, fout = last_data[0].batch, last_data[1]
        next_tokens_cpu, copy_done = fout.next_tokens_cpu, fout.copy_done_event
        copy_done.synchronize()
        spec_payload = fout.spec
        if spec_payload is None and getattr(fout, "spec_cpu", None) is not None:
            # graphed spec step: the pinned payload row landed with the copy_done event
            v = fout.spec_cpu.tolist()
            spec_payload = {"y1": int(v[0]), "y2": int(v[1]), "draft": int(v[2]),
                            "accept": bool(v[3])}
        # Signal the in-flight batch's PLE rows at the earliest safe instant: its
        # readback event is a hair past this copy_done point, and every host us
        # spent here before the fill is a us the replay idles at its captured WAIT.
        # (normal_loop drains the batch it just issued and runs the fill BEFORE this
        # point, so this is a no-op there -- draining first would deadlock.)
        self.engine.run_pending_host_fill()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        spec_two_row = getattr(batch, "spec_mode", None) == "verify"
        if getattr(self.config, "speculative", "none") == "mtp":
            # counted on the drain, where the batch shape is known: a spec batch IS a decode
            # step (the verify rides phase="prefill" for the extend machinery), and the split
            # between the two is the coverage the acceptance numbers do not show
            if getattr(batch, "spec_mode", None) is not None:
                self._spec_steps = getattr(self, "_spec_steps", 0) + 1
            elif batch.is_decode:
                self._spec_decode_steps = getattr(self, "_spec_decode_steps", 0) + 1
        with self.cache_manager.lazy_free_region():
            if spec_two_row:
                # The spec drain owns this batch's bookkeeping (accept: complete_one +
                # two tokens; reject: rollback to the post-row-0 state); the generic loop
                # below must not also run (its prefill branch would radix-commit).
                self._drain_spec(batch, spec_payload, reply, new_finished_reqs)
            for i, req in enumerate(() if spec_two_row else batch.reqs):
                if batch.is_prefill and req.spec_off and req.spec_off_reason:
                    # The engine's residual-stash gate declined this request during this
                    # forward. Prefill only: the scheduler's own declines set the same
                    # field, and re-adopting one at every decode drain would bury the
                    # counters the periodic acceptance line prints.
                    self._count_spec_rejection(req.spec_off_reason, req)
                    req.spec_off_reason = ""
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
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                # EOS / stop-string -> "stop", output budget exhausted -> "length";
                # EOS and stop strings win over length.
                # Overlap can advance device_len ahead of the token delivered to the host.
                hit_length = req.input_ids.numel() >= req.max_device_len
                hit_eos = (
                    not req.sampling_params.ignore_eos and next_token in self.eos_token_ids
                )
                matched_stop = (
                    self._match_stop_str(req)
                    if not hit_eos and req.sampling_params.stop_strs
                    else None
                )
                finished = hit_length or hit_eos or matched_stop is not None
                finish_reason = (
                    ("stop" if (hit_eos or matched_stop is not None) else "length")
                    if finished
                    else None
                )
                if (
                    next_token == self.toolcall_anchor_id
                    and req.toolcall_anchor_len is None
                    and not finished
                ):
                    req.toolcall_anchor_len = req.input_ids.numel()
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=next_token,
                        finished=finished,
                        finish_reason=finish_reason,
                        matched_stop=matched_stop,
                        stop_strs=req.sampling_params.stop_strs or None,
                    )
                )

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in self.finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif batch.is_prefill and req.table_idx != -1:
                    # for prefill, non-chunk req, cache the prefix.
                    # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
                    # the generic manager inserts the prefix into its radix/naive cache.
                    # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
                    # instead of freeing them (handled above), so a freed request should
                    # never reach this commit -- but if a future path frees one early, skip
                    # rather than re-read the freed page-table row (and on hybrid, deref the
                    # None'd GDN ping-pong slots).
                    self.cache_manager.cache_req(req, finished=False)

            if getattr(batch, "spec_mode", None) == "prologue_decode" and spec_payload is not None:
                # a regular decode drain ran for this batch; all that remains is handing
                # the head's first draft to the request for its next (verify) step.
                for req in batch.reqs:
                    if req.table_idx != -1 and not req.aborted:
                        req.spec_draft = spec_payload.get("draft")

        self.finished_reqs = new_finished_reqs
        self._account_prefill_debt(batch)
        self._observe_chunk_time(batch)
        # Stamp each reply with the post-batch KV page occupancy so the frontend (shell
        # status bar) can show live KV usage without a separate query.
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        spec = getattr(self.config, "speculative", "none") == "mtp"
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            kv_cached = self.cache_manager.evictable_kv_pages
            mamba_cached = self.cache_manager.evictable_mamba_slots
            swa_cached = self.cache_manager.evictable_swa_tokens
            moe_used, moe_live, moe_total = self._moe_residency()
            spec_snapshot = self._spec_snapshot() if spec else None
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
                if spec_snapshot is not None:
                    (m.spec_verifies, m.spec_accepted, m.spec_declines, m.spec_steps,
                     m.spec_decode_steps, m.spec_cold_seeds) = spec_snapshot
        spec_info = None
        if spec:
            spec_info = self._spec_status()
            # also on a clock: a run whose verifies never reach the window would otherwise sit
            # silent for its whole life, which is the failure this line exists to rule out
            self._report_spec_tally()
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
            spec=spec_info,
            # the drain counted what this forward shipped: two tokens on an MTP accept, one
            # on a reject (and none when the request was already gone).
            generated_tokens=(
                len(reply) if getattr(batch, "spec_mode", None) is not None else None
            ),
        )
        self.send_result(reply)

    def _account_prefill_debt(self, batch) -> None:
        """Accrue the wall time a prefill batch held the engine while decodes waited.

        Schedule->drain is the window a victim perceives as its inter-token stall.
        Only counted while a decode is actually runnable, so a pure-prefill stretch
        with nobody waiting accrues nothing (and cannot suppress the next admission).
        """
        if (
            self._prefill_debt_s <= 0
            or not batch.is_prefill
            or getattr(batch, "spec_mode", None) is not None
            or not getattr(batch, "scheduled_at", 0.0)
            or not self.decode_manager.runnable
        ):
            return
        discharged = self._debt_discharge
        if discharged is not None:
            self._debt_discharge = None
            if batch is discharged:
                return  # this chunk bought the concede the guard just granted
        self._prefill_debt += time.perf_counter() - batch.scheduled_at

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
        if not batch.is_prefill or getattr(batch, "spec_mode", None) is not None:
            return None, 0.0  # spec rides phase="prefill" but is a decode step
        if not getattr(batch, "scheduled_at", 0.0):
            return None, 0.0  # unstamped: now - 0 would concede unconditionally
        return batch, time.perf_counter() - batch.scheduled_at

    def _observe_chunk_time(self, batch) -> None:
        """EMA of prefill seconds-per-token, from the same schedule->drain window the
        debt accounts. Feeds the adaptive chunk budget; one update per prefill batch."""
        if (
            self._chunk_target_s <= 0
            or not batch.is_prefill
            or getattr(batch, "spec_mode", None) is not None
            or not getattr(batch, "scheduled_at", 0.0)
        ):
            return
        tokens = getattr(batch, "log_new_tokens", 0)
        if tokens <= 0:
            return
        spt = (time.perf_counter() - batch.scheduled_at) / tokens
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

    def _count_spec_rejection(self, reason: str, req: Req | None = None) -> None:
        """One more request that MTP declined to draft, by reason, with a line where it happens.

        The per-decline line is the point. The breakdown used to ride only inside the verify
        tally, so a workload that never qualified -- every prompt chunked, or every prompt a
        prefix hit -- printed nothing, and that was indistinguishable from MTP being off. A
        terminal reason counts once per request; the episodic ones (resync, streak,
        suspend_hole, a low_acceptance pause) count once per episode, and the per-reason line
        cap bounds the log either way."""
        counts = getattr(self, "_spec_rejections", None)
        if counts is None:
            counts = self._spec_rejections = {}
        counts[reason] = counts.get(reason, 0) + 1
        self._log_spec_decline(reason, req)

    def _log_spec_decline(self, reason: str, req: Req | None) -> None:
        lines = getattr(self, "_spec_decline_lines", None)
        if lines is None:
            lines = self._spec_decline_lines = {}
        shown = lines.get(reason, 0)
        if shown >= _MTP_DECLINE_LOG_CAP:
            if shown == _MTP_DECLINE_LOG_CAP:
                logger.info_rank0(
                    f"MTP spec: {reason} declined further requests without a line "
                    f"(total {getattr(self, '_spec_rejections', {}).get(reason, 0)}, "
                    f"first {_MTP_DECLINE_LOG_CAP} shown)"
                )
            lines[reason] = shown + 1
            return
        lines[reason] = shown + 1
        logger.info_rank0(f"MTP spec: declined {reason}{self._decline_detail(req)}")

    def _decline_detail(self, req: Req | None) -> str:
        """The numbers a decline was decided on, in the state the decider saw.

        ``prefix_hit`` means nothing until you can see the hit was 35904 of a 35904 token
        prompt, and ``over_budget`` nothing until you can see the cap it tripped. The engine's
        residual gate hands its own snapshot over through ``spec_off_detail``: by the time the
        scheduler drains that prefill, ``complete_one`` has moved ``cached_len`` up to the full
        prompt and ``extend_len`` reads 0, so guessing from the live request would lie."""
        if req is None:
            return ""
        detail = getattr(req, "spec_off_detail", "") or ""
        req.spec_off_detail = ""  # consumed, so a later decline describes itself
        if not detail:
            parts = [
                f"prompt={req.input_ids.numel()}",
                f"new={req.extend_len}",
                f"cached={req.cached_len}",
                f"out={req.output_len}",
            ]
            sp = getattr(req, "sampling_params", None)
            if sp is not None:
                parts.append(f"sampling=(T={sp.temperature}, top_k={sp.top_k}, top_p={sp.top_p})")
            detail = " ".join(parts)
        return f" for req {req.uid}: {detail}"

    def _spec_reject_summary(self) -> str:
        counts = getattr(self, "_spec_rejections", None) or {}
        return " ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "-"

    def _report_spec_config(self) -> None:
        """Say, once at startup, what MTP will consider draftable -- before any traffic exists.

        Every eligibility rule that lives only in a gate downstream (prompt size, single-chunk
        cold prefills, one-request decode batches, greedy-only graph capture) otherwise has to
        be inferred from a line that needs verifies to print, so an unqualified workload reads
        as a broken engine. The stash size is spelled out in bytes too: the cap is prompt-sized
        and its unit is the residual row width, not tokens."""
        cfg = self.config
        budget = mtp_stash_budget()
        hidden = getattr(cfg.model_config, "hidden_size", None)
        hc = getattr(getattr(self.engine.model, "model", None), "hc_count", None)
        # one residual row is hc_count hyper-connection streams of hidden, in the model's dtype
        stash = f"{budget} tokens"
        if hidden and hc:
            stash += f" (~{budget * hidden * hc * 2 / 2 ** 20:.0f} MiB/req)"
        sampling = "on" if sampling_path_enabled() else "off (FREETOKEN_MTP_SAMPLING=0)"
        span = (
            "prefix-hit and over-budget prompts draft cold"
            if mtp_cold_start()
            else "prefix-hit, over-budget and image rows never draft"
        )
        logger.info_rank0(
            f"MTP spec: enabled | window={_MTP_ACCEPT_WINDOW} "
            f"min_acceptance={_MTP_MIN_ACCEPTANCE:.2f} resume_after={_MTP_RESUME_AFTER} "
            f"reject_streak={_MTP_REJECT_STREAK}x{_MTP_SKIP_STEPS} "
            f"resync_lag={_MTP_RESYNC_LAG} "
            f"tally_every={self._spec_report_s:.0f}s decline_log={_MTP_DECLINE_LOG_CAP} | "
            f"draftable: single-request decode batches whose head catches up over the prompt "
            f"stash ({stash}, any number of prefill chunks); sampled drafting {sampling}; {span} "
            f"| CUDA graph capture is greedy-only"
        )

    def _report_spec_tally(self, force: bool = False) -> None:
        """Print the MTP tally: lifetime verify/accept, the span since the last line, and the
        decline breakdown.

        Driven by three things, because verifies alone were not enough to make a run readable:
        the auto-off window, the drain on a clock, and a forced flush when the queue empties
        (a short burst otherwise dies below the window and never reports).
        """
        reported_verifies, reported_rejections = getattr(self, "_spec_reported", (0, {}))
        rejections = getattr(self, "_spec_rejections", None) or {}
        n, a = getattr(self, "_spec_verifies", 0), getattr(self, "_spec_accepted", 0)
        wn, wa = (
            getattr(self, "_spec_win_verifies", 0),
            getattr(self, "_spec_win_accepted", 0),
        )
        # something new, always: run_when_idle is polled on every blocking wait, and a tally that
        # had nothing to add would reprint itself for as long as the server sits idle
        if n == reported_verifies and rejections == reported_rejections:
            return
        now = time.monotonic()
        if not force and now - getattr(self, "_spec_last_report", now) < getattr(
            self, "_spec_report_s", 0.0
        ):
            return
        self._spec_last_report = now
        self._spec_reported = (n, dict(rejections))
        self._spec_win_verifies = self._spec_win_accepted = 0
        since = f" (last {wa}/{wn})" if wn else ""
        # a decline-only run has nothing to divide: reporting the breakdown without the
        # acceptance number is the point (it is how a no-drafting run stays readable), but the
        # first version of this line still did a / n and killed the backend worker on idle flush
        tally = (
            f"acceptance {a}/{n} = {a / n:.3f}{since}{self._spec_coverage()}"
            if n
            else "no verifies"
        )
        logger.info_rank0(f"MTP spec: {tally} | declined: {self._spec_reject_summary()}")

    def _spec_coverage(self) -> str:
        """``drafted s/(s+d)``: the share of decode steps that ran as spec steps.

        Without it one acceptance number reads the same whether the head drafted the whole
        answer or was switched off sixteen verifies in and spent the rest decoding normally
        -- a bad head and a bad policy look identical otherwise. ``cold`` counts the seeds that
        had no catch-up pass behind them (cold start, or a hole the head never saw): drafts
        worth less than the average, which is why they must not be read silently."""
        s = getattr(self, "_spec_steps", 0)
        d = getattr(self, "_spec_decode_steps", 0)
        cold = getattr(self, "_spec_cold_seeds", 0)
        if not (s or d):
            return ""
        tail = f" cold={cold}" if cold else ""
        return f", drafted {s}/{s + d} ({s / (s + d):.2f}){tail}"

    def _spec_status(self) -> str:
        """The one-field MTP snapshot for the periodic Decode line.

        The Decode line is the only status output with a cadence that does not depend on
        verifies, so it carries whether drafting ran at all -- the question the acceptance line
        structurally cannot answer when no verify ever happened."""
        n, a = getattr(self, "_spec_verifies", 0), getattr(self, "_spec_accepted", 0)
        # 0/0 = 0.000 reads as a head that never lands a draft; it means nothing was drafted
        tally = f"acceptance {a}/{n} = {a / n:.3f}" if n else "no verifies"
        return f"{tally}{self._spec_coverage()}, declined: {self._spec_reject_summary()}"

    def _spec_snapshot(self) -> tuple[int, int, dict[str, int], int, int, int] | None:
        """The MTP counters to stamp onto a reply, or None when the engine does not draft.

        One snapshot per batch, and the decline map is copied rather than aliased: it keeps
        growing while the frontend serializes the replies that carry it."""
        if getattr(self.config, "speculative", "none") != "mtp":
            return None
        return (
            getattr(self, "_spec_verifies", 0),
            getattr(self, "_spec_accepted", 0),
            dict(getattr(self, "_spec_rejections", None) or {}),
            getattr(self, "_spec_steps", 0),
            getattr(self, "_spec_decode_steps", 0),
            getattr(self, "_spec_cold_seeds", 0),
        )

    def _release_spec_scratch(self, req: Req) -> None:
        """Drop everything the spec path holds between steps: draft, residual stash, scratch slot.

        Every path that stops drafting must call this. The scratch slot rides the same
        free-list as the live/ping-pong slots, so a leaked one is a slot the next admission
        cannot get -- and LinearStatePool.alloc raises mid-decode, where there is no way
        back out but killing the scheduler."""
        req.spec_draft = None
        req.spec_residual = None
        req.spec_draft_probs = None
        if req.spec_slot_idx is not None:
            pool = self.engine.linear_state_pool
            if pool is not None:
                pool.free(req.spec_slot_idx)
            req.spec_slot_idx = None

    def _spec_reject(self, req: Req, reason: str) -> None:
        """Turn MTP off for one request, release its scratch state, and count the reason.

        Counting happens on the transition only: a request declined every decode step
        (because the batch is never size 1) would otherwise bury the counters."""
        if not req.spec_off:
            self._count_spec_rejection(reason, req)
        req.spec_off = True
        req.spec_off_reason = reason
        self._release_spec_scratch(req)

    def _tally_verify(self, req: Req, accepted: bool) -> None:
        """Fold one verify into the running tally and the request's auto-off window.

        A request whose drafts keep missing pays a second target row for nothing, so the
        window closes shop below ``FREETOKEN_MTP_MIN_ACCEPTANCE`` (0 disables the check and
        leaves the call to whoever reads the printed acceptance): the request stands down for
        ``FREETOKEN_MTP_RESUME_AFTER`` decode steps and probes again, because one window is too
        small a sample to end an answer on.
        """
        self._spec_verifies = getattr(self, "_spec_verifies", 0) + 1
        self._spec_accepted = getattr(self, "_spec_accepted", 0) + int(accepted)
        self._spec_win_verifies = getattr(self, "_spec_win_verifies", 0) + 1
        self._spec_win_accepted = getattr(self, "_spec_win_accepted", 0) + int(accepted)
        req.spec_verifies += 1
        req.spec_accepted += int(accepted)
        if accepted:
            req.spec_rejects = 0
        elif _MTP_REJECT_STREAK:
            req.spec_rejects += 1
            if req.spec_rejects >= _MTP_REJECT_STREAK and not req.spec_suspend:
                # the next draft after a short miss streak lands ~15% of the time while a
                # verify costs ~2.1 decode steps: spend those steps decoding instead, then
                # probe again (the same stand-down the slow window uses, one scale smaller)
                req.spec_rejects = 0
                req.spec_suspend = _MTP_SKIP_STEPS
                self._count_spec_rejection("streak", req)
        if self._spec_win_verifies >= _MTP_ACCEPT_WINDOW:
            self._report_spec_tally(force=True)
        if req.spec_verifies >= _MTP_ACCEPT_WINDOW:
            rate = req.spec_accepted / req.spec_verifies
            req.spec_verifies = req.spec_accepted = 0
            if rate < _MTP_MIN_ACCEPTANCE:
                if _MTP_RESUME_AFTER:
                    # stand down and re-probe: one window is too small a sample to end a
                    # request's whole answer on (see _MTP_RESUME_AFTER), and the head's KV and
                    # scratch slot survive the pause, so resuming costs nothing
                    if not req.spec_suspend:
                        self._count_spec_rejection("low_acceptance", req)
                        logger.info_rank0(
                            f"MTP spec: req {req.uid} acceptance {rate:.3f} below "
                            f"{_MTP_MIN_ACCEPTANCE:.2f}; drafting suspended for "
                            f"{_MTP_RESUME_AFTER} decode steps"
                        )
                    req.spec_suspend = _MTP_RESUME_AFTER
                else:
                    logger.info_rank0(
                        f"MTP spec: req {req.uid} acceptance {rate:.3f} below "
                        f"{_MTP_MIN_ACCEPTANCE:.2f}, drafting disabled"
                    )
                    self._spec_reject(req, "low_acceptance")

    def _free_req_resources(self, req: Req) -> None:
        # Idempotent: an EOS-finished request can stay in running_reqs (output budget left), so an
        # abort in the same overlap iteration races _process_last_data and would free it twice --
        # double-freeing its table_idx and (hybrid) GDN slots onto the free-list, handing the same
        # slots to two later requests. table_idx == -1 marks an already-freed request.
        if req.table_idx == -1:
            return
        # MTP spec resources: the stash/draft references and the scratch GDN slot must not
        # outlive the request (the slot shares the free-list with the live/ping-pong slots).
        self._release_spec_scratch(req)
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

    def _current_cache_geometry(self) -> dict:
        """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
        the single source for _reply_rebuild's readout. None for a pool this model lacks
        (rebuild_cache skips those; the reply maps them to the wire format's 0). num_swa_pages is
        the CONCRETE current window (usable pages) so a rollback restores it byte-for-byte,
        whether it was pinned or ratio-derived."""
        eng = self.engine
        config = self.config
        mc = config.model_config
        num_swa_pages = None
        if getattr(mc, "dsv4_args", None) is not None:
            sizes = getattr(eng.kv_cache, "sizes", None)
            if sizes is not None:  # usable window pages = physical n_win_pages minus the dummy page
                num_swa_pages = max(0, sizes.n_win_pages - 1)
        elif getattr(mc, "has_swa_attention", False) and (
            getattr(config, "cache_type", None) == "swa_radix"
        ):  # usable window tokens = pool tokens minus the slot-0 sentinel
            num_swa_pages = max(0, int(getattr(eng.kv_cache, "swa_num_tokens", 0) or 0) - 1)
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

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        self.engine.graph_runner.pad_batch(batch)
        self._forward_iter += 1
        if batch.is_decode:
            # Free each decoding request's now-out-of-window SWA slots BEFORE the alloc below,
            # so they can back the new token -- this is what bounds the per-request swa
            # footprint during decode. (no-op unless the model is SWA / paged swa pool.)
            self.cache_manager.maybe_free_swa_out_of_window(
                batch.reqs, forward_iter=self._forward_iter)
            for req in batch.reqs:
                req.decode_batch_idx += 1
        else:
            # Prefill sibling of the decode driver: free out-of-window swa BEFORE allocating
            # this chunk, so a chunked prompt longer than the swa pool never accumulates its
            # whole swa footprint (which would exhaust alloc_swa). No-op unless SWA/paged.
            self.cache_manager.free_swa_out_of_window_extend(batch.reqs)
        # Polymorphic page allocation: DSV4 allocates window pages + cmp/idx blocks into its
        # slot maps; the generic manager allocates KV pages into the page table. Spec batches
        # record their charges so rejection can free pages beyond the committed prefix.
        recorded = self.cache_manager.allocate_paged(
            batch.reqs, record=getattr(batch, "spec_mode", None) == "verify"
        )
        if recorded:
            batch.spec_pages = recorded
        if batch.is_prefill:
            self._gather_multimodal(batch)
        batch.positions = _make_positions(batch, self.device)
        if self._model_is_mrope:
            batch.mrope_positions = _make_mrope_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if self.engine.linear_state_pool is not None:
            if batch.is_decode:
                # GPU GDN-state slot (one per padded request) for the decode gather/scatter;
                # lands in the CUDA-graph input buffer via copy_from. Gate on the cache mode,
                # NOT on whether any padded req has a linear_slot_idx -- the persistent dummy
                # req always carries one (= padding_slot), so that test is True even for naive
                # and would collapse all real naive reqs onto the padding slot. Hybrid: build
                # per padded req from Req.linear_slot_idx (dummy -> padding_slot). Naive: keep
                # the old keying = input_mapping's table_idx column (already staged, no H2D).
                if self.cache_manager.is_hybrid:
                    pool = self.engine.linear_state_pool
                    slots = [r.linear_slot_idx if r.linear_slot_idx is not None
                             else pool.padding_slot for r in batch.padded_reqs]
                    batch.linear_table_idx = torch.tensor(
                        slots, dtype=torch.int32, device="cpu", pin_memory=True
                    ).to(self.device, non_blocking=True)
                else:
                    batch.linear_table_idx = input_mapping[0].to(torch.int32)
            # Per-forward GDN metadata (cu_seqlens / cache_indices / continuation flags),
            # built once here instead of rebuilt in each of the 30 GDN layers. For decode
            # under CUDA graph the persistent cu_seqlens buffer is supplied by set_batch.
            batch.fla_metadata = build_fla_metadata(batch, self.device)
        if batch.is_decode:
            # This batch's padded per-row page-table rows. Backends that snapshot the table for
            # a captured replay (DSV4) read them in prepare_metadata / prepare_for_replay.
            batch.active_table_idx = input_mapping[0].view(-1)
        self.engine.attn_backend.prepare_metadata(batch)
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
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
        if not (owed and self.decode_manager.runnable):
            batch = self.prefill_manager.schedule_next_batch(self._adaptive_prefill_budget())
        if batch is None:
            batch = self.decode_manager.schedule_next_batch()
            if batch is not None:
                upgraded = self._as_spec_batch(batch)
                if upgraded is not None:
                    batch = upgraded
                elif getattr(self.config, "speculative", "none") == "mtp":
                    # A regular decode batch runs the graph and never feeds the head, so
                    # every row placed here is a row the head will never see.
                    for req in getattr(batch, "reqs", ()):
                        self._on_regular_decode(req)
        if batch is None:
            return None
        # Spec batches ride phase="prefill" for the extend machinery but ARE decode steps:
        # they must not feed the prefill-interleave streak.
        is_spec = getattr(batch, "spec_mode", None) is not None
        if batch.is_prefill and not is_spec:
            self._prefill_streak += 1
        else:
            self._prefill_streak = 0
            # the chunk in flight bought this concede with the time it is about to report;
            # discharging it is what keeps one slow chunk from buying two decode steps
            self._debt_discharge = inflight_batch if inflight_s > 0 else None
            self._prefill_debt = 0.0
        forward_input = self._prepare_batch(batch)
        self._report_prompt_admissions(batch)
        return forward_input

    def _on_regular_decode(self, req: Req) -> None:
        """Reconcile one request's spec state after a regular (non-spec) decode step.

        A regular step means the batch was not upgraded -- it grew past one request, or the
        draft was declined for the last row. The head's KV stays valid for every row it did
        see; only the staged draft is now aimed at a position that has already committed, so it
        is dropped and the next spec step re-seeds one (one rejected verify, then normal
        drafting). Ending the request here -- the old terminal ``batch_killed`` -- took drafting
        away from every concurrent turn, which is most of the traffic on a shared server.

        A request that never reached its first spec step still holds the prompt stash and
        resumes unchanged once it runs alone again; past the lag bound the span can no longer
        cover the prompt, so the stash (prompt-sized, and otherwise pinned for the whole life
        of a request that never runs alone) is released -- and the draft goes cold with it when
        cold start is on."""
        if req.spec_off:
            return
        if req.spec_suspend:
            # standing down by the acceptance floor: the head is falling behind ON PURPOSE, so
            # neither the lag nor the batch shape may turn that pause into the terminal
            # decline it used to be -- that is what never came back
            req.spec_suspend -= 1
            req.spec_suspend_missed += 1
            if req.spec_suspend == 0:
                # the hole is PER EPISODE: reset at every countdown end, or two short pauses
                # accumulate (the streak default sits exactly at the lag bound) and every
                # other one re-seeds -- measured live as streak=48 with suspend_hole=22
                missed, req.spec_suspend_missed = req.spec_suspend_missed, 0
                if missed > _MTP_RESYNC_LAG and (
                    req.spec_draft is not None or req.spec_draft_probs is not None
                ):
                    # the pause outran the hole the head may miss: the staged draft and the
                    # density that produced it were computed `missed` committed tokens ago,
                    # so the resume drops both and cold-seeds a fresh pair instead of paying
                    # a verify aimed at the wrong position (measured: that verify rejects
                    # every time). Not terminal -- same trade as the resync above.
                    req.spec_draft = None
                    req.spec_draft_probs = None
                    self._count_spec_rejection("suspend_hole", req)
            return
        if req.spec_head_len or req.spec_slot_idx is not None:
            if req.spec_draft is not None:
                req.spec_draft = None
                self._count_spec_rejection("resync", req)
            return
        req.spec_draft = None
        stash = req.spec_residual
        if stash is not None and req.cached_len - stash.shape[0] > _MTP_RESYNC_LAG:
            if mtp_cold_start():
                req.spec_residual = None  # the span is stale; the head can still draft cold
                self._count_spec_rejection("lag_overflow", req)
            else:
                self._spec_reject(req, "lag_overflow")

    def _as_spec_batch(self, batch: Batch) -> Batch | None:
        """Upgrade a single-request decode batch into an MTP spec batch, or None to keep it regular.

        Only a single request with room for two output tokens is eligible; its sampling is
        verified by id comparison (greedy) or by rejection sampling (non-greedy).
        The draft is staged after the last placed token; both rows use the extend
        attention path, with intermediate GDN/PLE state saved for rejection. Every decline
        that is final for the request goes through ``_spec_reject`` so it is counted and
        its scratch state released.
        """
        if getattr(self.config, "speculative", "none") != "mtp" or len(batch.reqs) != 1:
            return None
        req = batch.reqs[0]
        if req.spec_off or not req.can_decode:
            return None
        if req.spec_suspend:
            # standing down by the acceptance floor: no upgrade here, and the regular step it
            # falls back to is what counts the pause down. The stale draft is deliberately kept
            # -- it costs one reject after the resume, while dropping it would leave neither a
            # draft nor a stash to work from, which is a terminal decline by the rules above.
            return None
        if not spec_supported(req.sampling_params):
            self._spec_reject(req, "nongreedy")
            return None
        if getattr(req, "mm_items", None):
            self._spec_reject(req, "mm")  # Phase 1: image rows keep the regular decode path
            return None
        if req.remain_len < 2:
            # One row of output budget left: an accept could not pay out, so neither a draft nor
            # the catch-up pass is worth anything. Declining here also keeps the breakdown honest
            # -- left to _on_regular_decode this reads as a resync, blaming concurrency for
            # what is just the end of the generation (measured: every drafted request logged it)
            # -- and it releases the prompt-sized stash instead of holding it until finish.
            self._spec_reject(req, "no_room")
            return None
        if req.spec_draft is not None:
            mode = "verify"
        elif req.spec_residual is not None:
            mode = "prologue_decode"
        elif req.spec_head_len or mtp_cold_start():
            # No stash: either the head already ran and a mixed batch left a hole (its KV is
            # valid for everything it saw), or cold start is on and we draft from the first row.
            # No prologue batch is built, so the head pass of this step produces the seed draft.
            # Counted, because a cold draft is worth much less than a caught-up one and the
            # acceptance number has to be read against how often it happened.
            mode = "prologue_decode"
            self._spec_cold_seeds = getattr(self, "_spec_cold_seeds", 0) + 1
        else:
            # Neither: this request has no rows to catch the head up on and never had a draft.
            # The prefill forward declines at the BATCH level (its gate covers every request in
            # one test) and so names no request, which leaves the reason to be recovered from the
            # request here -- and a committed prompt can never grow a stash afterwards.
            if not spec_supported(req.sampling_params):
                reason = "nongreedy"  # FREETOKEN_MTP_SAMPLING=0 starved the whole batch
            elif getattr(req, "mm_items", None):
                reason = "mm"
            else:
                reason = "no_stash"
            self._spec_reject(req, reason)
            return None
        if req.spec_slot_idx is None:
            pool = self.engine.linear_state_pool
            if pool is None:
                self._spec_reject(req, "no_gdn_pool")
                return None
            try:
                req.spec_slot_idx = pool.alloc(1)[0]
            except RuntimeError:
                # _linear_spec_reserve makes this unreachable on a correctly sized pool,
                # but declining one request beats an uncaught raise in the decode loop:
                # the slot is already free-listed, so nothing leaked.
                self._spec_reject(req, "exhausted")
                logger.info_rank0(
                    f"MTP spec: no free GDN state slot for req {req.uid}, drafting disabled"
                )
                return None
        batch.spec_mode = mode
        batch.spec_sample = not req.sampling_params.is_greedy
        if mode == "verify":
            # The draft is staged into BOTH stores: the token pool feeds the forward
            # rows, and the host id buffer feeds the PLE extend fill (which builds this
            # batch's n-gram rows from req.input_ids[cached:device] -- without the host
            # copy the draft row would read a stale PLE row). The drain reconciles: an
            # accept keeps the draft as the placed y1 and appends only y2; a reject
            # overwrites the slot with the true token.
            self.token_pool[req.table_idx, req.device_len] = req.spec_draft
            req.append_host(torch.tensor([req.spec_draft], dtype=torch.int32))
            req.device_len += 1
            batch.spec_draft_id = req.spec_draft
            batch.phase = "prefill"
        else:
            if req.spec_residual is not None:
                batch.spec_prologue = self._build_spec_prologue(req)
            # else: a cold seed step -- no catch-up pass, the head pass of this very step
            # produces the first draft from the row it is decoding
        return batch

    def _build_spec_prologue(self, req: Req) -> Batch:
        """The head-only catch-up batch over the rows the head has not seen.

        See Engine._run_head_prologue. A shadow request view makes the sparse backend see
        a fresh length-t prefill: the head layer's slab/ring start empty and the page slots
        already exist (the main forward charged them), so out_loc is just the page-table row
        slice and the embeds are the pool's tokens 1..t (teacher forcing; pool[t] is the
        token that followed the last row). t is capped by the stash: rows placed after the
        prompt (a request that decoded in a multi-request batch before its first spec step)
        have no residual, so the head starts with a hole there -- it costs draft quality,
        never correctness.
        """
        from types import SimpleNamespace

        t = min(int(req.cached_len), int(req.spec_residual.shape[0]))
        assert t >= 1 and req.device_len >= t + 1, (t, req.device_len)
        shadow = SimpleNamespace(
            table_idx=req.table_idx, cached_len=0, device_len=t, extend_len=t,
            linear_slot_idx=req.linear_slot_idx,
        )
        pb = Batch(reqs=[shadow], phase="prefill")
        pb.positions = torch.arange(t, dtype=torch.int32, device=self.device)
        if self._model_is_mrope:
            # text-only prompt rows: the three mrope axes share the sequence index
            # (mrope_delta is 0 and mrope_positions_full is None for such requests)
            pb.mrope_positions = pb.positions.unsqueeze(0).expand(3, -1).contiguous()
        pb.input_ids = self.token_pool[req.table_idx, 1:t + 1]
        pb.out_loc = self.engine.page_table[req.table_idx, 0:t]
        return pb

    def _restore_spec_prefix(self, batch: Batch) -> None:
        req = batch.reqs[0]
        # the scratch slot a rollback restores from is freed only after the commit (see
        # _drain_spec): a None here means something reordered the tally ahead of the rollback
        # and the pool copy below would fail as a tensor-shape error on the engine stream
        assert req.spec_slot_idx is not None, "spec rollback without a scratch slot"
        self.engine.linear_state_pool.copy_from(req.spec_slot_idx, req.linear_slot_idx)
        req.cached_len = req.device_len - 1
        if batch.spec_pages:
            first_unused = div_ceil(req.cached_len, self.config.page_size)
            unused = [(table, max(lo, first_unused), hi)
                      for table, lo, hi in batch.spec_pages if hi > first_unused]
            if unused:
                self.cache_manager.release_paged(unused)

    def _drain_spec(self, batch: Batch, spec: dict | None,
                    reply: List[DetokenizeMsg], finished_out: Set[Req]) -> None:
        """Commit one or two verified inputs and emit the corresponding target tokens."""
        req = batch.reqs[0]
        if req in self.finished_reqs:
            return
        if req.aborted:
            self._restore_spec_prefix(batch)
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)
            finished_out.add(req)
            return
        if spec is None:
            raise RuntimeError("MTP spec step returned no payload")
        if spec["accept"]:
            req.complete_one()
            req.spec_draft = spec.get("draft")
            tokens = (spec["y1"], spec["y2"])
            skip_first_append = True
        else:
            # The scratch slot holds the state AFTER row 0, so a reject commits one
            # input token and can verify again without re-running the main model.
            self._restore_spec_prefix(batch)
            self.token_pool[req.table_idx, req.device_len - 1] = spec["y1"]
            req.input_ids[req.device_len - 1] = spec["y1"]  # the staged draft slot
            req.spec_draft = spec.get("draft")
            tokens = (spec["y1"],)
            skip_first_append = True
        # The tally runs last because it can end the draft: a request that trips the
        # low-acceptance floor is turned off through _spec_reject, which frees the scratch
        # slot the rollback above restores FROM. Doing it first hands the pool a None index
        # (a shape error out of the GDN copy, on the engine stream, mid-decode).
        if batch.spec_mode == "verify":
            self._tally_verify(req, bool(spec["accept"]))
        if debug_traced():
            logger.info_rank0(
                f"[mtp] {batch.spec_mode} accept={spec['accept']} y1={spec['y1']} "
                f"y2={spec.get('y2')} draft={spec.get('draft')} emit={tokens} "
                f"lens=({req.cached_len},{req.device_len}) host={req.input_ids.tolist()[-4:]}"
            )
        finished = False
        for tok_i, tok in enumerate(tokens):
            if not (tok_i == 0 and skip_first_append):
                req.append_host(torch.tensor([tok], dtype=torch.int32))
            hit_length = not req.can_decode
            hit_eos = not req.sampling_params.ignore_eos and tok in self.eos_token_ids
            matched_stop = (
                self._match_stop_str(req)
                if not hit_eos and req.sampling_params.stop_strs
                else None
            )
            finished = hit_length or hit_eos or matched_stop is not None
            finish_reason = (
                ("stop" if (hit_eos or matched_stop is not None) else "length")
                if finished
                else None
            )
            if (
                tok == self.toolcall_anchor_id
                and req.toolcall_anchor_len is None
                and not finished
            ):
                req.toolcall_anchor_len = req.input_ids.numel()
            reply.append(
                DetokenizeMsg(
                    uid=req.uid,
                    next_token=tok,
                    finished=finished,
                    finish_reason=finish_reason,
                    matched_stop=matched_stop,
                    stop_strs=req.sampling_params.stop_strs or None,
                )
            )
            if finished:
                break
        if finished:
            self.decode_manager.remove_req(req)
            self._free_req_resources(req)
            finished_out.add(req)

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
        if self.toolcall_anchor_id is not None and not batch.is_prefill:
            self.cache_manager.snapshot_toolcall_anchor(batch.reqs)
        if _dbg is not None:
            _qa = _time.perf_counter()
            _dbg.host_phase(_pk + "fw.anchor", _qa - _q1)
        forward_output = self.engine.forward_batch(batch, sample_args)
        if _dbg is not None:
            _q2 = _time.perf_counter()
            _dbg.host_phase(_pk + "fw.fbcall", _q2 - _qa)
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
