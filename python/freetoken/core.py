from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Tuple

import torch

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend, BaseAttnMetadata
    from freetoken.attention.linear import FLAMetadata
    from freetoken.kvcache import BaseCacheHandle, BaseKVCachePool
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.blocks import FeatureSink


@dataclass
class SamplingParams:
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024
    # Stop strings (OpenAI `stop` / Anthropic `stop_sequences`). Generation finishes when one
    # appears in the decoded output; the matched substring (and anything after) is trimmed.
    stop_strs: list[str] = field(default_factory=list)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    # min-p (typicality) filtering: tokens below min_p * p_max are dropped before top-k/p.
    # At high temperature it trims the tail that produces particle-drop / fragment decay.
    min_p: float = 0.0
    # Stamp the ReqStats snapshot on every reply, not just the terminal one (llama.cpp's
    # timings_per_token; the API layer has already ANDed it with --enable-timings-report).
    timings_per_token: bool = False

    @property
    def is_greedy(self) -> bool:
        return self.temperature <= 0.0 or self.top_k == 1


@dataclass
class ReqStats:
    """Per-request timing and draft counters, stamped by the scheduler with
    time.perf_counter(). Spans follow llama.cpp's server_slot_stats plus a queue
    span: queue = waiting queue -> first prefill chunk launch, prompt = first
    chunk launch -> first sampled token on the host (chunk gaps included),
    predicted = first token -> last committed token. One instance is shared by
    every chunk Req of a request, so the stamps survive chunked prefill."""

    enqueued_at: float | None = None     # entered the scheduler's waiting queue
    first_chunk_at: float | None = None  # first prefill chunk launched
    first_token_at: float | None = None  # first sampled token back on the host
    last_token_at: float | None = None   # last committed token
    cache_n: int = 0                     # prompt tokens served from the prefix cache at admission
    prompt_n: int = 0                    # prompt tokens actually computed
    draft_n: int = 0                     # drafted tokens summed over verify steps
    draft_n_accepted: int = 0            # accepted drafts that were committed, after stop truncation

    def stamp_token(self, now: float) -> None:
        if self.first_token_at is None:
            self.first_token_at = now
        self.last_token_at = now

    def finish(self, completion_tokens: int) -> dict | None:
        """Raw spans and counts for the terminal reply; None when no prefill ever
        launched or no token was committed (errors and aborts emit nothing)."""
        if self.first_chunk_at is None or self.first_token_at is None:
            return None
        return {
            "queue_ms": (self.first_chunk_at - self.enqueued_at) * 1000.0
            if self.enqueued_at is not None else 0.0,
            "cache_n": self.cache_n,
            "prompt_n": self.prompt_n,
            "prompt_ms": (self.first_token_at - self.first_chunk_at) * 1000.0,
            "predicted_n": completion_tokens,
            # 1 us floor as in llama.cpp: the first token can land in the same instant
            "predicted_ms": max((self.last_token_at - self.first_token_at) * 1000.0, 0.001),
            "draft_n": self.draft_n,
            "draft_n_accepted": self.draft_n_accepted,
        }


@dataclass(eq=False)
class Req:
    input_ids: torch.Tensor  # cpu tensor
    table_idx: int
    cached_len: int
    output_len: int
    uid: int
    sampling_params: SamplingParams
    cache_handle: BaseCacheHandle
    # per-item processor outputs and the tokenizer's precomputed mrope rows and delta
    mm_items: list | None = None
    mrope_positions_full: torch.Tensor | None = None  # [3, prompt_len] int32, CPU
    mrope_delta: int = 0
    # The whole prompt's length; 0 = this request's input_ids (a ChunkedReq's end at its chunk).
    prompt_len: int = 0

    # --- hybrid-radix (GDN linear-state) per-request slots; None for non-hybrid models or
    # until allocated from LinearStatePool. Set by the scheduler (P2). ---
    linear_slot_idx: int | None = None              # live GDN state slot (sglang mamba_pool_idx)
    mamba_ping_pong: tuple[int, int] | None = None  # 2 donatable track slots under overlap
    mamba_next_track_idx: int = 0                   # which ping-pong slot is the next snapshot dst (0/1)
    mamba_last_track_seqlen: int | None = None      # chunk-aligned committed len of the last snapshot
    mamba_restore_src: int | None = None            # on a prefix hit: tree snapshot slot to COW into the live slot (first chunk only)
    mamba_restore_host: int | None = None           # snapshot host-tiered: host-cache buffer id to H2D instead of the slot COW above
    swa_evicted_seqlen: int = 0                      # SWA radix: positions < this had their swa KV freed (slid out of window) during decode
    decode_batch_idx: int = 0                        # SWA radix: # of decode forwards done; the proactive free_swa skips the first (overlap guard)
    # Set once, at the first sampled tool-call opener token (scheduler detection): the state
    # length just after that token (its index + 1). A client-side rewrite of the echoed tool
    # call diverges strictly after this point, so it is the deepest reuse boundary that
    # survives such a rewrite. GDN: the state is frozen into a ping-pong slot when cached_len
    # reaches it (snapshot_toolcall_anchor) and donated at finish. SWA: caps the proactive
    # out-of-window eviction so the window ending here stays resumable.
    toolcall_anchor_len: int | None = None
    # Abort arrived while this request's forward was in flight (overlap scheduling). The abort
    # handler must not free resources under an in-flight forward; it sets this flag and
    # _process_last_data frees the request when the batch drains (after copy_done.synchronize).
    aborted: bool = False
    # Timing/draft counters shared with every other chunk Req of this request (set by
    # the PrefillManager from the PendingReq's instance; a default blank otherwise).
    stats: ReqStats = field(default_factory=ReqStats, init=False, repr=False)
    output_token_counts: torch.Tensor | None = field(default=None, init=False, repr=False)

    # Phase timings in ms, measured in the scheduler and reported to the frontend on the first
    # sampled token: time held in the prefill queue, and the schedule->drain prefill window.
    queue_ms: float = 0.0
    prefill_ms: float = 0.0
    # Tokens forwarded in the prefill batch currently in flight, snapshotted when it was
    # scheduled: extend_len reads 0 by the time the drain splits that batch's wall time.
    prefill_chunk_tokens: int = 0

    def __post_init__(self) -> None:
        assert self.input_ids.is_cpu
        self.device_len = len(self.input_ids)
        self.max_device_len = len(self.input_ids) + self.output_len
        self.prompt_len = self.prompt_len or self.device_len
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len
        self._alloc_ids_buf()

    def _alloc_ids_buf(self) -> None:
        self._ids_buf = torch.empty(self.max_device_len, dtype=self.input_ids.dtype)
        self._ids_buf[: self.device_len] = self.input_ids
        self.input_ids = self._ids_buf[: self.device_len]

    @property
    def remain_len(self) -> int:
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        self.cached_len = self.device_len
        self.device_len += 1

    def widen_pending(self, k: int) -> None:
        """Extend the pending token at cached_len by k provisional draft tokens."""
        self.device_len += k

    def commit_pending(self, n: int) -> None:
        """Make n tokens final and leave the last committed one pending at cached_len."""
        self.cached_len += n
        self.device_len = self.cached_len + 1

    def append_host(self, next_token: torch.Tensor) -> None:
        n = self.input_ids.numel()
        m = n + next_token.numel()
        assert m <= self.max_device_len
        self._ids_buf[n:m] = next_token
        self.input_ids = self._ids_buf[:m]

    def truncate_host(self, n: int) -> None:
        """Drop staged speculative rows from the host view after a partial accept."""
        assert n <= self.input_ids.numel()
        self.input_ids = self._ids_buf[:n]

    @property
    def can_decode(self) -> bool:
        return self.remain_len > 0

    def __repr__(self) -> str:
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )



@dataclass
class Batch:
    reqs: List[Req]
    phase: Literal["prefill", "decode", "verify"]
    # query rows per request: 1 on decode, more on verify, None on prefill (ragged)
    rows_per_req: int | None = None
    # these fields should be set by scheduler
    input_ids: torch.Tensor = field(init=False)
    positions: torch.Tensor = field(init=False)
    # [3, n] t/h/w rope positions on mrope models; positions keeps its sequence-index meaning for token_pool / page_table
    mrope_positions: torch.Tensor | None = field(default=None, init=False)
    out_loc: torch.Tensor | None = field(init=False)
    # Per-forward GatedDeltaNet metadata (cu_seqlens / cache_indices / continuation
    # flags), built once and shared by all GDN layers. Lazily built by the GDN op if
    # the scheduler/graph didn't set it.
    fla_metadata: "FLAMetadata | None" = field(default=None, init=False)
    padded_reqs: List[Req] = field(init=False)
    # DSV4 paged-KV out-locations for this batch (None for non-DSV4 models). Set by the scheduler.
    # This decode batch's padded per-row page-table rows. Attention backends that must read
    # positions anywhere in a request's history snapshot those rows before a captured replay
    # (DSV4), since the next batch's allocate_paged mutates the live table.
    active_table_idx: "torch.Tensor | None" = None
    # Running requests riding a phase="prefill" batch (FREETOKEN_MIXED_PREFILL_TOKENS):
    # extend_len==1 rows of live decodes, not prompt chunks. The drain, the SWA drivers,
    # the toolcall-anchor snapshot and the prefill-time charge branch on this membership,
    # never on phase. None for a pure prefill / decode batch.
    decoding_reqs: "List[Req] | None" = field(default=None, init=False)
    # this field should be set by attention backend
    attn_metadata: BaseAttnMetadata = field(init=False)
    # concatenated multimodal soft-token embeddings for a prefill batch (or None) and the batch rows they land on
    mm_embeds: torch.Tensor | None = field(default=None, init=False)
    mm_rows: torch.Tensor | None = field(default=None, init=False)
    # per batch token, the end (exclusive, in its request) of the image span holding it, 0 for text: the block a bidirectional layer attends within
    mm_block_ends: torch.Tensor | None = field(default=None, init=False)
    # this chunk's cache-miss items to encode and the gather plan [(uid, hash, row_lo, row_hi, n, pos), ...] in scatter order
    mm_encoder_jobs: list | None = field(default=None, init=False)
    mm_gather_plan: list | None = field(default=None, init=False)
    # Prefill log stats snapshotted at schedule time (before forward's complete_one()
    # advances cached_len), so the prefill log reports the tokens actually forwarded and
    # the prefix-cache hit -- matching SGLang's #new-token / #cached-token. Set by the
    # PrefillManager; 0 on decode batches.
    log_new_tokens: int = field(default=0, init=False)
    log_cached_tokens: int = field(default=0, init=False)
    # perf_counter() stamp of when the batch was scheduled; the status reporter measures
    # input throughput over this batch's own schedule->completion window. 0 = unstamped.
    scheduled_at: float = field(default=0.0, init=False)
    # (uid, complete prompt length, prefix-cache hit) for requests entering their first
    # prepared prefill batch. The scheduler turns these into PromptAdmittedMsg only AFTER
    # _prepare_batch succeeds. Continuation chunks leave this empty, so accounting is
    # exactly-once.
    prompt_admissions: List[Tuple[int, int, int]] = field(default_factory=list, init=False)
    # per-row hidden states a draft model reads from this forward; None when nothing reads them
    features: "FeatureSink | None" = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.rows_per_req is None and self.phase == "decode":
            self.rows_per_req = 1
        # without it, input_mapping[0][::None] would read every row as its own request
        assert self.phase != "verify" or self.rows_per_req is not None, "a verify batch needs rows_per_req"

    # Dual-microbatch decode (FREETOKEN_DUAL_STREAM_DECODE): a bs4 decode step runs as
    # two bs2 half-batches on skewed streams. Each half is a Batch VIEW carrying its
    # slice of the static graph buffers plus these two markers:
    # ple_row_offset -- this half's first row in the graph-pinned PLE staging buffer
    #   (the fill stages rows in FULL-batch order before the launch);
    # dual_slot -- 0 for the leading half (main stream), 1 for the trailing half
    #   (side stream): it selects the per-instance all-reduce, QSA staging profile
    #   and MoE fetch plan, keeping the two halves' shared-state mutations disjoint.
    ple_row_offset: int = field(default=0, init=False)
    dual_slot: int = field(default=-1, init=False)

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    def get_attn_positions(self) -> torch.Tensor:
        return self.mrope_positions if self.mrope_positions is not None else self.positions

    @property
    def size(self) -> int:
        return len(self.reqs)

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)


@dataclass
class Context:
    page_size: int
    # NOTE: this table always treat page_size = 1
    page_table: torch.Tensor = field(init=False)
    attn_backend: BaseAttnBackend = field(init=False)
    kv_cache: BaseKVCachePool = field(init=False)
    # Per-request recurrent state for GatedDeltaNet layers; set by the engine for
    # hybrid linear-attention models, otherwise None.
    linear_state_pool: LinearStatePool | None = None
    # The active batch is per-thread for the piped-prefill thread (FREETOKEN_PIPED_PREFILL):
    # a prefill forward runs there WHILE the main loop issues a decode step, and each
    # forward's layers must read their own batch. The main thread keeps the plain instance
    # slot, so capture, dual-decode swap_batch and every existing path are untouched; a
    # registered pipe thread sees only its thread-local slot.
    _batch: Batch | None = field(default=None, init=False)
    _pipe_tl: "threading.local | None" = field(default=None, init=False, repr=False)

    def register_pipe_thread(self) -> None:
        """Call ONCE from the piped-prefill worker thread; marks it as batch-scoped."""
        if self._pipe_tl is None:
            self._pipe_tl = threading.local()
        self._pipe_tl.is_pipe = True

    @property
    def _active(self) -> "Batch | None":
        if self._pipe_tl is not None and getattr(self._pipe_tl, "is_pipe", False):
            return getattr(self._pipe_tl, "batch", None)
        return self._batch

    @_active.setter
    def _active(self, batch: "Batch | None") -> None:
        if self._pipe_tl is not None and getattr(self._pipe_tl, "is_pipe", False):
            self._pipe_tl.batch = batch
        else:
            self._batch = batch

    @property
    def batch(self) -> Batch:
        cur = self._active
        assert cur is not None, "No active batch in context"
        return cur

    @contextmanager
    def override(
        self,
        *,
        kv_cache: BaseKVCachePool | None = None,
        attn_backend: BaseAttnBackend | None = None,
        hide_target_state: bool = False,
    ):
        """Swap the KV pool and attention backend for a draft forward; hide_target_state also hides the target's GDN pool."""
        swaps = {}
        if kv_cache is not None:
            swaps["kv_cache"] = kv_cache
        if attn_backend is not None:
            swaps["attn_backend"] = attn_backend
        if hide_target_state:
            swaps["linear_state_pool"] = None
        saved = {name: getattr(self, name) for name in swaps}
        try:
            for name, value in swaps.items():
                setattr(self, name, value)
            yield
        finally:
            for name, value in saved.items():
                setattr(self, name, value)

    @contextmanager
    def forward_batch(self, batch: Batch, **override):
        assert self._active is None, "Nested forward_batch is not allowed"
        self._active = batch
        try:
            with self.override(**override):
                yield
        finally:
            self._active = None

    @contextmanager
    def swap_batch(self, batch: Batch):
        """Re-entrant ``forward_batch``: restores the PREVIOUS batch on exit.

        The dual-microbatch decode driver swaps the half-batch views in and out
        under the engine's outer forward_batch scope (per half, per layer)."""
        prev = self._active
        self._active = batch
        try:
            yield
        finally:
            self._active = prev


_GLOBAL_CTX: Context | None = None


def set_global_ctx(ctx: Context):
    global _GLOBAL_CTX
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
