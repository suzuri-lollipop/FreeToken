"""CUDA graph for single-request, fixed-size prefill continuation chunks.

Small adaptive chunks are host-launch bound: a T=128 chunk issues ~2000 Python
kernel launches (~340 ms) against ~60 ms of GPU work (measured breakdown in
_scratch/agent3/RESULTS.md, rounds 5-6). A mid-prompt continuation chunk samples
nothing (ChunkedReq), so its graph runs the backbone only: KV / GDN / QSA state
advance, PLE rows pre-signaled exactly like the MTP spec graph, and the chunk's
inputs gathered IN-GRAPH from the persistent token_pool / page_table so a replay
picks up the staged scalars without any host-built tensors.

Capture follows the SpecGraphRunner protocol: taken after a real eager chunk of
the same shape, with the GDN live slot saved/restored around the warm, capture
and verification runs, and the replay's post-state compared against the eager
chunk's post-state before the graph is adopted. Any mismatch or capture failure
disables the runner and restores the eager post-state, so the engine continues
eager with correct state.

Phase 1 scope (everything else falls back to eager): the qsa_sparse backend, the
hybrid GDN track path, slot-direct promoted MoE staging, T == PREFILL_GRAPH_T,
single non-mm ChunkedReq continuations with cached_len > 0.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from freetoken.attention.linear import FLAMetadata
from freetoken.core import Batch
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.engine.engine import Engine

logger = init_logger(__name__)

# Must match the scheduler's adaptive chunk floor (_CHUNK_MIN): the controller
# settles there while a decode waits, so it is the only T worth capturing.
PREFILL_GRAPH_T = int(os.getenv("FREETOKEN_PREFILL_GRAPH_T", "128"))

# scalar staging layout (one pinned int64 row -> one H2D per chunk)
S_POS, S_TABLE, S_LIVE, S_SEQ, S_N = 0, 1, 2, 3, 4


def prefill_graph_enabled() -> bool:
    return os.getenv("FREETOKEN_PREFILL_GRAPH", "0") == "1"


class PrefillGraphRunner:
    """Lazy capture + per-chunk staging/replay of one T-bucket prefill chunk."""

    WARM_STEPS = 2  # eligible eager chunks before capturing (kernels/allocator warm)

    def __init__(self, engine: "Engine") -> None:
        self.engine = engine
        self.device = engine.device
        self.T = PREFILL_GRAPH_T
        self.graph: torch.cuda.CUDAGraph | None = None
        self.disabled = False
        self.warm = 0
        self._stream = torch.cuda.Stream()
        self._static = False

    # ------------------------------------------------------------------ eligibility
    def eligible(self, engine: "Engine", batch: Batch) -> bool:
        if self.disabled:
            return False
        if not batch.is_prefill or getattr(batch, "spec_mode", None) is not None:
            return False
        if batch.mm_gather_plan:
            return False
        reqs = batch.reqs
        if len(reqs) != 1:
            return False
        req = reqs[0]
        # Mid-prompt continuation chunks only: they sample nothing, so the graph can
        # run the backbone and return a dummy ForwardOutput.
        if req.can_decode or req.extend_len != self.T or req.cached_len <= 0:
            return False
        if getattr(req, "mm_items", None):
            return False
        if getattr(req, "spec_residual", None) is not None and not getattr(req, "spec_off", False):
            # The replay returns before the engine's residual-stash gate, so a graphed chunk
            # would commit rows the MTP head never gets a residual for -- hole in the span,
            # and the request declines for the rest of its life. Keep it eager while its
            # stash is live; a request already off-spec has nothing left to protect.
            return False
        if req.linear_slot_idx is None or engine.linear_state_pool is None:
            return False
        fla = batch.fla_metadata
        if fla is None or fla.track_dst is None:
            return False  # phase 1: the hybrid track path with one tracked req
        backend = engine.attn_backend
        if not hasattr(backend, "_block_table"):
            return False  # phase 1: qsa_sparse-style in-graph block-table gather
        cache = getattr(engine, "moe_offload_cache", None)
        if cache is None or getattr(cache, "promote_auto_frac", 0.0) <= 0:
            return False  # the captured MoE branch is the slot-direct promoted one
        from freetoken.layers.moe import _PREFILL_SLOT_DIRECT

        if not _PREFILL_SLOT_DIRECT:
            return False
        if getattr(engine, "spec_token_pool", None) is None:
            return False
        return True

    # ------------------------------------------------------------------ statics
    def _alloc_static(self, engine: "Engine") -> None:
        if self._static:
            return
        d = self.device
        T = self.T
        pool = engine.linear_state_pool
        km1 = pool.conv_states.shape[-1]
        from freetoken.kernel.fla.chunk import CHUNK_SIZE

        if T % CHUNK_SIZE or (T - 1) // CHUNK_SIZE != 1:
            # track constants below assume exactly one mid-chunk ×CHUNK boundary
            self.disabled = True
            raise ValueError(f"PREFILL_GRAPH_T={T} must sit in (CHUNK, 2*CHUNK]")
        self.h_scal = torch.empty(S_N, dtype=torch.int64, pin_memory=True)
        self.d_scal = torch.empty(S_N, dtype=torch.int64, device=d)
        self.c_off = torch.arange(T, dtype=torch.int64, device=d)
        self.d_positions = torch.empty(T, dtype=torch.int32, device=d)
        self.d_mrope = torch.empty(3, T, dtype=torch.int32, device=d)
        self.d_seq = torch.empty(1, dtype=torch.int32, device=d)
        self.d_live = torch.empty(1, dtype=torch.int32, device=d)
        self.d_table = torch.empty(1, dtype=torch.int64, device=d)
        self.d_track = torch.empty(1, dtype=torch.int64, device=d)
        # int64 cu buffer: fla's chunk helpers are tensor_cache-keyed and call
        # .to(torch.int64); a pre-typed buffer keeps the capture free of host ops
        # (same trick as SpecGraphRunner). QSA wants int32.
        self.c_cu64 = torch.tensor([0, T], dtype=torch.int64, device=d)
        self.c_cu32 = torch.tensor([0, T], dtype=torch.int32, device=d)
        self.c_t2r = torch.zeros(T, dtype=torch.int32, device=d)
        self.c_last = torch.tensor([T - 1], dtype=torch.int32, device=d)
        self.c_true = torch.tensor([True], device=d)
        # GDN track constants for the single mid-chunk boundary (one tracked req)
        self.c_h_row = torch.tensor([1], dtype=torch.int64, device=d)
        self.c_bound = torch.tensor([CHUNK_SIZE], dtype=torch.int64, device=d)
        self.c_conv_src = torch.tensor(
            [[CHUNK_SIZE - km1 + j for j in range(km1)]], dtype=torch.int64, device=d
        )
        self.d_next = torch.zeros(1, dtype=torch.int32, device=d)
        self.next_cpu = torch.zeros(1, dtype=torch.int32, pin_memory=True)

        shadow = SimpleNamespace(
            extend_len=T, cached_len=0, device_len=T, table_idx=0, linear_slot_idx=0,
            mamba_ping_pong=(0, 0), mamba_next_track_idx=0, mamba_last_track_seqlen=None,
            mamba_restore_src=None, spec_slot_idx=None,
        )
        batch = Batch(reqs=[shadow], phase="prefill")
        batch.padded_reqs = [shadow]
        batch.positions = self.d_positions
        batch.mrope_positions = self.d_mrope
        batch.linear_table_idx = self.d_live
        batch.input_ids = torch.zeros(T, dtype=torch.int32, device=d)
        batch.out_loc = torch.zeros(T, dtype=torch.int64, device=d)
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.c_cu64,
            cache_indices=self.d_live,
            has_initial_state=self.c_true,
            fresh_state_indices=None,
            track_dst=self.d_track,
            track_h_row=self.c_h_row,
            track_conv_src=self.c_conv_src,
            track_boundary_row=self.c_bound,
        )
        from freetoken.attention.qsa_sparse import QSASparseMetadata

        md = QSASparseMetadata(
            is_decode=False,
            last_indices=self.c_last,
            qo_indptr_cpu=torch.tensor([0, T], dtype=torch.int32),
            kv_len_cpu=torch.tensor([T], dtype=torch.int32),
        )
        md.cu_seqlens = self.c_cu32
        md.token_to_req = self.c_t2r
        md.seq_lens = self.d_seq
        batch.attn_metadata = md
        self.batch = batch
        self.md = md
        self._static = True

    # ------------------------------------------------------------------ staging
    def _stage(self, batch: Batch, req) -> None:
        h = self.h_scal
        h[S_POS] = req.cached_len
        h[S_TABLE] = req.table_idx
        h[S_LIVE] = req.linear_slot_idx
        h[S_SEQ] = req.device_len
        self.d_scal.copy_(h, non_blocking=True)
        # the scheduler's build_fla_metadata already resolved this chunk's track slot
        self.d_track.copy_(batch.fla_metadata.track_dst.reshape(1), non_blocking=True)

    def _fanout(self) -> None:
        """In-graph derivations from the staged scalars (recorded once at capture)."""
        self.d_positions.copy_((self.d_scal[S_POS] + self.c_off).to(torch.int32))
        self.d_mrope.copy_(self.d_positions.unsqueeze(0).expand(3, self.T))
        self.d_seq.copy_(self.d_scal[S_SEQ : S_SEQ + 1].to(torch.int32))
        self.d_live.copy_(self.d_scal[S_LIVE : S_LIVE + 1].to(torch.int32))
        self.d_table.copy_(self.d_scal[S_TABLE : S_TABLE + 1])

    def _gather_inputs(self, engine: "Engine") -> None:
        """In-graph gathers from the persistent engine tensors (auto-fresh per replay)."""
        batch = self.batch
        table_col = self.d_table.expand(self.T)
        pos64 = self.d_positions.to(torch.int64)
        batch.input_ids = engine.spec_token_pool[table_col, pos64]
        batch.out_loc = engine.page_table[table_col, pos64]
        self.md.ring_slots = self.d_table.to(torch.int32)
        self.md.block_table = engine.attn_backend._block_table(self.md.ring_slots.to(torch.int64))

    def _body(self, engine: "Engine", model) -> None:
        self._fanout()
        self._gather_inputs(engine)
        batch = self.batch
        # backbone only: a continuation chunk samples nothing (no lm_head, no sampler)
        model.model.forward_with_residual(batch.input_ids, batch)

    # ------------------------------------------------------------------ state save/restore
    @staticmethod
    def save_state(engine: "Engine", slot: int):
        pool = engine.linear_state_pool
        tensors = [pool.conv_states, pool.recurrent_states, *pool.slot_states.values()]
        return [t[:, slot].clone() for t in tensors]

    @staticmethod
    def restore_state(engine: "Engine", slot: int, saved) -> None:
        pool = engine.linear_state_pool
        tensors = [pool.conv_states, pool.recurrent_states, *pool.slot_states.values()]
        for tensor, value in zip(tensors, saved, strict=True):
            tensor[:, slot].copy_(value)

    # KV rows the chunk wrote (sparse-attention layers only, addressed by out_loc).
    # The GDN check alone would miss a broken out_loc/page-table gather: GDN reads the
    # token stream, KV writes ride out_loc -- so the capture verification covers both.
    @staticmethod
    def _kv_caches(engine: "Engine"):
        lids = sorted(getattr(engine.attn_backend, "_idx_slot", {}).keys())
        pool = engine.kv_cache
        return [(pool.k_cache(lid), pool.v_cache(lid)) for lid in lids]

    def save_kv_rows(self, engine: "Engine", batch: Batch):
        caches = self._kv_caches(engine)
        if not caches:
            # a backend without the qsa _idx_slot map would make the KV half of the
            # capture verification vacuous -- say so instead of silently passing
            logger.warning_rank0(
                "prefill graph: no sparse-attention KV caches found; "
                "capture verification covers GDN state only"
            )
        loc = batch.out_loc.to(torch.int64)
        return [(k[loc].clone(), v[loc].clone()) for k, v in caches]

    @staticmethod
    def restore_kv_rows(engine: "Engine", batch: Batch, saved) -> None:
        loc = batch.out_loc.to(torch.int64)
        for (k, v), (ks, vs) in zip(PrefillGraphRunner._kv_caches(engine), saved, strict=True):
            k[loc] = ks
            v[loc] = vs

    @staticmethod
    def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
        if torch.equal(a, b):
            return True
        return torch.allclose(a.float(), b.float(), rtol=1e-3, atol=1e-3)

    # ------------------------------------------------------------------ capture
    def invalidate(self) -> None:
        """Drop the capture (a cache rebuild moved the pools baked into the graph)."""
        self.graph = None
        self.warm = 0

    def capture_after_chunk(self, engine: "Engine", model, batch: Batch, req,
                            before_state, after_state) -> bool:
        """Capture on the just-completed eager chunk's context; the GDN live slot is
        restored around every warm/capture/replay execution, and the final replay must
        reproduce the eager chunk's post-state bitwise (or within 1e-3) or the runner
        disables itself. On any failure the eager post-state is restored, so the engine
        continues eager with correct state."""
        table = getattr(model, "_ple_table", None)
        if table is None or not hasattr(table, "prefill_runs"):
            self.disabled = True
            logger.warning_rank0("prefill graph needs the disk PLE backend; staying eager")
            return False
        cache = getattr(engine, "moe_offload_cache", None)
        slot = req.linear_slot_idx
        after_kv = self.save_kv_rows(engine, batch)  # the eager chunk's KV rows
        try:
            self._alloc_static(engine)
            self._stage(batch, req)
            if cache is not None:
                cache._promote_frozen = True  # no policy flips mid-capture
            runs = table.prefill_runs(batch)

            def _fill() -> None:
                table.fill(runs, graph=True)

            _fill()
            self.restore_state(engine, slot, before_state)
            torch.cuda.synchronize(self.device)
            with engine.ctx.forward_batch(self.batch):
                s = self._stream
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    self._body(engine, model)
                torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize(self.device)
            diag = os.getenv("FREETOKEN_PREFILL_GRAPH_DIAG", "0")
            if diag != "0":
                # Eager determinism control: run the (eager!) warm body a second and
                # third time from the same restored state and compare. Modes:
                #   1 = side-stream warms (as the capture warm runs)
                #   2 = engine-stream warms + preallocated comparison buffers +
                #       KV-row comparison (bisects side-stream and allocator churn)
                engine_stream_warm = diag == "2"
                names = ["conv", "recurrent"] + [
                    f"slot:{k}" for k in engine.linear_state_pool.slot_states
                ]
                w1 = [t.clone() for t in before_state]
                w2 = [t.clone() for t in before_state]

                def _save_into(dest) -> None:
                    pool = engine.linear_state_pool
                    tensors = [pool.conv_states, pool.recurrent_states,
                               *pool.slot_states.values()]
                    for dd, tt in zip(dest, tensors, strict=True):
                        dd.copy_(tt[:, slot])

                def _warm() -> None:
                    _fill()
                    self.restore_state(engine, slot, before_state)
                    torch.cuda.synchronize(self.device)
                    with engine.ctx.forward_batch(self.batch):
                        if engine_stream_warm:
                            self._body(engine, model)
                        else:
                            s = self._stream
                            s.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(s):
                                self._body(engine, model)
                            torch.cuda.current_stream().wait_stream(s)
                    torch.cuda.synchronize(self.device)

                _warm()
                _save_into(w1)
                k1 = self.save_kv_rows(engine, batch)
                _warm()
                _save_into(w2)
                k2 = self.save_kv_rows(engine, batch)
                bad = [n for n, a, b in zip(names, w1, w2) if not self._same(a, b)]
                kvbad = sum(
                    0 if (self._same(a[0], b[0]) and self._same(a[1], b[1])) else 1
                    for a, b in zip(k1, k2)
                )
                dmax = max(
                    ((a.float() - b.float()).abs().max().item() if n in bad else 0.0)
                    for n, a, b in zip(names, w1, w2)
                ) if bad else 0.0
                logger.info_rank0(
                    f"[p6-diag{diag}] warm1==warm2 gdn: {not bad}"
                    + (f" (differ: {', '.join(bad)}; max|d|={dmax:.3e})" if bad else "")
                    + f" | kv differing layers: {kvbad}/{len(k1)}"
                )
            self.restore_state(engine, slot, before_state)
            torch.cuda.synchronize(self.device)

            _fill()
            graph = torch.cuda.CUDAGraph()
            with engine.ctx.forward_batch(self.batch):
                with torch.cuda.graph(graph, stream=self._stream):
                    self._body(engine, model)
            torch.cuda.synchronize(self.device)
            self.restore_state(engine, slot, before_state)
            torch.cuda.synchronize(self.device)

            _fill()
            graph.replay()
            torch.cuda.synchronize(self.device)
            got = self.save_state(engine, slot)
            got_kv = self.save_kv_rows(engine, batch)
            if os.getenv("FREETOKEN_PREFILL_GRAPH_DIAG", "0") != "0":
                # Discriminate kernel nondeterminism from a structural input mismatch:
                # restore the pre-chunk state and replay again. replay1 != replay2 =>
                # the graphed kernels themselves are nondeterministic; replay1 ==
                # replay2 != eager => the replay consumed different inputs.
                self.restore_state(engine, slot, before_state)
                torch.cuda.synchronize(self.device)
                _fill()
                graph.replay()
                torch.cuda.synchronize(self.device)
                got2 = self.save_state(engine, slot)
                r1r2 = all(self._same(a, b) for a, b in zip(got, got2))
                r1e = all(self._same(a, b) for a, b in zip(got, after_state))
                logger.info_rank0(
                    f"[p6-diag] replay1==replay2: {r1r2} | replay1==eager: {r1e}"
                )
                # put the live state back to the authoritative eager post-state so
                # adoption/verification below are independent of replay2's outcome
                self.restore_state(engine, slot, after_state)
                self.restore_kv_rows(engine, batch, after_kv)
                torch.cuda.synchronize(self.device)
            state_ok = all(self._same(a, b) for a, b in zip(got, after_state))
            kv_ok = all(
                self._same(ak, bk) and self._same(av, bv)
                for (ak, av), (bk, bv) in zip(got_kv, after_kv)
            )
            if not (state_ok and kv_ok):
                # pinpoint the divergence: which state tensor, and at what magnitude
                # (ulp-scale => kernel run-to-run nondeterminism; large => structural)
                names = ["conv", "recurrent"] + [
                    f"slot:{k}" for k in engine.linear_state_pool.slot_states
                ]
                detail = []
                for name, a, b in zip(names, got, after_state):
                    if not self._same(a, b):
                        d = (a.float() - b.float()).abs()
                        scale = b.float().abs().max().item()
                        detail.append(
                            f"{name}: max|d|={d.max().item():.3e} (state max {scale:.1e})"
                        )
                self.restore_state(engine, slot, after_state)
                self.restore_kv_rows(engine, batch, after_kv)
                self.disabled = True
                self.graph = None
                logger.warning_rank0(
                    f"prefill graph replay diverged from the eager chunk "
                    f"(gdn={state_ok} kv={kv_ok}); staying eager. "
                    + "; ".join(detail[:6])
                )
                return False
            self.graph = graph
            logger.info_rank0(
                f"prefill chunk graph captured (single-req T={self.T} continuation)"
            )
            return True
        except Exception as e:  # noqa: BLE001
            # best effort: put the live slot and the chunk's KV rows back to the eager
            # chunk's post-state and fall back to eager permanently (this runner has no
            # host-state side effects)
            try:
                self.restore_state(engine, slot, after_state)
                self.restore_kv_rows(engine, batch, after_kv)
            except Exception:  # noqa: BLE001
                pass
            self.disabled = True
            self.graph = None
            logger.error_rank0(f"prefill graph capture failed ({e!r}); staying eager")
            return False
        finally:
            if cache is not None:
                cache._promote_frozen = False

    # ------------------------------------------------------------------ replay
    def run(self, engine: "Engine", model, batch: Batch):
        from freetoken.engine.engine import ForwardOutput

        req = batch.reqs[0]
        cache = getattr(engine, "moe_offload_cache", None)
        if cache is not None:
            # the in-graph rotation (baked at layer 0) refills the pin each replay;
            # the host side of the policy feedback lives here, between replays
            cache.harvest_promote_stats()
        self._stage(batch, req)
        table = getattr(model, "_ple_table", None)
        table.fill(table.prefill_runs(batch), graph=True)
        self.graph.replay()
        event = torch.cuda.Event()
        event.record(engine.stream)
        # mid-prompt chunks sample nothing; the drain's ChunkedReq branch never reads
        # the tokens, it only syncs the event
        return ForwardOutput(self.d_next, self.next_cpu, event)
