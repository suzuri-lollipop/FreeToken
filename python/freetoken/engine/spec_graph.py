"""CUDA graph for the bs-1 MTP spec step (Phase 3; design: _scratch/mtp_design.md).

One capture serves every verify/replay step of every request: the two-row shape is
invariant, and every per-step value enters through either a small static buffer staged
before the replay or an in-graph gather from persistent engine tensors (token_pool,
page_table, the linear-state pool). Recorded op order: the GDN/PLE snapshot
(live -> scratch), the two-row main forward (chunk GDN, extend QSA, device-side MoE
decode staging), the dual argmax, the accept/canary compare, the MTP head pass
producing the next draft, the accepted token's pool write and the payload pack.

The PLE rows are filled AT ISSUE TIME (``fill_spec_rows`` stages the graph-pinned rows
and signals the flag) so the captured lookup's memop WAIT is already satisfied when the
replay reaches it -- legal because a spec step's two input tokens ([last placed, draft]
or the replay pair) are host-known before dispatch. The reject-side state restore stays
eager in the drain: the normal_loop drains before the next issue, so no branch-free
in-graph commit is needed. Cross-step safety rests on the audits recorded in the design
doc: the QSA extend kernels take no host scalars (grids derive from the constant row
count), the fla chunk metadata is static except the staged slot, PLE lookups consume
positional graph rows, and block_table/out_loc/input_ids are re-gathered IN the graph
from the persistent tensors, so page changes need no recapture.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from freetoken.attention.linear import FLAMetadata
from freetoken.core import Batch
from freetoken.utils import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
    from freetoken.engine.engine import Engine
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

# scalar staging layout (one pinned int64 row -> one H2D per step)
S_POS, S_TABLE, S_LIVE, S_SCRATCH, S_DRAFT, S_MODE, S_SEQ, S_N = 0, 1, 2, 3, 4, 5, 6, 8
MODE_VERIFY, MODE_REPLAY = 0, 1


class SpecGraphRunner:
    """Lazy capture + per-step staging/replay of the two-row spec step."""

    WARM_STEPS = 3  # eager spec steps before capturing (kernel/allocator warm at T=2)

    def __init__(self, engine: "Engine") -> None:
        self.engine = engine
        self.device = engine.device
        self.graph: torch.cuda.CUDAGraph | None = None
        self.disabled = False
        self.warm = 0
        self._stream = torch.cuda.Stream()

    # ------------------------------------------------------------------ statics
    def _alloc_static(self, model: "Qwen4ExpForCausalLM") -> None:
        d = self.device
        self.h_scal = torch.empty(S_N, dtype=torch.int64, pin_memory=True)
        self.d_scal = torch.empty(S_N, dtype=torch.int64, device=d)
        self.d_positions = torch.empty(2, dtype=torch.int32, device=d)
        self.d_mrope = torch.empty(3, 2, dtype=torch.int32, device=d)
        self.d_seq = torch.empty(1, dtype=torch.int32, device=d)
        self.d_live = torch.empty(1, dtype=torch.int32, device=d)
        self.d_scratch = torch.empty(1, dtype=torch.int64, device=d)
        self.d_table = torch.empty(1, dtype=torch.int64, device=d)
        self.c_off = torch.tensor([0, 1], dtype=torch.int64, device=d)
        # two cu_seqlens buffers: fla's chunk path calls .to(torch.int64) on the metadata
        # cu_seqlens and its prepare_chunk_* helpers are tensor_cache-keyed -- the int64
        # buffer makes that .to() a no-op so the warm run fills the cache and the capture
        # never sees their host-side ops (.tolist/new_tensor). QSA wants int32.
        self.c_cu64 = torch.tensor([0, 2], dtype=torch.int64, device=d)
        self.c_cu32 = torch.tensor([0, 2], dtype=torch.int32, device=d)
        self.c_t2r = torch.tensor([0, 0], dtype=torch.int32, device=d)
        self.c_last = torch.tensor([1], dtype=torch.int32, device=d)
        self.c_true = torch.tensor([True], device=d)
        self.d_payload = torch.empty(4, dtype=torch.int64, device=d)
        self.d_next = torch.empty(1, dtype=torch.int32, device=d)
        self.payload_cpu = torch.empty(4, dtype=torch.int64, pin_memory=True)
        self.next_cpu = torch.empty(1, dtype=torch.int32, pin_memory=True)

        shadow = SimpleNamespace(
            extend_len=2, cached_len=0, device_len=2, table_idx=0, linear_slot_idx=0,
        )
        batch = Batch(reqs=[shadow], phase="prefill")
        batch.padded_reqs = [shadow]
        batch.spec_mode = "verify"  # the MoE routing gate only checks not-None
        batch.positions = self.d_positions
        batch.mrope_positions = self.d_mrope
        batch.linear_table_idx = self.d_live
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.c_cu64,
            cache_indices=self.d_live,
            has_initial_state=self.c_true,
            fresh_state_indices=None,
        )
        from freetoken.attention.qsa_sparse import QSASparseMetadata

        md = QSASparseMetadata(
            is_decode=False,
            last_indices=self.c_last,
            qo_indptr_cpu=torch.tensor([0, 2], dtype=torch.int32),
            kv_len_cpu=torch.tensor([2], dtype=torch.int32),
        )
        md.cu_seqlens = self.c_cu32
        md.token_to_req = self.c_t2r
        md.seq_lens = self.d_seq
        batch.attn_metadata = md
        self.batch = batch
        self.md = md

    # ------------------------------------------------------------------ staging
    def _stage(self, batch: "Batch", req) -> None:
        # rows are [cached_len, cached_len+1]; device_len == cached_len + 2 here
        h = self.h_scal
        h[S_POS] = req.cached_len
        h[S_TABLE] = req.table_idx
        h[S_LIVE] = req.linear_slot_idx
        h[S_SCRATCH] = req.spec_slot_idx
        h[S_DRAFT] = batch.spec_draft_id
        h[S_MODE] = MODE_VERIFY if batch.spec_mode == "verify" else MODE_REPLAY
        h[S_SEQ] = req.device_len
        self.d_scal.copy_(h, non_blocking=True)

    def _fanout(self) -> None:
        """In-graph derivations from the staged scalars (recorded once at capture)."""
        self.d_positions.copy_((self.d_scal[S_POS] + self.c_off).to(torch.int32))
        self.d_mrope.copy_(self.d_positions.unsqueeze(0).expand(3, 2))
        self.d_seq.copy_(self.d_scal[S_SEQ:S_SEQ + 1].to(torch.int32))
        self.d_live.copy_(self.d_scal[S_LIVE:S_LIVE + 1].to(torch.int32))
        self.d_scratch.copy_(self.d_scal[S_SCRATCH:S_SCRATCH + 1])
        self.d_table.copy_(self.d_scal[S_TABLE:S_TABLE + 1])

    def _gather_inputs(self, engine: "Engine") -> None:
        """In-graph gathers from the persistent engine tensors (auto-fresh per replay)."""
        batch = self.batch
        table_col = self.d_table.expand(2)
        pos64 = self.d_positions.to(torch.int64)
        batch.input_ids = engine.spec_token_pool[table_col, pos64]
        batch.out_loc = engine.page_table[table_col, pos64]
        backend = engine.attn_backend
        self.md.ring_slots = self.d_table.to(torch.int32)
        self.md.block_table = backend._block_table(self.md.ring_slots.to(torch.int64))

    def _snapshot(self, engine: "Engine") -> None:
        """In-graph GDN+PLE state snapshot (live -> scratch), slot-indexed on device."""
        pool = engine.linear_state_pool
        live, scratch = self.d_live.to(torch.int64), self.d_scratch
        pool.conv_states.index_copy_(1, scratch, pool.conv_states.index_select(1, live))
        pool.recurrent_states.index_copy_(1, scratch, pool.recurrent_states.index_select(1, live))
        for t in pool.slot_states.values():
            t.index_copy_(1, scratch, t.index_select(1, live))

    def _body(self, engine: "Engine", model: "Qwen4ExpForCausalLM") -> None:
        self._fanout()
        self._gather_inputs(engine)
        self._snapshot(engine)
        batch = self.batch
        mixed, residual = model.model.forward_with_residual(batch.input_ids, batch)
        logits = model.full_vocab_logits(mixed)
        y1 = logits[0].argmax(-1)
        y2 = logits[1].argmax(-1)
        draft_in = torch.stack([y1, y2]).to(torch.int32)
        d = model.draft(residual, draft_in, batch)
        # accept: verify compares against the staged draft; replay against the placed
        # row-1 token (the canary)
        is_replay = self.d_scal[S_MODE] == MODE_REPLAY
        cmp_tok = torch.where(is_replay, batch.input_ids[1].to(torch.int64), self.d_scal[S_DRAFT])
        accept = (y1 == cmp_tok).to(torch.int64)
        self.d_payload[0] = y1
        self.d_payload[1] = y2
        self.d_payload[2] = d[1]
        self.d_payload[3] = accept
        self.d_next.copy_(y2.view(1).to(torch.int32))
        # the accepted bonus token lands at position device_len (the scheduler's write
        # tuple targets the same slot; writing it here keeps the graph self-contained)
        engine.spec_token_pool.index_put_(
            (self.d_table, self.d_scal[S_SEQ:S_SEQ + 1]), y2.to(torch.int32).view(1)
        )

    # ------------------------------------------------------------------ capture
    def invalidate(self) -> None:
        """Drop the capture (a cache rebuild reallocated pool addresses baked into the
        graph); the next WARM_STEPS spec steps run eager and recapture."""
        self.graph = None
        self.warm = 0

    def capture_after_step(self, engine: "Engine", model: "Qwen4ExpForCausalLM",
                           batch: "Batch", req, eager_payload: dict) -> bool:
        """Capture using the just-run eager step's context (state restored around the
        warm/capture/replay executions, exactly the sequence the P3 spike proved). The
        final replay must reproduce the eager step's payload or the graph is disabled."""
        table = getattr(model, "_ple_table", None)
        if table is None or not hasattr(table, "fill_spec_rows"):
            self.disabled = True
            logger.warning_rank0("MTP spec graph needs the disk PLE backend; staying eager")
            return False
        self._alloc_static(model)
        self._stage(batch, req)
        pool = engine.linear_state_pool
        try:
            table.fill_spec_rows(req)
            pool.copy_from(req.spec_slot_idx, req.linear_slot_idx)  # back to pre-step
            torch.cuda.synchronize(self.device)
            with engine.ctx.forward_batch(self.batch):
                s = self._stream
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    self._body(engine, model)
                torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize(self.device)
            pool.copy_from(req.spec_slot_idx, req.linear_slot_idx)
            torch.cuda.synchronize(self.device)

            table.fill_spec_rows(req)
            graph = torch.cuda.CUDAGraph()
            with engine.ctx.forward_batch(self.batch):
                with torch.cuda.graph(graph, stream=self._stream):
                    self._body(engine, model)
            torch.cuda.synchronize(self.device)
            pool.copy_from(req.spec_slot_idx, req.linear_slot_idx)
            torch.cuda.synchronize(self.device)

            table.fill_spec_rows(req)
            graph.replay()
            torch.cuda.synchronize(self.device)
            got = [int(v) for v in self.d_payload.tolist()]
            want = [eager_payload["y1"], eager_payload["y2"], eager_payload["draft"],
                    int(bool(eager_payload["accept"]))]
            if got != want:
                raise RuntimeError(f"spec graph replay diverged from eager: {got} != {want}")
            self.graph = graph
            logger.info_rank0("MTP spec graph captured (two-row verify/replay step)")
            return True
        except Exception as e:  # any capture problem -> eager forever (correct, slower)
            import traceback

            self.disabled = True
            self.graph = None
            logger.warning_rank0(
                "MTP spec graph capture failed; staying eager:\n" + traceback.format_exc()
            )
            return False

    # ------------------------------------------------------------------ replay
    def run(self, engine: "Engine", model: "Qwen4ExpForCausalLM",
            batch: "Batch", req):
        import os
        import time as _time

        from freetoken.engine.engine import ForwardOutput

        dbg = os.getenv("FREETOKEN_MTP_DEBUG")
        t0 = _time.perf_counter() if dbg else 0.0
        self._stage(batch, req)
        t1 = _time.perf_counter() if dbg else 0.0
        model._ple_table.fill_spec_rows(req)
        t2 = _time.perf_counter() if dbg else 0.0
        self.graph.replay()
        t3 = _time.perf_counter() if dbg else 0.0
        self.payload_cpu.copy_(self.d_payload, non_blocking=True)
        self.next_cpu.copy_(self.d_next, non_blocking=True)
        event = torch.cuda.Event()
        event.record(engine.stream)
        if dbg:
            self._runs = getattr(self, "_runs", 0) + 1
            if self._runs <= 3 or self._runs % 64 == 0:
                logger.info_rank0(
                    f"[mtp-g] run#{self._runs} stage={(t1-t0)*1e3:.2f}ms "
                    f"fill={(t2-t1)*1e3:.2f}ms replay_issue={(t3-t2)*1e3:.2f}ms"
                )
        return ForwardOutput(self.d_next, self.next_cpu, event, None, self.payload_cpu)
