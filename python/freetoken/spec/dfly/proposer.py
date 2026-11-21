"""DFly2 proposer (AngelSpec, arxiv 2607.25852) for hy4-flash.

Port of vLLM PR #68296 ``dfly_speculator.py`` onto freetoken's spec runtime. Unlike
MTP (an autoregressive chain over a PRIVATE draft KV pool), DFly drafts a whole
query block of ``k + 1`` positions in one draft forward whose layers attend the
PAIRED TARGET layer's KV (``Hy4DSABackend.dfly_mla_forward``) with reused indexer
selections, then runs ``k`` autoregressive ``hidden_correction`` + sample steps.

Drafting (vLLM ``HYV4DFlyQueryModel.forward``):
  block ids = [bonus, mask, mask, ...] (query_width = k + 1)
  embeds[::query_width] = mtp_fusion(embeds[::query_width], target_hidden)
  hidden = draft.forward_block(embeds, target_hidden, query_width)   # 5 layers
  for step in range(k):
      corrected = hidden_correction(hidden[:, step], embed(previous))
      previous = sample(corrected) via the gathered target head
      draft[:, step] = previous

The draft KV is written into the target pool at the speculative positions; the
radix cache rolls it back on rejection (no private draft KV chain, so
``radix_lookahead`` is False). The target indexer selections at the paired sparse
layers are snapshotted after each target forward and expanded into the draft
reuse indices by ``build_dfly_reuse_indices``.

Status: core drafting algorithm + registration. The runtime wiring (attach binding
the draft backend to the target pool, feature_sink snapshot of ``md.sel``, graph
capture, slot-state budget) follows the MTP pattern and is validated end-to-end in
Phase 5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

import torch

from freetoken.attention.dfly_reuse import build_dfly_reuse_indices, dfly_out_width
from freetoken.models.hy_v4_flash.dfly import (
    DFLY_MASK_TOKEN_ID,
    DFLY_NUM_LAYERS,
    DFLY_TARGET_LAYERS,
    Hy4DFlyDraft,
    load_dfly_draft,
)
from freetoken.spec.proposer import Proposer, SpecProposal, check_draft_path

if TYPE_CHECKING:
    from freetoken.core import Batch, Req
    from freetoken.models.blocks import FeatureSink
    from freetoken.spec.proposer import DraftContext


class _DFlySink:
    """Captures the target's final hidden rows (the fusion's target_hidden) into a
    persistent buffer a captured verify graph also writes."""

    wanted_layers: frozenset[int] = frozenset()

    def __init__(self, buf: torch.Tensor) -> None:
        self.buf = buf

    def final_hidden(self, x: torch.Tensor) -> None:
        self.buf[: x.shape[0]].copy_(x)

    def layer_output(self, layer_id, x, residual=None, num_streams=1) -> None:
        pass


class _DFlyPrefillSink:
    """Stores every prefill row's final hidden + each request's [cached_len, device_len)
    span, so on_prefill can relay the last prompt hidden into the verify relay."""

    wanted_layers: frozenset[int] = frozenset()

    def __init__(self, spans) -> None:
        self.spans = spans
        self.hidden = None

    def final_hidden(self, x: torch.Tensor) -> None:
        self.hidden = x

    def layer_output(self, layer_id, x, residual=None, num_streams=1) -> None:
        pass


class DFlyProposer(Proposer):
    radix_lookahead = False
    tp_supported = True  # the draft argmax reads gathered full-vocab logits

    @classmethod
    def num_steps(cls, config) -> tuple[int, int]:
        return DFLY_NUM_LAYERS, DFLY_NUM_LAYERS  # query_width = k + 1 = 6

    @classmethod
    def adjust_config(cls, config, override) -> None:
        super().adjust_config(config, override)
        if "dfly" not in dict(config.model_spec.spec_drafts):
            raise ValueError(f"DFly is not supported for {config.hf_config.architectures[0]} yet")
        if config.use_dummy_weight:
            raise ValueError("DFly reads its draft from the checkpoint, so it cannot run with --dummy-weight")
        path = config.speculative_draft_model_path or config.model_path
        if config.speculative_draft_model_path:
            check_draft_path(path)

    @classmethod
    def _draft_class(cls, config):
        return Hy4DFlyDraft

    def __init__(self, config, target_model) -> None:
        super().__init__(config, target_model)
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = config.dtype
        self.query_width = self.k + 1
        path = config.speculative_draft_model_path or config.model_path
        self.draft = load_dfly_draft(config.model_config, target_model, path, self.device, config.dtype)
        self.lm_head = target_model.lm_head
        self.target_layers = DFLY_TARGET_LAYERS
        self.history_width = config.hy4_args.index_topk if hasattr(config, "hy4_args") else 2048
        self.out_width = dfly_out_width(self.history_width, 0, self.query_width)
        self.kv = None
        self.backend = None
        self.verify_hidden = None
        self.snapshots = None
        self._sink = None

    # ------------------------------------------------------------------ lifecycle
    def attach(self, ctx: "DraftContext") -> None:
        from freetoken.attention import create_attention_backend
        from freetoken.core import get_global_ctx

        super().attach(ctx)
        rows = self.config.max_running_req + 1
        # DFly attends the TARGET KV pool: bind the draft backend to it (no private pool)
        self.kv = get_global_ctx().kv_cache
        with get_global_ctx().override(kv_cache=self.kv):
            self.backend = create_attention_backend("dsa", self.draft.config)
        width = self.draft.relay_width(self.config.model_config)
        self.verify_hidden = torch.zeros(rows * self.query_width, width, dtype=self.dtype, device=self.device)
        # separate relay (per table_idx): the verify graph overwrites verify_hidden's
        # row region every step, so the last-accepted hidden must live apart from it
        self.relay_hidden = torch.zeros(rows, width, dtype=self.dtype, device=self.device)
        # one snapshot slot per draft layer of the target indexer's selected rows
        self.snapshots = torch.full((DFLY_NUM_LAYERS, rows, self.out_width), -1, dtype=torch.int32, device=self.device)
        self._sink = _DFlySink(self.verify_hidden)
        self._prefill_sink = None

    def detach(self, *, keep_kv: bool) -> None:
        self.backend = None
        self.verify_hidden = None
        self.relay_hidden = None
        self.snapshots = None
        self._sink = None
        self.kv = None  # the target pool; never owned by the draft
        super().detach(keep_kv=keep_kv)

    # ------------------------------------------------------------------ drafting
    def _sample(self, step: int, corrected: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head.gather_logits(self.lm_head.logits(self.draft.head_input(corrected)))
        return logits.argmax(dim=-1)

    def feature_sink(self, batch: "Batch") -> "FeatureSink":
        if getattr(batch, "phase", None) == "verify":
            return self._sink
        self._prefill_sink = _DFlyPrefillSink([(r.cached_len, r.device_len) for r in batch.reqs])
        return self._prefill_sink

    def _snapshot_target_selections(self, batch: "Batch") -> None:
        """Copy the target indexer's selected rows (md.sel) for the paired sparse
        layers into the reuse snapshots, one slot per draft layer."""
        md = batch.attn_metadata
        reqs = batch.padded_reqs if hasattr(batch, "padded_reqs") else batch.reqs
        table = torch.tensor([r.table_idx for r in reqs], dtype=torch.int64, device=self.device)
        for slot, target_layer in enumerate(self.target_layers):
            pos = md.sel_pos.get(target_layer)
            if pos is None:
                continue
            # positions [bs, m, width]; keep the first query row's selection per request,
            # stored by table_idx (the propose reads it back by table_idx, not batch order)
            s = pos[:, 0] if pos.dim() == 3 else pos
            w = min(s.shape[-1], self.history_width)
            self.snapshots[slot, table[: s.shape[0]], :w] = s[:, :w]

    def propose(self, reqs: List["Req"], rows: torch.Tensor, pending: torch.Tensor) -> SpecProposal:
        from freetoken.core import get_global_ctx

        bs = len(reqs)
        anchors = torch.tensor([r.cached_len for r in reqs], dtype=torch.int64, device=self.device)
        active = torch.ones(bs, dtype=torch.bool, device=self.device)
        bonus = pending[:bs].long()
        # query block ids: bonus at position 0, mask tokens after
        ids = torch.full((bs, self.query_width), DFLY_MASK_TOKEN_ID, dtype=torch.int64, device=self.device)
        ids[:, 0] = bonus
        positions = anchors[:, None] + torch.arange(self.query_width, device=self.device)[None, :]
        target_hidden = self.relay_hidden[rows]  # relayed last accepted hidden (per table_idx)
        embeds = self.draft._embed_tokens.forward(ids.view(-1))
        # reuse indices from the snapshotted target selections + the draft query block
        source_rows = rows
        layer_slots = torch.arange(DFLY_NUM_LAYERS, dtype=torch.int32, device=self.device)
        sel_buf = torch.full((DFLY_NUM_LAYERS, bs * self.query_width, self.out_width), -1, dtype=torch.int32, device=self.device)
        build_dfly_reuse_indices(sel_buf, self.snapshots, layer_slots, active, source_rows, anchors,
                                 self.history_width, 0, self.query_width)
        # sel_buf holds token POSITIONS (history snapshot + causal draft block); _attend
        # reads physical KV rows, so map every position through the request's page-table
        # row (the causal block lands on the draft KV slots dfly_mla_forward writes).
        req_of_row = torch.arange(bs * self.query_width, device=self.device) // self.query_width
        pt_rows = get_global_ctx().page_table[rows[req_of_row]]          # [bs*m, W]
        for layer in range(DFLY_NUM_LAYERS):
            p = sel_buf[layer]
            phys = pt_rows.gather(1, p.clamp_min(0).long())
            sel_buf[layer] = torch.where(p >= 0, phys, p.new_full((), -1)).to(torch.int32)
        batch = self._draft_batch(reqs, ids.view(-1), positions.view(-1), sel_buf, bs)
        with get_global_ctx().forward_batch(batch, kv_cache=self.kv, attn_backend=self.backend, hide_target_state=True):
            hidden = self.draft.forward_block(embeds, target_hidden, self.query_width).view(bs, self.query_width, -1)
            drafts = self.draft.draft_tokens(hidden, bonus, self._sample, active, self.query_width)
        return SpecProposal(drafts)

    def _draft_batch(self, reqs, input_ids, positions, sel_buf, bs):
        from freetoken.core import Batch

        views = [_DraftView(r.cached_len, r.cached_len + self.query_width, r.table_idx) for r in reqs]
        batch = Batch(reqs=views, phase="decode", rows_per_req=self.query_width)
        batch.padded_reqs = views
        batch.input_ids = input_ids
        batch.positions = positions
        rows = torch.cat([torch.full((self.query_width,), r.table_idx, dtype=torch.int64) for r in reqs]).to(self.device)
        batch.out_loc = self.ctx.page_table[rows, positions.to(torch.int64)]
        self.backend.prepare_metadata(batch)
        md = batch.attn_metadata
        for layer in range(DFLY_NUM_LAYERS):
            md.dfly_sel[layer] = (sel_buf[layer].view(bs, self.query_width, self.out_width),
                                  torch.full((bs, self.query_width), self.out_width, dtype=torch.int32, device=self.device))
        return batch

    def post_verify(self, batch: "Batch", out: torch.Tensor, emitted: torch.Tensor) -> None:
        # relay each request's last accepted target hidden + bonus token for the next propose
        self._snapshot_target_selections(batch)
        bs = batch.size
        k1 = self.query_width
        picked_rows = torch.arange(bs, device=self.device) * k1 + emitted[:bs]
        table = torch.tensor([r.table_idx for r in batch.reqs], dtype=torch.int64, device=self.device)
        self.relay_hidden[table] = self.verify_hidden[picked_rows]
        self._bonus = out[:bs, -1] if out.dim() == 2 else out[:bs]

    def on_prefill(self, batch: "Batch", next_tokens_gpu: torch.Tensor) -> None:
        # relay the target's last prompt hidden into the verify relay slot so the
        # first propose's fusion reads it as target_hidden (DFly drafts no prefill chunk).
        sink, self._prefill_sink = getattr(self, "_prefill_sink", None), None
        if sink is not None and sink.hidden is not None:
            hidden = sink.hidden
            offset = 0
            for req, (s, e) in zip(batch.reqs, sink.spans):
                rows = hidden[offset : offset + e - s]
                offset += e - s
                if e == req.prompt_len and rows.shape[0] > 0:
                    self.relay_hidden[req.table_idx] = rows[-1]
        self._snapshot_target_selections(batch)


class _DraftView:
    def __init__(self, cached_len: int, device_len: int, table_idx: int) -> None:
        self.cached_len = cached_len
        self.device_len = device_len
        self.extend_len = device_len - cached_len
        self.table_idx = table_idx
        self.mrope_delta = 0
        self.mrope_positions_full = None


__all__ = ["DFlyProposer"]
