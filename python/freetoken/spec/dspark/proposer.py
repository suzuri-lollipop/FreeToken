"""DSparkProposer: Inkling-2-Mini's official drafter (mtp.safetensors), the Maya v1.1.73 port.

One step drafts a block of 7 after the pending token: the target's taps of the committed
positions (the streams at the start of visits 5 / 23 / 41 and the final normed hidden,
hand-ed over by the model's FeatureSink) mix into the drafter's per-request context caches;
the anchor embedded as the target embeds plus 7 mask rows run the 3 dense draft layers, and
the Markov head reads the block's hiddens one token at a time. The target verifies the block
as one window (the engine's reject / commit), so drafting never changes an answer.

The context caches and f_rnn ride the linear-state slots (Proposer.slot_states): the pool
resets them with the conv state on a new request and copies them on a rebuild. The conv
history's own verify handler (Inkling2ConvVerify) rolls the target's convs to the accepted
rows; this proposer writes the drafter's context for exactly those rows in post_verify.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List

import torch

from freetoken.attention.linear import gdn_slot
from freetoken.models.config import SlotStateSpec
from freetoken.models.inkling2.draft import BLOCK, N_TAPS, RING, TAP_VISITS, WINDOW, DSparkDraft
from freetoken.spec.proposer import DraftContext, Proposer, SpecProposal, check_draft_path
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.core import Batch, Req
    from freetoken.models.config import ModelConfig

logger = init_logger(__name__)

CTX_LOCAL = "ik2_draft_ctx_local"
CTX_FULL = "ik2_draft_ctx_full"
FRNN = "ik2_draft_frnn"
# the full-context draft layer's kept positions (Maya's IK2_DRAFT_CTX); the windowed layers
# always keep Maya's 2048-row ring. Smaller than the context costs draft quality on long
# conversations only -- the verify keeps every answer exact.
DEFAULT_FULL_CTX = 8192


def _full_cap(config) -> int:
    want = int(os.environ.get("FREETOKEN_IK2_DRAFT_CTX", DEFAULT_FULL_CTX))
    return min(config.max_seq_len, max(want, RING))


class _VerifySink:
    """The verify rows' taps into the proposer's persistent buffer (a captured graph writes it too)."""

    wanted_layers = frozenset(TAP_VISITS)

    def __init__(self, buffer: torch.Tensor) -> None:
        self.buffer = buffer

    def layer_output(self, layer_id: int, x: torch.Tensor, residual: torch.Tensor | None = None, num_streams: int = 1) -> None:
        self.buffer[: x.shape[0], TAP_VISITS.index(layer_id)].copy_(x)

    def final_hidden(self, x: torch.Tensor) -> None:
        self.buffer[: x.shape[0], N_TAPS - 1].copy_(x)


class _PrefillSink:
    """Every prefill row's taps (bf16: a chunk's rows are many) and the request spans."""

    wanted_layers = frozenset(TAP_VISITS)

    def __init__(self, buffer: torch.Tensor, spans: List[tuple[int, int]]) -> None:
        self.buffer = buffer
        self.spans = spans

    def layer_output(self, layer_id: int, x: torch.Tensor, residual: torch.Tensor | None = None, num_streams: int = 1) -> None:
        n = min(x.shape[0], self.buffer.shape[0])
        self.buffer[:n, TAP_VISITS.index(layer_id)].copy_(x[:n])

    def final_hidden(self, x: torch.Tensor) -> None:
        n = min(x.shape[0], self.buffer.shape[0])
        self.buffer[:n, N_TAPS - 1].copy_(x[:n])


class DSparkProposer(Proposer):
    # the target's KV at position i reads tokens <= i only (the drafter keeps no paged KV),
    # so the radix keys need no lookahead; the draft argmax is tp 1 for now (the dense draft
    # weights are replicated, and a cross-rank agreement is not built yet)
    radix_lookahead = False
    tp_supported = False

    @classmethod
    def num_steps(cls, config) -> tuple[int, int]:
        return BLOCK - 1, BLOCK - 1

    @classmethod
    def adjust_config(cls, config, override) -> None:
        super().adjust_config(config, override)
        mc = config.model_config
        args = getattr(mc, "inkling2_args", None)
        if args is None:
            raise ValueError("dspark drafts Inkling-2-Mini (inkling2_loop) only")
        if args.hidden_size != 2304 or args.num_visits != 48:
            raise ValueError("dspark needs the Inkling-2-Mini geometry (width 2304, 48 visits)")
        if config.use_dummy_weight:
            raise ValueError("dspark reads its drafter from mtp.safetensors, so it cannot run with --dummy-weight")
        if config.speculative_token_map:
            raise ValueError("--speculative-token-map is an mtp feature; dspark drafts over the whole vocabulary")
        path = config.speculative_draft_model_path or config.model_path
        if not config.speculative_draft_model_path and (str(path).endswith(".gguf") or _is_ftw(path)):
            raise ValueError(
                "a GGUF or FTW target carries no mtp.safetensors: pass --speculative-draft-model-path "
                "pointing at the official drafter file (or the HF folder holding it)"
            )
        if config.speculative_draft_model_path:
            check_draft_path(path)
        DSparkDraft.check_source(path)

    @classmethod
    def slot_states(cls, config) -> tuple[SlotStateSpec, ...]:
        args = config.model_config.inkling2_args
        kv_dim = args.kv_dim
        # Mini's draft layers 0 and 2 windowed (F32 rings), layer 1 full-context (FP16)
        return (
            SlotStateSpec(name=CTX_LOCAL, shape=(2, RING, kv_dim), layer_ids=(0, 2), dtype=torch.float32),
            SlotStateSpec(name=CTX_FULL, shape=(2, _full_cap(config), kv_dim), layer_ids=(1,), dtype=torch.float16),
            SlotStateSpec(name=FRNN, shape=(args.hidden_size,), layer_ids=(), dtype=torch.float32),
        )

    @classmethod
    def budget(cls, config) -> tuple[int, int]:
        mc = config.model_config
        args = mc.inkling2_args
        C, V = args.hidden_size, args.padded_vocab_size
        R = 256  # the Markov latent width (rnn_proj's 3R rows)
        # the drafter's BF16 weights: 3 dense layers + the Markov head (mtp.safetensors ~0.6 GiB)
        per_layer = C * (2 * args.num_qo_heads * 128 + 4 * args.kv_dim + 2 * 4 * C + 4 * C)
        weights = 2 * (3 * per_layer + V * R + 3 * R * (2 * R + 2 * C) + C * R + 8 * C)
        fold = 2 * (V * R + V)  # the folded Markov bias, FP16
        rows = config.max_running_req + 1
        k = config.speculative_num_steps or BLOCK - 1
        verify_taps = rows * (k + 1) * N_TAPS * C * 4
        # the prefill tap buffer grows to the largest chunk seen; 8192 is the launcher's default
        # chunk, the plan's estimate (bf16 rows: half the F32 verify taps' bytes)
        prefill_taps = 8192 * N_TAPS * C * 2
        return weights + fold + verify_taps + prefill_taps, 0

    def __init__(self, config, target_model) -> None:
        super().__init__(config, target_model)
        self.target = target_model
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.path = config.speculative_draft_model_path or config.model_path
        self.draft: DSparkDraft | None = None
        self.fold_w = self.fold_b = None
        self.verify_taps = None
        self.prefill_taps = None
        self._prefill_sink: _PrefillSink | None = None

    # ------------------------------------------------------------------ lifecycle

    def attach(self, ctx: DraftContext) -> None:
        super().attach(ctx)
        mc = self.config.model_config
        if self.draft is None:
            self.draft = DSparkDraft.load(self.path, ctx.device, mc.vocab_size)
            head = self.target.lm_head
            self.head_logits = lambda x: head.gather_logits(head.logits(x))
            self.fold_w, self.fold_b = self.draft.fold(self.head_logits)
            # _NormedEmbedding is a plain op (no __call__); the proposer invokes it directly
            self.embed = self.target.model._normed_embedding.forward
            logger.info_rank0(
                f"dspark: the official drafter loaded ({self.draft.weight_bytes() / 2**20:.0f} MiB), "
                f"full-context layer keeps {_full_cap(self.config)} positions"
            )
        args = mc.inkling2_args
        rows = self.config.max_running_req + 1
        self.verify_taps = torch.zeros(rows * (self.k + 1), N_TAPS, args.hidden_size, dtype=torch.float32, device=ctx.device)
        self.prefill_taps = None

    def detach(self, *, keep_kv: bool) -> None:
        self.verify_taps = None
        self.prefill_taps = None
        self._prefill_sink = None
        super().detach(keep_kv=keep_kv)

    # ------------------------------------------------------------------ context state

    def _rings(self, layer: int) -> tuple[torch.Tensor, int]:
        """(the layer's ring tensor [num_slots, 2 (K/V), cap, kv_dim], its cap)."""
        if self.draft.layers[layer].local:
            return self.ctx.slot_states[CTX_LOCAL][0 if layer == 0 else 1], RING
        return self.ctx.slot_states[CTX_FULL][0], _full_cap(self.config)

    def _write_context(self, slot: int, taps: torch.Tensor, pos0: int, m: int) -> None:
        """The drafter's context rows of positions [pos0, pos0+m) from their [m, taps, C] taps."""
        positions = torch.arange(pos0, pos0 + m, device=taps.device)
        for li in range(len(self.draft.layers)):
            mixed = self.draft.mix_taps(taps, li)
            k, v = self.draft.context_rows(mixed, li, positions)
            ring, cap = self._rings(li)
            k, v = k.to(ring.dtype), v.to(ring.dtype)
            # wrap-free segments (Maya's write_context chunks): a span longer than the ring
            # revisits indices, and index_copy_ gives no order guarantee on duplicates
            start = 0
            while start < m:
                at = (pos0 + start) % cap
                n = min(m - start, cap - at)
                idx = torch.arange(at, at + n, device=taps.device)
                ring[slot, 0].index_copy_(0, idx, k[start : start + n])
                ring[slot, 1].index_copy_(0, idx, v[start : start + n])
                start += n

    def _read_context(self, slot: int, P: int) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        """The window rows each draft layer attends at anchor position P (Maya's k_lo)."""
        ks, vs = [], []
        for li in range(len(self.draft.layers)):
            ring, cap = self._rings(li)
            win = WINDOW - 1 if self.draft.layers[li].local else cap
            k_lo = max(0, P - win)
            idx = torch.arange(k_lo, P, device=ring.device) % cap
            ks.append(ring[slot, 0].index_select(0, idx).float())
            vs.append(ring[slot, 1].index_select(0, idx).float())
        return ks, vs

    # ------------------------------------------------------------------ one step

    def feature_sink(self, batch: Batch):
        if batch.phase == "verify":
            return _VerifySink(self.verify_taps)
        total = sum(e - s for s, e in ((r.cached_len, r.device_len) for r in batch.reqs))
        if self.prefill_taps is None or self.prefill_taps.shape[0] < total:
            self.prefill_taps = torch.zeros(total, N_TAPS, self.config.model_config.hidden_size,
                                            dtype=torch.bfloat16, device=self.device)
        self._prefill_sink = _PrefillSink(self.prefill_taps, [(r.cached_len, r.device_len) for r in batch.reqs])
        return self._prefill_sink

    def on_prefill(self, batch: Batch, next_tokens_gpu: torch.Tensor) -> None:
        """Write the chunk's context rows and set f_rnn at each request's last committed
        position: a prompt chunk's end (the next chunk overwrites it), or a ride-along decode
        row of a mixed batch (no verify upgrade there, the row commits as a plain decode)."""
        sink, self._prefill_sink = self._prefill_sink, None
        taps = sink.buffer
        offset = 0
        frnn = self.ctx.slot_states[FRNN]
        for i, (req, (s, e)) in enumerate(zip(batch.reqs, sink.spans)):
            m = e - s
            rows = taps[offset : offset + m]
            offset += m
            slot = gdn_slot(req)
            self._write_context(slot, rows, s, m)
            frnn[slot] = self.draft.f_rnn(rows[-1].float())

    def propose(self, reqs: List[Req], rows: torch.Tensor, pending: torch.Tensor) -> SpecProposal:
        drafts = []
        frnn = self.ctx.slot_states[FRNN]
        for i, req in enumerate(reqs):
            P = req.cached_len
            slot = gdn_slot(req)
            anchor_h = self.embed(pending[i : i + 1])[0].float()
            ks, vs = self._read_context(slot, P)
            drafts.append(self.draft.propose(
                pending[i], anchor_h, P, ks, vs, frnn[slot], self.head_logits, self.fold_w, self.fold_b, self.k))
        return SpecProposal(torch.stack(drafts, dim=0))

    def post_verify(self, batch: Batch, out: torch.Tensor, emitted: torch.Tensor) -> None:
        """The kept positions are the anchor + the accepted drafts (emitted+1 rows): their taps
        came from the verify forward; write their context and set f_rnn at the last kept row."""
        k1 = self.k + 1
        counts = (emitted[: batch.size] + 1).tolist()
        frnn = self.ctx.slot_states[FRNN]
        for i, req in enumerate(batch.reqs):
            n = counts[i]
            taps = self.verify_taps[i * k1 : i * k1 + n]
            slot = gdn_slot(req)
            self._write_context(slot, taps, req.cached_len, n)
            frnn[slot] = self.draft.f_rnn(taps[n - 1])


def _is_ftw(path) -> bool:
    from freetoken.checkpoint.ftw import is_ftw_checkpoint

    try:
        return is_ftw_checkpoint(path)
    except (OSError, ValueError):
        return False


__all__ = ["DSparkProposer"]
