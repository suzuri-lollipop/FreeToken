"""ERNIE 5.5's DSpark proposer: the bundled model.mtp.* drafter over the HCA target.

Reuses the inkling2 DSparkProposer's context-ring / Markov state machine (the drafter keeps
no paged KV; the target verifies every token) and swaps the geometry: the taps are the
contracted mHC stream at each ``dspark_target_layer_ids`` layer plus the final normed
hidden, the drafter loads from the main checkpoint's model.mtp.* through the family reader,
and the anchor is the target's input embedding. Drafts only -- the bf16 draft roundings
never reach an answer.
"""

from __future__ import annotations

import os
from typing import List, TYPE_CHECKING

import torch

from freetoken.models.ernie5_5_moe.draft import Ernie55DSparkDraft
from freetoken.spec.dspark.proposer import (
    CTX_FULL, CTX_LOCAL, FRNN, DSparkProposer, _full_cap,
)
from freetoken.spec.proposer import Proposer, SpecProposal
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.core import Batch, Req

logger = init_logger(__name__)


class _ErnieVerifySink:
    wanted_layers: frozenset

    def __init__(self, buffer: torch.Tensor, tap_layers: tuple):
        self.buffer = buffer
        self.tap_layers = tap_layers
        self.wanted_layers = frozenset(tap_layers)

    def layer_output(self, layer_id: int, x: torch.Tensor, residual=None, num_streams: int = 1) -> None:
        self.buffer[: x.shape[0], self.tap_layers.index(layer_id)].copy_(x)

    def final_hidden(self, x: torch.Tensor) -> None:
        self.buffer[: x.shape[0], len(self.tap_layers)].copy_(x)


class _ErniePrefillSink:
    wanted_layers: frozenset

    def __init__(self, buffer: torch.Tensor, spans: List[tuple], tap_layers: tuple):
        self.buffer = buffer
        self.spans = spans
        self.tap_layers = tap_layers
        self.wanted_layers = frozenset(tap_layers)

    def layer_output(self, layer_id: int, x: torch.Tensor, residual=None, num_streams: int = 1) -> None:
        n = min(x.shape[0], self.buffer.shape[0])
        self.buffer[:n, self.tap_layers.index(layer_id)].copy_(x[:n])

    def final_hidden(self, x: torch.Tensor) -> None:
        n = min(x.shape[0], self.buffer.shape[0])
        self.buffer[:n, len(self.tap_layers)].copy_(x[:n])


class Ernie55DSparkProposer(DSparkProposer):
    radix_lookahead = False
    tp_supported = False

    @classmethod
    def _args(cls, config):
        return config.model_config.dsv4_args

    @classmethod
    def num_steps(cls, config) -> tuple[int, int]:
        k = cls._args(config).dspark_block_size
        return k, k

    @classmethod
    def adjust_config(cls, config, override) -> None:
        Proposer.adjust_config(config, override)
        from freetoken.models.ernie5_5_moe.args import Ernie55Args
        args = getattr(config.model_config, "dsv4_args", None)
        if not isinstance(args, Ernie55Args):
            raise ValueError("dspark (ernie55) drafts ernie5_5_moe only")
        if not args.dspark_target_layer_ids:
            raise ValueError("dspark (ernie55) needs dspark_target_layer_ids (the tap layers)")
        if config.use_dummy_weight:
            raise ValueError("dspark reads its drafter from model.mtp.*, so it cannot run with --dummy-weight")
        if config.speculative_token_map:
            raise ValueError("--speculative-token-map is an mtp feature; dspark drafts over the whole vocabulary")

    @classmethod
    def slot_states(cls, config):
        from freetoken.models.config import SlotStateSpec
        args = cls._args(config)
        kv_dim = args.dspark_kv_heads * args.dspark_head_dim
        n = args.dspark_num_layers
        full = (n // 2,)
        local = tuple(i for i in range(n) if i not in full)
        # shape's leading 2 is the K/V axis (the layer axis comes from layer_ids)
        return (
            SlotStateSpec(name=CTX_LOCAL, shape=(2, args.dspark_ring, kv_dim), layer_ids=local, dtype=torch.float32),
            SlotStateSpec(name=CTX_FULL, shape=(2, _full_cap(config), kv_dim), layer_ids=full, dtype=torch.float16),
            SlotStateSpec(name=FRNN, shape=(args.dim,), layer_ids=(), dtype=torch.float32),
        )

    @classmethod
    def budget(cls, config) -> tuple[int, int]:
        args = cls._args(config)
        C, V = args.dim, args.vocab_size
        R = args.dspark_markov_rank
        hd, H, KV = args.dspark_head_dim, args.dspark_num_qo_heads, args.dspark_kv_heads
        F = getattr(args, "dspark_intermediate_size", 4 * C)
        per_layer = C * (2 * H * hd + 4 * KV * hd + 2 * F + 4 * C)
        weights = 2 * (args.dspark_num_layers * per_layer + V * R + 3 * R * (2 * R + 2 * C) + C * R + 8 * C)
        fold = 2 * (V * R + V)
        rows = config.max_running_req + 1
        k = config.speculative_num_steps or args.dspark_block_size
        n_taps = len(args.dspark_target_layer_ids) + 1
        verify_taps = rows * (k + 1) * n_taps * C * 4
        prefill_taps = 8192 * n_taps * C * 2
        return weights + fold + verify_taps + prefill_taps, 0

    def __init__(self, config, target_model) -> None:
        super().__init__(config, target_model)
        self.tap_layers = tuple(self._args(config).dspark_target_layer_ids)

    def attach(self, ctx) -> None:
        Proposer.attach(self, ctx)
        mc = self.config.model_config
        args = mc.dsv4_args
        if self.draft is None:
            from freetoken.models.ernie5_5_moe.weight import _ShardReader, _weight_map
            reader = _ShardReader(self.path, _weight_map(self.path), ctx.device)
            try:
                self.draft = Ernie55DSparkDraft.load(reader, args, mc.vocab_size)
            finally:
                reader.close()
            head = self.target.model.head
            self.head_logits = lambda x: head.gather_logits(head.logits(x))
            self.fold_w, self.fold_b = self.draft.fold(self.head_logits)
            self.embed = self.target.model.embed.forward
            logger.info_rank0(f"dspark (ernie55): drafter loaded ({self.draft.weight_bytes() / 2**20:.0f} MiB)")
        rows = self.config.max_running_req + 1
        n_taps = len(self.tap_layers) + 1
        self.verify_taps = torch.zeros(rows * (self.k + 1), n_taps, args.dim, dtype=torch.float32, device=ctx.device)
        self.prefill_taps = None

    def _rings(self, layer: int):
        args = self._args(self.config)
        if self.draft.layers[layer].local:
            local_ids = [i for i, l in enumerate(self.draft.layers) if l.local]
            return self.ctx.slot_states[CTX_LOCAL][local_ids.index(layer)], args.dspark_ring
        return self.ctx.slot_states[CTX_FULL][0], _full_cap(self.config)

    def _read_context(self, slot: int, P: int):
        args = self._args(self.config)
        ks, vs = [], []
        for li in range(len(self.draft.layers)):
            ring, cap = self._rings(li)
            win = args.dspark_window - 1 if self.draft.layers[li].local else cap
            k_lo = max(0, P - win)
            idx = torch.arange(k_lo, P, device=ring.device) % cap
            ks.append(ring[slot, 0].index_select(0, idx).float())
            vs.append(ring[slot, 1].index_select(0, idx).float())
        return ks, vs

    def feature_sink(self, batch):
        if batch.phase == "verify":
            return _ErnieVerifySink(self.verify_taps, self.tap_layers)
        total = sum(e - s for s, e in ((r.cached_len, r.device_len) for r in batch.reqs))
        n_taps = len(self.tap_layers) + 1
        if self.prefill_taps is None or self.prefill_taps.shape[0] < total:
            self.prefill_taps = torch.zeros(total, n_taps, self._args(self.config).dim,
                                            dtype=torch.bfloat16, device=self.device)
        self._prefill_sink = _ErniePrefillSink(self.prefill_taps, [(r.cached_len, r.device_len) for r in batch.reqs], self.tap_layers)
        return self._prefill_sink


__all__ = ["Ernie55DSparkProposer"]