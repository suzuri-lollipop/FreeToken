"""Qwen3.8-Flash-Next decoder stack (text-only).

The residual state is ``R [T, hc_count*hidden]`` end to end: the embedding is repeated over the
``hc_count`` streams, every layer mixes them down to one ``[T, hidden]`` block input and injects
its output back, and the top-level mixer collapses them once before ``lm_head``. There is no
input/post layernorm and no final ``model.norm`` -- the hyper-connection norms are the only ones.

Layer contract (frozen): ``forward(R [T, hc*hidden], batch) -> R' [T, hc*hidden]`` with an
immediate combine::

    R  = R + ple(R, batch)                 # zero-based layer 1 only
    x, s = attn_hc.mix(R); y = (GDN | QSA)(x); R = attn_hc.combine(R, y, s)
    x, s = mlp_hc.mix(R);  y = MoE(x);        R = mlp_hc.combine(R, y, s)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import BaseOP, OPList, ParallelLMHead, VocabParallelEmbedding
from freetoken.models.blocks import BaseLLMModel
from freetoken.utils import nvtx_annotate

from .attention import Qwen4ExpAttention
from .hc import GatedResidual
from .moe import Qwen4ExpMoE
from .mtp import Qwen4ExpMTPHead
from .ple import PLELayer
from freetoken.models.blocks import embed_input_ids
from freetoken.models.qwen3_vl.vision import Qwen3VLVisionModel, QwenVLVisionMixin

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig


def build_linear_mixer(config: ModelConfig, layer_id: int, prefix: str) -> BaseOP:
    """GDN mixer of a linear_attention layer (Qwen3.5's GDN with a configurable output gate)."""
    from .gdn import Qwen4ExpGatedDeltaNet

    g = config.linear_attention_group()
    return Qwen4ExpGatedDeltaNet(
        hidden_size=config.hidden_size,
        num_k_heads=g.num_key_heads,
        num_v_heads=g.num_value_heads,
        head_k_dim=g.key_head_dim,
        head_v_dim=g.value_head_dim,
        conv_kernel_size=g.conv_kernel_dim,
        rms_norm_eps=config.rms_norm_eps,
        layer_id=layer_id,
        output_gate=g.output_gate,
        quant_config=config.quant,
        prefix=prefix,
    )


class Qwen4ExpDecoderLayer(BaseOP):
    """One decoder layer over the hyper-connection streams (see the module docstring for the flow)."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = "") -> None:
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            self.linear_attn = build_linear_mixer(config, layer_id, f"{prefix}.linear_attn")
        else:
            self.self_attn = Qwen4ExpAttention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp = Qwen4ExpMoE(config, layer_id, prefix=f"{prefix}.mlp")
        self.attn_hyper_connection = GatedResidual(config, prefix=f"{prefix}.attn_hyper_connection")
        self.mlp_hyper_connection = GatedResidual(config, prefix=f"{prefix}.mlp_hyper_connection")
        self.ple = (
            PLELayer(config, layer_id, prefix=f"{prefix}.ple") if layer_id in config.qwen4_args.ple_layer_ids else None
        )

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        if self.ple is not None:
            hidden = hidden + self.ple.forward(hidden, batch)
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        if self._is_linear:
            block_output = self.linear_attn.forward(block_input)
        else:
            block_output = self.self_attn.forward(block_input, batch)
        # Fused combine + MLP norm: eliminates one write+read of the full residual
        hidden, mlp_rn = self.attn_hyper_connection.combine_norm(
            hidden, block_output, inject,
            self.mlp_hyper_connection.hc_norm.weight,
            self.mlp_hyper_connection.hc_norm.eps,
        )
        block_input, inject = self.mlp_hyper_connection.mix_from_normed(mlp_rn)
        return self.mlp_hyper_connection.combine(hidden, self.mlp.forward(block_input), inject)


class Qwen4ExpModel(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model") -> None:
        self.hc_count = config.qwen4_args.hc_count
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                Qwen4ExpDecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.hyper_connection_mixer = GatedResidual(config, use_combine=False, prefix=f"{prefix}.hyper_connection_mixer")
        # plain tuple (not an OP child), so it never shows up in the state dict
        self._ple = tuple(layer.ple for layer in self.layers.op_list if layer.ple is not None)

    @property
    def ple_layers(self) -> List[PLELayer]:
        """The PLE layers in decoder order -- the seam the loader attaches table backends to."""
        return list(self._ple)

    def forward_with_residual(self, input_ids: torch.Tensor, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """``forward`` plus the raw hyper-connection residual: ``(mixed [T,H], R [T,HC*H])``.

        The MTP spec path consumes R (the head's fc_hidden input) alongside the logits,
        so the top mixer's INPUT has to escape the model too.
        """
        hidden = embed_input_ids(self.embed_tokens, input_ids, batch)
        hidden = hidden.repeat(1, self.hc_count)
        meta = None
        if self._ple:
            from .ple import build_ple_metadata, commit_ngram_context

            meta = build_ple_metadata(batch, self._ple[0].args, input_ids.device)
            for ple in self._ple:  # gather the pinned-host PLE rows while the early layers run
                ple.start_prefetch(batch, meta)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden, batch)
        if meta is not None:
            # single writer: the layers only read the context, so a second PLE layer's
            # prefetch sees the un-rolled window
            commit_ngram_context(meta, getattr(batch, "fla_metadata", None))
        return self.hyper_connection_mixer.mix(hidden)[0], hidden

    def forward(self, input_ids: torch.Tensor, batch: Batch) -> torch.Tensor:
        return self.forward_with_residual(input_ids, batch)[0]

    def forward_dual(self, batchA: Batch, batchB: Batch, side: torch.cuda.Stream) -> torch.Tensor:
        """Dual-microbatch decode: run the two half-batches on skewed streams.

        Per layer the leading half issues first (current stream), then the trailing
        half (side stream); the MoE cross-waits (layers/moe.py _decode_dual_moe) make
        one half's PCIe miss fetch overlap the other half's dense/attention SM work.
        Each half runs with its own all-reduce instance, QSA staging profile and MoE
        fetch plan (batch.dual_slot), so the streams share no mutable state besides
        the slot cache's LRU bookkeeping, which the per-layer events serialize.
        Returns the merged mixed hidden [TA+TB, H] in batch row order."""
        import contextlib

        from freetoken.distributed import ar_instance
        from freetoken.kernel.triton.w8a16_linear import restore_ws_slot, set_ws_slot

        from .ple import build_ple_metadata, commit_ngram_context

        ctx = get_global_ctx()
        profile = getattr(getattr(ctx, "attn_backend", None), "dual_profile", None)
        main = torch.cuda.current_stream()
        fork = torch.cuda.Event()
        fork.record(main)
        side.wait_event(fork)

        def half_stack(batch: Batch, slot: int, on_side: bool) -> contextlib.ExitStack:
            st = contextlib.ExitStack()
            st.enter_context(ctx.swap_batch(batch))
            # resource slot: batch.dual_slot (the DUAL_SLOT0 probe collapses both
            # halves onto instance 0); the QSA staging profile always follows the
            # true slot so the halves' static buffers stay disjoint.
            rslot = getattr(batch, "dual_slot", slot)
            st.enter_context(ar_instance(rslot if rslot >= 0 else slot))
            # per-half split-K workspace partition (the ws address bakes into the
            # graph; the halves run concurrently and must not alias)
            prev_ws = set_ws_slot(rslot if rslot >= 0 else slot)
            st.callback(restore_ws_slot, prev_ws)
            if profile is not None:
                st.enter_context(profile(slot))
            if on_side:
                st.enter_context(torch.cuda.stream(side))
            return st

        def embed_half(batch: Batch, slot: int, on_side: bool):
            with half_stack(batch, slot, on_side):
                hidden = embed_input_ids(self.embed_tokens, batch.input_ids, batch)
                hidden = hidden.repeat(1, self.hc_count)
                if self._ple:
                    meta = build_ple_metadata(
                        batch, self._ple[0].args, batch.input_ids.device
                    )
                    # PLELayer.forward picks this up (the start_prefetch pair would
                    # race on the single _pending slot across the two halves).
                    batch.ple_meta_pre = meta
                    return hidden, meta
                return hidden, None

        # FREETOKEN_DUAL_SERIAL=1: fully serialize the halves (correctness probe --
        # isolates overlap-mechanics races from half-view plumbing bugs). Every
        # phase boundary gets an event pair: the capture-pool allocator reuses a
        # freed block on the OTHER stream with no execution ordering, so any
        # concurrent phase could write a block the peer's kernels still read.
        serial = os.getenv("FREETOKEN_DUAL_SERIAL", "0") == "1"

        def sync_to_side():
            ev = torch.cuda.Event()
            ev.record(main)
            side.wait_event(ev)

        def sync_to_main():
            ev = torch.cuda.Event()
            ev.record(side)
            main.wait_event(ev)

        hiddenA, metaA = embed_half(batchA, 0, False)
        if serial:
            sync_to_side()
        hiddenB, metaB = embed_half(batchB, 1, True)
        if serial:
            sync_to_main()
        for layer in self.layers.op_list:
            with half_stack(batchA, 0, False):
                hiddenA = layer.forward(hiddenA, batchA)
            if serial:
                sync_to_side()
            with half_stack(batchB, 1, True):
                hiddenB = layer.forward(hiddenB, batchB)
            if serial:
                sync_to_main()
        if self._ple:
            if serial:
                sync_to_side()
            with half_stack(batchA, 0, False):
                commit_ngram_context(metaA, getattr(batchA, "fla_metadata", None))
            if serial:
                sync_to_main()
                sync_to_side()
            with half_stack(batchB, 1, True):
                commit_ngram_context(metaB, getattr(batchB, "fla_metadata", None))
            if serial:
                sync_to_main()
        with half_stack(batchA, 0, False):
            mixedA = self.hyper_connection_mixer.mix(hiddenA)[0]
        if serial:
            sync_to_side()
        with half_stack(batchB, 1, True):
            mixedB = self.hyper_connection_mixer.mix(hiddenB)[0]
            join = torch.cuda.Event()
            join.record(side)
        main.wait_event(join)
        return torch.cat([mixedA, mixedB], dim=0)


class Qwen4ExpForCausalLM(BaseLLMModel):
    # the graph runner dual-captures the bs4 decode graph through model.forward_dual
    supports_dual_decode = True
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self.model = Qwen4ExpModel(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        # Speculative MTP head (--speculative mtp): attached under `mtp` so the strict
        # weight load lines up with the checkpoint's mtp.* names. Nothing in the serving
        # path calls it until the spec step lands (Phase 1 of _scratch/mtp_design.md);
        # the wiring variant is the probe knob FREETOKEN_MTP_WIRING.
        self.mtp = (
            Qwen4ExpMTPHead(
                config,
                layer_id=config.num_layers,
                wiring=os.getenv("FREETOKEN_MTP_WIRING", "norm_mixfrom_fc"),
            )
            if config.qwen4_args.mtp_enabled
            else None
        )
        super().__init__()

    def load_host_tables(self, engine_config) -> int:
        """Attach the PLE n-gram table (pinned checkpoint bank, or zeros for dummy weights); returns the pinned host bytes the engine reserves from its pin budget."""
        ple_layers = self.model.ple_layers
        if not ple_layers:
            return 0
        from .ple import PinnedUVATable, ZeroTable, derive_ngram_hash_constants

        if getattr(engine_config, "use_dummy_weight", False):
            # Dummy fill leaves the int64 hash buffers garbage (a zero vocab size divides by
            # zero in the hash), so re-derive the real constants and read a zero table.
            for ple in ple_layers:
                args = ple.args
                mult, sizes, offsets = derive_ngram_hash_constants(
                    vocab_size=self._config.vocab_size,
                    ngram_size=args.ngram_size,
                    num_ngram_heads=args.num_ngram_heads,
                    ngram_vocab_size_base=args.ngram_vocab_size_base,
                    ple_layer_index=ple.ple_index,
                )
                emb = ple.ple_embedding
                emb.layer_multipliers.copy_(torch.tensor(mult, dtype=torch.int64))
                emb.ngram_heads_vocab_sizes.copy_(torch.tensor(sizes, dtype=torch.int64))
                emb.ngram_heads_offsets.copy_(torch.tensor(offsets, dtype=torch.int64))
                emb.attach_table(ZeroTable(offsets[-1] + sizes[-1], args.ngram_head_dim))
            return 0

        if engine_config.ple_backend == "disk":
            from freetoken.utils import download_hf_weight

            from .ple_disk import DiskRowTable, resolve_row_source

            folder = download_hf_weight(engine_config.model_path)
            # one WAIT node per captured graph: the flag protocol supports a single consume
            assert len(ple_layers) == 1, "disk PLE backend expects exactly one PLE layer"
            emb, args = ple_layers[0].ple_embedding, ple_layers[0].args
            # hash with the state-dict-loaded constants, the same source the pinned path reads
            constants = {
                "num_ngram_heads": args.num_ngram_heads,
                "layer_multipliers": emb.layer_multipliers.tolist(),
                "per_head_vocab_sizes": emb.ngram_heads_vocab_sizes.tolist(),
                "per_head_offsets": emb.ngram_heads_offsets.tolist(),
                "eos_token_id": args.ngram_boundary_token_id,
                "image_token_id": args.image_token_id,
            }
            disk_table = DiskRowTable(
                resolve_row_source(folder),
                constants,
                max_graph_rows=max(256, engine_config.cuda_graph_max_bs or 0),
                max_extend_tokens=engine_config.max_extend_tokens,
            )
            self._ple_table = disk_table
            for ple in ple_layers:
                ple.ple_embedding.attach_table(disk_table)
            # engine enters this around every dispatch; the graph itself never waits on the disk
            self.forward_host_ctx = disk_table.forward_host_ctx
            return 0

        from .weight import load_ple_table

        table = load_ple_table(engine_config.model_path, self._config.qwen4_args)
        self._ple_table = table  # owns the pinned HostBank; keep it alive
        for ple in ple_layers:
            ple.ple_embedding.attach_table(
                PinnedUVATable(table.bank.tensor, float(table.weight_scale))
            )
        return table.bank.nbytes

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        dual = getattr(self, "_dual_halves", None)
        if dual is not None:
            # dual-microbatch decode capture/warm: the halves carry their slices of
            # the static graph buffers; the head pass stays on the merged rows with
            # the FULL batch as ctx (same row selection as the single-stream graph).
            return self.lm_head.forward(
                self.model.forward_dual(dual[0], dual[1], self._dual_stream)
            )
        return self.lm_head.forward(self.model.forward(batch.input_ids, batch))

    def forward_with_residual_ctx(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``forward()`` over the ctx batch, plus the raw residual (the engine's MTP stash)."""
        batch = get_global_ctx().batch
        mixed, residual = self.model.forward_with_residual(batch.input_ids, batch)
        return self.lm_head.forward(mixed), residual

    def full_vocab_logits(self, mixed: torch.Tensor) -> torch.Tensor:
        """Full ``[T, vocab]`` logits from mixed hidden rows.

        The spec path needs EVERY row (ParallelLMHead.forward's row selection is
        ctx-driven: per-request last row), so this is the direct shard matmul plus the
        head's own vocab gather (rank-major order, padding trimmed) -- the same contract
        ParallelLMHead.forward implements for its selected rows.
        """
        head = self.lm_head
        weight = head.tied_embedding.weight if head.tied_embedding is not None else head.weight
        logits = torch.nn.functional.linear(mixed, weight, head.bias)
        if head.tp_size == 1:
            return logits
        shape = logits.shape
        gathered = head._comm.all_gather(logits)
        gathered = gathered.view((head.tp_size,) + shape).permute(1, 0, 2).contiguous()
        return gathered.reshape(shape[0], head.tp_size * shape[1])[:, : head.num_embeddings]

    def greedy_ids(self, mixed: torch.Tensor) -> torch.Tensor:
        """Select every row's global argmax with one candidate per vocab shard."""
        head = self.lm_head
        weight = head.tied_embedding.weight if head.tied_embedding is not None else head.weight
        logits = torch.nn.functional.linear(mixed, weight, head.bias)
        if head.tp_size == 1:
            return logits.argmax(-1)
        start, count = head.vocab_range
        logits[:, count:] = -torch.inf
        values, indices = logits.max(-1)
        # Float64 preserves both FP32 scores and token ids in a single collective.
        candidates = torch.stack(
            (values.to(torch.float64), (indices + start).to(torch.float64)), -1
        )
        # PyNCCL accepts only 16-bit floats; transport the bits without conversion.
        gathered = head._comm.all_gather(candidates.view(torch.bfloat16))
        gathered = gathered.view(torch.float64).view(head.tp_size, mixed.shape[0], 2)
        # Rank-major order preserves full-vocab argmax's lowest-id tie break.
        winners = gathered[:, :, 0].argmax(0)
        return gathered[:, :, 1].gather(0, winners.unsqueeze(0)).squeeze(0).to(torch.int64)

    def draft(self, residual: torch.Tensor, next_ids: torch.Tensor, batch: Batch,
              *, select_row: torch.Tensor | None = None) -> torch.Tensor:
        """Greedy MTP draft ids for the rows of ``residual`` (the spec step's head pass).

        All rows update head KV. select_row chooses which verified prefix needs
        a new draft before the shared mixer and vocabulary projection.
        """
        if self.mtp is None:
            raise AssertionError("draft() needs the MTP head (--speculative mtp)")
        next_embed = self.model.embed_tokens.forward(next_ids)
        head_out = self.mtp.forward(residual, next_embed, batch)
        if select_row is not None:
            head_out = head_out.index_select(0, select_row)
        mixed = self.model.hyper_connection_mixer.mix(head_out)[0]
        return self.greedy_ids(mixed)


class Qwen4ExpForConditionalGeneration(QwenVLVisionMixin, Qwen4ExpForCausalLM):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        if config.is_multimodal:
            assert not config.vision_config.deepstack_visual_indexes, "Qwen3.8 consumes no DeepStack features"
            self.visual = Qwen3VLVisionModel(config.vision_config, quant_config=config.quant, prefix="visual")


__all__ = [
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpModel",
    "build_linear_mixer",
]
