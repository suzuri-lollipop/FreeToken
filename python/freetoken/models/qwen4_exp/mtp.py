"""MTP speculative head for Qwen3.8-Flash-Next (qwen4_exp).

One extra full-attention decoder layer over the hyper-connection streams plus the
fc/norm front end that fuses the next-token embedding with the main model's final
residual ``R [T, hc_count*hidden]``. Shares ``embed_tokens``/``lm_head``/the top-level
mixer with the main model (the caller passes them in; nothing is double-registered).

Attach the head as the owning model's ``mtp`` attribute: its state-dict names then
match the checkpoint's ``mtp.*`` dense keys exactly (loader fusions aside: q|k|v ->
qkv_proj and down+inject -> input_mix_weight_down_block_inject, both already handled
for the main layers).

Wiring (design doc _scratch/mtp_design.md section 4; the checkpoint ships no HF
reference, so the front-end order is an empirical question settled by acceptance-rate
probing -- ``wiring`` selects the hypothesis under test):

    e  = fc_embedding(pre_fc_norm_embedding(embed(next_ids)))        # [T, H]
    h  = fc_hidden(mixer.<mix>(pre_fc_norm_hidden(R)))               # [T, H]
    R0 = (e + h).repeat(1, hc_count)                                 # [T, HC*H]
    -- one Qwen4ExpDecoderLayer-shaped pass (attn HC -> QSA -> mlp HC -> MoE) --
    R_out                                                            # [T, HC*H]

and the spec step collapses ``R_out`` through the main model's top-level mixer and
lm_head. The front-end order was settled empirically (the checkpoint ships no HF
reference): the acceptance-rate probe on real weights (_scratch/mtp_probe.py,
2026-09-23, 550 teacher-forced positions) measured "norm_mixfrom_fc" -- pre_fc_norm_hidden
norms R and the mixer consumes it via mix_from_normed, its own hc_norm idle -- at
0.560 greedy acceptance vs 0.184 for the double-normed "norm_mix_fc" variant.
rope_theta is shared: the released config has mtp.rope_theta equal to the main
rope_theta (1e7), asserted at build.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from freetoken.layers import BaseOP, LinearReplicated, OPList

from .attention import Qwen4ExpAttention
from .hc import GatedResidual, GroupedPlusOneRMSNorm
from .moe import Qwen4ExpMoE

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig

MTP_WIRINGS = ("norm_mix_fc", "norm_mixfrom_fc")


class Qwen4ExpMTPLayer(BaseOP):
    """One MTP decoder layer: the frozen main-layer contract (model.py docstring)
    minus PLE -- the head has no n-gram layer of its own."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = "") -> None:
        self.attn_hyper_connection = GatedResidual(config, prefix=f"{prefix}.attn_hyper_connection")
        self.self_attn = Qwen4ExpAttention(config, layer_id, prefix=f"{prefix}.self_attn")
        self.mlp_hyper_connection = GatedResidual(config, prefix=f"{prefix}.mlp_hyper_connection")
        # The head's experts are RESIDENT whatever the engine's moe_strategy is: the
        # offload cache is keyed by main-decoder layer ids (0..num_layers-1), so an
        # OffloadMoELayer at the synthetic index would fall outside every cache. The
        # fp8-block resident method is TP1-only today (fp8_block.py tp_ok=False) --
        # TP2 placement is Phase 2 of the design doc; until then a TP2 engine building
        # the head fails loudly at MoE construction.
        self.mlp = Qwen4ExpMoE(
            replace(config, moe_strategy="fused"), layer_id, prefix=f"{prefix}.mlp"
        )

    def forward(self, hidden: torch.Tensor, batch: "Batch") -> torch.Tensor:
        block_input, inject = self.attn_hyper_connection.mix(hidden)
        block_output = self.self_attn.forward(block_input, batch)
        hidden, mlp_rn = self.attn_hyper_connection.combine_norm(
            hidden, block_output, inject,
            self.mlp_hyper_connection.hc_norm.weight,
            self.mlp_hyper_connection.hc_norm.eps,
        )
        block_input, inject = self.mlp_hyper_connection.mix_from_normed(mlp_rn)
        return self.mlp_hyper_connection.combine(hidden, self.mlp.forward(block_input), inject)


class Qwen4ExpMTPHead(BaseOP):
    """The single-layer MTP head. ``layer_id`` is the synthetic decoder index the
    attention backend and MoE cache see (num_layers of the main model)."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        layer_id: int,
        wiring: str = "norm_mixfrom_fc",
        prefix: str = "mtp",
    ) -> None:
        if wiring not in MTP_WIRINGS:
            raise ValueError(f"unknown MTP wiring {wiring!r}; expected one of {MTP_WIRINGS}")
        args = config.qwen4_args
        if args.mtp_num_layers < 1:
            raise ValueError("config carries no MTP block (mtp_num_layers == 0)")
        if args.mtp_rope_theta is not None:
            main_theta = config.rotary_config.base
            if float(args.mtp_rope_theta) != float(main_theta):
                raise NotImplementedError(
                    f"MTP rope_theta {args.mtp_rope_theta} != main {main_theta}: the head "
                    "shares config.rotary_config and needs its own RotaryConfig first"
                )
        self.wiring = wiring
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        width = args.ple_state_width  # hc_count * hidden

        self.pre_fc_norm_embedding = GroupedPlusOneRMSNorm(self.hidden_size, config.rms_norm_eps, 1)
        self.pre_fc_norm_hidden = GroupedPlusOneRMSNorm(width, config.rms_norm_eps, self.hc_count)
        self.fc_embedding = LinearReplicated(
            self.hidden_size, self.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.fc_embedding",
        )
        self.fc_hidden = LinearReplicated(
            self.hidden_size, self.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.fc_hidden",
        )
        self.hyper_connection_mixer = GatedResidual(
            config, use_combine=False, prefix=f"{prefix}.hyper_connection_mixer"
        )
        self.layers = OPList(
            [Qwen4ExpMTPLayer(config, layer_id, prefix=f"{prefix}.layers.{i}")
             for i in range(args.mtp_num_layers)]
        )

    def fuse_input(self, residual: torch.Tensor, next_embed: torch.Tensor) -> torch.Tensor:
        """Front end: (R, embed(next_ids)) -> the head-layer input stream R0 [T, HC*H].

        Split out from forward so the wiring probe can drive it without attention.
        """
        e = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(next_embed))
        if self.wiring == "norm_mix_fc":
            mixed = self.hyper_connection_mixer.mix(self.pre_fc_norm_hidden.forward(residual))[0]
        else:  # norm_mixfrom_fc
            mixed = self.hyper_connection_mixer.mix_from_normed(
                self.pre_fc_norm_hidden.forward(residual)
            )[0]
        h = self.fc_hidden.forward(mixed)
        return (e + h).repeat(1, self.hc_count)

    def forward(self, residual: torch.Tensor, next_embed: torch.Tensor, batch: "Batch") -> torch.Tensor:
        """Run the head over one step's final residual; returns R_out [T, HC*H] for the
        caller to collapse through the shared top-level mixer + lm_head."""
        hidden = self.fuse_input(residual, next_embed)
        for layer in self.layers.op_list:
            hidden = layer.forward(hidden, batch)
        return hidden


__all__ = ["MTP_WIRINGS", "Qwen4ExpMTPHead", "Qwen4ExpMTPLayer"]
