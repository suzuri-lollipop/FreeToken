"""Hy4-flash LatentMoE: routed experts run in a 1536-wide latent space.

``routed_expert_down_proj`` maps the hidden states into the latent width before the
experts and ``routed_expert_up_proj`` maps their combined output back; the router
and the shared expert keep seeing the full-width input. Routing is sigmoid scores
plus a selection-only bias (noaux_tc with n_group=1), top-16, renormalized and
scaled by 3.998. The expert SwiGLU clamps gate at +10 and up at +/-10 (the MoE
layer's ``limit``). ``combine_output_norm`` closes the block (sandwich norm).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.distributed import get_tp_info
from freetoken.kernel.triton.moe_router import fused_topk_sigmoid
from freetoken.layers import BaseOP, LinearReplicated, LinearRowParallel, RMSNorm, make_moe_layer, silu_and_mul

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Hy4DenseMLP(BaseOP):
    """Layer 0's dense FFN, with the extra post-fc2 norm of a standalone residual sublayer.

    TP: gate/up column-parallel over the intermediate (local output), down
    row-parallel (one all-reduce).
    """

    def __init__(self, config: ModelConfig, *, prefix: str = "") -> None:
        args = config.hy4_args
        tp = get_tp_info().size
        if args.intermediate_size % tp:
            raise ValueError(f"Hy4-flash dense intermediate {args.intermediate_size} not divisible by TP {tp}")
        local_inter = args.intermediate_size // tp
        self.gate_proj = LinearReplicated(
            args.dim, local_inter, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = LinearReplicated(
            args.dim, local_inter, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.up_proj",
        )
        self.down_proj = LinearRowParallel(
            args.intermediate_size, args.dim, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.down_proj",
        )
        self.post_fc2_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = silu_and_mul(torch.cat([self.gate_proj.forward(x), self.up_proj.forward(x)], dim=-1))
        return self.post_fc2_norm.forward(self.down_proj.forward(y))


class Hy4SharedExpert(BaseOP):
    """The always-on shared expert (full width, no post norm: the block carries it).

    TP: gate/up column-parallel, down row-parallel, as the dense FFN.
    """

    def __init__(self, config: ModelConfig, *, prefix: str = "") -> None:
        args = config.hy4_args
        inter = args.moe_inter_dim * args.n_shared_experts
        tp = get_tp_info().size
        if inter % tp:
            raise ValueError(f"Hy4-flash shared expert intermediate {inter} not divisible by TP {tp}")
        local_inter = inter // tp
        self.gate_proj = LinearReplicated(
            args.dim, local_inter, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = LinearReplicated(
            args.dim, local_inter, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.up_proj",
        )
        self.down_proj = LinearRowParallel(
            inter, args.dim, has_bias=False, quant_config=config.quant, prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = silu_and_mul(torch.cat([self.gate_proj.forward(x), self.up_proj.forward(x)], dim=-1))
        return self.down_proj.forward(y)


class Hy4LatentMoE(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = "") -> None:
        args = config.hy4_args
        self.layer_id = layer_id
        self.top_k = args.n_activated_experts
        self.num_experts = args.n_routed_experts
        self.norm_topk_prob = args.norm_topk_prob
        self.routed_scaling_factor = args.route_scale
        self.gate = LinearReplicated(args.dim, args.n_routed_experts, has_bias=False)
        self.e_score_correction_bias = torch.empty(args.n_routed_experts, dtype=torch.float32)
        self.routed_expert_down_proj = LinearReplicated(
            args.dim, args.latent_dim, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.routed_expert_down_proj",
        )
        # the offload cache indexes experts by MoE layer (global minus first_k_dense_replace)
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id - args.first_k_dense_replace,
            hidden_size=args.latent_dim,
            intermediate_size=args.moe_inter_dim,
            renormalize=args.norm_topk_prob,
            # clamp(gate, max=limit) * sigmoid(gate) * clamp(up, +-limit): the reference _apply_gate
            activation="swiglu_clamp" if args.swiglu_limit > 0 else "silu",
            limit=args.swiglu_limit if args.swiglu_limit > 0 else None,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        self.routed_expert_up_proj = LinearReplicated(
            args.latent_dim, args.dim, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.routed_expert_up_proj",
        )
        self.shared_experts = Hy4SharedExpert(config, prefix=f"{prefix}.shared_experts")
        self.combine_output_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # bf16-trained model: keep the routed activations bf16 (W8A16), see attention.py
        self.experts.fp8_w8a16 = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.mm(x, self.gate.weight.t(), out_dtype=torch.float32)
        topk_weights, topk_ids = fused_topk_sigmoid(
            logits, self.e_score_correction_bias, self.top_k, self.norm_topk_prob,
            routed_scale=self.routed_scaling_factor,
        )
        # routed kernels may overwrite their input; the shared expert needs the original
        shared = self.shared_experts.forward(x)
        latent = self.routed_expert_down_proj.forward(x)
        routed = self.experts.routed_forward(latent, topk_weights, topk_ids)
        routed = self.routed_expert_up_proj.forward(routed)
        return self.combine_output_norm.forward(routed + shared)


__all__ = ["Hy4LatentMoE", "Hy4DenseMLP", "Hy4SharedExpert"]
