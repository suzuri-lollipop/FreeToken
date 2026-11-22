"""Inkling-2 MoE: fp32 sigmoid router with a selection bias, top-8 of 128 SITU experts.

Routing follows the DeepSeek/GLM family (fused_topk_sigmoid): sigmoid scores, the bias
shifts only the top-k selection, the unbiased gathered scores renormalize (eps 1e-20)
and scale by route_scale (8). The experts run the tanh-softcapped SwiGLU ("situ"):
gate cap 4, up cap 25, sigmoid on the raw gate. Layers 0-19 store NVFP4 experts, layers
20-23 MXFP8 (served dequantized to bf16); the per-layer scheme comes from the registry's
unquantized_modules, so make_moe_layer picks the method per physical layer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.distributed import get_tp_info
from freetoken.kernel.triton.moe_router import fused_topk_sigmoid
from freetoken.layers import BaseOP, LinearReplicated, make_moe_layer
from freetoken.models.config import ModelConfig

from .args import Inkling2Args


class Inkling2MoE(BaseOP):
    def __init__(self, config: ModelConfig, phys_layer: int, *, prefix: str = ""):
        args: Inkling2Args = config.inkling2_args
        self.top_k = config.num_experts_per_tok
        self.route_scale = args.route_scale
        tp = get_tp_info()
        self.tp_size = tp.size
        self.tp_rank = tp.rank
        self.num_experts = config.num_experts
        self.local_experts = config.num_experts // tp.size
        self.expert_offset = self.tp_rank * self.local_experts
        # the reference keeps the router fp32: the bias sits on the top-8 selection boundary
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.gate.weight = torch.empty(config.num_experts, config.hidden_size, dtype=torch.float32)
        self.gate_bias = torch.empty(config.num_experts, dtype=torch.float32)
        self.experts = make_moe_layer(
            config,
            layer_id=phys_layer,
            renormalize=True,
            activation="situ",
            alpha=args.situ_gate_cap,
            limit=args.situ_up_cap,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
            num_experts=self.local_experts,
        )
        if self.tp_size > 1:
            # EP: each rank holds full-intermediate experts; the intermediate is NOT TP-split.
            # Override the quant cfg so bank shapes and kernels use the full width.
            object.__setattr__(self.experts.quant_method.cfg, "tp_size", 1)
            object.__setattr__(self.experts.quant_method.cfg, "tp_rank", 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = F.linear(x.float(), self.gate.weight)
        topk_weights, topk_ids = fused_topk_sigmoid(
            logits, self.gate_bias, self.top_k, True, routed_scale=self.route_scale
        )
        if self.tp_size > 1:
            # EP: remap global expert ids to local; non-local routes become -1 (inactive)
            local_ids = topk_ids - self.expert_offset
            local_ids = torch.where(
                (local_ids >= 0) & (local_ids < self.local_experts), local_ids, -1
            )
            topk_ids = local_ids
        return self.experts.routed_forward(x, topk_weights, topk_ids)


__all__ = ["Inkling2MoE"]
