"""ERNIE 5.5 FFN: dense SiTU-GLU MLP and the Multi-Router Latent MoE.

Routing (``moe_split_feature_routing`` + learnable per-expert scale): two routers
score every token through ONE fp32 ``[2E, H]`` GEMM; ``sigmoid(l0) + sigmoid(l1)``
are bias-corrected (``e_score_correction_bias``), top-k'd, renormalized and
multiplied by ``routed_scaling_factor_param[expert]``.

Experts run in the ``moe_latent_size`` space between ``fc1_latent_proj`` and
``latent_norm`` + ``fc2_latent_proj`` (applied AFTER the expert combine -- the
norm is non-linear); the shared SiTU-GLU expert sees the original hidden states.
Routed experts are MXFP4 (W4A8) on the offload cache, SituGLU activation.
"""

from __future__ import annotations

import torch

from freetoken.kernel.triton.ernie55 import split_sigmoid_sum
from freetoken.layers import BaseOP, LinearColParallelMerged, LinearReplicated, LinearRowParallel, MoELayer, OffloadMoELayer, RMSNorm
from freetoken.layers.activation import situ_and_mul
from freetoken.layers.moe import is_offload_moe_strategy

from .args import Ernie55Args


def ernie55_routing(
    scores: torch.Tensor, topk: int, renormalize: bool,
    bias: torch.Tensor, scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bias-corrected top-k over the split-feature sigmoid scores (torch reference
    semantics: ``topk(sorted=False)`` slots, renorm clamp 1e-20, per-expert scale)."""
    ids = torch.topk(scores + bias, topk, dim=-1, sorted=False).indices
    weights = scores.gather(1, ids)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-20)
    weights = weights * scale[ids]
    return weights.float(), ids.to(torch.int32)


class Ernie55MLP(BaseOP):
    """SiTU-GLU feed-forward block (dense layer 0 and the shared expert)."""

    def __init__(self, dim: int, inter_dim: int, args: Ernie55Args, *, quant_config=None, prefix: str = ""):
        # TP: gate/up column-parallel (this rank's intermediate rows), down row-parallel
        # (all-reduce). At TP1 the local sizes equal the full ones, so the path is identical.
        self.gate_proj = LinearColParallelMerged(dim, [inter_dim], has_bias=False, quant_config=quant_config, prefix=f"{prefix}.gate_proj")
        self.up_proj = LinearColParallelMerged(dim, [inter_dim], has_bias=False, quant_config=quant_config, prefix=f"{prefix}.up_proj")
        self.down_proj = LinearRowParallel(inter_dim, dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.down_proj")
        self.beta = args.situ_beta
        self.linear_beta = args.situ_linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = torch.cat([self.gate_proj.forward(x), self.up_proj.forward(x)], dim=-1)
        h = situ_and_mul(gate_up, alpha=self.beta, limit=self.linear_beta)
        return self.down_proj.forward(h)


class Ernie55MoE(BaseOP):
    """Latent MoE: split-feature routing -> MXFP4 experts in latent space ->
    latent_norm + fc2, plus the shared expert on the original hidden states."""

    def __init__(
        self, layer_id: int, args: Ernie55Args, *, strategy: str = "offload",
        decode_target: str = "gpu", quant_config=None, prefix: str = "",
    ):
        self.dim = args.dim
        self.latent_size = args.moe_latent_size
        self.topk = args.n_activated_experts
        self.num_experts = args.n_routed_experts
        self.norm_topk_prob = args.norm_topk_prob
        # the two split-feature routers share one [2E, H] fp32 GEMM
        self.gate_weight = torch.empty(2 * args.n_routed_experts, args.dim, dtype=torch.float32)
        self.e_score_correction_bias = torch.empty(args.n_routed_experts, dtype=torch.float32)
        self.routed_scaling_factor_param = torch.empty(args.n_routed_experts, dtype=torch.float32)

        self.fc1_latent_proj = LinearReplicated(args.dim, self.latent_size, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.fc1_latent_proj")
        self.latent_norm = RMSNorm(self.latent_size, args.norm_eps)
        self.fc2_latent_proj = LinearReplicated(self.latent_size, args.dim, has_bias=False, quant_config=quant_config, prefix=f"{prefix}.fc2_latent_proj")
        self.shared_experts = Ernie55MLP(
            args.dim, args.moe_inter_dim * args.n_shared_experts, args,
            quant_config=quant_config, prefix=f"{prefix}.shared_experts",
        )
        # the strategy seam: offload family (MXFP4 banks) vs resident (bf16 parity path)
        expert_cls = OffloadMoELayer if is_offload_moe_strategy(strategy) else MoELayer
        self.experts = expert_cls(
            # bank-relative id (the leading dense layers own no expert bank)
            layer_id=layer_id - args.first_k_dense_replace,
            num_experts=args.n_routed_experts,
            top_k=args.n_activated_experts,
            hidden_size=self.latent_size,
            intermediate_size=args.moe_inter_dim,
            # the custom router already renormalizes + scales the weights
            renormalize=False,
            activation="situ",
            alpha=args.situ_beta,
            limit=args.situ_linear_beta,
            strategy=strategy,
            decode_target=decode_target,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = x.float() @ self.gate_weight.t()
        scores = split_sigmoid_sum(logits, self.num_experts)
        return ernie55_routing(
            scores, self.topk, self.norm_topk_prob,
            self.e_score_correction_bias, self.routed_scaling_factor_param,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.route(x)
        # shared expert enqueued before routed_forward (hybrid decode overlap,
        # DSV4 precedent); it reads the ORIGINAL hidden states.
        shared = self.shared_experts.forward(x)
        latent = self.fc1_latent_proj.forward(x)
        routed = self.experts.routed_forward(
            latent, weights.float().contiguous(), indices.to(torch.int32).contiguous()
        )
        out = self.fc2_latent_proj.forward(self.latent_norm.forward(routed)) + shared
        return out.view(shape)


__all__ = ["Ernie55MLP", "Ernie55MoE", "ernie55_routing"]
