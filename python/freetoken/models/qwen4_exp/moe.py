from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen4ExpMoE(Qwen3_5MoE):
    """Qwen3_5MoE with the shared-expert gate on triton instead of gemv + sigmoid + mul + add.

    Same weights, same state dict. The gate reduction stays ahead of the routed experts, which may write into ``hidden_states`` in place.

    The shared expert and the routed experts are both row-parallel over the same output,
    so under TP their partials are summed BEFORE one all-reduce instead of two: the gate
    is rank-replicated, hence ``AR(routed) + AR(shared)*gate == AR(routed + shared*gate)``.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward_partial(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        routed = self.experts.routed_partial(hidden_states, router_logits)
        combined = shared_gate_mul_add(routed, shared, gate)
        return self.experts.reduce_partial(combined).view(num_tokens, hidden_dim)


__all__ = ["Qwen4ExpMoE"]
