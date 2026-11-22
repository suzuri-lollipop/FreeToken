"""Inkling-2-Mini geometry: the loop schedule, attention dims and short-conv state layout.

The checkpoint stores 24 physical decoder layers; the middle 12 run three times, so a
forward executes 48 layer visits. Every visit keeps its own KV and short-conv state
(the engine's logical layer id), while the weights and the MoE expert banks are shared
by physical layer. Reference: llama.cpp's inkling2_loop graph (ggml-org/llama.cpp#32694)
and the transformers inkling2 modules for the per-layer math.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# per-request conv history riding the linear-state slots (see LinearStatePool.slot_states)
CONV_STATE = "inkling2_conv"


@dataclass(frozen=True)
class Inkling2Args:
    hidden_size: int
    num_phys_layers: int
    num_visits: int
    core_start: int
    core_end: int
    num_passes: int
    gate_rank: int
    block_size: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    d_rel: int
    rel_extent: int
    sliding_window: int
    qk_clip: float
    rel_clip: float
    log_scale_floor: int
    log_scale_alpha: float
    conv_kernel: int
    situ_gate_cap: float
    situ_up_cap: float
    route_scale: float
    rms_norm_eps: float
    final_logit_softcapping: float
    padded_vocab_size: int
    unpadded_vocab_size: int
    image_token_id: int
    # physical layers using sliding-window attention; the rest attend the whole history
    swa_physical: Tuple[int, ...]
    # physical layer of every visit, in execution order
    visit_physical: Tuple[int, ...]

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def q_dim(self) -> int:
        return self.num_qo_heads * self.head_dim

    @property
    def conv_channels(self) -> int:
        # K and V convs, then the attention-output and FFN-output convs
        return 2 * self.kv_dim + 2 * self.hidden_size

    @property
    def conv_state_len(self) -> int:
        return self.conv_kernel - 1

    def is_swa_visit(self, visit: int) -> bool:
        return self.visit_physical[visit] in self.swa_physical

    def physical(self, visit: int) -> int:
        return self.visit_physical[visit]

    def swa_visits(self) -> Tuple[int, ...]:
        return tuple(v for v in range(self.num_visits) if self.is_swa_visit(v))

    def full_visits(self) -> Tuple[int, ...]:
        return tuple(v for v in range(self.num_visits) if not self.is_swa_visit(v))

    def rel_table_width(self, visit: int) -> int:
        # the bias covers [0, extent); the extra column holds the zero bias of farther keys
        extent = self.sliding_window if self.is_swa_visit(visit) else self.rel_extent
        return extent + 1


def build_args(
    *,
    hidden_size: int,
    num_hidden_layers: int,
    loop_config: dict,
    local_layer_ids: list[int],
    n_layers_per_attn_res_block: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    d_rel: int,
    rel_extent: int,
    sliding_window_size: int,
    qk_l2_clip_threshold: float,
    rel_l2_clip_threshold: float,
    log_scaling_n_floor: int,
    log_scaling_alpha: float,
    sconv_kernel_size: int,
    mlp_gate_softcap: float,
    mlp_up_softcap: float,
    route_scale: float,
    rms_norm_eps: float,
    final_logit_softcapping: float,
    vocab_size: int,
    unpadded_vocab_size: int,
    image_token_id: int,
) -> Inkling2Args:
    start, end = int(loop_config["core_start"]), int(loop_config["core_end"])
    passes, rank = int(loop_config["num_passes"]), int(loop_config["gate_rank"])
    block = int(n_layers_per_attn_res_block)
    if not (0 < start < end <= num_hidden_layers) or start % block or end % block or passes < 2 or rank <= 0:
        raise ValueError(f"inkling2_loop: invalid loop_config {loop_config} for {num_hidden_layers} layers")
    if bool(loop_config.get("is_varloop", False)):
        raise NotImplementedError("inkling2_loop: the variable-pass research variant is not served")
    visits = tuple(range(start)) + tuple(range(start, end)) * passes + tuple(range(end, num_hidden_layers))
    return Inkling2Args(
        hidden_size=hidden_size,
        num_phys_layers=num_hidden_layers,
        num_visits=len(visits),
        core_start=start,
        core_end=end,
        num_passes=passes,
        gate_rank=rank,
        block_size=block,
        num_qo_heads=num_attention_heads,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        d_rel=d_rel,
        rel_extent=rel_extent,
        sliding_window=sliding_window_size,
        qk_clip=qk_l2_clip_threshold,
        rel_clip=rel_l2_clip_threshold,
        log_scale_floor=log_scaling_n_floor,
        log_scale_alpha=log_scaling_alpha,
        conv_kernel=sconv_kernel_size,
        situ_gate_cap=mlp_gate_softcap,
        situ_up_cap=mlp_up_softcap,
        route_scale=route_scale,
        rms_norm_eps=rms_norm_eps,
        final_logit_softcapping=final_logit_softcapping,
        padded_vocab_size=vocab_size,
        unpadded_vocab_size=unpadded_vocab_size,
        image_token_id=image_token_id,
        swa_physical=tuple(sorted(local_layer_ids)),
        visit_physical=visits,
    )


__all__ = ["CONV_STATE", "Inkling2Args", "build_args"]
