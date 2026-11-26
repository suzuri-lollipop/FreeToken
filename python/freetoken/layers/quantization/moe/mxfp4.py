"""MXFP4 experts: the standard OCP form (DeepSeek-V4) and the gpt-oss variant with biases."""

from __future__ import annotations

import torch

from ..registry import LayerKind, register_method
from ..scheme import MX_GROUP as GROUP, QuantKind
from .base import BankSpec, ExpertView, fused_piece, is_resident, limit_or_inf, MoEConfig, MoEKernel, MoEMethod, tp_piece, tp_rows

E8M0 = torch.float8_e8m0fnu


class TritonMxfp4MoEKernel(MoEKernel):
    """Standard OCP MXFP4 experts (DeepSeek-V4 / V4.1 ds_fp4): e2m1 pairs + e8m0 scales, no bias;
    W4A8 with the fp8 activation round-trips at the checkpoint's block."""

    name = "triton"
    cpu_format = "ds_fp4"
    supports_inactive_slots = True
    supports_skip_w0 = True  # _decode_dsfp4_moe_kernel zero-stores w==0 routes unread

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        if cfg.interleaved:
            return "standard MXFP4 kernel reads the concatenated gate|up row order"
        if cfg.activation == "situ":
            # ERNIE 5.5 SituGLU: alpha is the gate tanh softcap, limit the up softcap.
            if cfg.beta != 0.0 or cfg.limit is None:
                return "situ experts need beta=0 and an up softcap (limit)"
        elif (cfg.alpha, cfg.beta) != (1.0, 0.0):
            return "standard MXFP4 kernel has no alpha / beta in its swiglu"
        if cfg.act_block is None:
            return "W4A8 ds_fp4 experts need the fp8 activation quant block"
        # TP: the banks hold this rank's intermediate slice (layout/pack above), the down
        # GEMM sums only that slice, and the MoE layer all-reduces the partials. The decode
        # GEMV needs K//2 % 128 == 0 (local intermediate a multiple of 256); the prefill
        # grouped GEMM has no such floor.
        return self._common_reject(cfg, tp_ok=True, cpu_ok=True, plain_silu_only=False)

    def cpu_options(self, cfg: MoEConfig) -> dict[str, int]:
        return {"act_block": int(cfg.act_block)}

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        # banks hold this rank's slice of the expert intermediate (local == full at TP1,
        # so the resident/offload TP1 layout is byte-identical); pack() cuts the matching slice
        i, h = cfg.local_intermediate, cfg.hidden
        return {
            "gate_up": BankSpec((2 * i, h // 2), torch.uint8),
            "gate_up_scale": BankSpec((2 * i, h // GROUP), E8M0),
            "down": BankSpec((h, i // 2), torch.uint8),
            "down_scale": BankSpec((h, i // GROUP), E8M0),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        gu = fused_piece(pieces, "gate_up").view(torch.uint8)
        gus = fused_piece(pieces, "gate_up_scale").view(E8M0)
        down = pieces["down"].view(torch.uint8)
        downs = pieces["down_scale"].view(E8M0)
        if cfg.tp_size > 1:
            # the reader hands over full experts; this rank banks its own intermediate slice
            # (gate|up rows, down columns). tp_piece takes the cut in each piece's own grain
            # (packed e2m1 pairs /2, e8m0 scale groups /GROUP), so the scales stay aligned.
            lo, hi = cfg.local_intermediate_range
            total = cfg.intermediate
            # pieces carry the expert axis first: the intermediate is dim 1 of the fused
            # gate|up (2*total rows) and dim 2 of down (packed columns) -- the NVFP4 axes
            gu = tp_rows(gu, 1, lo, hi, total)
            gus = tp_rows(gus, 1, lo, hi, total)
            down = tp_piece(down, 2, lo, hi, total)
            downs = tp_piece(downs, 2, lo, hi, total)
        out["gate_up"].copy_(gu)
        out["gate_up_scale"].copy_(gus)
        out["down"].copy_(down)
        out["down_scale"].copy_(downs)
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        t = view.tensors
        banks = (t["gate_up"], t["gate_up_scale"], t["down"], t["down_scale"])
        limit = limit_or_inf(layer)
        act_block = layer.quant_method.cfg.act_block
        situ_alpha = float(layer.alpha) if layer.activation == "situ" else None
        if is_prefill and view.n is not None:
            from freetoken.moe.fused_ds_fp4 import routed_experts_fp4_prefill

            return routed_experts_fp4_prefill(x, topk_ids, topk_weights, *banks, limit, view.n, act_block=act_block, situ_alpha=situ_alpha)
        from freetoken.moe.fused_ds_fp4 import routed_experts_fp4

        # _moe_skip_w0 is a capture-time transient set by the fetch-overlap decode
        # (layers/moe.py): each pass zero-stores the other half's routes unread.
        return routed_experts_fp4(x, topk_ids, topk_weights, *banks, limit, act_block=act_block, situ_alpha=situ_alpha,
                                  skip_w0=getattr(layer, "_moe_skip_w0", False))


class TritonGptossMxfp4MoEKernel(MoEKernel):
    """gpt-oss MXFP4: bias, alpha / limit swiglu, interleaved gate|up rows, transposed split-K banks."""

    name = "triton_gptoss"
    cpu_format = "mxfp4_triton"

    def unusable_reason(self, cfg: MoEConfig) -> str | None:
        if not cfg.has_bias:
            return "gpt-oss kernel needs the expert biases"
        return None

    def layout(self, cfg: MoEConfig) -> dict[str, BankSpec]:
        i, h = cfg.local_intermediate, cfg.hidden
        return {
            "gate_up": BankSpec((h // 2, 2 * i), torch.uint8),
            "gate_up_scale": BankSpec((h // GROUP, 2 * i), torch.uint8),
            "gate_up_bias": BankSpec((2 * i,), torch.bfloat16),
            "down": BankSpec((i // 2, h), torch.uint8),
            "down_scale": BankSpec((i // GROUP, h), torch.uint8),
            "down_bias": BankSpec((h,), torch.bfloat16),
        }

    def pack(self, pieces, cfg: MoEConfig, out):
        # HF [e, N, K//32, 16] blocks / [e, N, K//32] scales -> split-K layout with N innermost
        def blocks_t(t):
            e, n = t.shape[0], t.shape[1]
            return t.reshape(e, n, -1).permute(0, 2, 1)

        out["gate_up"].copy_(blocks_t(fused_piece(pieces, "gate_up")))
        out["gate_up_scale"].copy_(fused_piece(pieces, "gate_up_scale").permute(0, 2, 1))
        out["gate_up_bias"].copy_(fused_piece(pieces, "gate_up_bias"))
        out["down"].copy_(blocks_t(pieces["down"]))
        out["down_scale"].copy_(pieces["down_scale"].permute(0, 2, 1))
        out["down_bias"].copy_(pieces["down_bias"])
        return {}

    def apply(self, layer, x, topk_weights, topk_ids, view: ExpertView, *, is_prefill: bool):
        from freetoken.moe.fused_mxfp4 import MXFP4_DECODE_MAX_TOKENS, run_mxfp4_prefill_experts_t, run_mxfp4_splitk_decode_experts

        t = view.tensors
        # the resident layer picks the split-K decode kernel by token count, not by phase
        decode = x.shape[0] <= MXFP4_DECODE_MAX_TOKENS if is_resident(layer) else not is_prefill
        run = run_mxfp4_splitk_decode_experts if decode else run_mxfp4_prefill_experts_t
        return run(
            x, topk_weights, topk_ids,
            t["gate_up"], t["gate_up_scale"], t["gate_up_bias"], t["down"], t["down_scale"], t["down_bias"],
            top_k=layer.top_k, hidden_act_alpha=float(layer.alpha), swiglu_limit=layer.limit,
        )


@register_method(QuantKind.MXFP4, LayerKind.MOE)
class Mxfp4MoEMethod(MoEMethod):
    candidates = (TritonMxfp4MoEKernel, TritonGptossMxfp4MoEKernel)
