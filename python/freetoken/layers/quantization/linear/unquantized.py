"""bf16 Linear: one kernel (torch), no scheme.

When FREETOKEN_FP8_DECODE_LINEAR=1, large projections are pre-quantized to fp8
at finalize() and use torch._scaled_mm during decode for ~1.7x GEMM speedup.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import LinearKernel, LinearMethod

# FP8 decode acceleration settings
_FP8_MIN_ELEMENTS = int(os.environ.get("FREETOKEN_FP8_DECODE_MIN_ELEMENTS", "4000000"))
_FP8_MAX_BATCH = int(os.environ.get("FREETOKEN_FP8_DECODE_MAX_BATCH", "8"))
_FP8_ENABLED = os.environ.get("FREETOKEN_FP8_DECODE_LINEAR", "0") == "1"


class TorchLinearKernel(LinearKernel):
    name = "torch"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        # an fp32 activation stream (DeepSeek-V4's compressors) upcasts the bf16 weight on the fly, as the reference does
        w, b = layer.weight, layer.bias

        # FP8 decode path: use pre-quantized fp8 weight for large projections at small batch
        fp8_w = getattr(layer, "_fp8_weight", None)
        fp8_scale = getattr(layer, "_fp8_scale", None)

        if (
            _FP8_ENABLED
            and fp8_w is not None
            and fp8_scale is not None
            and x.shape[0] <= _FP8_MAX_BATCH
            and b is None
            and x.is_cuda
        ):
            try:
                return self._fp8_linear(x, fp8_w, fp8_scale, w.dtype)
            except Exception:
                pass  # Fall back to bf16 on any error

        if w.dtype != x.dtype:
            w = w.to(x.dtype)
            b = b.to(x.dtype) if b is not None else None
        return F.linear(x, w, b)

    @staticmethod
    def _fp8_linear(
        x: torch.Tensor,
        w_fp8: torch.Tensor,
        w_scale: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Run fp8 GEMM: x @ w_fp8.T with per-tensor scaling."""
        x_amax = x.abs().max()
        x_scale = (x_amax / 448.0).clamp(min=1e-12).to(torch.float32)
        x_fp8 = (x / x_scale).to(torch.float8_e4m3fn)
        combined_scale = x_scale * w_scale

        return torch._scaled_mm(
            x_fp8,
            w_fp8.T,
            scale_a=combined_scale,
            scale_b=torch.tensor(1.0, dtype=torch.float32, device=x.device),
            out_dtype=out_dtype,
        )


@register_method(QuantKind.NONE, LayerKind.LINEAR)
class UnquantizedLinearMethod(LinearMethod):
    candidates = (TorchLinearKernel,)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        layer.weight = torch.empty(g.out_features, g.in_features)

    def finalize(self, layer: Any) -> None:
        """Pre-quantize large weights to fp8 for decode acceleration."""
        if not _FP8_ENABLED:
            return

        w = layer.weight
        if w is None or not w.is_cuda:
            return

        K = w.shape[1]  # in_features
        N = w.shape[0]  # out_features

        if K * N < _FP8_MIN_ELEMENTS:
            return

        # Pre-quantize weight to fp8 (persistent buffer, CUDA-graph compatible)
        amax = w.abs().max()
        scale = (amax / 448.0).clamp(min=1e-12).to(torch.float32)
        w_fp8 = (w / scale).to(torch.float8_e4m3fn)

        layer._fp8_weight = w_fp8
        layer._fp8_scale = scale
