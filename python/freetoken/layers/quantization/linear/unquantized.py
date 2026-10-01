"""bf16 Linear: one kernel (torch), no scheme.

Two decode-time weight-precision paths, both resolved at finalize():

* **W8A16 (default)**: class-opted large projections are RE-quantized to fp8-e4m3
  per output channel and the bf16 tensor is DROPPED (freeing half their VRAM to
  the expert slot cache). Decode (M <= ``_W8A16_MAX_BATCH``) runs the fused triton
  kernel (kernel/triton/w8a16_linear.py: no activation quantization at all);
  prefill dequantizes on the fly. Kill switch: FREETOKEN_W8A16_DECODE_LINEAR=0.
* **W8A8 legacy (FREETOKEN_FP8_DECODE_LINEAR=1)**: pre-quantized fp8 copies kept
  BESIDE the bf16 weights, torch._scaled_mm at decode. Superseded by W8A16 where
  that is enabled (W8A16 skips the ~25 us/call activation quantization that made
  W8A8 lose on small shapes -- docs/qwen38_flash_next_optimizations.md), but kept
  as the fallback for classes/shapes W8A16 does not claim.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F

from freetoken.utils import init_logger

from ..registry import LayerKind, register_method
from ..scheme import QuantKind
from .base import LinearKernel, LinearMethod

logger = init_logger(__name__)

# FP8 decode acceleration settings
_FP8_MIN_ELEMENTS = int(os.environ.get("FREETOKEN_FP8_DECODE_MIN_ELEMENTS", "4000000"))
_FP8_MAX_BATCH = int(os.environ.get("FREETOKEN_FP8_DECODE_MAX_BATCH", "8"))
_FP8_ENABLED = os.environ.get("FREETOKEN_FP8_DECODE_LINEAR", "0") == "1"

# W8A16 decode settings (weight-only fp8; the bf16 master copy is dropped)
_W8A16_ENABLED = os.environ.get("FREETOKEN_W8A16_DECODE_LINEAR", "1") == "1"
_W8A16_MIN_ELEMENTS = int(os.environ.get("FREETOKEN_W8A16_MIN_ELEMENTS", "2000000"))
# Keep the ceiling in sync with kernel/triton/w8a16_linear._MAX_DECODE_M: the
# kernel asserts above it, so a larger env value is clamped instead of honored.
_W8A16_KERNEL_MAX_M = 32
_W8A16_MAX_BATCH = min(
    int(os.environ.get("FREETOKEN_W8A16_MAX_BATCH", "32")), _W8A16_KERNEL_MAX_M
)

# One-shot log latches. Activation used to be invisible and the per-call fallback
# silent, so a broken fp8 path was indistinguishable from a disabled one.
_FP8_LOGGED = False
_FP8_FALLBACK_WARNED = False
_W8A16_LOGGED = False

# scale_b is a constant 1.0 and must be a CACHED device tensor: allocating it per
# call is a pageable H2D copy, which is illegal during CUDA graph capture (the
# capture-time exception silently dropped the whole optimization to bf16).
_FP8_ONES: dict[torch.device, torch.Tensor] = {}


def _fp8_ones(device: torch.device) -> torch.Tensor:
    ones = _FP8_ONES.get(device)
    if ones is None:
        ones = torch.tensor(1.0, dtype=torch.float32, device=device)
        _FP8_ONES[device] = ones
    return ones


def fp8_decode_candidate(layer: Any, min_elements: int | None = None) -> bool:
    """Whether a bf16 layer's local weight is an fp8-decode win: class opt-in
    (tall merged column projections) and large enough that the GEMM saving
    exceeds the activation-quantization overhead."""
    if min_elements is None:
        min_elements = _FP8_MIN_ELEMENTS
    if not getattr(layer, "fp8_decode_ok", False):
        return False
    w = getattr(layer, "weight", None)
    return w is not None and w.numel() >= min_elements


def w8a16_decode_candidate(layer: Any, min_elements: int | None = None) -> bool:
    """Whether a bf16 layer's weight should be REPLACED by its per-channel fp8 form.

    Class opt-in (``w8a16_decode_ok``) plus a size floor (tiny GEMVs are launch-
    bound, so halving their bytes does not pay) plus a name guard: routers decide
    the expert set, so their logits must keep full precision even when large.
    """
    if min_elements is None:
        min_elements = _W8A16_MIN_ELEMENTS
    if not getattr(layer, "w8a16_decode_ok", False):
        return False
    prefix = getattr(layer, "prefix", "") or ""
    if prefix.endswith(".gate") or prefix.endswith(".router") or "router" in prefix.split("."):
        return False
    w = getattr(layer, "weight", None)
    return w is not None and w.numel() >= min_elements


class TorchLinearKernel(LinearKernel):
    name = "torch"

    def apply(self, layer: Any, x: torch.Tensor) -> torch.Tensor:
        # W8A16 replaced layers: the bf16 weight is gone; decode takes the fused
        # triton kernel (activation stays bf16), prefill dequantizes on the fly.
        w8 = getattr(layer, "_w8a16_weight", None)
        if w8 is not None:
            from freetoken.kernel.triton.w8a16_linear import (
                dequantize_w8a16,
                w8a16_linear_decode,
            )

            if (
                x.is_cuda
                and x.shape[0] <= _W8A16_MAX_BATCH
                and x.dtype == torch.bfloat16  # the kernel's dot path is bf16-typed
            ):
                return w8a16_linear_decode(x, w8, layer._w8a16_scale, layer.bias)
            return F.linear(x, dequantize_w8a16(w8, layer._w8a16_scale, x.dtype), layer.bias)

        # an fp32 activation stream (DeepSeek-V4's compressors) upcasts the bf16 weight on the fly, as the reference does
        w, b = layer.weight, layer.bias

        # FP8 decode path: use pre-quantized fp8 weight for selected projections at small batch
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
            except Exception as e:
                global _FP8_FALLBACK_WARNED
                if not _FP8_FALLBACK_WARNED:
                    _FP8_FALLBACK_WARNED = True
                    logger.warning_rank0(f"fp8 decode linear fell back to bf16: {e!r}")

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
            scale_b=_fp8_ones(x.device),
            out_dtype=out_dtype,
        )


@register_method(QuantKind.NONE, LayerKind.LINEAR)
class UnquantizedLinearMethod(LinearMethod):
    candidates = (TorchLinearKernel,)

    def create_weights(self, layer: Any) -> None:
        g = self.cfg
        layer.weight = torch.empty(g.out_features, g.in_features)

    def finalize(self, layer: Any) -> None:
        """Pre-quantize selected weights for decode acceleration.

        W8A16 (default): REPLACE the bf16 weight with its per-channel fp8 form and
        free the original -- the VRAM goes straight into the expert slot cache
        (the resolver measures the weights AFTER finalize), and decode reads half
        the HBM bytes. Prefill dequantizes on the fly.
        Legacy W8A8 (FREETOKEN_FP8_DECODE_LINEAR): keep an fp8 COPY beside the
        bf16 weight for torch._scaled_mm at decode.
        """
        w = layer.weight
        if w is None or not w.is_cuda:
            return

        if _W8A16_ENABLED and w8a16_decode_candidate(layer):
            from freetoken.kernel.triton.w8a16_linear import quantize_weight_w8a16

            w8, scale = quantize_weight_w8a16(w)
            layer._w8a16_weight = w8
            layer._w8a16_scale = scale
            layer.weight = None  # drop the bf16 master: decode=kernel, prefill=dequant
            global _W8A16_LOGGED
            if not _W8A16_LOGGED:
                _W8A16_LOGGED = True
                logger.info_rank0(
                    "W8A16 decode linear: replacing projections with >= "
                    f"{_W8A16_MIN_ELEMENTS:,} elements by per-channel fp8 weights "
                    f"(decode batch <= {_W8A16_MAX_BATCH}, bf16 copy freed)"
                )
            return

        if not _FP8_ENABLED:
            return

        if not fp8_decode_candidate(layer):
            return

        # Pre-quantize weight to fp8 (persistent buffer, CUDA-graph compatible)
        amax = w.abs().max()
        scale = (amax / 448.0).clamp(min=1e-12).to(torch.float32)
        w_fp8 = (w / scale).to(torch.float8_e4m3fn)

        layer._fp8_weight = w_fp8
        layer._fp8_scale = scale
        # Warm the cached scale_b while still eager, so no graph capture ever
        # allocates it (a pageable H2D inside a capture is illegal).
        _fp8_ones(w.device)

        global _FP8_LOGGED
        if not _FP8_LOGGED:
            _FP8_LOGGED = True
            logger.info_rank0(
                "fp8 decode linear: pre-quantizing merged column projections with "
                f">= {_FP8_MIN_ELEMENTS:,} elements (decode batch <= {_FP8_MAX_BATCH})"
            )
