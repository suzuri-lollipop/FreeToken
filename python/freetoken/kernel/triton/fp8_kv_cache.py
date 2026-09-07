"""FP8 KV cache quantization utilities.

Per-tensor dynamic quantization: scale = max_abs(x) / FP8_MAX, then x_fp8 = x / scale.
The scale is computed fresh each forward pass from the bf16 K/V input.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8_E4M3_MAX = 448.0
# Floor on the per-tensor scale. An all-zero (or subnormal) K/V tile would give
# scale == 0 and x / scale == NaN, poisoning the whole attention output for the
# layer. Clamping keeps the round-trip exact for zero (0 / eps -> 0 -> 0 * eps)
# and only costs precision for activations below ~448 * MIN_SCALE (~4.5e-4).
MIN_SCALE = 1e-6


@triton.jit
def _compute_scale_kernel(
    x_ptr,
    scale_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    MIN_SCALE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    abs_x = tl.abs(x.to(tl.float32))
    local_max = tl.max(abs_x, axis=0)
    # Per-block floor so the atomic_max result is never 0 (see MIN_SCALE above).
    local_scale = tl.maximum(local_max / FP8_MAX, MIN_SCALE)
    tl.atomic_max(scale_ptr, local_scale)


def compute_fp8_scale(x: torch.Tensor, scale_out: torch.Tensor) -> None:
    """Compute max(max_abs(x) / FP8_E4M3_MAX, MIN_SCALE) into scale_out (scalar tensor).

    Uses atomic_max so multiple blocks can contribute. Caller must zero-initialize scale_out.
    The result is clamped to MIN_SCALE so quantize_fp8 never divides by zero.
    """
    n = x.numel()
    BLOCK = 4096
    grid = (triton.cdiv(n, BLOCK),)
    _compute_scale_kernel[grid](
        x, scale_out, n, BLOCK_SIZE=BLOCK, FP8_MAX=FP8_E4M3_MAX, MIN_SCALE=MIN_SCALE
    )
    if n == 0:
        # No blocks ran; scale_out keeps its caller-zeroed value. Floor it here.
        scale_out.clamp_(min=MIN_SCALE)


def quantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize bf16 tensor to fp8_e4m3 using the given scale. Returns fp8 tensor."""
    return (x / scale.to(x.dtype)).to(torch.float8_e4m3fn)

