"""FP8 KV cache quantization utilities.

Per-(token, head) quantization: one scale per row of ``head_dim`` elements, derived from
that row alone and stored beside the token it describes. The cache outlives the batch that
filled it, so a layer-wide scale recomputed each forward would retroactively rescale every
token an earlier forward had already written.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

FP8_E4M3_MAX = 448.0
# Floor on a row scale. An all-zero (or subnormal) row would give scale == 0 and
# x / scale == NaN, poisoning the whole attention output for the layer. Clamping keeps the
# round-trip exact for zero (0 / eps -> 0 -> 0 * eps) and only costs precision for rows
# whose amax stays below ~448 * MIN_SCALE (~4.5e-4).
MIN_SCALE = 1e-6


@triton.jit
def _quantize_rows_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    ROW: tl.constexpr,
    BLOCK: tl.constexpr,
    FP8_MAX: tl.constexpr,
    MIN_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    mask = lanes < ROW
    offs = row * ROW + lanes
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # Lanes past ROW load as 0.0, so they cannot raise the row amax.
    scale = tl.maximum(tl.max(tl.abs(x), axis=0) / FP8_MAX, MIN_SCALE)
    tl.store(scale_ptr + row, scale)
    tl.store(out_ptr + offs, (x / scale).to(out_ptr.dtype.element_ty), mask=mask)


def quantize_fp8_rows(x: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[T, H * head_dim]`` to fp8 with one scale per ``(token, head)`` row.

    Returns the fp8 tensor in the same ``[T, H * head_dim]`` layout and the ``[T, H]`` fp32
    scales. Amax and quotient come out of a single pass: one row is exactly one block, so
    the scale is known before the divide without reading x twice.
    """
    if x.shape[-1] % head_dim:
        raise ValueError(f"row width {x.shape[-1]} is not a multiple of head_dim {head_dim}")
    assert x.is_contiguous() and x.is_cuda
    num_heads = x.shape[-1] // head_dim
    rows = x.numel() // head_dim
    scales = torch.empty(rows, dtype=torch.float32, device=x.device)
    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    if rows:
        _quantize_rows_kernel[(rows,)](
            x,
            scales,
            out,
            ROW=head_dim,
            BLOCK=triton.next_power_of_2(head_dim),
            FP8_MAX=FP8_E4M3_MAX,
            MIN_SCALE=MIN_SCALE,
        )
    return out, scales.view(-1, num_heads)
