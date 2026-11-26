# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted for FreeToken from baidu/vLLM-ERNIE (ernie5_5/nvidia/model.py and
# ernie5_5/nvidia/moe.py): the learned multimax logit correction and the
# split-feature router's sigmoid sum.
"""ERNIE 5.5 elementwise ops: multimax logit correction + split-sigmoid routing score.

``multimax`` is the learned piecewise correction the checkpoint applies to the
lm_head logits (fp32): ``y = x + t0*relu(r0-x) + t1*relu(x-r1) +
t2*relu(r2-x)^2 + t3*relu(x-r3)^2``. The kernel keeps the torch expression's
op order with FMA contraction off so each op rounds like eager torch.

``split_sigmoid_sum`` reduces the two split-feature routers' logits
``[T, 2E]`` to ``sigmoid(l0) + sigmoid(l1)`` ``[T, E]`` (fp32, denormals kept).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _relu(x):
    # torch.relu: NaN stays NaN, -0.0 becomes +0.0.
    return tl.where(x <= 0, 0.0, x)


@triton.jit
def _multimax_kernel(logits_ptr, ranges_ptr, ts_ptr, stride, V, BLOCK: tl.constexpr):
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < V
    ptrs = logits_ptr + tl.program_id(0).to(tl.int64) * stride + offs
    x = tl.load(ptrs, mask=mask)
    a = _relu(tl.load(ranges_ptr) - x)
    b = _relu(x - tl.load(ranges_ptr + 1))
    c = _relu(tl.load(ranges_ptr + 2) - x)
    d = _relu(x - tl.load(ranges_ptr + 3))
    # Same op order as the torch expression in multimax_logits.
    y = x + tl.load(ts_ptr) * a
    y = y + tl.load(ts_ptr + 1) * b
    y = y + tl.load(ts_ptr + 2) * (c * c)
    y = y + tl.load(ts_ptr + 3) * (d * d)
    tl.store(ptrs, y, mask=mask)


def multimax_logits(
    logits: torch.Tensor, ranges: torch.Tensor, ts: torch.Tensor
) -> torch.Tensor:
    """Apply the learned multimax correction; fp32 CUDA logits are updated in place.

    ``ranges`` / ``ts`` are fp32 ``[4]`` (the reader upcasts the checkpoint's bf16).
    Non-CUDA / non-fp32 inputs take the eager torch path (a fresh tensor).
    """
    if (
        logits.is_cuda
        and logits.dtype == torch.float32
        and logits.dim() == 2
        and logits.stride(1) == 1
    ):
        num_tokens, vocab = logits.shape
        if num_tokens and vocab:
            block = 2048
            _multimax_kernel[(num_tokens, triton.cdiv(vocab, block))](
                logits, ranges, ts, logits.stride(0), vocab,
                BLOCK=block, num_warps=4,
                # No FMA contraction, so each op rounds like torch's.
                enable_fp_fusion=False,
            )
        return logits
    r, t = ranges, ts
    return (
        logits
        + t[0] * (r[0] - logits).relu()
        + t[1] * (logits - r[1]).relu()
        + t[2] * (r[2] - logits).relu().square()
        + t[3] * (logits - r[3]).relu().square()
    )


@triton.jit
def _sigmoid(x):
    # torch's fp32 CUDA sigmoid: 1 / (1 + expf(-x)), IEEE division.
    return tl.div_rn(1.0, 1.0 + libdevice.exp(-x))


@triton.jit
def _split_sigmoid_sum_kernel(logits_ptr, out_ptr, stride, E: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, E)
    l0 = tl.load(logits_ptr + row * stride + offs)
    l1 = tl.load(logits_ptr + row * stride + E + offs)
    tl.store(out_ptr + row * E + offs, _sigmoid(l0) + _sigmoid(l1))


def split_sigmoid_sum(logits: torch.Tensor, num_experts: int) -> torch.Tensor:
    """``sigmoid(logits[:, :E]) + sigmoid(logits[:, E:])`` in one kernel (fp32)."""
    num_tokens = logits.shape[0]
    out = torch.empty(num_tokens, num_experts, dtype=torch.float32, device=logits.device)
    if (
        num_tokens
        and logits.is_cuda
        and logits.dtype == torch.float32
        and logits.stride(1) == 1
        and triton.next_power_of_2(num_experts) == num_experts
    ):
        _split_sigmoid_sum_kernel[(num_tokens,)](
            logits, out, logits.stride(0), E=num_experts,
            # Keep denormal results, as torch does.
            enable_reflect_ftz=False,
        )
    elif num_tokens:
        torch.add(
            logits[:, :num_experts].sigmoid(),
            logits[:, num_experts:].sigmoid(),
            out=out,
        )
    return out


__all__ = ["multimax_logits", "split_sigmoid_sum"]
