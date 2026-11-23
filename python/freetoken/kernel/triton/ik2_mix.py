"""Fused Inkling-2 depth-stream mix (the BlockAttentionResidual).

``mix`` runs ~120x per decode step (attn-res + ffn-res per visit, plus loop messages);
as plain torch it is a stack, an RMS norm, two einsums and a softmax -- ~14 tiny kernels
and several fp32 memory passes each. One Triton kernel reads the stacked streams once,
RMS-norms them, scores against the query, softmaxes over the streams and writes the
weighted sum, fp32 throughout as the reference. CPU tensors fall back to torch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mix_kernel(
    OUT, STATES, QUERY,
    S: tl.constexpr, Q: tl.constexpr, H: tl.constexpr,
    eps: tl.constexpr, scale: tl.constexpr,
    S_PAD: tl.constexpr, BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)
    q = tl.program_id(1)
    h = tl.arange(0, BLOCK_H)
    s = tl.arange(0, S_PAD)
    hmask = h < H
    smask = s < S
    # states[t, s, h] bf16 -> fp32; rows s >= S load zero
    soff = t.to(tl.int64) * S * H + s[:, None] * H + h[None, :]
    x = tl.load(STATES + soff, mask=smask[:, None] & hmask[None, :], other=0.0).to(tl.float32)
    rms = tl.math.rsqrt(tl.sum(x * x, axis=1) / H + eps)
    xn = x * rms[:, None]
    qv = tl.load(QUERY + q * H + h, mask=hmask, other=0.0).to(tl.float32)
    score = tl.sum(xn * qv[None, :], axis=1) * scale
    score = tl.where(smask, score, -1.0e30)  # padded streams get zero softmax weight
    mx = tl.max(score, axis=0)
    e = tl.exp(score - mx)
    probs = e / tl.sum(e, axis=0)
    out = tl.sum(probs[:, None] * xn, axis=0)
    ooff = t.to(tl.int64) * Q * H + q * H + h
    tl.store(OUT + ooff, out, mask=hmask)


def mix_fused(states: torch.Tensor, query: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``states [T, S, H]`` (bf16) and ``query [Q, H]`` (fp32) -> ``[T, Q, H]`` fp32.

    RMS-norms each stream over H, scores by query/sqrt(H), softmax over S, weighted sum.
    """
    T, S, H = states.shape
    Q = query.shape[0]
    out = torch.empty(T, Q, H, dtype=torch.float32, device=states.device)
    if T == 0:
        return out
    BLOCK_H = triton.next_power_of_2(H)
    S_PAD = triton.next_power_of_2(S)
    _mix_kernel[(T, Q)](
        out, states, query, S, Q, H, eps, H**-0.5,
        S_PAD=S_PAD, BLOCK_H=BLOCK_H, num_warps=8, num_stages=1,
    )
    return out


def mix_ref(blocks, cur, query, eps: float = 1e-6) -> torch.Tensor:
    """The plain-torch reference (also the CPU fallback)."""
    from freetoken.models.inkling2.attention import unweighted_rms_norm

    states = torch.stack(blocks + [cur], dim=1).float()
    states = unweighted_rms_norm(states, eps)
    hidden = states.shape[-1]
    scores = torch.einsum("tsh,qh->tsq", states, query.float()) * hidden**-0.5
    probs = torch.softmax(scores, dim=1)
    return torch.einsum("tsq,tsh->tqh", probs, states)


__all__ = ["mix_fused", "mix_ref"]
