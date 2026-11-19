"""Identity hyper-connections (iHC) for Hy4-flash.

``hc_mult`` residual streams ``[T, hc, dim]``; each sublayer boundary predicts
PRE gates (mix the streams into the sublayer input) and POST gates (scale the
sublayer output back into every stream) from one fp32 GEMM over the flattened
streams, normalized by the streams' own unweighted RMS. Unlike DeepSeek-V4's mHC
there is no Sinkhorn / comb: the gates are independent sigmoids, the post branch
carries ``hc_magnitude``. The reference forces fp32 throughout; so does this.

Checkpoint names (under ``hc_attn_layer.hc_pre`` / ``hc_mlp_layer.hc_pre``):
``hc_fn [2*hc, hc*dim]``, ``hc_base [2*hc]``, ``hc_scale [2]`` -- all fp32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from freetoken.layers import BaseOP


class IHCPre(BaseOP):
    """One sublayer's hyper-connection: ``forward(streams) -> (post, x)``."""

    def __init__(self, hc_mult: int, dim: int, rms_eps: float, hc_eps: float, magnitude: float) -> None:
        self.hc = hc_mult
        self.dim = dim
        self.rms_eps = rms_eps
        self.hc_eps = hc_eps
        self.magnitude = magnitude
        self.hc_fn = torch.empty(2 * hc_mult, hc_mult * dim, dtype=torch.float32)
        self.hc_base = torch.empty(2 * hc_mult, dtype=torch.float32)
        self.hc_scale = torch.empty(2, dtype=torch.float32)

    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = streams.flatten(1).float()
        inv = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.rms_eps)
        mixes = F.linear(flat, self.hc_fn) * inv
        pre_logits, post_logits = mixes.split(self.hc, dim=-1)
        pre_b, post_b = self.hc_base.split(self.hc)
        pre_scale, post_scale = self.hc_scale.unbind(0)
        pre = torch.sigmoid(pre_logits * pre_scale + pre_b) + self.hc_eps
        post = self.magnitude * torch.sigmoid(post_logits * post_scale + post_b) + self.hc_eps
        x = (pre.unsqueeze(-1) * streams.float()).sum(dim=1)
        return post, x.to(streams.dtype)


class IHCHead(BaseOP):
    """The final stream collapse before the model norm (``model.hc_head``)."""

    def __init__(self, hc_mult: int, dim: int, rms_eps: float, hc_eps: float) -> None:
        self.hc = hc_mult
        self.rms_eps = rms_eps
        self.hc_eps = hc_eps
        self.hc_head_fn = torch.empty(hc_mult, hc_mult * dim, dtype=torch.float32)
        self.hc_head_base = torch.empty(hc_mult, dtype=torch.float32)
        self.hc_head_scale = torch.empty(1, dtype=torch.float32)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        flat = streams.flatten(1).float()
        inv = torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.rms_eps)
        mixes = F.linear(flat, self.hc_head_fn) * inv
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        return (pre.unsqueeze(-1) * streams.float()).sum(dim=1).to(streams.dtype)


def ihc_inject(streams: torch.Tensor, y: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
    """``streams + post * y`` per stream, fp32 as the reference, cast back at the store."""
    return (post.float().unsqueeze(-1) * y.float().unsqueeze(-2) + streams.float()).to(streams.dtype)


__all__ = ["IHCPre", "IHCHead", "ihc_inject"]
