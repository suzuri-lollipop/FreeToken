"""Compatibility shim: the canonical GDN op moved to ``freetoken.layers.linear_attn.gdn``
(upstream PR #841 deleted the per-model copies). Kept only for importers of the old
per-model path (TP shard math tests and the qwen3_5 weight tests); new code imports
the shared op directly."""
from freetoken.layers.linear_attn.gdn import GatedDeltaNet, gdn_local_dims

# The qwen3_5 op is the shared op with its default silu output gate.
Qwen3_5GatedDeltaNet = GatedDeltaNet

__all__ = ["Qwen3_5GatedDeltaNet", "gdn_local_dims"]
