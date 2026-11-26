"""ERNIE 5.5 Flash Preview (``ernie5_5_moe``) support for FreeToken.

280B-total / 14B-active MoE: 43 layers (1 dense + 42 latent MoE), 37 HCA
(Heavily Compressed Attention: window-128 + 128:1 gated-pooling compressed KV,
the DSV4 ratio-128 machinery) + 6 MLA-DSA (token-level top-2048 lightning
indexer over full-history latent KV, trailing window always attended), mHC
residual streams (4, Sinkhorn-20, ERNIE RMS factor), VHA head post-mix, sigmoid
output gates, Multi-Router Latent MoE (512 MXFP4 routed top-10 + 1 shared,
SituGLU), block-fp8 dense projections, multimax lm_head. 1M context (YaRN on
HCA at compress_rope_theta; plain rope on DSA).

The HCA tiers ride the DSV4 paged pool / backend (``dsv4_args`` carries
:class:`Ernie55Args`); the DSA layers add full-history latent + index slabs to
the same pool (:mod:`freetoken.kvcache.dsv4.ernie55_pool`) and are served by
:mod:`freetoken.attention.ernie55_sparse`. Routed experts are W4A8 MXFP4 on the
offload cache (``--moe-strategy offload``). The bundled DSpark draft is not
served yet.
"""

from .args import Ernie55Args, load_args
from .config import model_config_from_args, parse_config
from .gguf import iter_gguf_weights, parse_gguf_config, to_gguf
from .model import Ernie55ForCausalLM
from .weight import iter_expert_pieces, iter_weights

__all__ = [
    "Ernie55Args",
    "Ernie55ForCausalLM",
    "load_args",
    "model_config_from_args",
    "parse_config",
    "iter_weights",
    "iter_expert_pieces",
    "parse_gguf_config",
    "iter_gguf_weights",
    "to_gguf",
]
