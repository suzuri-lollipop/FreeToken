"""Engram conditional memory for Hy4-flash: hashed bigram/trigram lookups added
to the iHC streams before attention.

Each n-gram order (2..max_ngram) is mixed into ``n_heads`` hash ids by XOR of
``compressed_id * odd_multiplier`` lanes, taken modulo a per-head prime vocabulary;
the ids index one flat table (n-gram-major head offsets). The looked-up vectors
are fp8 round-tripped (as trained), RMS-normed per head, concatenated and projected
into a shared value, then gated per iHC stream by a signed-sqrt dot-product gate.

The hash runs on the HOST (``EngramHash``): it needs the two tokens before each
position, which a decode step no longer forwards. The table itself stays on disk
(``engram_table.py``); the layer only sees staged rows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from freetoken.layers import BaseOP, LinearReplicated, OPList, RMSNorm

if TYPE_CHECKING:
    from freetoken.layers.quantization import QuantConfig

    from .args import HyV4FlashArgs


class BareTensor(BaseOP):
    """A tensor whose state-dict key is the bare prefix (checkpoint stores
    ``unified_per_hc_scales.0`` with no ``.weight`` leaf)."""

    def __init__(self, size: int) -> None:
        self.data = torch.empty(size)

    def forward(self) -> torch.Tensor:
        return self.data

    def state_dict(self, *, prefix: str = "", result=None):
        result = result if result is not None else {}
        result[prefix] = self.data
        return result

    def load_state_dict(self, state_dict, *, prefix: str = "", _internal: bool = False) -> None:
        item = state_dict.pop(prefix)
        assert self.data.shape == item.shape and self.data.dtype == item.dtype, prefix
        self.data = item


def draw_hash_multipliers(seed: int, layer_id_1based: int, max_ngram: int, compressed_vocab_size: int) -> torch.Tensor:
    """The odd multipliers the memory was built with (reference ``draw_hash_multipliers``):
    NumPy's default generator seeded per layer, bounded so a product with any compressed
    id stays within int64."""
    half_bound = max(1, np.iinfo(np.int64).max // compressed_vocab_size // 2)
    generator = np.random.default_rng(seed + 10007 * layer_id_1based)
    drawn = generator.integers(0, half_bound, size=max_ngram, dtype=np.int64)
    return torch.tensor((drawn * 2 + 1).tolist(), dtype=torch.int64)


class EngramHash:
    """Host-side n-gram -> global table row mapping for ONE engram layer."""

    def __init__(self, args: "HyV4FlashArgs", layer_id: int, lookup_table: torch.Tensor) -> None:
        self.max_ngram = args.engram_max_ngram
        self.n_heads = args.engram_n_heads
        self.pad_id = args.engram_pad_id
        self.lookup = lookup_table.to(torch.int64).cpu()
        self.table_size = self.lookup.numel()
        # The reference (_init_weights) redraws from the loaded map bounded by its
        # largest compressed id + 1, not the table length: the released map compresses
        # to fewer ids than the table holds, and the wrong bound draws different
        # multipliers -> wrong hashes -> unreadable memory.
        self.multipliers = draw_hash_multipliers(
            args.engram_seed, layer_id + 1, self.max_ngram, int(self.lookup.max().item()) + 1
        )
        sizes = args.engram_head_vocab_sizes[layer_id]
        assert len(sizes) == (self.max_ngram - 1) * self.n_heads
        self.moduli = torch.tensor(sizes, dtype=torch.int64)
        self.offsets = torch.tensor([0] + list(sizes[:-1]), dtype=torch.int64).cumsum(0)
        self.n_cols = len(sizes)
        self.pad_compressed = int(self.lookup[min(self.pad_id, self.table_size - 1)].item())

    def row_ids(self, tokens: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """``tokens [T]`` with the up-to-(max_ngram-1) preceding ids ``ctx`` -> global rows [T, n_cols]."""
        t = tokens.numel()
        ngram = torch.empty(t, self.max_ngram, dtype=torch.int64)
        ngram[:, 0] = tokens
        # shift the [ctx | tokens] window; positions reaching past the start read pad
        full = torch.cat([ctx, tokens])
        for off in range(1, self.max_ngram):
            start = ctx.numel() - off
            shifted = torch.full((t,), self.pad_id, dtype=torch.int64)
            lo = max(0, start)
            hi = start + t
            if hi > lo:
                shifted[lo - start : hi - start] = full[lo:hi]
            ngram[:, off] = shifted
        compressed = self.lookup[ngram.clamp_max(self.table_size - 1)]
        compressed = torch.where(ngram < self.table_size, compressed, torch.tensor(self.pad_compressed, dtype=torch.int64))
        mixed = compressed * self.multipliers
        ids = []
        for order in range(2, self.max_ngram + 1):
            mix = mixed[:, 0]
            for off in range(1, order):
                mix = mix ^ mixed[:, off]
            start = (order - 2) * self.n_heads
            ids.append(mix.unsqueeze(-1) % self.moduli[start : start + self.n_heads])
        return torch.cat(ids, dim=-1) + self.offsets


class ZeroEngramTable:
    """Dummy-weight stand-in: every lookup reads zeros."""

    def __init__(self, width: int, device: torch.device) -> None:
        self.width = width
        self.device = device

    def stage(self, row_ids: torch.Tensor, *, graph: bool) -> None:
        pass

    def flush(self, *, signal: bool) -> None:
        pass

    def lookup(self, num_tokens: int) -> torch.Tensor:
        return torch.zeros(num_tokens, self.width, dtype=torch.bfloat16, device=self.device)


def fp8_round_trip(x: torch.Tensor) -> torch.Tensor:
    """Per-vector e4m3 quantize-and-back, as the embeddings were trained to be read
    (reference ``_fp8_round_trip``; the forward value is the quantized one)."""
    info = torch.finfo(torch.float8_e4m3fn)
    scale = x.detach().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / info.max
    quantized = (x / scale).clamp(-info.max, info.max).to(torch.float8_e4m3fn).to(x.dtype) * scale
    return quantized


class EngramLayer(BaseOP):
    def __init__(self, args: "HyV4FlashArgs", layer_id: int, quant_config: "QuantConfig | None", *, prefix: str = "") -> None:
        self.layer_id = layer_id
        self.dim = args.dim
        self.hc = args.hc_mult
        self.n_cols = (args.engram_max_ngram - 1) * args.engram_n_heads
        self.head_dim = args.engram_head_dim
        self.width = args.engram_width
        self.fp8_embedding = args.engram_fp8_embedding
        self.lookup_table = torch.empty(args.engram_lookup_size, dtype=torch.int32)
        self.value_proj = LinearReplicated(
            self.width, args.dim, has_bias=True, quant_config=quant_config, prefix=f"{prefix}.value_proj",
        )
        self.key_projs = OPList([
            LinearReplicated(self.width, args.dim, has_bias=True, quant_config=quant_config, prefix=f"{prefix}.key_projs.{i}")
            for i in range(self.hc)
        ])
        # nn.RMSNorm defaults: eps 1e-5, elementwise affine
        self.norm_key = OPList([RMSNorm(args.dim, eps=1e-5) for _ in range(self.hc)])
        self.norm_query = OPList([RMSNorm(args.dim, eps=1e-5) for _ in range(self.hc)])
        self.lookup_per_head_norms = OPList([RMSNorm(self.head_dim, eps=1e-5) for _ in range(self.n_cols)])
        self.unified_pre_gamma = torch.empty(args.dim)
        self.unified_per_hc_scales = OPList([BareTensor(args.dim) for _ in range(self.hc)])
        self._table = None
        self._hash: EngramHash | None = None

    def attach_table(self, table) -> None:
        self._table = table

    def hash(self) -> EngramHash:
        if self._hash is None:
            raise RuntimeError("engram hash not initialized (load_host_tables / prepare_for_runtime)")
        return self._hash

    def init_hash(self, args: "HyV4FlashArgs") -> EngramHash:
        self._hash = EngramHash(args, self.layer_id, self.lookup_table)
        return self._hash

    def forward(self, streams: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``streams [T, hc, dim]``, ``mask [T]`` (positions owning an n-gram) -> the residual to add."""
        t = streams.shape[0]
        emb = self._table.lookup(t).view(t, self.n_cols, self.head_dim)
        if self.fp8_embedding:
            emb = fp8_round_trip(emb)
        emb = torch.stack(
            [norm.forward(emb[:, h]) for h, norm in enumerate(self.lookup_per_head_norms.op_list)], dim=1
        )
        emb = emb.flatten(1)
        gates = []
        for hc in range(self.hc):
            key = self.norm_key.op_list[hc].forward(self.key_projs.op_list[hc].forward(emb))
            query = self.norm_query.op_list[hc].forward(streams[:, hc])
            gate = (key * query).sum(dim=-1) * self.dim**-0.5
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gates.append(gate.sigmoid())
        gates = torch.stack(gates, dim=1).unsqueeze(-1)
        value = self.value_proj.forward(emb)
        main = self.unified_pre_gamma * (
            value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-5)
        ).to(value.dtype)
        branch = torch.stack([s.data * main for s in self.unified_per_hc_scales.op_list], dim=1)
        out = gates * main.unsqueeze(1) + branch
        return out.masked_fill(~mask[:, None, None], 0.0)


__all__ = ["EngramLayer", "EngramHash", "ZeroEngramTable", "BareTensor", "fp8_round_trip", "draw_hash_multipliers"]
