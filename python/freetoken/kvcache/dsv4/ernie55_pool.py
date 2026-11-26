"""ERNIE 5.5 paged KV pool: DSV4 window+compressed tiers + token-level DSA slabs.

ERNIE's 37 HCA layers ARE DSV4 ratio-128 attention (window ring + compressed
blocks + compress-state ring), so the pool subclasses :class:`DSV4PagedKVCache`
and inherits its allocation, full-loc mapping and rebuild surface. The six DSA
layers add two full-history slabs each, addressed ARITHMETICALLY by full loc
(like the compressed tier at ratio 1 -- no separate allocator):

* ``dsa_latent[L]`` -- ``[full_token, kv_lora_rank + qk_rope_head_dim]`` bf16,
  the MLA latent ``ckv | kpe`` row per token (K == V absorption: values are the
  ckv half, the rope tail only scores).
* ``dsa_index[L]``  -- ``[full_token, index_head_dim]`` bf16, the lightning
  indexer's key per token.

Both ride the shared page table: a token's full loc IS its row, so page
sharing / eviction / radix reuse cover them for free, and ``rebuild`` resizes
every slab atomically with the DSV4 tiers.

The cost model adds the DSA bytes/token on top of the DSV4 per-tier model; the
window tier still allocates on every layer (DSA layers never write it -- the
6/43 window overhead is kept so allocation and cost stay in lockstep).
"""

from __future__ import annotations

from typing import ClassVar

import torch

from freetoken.utils import init_logger

from ..format import NONE, ROW_SCALE_FORMATS, KVFormat
from .v4_cost_model import (
    DSV4PoolSizes,
    _dsv4_pool_sizes,
    _dsv4_swa_ratio,
    _dsv4_window_floor_pages,
    dsv4_auto_cost_model,
    dsv4_cache_per_page,
    dsv4_kv_unit_bytes,
    dsv4_pool_bytes,
    dsv4_pool_sizes,
    dsv4_solve_num_pages,
    dsv4_window_unit_bytes,
)
from .v4_pool import DSV4PagedKVCache

logger = init_logger(__name__)


# ---- cost model: DSV4 tiers + the full-history DSA slabs ------------------------------


def ernie55_dsa_bytes_per_token(args, kv_format: KVFormat = NONE) -> int:
    """Latent + index rows per full-history token, summed over DSA layers. The latent
    slab carries ``kv_format`` (int8 codes + two fp32 row scales when quantized); the
    index slab stays bf16 (the lightning-indexer scoring reads it unquantized)."""
    if kv_format.quantized:
        latent = kv_format.row_bytes(args.mla_latent_dim, torch.bfloat16, segments=2)
    else:
        latent = args.mla_latent_dim * 2
    return len(args.dsa_layer_ids) * (latent + args.index_head_dim * 2)


def ernie55_cache_per_page(args, swa_ratio: float, P: int = 128, kv_format: KVFormat = NONE) -> int:
    return dsv4_cache_per_page(args, swa_ratio, P) + ernie55_dsa_bytes_per_token(args, kv_format) * P


def ernie55_pool_bytes(sizes: DSV4PoolSizes, args, n_scratch: int = 1, kv_format: KVFormat = NONE) -> int:
    return dsv4_pool_bytes(sizes, args, n_scratch) + ernie55_dsa_bytes_per_token(args, kv_format) * sizes.full_token


def ernie55_solve_num_pages(
    available_bytes: int, args, swa_ratio: float, floor_win_pages: int,
    P: int = 128, n_scratch: int = 1, kv_format: KVFormat = NONE,
) -> DSV4PoolSizes:
    """Largest budget-respecting pool (the dsv4 binary search with the DSA bytes)."""
    def _sizes(num: int) -> DSV4PoolSizes:
        win = max(floor_win_pages, (round(swa_ratio * num * P) + P - 1) // P)
        return dsv4_pool_sizes(num, args, swa_ratio, P=P, n_win_pages=win)

    lo = max(floor_win_pages, 2)
    if ernie55_pool_bytes(_sizes(lo), args, n_scratch, kv_format) > available_bytes:
        raise ValueError(
            f"ERNIE 5.5 KV budget {available_bytes} bytes cannot fit the minimal pool "
            f"({lo} pages incl. the window working-set floor {floor_win_pages}); "
            "raise memory_ratio or lower max_running_req/max_seq_len"
        )
    hi = max(lo, available_bytes // max(1, ernie55_cache_per_page(args, 0.0, P, kv_format)))
    while ernie55_pool_bytes(_sizes(hi), args, n_scratch, kv_format) <= available_bytes:
        hi *= 2
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if ernie55_pool_bytes(_sizes(mid), args, n_scratch, kv_format) <= available_bytes:
            lo = mid
        else:
            hi = mid
    return _sizes(lo)


def ernie55_auto_cost_model(args, swa_ratio, floor_win_pages, P=128, n_scratch=1, kv_format: KVFormat = NONE):
    per_page, fixed, min_reserve = dsv4_auto_cost_model(args, swa_ratio, floor_win_pages, P, n_scratch)
    return per_page + ernie55_dsa_bytes_per_token(args, kv_format) * P, fixed, min_reserve


def ernie55_kv_unit_bytes(args, P: int = 128, kv_format: KVFormat = NONE) -> int:
    return dsv4_kv_unit_bytes(args, P) + ernie55_dsa_bytes_per_token(args, kv_format)


def _ernie55_pool_sizes(config, num_pages: int, num_swa_pages: int | None = None):
    # Window sizing is the DSV4 policy verbatim; the DSA slabs size off full_token.
    return _dsv4_pool_sizes(config, num_pages, num_swa_pages=num_swa_pages)


class Ernie55PagedKVCache(DSV4PagedKVCache):
    # The full-history DSA latent slab carries a row-scale KV format (int8/fp8); the
    # bounded HCA window/compressed tiers and the DSA index slab stay bf16.
    kv_formats: ClassVar[frozenset[str]] = ROW_SCALE_FORMATS

    def __init__(
        self,
        sizes: DSV4PoolSizes,
        args,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        P: int = 128,
        n_scratch: int = 1,
        kv_format: KVFormat = NONE,
    ) -> None:
        self.kv_format = kv_format
        self.dsa_layer_ids = tuple(args.dsa_layer_ids)
        self.dsa_latent_dim = args.mla_latent_dim
        self.dsa_index_head_dim = args.index_head_dim
        super().__init__(sizes, args, device, dtype, P=P, n_scratch=n_scratch)

    def _alloc_buffers(self) -> None:
        super()._alloc_buffers()
        # Full-history DSA slabs: row == full loc (arithmetic, no allocator). The latent
        # slab stores kv_format codes (+ a per-row ckv/kpe scale pair); index stays bf16.
        fmt = self.kv_format
        self.dsa_latent: dict[int, torch.Tensor] = {
            L: torch.zeros(
                self.sizes.full_token, fmt.elem_dim(self.dsa_latent_dim),
                device=self._device, dtype=fmt.elem_tensor_dtype(self._dtype),
            )
            for L in self.dsa_layer_ids
        }
        self._dsa_latent_scale: dict[int, torch.Tensor] | None = None
        if fmt.scale_dtype is not None:
            self._dsa_latent_scale = {
                L: torch.zeros(self.sizes.full_token, 2, device=self._device, dtype=fmt.scale_store_dtype)
                for L in self.dsa_layer_ids
            }
        self.dsa_index: dict[int, torch.Tensor] = {
            L: torch.zeros(
                self.sizes.full_token, self.dsa_index_head_dim,
                device=self._device, dtype=self._dtype,
            )
            for L in self.dsa_layer_ids
        }

    def _drop_buffers(self) -> None:
        super()._drop_buffers()
        self.dsa_latent = self.dsa_index = self._dsa_latent_scale = None

    # ----- DSA reads / writes (addressed by full loc) -----
    def dsa_latent_rows(self, layer_id: int) -> torch.Tensor:
        return self.dsa_latent[layer_id]

    def dsa_latent_scale(self, layer_id: int) -> torch.Tensor | None:
        return None if self._dsa_latent_scale is None else self._dsa_latent_scale[layer_id]

    def dsa_index_rows(self, layer_id: int) -> torch.Tensor:
        return self.dsa_index[layer_id]

    def store_dsa_kv(self, c_kv: torch.Tensor, k_rope: torch.Tensor, out_loc: torch.Tensor, layer_id: int) -> None:
        rows = self.dsa_latent[layer_id]
        if self.kv_format.quantized:
            from freetoken.kernel.triton.kv_rows import store_latent_rows

            store_latent_rows(self.kv_format, c_kv, k_rope, out_loc, rows, self._dsa_latent_scale[layer_id])
            return
        split = rows.shape[1] - k_rope.shape[-1]
        rows[out_loc, :split] = c_kv
        rows[out_loc, split:] = k_rope

    def store_dsa_index_k(self, k: torch.Tensor, out_loc: torch.Tensor, layer_id: int) -> None:
        self.dsa_index[layer_id][out_loc] = k

    def total_bytes(self) -> int:
        n = super().total_bytes()
        n += sum(t.numel() * t.element_size() for t in self.dsa_latent.values())
        n += sum(t.numel() * t.element_size() for t in self.dsa_index.values())
        if self._dsa_latent_scale is not None:
            n += sum(t.numel() * t.element_size() for t in self._dsa_latent_scale.values())
        return int(n)

    # ----- engine-facing sizing (the DSV4 surface with the DSA bytes) -----
    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        args = config.model_config.dsv4_args
        P = args.window_size
        floor = _dsv4_window_floor_pages(config, P)
        per_page, fixed, min_reserve_tokens = ernie55_auto_cost_model(
            args, _dsv4_swa_ratio(config), floor, P=P, n_scratch=config.max_running_req + 1,
            kv_format=getattr(config, "kv_format", NONE),
        )
        return per_page, fixed, config.page_size, min_reserve_tokens

    @classmethod
    def solve_num_pages(cls, config, available_memory: int) -> int:
        from freetoken.utils import mem_GB

        args = config.model_config.dsv4_args
        P = args.window_size
        num_pages = config.num_page_override
        if num_pages is None:
            sizes = ernie55_solve_num_pages(
                available_memory, args, _dsv4_swa_ratio(config),
                floor_win_pages=_dsv4_window_floor_pages(config, P), P=P,
                n_scratch=config.max_running_req + 1,
                kv_format=getattr(config, "kv_format", NONE),
            )
            num_pages = sizes.full_token // P - 1
        else:
            floor = _dsv4_window_floor_pages(config, P)
            if num_pages < floor:
                raise ValueError(
                    f"--num-pages {num_pages} ({num_pages * P} tokens) is below the ERNIE 5.5 "
                    f"window working-set floor {floor} pages ({floor * P} tokens); raise "
                    f"--num-pages or lower max_running_req/max_seq_len"
                )
            sizes = _ernie55_pool_sizes(config, num_pages + 1)
        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        real = ernie55_pool_bytes(sizes, args, config.max_running_req + 1, getattr(config, "kv_format", NONE))
        logger.info(
            f"Allocating {num_pages * P} tokens for ERNIE 5.5 KV cache "
            f"({sizes.n_win_pages} window pages, {len(args.dsa_layer_ids)} DSA slabs), "
            f"total = {mem_GB(real)}"
        )
        return num_pages

    @classmethod
    def window_spec(cls, config):
        from ..base import WindowPoolSpec

        P = config.model_config.dsv4_args.window_size
        return WindowPoolSpec(P, _dsv4_window_floor_pages(config, P) - 1)

    @classmethod
    def min_kv_tokens(cls, config) -> int:
        P = config.model_config.dsv4_args.window_size
        return _dsv4_window_floor_pages(config, P) * P

    def validate_rebuild(
        self, config, *, num_pages: int | None, target_moe: int, per_expert_bytes: int,
        baseline_free: int, weights_bytes: int, current_num_pages: int,
        extra_fixed_bytes: int = 0, extra_note: str = "",
        num_swa_pages: int | None = None,
        device_total: int = 0, nonpool_overhead_bytes: int = 0, **targets,
    ) -> None:
        from freetoken.engine.cache_budget import net_cache_budget_bytes
        from freetoken.utils import mem_GB

        from ..base import CacheRebuildRejected

        args = config.model_config.dsv4_args
        if num_pages is not None:
            floor = _dsv4_window_floor_pages(config, args.window_size)
            if num_pages < floor:
                raise CacheRebuildRejected(
                    f"num_pages {num_pages} is below the ERNIE 5.5 window working-set floor {floor} "
                    f"(max_running_req={config.max_running_req}); admission would deadlock"
                )
        if num_pages is not None or num_swa_pages is not None:
            target_pages = num_pages if num_pages is not None else current_num_pages
            kv_sizes = _ernie55_pool_sizes(config, target_pages + 1, num_swa_pages=num_swa_pages)
        else:
            kv_sizes = self.sizes
        budget = net_cache_budget_bytes(
            config.memory_ratio, baseline_free, weights_bytes, 0,
            device_total=device_total, nonpool_overhead_bytes=nonpool_overhead_bytes,
        )
        need = target_moe * per_expert_bytes + ernie55_pool_bytes(
            kv_sizes, args, config.max_running_req + 1
        )
        if need > budget:
            kv_part = f"kv={num_pages} P-pages" if num_pages is not None else "kv=current pool"
            raise CacheRebuildRejected(
                f"requested cache (moe={target_moe} slots, {kv_part}) needs "
                f"{mem_GB(need)} > budget {mem_GB(budget)}; old cache kept, still serving"
            )

    def rebuild_from_config(self, config, num_pages: int, *, num_swa_pages: int | None = None) -> None:
        self.rebuild(_ernie55_pool_sizes(config, num_pages + 1, num_swa_pages=num_swa_pages))

    def unit_bytes(self) -> tuple[int, int]:
        kv, swa = super().unit_bytes()
        return kv + ernie55_dsa_bytes_per_token(self.args), swa


__all__ = ["Ernie55PagedKVCache", "ernie55_pool_bytes", "ernie55_dsa_bytes_per_token"]
