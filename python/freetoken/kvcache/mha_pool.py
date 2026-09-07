from __future__ import annotations

from typing import Sequence

import torch
from freetoken.distributed import get_tp_info
from freetoken.utils import div_even

from .base import BaseKVCachePool

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    ``layer_ids`` lets the pool back only a *subset* of the model's layers while
    callers keep indexing by their global ``layer_id``. Hybrid models (e.g. the
    Qwen3.5 GatedDeltaNet/full-attention stack) interleave linear-attention layers
    that hold no paged KV; passing the full-attention layer ids here allocates one
    storage slab per KV layer (not per model layer) and remaps the global id to its
    dense slot, avoiding a multiple-x over-allocation of unused slabs.

    When ``dtype`` is an FP8 type the pool stores quantized KV with one scale per
    ``(token, kv head)`` pair (``k_scale`` / ``v_scale``). ``store_kv`` quantizes the bf16
    input on the fly and scatters those scales into the slots it wrote; attention backends
    read a token's own scale back when they dequantize it.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        layer_ids: Sequence[int] | None = None,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._num_layers = num_layers
        self._is_fp8 = dtype in _FP8_DTYPES
        if layer_ids is None:
            num_storage_layers = num_layers
            self._layer_map: list[int] | None = None
        else:
            num_storage_layers = len(layer_ids)
            layer_map = [-1] * num_layers
            for dense, global_id in enumerate(layer_ids):
                if global_id < 0 or global_id >= num_layers:
                    raise ValueError(f"KV layer id {global_id} outside [0, {num_layers})")
                layer_map[global_id] = dense
            self._layer_map = layer_map
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
        if self._is_fp8:
            self._k_scales, self._v_scales = self._alloc_scales(
                num_storage_layers, num_pages * page_size, local_kv_heads, device
            )

    @staticmethod
    def _alloc_scales(num_storage_layers, num_slots, local_kv_heads, device):
        # Keyed by SLOT, not layer: a layer-wide scale recomputed each forward would
        # retroactively rescale tokens an earlier forward already wrote. Ones, so an
        # unwritten slot dequantizes its zeroed KV to zero.
        shape = (num_storage_layers, num_slots, local_kv_heads)
        return (
            torch.ones(shape, dtype=torch.float32, device=device),
            torch.ones(shape, dtype=torch.float32, device=device),
        )

    def rebuild(self, num_pages: int) -> None:
        """Reallocate the KV buffer for ``num_pages`` pages IN PLACE.

        Geometry (storage layers, page_size, kv heads, head_dim) is taken from the
        existing buffer; only the page count changes. Views and ``_storage_shape`` are
        refreshed. Object identity is preserved so cached backend references stay valid.
        """
        _, num_storage_layers, _old_pages, page_size, local_kv_heads, head_dim = self._kv_buffer.shape
        dtype = self._kv_buffer.dtype
        device = self._device
        self._k_buffer = None
        self._v_buffer = None
        self._kv_buffer = None
        if self._is_fp8:
            self._k_scales = None
            self._v_scales = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        self._kv_buffer = torch.empty(
            (2, num_storage_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
        if self._is_fp8:
            self._k_scales, self._v_scales = self._alloc_scales(
                num_storage_layers, num_pages * page_size, local_kv_heads, device
            )

    @classmethod
    def kv_cost(cls, config) -> tuple[int, int, int, int]:
        from .base import spec_kv_bytes_per_token

        per_token = sum(
            spec_kv_bytes_per_token(spec, config)
            for spec in config.model_config.kv_cache_group_specs()
            if not spec.is_swa
        )
        return per_token * config.page_size, 0, config.page_size, 0

    def rebuild_from_config(
        self, config, num_pages: int, *, num_swa_pages: int | None = None
    ) -> None:
        self.rebuild(num_pages + 1)  # +1 for the dummy page (matches create_kvcache_pool)

    def unit_bytes(self) -> tuple[int, int]:
        buf = self._kv_buffer
        tokens = int(buf.shape[2]) * int(buf.shape[3])
        per_token = int(buf.numel() * buf.element_size()) // tokens
        if self._is_fp8:
            # The scales are allocated with the slab, so the page-count planner has to see
            # them or it budgets a pool that does not fit. fp32 x 2 (K, V) x heads per token.
            per_token += int(self._k_scales.numel() * self._k_scales.element_size()) * 2 // tokens
        return per_token, 0

    def _dense(self, layer_id: int) -> int:
        if self._layer_map is None:
            return layer_id
        dense = self._layer_map[layer_id]
        if dense < 0:
            raise KeyError(f"layer {layer_id} has no paged KV storage")
        return dense

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[self._dense(index)]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[self._dense(index)]

    def store_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        out_loc: torch.Tensor,
        layer_id: int,
    ) -> None:
        from freetoken.kernel import store_cache

        dense = self._dense(layer_id)
        if not self._is_fp8:
            store_cache(
                k_cache=self._k_buffer[dense].view(self._storage_shape),
                v_cache=self._v_buffer[dense].view(self._storage_shape),
                indices=out_loc,
                k=k,
                v=v,
            )
            return
        from freetoken.kernel.triton.fp8_kv_cache import quantize_fp8_rows

        head_dim = self._storage_shape[2]
        k_fp8, k_scales = quantize_fp8_rows(k, head_dim)
        v_fp8, v_scales = quantize_fp8_rows(v, head_dim)
        store_cache(
            k_cache=self._k_buffer[dense].view(self._storage_shape),
            v_cache=self._v_buffer[dense].view(self._storage_shape),
            indices=out_loc,
            k=k_fp8,
            v=v_fp8,
        )
        # Scatter each row's scale into the slot it landed in, so a later forward dequantizes
        # a token with that token's OWN scale instead of the newest batch's.
        slots = out_loc if out_loc.dtype == torch.int64 else out_loc.to(torch.int64)
        self._k_scales[dense].index_copy_(0, slots, k_scales)
        self._v_scales[dense].index_copy_(0, slots, v_scales)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers

    @property
    def is_fp8(self) -> bool:
        return self._is_fp8

    def k_scale(self, layer_id: int) -> torch.Tensor:
        """Per-slot FP8 scales for K at the given layer: ``[num_slots, num_kv_heads]`` fp32."""
        return self._k_scales[self._dense(layer_id)]

    def v_scale(self, layer_id: int) -> torch.Tensor:
        """Per-slot FP8 scales for V at the given layer: ``[num_slots, num_kv_heads]`` fp32."""
        return self._v_scales[self._dense(layer_id)]
