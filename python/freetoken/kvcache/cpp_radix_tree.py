"""C++-backed prefix caches: same public API as the three Python radix classes.

``_radix_tree`` (kernel/csrc/radix_tree/radix_tree_ext.cpp) owns the tree; these
facades keep the Python surface the scheduler and the test battery consume:

- ``CppRadixPrefixCache``  <-> ``RadixPrefixCache``  (BasePrefixCache)
- ``CppSWARadixCache``     <-> ``SWARadixCache``
- ``CppHybridRadixCache``  <-> ``HybridRadixCache``

Nodes surface as stable ``NodeRef`` objects (one Python object per live node id,
strongly cached in the C++ registry and pruned on unlink), so ``is``/``id()``
identity, ``node.parent``/``children``/``_key``/``value`` introspection and the
split identity contract behave exactly like the Python ``RadixTreeNode``.

Counters (``evictable_size``, ``full_evictable``, ...) are mirrored as plain
writable Python attributes refreshed after every mutating op -- the battery's
tamper-a-counter self-test keeps working, and the C++ side stays canonical.

``FREETOKEN_RADIX_BACKEND``: ``auto`` (default; C++ when the extension imports,
Python otherwise), ``cpp`` (hard failure when unavailable), ``py``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from freetoken.core import get_global_ctx

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo
from .hybrid_radix_cache import EvictResult, HybridCacheHandle, HybridMatch
from .radix_cache import RadixCacheHandle, _get_key_fn
from .swa_radix_cache import SWACacheHandle, SWAEvictResult, SWAMatch

_EXT = None
_EXT_ERROR: Optional[BaseException] = None
_BACKEND_LOGGED = False


def _log_backend_once(kind: str, backend: str) -> None:
    global _BACKEND_LOGGED
    if _BACKEND_LOGGED:
        return
    _BACKEND_LOGGED = True
    from freetoken.utils import init_logger

    init_logger(__name__).info(
        f"Prefix cache ({kind}): {'C++ _radix_tree' if backend == 'cpp' else 'Python'} backend"
    )


def _load_ext():
    global _EXT, _EXT_ERROR
    if _EXT is None and _EXT_ERROR is None:
        try:
            from freetoken.kernel import _radix_tree

            _EXT = _radix_tree
        except (ImportError, OSError) as exc:  # unbuilt checkout / incompatible ABI
            _EXT_ERROR = exc
    return _EXT


def cpp_radix_available() -> bool:
    return _load_ext() is not None


def radix_backend() -> str:
    """'cpp' or 'py' per FREETOKEN_RADIX_BACKEND (auto|cpp|py, default auto)."""
    want = os.getenv("FREETOKEN_RADIX_BACKEND", "auto").strip().lower()
    if want not in ("auto", "cpp", "py"):
        raise ValueError(f"FREETOKEN_RADIX_BACKEND must be auto|cpp|py, got {want!r}")
    if want == "py":
        return "py"
    if cpp_radix_available():
        return "cpp"
    if want == "cpp":
        raise RuntimeError(f"FREETOKEN_RADIX_BACKEND=cpp but _radix_tree is unavailable: {_EXT_ERROR}")
    return "py"


def _nid(node) -> int:
    """Node identity -> the C++ id64 (accepts a NodeRef or a raw int)."""
    return node.id64 if hasattr(node, "id64") else int(node)


@dataclass(frozen=True)
class CppRadixCacheHandle(RadixCacheHandle):
    """Plain-radix handle whose path concat is ONE C++ call (no per-hop proxy
    walk). Passes isinstance(RadixCacheHandle) for lock_handle/adapter checks."""

    def get_matched_indices(self) -> torch.Tensor:
        return self.node.path_kv()


class _CppTreeMixin:
    """Shared facade plumbing: tree construction, counter mirrors, unwrap helpers."""

    _MODE = 0  # TreeCore mode enum

    def _init_tree(self, device: torch.device, page_size: int | None, window: int = 0) -> None:
        ext = _load_ext()
        if ext is None:
            raise RuntimeError(
                f"_radix_tree extension is not built ({_EXT_ERROR}); "
                "run `python setup.py build_ext --inplace` or use FREETOKEN_RADIX_BACKEND=py"
            )
        self.device = device
        self.page_size = get_global_ctx().page_size if page_size is None else int(page_size)
        self.key_fn = _get_key_fn(self.page_size)
        self._tree = ext.RadixTree(self._MODE, self.page_size, window, str(device))
        self._sync()

    def _sync(self) -> None:
        raise NotImplementedError


class CppRadixPrefixCache(_CppTreeMixin, BasePrefixCache):
    """C++ twin of RadixPrefixCache."""

    _MODE = 0

    def __init__(self, device: torch.device, page_size: int | None = None) -> None:
        super().__init__()
        self._init_tree(device, page_size)
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        self.root_node = self._tree.root_ref()

    def _sync(self) -> None:
        c = self._tree.counters()
        self.evictable_size = c[0]
        self.protected_size = c[1]

    @property
    def root(self):  # adapter parity: hasattr(cache, "root") selects the attribute
        return self.root_node

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        self._tree.lock_plain(_nid(handle.node), unlock)
        self._sync()

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        node, prefix_len = self._tree.match_plain(input_ids)
        return MatchResult(CppRadixCacheHandle(int(prefix_len), node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        prefix_len, insert_len, node = self._tree.insert_plain(input_ids, indices)
        self._sync()
        return InsertResult(int(prefix_len), CppRadixCacheHandle(int(insert_len), node))

    def evict(self, size: int) -> torch.Tensor:
        out = self._tree.evict_plain(size)
        self._sync()
        return out

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=self.evictable_size, protected_size=self.protected_size)

    def check_integrity(self) -> None:
        err = self._tree.check_integrity()
        assert not err, err


class CppSWARadixCache(_CppTreeMixin):
    """C++ twin of SWARadixCache."""

    _MODE = 1

    def __init__(self, device: torch.device, page_size: int, sliding_window_size: int) -> None:
        assert sliding_window_size > 0, "SWARadixCache requires a positive sliding window"
        self.sliding_window_size = sliding_window_size
        self._init_tree(device, page_size, sliding_window_size)
        self.empty = torch.empty(0, dtype=torch.int32, device=device)
        self.root = self._tree.root_ref()

    def _sync(self) -> None:
        c = self._tree.counters()
        self.full_evictable = c[0]
        self.full_protected = c[1]
        self.swa_evictable = c[2]
        self.swa_protected = c[3]
        self._revives = c[6]

    def match_prefix(self, input_ids: torch.Tensor) -> SWAMatch:
        kv, cached_len, node = self._tree.match_swa(input_ids)
        return SWAMatch(kv, int(cached_len), node)

    def insert(
        self, input_ids: torch.Tensor, kv_indices: torch.Tensor,
        swa_evicted_seqlen: int = 0, update_kv_after_len: int = 0,
    ) -> Tuple[int, torch.Tensor]:
        matched, freed = self._tree.insert_swa(
            input_ids, kv_indices, swa_evicted_seqlen, update_kv_after_len)
        self._sync()
        return int(matched), freed

    def inc_lock(self, node) -> Optional[int]:
        uuid = self._tree.inc_lock_swa(_nid(node))
        self._sync()
        return uuid

    def dec_lock(self, node, swa_uuid_for_lock: Optional[int] = None, skip_swa: bool = False) -> None:
        self._tree.dec_lock_swa(_nid(node), swa_uuid_for_lock, skip_swa)
        self._sync()

    def evict_full(self, num_tokens: int) -> SWAEvictResult:
        kv, swa = self._tree.evict_full_swa(num_tokens)
        self._sync()
        return SWAEvictResult(kv, swa)

    def evict_swa(self, num_tokens: int) -> SWAEvictResult:
        kv, swa = self._tree.evict_swa(num_tokens)
        self._sync()
        return SWAEvictResult(kv, swa)

    def trim_head_swa(self, input_ids: torch.Tensor, keep_from: int) -> torch.Tensor:
        out = self._tree.trim_head_swa(input_ids, keep_from)
        self._sync()
        return out

    @property
    def full_evictable_size(self) -> int:
        return self.full_evictable

    @property
    def swa_evictable_size(self) -> int:
        return self.swa_evictable

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=self.full_evictable, protected_size=self.full_protected)

    def check_integrity(self) -> None:
        err = self._tree.check_integrity()
        assert not err, err


class CppHybridRadixCache(_CppTreeMixin):
    """C++ twin of HybridRadixCache."""

    _MODE = 2

    def __init__(self, device: torch.device, page_size: int) -> None:
        from freetoken.kernel.fla.chunk import CHUNK_SIZE

        assert CHUNK_SIZE % page_size == 0, (
            f"hybrid_radix needs CHUNK_SIZE({CHUNK_SIZE}) % page_size({page_size}) == 0"
        )
        self._init_tree(device, page_size)
        self.empty = torch.empty(0, dtype=torch.int32, device=device)
        self.root = self._tree.root_ref()

    def _sync(self) -> None:
        c = self._tree.counters()
        self.full_evictable = c[0]
        self.full_protected = c[1]
        self.mamba_evictable = c[4]
        self.mamba_protected = c[5]

    def match_prefix(self, input_ids: torch.Tensor) -> HybridMatch:
        kv, cached_len, mamba_value, node = self._tree.match_hybrid(input_ids)
        return HybridMatch(kv, int(cached_len), mamba_value, node)

    def insert(
        self, input_ids: torch.Tensor, kv_indices: torch.Tensor, mamba_value: int
    ) -> Tuple[int, bool]:
        prefix_len, mamba_exist = self._tree.insert_hybrid(input_ids, kv_indices, mamba_value)
        self._sync()
        return int(prefix_len), bool(mamba_exist)

    def inc_lock(self, node) -> None:
        self._tree.inc_lock_hybrid(_nid(node))
        self._sync()

    def dec_lock(self, node) -> None:
        self._tree.dec_lock_hybrid(_nid(node))
        self._sync()

    def evict_full(self, num_tokens: int) -> EvictResult:
        kv, mamba = self._tree.evict_full_hybrid(num_tokens)
        self._sync()
        return EvictResult(kv, list(mamba))

    def evict_mamba(self, num: int) -> EvictResult:
        kv, mamba = self._tree.evict_mamba(num)
        self._sync()
        return EvictResult(kv, list(mamba))

    @property
    def full_evictable_size(self) -> int:
        return self.full_evictable

    @property
    def mamba_evictable_size(self) -> int:
        return self.mamba_evictable

    @property
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=self.full_evictable, protected_size=self.full_protected)

    def check_integrity(self) -> None:
        err = self._tree.check_integrity()
        assert not err, err


def make_prefix_cache(kind: str, device: torch.device, page_size: int | None = None,
                      sliding_window_size: int | None = None):
    """Backend-selecting factory for the three radix trees ('radix', 'swa_radix',
    'hybrid_radix'). 'naive' never routes here. Falls back to the Python class
    when the extension is missing (auto) and raises for an explicit cpp request."""
    backend = radix_backend()
    _log_backend_once(kind, backend)
    if kind == "hybrid_radix":
        if backend == "cpp":
            return CppHybridRadixCache(device, page_size)
        from .hybrid_radix_cache import HybridRadixCache

        return HybridRadixCache(device, page_size)
    if kind == "swa_radix":
        if backend == "cpp":
            return CppSWARadixCache(device, page_size, sliding_window_size)
        from .swa_radix_cache import SWARadixCache

        return SWARadixCache(device, page_size, sliding_window_size)
    if kind == "radix":
        if backend == "cpp":
            return CppRadixPrefixCache(device, page_size=page_size)
        from .radix_cache import RadixPrefixCache

        return RadixPrefixCache(device=device, page_size=page_size)
    raise ValueError(f"unknown radix cache kind {kind!r}")


__all__ = [
    "CppRadixPrefixCache",
    "CppSWARadixCache",
    "CppHybridRadixCache",
    "CppRadixCacheHandle",
    "cpp_radix_available",
    "radix_backend",
    "make_prefix_cache",
]
