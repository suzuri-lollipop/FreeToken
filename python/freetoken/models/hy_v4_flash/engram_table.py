"""Engram tables streamed from the checkpoint shards (bf16 rows, no scales).

Each Hy4-flash engram layer owns a ``[165M, 96]`` bf16 table (~31.7 GB). The table
stays where the checkpoint put it: ``EngramDiskTable`` maps the shard tensor as one
``RowStore`` extent and reads the 16 rows a token needs per fill (io_uring, O_DIRECT),
staging them into pinned memory the device copies from. ``EngramDenseTable`` keeps a
(small) table resident on the device for tests and tiny checkpoints.

Protocol (the DeepSeek-V4.1 engram / Qwen3.8 PLE disk table's): the engine wraps every
dispatch in ``EngramHost.forward_host_ctx``. Eager: hash + stage + read before the
launch. CUDA-graph decode: the sampled token is read back, a worker thread hashes it
against the dispatch snapshot, fills the graph's pinned staging and releases a
per-table flag the captured ``lookup`` WAITs on (or fills before the replay without
stream memops). See deepseek_v41/engram_table.py for the full rationale.
"""

from __future__ import annotations

import json
import os
import struct
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import torch
from freetoken.core import Batch
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.utils import init_logger

from .engram import EngramHash

logger = init_logger(__name__)


@dataclass(frozen=True)
class _TensorExtent:
    path: str
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str


def _tensor_extent(folder: str, name: str) -> _TensorExtent:
    from freetoken.models.loader import safetensors_weight_map

    path = os.path.join(folder, safetensors_weight_map(folder)[name])
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    meta = header[name]
    start, end = meta["data_offsets"]
    return _TensorExtent(path, 8 + n + start, end - start, tuple(meta["shape"]), meta["dtype"])


def engram_row_source(folder: str, layer_id: int) -> _TensorExtent:
    """Locate a layer's bf16 embedding table in the checkpoint directory."""
    weight = _tensor_extent(folder, f"model.layers.{layer_id}.engram_layer.multi_head_embedding.embedding.weight")
    if weight.dtype != "BF16":
        raise ValueError(f"engram table of layer {layer_id} is {weight.dtype}, expected BF16")
    return weight


class EngramDiskTable:
    """One layer's table: stages ``[T, n_cols]`` bf16 rows into pinned memory on the
    host; the device copies them in. Implements the ``stage / flush / lookup`` protocol."""

    def __init__(
        self, source: _TensorExtent, n_cols: int, device: torch.device, *, max_graph_rows: int, max_extend_tokens: int,
        use_io_uring: bool = True,
    ) -> None:
        from freetoken.kernel.row_store import RowStore

        rows, head_dim = source.shape
        self.head_dim = head_dim
        self.n_cols = n_cols
        self.num_rows = rows
        self.row_bytes = head_dim * 2  # bf16
        self.device = device
        self.store = RowStore(
            paths=[source.path], extent_file=[0], extent_base=[source.offset], rows_per_extent=rows,
            row_bytes=self.row_bytes, row_stride=self.row_bytes, use_io_uring=use_io_uring,
        )
        self.max_graph_rows = max_graph_rows
        self._graph_pinned = alloc_pinned_tensor(max_graph_rows * n_cols * self.row_bytes, dtype=torch.uint8)
        self._graph_pinned.zero_()
        self._graph_dev = torch.empty(max_graph_rows * n_cols * self.row_bytes, dtype=torch.uint8, device=device)
        self.graph_consumed = torch.cuda.Event()
        self._eager_pinned: list[torch.Tensor] = []
        self._eager_read = [torch.cuda.Event(), torch.cuda.Event()]
        self._eager_slot = 0
        self._eager_dev = None
        self._grow_eager(max_extend_tokens)
        self.flag = None
        self.wait_sync = False

    def _grow_eager(self, num_tokens: int) -> None:
        nbytes = num_tokens * self.n_cols * self.row_bytes
        if self._eager_pinned and self._eager_pinned[0].numel() >= nbytes:
            return
        for event in self._eager_read:
            event.synchronize()
        self._eager_pinned = [alloc_pinned_tensor(nbytes, dtype=torch.uint8) for _ in self._eager_read]
        for buf in self._eager_pinned:
            buf.zero_()
        self._eager_dev = torch.empty(nbytes, dtype=torch.uint8, device=self.device)

    def stage(self, row_ids: torch.Tensor, *, graph: bool) -> None:
        n = row_ids.numel()
        if graph:
            self.graph_consumed.synchronize()
            pinned = self._graph_pinned
            assert n <= self.max_graph_rows * self.n_cols
        else:
            self._grow_eager(n // self.n_cols)
            self._eager_slot ^= 1
            self._eager_read[self._eager_slot].synchronize()
            pinned = self._eager_pinned[self._eager_slot]
        ids = row_ids.reshape(-1).to(torch.int64).contiguous()
        rows = pinned[: n * self.row_bytes].view(n, self.row_bytes)
        self.store.stage_rows(ids.data_ptr(), n, rows.data_ptr(), self.row_bytes)

    def flush(self, *, signal: bool) -> None:
        self.store.flush(self.flag.data_ptr() if signal else 0)

    def lookup(self, num_tokens: int) -> torch.Tensor:
        from freetoken.kernel.row_store import wait_reset

        stream = torch.cuda.current_stream(self.device)
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and self.wait_sync:
            wait_reset(stream, self.flag)
        pinned, dev = (self._graph_pinned, self._graph_dev) if capturing else (self._eager_pinned[self._eager_slot], self._eager_dev)
        n = num_tokens * self.n_cols
        nbytes = n * self.row_bytes
        dev[:nbytes].copy_(pinned[:nbytes], non_blocking=True)
        if not capturing:
            self._eager_read[self._eager_slot].record(stream)
        return dev[:nbytes].view(torch.bfloat16).view(num_tokens, self.n_cols * self.head_dim)


class EngramDenseTable:
    """A device-resident table (tests / tiny checkpoints): the host stages row IDS,
    the captured lookup gathers them from the resident embedding."""

    def __init__(self, embedding: torch.Tensor, n_cols: int, device: torch.device, *, max_graph_rows: int, max_extend_tokens: int) -> None:
        self.embedding = embedding.to(device)
        self.n_cols = n_cols
        self.head_dim = embedding.shape[1]
        self.device = device
        self.max_graph_rows = max_graph_rows
        self._graph_ids = torch.zeros(max_graph_rows * n_cols, dtype=torch.int64, device=device)
        self._eager_ids = torch.zeros(max(max_extend_tokens, 1) * n_cols, dtype=torch.int64, device=device)
        self.graph_consumed = torch.cuda.Event()
        self.flag = None
        self.wait_sync = False

    def stage(self, row_ids: torch.Tensor, *, graph: bool) -> None:
        ids = row_ids.reshape(-1).to(torch.int64)
        if graph:
            self.graph_consumed.synchronize()
            assert ids.numel() <= self._graph_ids.numel()
            self._graph_ids[: ids.numel()].copy_(ids.to(self.device, non_blocking=True))
        else:
            if ids.numel() > self._eager_ids.numel():
                self._eager_ids = torch.zeros(ids.numel(), dtype=torch.int64, device=self.device)
            self._eager_ids[: ids.numel()].copy_(ids.to(self.device, non_blocking=True))

    def flush(self, *, signal: bool) -> None:
        pass

    def lookup(self, num_tokens: int) -> torch.Tensor:
        n = num_tokens * self.n_cols
        capturing = torch.cuda.is_current_stream_capturing()
        ids = (self._graph_ids if capturing else self._eager_ids)[:n]
        return self.embedding.index_select(0, ids).view(num_tokens, self.n_cols * self.head_dim)


def _context(ids: torch.Tensor, position: int, width: int) -> torch.Tensor:
    return ids[max(0, position - width) : position]


class EngramHost:
    """Per-model coordinator: hashes each batch's token runs and fills every layer's table."""

    def __init__(self, hashes: Sequence[EngramHash], tables: Sequence, device: torch.device, *, sync_mode: str = "auto") -> None:
        from freetoken.kernel.row_store import probe_wait_sync

        self.hashes = list(hashes)
        self.tables = list(tables)
        self.device = device
        self.wait_sync = probe_wait_sync(sync_mode, device)
        for t in self.tables:
            t.wait_sync = self.wait_sync
            if self.wait_sync and getattr(t, "flag", None) is None and hasattr(t, "store"):
                from freetoken.kernel.row_store import alloc_sync_flag

                with torch.cuda.device(device):
                    t.flag = alloc_sync_flag()
        graph_cap = min(getattr(t, "max_graph_rows", 1) for t in self.tables)
        self._token_readback = alloc_pinned_tensor(graph_cap, dtype=torch.int32)
        self._readback_event = torch.cuda.Event()
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hy4-engram-fill")
        self._pending: Future | None = None
        backend = self.tables[0].store.io_backend() if hasattr(self.tables[0], "store") else "dense"
        logger.info_rank0(f"Hy4 engram backend: {backend}, {'wait-sync' if self.wait_sync else 'launch-gating'}")

    def close(self) -> None:
        self._worker.shutdown(wait=True)

    def __del__(self) -> None:
        worker = getattr(self, "_worker", None)
        if worker is not None:
            worker.shutdown(wait=False)

    def fill(self, runs: Sequence[tuple[torch.Tensor, torch.Tensor]], *, graph: bool) -> None:
        """``runs``: per request ``(context ids, token ids)`` in batch order."""
        for l, table in enumerate(self.tables):
            h = self.hashes[l]
            ids = torch.cat([h.row_ids(tokens, ctx) for ctx, tokens in runs]) if runs else torch.empty(0, h.n_cols, dtype=torch.int64)
            table.stage(ids, graph=graph)
            table.flush(signal=graph and self.wait_sync)

    @torch.inference_mode()
    def _deferred_fill(self, bs: int, histories: list[tuple[int, ...]]) -> None:
        try:
            self._readback_event.synchronize()
            tokens = self._token_readback[:bs].to(torch.int64)
            runs = [(torch.tensor(hist, dtype=torch.int64), tokens[i : i + 1]) for i, hist in enumerate(histories)]
            self.fill(runs, graph=True)
        except BaseException:
            from freetoken.kernel.row_store import signal

            for t in self.tables:
                if getattr(t, "flag", None) is not None:
                    signal(t.flag)
            raise

    def _await_pending(self) -> None:
        if self._pending is not None:
            pending, self._pending = self._pending, None
            pending.result()

    def host_fill_batch(self, batch: Batch, use_graph: bool) -> None:
        self._await_pending()
        width = self.hashes[0].max_ngram - 1
        if batch.is_decode:
            reqs = list(batch.padded_reqs)
            histories = [tuple(_context(r.input_ids, r.device_len - 1, width).tolist()) for r in reqs]
            if use_graph and self.wait_sync:
                bs = batch.padded_size
                self._token_readback[:bs].copy_(batch.input_ids, non_blocking=True)
                self._readback_event.record(torch.cuda.current_stream(self.device))
                self._pending = self._worker.submit(self._deferred_fill, bs, histories)
                return
            tokens = batch.input_ids.to("cpu").to(torch.int64)
            runs = [(torch.tensor(hist, dtype=torch.int64), tokens[i : i + 1]) for i, hist in enumerate(histories)]
            self.fill(runs, graph=use_graph)
            return
        if getattr(batch, "phase", None) == "verify":
            # multi-row verify (DFly): m query rows per request, each with its n-gram
            # context; stage bs*m rows to match the engram lookup's num_tokens=bs*m.
            reqs = list(batch.padded_reqs)
            runs = []
            for r in reqs:
                m = r.extend_len
                cached = r.device_len - m
                for j in range(m):
                    pos = cached + j
                    runs.append((_context(r.input_ids, pos, width), r.input_ids[pos : pos + 1]))
            self.fill(runs, graph=use_graph)
            return
        # prefill: this chunk's per-request ranges off the DSA metadata (no segments field)
        md = batch.attn_metadata
        qo = md.qo_indptr_cpu.tolist()
        kv_lens = md.kv_len_cpu.tolist()
        runs = []
        for i, r in enumerate(batch.reqs):
            m = qo[i + 1] - qo[i]
            start = kv_lens[i] - m
            runs.append((_context(r.input_ids, start, width), r.input_ids[start : start + m]))
        self.fill(runs, graph=False)

    @contextmanager
    def forward_host_ctx(self, batch: Batch, use_graph: bool):
        self.host_fill_batch(batch, use_graph)
        yield
        self._await_pending()
        if use_graph:
            stream = torch.cuda.current_stream(self.device)
            for t in self.tables:
                t.graph_consumed.record(stream)


__all__ = ["EngramDiskTable", "EngramDenseTable", "EngramHost", "engram_row_source"]
