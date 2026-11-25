"""Minimal in-process engine for the tiny ERNIE 5.5 model: builds the hybrid
pool + backend + context by hand (no scheduler), loads a synthetic checkpoint
(bf16 resident or fp8/MXFP4 offloaded experts) and drives ragged prefill /
batched decode through ``model.forward()``. Shared by the parity and smoke tests.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from freetoken.core import Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.distributed.info import set_tp_info, try_get_tp_info
from freetoken.engine.engine import _materialize_loaded_weight_state_dict
from freetoken.kvcache.dsv4.ernie55_pool import Ernie55PagedKVCache
from freetoken.kvcache.dsv4.v4_cost_model import dsv4_pool_sizes
from freetoken.layers import set_rope_device
from freetoken.layers.embedding import VocabParallelEmbedding, set_w8a16_embed_default
from freetoken.layers.quantization import NoQuantConfig
from freetoken.layers.quantization.method import finalize_quant
from freetoken.models import create_model
from freetoken.models.ernie5_5_moe.config import parse_config
from freetoken.models.ernie5_5_moe.weight import iter_weights
from freetoken.utils.hf import cached_load_hf_config
from freetoken.utils.torch_utils import torch_dtype

P = 128


class TinyEngine:
    def __init__(self, checkpoint: str, *, max_seq_len: int = 1024, max_running_req: int = 2, quantized: bool = False, gguf: bool = False, w8a16_embed: bool = True, kv_format=None):
        self.device = torch.device("cuda")
        self.checkpoint = checkpoint
        if try_get_tp_info() is None:
            set_tp_info(0, 1)
        set_rope_device(self.device)
        if gguf:
            from freetoken.models.ernie5_5_moe.gguf import iter_gguf_weights, parse_gguf_config
            from freetoken.models.gguf.config import build_gguf_shim

            hf_config = build_gguf_shim(checkpoint)
            mc = parse_gguf_config(hf_config)
            self._weight_iter = iter_gguf_weights
        else:
            hf_config = cached_load_hf_config(checkpoint)
            mc = parse_config(hf_config)
            self._weight_iter = iter_weights
        args = mc.dsv4_args
        args.max_seq_len = max_seq_len
        args.max_batch_size = max_running_req + 1
        self.args = args
        quant = NoQuantConfig()
        if quantized:
            import copy

            from freetoken.models.register import checkpoint_quant_config, get_model_spec

            spec = get_model_spec("Ernie5_5_MoeForCausalLM")
            # EngineConfig.model_config applies the spec's quant_overrides before the
            # dialect reads the config (ERNIE: scale_fmt "" -- the dense scales are F32
            # weight_scale_inv despite the checkpoint's ue8m0 expert-scale field).
            if spec.quant_overrides:
                from freetoken.layers.quantization.configs.base import quantization_config_of

                hf_config = copy.copy(hf_config)
                q = quantization_config_of(hf_config)
                if q is not None:
                    hf_config.quantization_config = {**q, **dict(spec.quant_overrides)}
            quant = checkpoint_quant_config(checkpoint, hf_config, spec)
        self.config = replace(mc, moe_strategy="offload" if quantized else "fused", decode_target="gpu", quant=quant)
        mc = self.config
        ambient = VocabParallelEmbedding._W8A16_EMBED_DEFAULT
        set_w8a16_embed_default(False)
        try:
            with torch.device("meta"), torch_dtype(torch.bfloat16):
                self.model = create_model(self.config)
        finally:
            set_w8a16_embed_default(ambient if w8a16_embed else False)
        state = _materialize_loaded_weight_state_dict(
            self.model.state_dict(), self._weight_iter(checkpoint, torch.device("cpu"), include_moe_experts=False), device=self.device,
        )
        self.model.load_state_dict(state)
        finalize_quant(self.model)
        self.model.prepare_for_runtime()

        from freetoken.moe.expert_banks import load_expert_banks

        moe_layers = [l for l in self.model.model.layers.op_list if l.layer_id >= args.first_k_dense_replace]
        if moe_layers:
            self._load_experts(checkpoint, mc, moe_layers, quantized)

        self.max_running_req = max_running_req
        num_pages = max_seq_len // P
        from freetoken.kvcache.format import NONE

        self.pool = Ernie55PagedKVCache(
            dsv4_pool_sizes(num_pages + 1, args, 1.0, P), args, self.device, n_scratch=max_running_req + 1,
            kv_format=NONE if kv_format is None else kv_format,
        )
        self.pool._init_paged_state(max_running_req, radix=False)
        self.page_table = torch.zeros(max_running_req + 1, max_seq_len, dtype=torch.int32, device=self.device)
        self.page_table[max_running_req].fill_(num_pages * P)
        self.pool.attach_page_table(self.page_table)
        try:
            ctx = get_global_ctx()
        except AssertionError:
            ctx = Context(page_size=P)
            set_global_ctx(ctx)
        from freetoken.attention.ernie55_sparse import Ernie55SparseAttnBackend

        self.ctx = ctx
        ctx.page_table = self.page_table
        ctx.kv_cache = self.pool
        self.backend = Ernie55SparseAttnBackend(self.config)
        self._next_page = 0
        self._bind()

    def _load_experts(self, checkpoint, mc, moe_layers, quantized):
        from freetoken.moe.expert_banks import load_expert_banks

        method = moe_layers[0].ffn.experts.quant_method
        if quantized:
            from freetoken.moe.offload_cache import OffloadMoeCache, attach_offload_moe_cache

            self.banks = load_expert_banks(checkpoint, mc, method=method, device=self.device, parallel=False)
            self.expert_cache = OffloadMoeCache(
                num_layers=mc.num_moe_layers, num_experts=mc.num_experts,
                cache_size=mc.num_moe_layers * mc.num_experts, device=self.device,
                quant_format=self.banks.quant_format, layout=method.layout(),
            )
            self.expert_cache.set_bank_sources(self.banks.sources, layer_residency=self.banks.layer_residency)
            attach_offload_moe_cache(self.model, self.expert_cache)
        else:
            from freetoken.layers import iter_moe_layers
            from freetoken.moe.expert_banks import attach_resident_banks

            attach_resident_banks(list(iter_moe_layers(self.model)), load_expert_banks(
                checkpoint, mc, method=method, device=self.device, parallel=False, resident=True,
            ))

    def _bind(self) -> None:
        ctx = self.ctx
        if (
            getattr(ctx, "attn_backend", None) is self.backend
            and getattr(ctx, "kv_cache", None) is self.pool
            and getattr(ctx, "page_table", None) is self.page_table
        ):
            return
        ctx.page_table = self.page_table
        ctx.kv_cache = self.pool
        ctx.attn_backend = self.backend
        self.model.mark_for_rebind()

    def new_request(self, table_idx: int, tokens: list[int], max_new: int = 64) -> Req:
        n_pages = -(-(len(tokens) + max_new) // P)
        first = self._next_page
        self._next_page += n_pages
        locs = torch.arange(first * P, (first + n_pages) * P, device=self.device)
        self.page_table[table_idx, : n_pages * P] = locs.to(torch.int32)
        self.pool.alloc_swa(locs)
        return Req(input_ids=torch.tensor(tokens, dtype=torch.int32), table_idx=table_idx, cached_len=0, output_len=max_new,
                   uid=table_idx, sampling_params=SamplingParams(), cache_handle=None)

    def new_request_on_prefix(self, table_idx: int, donor: Req, tokens: list[int], max_new: int = 64) -> Req:
        """A request whose first ``hit`` positions alias the donor's pages (what a
        radix prefix hit does through the page table); the rest get fresh pages.
        The hit is page-aligned, as the scheduler admits it."""
        hit = min(donor.device_len, len(tokens) - 1) // P * P
        assert tokens[:hit] == donor.input_ids[:hit].tolist()
        self.page_table[table_idx, :hit] = self.page_table[donor.table_idx, :hit]
        n_pages = -(-(len(tokens) + max_new - hit) // P)
        first = self._next_page
        self._next_page += n_pages
        locs = torch.arange(first * P, (first + n_pages) * P, device=self.device)
        self.page_table[table_idx, hit : hit + n_pages * P] = locs.to(torch.int32)
        self.pool.alloc_swa(locs)
        return Req(input_ids=torch.tensor(tokens, dtype=torch.int32), table_idx=table_idx, cached_len=hit, output_len=max_new,
                   uid=table_idx, sampling_params=SamplingParams(), cache_handle=None)

    def _out_loc(self, reqs: list[Req]) -> torch.Tensor:
        rows, cols = [], []
        for r in reqs:
            for pos in range(r.cached_len, r.device_len):
                rows.append(r.table_idx)
                cols.append(pos)
        return self.page_table[torch.tensor(rows, device=self.device), torch.tensor(cols, device=self.device)]

    def prefill(self, reqs: list[Req]) -> torch.Tensor:
        batch = Batch(reqs=reqs, phase="prefill")
        batch.padded_reqs = list(reqs)
        batch.input_ids = torch.cat([r.input_ids[r.cached_len : r.device_len] for r in reqs]).to(self.device)
        batch.positions = torch.cat([torch.arange(r.cached_len, r.device_len) for r in reqs]).to(self.device)
        batch.out_loc = self._out_loc(reqs)
        self._bind()
        self.backend.prepare_metadata(batch)
        with self.ctx.forward_batch(batch), self.model.forward_host_ctx(batch, False):
            return self.model.forward().float()

    def decode(self, reqs: list[Req], tokens: list[int]) -> torch.Tensor:
        for r, t in zip(reqs, tokens):
            r.append_host(torch.tensor([t], dtype=torch.int32))
            r.complete_one()
        batch = Batch(reqs=reqs, phase="decode")
        batch.padded_reqs = list(reqs)
        batch.input_ids = torch.tensor(tokens, dtype=torch.int32, device=self.device)
        batch.positions = torch.tensor([r.device_len - 1 for r in reqs], device=self.device)
        batch.active_table_idx = torch.tensor([r.table_idx for r in reqs], device=self.device)
        batch.out_loc = self.page_table[
            batch.active_table_idx, batch.positions
        ]
        self._bind()
        self.backend.prepare_metadata(batch)
        with self.ctx.forward_batch(batch), self.model.forward_host_ctx(batch, False):
            return self.model.forward().float()

    def finish_prefill(self, reqs: list[Req]) -> None:
        for r in reqs:
            r.cached_len = r.device_len
