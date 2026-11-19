"""A minimal in-process engine for the tiny Hy4-flash model: builds the kpool MLA pool,
the hy4 DSA backend and context by hand (no scheduler), loads the synthetic bf16 checkpoint
with resident experts and a device-resident engram table, and drives ragged prefill /
batched decode batches through ``model.forward()``."""

from __future__ import annotations

from dataclasses import replace

import torch
from freetoken.attention import create_attention_backend
from freetoken.core import Batch, Context, Req, SamplingParams, get_global_ctx, set_global_ctx
from freetoken.distributed.info import set_tp_info, try_get_tp_info
from freetoken.engine.engine import _materialize_loaded_weight_state_dict
from freetoken.kvcache import create_kvcache_pool
from freetoken.layers import set_rope_device
from freetoken.layers.embedding import VocabParallelEmbedding, set_w8a16_embed_default
from freetoken.layers.quantization import NoQuantConfig
from freetoken.layers.quantization.method import finalize_quant
from freetoken.models import create_model
from freetoken.models.hy_v4_flash.config import parse_config
from freetoken.models.hy_v4_flash.engram_table import EngramDenseTable, EngramHost
from freetoken.models.hy_v4_flash.weight import iter_weights
from freetoken.utils.hf import cached_load_hf_config
from freetoken.utils.torch_utils import torch_dtype

P = 64  # kpool page size (the engine coerces to 64 for index_ratio 4)


class TinyEngine:
    def __init__(self, checkpoint: str, tensors: dict | None = None, *, max_seq_len: int = 1024, max_running_req: int = 2, quantized: bool = False, moe_strategy: str | None = None):
        self.device = torch.device("cuda")
        if try_get_tp_info() is None:
            set_tp_info(0, 1)
        set_rope_device(self.device)
        hf_config = cached_load_hf_config(checkpoint)
        mc = parse_config(hf_config)
        quant = NoQuantConfig()
        if quantized:
            from freetoken.models.register import checkpoint_quant_config, get_model_spec

            quant = checkpoint_quant_config(checkpoint, hf_config, get_model_spec(mc.architectures[0]))
        # the engine installs the checkpoint's QuantConfig before the weights load
        # (expert readers query its stored-tensor names)
        from freetoken.layers.quantization import set_quant_config

        set_quant_config(quant)
        strategy = moe_strategy or ("offload" if quantized else "fused")
        self.config = replace(mc, moe_strategy=strategy, decode_target="gpu", quant=quant)
        mc = self.config
        self.args = mc.hy4_args
        # bf16 embedding parity (no W8A16 fp8 table)
        ambient = VocabParallelEmbedding._W8A16_EMBED_DEFAULT
        set_w8a16_embed_default(False)
        try:
            with torch.device("meta"), torch_dtype(torch.bfloat16):
                self.model = create_model(self.config)
        finally:
            set_w8a16_embed_default(ambient)
        state = _materialize_loaded_weight_state_dict(
            self.model.state_dict(),
            iter_weights(checkpoint, self.device, include_moe_experts=False, include_non_moe=True),
            device=self.device,
        )
        self.model.load_state_dict(state)
        finalize_quant(self.model)
        from freetoken.layers import iter_moe_layers
        from freetoken.moe.expert_banks import attach_resident_banks, load_expert_banks

        method = next(iter(iter_moe_layers(self.model))).quant_method
        if strategy == "offload":
            from freetoken.moe.offload_cache import OffloadMoeCache, attach_offload_moe_cache

            self.banks = load_expert_banks(checkpoint, mc, method=method, device=self.device, parallel=False)
            self.expert_cache = OffloadMoeCache(
                num_layers=mc.num_moe_layers, num_experts=mc.num_experts,
                cache_size=min(mc.num_moe_layers * mc.num_experts, 512), device=self.device,
                quant_format=self.banks.quant_format, layout=method.layout(),
            )
            self.expert_cache.set_bank_sources(self.banks.sources, layer_residency=self.banks.layer_residency)
            attach_offload_moe_cache(self.model, self.expert_cache)
        else:
            attach_resident_banks(list(iter_moe_layers(self.model)), load_expert_banks(
                checkpoint, mc, method=method, device=self.device, parallel=False, resident=True,
            ))
        # engram: device-resident tables (synthetic fixture) or disk streaming (real checkpoint)
        tables, hashes = [], []
        for layer in self.model.engram_layers():
            hashes.append(layer.init_hash(self.args))
            if tensors is not None:
                emb = tensors[f"model.layers.{layer.layer_id}.engram_layer.multi_head_embedding.embedding.weight"]
                tables.append(EngramDenseTable(
                    emb, layer.n_cols, self.device,
                    max_graph_rows=max_running_req + 1, max_extend_tokens=max_seq_len,
                ))
            else:
                from freetoken.models.hy_v4_flash.engram_table import EngramDiskTable, engram_row_source

                tables.append(EngramDiskTable(
                    engram_row_source(checkpoint, layer.layer_id), layer.n_cols, self.device,
                    max_graph_rows=max_running_req + 1, max_extend_tokens=max_seq_len,
                ))
            layer.attach_table(tables[-1])
        if tables:
            self.engram_host = EngramHost(hashes, tables, self.device, sync_mode="gate")
            self.model.forward_host_ctx = self.engram_host.forward_host_ctx
        self.model.prepare_for_runtime()

        self.max_running_req = max_running_req
        num_pages = max_seq_len // P + 1  # +1 dummy page
        self.pool = create_kvcache_pool(
            mc, num_pages, P, torch.bfloat16, self.device, num_req_slots=max_running_req + 1,
        )
        self.page_table = torch.zeros(max_running_req + 1, max_seq_len, dtype=torch.int32, device=self.device)
        self.page_table[max_running_req].fill_((num_pages - 1) * P)
        try:
            ctx = get_global_ctx()
        except AssertionError:
            ctx = Context(page_size=P)
            set_global_ctx(ctx)
        self.ctx = ctx
        ctx.page_table = self.page_table
        ctx.kv_cache = self.pool
        self.backend = create_attention_backend("dsa", mc)
        ctx.attn_backend = self.backend
        self._next_page = 0

    def new_request(self, table_idx: int, tokens: list[int], max_new: int = 64) -> Req:
        n_pages = -(-(len(tokens) + max_new) // P)
        first = self._next_page
        self._next_page += n_pages
        assert (first + n_pages) * P <= self.page_table.shape[1], "test pool exhausted"
        locs = torch.arange(first * P, (first + n_pages) * P, device=self.device)
        self.page_table[table_idx, : n_pages * P] = locs.to(torch.int32)
        return Req(
            input_ids=torch.tensor(tokens, dtype=torch.int32), table_idx=table_idx, cached_len=0,
            output_len=max_new, uid=table_idx, sampling_params=SamplingParams(), cache_handle=None,
        )

    def _out_loc(self, reqs: list[Req], positions: torch.Tensor) -> torch.Tensor:
        table_idx = torch.cat([
            torch.full((r.extend_len if r.extend_len > 0 else 1,), r.table_idx, dtype=torch.long)
            for r in reqs
        ])
        return self.page_table[table_idx, positions.long()]

    def prefill(self, reqs: list[Req]) -> torch.Tensor:
        batch = Batch(reqs=reqs, phase="prefill")
        batch.padded_reqs = list(reqs)
        batch.input_ids = torch.cat([r.input_ids[r.cached_len : r.device_len] for r in reqs]).to(self.device)
        batch.positions = torch.cat([torch.arange(r.cached_len, r.device_len) for r in reqs]).to(self.device)
        batch.out_loc = self._out_loc(reqs, batch.positions)
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
        batch.out_loc = self.page_table[batch.active_table_idx.long(), batch.positions.long()]
        self.backend.prepare_metadata(batch)
        with self.ctx.forward_batch(batch), self.model.forward_host_ctx(batch, False):
            return self.model.forward().float()

    def finish_prefill(self, reqs: list[Req]) -> None:
        for r in reqs:
            r.cached_len = r.device_len


__all__ = ["TinyEngine", "P"]
