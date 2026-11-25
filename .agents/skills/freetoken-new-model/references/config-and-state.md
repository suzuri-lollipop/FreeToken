# Config, registration and stateful infrastructure

## parse_config: HF config -> ModelConfig

Pattern from `models/qwen4_exp/config.py`:

- Accept the multimodal wrapper: `text = getattr(hf_config, "text_config",
  hf_config)`. Weights live under `model.language_model.`; the reader strips
  the prefix (`rename_vl_prefix`).
- Normalize HF quirks at the boundary, with a comment naming the HF source:
  - `layer_types` names move between transformers releases
    (`qwen_sparse_attention` <= 5.17, `indexed_attention` >= 5.18 -> both map
    to this repo's `full_attention`); fall back to `full_attention_interval`
    (every Nth layer, 1-indexed, is full).
  - `head_dim` may be explicit or `hidden_size // num_attention_heads`.
  - `rotary_dim = int(head_dim * partial_rotary_factor)` - int(), not
    round(): HF truncates.
  - Default rope type: drop the unhashable `mrope_section` list from the
    scaling dict so it never reaches `get_rope`'s cache key; keep mrope only
    when a vision tower is built.
  - `ple_layer_ids` are ONE-indexed in HF -> convert to zero-based; validate
    each sits on a linear_attention layer.
  - `eos_token_id` may be a list -> take the first entry.
  - A null `output_gate_type` resolves to `hidden_act` (mirror HF, do not
    stringify None).
  - `norm_topk_prob` absent from the shipped config: default True to match
    the reused Qwen3_5MoE block, which renormalizes unconditionally.
- Expert quant flag for the engine's strategy decisions:
  `QuantConfig.from_hf(hf_config).scheme_for_name("model.language_model.layers.0.mlp.experts.0.gate_proj")`
  -> `expert_quant` ("none" / kind string). Every module still takes its own
  scheme from the QuantConfig when built.
- Family geometry goes in a frozen dataclass (`Qwen4ExpArgs`) stored on
  `ModelConfig.qwen4_args` (opaque to model-agnostic engine code). Derived
  quantities are `@property` (e.g. `index_topk_blocks = index_budget //
  index_ratio`, `num_ngram_heads = (ngram_size - 1) * heads_per_ngram`,
  `ple_conv_state_len = (kernel - 1) * dilation`).
- MTP geometry is record-only (`mtp_num_layers`, `mtp_rope_theta`,
  `mtp_use_hidden_state_from_layer`); nothing is built until
  `--speculative mtp` calls `extend_config_for_mtp`.

## Attention groups

`ModelConfig.attention_groups` is a tuple of group configs ordered by first
layer id (deterministic iteration):

- `FullAttentionGroupConfig(name, layer_ids, num_kv_heads, head_dim,
  rotary_config, index_head_dim, num_index_layers, index_ratio)` - the
  indexer fields size the QSA slab (AttnType derivation below).
- `LinearGatedDeltaGroupConfig(name, layer_ids, num_key_heads,
  num_value_heads, key_head_dim, value_head_dim, conv_kernel_dim,
  output_gate)`.
- Also available: `SWAAttentionGroupConfig`, `DSV4AttentionGroupConfig`.
- Helpers: `config.is_linear_layer(layer_id)`,
  `config.linear_attention_group()`, `config.has_hybrid_attention`.
- `AttnType` derives from the group (`_full_group_attn_type`): non-MLA with
  an index slab (`index_head_dim > 0`, `num_index_layers > 0`) -> QSA when
  `index_ratio > 1` (compressed keys), BSA otherwise; no slab -> FULL.

`extend_config_for_mtp` appends the synthetic layer id `num_layers` to the
full group's `layer_ids` and bumps `num_index_layers`: the KV/index slab
factory and the backend's layer->slot map both read the group, so one
append sizes the extra QSA layer everywhere. Idempotent; refuses non-qwen4_exp
and `mtp_num_layers != 1`.

## Slot states (per-request state on linear-attention slots)

For state that must survive chunked prefill, radix snapshot / COW and
speculative verify (qwen4_exp: PLE dilated-conv left context + n-gram
window). Declare in `parse_config` via `ModelConfig.slot_states`:

```python
SlotStateSpec(
    name="ple_conv",                       # read back: pool.slot_state(name, layer_id)
    shape=(hc_count * hidden, (kernel - 1) * dilation),  # per slot, TP-replicated
    layer_ids=ple_layer_ids,               # () -> one shared layer dimension
    dtype=None,                            # None -> pool compute dtype
    fill_value=0.0,
    verify=PLEConvVerify,                  # None -> speculative serving refused
)
```

- Allocated `[max(1, len(layer_ids)), num_slots, *shape]`; `copy_from`,
  `reset`, `rebuild` cover it with the GDN `conv_states` / `recurrent_states`.
- Requires a linear-attention group to ride on.
- The owner advances the state in forward and writes its prefill track
  snapshot into `fla_metadata.track_dst` AFTER the chunk's rows.
- `SlotVerify` protocol (owner supplies, pool only calls):
  - `nbytes(spec, max_bs, rows_per_req, dtype)` static - scratch bytes from
    the spec alone, before any pool exists (cache budget planning reads it).
  - `commit(pool, slots, counts)` - roll live slot `slots[i]` to after the
    first `counts[i]` accepted rows of verify request i; int64 device
    tensors; may run more than once per step; re-read state through
    `pool.slot_state` every call (rebuild replaces it).
  - optional `stash(meta)` - the verify forward stages what commit needs;
    retrieve with `pool.verify_handler(name)`. Scratch row = verify-batch
    row, never a pool slot; single-buffered (spec forces the normal loop).
- qwen4_exp's ngram context: `fill_value = eos_token_id` (hash boundary),
  int32, shared by every PLE layer; on verify the model stashes the meta and
  the commit rolls by accepted rows only; on normal steps a single
  `commit_ngram_context` after the layer loop (layers only read, so a second
  PLE layer's prefetch sees the un-rolled window).

## QSA KV pool (`kvcache/qsa_pool.py`)

`QSAKVCache(MHAKVCache)` - paged GQA K/V plus three index tiers:

- Compressed slab: ONE key row per `index_ratio`-token group, addressed by
  `slot // index_ratio`. Well-defined because `page_size % index_ratio == 0`
  (a group never straddles a page). The slab is a 1/ratio shadow of the KV
  pages: follows page sharing / eviction for free - no allocator, no free,
  no clear. Stale rows must stay unreachable: scoring clamps to
  `kvlen // index_ratio`.
- `pending_ring(slot)`: last `ring_capacity` PRE-RoPE index keys per running
  request, indexed by `Req.table_idx`, never cleared (a new tenant starts at
  a group boundary: cached_len is 0 or a page multiple).
- Scratch rows at `cmp_scratch_base`: one per request slot, write target for
  rows whose group does not close this forward (unconditional scatter, no
  negative index).
- Cost: slab amortized into per-token `unit_bytes`; ring + scratch fixed,
  priced through `kv_cost`'s `fixed_cache_size`. Index tiers always 2-byte.

## QSA attention backend (`attention/qsa_sparse.py`)

Model/backend split (same shape as MiniMax-M3's `bsa_forward`): the layer
owns weights and hands over RAW index projections; the backend owns
everything stateful. Per-layer forward, ragged over `[T, ...]`:

1. store K/V at `batch.out_loc`;
2. pool each row's closing group (new members from this forward, older from
   the pending ring), zero-centered rmsnorm in fp32, rope at the group's
   FIRST position, scatter to slab row `out_loc // index_ratio` (non-closing
   -> scratch);
3. store this forward's last `ring_capacity` raw index keys per request;
4. norm+rope indexer queries at their own positions;
5. score complete visible blocks: `sum_h relu(<q_h, k_bar_b>) /
   sqrt(index_head_dim)`, top `index_budget // index_ratio` blocks, expand to
   token indices + causal tail of the open group;
6. attend to exactly those tokens.

- The engine pins `page_size == 64` (backend's `page_sizes`);
  `block_table[req, p] = page_table[req, p * 64] // 64` names both the KV
  page and the block's slab page.
- CUDA-graph capturable: decode stages block table, live lengths, table_idx
  into static buffers (`prepare_for_replay`).
- QSA is exactly dense while a request sees <= `index_budget + index_ratio
  - 1` tokens - that is the equivalence oracle against
  `TorchDenseQSAReference`.
- TP: qkv/o_proj rank-local; indexer + slab REPLICATED so every rank selects
  the same blocks.

## Scheduler alignment

Hybrid snapshots are donated only at page-aligned boundaries:
`CacheManager.prefill_chunk_align = page_size if is_hybrid else 1`;
`PrefillAdder` aligns non-final chunk ends down, keeping the unaligned end
when the leftover budget cannot fill one whole unit (correctness is
unaffected - only prefix reuse is lost).

## Engine hooks and load order

In `engine/engine.py` (order matters):

1. weight load (dense reader + expert banks);
2. host embedding tables (`--embed-weights host`) +
   `model.load_host_tables(config)` if present - returns pinned bytes,
   added to `_host_tables_bytes`;
3. `_init_offload_moe_cache` - expert residency planning sees the pin quota
   already spent;
4. `model.prepare_for_runtime()` if present;
5. multimodal processor / encoder cache / warmup;
6. KV budget: `_startup_kv_budget(...)` minus `state_pool_bytes(config)`
   (slot states included), minus the speculative draft's share.

`load_host_tables` (qwen4_exp `model.py`) selects the PLE backend:
- dummy weights: re-derive the real hash constants (a zero vocab size
  divides by zero in the hash) and attach `ZeroTable`; return 0;
- `disk`: `DiskRowTable(resolve_row_source(model_path), constants,
  max_graph_rows, max_eager_tokens)`; exactly ONE PLE layer (one WAIT node
  per captured graph); sets `self.forward_host_ctx` (the engine enters it
  around every dispatch; the graph never waits on disk); returns 0;
- `gpu`: pageable read -> VRAM copy -> release host pages;
  `GpuResidentTable`; full copy per rank; returns 0;
- `pinned` (default fallback): `load_ple_table` -> `PinnedUVATable`;
  returns `table.bank.nbytes` (reserved from the pin budget).
Keep the table object alive on the model (`self._ple_table`).

Disk staging sizes (from the fix history - get them right first time):
- `max_graph_rows = max_running_req * decode_rows_per_req` when rows > 1
  (verify graphs), else `max(256, cuda_graph_max_bs)`;
- `max_eager_tokens = max(max_extend_tokens, max_running_req *
  decode_rows_per_req)` - an eager decode/verify step can exceed a prefill
  chunk.
