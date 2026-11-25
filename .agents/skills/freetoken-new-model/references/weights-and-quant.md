# Weight readers, quantization and checkpoint formats

Pattern source: `models/qwen4_exp/weight.py` (592 lines) + `gguf.py`.
Three separate paths because the checkpoint's weight classes live in
different places: dense tensors (`iter_weights`), the host-scale table
(`load_ple_table`), routed experts (`nvfp4_expert_spec` -> bank reader).

## iter_weights contract

```python
def iter_weights(model_path, device, *, include_moe_experts,
                 include_non_moe, include_vision=True) -> Iterator[(str, Tensor)]
```

- Yields the model's state-dict keys (checkpoint module names below the
  stripped prefix), TP-LOCAL tensors. `include_moe_experts` is accepted for
  the loader contract but yields nothing when experts have a bank reader.
- `if not include_non_moe: return` - the offload cache calls with
  `include_non_moe=False` to stream experts only.
- Rename/skip rules (`_rename`): drop `mtp.*` (draft reads it separately),
  the PLE table infix + scale (load_ple_table), routed-expert keys (bank
  reader), static KV scales (`.k_scale/.v_scale/.q_scale/.prob_scale` - the
  KV cache runs in the engine's dtype); `rename_vl_prefix` for the tower.
- Iterate `iter_weight_files(model_path)` with `safetensors.safe_open`,
  tqdm gated on `get_tp_info().is_primary()`.

## _DenseReader: every Linear under the checkpoint's QuantConfig

The core lesson of the fix history (#759, #426): do NOT assume a dtype.
Each Linear module is built with a scheme from `QuantConfig`; the reader
must route each checkpoint tensor to the buffer that scheme declares.

- `spec.packed_modules_mapping` is the part table (fused leaf -> source
  leaves). `reader.add(name, tensor)` accumulates parts; emits the fused
  buffer once every part has every role its scheme stores.
- Roles come from `quant.storage(scheme)` (dialect storage names): bf16 ->
  just `weight`; fp8/nvfp4 -> weight + scales (+ input_scale). A suffix the
  scheme does not declare is an ERROR naming the module and scheme, except
  `input_scale`/`input_global_scale` of schemes without that role (dropped).
- `_check_weight`: dtype must match the scheme's element type
  (`e4m3 -> float8_e4m3fn`, `e2m1 -> uint8` packed pairs); shape must be a
  multiple of the scale block; an unquantized module receiving a quant dtype
  is refused. A checkpoint disagreeing with its own quantization_config
  fails loudly at load, never silently dequantized.
- `_finish`: reciprocal scales inverted per `storage[role].reciprocal`;
  per-channel scales (`group cols < 0`) normalized to fp32 `[rows]` -
  exactly 1 or `rows` elements, any other count REFUSED (no broadcast onto
  wrong rows); block scales shape-checked `(rows//br, cols//bc)`; e4m3
  scales must actually be float8_e4m3fn; `weight_global` one scalar
  expanded to rows in fp32; `input_scale` scalar fp32.
- Fusions per role: attention q|k|v -> `qkv_proj`; GDN
  `in_proj_{qkv,z,b,a}` -> `in_proj`, or `in_proj_qkvz` + `in_proj_ba` when
  qkv|z is quantized (same test as the layer: scheme of `in_proj_qkvz`
  exists); shared-expert gate|up -> `gate_up_proj`; HC
  `input_mix_weight_down` | `block_inject_weight` ->
  `input_mix_weight_down_block_inject` with ZERO pad rows to a multiple of
  16 (vLLM hyperconnection pad_size). Fused `input_scale` = max over parts
  (same activation). The top-level mixer never fuses (no injection).
- The padded HC fusion is bf16-ONLY: a quant scheme on it raises.
- `reader.missing()` at EOF: one line per incomplete module with the roles
  still lacking - raise instead of loading a half-filled model.

## Zero-centered norms load RAW

HF `Qwen4ExpTextRMSNorm` weights are zero-centered; the runtime applies
`(1+w)` in fp32 (GroupedPlusOneRMSNorm / GemmaPlusOneRMSNorm / vendored
grouped_gemma_rmsnorm). Folding +1 into the bf16 weight at load would
double-apply it and round away small |w|. qwen4_exp's raw-load suffix list:
`.hc_norm.weight`, `.ple.norm_{key,query,conv}.weight`,
`.self_attn.{q,k}_norm.weight`, `.self_attn.indexer.{q,k}_layernorm.weight`.
The GDN gated norm (`linear_attn.norm`) is a plain weight*x norm - NOT in
the list. A test asserts the suffix list covers every such norm in the
state dict.

## TP slicing (_dense_slicer)

Final rule after the TP fix series (#430s): the reader emits fused
GLOBAL-shape buffers; the slicer splits a fused buffer back into its parts,
cuts each part by its own leaf (`shard.tensor(part, piece)`), re-concatenates
in part order. Leaves outside the shard table (routers, norms, HC/PLE
mixers, replicated fc front end) are replicated - cutting is a no-op.
Row-count mismatch between buffer and parts raises.

- `tp_shard(model_config)` + `ROLE_SUFFIXES` / `module_leaf` from
  `models/tp_shard.py`; the vision tower cuts by its own leaves
  (`vision_tp_shard`, `vision_module_leaf`).
- Construction side: `LinearColParallelMerged` takes FULL global output
  sizes (`[d * tp_size for d in split]`); forward uses TP-local split dims.
  `LinearOProj` / `LinearRowParallel` for o_proj/out_proj (all-reduce).
- KV heads must divide the TP size when the fused qkv slices rows of the
  k/v segments - raise a clear error at construction otherwise.
- Indexer + compressed slab stay replicated (every rank selects the same
  top-k blocks).
- Drop the dense shards' page cache after loading (`drop_page_cache`).
- The MTP draft reader shares `_dense_slicer` so draft buffers get the same
  rank-local cuts.

## Routed NVFP4 experts: nvfp4_expert_spec

```python
Nvfp4ExpertSourceSpec(
    key_pattern=...,        # regex over RAW weight_map keys, with named
                            # groups layer/expert/proj/kind; anchor
                            # model.language_model. excludes mtp.* stacked
    proj_to_role={"gate_proj": "gate", "up_proj": "up", "down_proj": "down"},
    layer_to_bank=lambda layer, config: layer,   # every layer is MoE
    desc=...,
    kind_map=...,           # dialect storage name -> canonical modelopt kind
                            # (weight_packed -> weight, weight_global_scale
                            #  -> weight_scale_2 for llm-compressor)
    global_reciprocal=...,  # checkpoint stores the QUANT-side global scale;
                            # banks keep its reciprocal
)
```

Build `kind_map` from `get_quant_config().stored_tensors(QuantKind.NVFP4)`
so ModelOpt and llm-compressor dialects both resolve. `moe/expert_pieces.py`
resolves the hook BEFORE any bank is allocated; a family without the hook
raises NotImplementedError. Block-fp8 experts: reuse
`qwen3_5_moe.weight.iter_expert_pieces` (same `model.language_model.layers.*`
key layout). NVFP4 global scales stay fp32 (breaking fix #739).

## PLE host table (load_ple_table / ftw_side_files)

- The checkpoint splits the table into `split_ngram_parts` equal row blocks
  named `...ngram_embedding.shard_<i>.weight`, scattered over shards in
  header order. Locate shards via `model.safetensors.index.json` weight_map.
- Parse safetensors headers directly (struct + json) for dtype/shape/offsets;
  validate: shards exactly `0..split_ngram_parts-1`, uniform shape/dtype,
  row width == `ngram_head_dim`; fp8 table REQUIRES the scalar
  `weight_scale`, bf16 without one is unscaled (scale 1).
- Fill one `HostBank((parts * rows, cols), dtype)` shard by shard at
  `shard_index * rows_per_shard` with `read_range_into` - chunked
  multi-threaded O_DIRECT (the ~47.7 GiB table must not also sit in the page
  cache). `bank.pin()` when CUDA is available; `pin=False` for the GPU path.
- `ftw_side_files(model_path, out_dir)`: write ONLY the table tensors into
  `ple-table-*.safetensors` (4 GiB rollover) next to an FTW checkpoint;
  resolved by name in `checkpoint/convert.py`. The table is served from
  safetensors in the checkpoint dir, never from FTW entries.

## GGUF path (gguf.py)

Register a second ModelSpec with `parse_gguf_config` / `iter_gguf_weights`
and map the arch string in `models/gguf/config.py`.

`parse_gguf_config(shim)`:
- `check_scales(model_path)`; refuse a scaled PLE table
  (`per_layer_token_embd.scale`) and decoder-less files (MTP-only packs).
- Build an HF-like `SimpleNamespace(text_config=..., model_type=...)` from
  GGUF metadata and reuse `parse_config` - one geometry pipeline.
- `_FAMILY_DEFAULTS` for fields the metadata does not carry (hidden_act,
  output_gate_type, indexer_kv_heads, ngram vocab constants, norm_topk_prob).
- `block_count` INCLUDES MTP layers: subtract `nextn_predict_layers`.
- GGUF `embedding_length_per_layer_input` is ONE n-gram head's row; HF
  `ple_embed_dim` counts all heads: multiply by `(ngram_size-1) *
  heads_per_ngram`. `ple.layers` zero-based in GGUF -> +1 for HF's
  one-indexed field.
- Result: `expert_quant="gguf"`, `gdn_out_heads_tiled=True`.

`iter_gguf_weights` - undo the llama.cpp converter's transforms:
- zero-centered norms stored as `w + 1` -> `float_tensor(...).float() - 1.0`;
- `A_log` stored as `-exp(A_log)` -> `log(-x)`;
- conv kernels squeezed -> `unsqueeze(1)`;
- GDN value heads in tiled order (k-head minor) -> `_untile` row reorder on
  the PACKED bytes for qkv/gate/beta/alpha/conv/dt_bias; `ssm_out` KEEPS its
  tiled columns and the GDN layer permutes its input instead
  (`out_heads_tiled` flag) - reordering packed quant rows is not always
  possible;
- indexer projection split into q/k -> fuse to `index_qk_proj`;
- quantized GDN: `in_proj_qkvz` + `in_proj_ba` split (same rule as HF);
- HC down+inject fused with pad_to=16;
- PLE hash constants come from GGUF metadata arrays
  (`ple.layer_multipliers`, `ple.head_vocab_sizes`, `ple.head_offsets`) as
  int64 tensors;
- vision tower from an mmproj (`find_mmproj`, `iter_mmproj_weights`), bf16.
- `to_gguf(name, tensors)` maps module names -> GGUF tensor names for the
  GGUF quant config (`GGUFConfig(types, to_gguf, name_map=NameMap(packed=...))`).

## MTP weight readers

- `iter_mtp_weights`: `mtp.*` minus its routed experts, renamed (strip
  `mtp.`) and fused like the target's dense weights, TP-cut by the shared
  slicer; raises when no `mtp.*` exists or projections are incomplete.
- `iter_mtp_expert_pieces`: the stacked bf16 pair (`gate_up_proj` /
  `down_proj`) OR per-expert projections with block-fp8 scales, via
  `moe/expert_pieces.iter_extra_layer_expert_pieces` with a family `locate`
  function. Require BOTH stacked tensors (fix #674d671).
