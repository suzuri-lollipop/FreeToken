# Anatomy: what a new model touches

The complete touch map of the Qwen3.8-Flash-Next support commit `e3b6d7b`
(#257, 66 files, ~10k insertions), generalized into a checklist. Paths are
relative to the repository root. A simpler model needs a subset; the order
of the phases in SKILL.md still holds.

## 1. The model package: `python/freetoken/models/<family>/`

qwen4_exp's layout (file -> responsibility):

| File | Responsibility |
|---|---|
| `__init__.py` | Re-exports the hooks the engine resolves: `parse_config`, model classes, `iter_weights`, `nvfp4_expert_spec`, `load_ple_table`, `ftw_side_files`, `iter_vision_weights`, GGUF functions. Module docstring carries the architecture summary and the cross-module contracts. Reuses `qwen3_5_moe.weight.iter_expert_pieces` for block-fp8 experts. |
| `config.py` | `Qwen4ExpArgs` frozen dataclass (family geometry beyond generic `ModelConfig`), `parse_config(hf_config) -> ModelConfig`, `ple_slot_states`, `extend_config_for_mtp`. |
| `model.py` | `Qwen4ExpModel` (embed + `OPList` of layers + top-level mixer), `Qwen4ExpDecoderLayer`, `Qwen4ExpForCausalLM(BaseLLMModel)`, `Qwen4ExpForConditionalGeneration(QwenVLVisionMixin, ...)`, `load_host_tables`, `forward_dual`. |
| `attention.py` | The full-attention layer module (`Qwen4ExpAttention`), the frozen `QSAIndexerInputs` dataclass, the `QSAAttentionBackend` Protocol, and `TorchDenseQSAReference` (fp32 dense oracle, test-only, never registered). |
| `hc.py` | Hyper-connection `GatedResidual` (mix / combine / combine_norm) + `GroupedPlusOneRMSNorm`; pure-torch reference chain and the Triton kernel path side by side. |
| `moe.py` | `Qwen4ExpMoE(Qwen3_5MoE)` - reuse the family MoE, override only the shared-gate epilogue with Triton kernels. |
| `ple.py` | `PLELayer`, `NGramEmbedding`, hash-constant derivation, `PLEMetadata`, table backends (`PinnedUVATable`, `GpuResidentTable`, `ZeroTable`), verify handlers (`PLEConvVerify`, `PLENgramVerify`). |
| `ple_disk.py` | `DiskRowTable` + `resolve_row_source` for `--ple-backend disk`. |
| `weight.py` | `_DenseReader` + `iter_weights`, `load_ple_table`, `ftw_side_files`, `nvfp4_expert_spec`, `iter_mtp_weights`, `iter_mtp_expert_pieces`, `iter_vision_weights`. |
| `gguf.py` | `parse_gguf_config`, `iter_gguf_weights`, `to_gguf` name map; undoes the llama.cpp converter's transforms. |
| `mtp.py` | `Qwen4ExpMTPDraft` (speculative head from `mtp.*` tensors or a Strata GGUF pack). |

The initial commit also had `gdn.py` + `gdn_reference.py` in the package;
they were later refactored into the SHARED op `layers/linear_attn/gdn.py`
(`GatedDeltaNet`, #841) because Qwen3.5 uses the same mixer. Prefer extending
a shared layer op over a family-private copy when the math matches.

## 2. Registration and engine-wide declarations

- `python/freetoken/models/register.py`
  - One `ModelSpec` per served architecture string. qwen4_exp registers
    `Qwen4ExpForConditionalGeneration` (HF) and `Qwen4ExpGGUFForCausalLM`
    (same model class, `parse_config="parse_gguf_config"`,
    `iter_weights="iter_gguf_weights"`).
  - `packed_modules_mapping`: fused leaf -> checkpoint source leaves.
    qwen4_exp = `_QWEN3_5_PACKED` (qkv, gate_up, GDN in_proj variants,
    per-expert probe) + `("input_mix_weight_down_block_inject",
    ("input_mix_weight_down", "block_inject_weight"))`; the GGUF spec adds
    `("index_qk_proj", ("index_q_proj", "index_k_proj"))`.
  - `checkpoint_roots`: attribute-path -> checkpoint-root for quant lookups
    (`_QWEN_VL_ROOTS` = model -> model.language_model, visual -> model.visual).
  - `tp_supported=True` only when the reader emits TP-local slices and every
    shardable Linear is column/row-parallel (pre-flightable at startup).
  - `encoders=(EncoderSpec("vision", "vision_config", ("image",),
    tp_sharded=True),)`, `mm_processor="freetoken.mm.processors.qwen_vl:QwenVLMMProcessor"`.
  - `spec_drafts=(("mtp", "freetoken.models.qwen4_exp.mtp:Qwen4ExpMTPDraft"),
    ("dflash", "freetoken.models.dflash:DFlash2DraftModel"))`.
  - `unquantized_modules` / `quant_overrides` for checkpoints whose
    quantization_config under-declares or covers modules built in bf16.
- `python/freetoken/models/config.py` - add the family payload field
  (`qwen4_args: Any | None`, opaque to model-agnostic engine code) and use
  the generic `attention_groups` / `slot_states` fields.
- `python/freetoken/models/gguf/config.py` - GGUF arch string map:
  `"qwen4exp": "Qwen4ExpGGUFForCausalLM"`.
- `python/freetoken/kernel/aot_models.py` - one `AotModel` entry per
  checkpoint family so the prebuilt kernel cache hits by spec name:
  `name="RadixArk/Qwen3.8-Flash-Next-NVFP4"`,
  `architecture="Qwen4ExpForConditionalGeneration"`, `hidden_size=2560`,
  `kv_groups=((2, 256),)`, `top_k=10`, `moe_intermediate_size=640`,
  `expert_formats=(*_NVFP4_FORMATS, "fp8_block")`,
  `arch_aliases=("Qwen4ExpGGUFForCausalLM",)`.
  Derivations must stay in lockstep with runtime call sites by hand; a
  drifted derivation silently falls back to JIT (needs nvcc).
- `python/freetoken/checkpoint/convert.py` - resolves the optional
  `ftw_side_files` hook by name when converting HF -> FTW (qwen4_exp writes
  `ple-table-*.safetensors` next to the FTW; the table is never an FTW entry).

## 3. Attention / KV cache / scheduler (only for new attention types)

- `python/freetoken/attention/qsa_sparse.py` (new backend, ~590 lines) -
  serves `AttnType.QSA`; entry point `qsa_forward`, NOT `forward` (the model
  has no dense attention layer). Registered in `attention/__init__.py`;
  `attention/base.py` + `attention/linear.py` gained the metadata plumbing.
- `python/freetoken/kvcache/qsa_pool.py` (new pool, ~236 lines) -
  `QSAKVCache(MHAKVCache)`: paged GQA K/V + compressed index-key slab +
  per-request pending ring + scratch rows. Registered in `kvcache/__init__.py`;
  `kvcache/base.py` cost hooks.
- `python/freetoken/kvcache/linear_state_pool.py` - declarative slot-sibling
  states (`ModelConfig.slot_states`) ride the GDN state slots through the
  hybrid-radix snapshot / COW / rebuild lifecycle.
- `python/freetoken/scheduler/cache.py` - `prefill_chunk_align` property:
  `page_size if is_hybrid else 1`.
- `python/freetoken/scheduler/prefill.py` - align non-final chunk ends down
  to that granularity (keep the unaligned end when the leftover budget
  cannot fill a whole unit).

## 4. MoE / host memory

- `python/freetoken/moe/fused.py`, `fused_nvfp4.py` - non-power-of-2 top-k
  router support, deep-K marlin decode config, one fp8 scale-bank padding
  rule shared with the AOT row table.
- `python/freetoken/moe/host_banks.py` - `HostBank` (page-locked host buffer,
  O_DIRECT-rounded) + `read_range_into` (chunked multi-threaded O_DIRECT
  reads straight into the bank; Windows uses FILE_FLAG_NO_BUFFERING).
- `python/freetoken/moe/offload_cache.py` + `moe/expert_pieces.py` - resolve
  the family's `nvfp4_expert_spec` hook BEFORE any bank is allocated.
- `python/freetoken/models/nvfp4_banks.py` - `Nvfp4ExpertSourceSpec`
  (key_pattern over RAW weight_map keys, proj_to_role, layer_to_bank,
  kind_map for dialect storage names, global_reciprocal).

## 5. Engine / CLI

- `python/freetoken/engine/engine.py` - host-table hook ordering: after the
  weight load (a load failure is not masked), BEFORE `_init_offload_moe_cache`
  (bank residency planning must see the pin quota the table already spent):
  `self._host_tables_bytes += int(self.model.load_host_tables(config) or 0)`.
- `python/freetoken/engine/config.py` - `ple_backend: str = "disk"`.
- `python/freetoken/server/args.py` - `--ple-backend` flag.
- `python/freetoken/server/function_call_parser.py` - the family's detector
  class, its `FunctionCallParser` map entry, and the opener tag in
  `TOOLS_TAG_LIST` (streaming tool-call detection).
- `python/freetoken/server/reasoning_parser.py` - the family's parser class
  and its `ReasoningParser` map entry (`force_reasoning` matches the chat
  template's thinking default).
- `python/freetoken/server/args.py` - the auto-resolution cascade
  (`_infer_tool_call_parser` / `_infer_reasoning_parser`) AND the explicit
  `choices` lists of `--tool-call-parser` / `--reasoning-parser`. These are
  four independent tables; a name the cascade resolves but the choices list
  lacks is rejected by argparse on explicit selection while `auto` works.
  Checklist and CPU verification: `references/delivery-completeness.md`.
- `python/freetoken/calibrate_multi.py` - `spec_backends_for`'s model_type
  map: the family's speculative contract (own drafter -> swap the arm;
  config-time refusal -> no arm). The generic fallback offers mtp + dflash
  and excludes the doomed arm at server start - registration, not fallback.

## 6. Kernels: `python/freetoken/kernel/triton/`

Vendored with SPDX provenance headers (keep them when adapting):

| Kernel | Provenance |
|---|---|
| `hc.py` (grouped_gemma_rmsnorm, hc_silu, hc_gate_mix, hc_combine, hc_combine_norm) | Apache-2.0, adapted from vLLM `vllm/models/qwen4_exp/nvidia/ops/hc.py` |
| `ple.py` (UVA row gather) | Apache-2.0, adapted from SGLang `python/sglang/srt/models/qwen4_exp.py` |
| `moe_router.py` (fused softmax/sigmoid top-k) | Apache-2.0, adapted from SGLang `kernels/ops/moe/moe_fused_gate.py` |
| `moe_shared_gate.py` (gate reduction + sigmoid-mul-add) | Apache-2.0, adapted from SGLang `kernels/ops/elementwise.py` |
| `qsa/attend.py`, `qsa/compress.py`, `qsa/score.py`, `qsa/expand.py` | Apache-2.0, adapted from vLLM `vllm/models/qwen4_exp/nvidia/ops/qsa.py` |
| `qsa/topk.py` | Original: MSB-first radix block top-k (only the winner SET matters, no sort) |

int64 row addressing throughout. Later perf work added `nvfp4_linear.py`,
`nvfp4_fused_moe.py`, `nvfp4_act_quant.py`, `nvfp4_dequant.py`,
`fp8_blockscale_moe.py`, `w8a16_linear.py`, Gluon sm_120 paths
(`kernel/gluon/nvfp4/`), and fused GDN kernels (`gdn_fused_layer.py`,
`gdn_hc_fused.py`) - see `docs/qwen38_flash_next_optimizations.md`.

## 7. Tests (mirror the subsystem layout)

- `tests/models/qwen4_exp/` - the family's own tests (see
  `references/tests-and-verification.md`).
- Subsystem tests extended in the same commit:
  `tests/kvcache/test_qsa_pool.py` (new), `test_linear_state_pool_alloc.py`,
  `tests/engine/test_attention_backend_matrix.py`, `test_cache_budget.py`,
  `tests/moe/test_fused_moe.py`, `tests/scheduler/test_hybrid_cache_manager.py`.

## 8. Docs

- `docs/models.md` - checkpoint table row (HF / NVFP4 links; siblings as
  extra links in the same row), GGUF family alias row (`qwen4exp`),
  image-token table, family notes (PLE backends, KV-cache dtype guidance
  for long context). The Notes entry records the family's serving facts: MoE
  strategy, TP, KV dtype support (the pool's `kv_formats`), draft support,
  rope scaling bundled or not + the trained ceiling, KV bytes/token from the
  cost model. TP / draft / GGUF are mandatory requirements, so a "limitation"
  there is only a surfaced exception the owner agreed to solve next - never a
  silent "not yet" shipped instead of the requirement
  (`references/delivery-completeness.md`).
- `docs/cli.md` - new flags (`--ple-backend`), speculative-draft options.
- `docs/qwen38_flash_next_optimizations.md` - perf analysis + A/B numbers
  (pattern to copy for performance work).

## 9. Delivery completeness - MANDATORY (TP / drafts / GGUF / parsers / calibration)

`tp_supported` and `spec_drafts` in the ModelSpec are required deliverables,
not defaults to leave unset: a family shipped TP1-only with an unserved
bundled draft head, no GGUF reader, and auto-only parser selection is NOT
delivered. TP >= 2, speculative draft, GGUF, the parser four-table
registration and the calibration family map are MANDATORY requirements -
implement and verify them on real hardware. The gates to clear (offload
expert TP refusal, window-tier KV spec refusal) and the checklist are in
`references/delivery-completeness.md`.
