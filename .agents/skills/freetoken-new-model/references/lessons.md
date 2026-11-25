# Lessons: silent failures the fix history already paid for

Extracted from the ~120 follow-up commits on `models/qwen4_exp/` and its
subsystems. Each entry: the failure mode -> the rule. Commit hashes are
this repository's history (merge duplicates exist; one hash per change).
Sweep this list before calling a new-model phase done.

## Tensor parallelism

- Fused buffers cut with one split rule land on wrong boundaries. Rule:
  cut per LEAF with its own rule - split the fused buffer back into parts,
  cut each part, re-concatenate in part order (`_dense_slicer`); row-count
  mismatch raises. (325e202, df46fdc)
- `LinearColParallelMerged` is constructed with FULL global output sizes;
  forward splits with TP-local dims. Row-parallel (`LinearOProj` /
  `LinearRowParallel`) for o_proj/out_proj takes TP-local input dims and
  all-reduces. Missing shard suffixes in the shard table silently replicate
  a leaf that must be cut. (dd63ed4, 461ac04)
- Fused qkv slices rows of the k/v segments: KV heads must divide the TP
  size, refuse at construction with a clear message - not mid-load.
- The indexer and its compressed slab stay REPLICATED under TP so every
  rank selects the same top-k blocks; only KV heads differ.
- Shared + routed expert partials sum BEFORE one all-reduce (gate is
  rank-replicated); two all-reduces also work but cost twice.
- Drop the dense shards' page cache after loading, or tens of GiB of host
  RAM stay charged. (726c527)
- The MTP draft's dense weights need the SAME rank-local cuts as the
  target's - share the slicer, do not write a second one. (ef16b35)

## Quantization

- Never assume a dtype: read every Linear under the scheme the checkpoint's
  QuantConfig declares for it, the same answer the module built its buffers
  from. A bf16-assuming reader silently misreads fp8/nvfp4 checkpoints.
  (163d1de, d113ac5)
- Expert quant kind comes from the QuantConfig, not a family constant.
  (d113ac5)
- A checkpoint tensor disagreeing with its declared scheme (dtype, shape
  vs scale block, scale element count) must RAISE at load with the module
  and scheme named - never broadcast or silently dequantize. Per-row
  scales: exactly 1 or `rows` elements.
- NVFP4 global scales stay fp32; keeping them in the checkpoint's bf16
  rounds the dequant math. Breaking fix, hence `fix(quant)!`. (0db5023)
- Dialect storage names move: modelopt mxfp8 also ships `weight_scale`
  (accept both names); llm-compressor uses `weight_packed` /
  `weight_global_scale` and stores the QUANT-side global (banks keep the
  reciprocal). Build kind maps from `quant.stored_tensors(kind)`. (b0b57a4)
- Fused parts sharing one activation: `input_scale` = max over parts.
  W4A4 linears use the LARGEST per-part global. (1f886fe)
- Padded fusions (HC down+inject, 16-row pad) are bf16-only - a quant
  scheme on them raises instead of padding packed bytes.
- MTP fc projections are quantized too: read them under the quant config
  like any other Linear. (1e3f418)

## GGUF / converter transforms

- The llama.cpp converter stores zero-centered norms as `w+1` (subtract 1
  at load), `A_log` as `-exp(A_log)` (take `log(-x)`), conv kernels squeezed
  (unsqueeze), GDN value heads tiled k-head-minor (untile row reorders on
  the packed bytes). Missing any of these loads "fine" and generates
  garbage. (gguf.py docstring, 128d0e2)
- `ssm_out` keeps its tiled columns - the layer permutes its input instead
  (`gdn_out_heads_tiled`); do not untile packed quant rows you cannot
  reorder safely.
- Quantized tensors need explicit paths everywhere they appear: indexer,
  HC down/inject/up, GDN ba, PLE key/value were each a separate load fix.
  Enumerate every quantizable leaf, not just the big projections.
  (64b85c3, dfbe98f, 8682d4e)
- PLE tables in F32/F16/BF16 read without ggml; a SCALED PLE table
  (`per_layer_token_embd.scale`) is refused, not served unscaled. (0ac025b)
- `block_count` includes MTP layers: subtract `nextn_predict_layers` or the
  decoder builds phantom layers. GGUF per-layer PLE dim is ONE head's row:
  multiply by `(ngram_size-1)*heads_per_ngram`. `ple.layers` zero-based ->
  +1 for HF.
- NVFP4 GGUF weight scales must be applied (gguf tensors carry them
  separately); NVFP4 token_embd rows need the vendored get_rows path.
  (31c6ef4, d026ffd, 20d5d93)

## PLE / host tables

- Hash disk rows with the STATE-DICT-LOADED constants, the same source the
  pinned path reads - re-deriving them drifts from the checkpoint. (697dedf)
- Dummy-weight fill leaves int64 hash buffers garbage (zero vocab divides
  by zero): re-derive real constants and attach ZeroTable.
- Size eager staging for `max(max_extend_tokens, max_running_req *
  decode_rows_per_req)` - an eager decode/verify step stages more tokens
  than a prefill chunk; graph rows: verify graphs stop at max_running_req
  requests, decode graphs at cuda_graph_max_bs. (d04a86b)
- Zero the eager staging for the warmup prefill; read back the PADDED
  decode batch for the fill. (ccddade, 6450c3c)
- Fence disk staging against in-flight lookups (a fill overwriting a row a
  kernel is reading corrupts silently); validate row coverage; the graph
  itself never waits on disk - the engine enters forward_host_ctx around
  dispatch. (d3512b4, 2b27ea2)
- The 47.7 GiB table reads O_DIRECT into the pinned bank; buffered reads
  double the host memory. Keep the table object alive on the model.
- Pin-budget ordering: load_host_tables runs AFTER weights (failures not
  masked) and BEFORE expert-bank residency planning. (e3b6d7b engine hook)
- FTW conversion must write the PLE table as side files next to the FTW;
  it is never an FTW entry. (234ecc0)

## Speculative decoding / MTP

- Refuse draft sources the draft cannot read BEFORE the target loads:
  GGUF target + folder draft is invalid (the folder reader would take the
  GGUF quant config); a pack must carry `mtp.*`. (495643f, 41e4f67)
- Require BOTH stacked expert tensors; read MTP head experts on the host.
  (674d671)
- The draft's MoE is forced RESIDENT in the proposer, checked after load.
  (5b46b07)
- Slot states need verify handlers or speculative serving must be refused
  (`SlotStateSpec.verify=None`); verify reads live state only, stages into
  handler scratch, commit rolls by ACCEPTED rows only and may run twice.
  (0939740, SlotVerify contract)
- `extend_config_for_mtp` appends the layer id to the full-attention group
  so slab factory + backend layer->slot map size it everywhere; the draft
  config cuts to one QSA layer without PLE/slot states/vision.
- Speculative logits route through the head projection. (522c326)
- Dual-microbatch decode: the capture-pool allocator reuses freed blocks
  across streams with NO execution ordering - every phase boundary needs
  an event pair when serializing; halves need disjoint static buffers,
  all-reduce instances and split-K workspaces. (model.py forward_dual)

## Scheduler / cache

- Hybrid snapshots donate only at page-aligned chunk ends: one unaligned
  non-final chunk costs every prefix-reuse point for the rest of the
  prompt. Align down; keep the unaligned end only when the leftover budget
  cannot fill a whole unit. (e3b6d7b scheduler)
- QSA slab rows are never cleared: scoring must clamp to
  `kvlen // index_ratio` so stale rows stay unreachable; the pending ring
  is never cleared either (new tenants start at group boundaries).
- `page_size % index_ratio == 0` is what makes `slot // index_ratio`
  well-defined - pin page_size (64) in the backend's `page_sizes`.

## Serving / runtime

- Quantized flash-class checkpoints drift into particle-drop fragmentation
  at temperature >= 1.0 without min_p (measured on this NVFP4 checkpoint):
  the sampling guard adds min_p=0.05, NO frequency penalty (count-linear
  penalties strip legitimate repeated punctuation/newlines). Explicit
  client min_p wins. (docs/cli.md FREETOKEN_SAMPLING_GUARD)
- Default bf16 KV on 2x24 GB holds only ~8k tokens for this model; long
  context needs `--kv-cache-dtype fp8` and/or `--num-tokens`; over-long
  prompts must fail as a clean 400 `context_length_exceeded` with the
  ceiling in the message. (docs/models.md)
- AOT kernel-shape entries must mirror runtime derivations by hand; a
  drifted entry misses the prebuilt cache by spec name and silently falls
  back to JIT (needs nvcc). (kernel/aot_models.py docstring)
- TP over offload-family MoE strategies is refused for non-NVFP4 experts
  (MXFP4 / block-FP8 / bf16 offload readers emit full width): serve those
  resident with `--moe-strategy fused`. Marlin / b12x packs need a wider
  slice than the shard leaves, so the selector falls back to Triton.
  (docs/cli.md "What is TP-sharded today")
