# Lessons: silent failures the fix history already paid for

Extracted from the ~120 follow-up commits on `models/qwen4_exp/` and its
subsystems, plus the registration gaps the ERNIE 5.5 addition exposed.
Each entry: the failure mode -> the rule. Commit hashes are this
repository's history (merge duplicates exist; one hash per change); entries
without a hash name the change they came from. Sweep this list before
calling a new-model phase done.

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
- A checkpoint's `scale_fmt` can describe the EXPERT scales while the dense
  projections store F32 `weight_scale_inv`: override the dialect at the
  ModelSpec (`quant_overrides=(("scale_fmt", ""),)`) or the fp8 reader
  misreads the dense scales. ERNIE 5.5 spells the OCP expert form "mxfp4"
  where V4 spells it "fp4" - accept both in the dialect. (ernie5_5_moe
  register.py / quantization/configs/fp8.py)

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

- A streaming tool-call detector that re-parses the whole growing call JSON every
  token (`BaseFormatDetector` + `partial_json_parser`) is O(n^2): a 20 KB argument
  stalled the frontend for ~160 s, the stream went silent past the 90 s frozen
  watchdog, and the supervisor rebuilt the backend mid-turn - a lost coding turn
  that looks like a model crash, not a parser bug. A drain re-feed that loops on an
  unchanged buffer (no progress guard) multiplies it ~21x. Rule: new families stream
  with an incremental cursor (string values char-by-char, typed values buffered),
  the `InvokeParamStreamMixin` / `Glm47Detector` / `Inkling2Detector` shape; an empty
  re-feed with an unchanged buffer returns immediately. Guard the re-parse count in
  a test. (inkling2 streaming migration, server/function_call_parser.py,
  tests/server/test_function_call_parser.py)

## Delivery completeness / family registration

- Parser registration is FOUR independent tables: the detector/parser class
  maps, the streaming opener tag, the auto cascade, and the explicit CLI
  choices. ERNIE 5.5 shipped the classes and the cascade but not the
  choices: `auto` worked, an explicit `--tool-call-parser ernie55` was
  rejected by argparse - silent partial support no parser test catches,
  because they all start from a name the factories know. Rule: register all
  four; the bookkeeping test drives the live registry's inferred names
  through the explicit choices. (ernie5_5_moe addition,
  tests/server/test_parser_auto_selection.py)
- `tp_supported` left unset ships a family TP1-only. ERNIE's MXFP4 offload
  experts emit full width, so TP is refused until the banks shard per rank
  (NVFP4 path) or serve fused. Rule: TP >= 2 is a MANDATORY requirement - do
  the per-rank expert sharding and set the claim; TP1-only is an incomplete
  delivery, not a deferral. (ernie5_5_moe addition; docs/cli.md "What is
  TP-sharded today")
- MXFP4 expert TP is layered, not one flag. The standard OCP kernel
  (`mxfp4.py` TritonMxfp4MoEKernel) was `tp_ok=False` + full-width layout. The
  bank-sharding step (done for ERNIE): layout uses `cfg.local_intermediate`,
  pack cuts the pieces with `tp_rows(gu, 1, ...)` / `tp_piece(down, 2, ...)`
  -- the pieces carry the expert axis FIRST, so the intermediate is dim 1 of
  the fused gate|up (2*total rows) and dim 2 of down (packed cols /2, e8m0
  scale cols /GROUP); TP1 stays byte-identical. The all-reduce is NOT in the
  kernel: `MoELayer._maybe_all_reduce` already sums `routed_partial` across
  ranks, and the kernel infers the intermediate from the bank shape
  (`gate_up_packed.shape[1]//2`), so per-rank banks just work -- verified
  rank0_partial + rank1_partial == TP1 full (prefill grouped GEMM; the decode
  GEMV has a K//2 % 128 == 0 floor, so the local intermediate must be a
  multiple of 256). With that proved, `tp_ok=True` + the reader handing full
  experts (no reader-side shard) completes the EXPERT TP. Dense MLP TP (done):
  `Ernie55MLP` gate/up -> `LinearColParallelMerged`, down -> `LinearRowParallel`
  (TP1 byte-identical); the reader's `_mlp_shard` cuts gate/up rows + down cols
  per instance -- the dense layer-0 MLP uses `intermediate_size`, the shared
  expert `moe_inter_dim*n_shared`, so the shard is built per instance, not from
  the generic `_leaf_units` (one intermediate). KEY design choice: keep the
  attention REPLICATED (MQA shared latent KV + the VHA cross-head mix couples
  all heads, so sharding query heads would need an all-gather before VHA);
  replicated attention + mHC, sharded MLPs + experts is a valid TP config that
  saves the dominant expert memory. The reader also shards the vocab
  (embed/lm_head, `_vocab_shard`, ceil cut); TP=2 model build + sharded dense
  load is verified (local MLP buffers, finite weights). STILL remaining for
  `tp_supported=True` end-to-end: the engine TP gate (engine.py ~467) requires
  a backend with `supports_tp=True` reading TP-local heads against a TP-local
  KV pool -- `ernie55_sparse` is `supports_tp=False` and the DSV4-based
  Ernie55PagedKVCache is TP1-only, so the replicated-attention design needs
  either sharded query heads + a VHA all-gather, or a replicated-attention TP
  mode (backend reads full heads, KV pool replicated) the engine currently
  refuses. `tp_supported=True` + `ernie55_sparse supports_tp=True` are set and
  the config-time gates pass at TP2 (query heads split, the single MQA KV head
  replicates, expert intermediate splits). A 2-rank NCCL forward harness exists
  (mirror inkling2/test_vision.py: mp.spawn + init_process_group, per-rank
  TinyEngine, diff TP1 logits). TP2 FORWARD NOW PASSES (rel_err < 0.05 vs TP1, both
  bf16 and the default W8A16 embed). The NaN hunt, localized by per-stage finite
  probes, ended at the EMBED, not mHC/attention: the layer-0 mhc_pre input was
  ~2e38 because the vocab-parallel embedding gathered garbage. Two real fixes:
  (a) kernel/index.py `indexing()` must zero the fresh output when vocab_range is
  set -- masked_index_kernel only writes in-range rows (pos is unsigned, so ids
  below start wrap past length and are skipped), leaving out-of-range rows as
  uninitialized garbage that the TP all-reduce sums into the embedding;
  (b) the vocab shard must materialize the tensor before the rank cut
  (_vocab_shard clones), and the harness reads dense weights on CPU then moves to
  device (a direct safetensors GPU load of the far rank's rows returned NaN).
  mHC/attention are replicated (no TP split) and were never the fault. The tiny
  geometry needs intermediate >= 512 so the TP2 slice clears the fp8/decode block
  floors (geo_overrides). Logits agree to a few percent relative: the bf16
  vocab/MLP all-reduce rounding is amplified by the mHC Sinkhorn (20 iters,
  alpha_res 5.3); expert partials alone match to 3e-5. The draft is separate.
  (ernie5_5_moe expert + dense MLP/vocab TP,
  tests/models/ernie5_5_moe/test_tp.py::test_tp2_forward_matches_tp1)
- A bundled draft head that is not served is a gap, not a finish. ERNIE ships
  `model.mtp.*` (DSpark) but registers no `spec_drafts`, AND its HCA layers are
  DSV4 window-tier KV, which `_adjust_spec_config` refuses for spec regardless
  of a draft. Rule: drafting is MANDATORY when the checkpoint ships a head -
  deliver BOTH a served draft class and a target the spec gate accepts
  (inkling2's SWA was lifted only after its eviction audit). (ernie5_5_moe
  addition, engine.py swa_spec_ok / _adjust_spec_config)
- `spec/dspark/proposer.py` (DSparkProposer) is Inkling-2-coupled: it imports
  `models/inkling2/draft.py` (3 dense draft layers, Markov head, taps at the
  loop visits 5/23/41) and the ik2 linear-state slots. A different family's
  DSpark head needs its own draft class + proposer subclass AND the target's
  FeatureSink taps, not a reuse of the ik2 proposer. DONE for ERNIE:
  `models/ernie5_5_moe/draft.py` (Ernie55DSparkDraft, the same DSpark math,
  geometry from dspark_* args, loads model.mtp.* through the family reader),
  `spec/dspark/ernie55.py` (Ernie55DSparkProposer subclasses DSparkProposer,
  reuses the context-ring/Markov state machine, overrides geometry + taps),
  FeatureSink taps in the model prefill/decode (hc_contract of the mHC stream at
  dspark_target_layer_ids + final_hidden), spec_drafts registered, and the
  synthetic builder writes model.mtp.*. The draft math is pinned by a parity
  test against the inkling2 DSparkDraft with identical weights (propose equal).
  The proposer attach + propose run on the real GPU. The slot_states shape's
  leading axis is K/V (=2), NOT the layer count (layer_ids carries the layers).
  The spec gate lift is an eviction audit, not a flag flip: ERNIE's HCA window
  tier is a WindowTierPagedPool (swa_paged), so the CacheManager's
  maybe_free_swa_out_of_window already floors frees at req.cached_len under spec
  (scheduler/cache.py) - the same committed-frontier audit as inkling2's SWA -
  so is_ernie55 joins swa_spec_ok. Two more integration pieces were needed and
  added: (a) spec/runner.py proposer_class must consult model_spec.spec_drafts
  BEFORE the global PROPOSERS registry, else the method name "dspark" resolves to
  the inkling2 proposer for every family; (b) the draft context state rides the
  linear-state slots, but ERNIE has no GDN tower, so engine.py builds a zero-layer
  LinearGatedDeltaGroupConfig ("spec_state") when config.slot_states is non-empty
  and linear_attention_group() is None, giving the proposer its slot_states pool.
  For a full engine spec run the DSV4 multi-row decode graph capture is being
  built (a separate backend task from the window-tier eviction audit): the
  ernie55_sparse BackendInfo now declares uniform_extend_graph="host_plan";
  dsv4_sparse.init_capture_graph sizes the snapshot in rows (max_bs*rows_per_req)
  instead of raising; prepare_for_capture stages batch.padded_size rows; the
  ernie55_sparse DSA buffers + _point_to_capture are row-sized and tolerate a
  capture batch with no attn_metadata; state_pool_bytes / verify_scratch_bytes
  accept a zero-layer "spec_state" group when slot_states exist without a GDN
  tower; and the ERNIE model forward sizes decode off input_ids.size(0) (the
  verify's k+1 rows/request), not padded_size. The full engine now CONSTRUCTS and
  GENERATES end to end with dspark on the real GPU. Fixes that got it there:
  prepare_for_capture stages padded_size*rows_per_req rows (padded_size is the
  REQUEST count); the ernie55 DSA kv_len is repeat_interleave'd per verify row;
  n_scratch (create_kv_pool + v4_pool cost model/solver) scales by decode_rows_per_req
  so each k+1 verify row gets a compress scratch row; store_window routes -1
  (unmapped / warmup dummy) to the reserved last dummy slot instead of index_copy_
  OOB. The LLM path also needs the family model_type registered with transformers
  AutoConfig (a PretrainedConfig subclass with model_type + max_position_embeddings
  set before super().__init__, since rope standardization reads it). RESOLVED: the
  spec output diverged from plain greedy decode; two REAL correctness bugs (not a
  bf16 near-tie) were the cause, found by layer-by-layer anchor-row taps (hook
  Block.decode_step / Transformer.decode, compare row 0 at the same position; a
  verify's non-anchor rows carry DRAFT inputs, so only the anchor row is comparable):
  (1) The block-fp8 dense projections (wkv/qkv/ffn/...) ran the W8A8 block GEMM for
  the k+1 verify rows but the W8A16 split-K GEMV for the 1-row decode. That is NOT
  an ulp: the GEMM quantizes the activation to fp8 (per-128-group), the decode GEMV
  keeps bf16, a ~2e-2 relative difference that drifts the stored KV. (The earlier
  "bf16 GEMM rounds per batch shape" guess was WRONG - F.linear and the fp8 act-quant
  are row-wise batch-invariant; it is the quant SCHEME that switches at M=1.) Fix:
  kernel/triton/fp8_block_linear.block_fp8_gemv_rows runs the exact M=1 decode GEMV
  per row (bit-identical by construction; a fused multi-row kernel re-layouts the
  128-K tl.sum and loses identity at large N/K), and TritonFp8BlockLinearKernel.apply
  dispatches to it for a single-request verify (batch.phase=="verify" and size==1).
  Wider verify batches keep the GEMM: plain decode at bs>1 uses it too and it is
  row-wise M-invariant. (2) The DSA trailing window for kvlen < window used
  `kvlen - window + j`, which goes negative and clamps every slot to row 0 (attends
  position 0 on repeat). Fix: `kvlen - wlen + j` with wlen=min(kvlen,window) in
  _dsa_window_rows AND _dsa_select_prefill. The compressor carry rollback (step one
  verify row at a time; commit_verify restores the pre-verify block and replays the
  accepted rows) is correct but was never the cause - the compressed tier is empty
  at these early positions. RESIDUAL after both fixes: a verify-vs-decode ulp from
  the attention splitk(m=1)-vs-single(m>1) reduction, amplified by the mHC residual
  (alpha_res) and MoE routing flips, can part the runs at a non-tie on the raw
  synthetic weights. This is the fork's documented verify-contract behavior; the
  lever is fan-in scaled weights (make_checkpoint fan_in=True) keeping activations
  O(1). With it, served DSpark reproduces plain greedy token-for-token (eager AND
  cuda-graph, k=1 and k=7). ERNIE is bf16-only (KV pool asserts), so the fork's fp16
  lever is unavailable. (ernie5_5_moe draft + spec parity,
  tests/models/ernie5_5_moe/test_draft.py, test_reference_parity.py)
- No GGUF reader means the family cannot serve the GGUF checkpoints llama.cpp
  converts and users download. Rule: GGUF support (gguf.py + arch registration
  + alias row) is a MANDATORY requirement, not a scope choice. (ernie5_5_moe
  addition shipped HF-only)
- `strata-gguf-py` is NOT Strata's own GGUF stack: its UPSTREAM.md says it is
  llama.cpp's gguf-py + converter copied byte-identical. "Reference Strata's own
  implementation" means `terminal/tools/gguf_reader.py` / `gguf_writer.py` (the
  own-stack reader that handles MXFP4=39 / Q2_0=42, which gguf-py cannot). For
  an HF-name-preserving GGUF (no llama.cpp transforms), share the HF dense name
  map and supply the vocab via `tokenizer.ggml.tokens` (the shared `_vocab_size`
  wants llama.cpp's `token_embd.weight`); the MXFP4 expert banks need their own
  read, not the `blk.N.ffn_*_exps` loader. (ernie5_5_moe GGUF,
  tests/models/ernie5_5_moe/test_gguf.py)
- Synthetic-harness verification hides the REAL serving path. Booting the actual
  ERNIE checkpoint at TP2 + int8 KV surfaced three gaps the synthetic tests never
  hit: (1) the engine TP offload gate hardcoded `nvfp4` and refused `ds_fp4` even
  though the MXFP4 kernel declares `tp_ok=True` and shards its banks - the gate must
  list every format that actually shards; (2) the TinyEngine harness builds the
  pool/model directly and BYPASSES the config gates, so a green synthetic TP2 test
  is not evidence the server boots at TP2; (3) ERNIE ships a custom ErnieChatTokenizer
  via `auto_map`, so `load_tokenizer` needs `trust_remote_code=True` AND a registered
  `PretrainedConfig` (Ernie55HFConfig, `max_position_embeddings` set before
  `super().__init__` for RoPE standardization) - the engine's RawConfigShim does not
  cover the tokenizer worker's AutoConfig call. Rule: download the real checkpoint
  and boot it through `ft serve` end to end. (ernie5_5_moe real serving,
  engine/config.py TP gate, models/ernie5_5_moe/config.py, utils/hf.py load_tokenizer)
- int8 KV for a hybrid window + full-history family quantizes the FULL-HISTORY slab
  (the long-context memory consumer), not the bounded window: ERNIE's DSA latent slab
  rides `store_latent_rows` (per-row ckv/kpe int8 + fp32 scale pair, read back through
  glm_dsa_sparse_attn's kv_format arm), while the HCA window/compressed tiers and the
  DSA index slab stay bf16. The pool declares `kv_formats=ROW_SCALE_FORMATS`, the
  backend passes `kv_format`+`scale`, and the cost model budgets the int8 row bytes.
  int8 is TP-safe (rank-local scales); the DSA slab is replicated under ERNIE's MQA
  replicated attention. (ernie5_5_moe int8 KV, kvcache/dsv4/ernie55_pool.py,
  attention/ernie55_sparse.py, tests/models/ernie5_5_moe/test_reference_parity.py)
- RAM-offload decode speed: the fork's decode fetch-overlap (hit-route GEMV on the
  compute stream while the miss experts' H2D runs on a side stream, then the miss
  GEMV, merge) was NVFP4-ONLY -- gated by the kernel's `supports_skip_w0`. The ds_fp4
  (MXFP4) decode GEMV lacked it, so ERNIE decode always serialised fetch+compute.
  Ported SKIP_W0 to `_decode_dsfp4_moe_kernel` (a zero-weighted route zero-stores and
  returns BEFORE reading its slot, so the concurrent copy is never read mid-write;
  gate_up carries the weights only to skip, down applies them) and set
  `supports_skip_w0=True` on TritonMxfp4MoEKernel. The hit+miss merge matches serial
  to bf16 sum-order noise (the NVFP4 overlap tests allow rtol 3e-2, not strict bits).
  MEASURED: at bs=1 the overlap is a wash -- 4.79 vs 4.99 tok/s (the narrow hit GEMV
  does not hide the H2D, and the two-pass fork/join costs more), so the
  `_FETCH_OVERLAP_MIN_ROWS=2` default is correct; the win is bs>=2.
  (kernel/triton/dsv4/fused_moe.py, moe/fused_ds_fp4.py,
  layers/quantization/moe/mxfp4.py, tests/moe/test_dsfp4_reference_semantics.py)
- Hot-expert GPU residency audit: the fork keeps hot experts in VRAM via the LRU slot
  cache (implicit -- frequently routed experts stay), plus opt-in prefill promotion
  (FREETOKEN_PREFILL_PROMOTE_FRAC / auto_promote) that slot-caches prefill misses for
  cross-request reuse. `--moe-hot-frac` / `_resolve_hot_experts` pin the profile's
  hottest experts in compact HOST banks for DISK layers only -- not GPU, and not
  applied to ERNIE's RAM-pinned offload. There is no explicit frequency-based
  "pin top-N decode experts in GPU" policy beyond LRU. (moe/offload_cache.py,
  engine/engine.py _resolve_hot_experts)
- The calibration tool's `spec_backends_for` offers mtp + dflash to unknown
  families; a family that refuses speculation at config time burns a doomed
  arm on every multi calibration. Rule: register the model_type - no arm for
  a refusal, the family's own drafter where one exists (inkling2 -> dspark).
  (ernie5_5_moe addition, calibrate_multi.py)
