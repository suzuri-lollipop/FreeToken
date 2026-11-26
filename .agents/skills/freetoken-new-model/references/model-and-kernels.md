# Model skeleton, layer modules and kernels

Pattern source: `models/qwen4_exp/{model,hc,attention,moe,ple,mtp}.py`,
`layers/linear_attn/gdn.py`, `kernel/triton/`.

## Model skeleton (model.py)

- `Qwen4ExpModel(BaseOP)`: `VocabParallelEmbedding` -> `OPList` of decoder
  layers -> top-level `GatedResidual(use_combine=False)` mixer. No
  input/post layernorm, no final `model.norm` - the hyper-connection norms
  are the only ones.
- Residual state `R [T, hc_count*hidden]` end to end (stream outer, hidden
  inner - the checkpoint layout): `embed_input_ids(...)` then
  `hidden.repeat(1, hc_count)`; the mixer collapses once before `lm_head`.
- `Qwen4ExpForCausalLM(BaseLLMModel)`: `ParallelLMHead` (tie via
  `tied_embedding=self.model.embed_tokens`), `forward()` reads
  `get_global_ctx().batch`, calls
  `lm_head.forward(model.forward(...), export_hidden=False)` - the model
  hands pre-mix streams to `batch.features` itself.
- Multimodal: `Qwen4ExpForConditionalGeneration(QwenVLVisionMixin,
  Qwen4ExpForCausalLM)` builds `Qwen3VLVisionModel(config.vision_config,
  quant_config=..., prefix="visual")` when `config.is_multimodal`; assert
  no DeepStack features. Reuse the shared tower, do not fork it.
- PLE layers kept in a plain tuple (`self._ple`, not an OP child - never in
  the state dict); `ple_layers` property is the seam `load_host_tables`
  attaches table backends to.
- `supports_dual_decode = True` + `forward_dual(batchA, batchB, side)` for
  dual-microbatch decode (see below).

## Frozen layer contract

```
forward(R [T, hc*hidden], batch) -> R' [T, hc*hidden]

R  = R + ple(R, batch)                 # zero-based layer 1 only
x, s = attn_hc.mix(R); y = (GDN | QSA)(x); R = attn_hc.combine(R, y, s)
x, s = mlp_hc.mix(R);  y = MoE(x);        R = mlp_hc.combine(R, y, s)
```

Perf fusion actually shipped: `attn_hc.combine_norm(R, y, s,
mlp_hc.hc_norm.weight, eps)` returns `(R', mlp_rn)` and
`mlp_hc.mix_from_normed(mlp_rn)` skips the second norm - eliminates one
write+read of the full residual. Keep the sequential torch path as the
reference the fused kernel is diffed against.

## Hyper-connections (hc.py)

`GatedResidual` formulas (HF modeling_qwen4_exp.py:959-969):

```
Rn      = groupRMSNorm(R) * (1 + hc_norm.weight)   # per-stream fp32 stats
lora, s = input_mix_weight_down_block_inject(Rn)   # merged GEMM [lowrank | hc_count | pad]
gate    = input_mix_weight_up(silu(lora / hc_count))
x       = mean_i(sigmoid(gate_i) * Rn_i)
R'_i    = R_i + 2*sigmoid(s_i / hc_count) * y
```

- `s` returned by mix is the RAW inject logit slice (pre 2*sigmoid) - what
  the vendored `hc_combine` kernel expects; combine applies the activation.
- Merged weight `[lowrank + hc_count + pad, hc*hidden]` (Qwen3.8: 320+4+12
  = 336 rows); pad rows zero, their GEMM output dropped; both slices keep
  unit inner stride (no copy).
- `use_combine=False` = top-level mixer: unmerged `input_mix_weight_down`,
  `s = None`, no combine.
- CUDA path: `grouped_gemma_rmsnorm` / `hc_silu` / `hc_gate_mix` /
  `hc_combine` / `hc_combine_norm` around two `F.linear` GEMMs; CPU path:
  pure torch with fp32 intermediates, cast back at the store. Launch budget:
  mix = 3 kernels around 2 GEMMs, combine = 1.
- `GroupedPlusOneRMSNorm(size, eps, num_groups)`: per-group fp32 statistic,
  (1+w) scale; kernel is 2D-only, higher-rank callers keep the torch chain.
  Exported for reuse (ple.py norm_key/query/conv).

## GDN linear-attention layers

Use the SHARED op `layers/linear_attn/gdn.py::GatedDeltaNet` (refactored out
of qwen4_exp in #841; Qwen3.5 shares it):

```python
GatedDeltaNet(hidden_size, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
              conv_kernel_size, rms_norm_eps, layer_id, output_gate,
              quant_config, prefix, out_heads_tiled)
```

- `gdn_local_dims(...)` derives the rank's head split; BOTH GatedDeltaNet
  and `LinearStatePool._linear_local_dims` use it, so fused `in_proj` slices
  whole heads (refuses KV-style replication). Opt-in uneven split via
  registered dense shard weights (`FREETOKEN_DENSE_SHARD_FRAC`).
- State: `conv_states` / `recurrent_states` on LinearStatePool slots;
  metadata via `fla_metadata` (cache_indices, track_dst).
- `out_heads_tiled=True` (GGUF): the layer permutes its input instead of
  untile-ing the packed `ssm_out` rows.
- Kernels: `kernel/causal_conv1d` (varlen / decode / verify), fla-style
  recurrent update; later fused decode kernels in
  `kernel/triton/gdn_fused_layer.py`, `gdn_hc_fused.py` and
  `kernel/fla/gdn_decode_fp16_state.py` (bf16 state prototype) - see
  `docs/qwen38_flash_next_optimizations.md` for the measured tradeoffs.

## QSA attention module (attention.py)

Gated GQA: `q_proj` twice as wide for the output gate.

```
q, gate = chunk(q_proj(x).view(-1, num_q, 2*head_dim), 2, -1)
q, k    = rope(q_norm(q), k_norm(k_proj(x)))      # first rotary_dim dims
o       = backend.qsa_forward(q, k, v_proj(x), indexer(x), layer_id, batch)
out     = o_proj(o * sigmoid(gate))
```

- q/k/v one merged GEMM (`qkv_proj`, split `[num_q*head_dim*2, kv, kv]`);
  checkpoint ships them separately -> loader concatenates along dim 0.
- Per-head zero-centered q/k norms (`GemmaPlusOneRMSNorm`,
  `forward_inplace`), partial NeoX rope (64 of 256 dims).
- `Qwen4ExpIndexer`: `index_qk_proj` = `LinearReplicated` with
  `output_sizes=[index_n_heads*index_head_dim, index_kv_heads*index_head_dim]`
  (Qwen3.8: [640, 2560] = 4 q heads x 128 then 1 k head x 128) + two
  per-head norms. Returns `QSAIndexerInputs` - RAW q/k slices, no norm, no
  rope; both norm weights travel with the call because the index k norm
  runs AFTER the fp32 mean over each group of `index_ratio` raw keys.
- rope64 for the indexer: same frequencies as main attention, different
  head_size - the backend builds its own cached `get_rope` instance.
- `QSAAttentionBackend` Protocol is the seam; `attention/qsa_sparse.py`
  implements it; `TorchDenseQSAReference` (fp32, own [slot, position] KV,
  no engine wiring) is the test oracle - never registered as a backend.

## MoE (moe.py)

Subclass the family MoE (`Qwen3_5MoE`), override only the epilogue:

```python
router_logits = self.gate.forward(hidden_states)
shared = self.shared_expert.forward_partial(hidden_states)
gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
routed = self.experts.routed_partial(hidden_states, router_logits)
combined = shared_gate_mul_add(routed, shared, gate)
return self.experts.reduce_partial(combined)
```

- Gate reduction BEFORE the routed experts (they may write `hidden_states`
  in place); mul-add after.
- TP: shared + routed partials summed BEFORE one all-reduce (gate is
  rank-replicated: `AR(routed) + AR(shared)*gate == AR(routed + shared*gate)`).
- Router: non-power-of-2 top-k (top-10) supported by the fused triton
  router; `norm_topk_prob` renormalization matches the reused block.
- Routed experts: NVFP4 / block-fp8 banks through the offload cache
  (`nvfp4_expert_spec`), never the dense state dict.

## PLE layer (ple.py)

Per token (HF Qwen4ExpTextNGramEmbedding / Qwen4ExpTextPLELayer):

```
E = table[hash(ngram)]                       # 16 heads (8x2-gram + 8x3-gram) x 160
K = norm_key(key_proj(E)).view(hc, hidden);  V = value_proj(E)
Q = norm_query(R).view(hc, hidden)
u = <K_i, Q_i> / sqrt(hidden)                # per stream
U = sigmoid(sign(u) * sqrt(max(|u|, 1e-6))) * V
D = U + silu(conv1d(norm_conv(U)))           # depthwise, kernel 4, dilation ngram_size
R += D                                       # before the attention HC mix
```

- Hash: splitmix64-derived per-layer prime multipliers, per-head vocab sizes
  (base 20M rounded to 128) and offsets; `derive_ngram_hash_constants` is
  the single source; windows never cross the eos boundary token; image
  placeholder tokens restored via `freetoken.mm.restore_placeholder`.
- `PLETableBackend` Protocol (frozen): `lookup(row_ids [T, heads] int64
  device) -> [T, heads*head_dim]` dequantized (fp8 -> dtype times scalar
  scale), optional `out` buffer reused by CUDA-graph decode; `prefetch` may
  start the gather on a side stream (model issues it before layer 0, joins
  in lookup). Rows addressed by GLOBAL hashed id (offset already added).
  Out-of-range ids store zeros.
- Backends: `PinnedUVATable` (47.7 GiB pinned host, GPU dereferences over
  PCIe - host VA on Linux/UVA, mapped device address on WDDM via
  `kernel/pinned.device_ptr`), `GpuResidentTable` (small-table oracle /
  `--ple-backend gpu`), `ZeroTable` (dummy weights), `DiskRowTable`
  (ple_disk.py, default).
- Conv left context + ngram window ride slot states (`ple_conv`,
  `ple_ngram_ctx`); prefill writes the track snapshot after the chunk's
  rows; separate `_decode_conv` / `_verify_conv` / `_prefill_conv` paths.
- `build_ple_metadata(batch, args, device)` once per forward;
  `commit_ngram_context(meta, fla_metadata)` single writer after the layer
  loop; verify path stashes into the verify handler instead.

## MTP draft (mtp.py)

Class contract the spec machinery resolves:
- `attention_backend = "qsa_sparse"`; `draft_config = staticmethod(...)` -
  target config cut to one QSA layer (id 0), no PLE, no slot states, no
  vision;
- `relay_width(config)` - hidden width the target relays
  (`hc_count * hidden`: the pre-mix residual streams, exported through
  `batch.features.final_hidden`);
- `check_source(path, target_path)` - refuse unreadable sources BEFORE the
  target loads: GGUF draft must carry `mtp.*`; a GGUF target needs a GGUF
  draft (Strata pack), not a folder; folder draft must carry `mtp.*`;
- `load(cls, config, target_model, model_path, device, dtype)` - build on
  meta device, strict state-dict match (unexpected AND missing both raise),
  routed experts packed by the layer's own expert method and attached as
  RESIDENT banks (`attach_resident_banks`), `finalize_quant` per module
  (not the root - it reaches the target's already-finalized embedding);
- `embed_tokens` property -> the target's own module (underscore attribute
  keeps it out of the draft state dict);
- `forward_hidden(embeds, prev_hidden)`:
  `e = fc_embedding(norm_e(embeds))`;
  `R_in = fc_hidden(norm_h(R_c)) per stream + e` (norm_h: ONE RMS statistic
  over all hc*hidden); one decoder layer; `head_input` = mixer.mix.
- Strata pack path: GGUF with HF `mtp.*` names, stacked ggml experts;
  `_pack_tensors` / `_iter_pack_weights` / `_pack_expert_banks` /
  `_pack_quant` (GGUFConfig with the target's packed mapping).
- Target side: `extend_config_for_mtp` sizes the extra QSA layer; the
  loader keeps `mtp.*` dense weights only when enabled.

## Dual-microbatch decode (forward_dual)

Two half-batches on skewed CUDA streams; per layer the leading half issues
first, MoE cross-waits overlap one half's PCIe miss fetch with the other's
SM work. Each half: own all-reduce instance (`ar_instance(slot)`), QSA
staging profile (`dual_profile(slot)`), split-K workspace partition
(`set_ws_slot`), `batch.dual_slot`. Static buffers disjoint; LRU bookkeeping
serialized by per-layer events. `FREETOKEN_DUAL_SERIAL=1` fully serializes
with event pairs at every phase boundary (correctness probe: the
capture-pool allocator reuses freed blocks across streams with no execution
ordering). PLE meta passed via `batch.ple_meta_pre` (the start_prefetch pair
would race on the single _pending slot).

## Feature sink (draft support)

`batch.features.wanted_layers` -> `features.layer_output(i, hidden,
num_streams=hc_count)` inside the layer loop; `features.final_hidden(hidden)`
before the final mix (a draft reads the residual streams pre-mix). Read the
declared `Batch.features` directly, not through getattr (refactor 20de52d).

## Kernel vendoring rules

- Keep SPDX headers naming the source project and file (vLLM / SGLang
  Apache-2.0 adaptations; original kernels get a plain docstring).
- Every Triton kernel keeps a pure-torch reference in the model module;
  tests diff them to fp32 rounding.
- int64 row addressing; `num_token_non_padded` row masks read the DEVICE
  tensor so they survive CUDA-graph capture.
- Graph-capturable: no host syncs inside capture, static staging buffers,
  `prepare_for_replay` for decode metadata.
- New kernel shapes need an `kernel/aot_models.py` entry or the prebuilt
  cache misses and JIT (nvcc) kicks in.
