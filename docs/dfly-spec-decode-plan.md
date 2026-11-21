# DFly2 speculative decoding for hy4-flash — implementation plan

Reference: vLLM PR #68296 (merged, head `8cbb80ef`, repo `jeejeelee/vllm`),
files under `vllm/models/hy_v4/nvidia/dfly*.py`,
`vllm/models/hy_v4_flash/dfly_config.py`, `vllm/v1/worker/gpu/spec_decode/dfly/`,
and `docs/features/speculative_decoding/dfly.md`. DFly = AngelSpec
(arxiv 2607.25852), a checkpoint-native multi-token draft that reuses target
hidden states, sparse attention indices and KV storage.

## What the checkpoint ships (`model.dfly_layers.*`, 117 tensors)

- 5 draft decoder layers (`dfly_num_hidden_layers=5`), each a gated-MLA
  `self_attn` (q_a/q_b/kv_a/kv_b/linear_gate/o_proj/learnable_sink/post_proj_norm)
  plus a **dense** MLP (gate/up/down, block-fp8) and input/post_attention norms.
  No engram, no latent MoE (`engram_layer_ids=None`,
  `routed_expert_hidden_size=None` in the derived draft config).
- `mtp_fusion` (`enorm`, `hnorm`, `eh_proj`, `gate`):
  `fused = eh_proj(cat(enorm(anchor_embed), hnorm(target_hidden)))`;
  `out = anchor_embed + gate * fused` (`dfly_mtp_fusion='slot0'`).
- `hidden_correction` (`hidden_norm`, `embed_norm`, `gate_up_proj`, `down_proj`):
  `delta = down_proj(silu_and_mul(gate_up_proj(cat(hidden_norm(h), embed_norm(prev_embed)))))`;
  `out = h + delta` (`dfly_enable_hidden_correction=True`).
- `final_norm`, `confidence_head` (the latter is ignored by vLLM).
- Draft reuses the target `embed_tokens` and `lm_head` (`has_own_lm_head=False`).
- `dfly_target_index_layer = dfly_target_kv_layers = [25,29,33,37,41]` (one sparse
  target layer per draft layer); `dfly_target_index_topk=2048`;
  `dfly_mask_token_id=120009`; `query_width = num_speculative_tokens+1 = 6`;
  `index_topk = round_up(2048 + context_local_window + 6, 128)`.

## freetoken scaffolding already present

- `engine/spec_protocol.py:SpecRuntime` (propose_into / verify_step /
  capture_graphs / slot_states / feature_sink / on_prefill).
- `spec/runner.py:SpecRunner`, `spec/proposer.py:Proposer`,
  `spec/mtp/proposer.py:MTPProposer` (multi-step chain drafter, 425 lines) and
  `spec/dflash/` — the template to mirror.
- `models/qwen3_5_moe/mtp.py` (157-line draft: `draft_config`, `relay_width`,
  `check_source`, `load`, `forward_hidden`, `head_input`).
- `models/register.py:ModelSpec.spec_drafts`; `--speculative` /
  `--speculative-num-steps` / `--speculative-draft-model-path` server args.

## Status

- **Phase 1 DONE (verified).** `models/hy_v4_flash/dfly.py`: `DraftAttention`
  (target indexed MLA geometry, no indexer; q_a/kv_a/kv_b bf16, q_b/linear_gate/
  o_proj block-fp8), `DraftMLP` (plain SwiGLU inter 9216), `DFlyBlock` (pre-norm,
  no iHC), `DFlyFusion`, `DFlyHiddenCorrection` (inter 3072, bf16), `Hy4DFlyDraft`,
  `draft_args`, `iter_dfly_weights` + `_tp_cut_dfly`, `load_dfly_draft`. Loading
  parity on `tencent/Hy4-flash-FP8`: **115/115 keys, 0 missing/unexpected/shape
  mismatch** (117 checkpoint tensors minus the 2 ignored `confidence_head`).
- **Phase 2 design.** vLLM aliases the draft KV storage to the target layer
  (`kv_sharing_target_layer_name`, equal `data_ptr`) and snapshots the target
  indexer `topk_indices` for the draft (`dfly_target_reuse.py` /
  `build_dfly_reuse_indices`). freetoken's `Hy4DSABackend` (extends
  `Glm5NextDSABackend`) keeps per-layer MLA latent rows (`MLAKVCache`/`DSAKVCache`)
  and per-layer compressed-indexer selection (each indexed layer its own leader).
  The port must: (a) let a draft layer read its paired target layer's latent pool
  + selection (pass the target `layer_id` for KV reads while using draft
  projections), (b) write the draft query-block KV into that aliased slot,
  (c) snapshot/reuse the target indexer top-k (target_index_topk=2048,
  query_width=6, index_topk=round_up(2048+window+6,128)), (d) skip draft KV
  allocation (alias). This is the bulk of the risk: a wrong alias/index yields
  low acceptance or incorrect tokens.
  - **DONE (verified):** `attention/dfly_reuse.py:build_dfly_reuse_indices` — the
    vectorized, graph-safe port of vLLM `_build_dfly_reuse_indices_kernel`
    (history | local window | causal draft block, `dfly_out_width` =
    round_up(topk+window+query_width,128) = 2176 for hy4). Pinned against a naive
    scalar transcription by `tests/models/hy_v4_flash/test_dfly.py` (3 tests, 8
    random seeds + the hy4 zero-window case).
  - **DONE (imports clean, hy4 suite 22 passed / 2 skipped, no regression):**
    `DraftAttention.forward` (full gated-MLA projection math mirroring
    `Hy4Attention.forward`: q_a/kv_a norms, q_b/kv_b absorption, interleaved rope,
    linear_gate, post_proj_norm) calling `Hy4DSABackend.dfly_mla_forward`, which
    stores the draft query-block latent rows into the paired target layer's pool
    (`store_kv(..., target_layer)`) and `_attend`s over `latent_rows(target_layer)`
    with `md.dfly_sel[layer_id]` (new `DSAMetadata.dfly_sel` dict).
  - **TODO (Phase 3, the bulk):** the DFly proposer (`spec/dfly/proposer.py`,
    mirror `spec/mtp/proposer.py` 425 lines). Unlike MTP (private draft KV chain +
    `chain_backend`), DFly binds the draft backend to the TARGET KV pool and must:
    `feature_sink` capture the target final hidden + the target indexer selections
    (`md.sel[L_i]`) + last sampled token; `propose` build the query block
    (bonus + `dfly_mask_token_id` at anchor..anchor+k), snapshot the target
    selections into `build_dfly_reuse_indices` → `md.dfly_sel`, drive
    `Hy4DFlyDraft.forward_hidden` (fusion + 5 layers + hidden_correction) for k
    steps and sample via the gathered target head; manage draft KV write/rollback
    on the aliased target slots; `tp_supported` via gathered-vocab argmax. Then
    Phase 4 (verify + graph capture + slot_states/budget) and Phase 5 (real
    acceptance/tok/s). Register `spec_drafts={"dfly": ...}` + a `dfly` speculative
    algorithm in the runner.

## Phase 3 status + first end-to-end blocker (2026-11-20)

- **DONE:** `spec/dfly/proposer.py:DFlyProposer` (block drafting: query block
  [bonus + mask], `mtp_fusion` at the anchor, `forward_block` over the 5 draft
  layers via `dfly_mla_forward`, then `draft_tokens` hidden-correction + gathered-
  head argmax loop; `build_dfly_reuse_indices` from snapshotted `md.sel`;
  `radix_lookahead=False`, `tp_supported=True`, `num_steps=(5,5)`), registered in
  `spec/__init__.py:PROPOSERS["dfly"]` + `spec_drafts` on both hy4 registry entries.
  Imports clean.
- **BLOCKER (smoke test `--speculative-algorithm dfly --speculative-num-steps 5`):**
  `engine.py:_adjust_spec_config` rejects the target backend —
  `attention_backend_info("dsa").uniform_extend_graph == "none"`, and spec decode
  requires a backend that captures multi-row verify graphs (fa/triton/fi). The
  SWA window-free guard passed (hy4 keeps a full radix pool, no window-tier frees),
  so the sole config-time blocker is **dsa multi-row verify-graph capture**.
  Resolving it means teaching the hy4 `dsa` backend to capture a verify-shaped
  (k+1 rows/request) extend graph — the `_prefill` path already attends multi-row
  eagerly, so the work is graph-capturing it (static shapes, device-only selection)
  and declaring `uniform_extend_graph` accordingly. The `qsa_sparse` backend
  (592 lines, `uniform_extend_graph="device"`, static-buffer `prepare_for_replay`)
  is the template.
  - **DONE (verified):** `attention/dfly_reuse.py:verify_select_pools` — the
    device-side, static-shape batched causal pool top-k (generalizes
    `indexer_select_prefill` to `[bs, m, n_blocks]` with per-row positions; live =
    (p+1)//ratio, pools past live masked/-1). Pinned against the scalar rule by
    `test_dfly.py::test_verify_select_pools_matches_scalar_prefill` (6 seeds). This
    is the graph-safe selection core the verify graph needs.
  - **DONE (imports clean):** `Hy4DSABackend._select_verify_indexed` — fully
    device-side, static-shape batched verify selection (batched `indexer_logits`
    over `[bs, m, H, D]` queries vs gathered pool keys, `verify_select_pools`
    causal top-k, `_expand_groups` to token rows; `max_pools` is the capture-time
    constant). No per-request host loop.
  - **TODO:** route the verify forward through `_select_verify_indexed`, add the
    static-buffer multi-row capture + `prepare_for_replay` (`qsa_sparse` pattern),
    declare `uniform_extend_graph="device"` for dsa, then Phase 5 iterative
    end-to-end debugging (verify-hidden relay indexing, snapshot timing, draft KV
    rollback on the aliased target slots, acceptance/tok/s).

## Phase 5 iterative debugging (2026-11-20)

Smoke tests (`--speculative-algorithm dfly --speculative-num-steps 5`) surfaced and
fixed, in order:
1. **dsa multi-row verify graph** config rejection → declared
   `uniform_extend_graph="host_plan"` for dsa + implemented `_select_verify_indexed`
   (device-side batched selection). The full static-buffer capture
   (`prepare_for_replay`, qsa_sparse pattern) is still TODO.
2. **Draft quant/layout**: the draft was built with the target's NVFP4 quant and
   loaded from the NVFP4 checkpoint, which **drops `model.dfly_layers.*`** (0 dfly
   tensors vs 117 in FP8). Fixed `load_dfly_draft` to always use `Fp8BlockConfig`
   for the draft (vLLM: draft BF16/FP8 only) — verified: 115 keys load from the FP8
   release, MLP `Fp8BlockLinearMethod`, q_a/kv_a/kv_b `UnquantizedLinearMethod`.
   **An NVFP4 target must set `--speculative-draft-model-path` to the FP8
   checkpoint** (or the NVFP4 conversion must preserve dfly_layers).
3. **TP sharding** (smoke test: `q_b_proj model (8192,1024) vs checkpoint
   (4096,1024)`): the draft built full-width heads/MLP while `iter_dfly_weights`
   TP-shards. Fixed `DraftAttention.num_heads = n_heads // tp`, `DraftMLP` /
   `DFlyHiddenCorrection` gate/up to local-intermediate (`inter // tp`, down stays
   full-width `LinearRowParallel`), and `o_proj` to FULL head width
   (`LinearOProj` shards internally). **Verified: `load_dfly_draft` loads 115 keys
   at TP=1 and TP=2.**
4. **Draft backend layer count** (scheduler `IndexError: tuple index out of
   range`): the draft `Hy4DSABackend` built with `config.num_layers=42` (target)
   while draft `swa_layers` has 5 entries. Fixed `load_dfly_draft` to set
   `dconfig.num_layers=DFLY_NUM_LAYERS`.
5. **Draft ModelConfig rebuilt** (`draft_model_config`): attention group over
   layers 0..4 (`num_index_layers=5`), `moe_enabled=False`, `num_layers=5`.
   Verified TP=2 loading 115 keys; resolved the scheduler `IndexError`.
6. **CORE BLOCKER (precise):** `dsa.py:353 init_capture_graph` raises
   `NotImplementedError("dsa captures decode graphs only")` for `rows_per_req != 1`.
   The decode capture allocates `_rows_buf[max_bs, W]` + `_kvlen_buf[max_bs]` and
   `_stage_decode` restages them per replay. The verify extension (qsa_sparse
   pattern) needs: `_rows_buf[max_bs, k+1, W]`, per-row `_kvlen_buf`/`_positions_buf`,
   a `prepare_for_replay` that stages the k+1 query rows, and a graph-capturable
   multi-row forward using `_select_verify_indexed` (device-side selection) +
   `_attend`. This is the substantial remaining backend work; then proposer runtime
   wiring (verify-hidden relay, snapshot timing, draft KV rollback) + acceptance/tok/s.
   Note: the NVFP4 bank sidecar rebuilds on each spec-config change (~150 s/rank),
   slowing iteration.

### Detailed verify-capture implementation design (derived round 10)

The hy4 `_attend(q_cat[B,m,H,D], layer_id, sel[B,m,W], cnt[B,m], sinks)` already
supports batched multi-row (the eager `_prefill` calls it per-request in a host
loop). The decode path (`_decode`) is the graph-capturable template: static
`_rows_buf`/`_kvlen_buf`, device-only selection, `_stage_decode` restage-per-replay.
The verify extension mirrors it for `m = k+1` rows:

1. `Hy4DSABackend.init_capture_graph(max_seq_len, bs_list, rows_per_req=m)`: drop
   the `rows_per_req != 1` guard (override in Hy4DSABackend); allocate
   `_rows_buf[max_bs, m, W]`, `_kvlen_buf[max_bs, m]`, `_positions_buf[max_bs, m]`,
   and indexer-query static buffers `_q_idx_buf[max_bs*m, H, D]`,
   `_w_buf[max_bs*m, H]` (the verify indexer inputs).
2. `prepare_for_replay`: stage the k+1 query rows' page-table rows, per-row kvlen
   (cached_len + row offset), positions, and the indexer q/w into the static buffers.
3. A graph-capturable `_verify(md, layer_id, q_nope, q_pe, inputs)`: reshape to
   `[bs, m, ...]`, call `_select_verify_indexed(slot, q_idx[bs,m,H,D], w[bs,m,H],
   rows[bs,m,W], positions[bs,m], kvlen[bs,m], max_pools=max_seq_len//kpool)` →
   `(sel[bs,m,W], cnt[bs,m])`, then `_attend(q_cat[bs,m,H,D], layer_id, sel, cnt,
   sinks)`. Store `md.sel[layer_id]` for the DFly draft reuse snapshot.
4. `mla_forward` dispatch: when the batch is a verify extend (`rows_per_req>1`,
   graph-bound), route to `_verify` instead of the eager `_prefill`.
5. The DFly draft (`dfly_mla_forward`) reads the snapshotted `md.sel[target_layer]`
   via `build_dfly_reuse_indices`; the draft KV writes alias the target pool slots.

This is the substantial remaining backend work (qsa_sparse is the 592-line
reference). Then proposer runtime wiring + iterative end-to-end debugging +
acceptance/tok/s. Lever 1 (pipeline) follows DFly per the 2→3→1 order.

### Implementation progress (round 10)

Implemented in `Hy4DSABackend` (hy4_sparse.py): `init_capture_graph` override
accepting `rows_per_req>1` (multi-row static buffers `_v_rows_buf[bs,m,W]`,
`_v_kvlen_buf`, `_v_pos_buf`, `_v_max_pools`); `_stage_verify` /
`prepare_for_capture` / `prepare_for_replay` / `reset_capture`; `_verify`
(device-side batched: SWA→`_window_select`, indexed→`_select_verify_indexed`,
then batched `_attend`, snapshots `md.sel[layer_id]` for DFly reuse);
`prepare_metadata` sets `md.verify_rows_per_req` for phase="verify";
`mla_forward` dispatches verify batches to `_verify`. `DSAMetadata.verify_rows_per_req`
field added; module-level `get_global_ctx` import. The engine already builds the
verify GraphRunner (`engine.py:2507 rows_per_req=k+1, phase="verify"`).

Smoke-test progression: "dsa captures decode graphs only" → RESOLVED (override);
"Batch has no positions" → RESOLVED (dummy-capture fallback); current: a tensor
expand shape mismatch (`expanded size (1) vs (8192) at dim 1, target [1,1]`)
during verify graph capture (bs=1). Hypothesis: the target dsa backend instance
is shared by the decode runner (rows_per_req=1) and the verify runner
(rows_per_req=k+1); `init_capture_graph` is called for both and the decode call
resets `_verify_rows_per_req`/buffers to 1-row, so the verify replay stages with
m=1 against a [8192] row tensor. Fix direction: in spec mode the verify graph
REPLACES decode (no separate 1-row decode graph for the target), or the backend
must keep separate decode/verify buffer sets keyed by rows_per_req. Needs the
full scheduler traceback (subprocess swallows it; enable scheduler traceback
capture) + the engine's spec graph-capture flow (engine.py:2507 verify runner vs
the decode runner). Iterative shape/wiring debugging continues; the ~150 s/rank
sidecar rebuild slows each iteration.

Round 11-12 fixes: added traceback capture to `_stage_verify`/`_verify`; fixed
`_verify_inner` passing 3-D `md.rows[bs,m,W]` to `_select_verify_indexed` (expects
2-D `[bs,W]`) → pass `rows[:,0,:]`. Next bug surfaced: the **engram disk-table
lookup** during verify capture — `dev[:nbytes].view(bf16).view(num_tokens=6,
n_cols*head_dim=1536)` fails (input size 3072 = 2 rows staged, not 6): the engram
host-fill (`EngramHost.forward_host_ctx`) stages rows for the verify dummy batch's
token count, which mismatches the multi-row lookup's `num_tokens`. The engram fill
must stage `bs*m` query rows for verify (and the graph-capture dummy batch must
carry `m` rows). This is the next integration layer (engram fill ↔ verify capture),
then proposer runtime wiring + KV rollback + acceptance/tok/s.

Round 13 analysis: `EngramHost.host_fill_batch` (engram_table.py:246) takes the
prefill branch for phase="verify" (is_decode=False) and stages runs from
`batch.reqs` + `md.qo_indptr_cpu`/`kv_len_cpu` (`m = qo[i+1]-qo[i]`,
`r.input_ids[start:start+m]`), while the engram forward (engram.py:178) looks up
`t = input_ids.shape[0] = bs*m` rows. For the verify dummy capture batch these
token counts disagree (staged 2 rows vs lookup 6). Fix direction: host_fill_batch
must handle phase="verify" explicitly — stage `bs*m` query rows (per-request m
rows at positions cached_len..cached_len+m-1, each with its n-gram context), matching
the multi-row lookup; the graph-capture dummy batch must carry m rows per request.
The engram fill is graph-captured via forward_host_ctx (wait-sync), so the verify
fill must be graph-safe (static row count bs*m).

Round 14: added a phase="verify" branch to `host_fill_batch` staging bs*m runs
(each verify token + its n-gram context). The lookup error persists because the
engram **graph buffer** is sized for decode: `load_host_tables` (model.py:152)
sets `graph_rows = max(max_running_req, cuda_graph_max_bs)` (=2), but the verify
graph needs `max_running_req * decode_rows_per_req` (=2*6=12) rows. Fix: size the
engram graph buffers (`max_graph_rows`, `_token_readback`) for the verify row
count when spec decode is on. This is the engram graph-buffer sizing integration;
then proposer runtime wiring + KV rollback + acceptance/tok/s.

### MILESTONE (round 15): DFly generates correctly end-to-end

Fixed the engram graph-buffer sizing (`graph_rows *= decode_rows_per_req`), the
proposer prefill/verify sink split (`_DFlyPrefillSink` + `feature_sink` phase
dispatch + `on_prefill` relaying the last prompt hidden into the verify relay),
and `dfly_mla_forward` heads (use the draft indexed `n_heads//tp`, not
`_heads(target_layer)` which indexes the 5-layer draft swa table). The DFly serve
now **starts, captures graphs, and generates correct output** ("日本の首都は東京です。",
finish=stop; long-form history correct). ~12 integration bugs fixed iteratively.

**But draft acceptance ≈ 0** (`accept len: 1.02` — only the bonus token accepted,
all 5 drafts rejected): DFly runs at ~4 tok/s vs the ~14 tok/s non-spec baseline
(pure drafting overhead, verify falls back to target greedy). The draft forward
produces wrong tokens. Next phase: debug draft acceptance — suspects are the
reuse-index construction (`build_dfly_reuse_indices` snapshot/anchor alignment),
the `target_hidden` relay (fusion reads the wrong hidden row), the draft KV alias
into the target pool (`store_kv` at `target_layer` vs the draft query positions),
and the draft attention rope/positions. Acceptance must rise well above 1 for DFly
to beat the baseline. The practical best config remains non-spec NVFP4 (~14 tok/s).

### Acceptance debugging (round 16)

Fixed the reuse-index semantics to match vLLM exactly: the snapshot now stores
token POSITIONS (`md.sel_pos`, threaded through `_expand_groups` returning
`(sel, cnt, pos)`), `build_dfly_reuse_indices` keeps vLLM's position semantics
(parity tests green), and the proposer maps the WHOLE sel_buf (history + causal
block positions) to physical KV rows via `page_table[table_idx, pos]`; snapshot
stored/read by `table_idx`; `target_topk = index_topk` (was `out_width-query_width`).
**Acceptance still ~1.07** (was 1.02) -- the reuse indices are now correct, so the
draft FORWARD itself produces wrong tokens. Next: parity-test the draft model
(`Hy4DFlyDraft.forward_block` / fusion / hidden_correction / DraftAttention) against
the vLLM reference on matched inputs; suspects are the fusion target_hidden relay
row, the draft attention rope/positions, the draft KV alias into the target pool,
and the TP-sharded draft weight forward (loading shapes verified, computation not).
This is deep multi-day draft-forward verification.

Round 17: verified the DFly block structure matches vLLM (the draft inherits the
target decoder layer but iHC `attn_hc`/`ffn_hc` and `post_mlp_layernorm` are
identity for the draft -- no such weights exist -- so plain pre-norm is correct;
fusion/correction math matches). Fixed a real relay bug: `verify_hidden` was used
for BOTH the verify-row capture AND the per-table_idx relay slot (overlap); added
a separate `relay_hidden[rows, dim]` (on_prefill/post_verify/propose use it).
Acceptance rose 1.02 -> 1.07 -> ~1.2 across the fixes -- slowly improving, but still
far from the ~3-4 needed to beat the baseline. The draft forward still produces
mostly-wrong tokens; remaining suspects are the draft attention KV alias/rope into
the target pool and the TP-sharded draft weight FORWARD computation (loading shapes
verified, numerics not). Next: layer-by-layer numeric parity of the draft against
the vLLM reference on matched inputs (embed -> fusion -> block0 attn/mlp -> ... ->
correction -> head), isolating the first divergent layer.

Round 18: ruled out more suspects by direct comparison with the vLLM reference --
DraftAttention rope (base=rope_theta_indexed, is_neox=False) and kv_b absorption
(w_uk/w_uv split + bmm) match the target indexed layers; the anchor/position
construction matches (`anchor = target_positions[source_row] + 1 = cached_len`,
positions = anchor + offsets, draft device_len = anchor + query_width). With reuse
indices, relay buffer, block structure, attention math, rope, and positions all
verified/fixed and acceptance still ~1.2, the divergence is in the draft KV alias
numerics (store_kv into the target pool + the sparse-MLA attend over aliased rows)
or the TP-sharded draft weight forward computation. These need layer-by-layer
numeric parity against a reference forward on matched inputs -- multi-day work.

## Phases

1. **Draft model** `models/hy_v4_flash/dfly.py` (this commit, foundation):
   `Hy4DFlyFusion`, `Hy4DFlyHiddenCorrection`, 5 `Hy4DecoderLayer`-based draft
   layers with dense MLP, `final_norm`, weight loader for `model.dfly_layers.*`
   (block-fp8 dequant for the draft MLP/attn), `draft_config` derivation
   (1-based target sparse layers, no engram/MoE). Verify: weight-loading parity
   (every draft tensor loads, shapes/dtypes match) on the FP8/NVFP4 checkpoint.
2. **Draft attention backend**: extend the hy4 `dsa` backend so a draft layer
   attends over the TARGET KV pool at its paired sparse target layer using
   external top-k indices (`dfly_target_reuse.py`), instead of a private draft KV
   chain. This is the hardest part: target-KV aliasing + the indexer index reuse.
3. **DFly proposer** `spec/dfly/proposer.py`: mirror `MTPProposer` but drive the
   fusion + 5 layers + hidden_correction per draft step, feeding target hidden
   states and reused indices; `num_steps=5`. Register `spec_drafts` +
   `speculative_algorithm='dfly'` + `SpecRuntime` wiring.
4. **Verify + graphs**: reuse the existing target verify path; capture draft
   graphs; slot-state accounting for the draft KV alias.
5. **Verification**: acceptance rate + tok/s on the 2x RTX PRO 4000 rig vs the
   ~14 tok/s non-spec baseline; correctness (greedy parity, sampled quality).

## Scope / risk

This is a multi-session engine feature, not a flag. Phase 2 (target-KV-reuse
attention) is the bulk of the risk: a wrong index/KV alias yields low acceptance
(no speedup) or incorrect tokens. The vLLM reference is the ground truth for the
math; freetoken's backend differs from vLLM's, so the alias must be re-derived
against freetoken's paged latent-KV pool. Potential payoff ~1.5-2x bs1 decode
once acceptance is high; vLLM's doc explicitly does not guarantee a speedup for
every workload.