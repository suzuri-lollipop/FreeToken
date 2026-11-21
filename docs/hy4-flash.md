# Hy4-flash (tencent/Hy4-flash)

Engine-native support for Tencent's Hy4-flash (`hy_v4_flash`, `HYV4FlashForCausalLM` /
`HYV4FlashForConditionalGeneration`): 42 layers of gated MLA on 4 identity
hyper-connection (iHC) residual streams, hybrid attention (12 compressed-indexer
sparse layers + 30 sliding-window layers), Engram n-gram memory at layers
1/11/21/31, and 512-expert top-16 LatentMoE running in a 1536-wide latent space.

Served today: **text only**. The vision tower and the DFly2 speculative draft ship
in the checkpoint but are not served (their tensors are skipped at load); the
multimodal wrapper architecture loads as the text model.

## FP8 serving (recommended quality)

The FP8 release serves directly -- block-fp8 attention projections and the
stacked block-fp8 experts through the offload cache (~149 GB pinned host banks).
The family marks its fp8 layers ``fp8_w8a16``: Hy4-flash is bf16-trained, and its
activations degenerate under the fp8 kernels' default dynamic per-128-group W8A8
quantization (repetitive, fragmented generation), so both phases serve the fp8
weights weight-only with bf16 activations:

```bash
ft serve --model-path ~/models/tencent--Hy4-flash-FP8 \
    --moe-strategy offload --moe-cache-auto \
    --max-running-requests 2 --num-tokens 32768
```

KV cache: bf16 (the default), `--kv-cache-dtype int8` (per-row amax int8), or
`--kv-cache-dtype fp8_per_token_head` (per-row amax fp8). The plain `fp8` alias
is the STATIC-scale format (`fp8_e4m3`, one config scale for every row) and it
fragments this model's generation into word salad -- the MLA latent rows need
per-row scaling. int8 and fp8_per_token_head both track the bf16 reference.

The family also opts its bf16 projections and embedding table out of the
engine's per-channel/per-row fp8 rewrites (`w8a16_decode_ok` / `w8a16_embed_ok`):
across 42 layers that extra weight noise breaks generation the same way.

The checkpoint's `modules_to_not_convert` names the text tower under the HF
wrapper (`language_model.model.<module>`); the family's `checkpoint_roots` map
the engine's module names onto that spelling, so the bf16 modules (kv_b, the
latent projections, shared expert, router, indexer, engram, iHC, embeddings,
lm_head) resolve correctly. Generation is coherent on this path -- it is the
quality reference the NVFP4 conversion is measured against.

## NVFP4 conversion

No official NVFP4 release exists yet; `scripts/convert_hy4_fp8_to_nvfp4.py`
converts the FP8 release:

```bash
python scripts/convert_hy4_fp8_to_nvfp4.py \
    --source ~/models/tencent--Hy4-flash-FP8 \
    --out ~/models/tencent--Hy4-flash-NVFP4
```

- Routed experts: stacked 128x128 block-fp8 -> per-expert ModelOpt NVFP4
  (`weight` packed e2m1 pairs, `weight_scale` e4m3 per 16 input elements,
  `weight_scale_2` fp32 dequant-side global). The block scale is the e4m3 value
  minimizing the block's post-rounding MSE among the nearest scale and its e4m3
  neighbors; codes round to nearest, ties to even.
- Attention projections: block-fp8 -> bf16 (the MLA absorption reads `kv_b`
  unquantized anyway).
- Engram tables: copied unchanged (bf16, ~127 GB); streamed from disk at serve
  time through the same io_uring row store the DeepSeek-V4.1 engram uses, with
  the trained fp8 per-vector round-trip applied at lookup.
- DFly2 draft layers: dropped. Vision tower: copied, not served.
- `--expert-dtype bf16` keeps the experts dequantized (stacked bf16) for A/B
  quality checks; `--max-layers N` converts a prefix for fast load tests.

The output is a W4A16 ModelOpt checkpoint (`quant_algo: W4A16_NVFP4`, everything
but the experts on the ignore list); the experts serve from the offload cache.

## Serving

Verified on 2x RTX PRO 4000 Blackwell (24 GB), TP=1, experts offloaded to host
RAM (~80 GB pinned banks), engram streamed from disk:

```bash
ft serve --model-path ~/models/tencent--Hy4-flash-NVFP4 \
    --moe-strategy offload --moe-cache-auto \
    --max-running-requests 2 --num-tokens 32768
```

Same KV guidance as FP8: bf16 or `fp8_per_token_head`, never the static-scale
`fp8` alias.

The attention backend resolves to `dsa` (the hy4 kpool variant): indexed layers
recall the 511 best complete 4-token groups strictly before the query's own
group plus every existing token of that group (2048 tokens total); SWA layers
recall the last 512 tokens; both attend with learned per-head sinks. Page size
is auto-adjusted to 64 for the kpool shadow slab.

## Tensor parallelism

`--tp-size N` shards the attention heads (q_b/kv_b/linear_gate rows, o_proj
columns, per-head sinks), the dense/shared-expert intermediate, the routed
expert intermediate (offload banks are cut per rank, with a per-rank FTW
sidecar), and the vocab. The latent KV pool, compressed indexer, engram tables,
iHC mixers and norms replicate on every rank: all ranks must recall the same
tokens and carry the same residual streams, and the row-parallel projections
all-reduce their partials.

Valid TP sizes divide the head counts (32 indexed / 64 SWA) and keep whole
scale blocks: block-fp8 experts need `1536 / tp` to be a multiple of 128
(tp 2, 3, 4, 6, 12); NVFP4 shards by 16-row groups. Verified at TP=2 on the
2x RTX PRO 4000 (FP8 and NVFP4, bf16/int8/fp8_per_token_head KV): the first
tokens and the reasoning split match the TP=1 serving; later greedy tokens can
diverge because the all-reduce rounding flips near-tied choices in this
loop-prone model (judge quality with sampling, as above).

## Verification

`tests/models/hy_v4_flash/` checks the port against a standalone eager reference
transcribed from the transformers PR (huggingface/transformers#50903):

- dense-regime prefill logits parity (bf16 tolerance),
- sparse-regime prefill + greedy decode with the selection sets cross-checked
  against the reference (near-tie flip rate bounded),
- chunked prefill (a compress group straddling two chunks), two-request batches,
- the released attention/engram widths (32/64 heads, 512-wide latent, 96-dim
  engram heads) against the reference's own bf16-vs-fp32 noise floor,
- the FP8 layout (stacked block-fp8 expert banks + fp8 attention, served W8A16)
  against the bf16 reference on the dequantized weights,
- the converter's packing/scale convention against the engine's dequant kernel.

## Quality notes

**Serving numerics.** Hy4-flash is bf16-trained. Three engine defaults that are
fine for fp8-trained models (DeepSeek-V3 style) break its generation, and the
family opts out of all three:

- the fp8 kernels' dynamic per-128-group W8A8 activation quantization
  (`fp8_w8a16` keeps activations bf16 at every batch size),
- the per-channel fp8 rewrite of bf16 projections (`w8a16_decode_ok`),
- the per-row fp8 rewrite of the embedding table (`w8a16_embed_ok`).

On the KV side, the static-scale `fp8` alias fragments generation into word
salad; use bf16 or `fp8_per_token_head` (per-row scales).

**Engram hash multipliers.** The odd multipliers that hash an n-gram into the
memory table are drawn bounded by the largest compressed id + 1 of the loaded
map (92620 on the released checkpoint), not by `config.engram_lookup_table_size`
(the table LENGTH, 120818). The checkpoint's normalized-vocab map compresses to
fewer ids (max 92619) than the table holds, so the two bounds differ and the
wrong one reads unrelated memory rows. The family draws them from the loaded
map, matching the reference `_init_weights` redraw (the PR's test pins the
released layer-2 multipliers); `test_engram_multiplier_bound` guards it (the
fixture uses a compression map so the bounds are distinguishable).

**RoPE convention (the word-salad root cause, fixed 2026-11-20).** Hy4-flash
rotates **adjacent (even, odd) channel pairs** -- the GLM / GPT-J *interleaved*
RoPE (`rotate_half` over `x[..., 0::2]` / `x[..., 1::2]`, each frequency
`repeat_interleave(2)`), as in the PR's `apply_rotary_pos_emb`. The port had
transcribed the DeepSeek **NeoX half-split** (`x[:d/2]` / `x[d/2:]`,
`cat(freqs, freqs)`, `is_neox=True`). The two rotate the same vectors
differently, so attention scores were wrong on trained weights and generation
fragmented into word salad. Random-weight parity tests hid it because the engine
and the eager reference shared the same wrong convention. A full-depth comparison
of the PR runtime (huggingface/transformers#50903, head `b865850d`) against the
reference on a real-weight bf16 prefix isolated the first divergence to layer-0
attention (input identical, output 17% rel L2); switching the reference to
interleaved made the attention output bit-exact (0 rel L2). The fix sets
`is_neox=False` on the main MLA and indexer ropes, rotates the indexer query and
the kpool group-key kernel (`_kpool_rope_rows_kernel`) over adjacent pairs, and
uses the interleaved `rotate_half` / `repeat_interleave` cache in the reference.
Verified on the FP8 release (TP=2, int8 KV): Japanese and English prompts now
answer correctly and stop cleanly. The weights themselves were never the problem
(re-downloaded at `bd0e136e`; upstream unchanged since 2026-11-18 apart from
`chat_template.jinja` dropping the `max` effort; block-fp8 experts dequantize to
the BF16 base within 3.4% rel L2, unquantized tensors bit-identical).

**Decode throughput (open).** On the 2x RTX PRO 4000 (PCIe, no P2P) FP8 decode
runs ~4-5 tok/s at bs1 (TP=2). Profiling (`FREETOKEN_MOE_STATS=1` /
`FREETOKEN_LATENCY=1`) shows ~90% of wall time host-blocking inside the forward
call: the VRAM expert slot cache hits ~80% of routes and the ~3.3 miss
experts/layer (x41 MoE layers) are decoded **synchronously on the CPU executor**
(`fetched=0`, `cpu=miss`) rather than streamed to the GPU over PCIe with the
decode fetch overlap. TP=1 is *slower* (~1.7 tok/s, forward block ~570ms): it
removes the per-layer all-reduce but doubles each expert's CPU GEMV (full-width
banks) and shrinks the VRAM slot cache, so the CPU decode volume -- not the
all-reduce -- is the cost. maya reaches ~18 tok/s on the same 2 GPUs by a
different strategy: it rejects TP (a member needs ~50 GB VRAM), **layer-splits**
(pipeline: GPU0 layers 0-15, GPU1 16-41, no per-layer all-reduce) and runs a
**44-thread CPU lane** for the 77% of experts not in VRAM, overlapping CPU expert
decode with GPU work across layers. The levers for this port are overlapping the
CPU miss decode with GPU compute (pipeline across layers), raising the VRAM slot
cache hit rate (smaller dense/KV footprint), or a faster CPU GEMV path. The auto
host-RAM budget also spills the head+tail MoE layers to NVMe CPU decode when
banks exceed it (`--moe-ram-budget` raises it), but that spill was not the
dominant cost here. The practical mitigation today is the NVFP4 conversion
(below): its half-size experts raise the VRAM cache hit rate enough to reach
~14 tok/s on the same rig without any engine change.

Measured bs1 decode on the 2x RTX PRO 4000 (600-token generation, int8 KV unless
noted; `FREETOKEN_MOE_STATS=1` / `FREETOKEN_LATENCY=1`):

| config | tok/s | VRAM-cache miss | forward host-block |
| --- | --- | --- | --- |
| FP8, TP=2, CUDA graph | ~4-5 | ~20% | ~178 ms (90%) |
| NVFP4, TP=2, CUDA graph | ~14 | ~15% | ~55 ms (88%) |
| NVFP4, TP=2, graph, engram pread-pool | ~15 | ~15% | ~54 ms |
| NVFP4, TP=2, eager (no graph) | ~4 | ~15% | ~231 ms |
| NVFP4, TP=1, CUDA graph | ~4 | ~36% | ~178 ms |
| maya layer-split (reference) | ~18 | — | — |

What the sweep shows: the cost tracks the **miss rate**, not the all-reduce
(TP=1 removes the all-reduce but doubles the miss rate and is slower); CUDA
graphs are essential (eager collapses to ~4 tok/s); the engram I/O path
(io_uring vs pread-pool) is worth only ~1 tok/s.

**Correction (counter semantics).** With every bank pinned (`--moe-ram-budget`
above the bank size) the offload cache resolves `decode_target=gpu`: there are
no CPU layers and **no CPU expert GEMV** — misses are gathered H2D into VRAM
slots and computed on the GPU (`fast_index_copy` + the NVFP4 marlin GEMV). The
`[moestat]` `fetched`/`cpu` fields are **hybrid-only** counters
(`OffloadMoeCache.stat_fetched`, "hybrid only" in offload_cache.py); on the
gpu-target path they stay 0, so `cpu=miss` there is an artifact, NOT evidence of
CPU decode. The generally-slow "moe + hybrid" mode is what the *auto* RAM budget
falls into when it spills the head+tail layers to the NVMe/CPU executor; pinning
all banks avoids it. **nsys trace of the bs1 decode graph (2026-11-20, NVFP4 TP=2 gpu-target,
`--cuda-graph-trace=node --trace=cuda,nvtx,osrt`).** The host block is
`cudaEventSynchronize` (77.5% of CUDA API time, ~10.8 calls/token, avg 9 ms) —
the forward call waiting on GPU completion events. On the device,
**`fast_index_copy_multi` (the miss-expert H2D gather over PCIe) is 68.8% of GPU
kernel time**, while the expert GEMV (`_decode_nvfp4_marlin_kernel`) is only
2.8% and the TP all-reduce (`_shm_ar_sum_kernel`) ~1%. So the gpu-target path is
**PCIe-copy-bound, not compute-bound**: the gen4 x4 links (~6.6 GB/s) move the
miss experts far slower than the GPU consumes them, and there is almost no
compute to hide the copy under — which is why enabling the fetch overlap
(`FREETOKEN_FETCH_OVERLAP_MIN_ROWS=1`) did not raise bs1 throughput. Switching
the whole MoE to CPU-direct decode (`--moe-strategy cpu`, all banks pinned,
21 executor threads/rank, no expert PCIe) gives ~13 tok/s — the same ceiling:
reading experts from RAM and GEMV-ing on the CPU costs about what copying them
over PCIe and GEMV-ing on the GPU costs. The ~14 tok/s plateau is therefore the
per-token expert-movement cost serialized per layer in TP lockstep. maya's
~18 tok/s comes from **layer-split pipeline parallelism** (GPU0 layers 0-15,
GPU1 16-41) overlapping its 44-thread CPU expert decode with GPU compute across
layers; this port has no pipeline mode, so that overlap is the missing lever.
Closing the gap, in order of leverage: (1) layer-split / pipeline parallelism to
overlap expert decode with GPU compute across layers (the structural fix);
(2) cut the miss volume with a measured hot-expert bank profile (`ft-profile` +
`--moe-hot-frac`) or a larger VRAM slot cache, shrinking the PCIe copy;
(3) speculative decoding from the checkpoint's DFly2 draft layers (no
`spec_drafts` entry for the family yet). The structural limit is that the ~87 GB
of NVFP4 experts exceed the rig's 48 GB of VRAM, so per-token expert movement is
unavoidable here. Best config today: NVFP4, TP=2, CUDA graph, all banks pinned,
`FREETOKEN_ENGRAM_IO_URING=0`, int8 KV (~14-15 tok/s).

Lever status (2026-11-20):
- **(2) miss-volume reduction — exhausted on this rig.** The VRAM is already
  saturated: KV is only 0.77 GiB (latent), the expert slot cache ~16 GiB
  (6139-6671 slots, ~160 of 512 experts/layer), CUDA graphs ~2 GiB, dense
  ~3.5 GiB. `--cuda-graph-max-bs 1` / smaller KV free negligible VRAM and leave
  the miss rate ~14% and throughput ~14 tok/s unchanged; `--moe-cache-mem` cannot
  grow the cache past the ~2 GiB headroom. `--moe-hot-frac` compact pinning
  targets *disk* layers (CPU executor), not the gpu-target VRAM cache. Cutting
  misses meaningfully needs more VRAM or fewer expert bytes, not a config knob.
- **(1) layer-split / pipeline parallelism — not present; limited bs1 payoff.**
  The port has only TP (plus a dual-microbatch decode reducer for batch overlap).
  A single decode token traverses layers sequentially, so splitting layers across
  GPUs adds handoff latency rather than hiding it; pipeline parallelism mainly
  raises batch throughput. maya's layer-split edge needs its CPU-lane overlap
  reproduced, which is a substantial engine feature, not a flag.
- **(3) DFly2 speculative decoding — the strongest bs1 lever, but no reference
  implementation exists to port.** The checkpoint ships a DFly2 draft
  (`model.dfly_layers.*`, 117 tensors): 5 gated-MLA + dense-MLP draft layers,
  `mtp_fusion` (gated embed+hidden, `slot0`), `hidden_correction` (SwiGLU from
  embed+hidden), a `confidence_head`, and target-index cross-attention to target
  layers [25,29,33,37,41] (`dfly_target_index_topk=2048`). The reader skips it;
  `spec_drafts` for the family is empty. The engine has a spec framework
  (`engine/spec_protocol.py`, scheduler verify) and a 157-line Qwen MTP draft to
  mirror. **However the transformers PR #50903 explicitly ignores `dfly_layers`
  (`_keys_to_ignore_on_load_unexpected`) and implements no DFly2 drafting, and
  vLLM ships only `hy_v3`/`hy_v3_mtp`, not hy4 DFly2.** Porting therefore requires
  the official DFly2 algorithm (Tencent / future vLLM-SGLang hy4 support) or a
  from-tensors reverse-engineering of the fusion/correction/target-index math —
  research-grade and high-risk (a wrong draft yields low acceptance or incorrect
  output). Potential ~1.5-2x bs1 once a correct reference is available; it is the
  recommended next lever ahead of pipeline, but blocked on that reference.

**NVFP4 PTQ serves correctly and faster (re-tested 2026-11-20 after the RoPE
fix).** The conversion is format-correct; its weight error matches a known-good
ModelOpt conversion (8.6% relative L2 on Hy4 experts with MSE-optimal block
scales, vs 9.8% on RadixArk/Qwen3.8-Flash-Next-NVFP4 against its FP8 source;
~9-16% routed-expert output error per layer on real hidden states). The earlier
finding that this noise "degrades generation" was **confounded by the RoPE bug
above**, which fragmented FP8 and NVFP4 alike. With the interleaved RoPE fix the
NVFP4 conversion answers Japanese and English prompts correctly and stops
cleanly (capital, arithmetic, list prompts verified on TP=2 / int8 KV). It is
also ~3x faster than the FP8 release on the same rig (~14 tok/s vs ~4-5 at bs1):
the 4-bit experts are roughly half the bytes, so the VRAM slot cache holds more
of them and far fewer routes spill to the synchronous CPU decode that dominates
FP8 latency. NVFP4 is now the recommended serve for this model on memory- or
speed-constrained rigs; keep FP8 when the last bit of weight fidelity matters.
An official calibration-aware NVFP4 (or QAT) release would still tighten the
expert noise floor.

## Maya comparison + parallel-inference throughput (2026-11-21)

Maya (C++/CUDA engine, `module/maya`) serves Hy4-flash at **22.8 tok/s** decode
(2555 tokens / 112 s, single stream). Its config differs fundamentally:
- **Layer split (pipeline), NOT TP**: GPU0 layers 0-19, GPU1 20-41. Maya explicitly
  rejects TP (a TP member needs ~50 GB VRAM; layer split fits). No per-layer
  all-reduce; each GPU holds the FULL experts of its 21 layers (cache concentrated).
- **CPU lane hybrid MoE**: 30% of routed experts in VRAM, the rest computed on 44
  CPU threads from the model file (NUMA-spread pages).
- **Lower-bit experts**: IQ4_XS / IQ2_XXS GGUF pack (2-4 bit), smaller than NVFP4.

FreeToken NVFP4 TP=2 measured (server-log gen throughput, DL380 Gen9 gen3 x8):
| config | single tok/s | conc-4 aggregate | conc-4 per-stream |
|---|---|---|---|
| offload (best) | ~15-16 | ~20 | ~5.1 |
| hybrid (benched 11.4% fetch) | ~15 | ~9.6 | ~2.4 |

`ft bench bw` (NVFP4 1536x768 top16, 44 threads): CPU-MoE 33.5 GB/s vs PCIe-gather
5.4 GB/s (6.2x), overlapped fetch split 5.9-11.4%. Hybrid is WORSE under concurrency:
the CPU executor (max_tokens=4) does not scale to bs=4 -- the GPU dense work is too
small to hide the CPU overflow, so it lands on the critical path. Offload remains best.
Fixed a merge-lost `_QUANT_TO_BENCH_FORMAT` definition in `moe/bench_profile.py`
(NameError blocked `load_hybrid_fetch_fraction`).

The parallel drop (single 15 -> conc-4 per-stream 5) is the gen3 x8 PCIe bandwidth
wall (5.6 GB/s) shared across TP ranks + concurrent fetch, plus TP all-reduce
inflation under contention (~9-12 ms/step, see `_scratch/parallel-perf/RESULTS.md`).
Maya's layer split sidesteps both (no TP AR, concentrated cache, CPU lane). The
software path to match Maya is **layer-split / pipeline parallelism (lever 1)** in
FreeToken; the hardware path is an x16 riser (719076-B21 primary / 719073-B21
secondary, ~2x bandwidth). DFly spec decode (lever 3) generates correctly end-to-end
but draft acceptance ~1.2 (still slower than baseline) -- see dfly-spec-decode-plan.md.
