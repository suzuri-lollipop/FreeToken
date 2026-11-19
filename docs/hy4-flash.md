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
memory table are drawn bounded by `config.engram_lookup_table_size` (the table
LENGTH, 120818), not by the largest compressed id + 1. The checkpoint's
normalized-vocab map compresses to fewer ids (max 92619) than the table holds,
so the two bounds differ and the wrong one reads unrelated memory rows. The
family draws them with the table length, matching the reference
`draw_hash_multipliers`; `test_engram_multiplier_bound` guards it (the fixture
uses a compression map so the bounds are distinguishable).

**Known issue: generation still degenerates.** Even with the fixes above, chat
and Japanese prompts produce repetitive / fragmented output. The eager bf16
reference (independent of the engine kernels, real weights dequantized to bf16)
degenerates the same way, so the gap is in the shared model computation rather
than the kernels. Ablating the engram residual makes the output more coherent
(though it still loops), and the engram residual is large (~1.6x the stream
norm) by the reference formula. The port matches the PR's `modular_hy_v4_flash.py`
component-by-component, but the PR's generated `modeling_hy_v4_flash.py` is stale
(it expects non-latent experts and `gate_proj` where the checkpoint stores
`linear_gate`), so it cannot load this checkpoint as a runtime ground truth. The
checkpoint appears to come from a slightly different architecture revision than
the public PR; resolving the remaining degeneration needs that revision's
reference code (or the official vLLM/SGLang `hy4-flash` images) to compare
against. Treat current text quality as experimental.

**Naive NVFP4 PTQ degrades this model.** The conversion is format-correct and
its weight error matches a known-good ModelOpt conversion (measured 8.6%
relative L2 on Hy4 experts with MSE-optimal block scales, vs 9.8% on
RadixArk/Qwen3.8-Flash-Next-NVFP4 against its FP8 source). But Hy4-flash is
sensitive to that noise: the routed-expert output error is ~9-16% per layer on
real hidden states, and across 41 MoE layers it accumulates -- factual answers
survive in the first tokens, then generation turns repetitive and fragmented
even with the serving fixes above. Until an official calibration-aware NVFP4
(or QAT) release lands, treat this conversion as best-effort PTQ and serve the
FP8 release directly when quality matters more than the ~2x expert memory
saving.
