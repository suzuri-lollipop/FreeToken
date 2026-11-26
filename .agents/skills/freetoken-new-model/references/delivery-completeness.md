# Delivery completeness: MANDATORY requirements for a new model

These are mandatory requirements (必須要件), not optional surfaces. A new
model implementation is NOT delivered until every item below is implemented
and verified on real hardware. The ERNIE 5.5 addition shipped TP1-only, no
speculative draft, and auto-only parser selection - treating these as "not
required this time" is exactly the gap this skill must prevent. There is no
"defer and document" escape hatch: a genuine architecture blocker is an
exception to surface and solve with the owner, not a default.

## Mandatory requirements (checklist)

1. **Parser registration in all four tables** (class maps, streaming opener
   tag, auto cascade, explicit CLI choices).
2. **TP >= 2 support** - the reader shards and the expert banks shard per rank.
3. **Speculative draft support** - serve the checkpoint's draft head and pass
   the engine's spec gate.
4. **GGUF support** - a GGUF reader + arch registration for the family.
5. **Calibration tool family map** - register the family's speculative contract.
6. **KV-cache quantization for long context** - the family serves its full-history
   KV under `--kv-cache-dtype int8` / `fp8` (the row-scale formats), verified under
   TP on the real server. A long-context family left bf16-KV-only is an incomplete
   delivery: the full-history slab is what makes 256K+ context fit a 24 GB card.
7. **Real-checkpoint serving** - the family boots end to end on the REAL checkpoint
   (download it), not only the synthetic harness: tokenizer loads (a custom
   `auto_map` tokenizer needs `trust_remote_code` + a registered `PretrainedConfig`),
   the config gates pass, and it generates. Synthetic-only verification hides the
   serving-path gaps (config registration, the TP offload gate, the tokenizer).

The deep mechanics live in the referenced files; this is the checklist that
makes them required rather than skipped.

## 1. Parser registration: four independent tables

`tests/server/test_parser_auto_selection.py` guards the live registry across
all four; a new family is covered the moment it is registered:

1. `server/function_call_parser.py` - the detector class, the
   `FunctionCallParser` name->class map, and the opener tag in
   `TOOLS_TAG_LIST` (streaming tool-call detection).
2. `server/reasoning_parser.py` - the parser class and the `ReasoningParser`
   name->class map. Mind `force_reasoning`: a chat template that ends the
   generation prompt with the open think tag (thinking ON by default) needs
   `force_reasoning=True`; `enable_thinking: false` pre-closing the block is
   the off switch.
3. `server/args.py` - the substring cascade in `_infer_tool_call_parser` /
   `_infer_reasoning_parser` (feeds `auto` only).
4. `server/args.py` - the explicit `choices` lists of `--tool-call-parser`
   and `--reasoning-parser`. A name missing here is rejected by argparse on
   explicit selection even though `auto` resolves it and the factories build
   it. Auto working is NOT evidence the registration is complete - this is
   the table the ERNIE addition missed.

CPU verification (no GPU, no checkpoint):

```python
from freetoken.server.args import parse_args
args, _ = parse_args(["--model", "/models/anon",
                      "--tool-call-parser", "<family>",
                      "--reasoning-parser", "<family>"])
assert args.tool_call_parser == "<family>"
assert args.reasoning_parser == "<family>"
```

## 2. Tensor parallelism (TP >= 2) - REQUIRED

`register.py` `tp_supported=True` must be earned and set: the reader emits
TP-local slices AND every shardable Linear is column/row parallel AND the
expert banks shard per rank. A family left at TP1 is an incomplete delivery.
The gates to clear (not to document around):

- Offload-family MoE strategies (`offload` / `cpu` / `hybrid`, and `auto`
  resolving to them) need expert banks that shard per rank. The NVFP4 and
  ds_fp4 (OCP MXFP4) kernels declare their banks at the rank's intermediate
  width and pack the matching slice (host RAM, the GPU slot cache and the PCIe
  stream halve per rank); the bf16 stack still emits full width. The required
  work for a new format is to give its banks per-rank slices, or serve resident
  `fused` where VRAM allows. **The engine's TP offload gate
  (`engine/config.py`, the `expert_quant not in (...)` check) must list every
  format whose banks actually shard** - a stale gate (it once allowed only
  `nvfp4`) refuses a family whose kernel already declares `tp_ok=True`.
  (docs/cli.md "What is TP-sharded today"; references/weights-and-quant.md,
  lessons.md "Tensor parallelism".)
- Fused buffers cut per LEAF with its own rule; KV heads must divide the TP
  size (refuse at construction); indexer + compressed slab stay replicated.
- An FTW directory cannot be sharded (tensors already fused at full width):
  point `--model` at the HF checkpoint for TP > 1.

Verify on the real multi-GPU rig (the synthetic-checkpoint harness runs a
full forward at TP=2 without the full-size weights). **The harness builds the
pool/model directly and BYPASSES the engine config gates** (the TP offload
gate, the spec gate, kv-format validation), so a green synthetic TP2 test is
NOT evidence the real server boots at TP2 - boot the real checkpoint through
`ft serve --tp-size 2` and generate. TP1-only is not an acceptable finish for a
family whose experts can shard.

## 3. Speculative decoding / draft model - REQUIRED

A family that ships a draft head must serve it. Two requirements, both
mandatory; a bundled-but-unserved head (ERNIE's `model.mtp.*` DSpark head) is
a gap to close, not a finish:

- `register.py` `spec_drafts=((name, "module:Class"), ...)` plus a draft class
  implementing the contract (references/model-and-kernels.md "MTP draft").
- The engine gate `_adjust_spec_config(windowed_kv=True)` refuses targets with
  sliding-window or DSV4 window-tier KV ("their window frees would release
  slots a verify step still reads"). ERNIE's HCA layers are DSV4 window-tier,
  so spec is refused at config time regardless of a draft. The required work
  is the eviction audit inkling2's SWA went through (free below the COMMITTED
  frontier under spec so a verify's reads stay live at any draft width;
  engine.py `swa_spec_ok`), then lift the ban for the family. Serving the
  draft head alone is not enough; clearing the gate alone is not enough.

Register the family's contract in the calibration tool (below) so it offers
the right arms. A family with no draft head at all is the only exception, and
that is a property of the checkpoint, not a deferral.

## 4. GGUF - REQUIRED

The family needs a `gguf.py` (parse_gguf_config / iter_gguf_weights / to_gguf
name map), a GGUF arch entry in register.py and `models/gguf/config.py`, and
the GGUF alias row in docs/models.md. A family without a GGUF reader cannot
serve GGUF checkpoints. This is a required surface, not a scope choice.

Two conventions, pick per family:

- **llama.cpp-converted** (the existing families): reverse the converter's
  transforms (zero-centered `w+1` norms, tiled heads, interleaved expert rows,
  `blk.N.*` names). references/weights-and-quant.md, lessons.md "GGUF /
  converter transforms".
- **Strata-convention (HF-name-preserving)** - preferred for ERNIE 5.5, do NOT
  reference llama.cpp: keep the checkpoint's HF tensor names and the HF config
  as a JSON metadata blob (`<arch>.hf_config`), so there are no converter
  transforms to undo. The dense reader shares the HF name map
  (`_iter_dense_from_reader`); `to_gguf` is the identity on HF names. Reference
  Strata's OWN reader/writer (`module/strata/terminal/tools/gguf_reader.py` /
  `gguf_writer.py`), NOT the vendored `strata-gguf-py` (that one IS llama.cpp's
  gguf-py/converter copied unchanged - see its UPSTREAM.md). Strata's own
  reader handles the quant types gguf-py cannot represent (MXFP4 = 39,
  NVFP4 = 40, Q2_0 = 42), which the MXFP4 expert banks need. The fork's shared
  GGUF helpers are llama.cpp-name-centric (`_vocab_size` wants
  `token_embd.weight`, the expert-bank loader wants `blk.N.ffn_*_exps`): an
  HF-name GGUF supplies the vocab via `tokenizer.ggml.tokens` metadata and
  needs its own expert-bank read (the offload dispatch in
  `moe/expert_banks.py` keys on `QuantKind.GGUF`). Verified: ERNIE dense weights
  round-trip bit-exact HF<->Strata GGUF (tests/models/ernie5_5_moe/test_gguf.py).

## 5. Calibration tool family map - REQUIRED

`python/freetoken/calibrate_multi.py::spec_backends_for` decides which
speculative arms `--spec auto` offers, by the target's `model_type`:

- a family with its own drafter swaps the arm (inkling2_loop -> dspark);
- a family that refuses speculation at config time gets no arm - register it,
  do not rely on the generic fallback where a doomed mtp arm fails at server
  start and is excluded with a reason;
- an explicit request for a refused backend is pre-skipped with the reason.

`calibrate_lookup` is model-agnostic (copy probes); no family entry. Extend
the `spec_backends_for` tests in `tests/test_calibrate_multi.py`.

## Docs and exceptions

docs/models.md Notes records the family's serving facts (MoE strategy, TP,
KV dtype, draft, rope scaling, KV bytes/token). A limitation may appear there
ONLY as a surfaced exception the owner has agreed to solve next - never as a
silent "not yet" that shipped instead of a mandatory requirement. If a
requirement above cannot be met, stop and raise it with the owner; do not
downgrade it to optional on your own.
