---
name: freetoken-new-model
description: FreeToken (this repository) に新しいモデルアーキテクチャを実装・追加するときに使う。Qwen3.8-Flash-Next (model_type qwen4_exp, NVFP4 / block-fp8 / GGUF) の実装全体から抽出した手順・凍結契約・チェックリスト・既知の落とし穴。Use when implementing or adding a new model / architecture / checkpoint family to FreeToken - model package, registry, weight readers, quantization dialects, attention backends, KV pools, slot states, Triton kernels, MoE expert banks, host tables, MTP drafts, GGUF support, TP sharding, server parser registration, TP and speculative-draft completeness, calibration tool family map, tests and hardware verification. Triggers - 新モデル実装, モデル追加, 新アーキテクチャ対応, パーサ登録, TP対応, ドラフト対応, GGUF対応, 必須要件, new model support, add model, model loader, parser choices, delivery completeness, mandatory requirements.
---

# FreeToken new-model implementation

Everything a new model family touches in this engine, extracted from the
Qwen3.8-Flash-Next implementation (`python/freetoken/models/qwen4_exp/`,
initial commit `e3b6d7b` "feat(qwen4_exp): support Qwen3.8-Flash-Next" (#257),
66 files / ~10k lines, plus ~120 follow-up fixes and features). Use it as the
reference anatomy when serving a new architecture: what to build, in what
order, which contracts are frozen, and which silent failures the follow-up
history already paid for.

The reference model in one paragraph: 48 decoder layers (36 GatedDeltaNet
linear-attention + 12 QSA compressed-block sparse attention) on 4
hyper-connection residual streams `R [T, 4*hidden]`; a PLE hashed n-gram
embedding layer at zero-based layer 1 backed by a 47.7 GiB fp8 host table;
512 NVFP4 / block-fp8 routed experts (top-10) plus a gated shared expert;
a shared Qwen3-VL vision tower; an MTP speculative head; HF safetensors
(bf16 / block-fp8 / ModelOpt NVFP4 / llm-compressor) and llama.cpp GGUF
checkpoints; TP=2 sharding the dense projections, the vision tower and the
NVFP4 expert banks (per-rank intermediate-width slices: 63.5 GiB of banks
become 31.8 GiB per rank).

## Ground rules (binding, from AGENTS.md / CONTRIBUTING.md)

- The human behind the change owns every line and must have run it on real
  hardware. Never report tests or benchmarks as run when they were not; if
  hardware or a checkpoint is missing, say so and say what ran instead.
- Do not run `git push`, `gh pr create`, `gh pr comment`, `gh issue create`.
  Only commit when the user asks. Conventional Commits, one line, imperative,
  lowercase, no trailing period: `feat(<family>): support <Model Name>`.
- Read `tests/README.md` ("What earns a place here") before writing a test.
  A new model's own tests go in `tests/models/<model>/` without asking;
  ask before adding test files anywhere else.
- Comments explain a non-obvious "why", one or two lines, ASCII only.
  Configuration files get no comments.

## Workflow

Work in this order; each phase is testable before the next starts. Details
and file-level checklists live in the references below - read the reference
for a phase before doing that phase.

1. **Survey** - HF `config.json` + `modeling_*.py` reference, checkpoint
   `weight_map`, quantization dialects, which existing families to reuse
   (qwen4_exp reuses `qwen3_5_moe` MoE + expert reader and `qwen3_vl` vision).
   Decide the attention type(s), the residual layout, and any host-scale
   tables. -> `references/anatomy.md`
2. **Config + registration** - `config.py` (`parse_config` -> `ModelConfig`,
   family args payload, attention groups, slot states), `register.py`
   `ModelSpec`, GGUF arch map, AOT kernel-shape entry.
   -> `references/config-and-state.md`, `references/anatomy.md`
3. **Weight reader** - `weight.py`: dense reader under the checkpoint's
   `QuantConfig`, fusions + padding, zero-centered norms, TP slicing, expert
   source spec, host-table loader, FTW side files. Synthetic-checkpoint tests
   first, real-checkpoint tests behind `needs_weights`.
   -> `references/weights-and-quant.md`
4. **Model + layers + kernels** - `model.py` skeleton and the frozen layer
   contract; per-block modules (attention / linear mixer / MoE / HC / PLE)
   each with a pure-torch reference; vendored Triton kernels with SPDX
   provenance headers, diffed against the torch path.
   -> `references/model-and-kernels.md`
5. **Stateful infrastructure** (only if the architecture needs it) -
   attention backend, KV pool family, `SlotStateSpec` + verify handlers,
   scheduler chunk alignment, engine `load_host_tables` / pin-budget hook.
   -> `references/config-and-state.md`
6. **Architecture-specific surfaces** (required when the architecture has the
   feature) - disk / pinned / GPU host-table backends, dual-microbatch decode,
   feature-sink exports.
   -> `references/model-and-kernels.md`, `references/weights-and-quant.md`
7. **Delivery completeness - MANDATORY requirements (必須要件)** - parser
   registration in all four tables (class maps, auto cascade, explicit CLI
   choices), an incremental streaming tool-call parser (cursor-based, not the
   O(n^2) full-buffer re-parse), TP >= 2 support, speculative draft support,
   GGUF support, and the calibration tool's model_type map. These are REQUIRED,
   not optional: a model that serves at TP1 but cannot shard, cannot draft, has
   no GGUF reader, or cannot be selected explicitly is NOT delivered. Implement
   and verify each on real hardware; a genuine blocker is an exception to raise
   with the owner, never a silent "not required this time".
   -> `references/delivery-completeness.md`
8. **Tests + verification + docs** - parity tests against independent
   references, subsystem tests for new pools/backends, real serve run,
   `docs/models.md` + `docs/cli.md` entries, A/B numbers for perf changes.
   -> `references/tests-and-verification.md`

Before calling any phase done, sweep `references/lessons.md`: it lists the
concrete bugs the qwen4_exp follow-up history fixed (TP fusion order, quant
dialect names, GGUF converter transforms, disk-PLE staging sizes, MTP source
checks, pin-budget ordering) and the delivery gaps the ERNIE 5.5 addition
exposed (parser choices, TP1-only, unserved draft head, calibration family
map). Most new-model bugs are repeats of these.

## Frozen contracts (do not rename or reshape)

- Layer contract: `forward(R [T, hc*hidden], batch) -> R' [T, hc*hidden]`
  with immediate mix/combine; the embedding repeats over the streams and the
  top-level mixer collapses them once before `lm_head`.
- `iter_weights(model_path, device, *, include_moe_experts, include_non_moe,
  include_vision=True)` yields `(state-dict key, TP-local tensor)`; routed
   experts never come through it when they have a bank reader.
- `QSAIndexerInputs`: raw pre-norm / pre-rope index q/k plus both norm
  weights; the backend norms AFTER the fp32 group mean. The pending ring
  stores pre-RoPE keys.
- `PLETableBackend.lookup(row_ids) -> dequantized rows` / `prefetch`; rows
  addressed by GLOBAL hashed id; int64 row addressing in every kernel.
- `SlotStateSpec` / `SlotVerify` (`nbytes`, `commit`, optional `stash`);
  read state back only through `pool.slot_state(name, layer_id)`.
- Zero-centered norms load RAW; `(1+w)` is applied at runtime in fp32.
- `ModelSpec` hooks the engine resolves by name: `parse_config`,
  `iter_weights`, `nvfp4_expert_spec`, `ftw_side_files`, `load_host_tables`,
  `prepare_for_runtime`, `spec_drafts`.

## References

Read the file for the phase you are in; the table is the complete list.

| Task | File |
|---|---|
| What a new model touches: full file map, package layout, registration + AOT checklist | `references/anatomy.md` |
| Config parsing, ModelConfig payload, attention groups, slot states, KV pool, attention backend, scheduler / engine hooks | `references/config-and-state.md` |
| Weight reader, QuantConfig dialects, fusions, TP slicing, PLE table, NVFP4 expert spec, GGUF transforms, FTW side files | `references/weights-and-quant.md` |
| Model skeleton, HC / GDN / QSA / MoE / PLE modules, kernel vendoring, MTP draft, dual decode, feature sink | `references/model-and-kernels.md` |
| Delivery completeness - MANDATORY requirements: parser registration (four tables), TP >= 2, speculative draft, GGUF, calibration tool family map | `references/delivery-completeness.md` |
| Test layout and parity oracles, the test bar, run commands, hardware verification, docs, commits | `references/tests-and-verification.md` |
| Silent-failure lessons from the qwen4_exp fix history and the ERNIE 5.5 registration gaps, grouped by subsystem | `references/lessons.md` |
