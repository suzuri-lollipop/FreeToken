# Tests, verification and docs

## The test bar (tests/README.md, binding)

A test must be able to fail on a plausible regression of production code:
name the future change that would turn it red. If the only answer is
"editing the test", do not write it. Admitted:

- bugs a normal run would not reveal (a wrong number that still looks like
  a number, a wrong cache slot, a crash only one interleaving/checkpoint hits);
- properties that took reasoning to get right, checked against something
  INDEPENDENT (torch reference, round-trip, CPU mirror, live registry);
- bookkeeping that is easy to forget (registry exports, arch -> parser maps,
  CLI choices lists, calibration tool family maps).

Not admitted: happy-path restatements, config tests echoing fixture values,
`assert_called*` / "no exception" checks. Extend an existing parametrize /
table / file before creating anything; ask the user before adding test
files, directories, fixtures or helpers outside `tests/models/<model>/`,
and before deleting a test. A new model's own tests go in
`tests/models/<model>/` without asking.

## Layout shipped with qwen4_exp

`tests/models/qwen4_exp/`:

| File | What it guards (oracle) |
|---|---|
| `conftest.py` | Runtime hygiene: `set_tp_info(rank=0, size=1)` once, `core._GLOBAL_CTX = None` before and after every test (autouse). Copy this for a new family. |
| `common.py` | Toy `hf_config()` builder: the SHIPPING geometry everywhere it matters for QSA (head_dim 256, index_head_dim 128 with 64-wide partial rope, index_ratio 4, budget 2048 -> 512 blocks -> 2051 selected tokens, page_size 64); only head counts / hidden / layer count scaled down to fit a shared GPU. `requires_cuda` skip marker. No tests itself. |
| `ple_hf_ref.py` | HF PLE reference implementation (independent oracle). |
| `test_config.py` | parse_config normalization rules (layer-type aliases, one-indexed PLE ids, rope truncation, defaults). |
| `test_skeleton.py` | HC mix/combine vs HF formulas; merged GEMM layout + top-level mixer; fused combine_norm vs sequential; PLE hash + forward vs HF; QSA layer vs HF dense; shared-gate fusion vs eager; full decoder stack prefill + decode. |
| `test_ple.py` | PLE internals: hash constants, windows/eos boundary, conv paths, slot-state lifecycle, table backends vs GpuResidentTable oracle. |
| `test_ple_disk.py` | DiskRowTable: bitwise row staging, row cache eviction byte accounting, decode-graph fill vs python row build, oracle parity per dtype, in-flight lookup fencing, graph sync protocol, platform gates. |
| `test_qsa_backend.py` | Sparse backend vs dense oracle: prefill/decode dense below budget, flashinfer dense vs sparse path, chunked prefill == one-shot, decode graph replay == eager, verify after rejected rows, row-chunked scoring, two layers keep separate slab slots. |
| `test_qsa_hf.py` | QSA indexer math vs the HF reference. |
| `test_qsa_kernels.py` | Each vendored triton kernel vs its torch reference (compress/score/topk/expand/attend). |
| `test_weight.py` | Synthetic safetensors checkpoints: fusion layouts (HC down+inject+pad, qkv round-trip, GDN in_proj splits, gate_up), zero-centered norms raw + suffix-list coverage, hash constants int64, PLE table concat/shard-count rejection, `read_range_into` correctness/thread-safety, AOT entry geometry + every registry arch claimed, emitted keys == model state dict, fp8 per-channel/per-tensor/mxfp8 dialect names, llm-compressor expert spec, checkpoint-vs-quant-config rejection, TP rank slices (text + tower), MTP dense TP sharding. |
| `test_weight_ckpt.py` | `needs_weights`, gated on `FREETOKEN_QWEN4EXP_MODEL` (local checkpoint dir): emitted names unique/complete == state dict, sampled tensors bitwise vs checkpoint, PLE table pinned load matches, sampled experts match, expert bank bytes == AOT row table. |

The GDN op tests moved to `tests/layers/test_gdn.py` when the op became
shared (#841); MTP draft coverage lives in `test_weight.py` and
`tests/spec/`. Follow the module, not the history.

Subsystem tests extended in the same commit (extend these tables for a new
pool/backend instead of creating parallel files):
`tests/kvcache/test_qsa_pool.py`, `test_linear_state_pool_alloc.py`,
`tests/engine/test_attention_backend_matrix.py`, `test_cache_budget.py`,
`tests/moe/test_fused_moe.py`, `tests/scheduler/test_hybrid_cache_manager.py`.

## Running

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[accel]"
uv run pytest tests/ -m "not slow"          # full suite minus long sweeps
uv run pytest tests/models/<family>/        # the family's own tests
FREETOKEN_QWEN4EXP_MODEL=/path/to/checkpoint uv run pytest tests/models/qwen4_exp/test_weight_ckpt.py
```

- GPU tests skip themselves without CUDA; marlin NVFP4 tests need a vllm
  venv (`vllm>=0.14,<0.15`).
- On tmpfs `/tmp` hosts point basetemp at a disk:
  `uv run pytest tests/ --basetemp=/home/$USER/pytest-tmp`; multi-GiB
  fixtures delete themselves in teardown.
- C++ extensions under `kernel/csrc/`: rebuild with
  `python setup.py build_ext --inplace` after changing them.

## Hardware verification (AI policy: mandatory, honest)

The human must run the real serve path; report exactly what ran. Pattern:

```bash
ft serve --model nvidia/Qwen3.8-Flash-Next-NVFP4 \
    --kv-cache-dtype fp8 --num-tokens <budget>      # long context on 24 GB cards
ft serve --model <checkpoint> --tp-size 2 --gpu 0,1
ft serve --model <checkpoint> --speculative-algorithm mtp
ft serve --model <checkpoint> --ple-backend disk|pinned|gpu
```

- Verify a fix by re-running what failed (the command/request/checkpoint
  from the issue) and report the result; if you cannot run it, say so and
  say what you ran instead. Verification alone does not call for a new test.
- Parser registration and the calibration family map are CPU-verifiable
  (explicit-choices parse, `spec_backends_for`) - run them. The streaming
  matrix tests the parser classes, not the CLI choices; auto resolution
  working is not evidence the choices list was extended. See
  `references/delivery-completeness.md`.
- Check both prefill and decode, chunked prefill + prefix reuse, CUDA-graph
  decode, and (if supported) speculative verify with rejected rows.
- Performance changes: A/B end-to-end numbers against `main` - same model,
  prompt, settings; tokens/s and TTFT; the commands used. Record them in a
  doc like `docs/qwen38_flash_next_optimizations.md` (baseline table,
  bottleneck breakdown, per-kernel A/B, accuracy error bounds).
- e2e gates live in `tests/e2e/` (real-server generation runs).

## Docs to update with a new family

- `docs/models.md`: supported-checkpoint table row (HF / FP8 / NVFP4 / GGUF
  links; siblings as extra links in the same row), GGUF family alias table,
  image-token table (if multimodal), family notes (host tables, KV dtype
  guidance, TP, draft). The Notes entry records the serving facts: MoE
  strategy, TP, KV dtype support, draft support, bundled rope scaling +
  trained ceiling, KV bytes/token. TP / draft / GGUF are mandatory, so a
  "limitation" there is only a surfaced exception the owner agreed to solve
  next - see `references/delivery-completeness.md`.
- `docs/cli.md`: new flags and speculative-draft sources.
- `docs/quickstart.md` when the serve example changes.
- Delivery completeness (TP >= 2, speculative draft, GGUF, parser four-table
  registration, calibration family map) is a set of MANDATORY requirements,
  not a follow-up: checklist in `references/delivery-completeness.md`. Each
  must be verified on real hardware - a TP=2 forward, a speculative verify
  with rejected rows, a GGUF load - not just the TP1 path. The synthetic-
  checkpoint harness (`tests/models/<family>/`) runs these on GPU without the
  full-size weights.

## Commits / PRs

- Conventional Commits, one line, imperative, lowercase, no trailing
  period: `feat(qwen4_exp): support Qwen3.8-Flash-Next`. Scope = the module
  or family. Breaking: `!` after scope + body why.
- Follow-up fixes keep the family scope: `fix(qwen4_exp): ...`,
  `perf(qwen4_exp): ...`; cross-subsystem refactors use that subsystem's
  scope (`refactor(layers): share one GatedDeltaNet op ...`).
- One change per PR, linked issue, hardware + checkpoint ID + exact command
  in the description. Roadmap items discussed with maintainers first;
  features not on the Roadmap start as an issue.
- Only commit when the user asks; never push / open PRs on their behalf.
  Attribution if wanted: `Assisted-by: <agent name>`.
