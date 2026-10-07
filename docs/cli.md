# CLI reference

```
ft <command> [args]
```

| Command | Purpose |
|---|---|
| `ft serve` | Start the API server (OpenAI `/v1/*`, Anthropic `/v1/messages`, Responses) |
| `ft shell` | Chat with a server in the terminal |
| `ft ctl` | Query and manage a running server over HTTP |
| `ft launch` | Configure and launch a coding agent against a server |
| `ft checkpoint` | Convert an HF checkpoint to the FTW fast-load format |
| `ft bench bw` | Benchmark CPU vs PCIe bandwidth to calibrate the MoE backend |

`ft --version` prints the installed version (torch-free; nightly wheels carry a
`+g<sha>` build stamp, tagged releases a bare version). Every command supports
`--help`.

## ft serve

```bash
ft serve --model <path-or-hf-id> [options]
```

`--model` is the only required flag — dtype, attention backend, MoE backend,
MoE cache size, KV capacity, CUDA-graph sizes and the tool-call/reasoning
parsers all resolve automatically from the checkpoint and the GPU.

### Model

| Flag | Default | Meaning |
|---|---|---|
| `--model-path`, `--model` | required | Local dir, HF repo id, or an FTW dir (auto-detected) |
| `--served-model-name` | basename of `--model` | Model id reported by `/v1/models` |
| `--hf-overrides` | — | JSON object applied to the checkpoint's config as vLLM's `--hf-overrides`: a nested config section updates key by key, any other value is replaced whole. A YaRN `rope_parameters` override serves `original_max_position_embeddings * factor` positions |

### Server & runtime

| Flag | Default | Meaning |
|---|---|---|
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 1919 | Bind port |
| `--gpu` | GPU 0 | GPU to run on, one per tensor-parallel rank: a UUID from `nvidia-smi -L` or an `nvidia-smi` index; see [below](#choosing-a-gpu) |
| `--tensor-parallel-size`, `--tp-size` | 1 | GPUs to shard one model across; see [Multiple GPUs](#multiple-gpus-tensor-parallelism) |
| `--max-running-requests` | 4 | Max concurrently running requests |
| `--max-output-tokens` | 32768 | Default output budget for requests that omit one |
| `--max-seq-len-override` | from checkpoint | Max sequence length |
| `--max-prefill-length` | 8192 | Chunked-prefill chunk size in tokens |
| `--cuda-graph-max-bs`, `--graph` | = max running requests | Max batch size captured as CUDA graphs |
| `--decode-log-interval` | 40 | Scheduler status line every N decode steps |

### Choosing a GPU

For example, a machine with an RTX 5090 and an RTX 3060 Ti:

```console
$ nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 3060 Ti (UUID: GPU-2f3a9b1c-8d7e-4a05-b6c1-0e5f9a3d7b42)
GPU 1: NVIDIA GeForce RTX 5090 (UUID: GPU-9e8d7c6b-5a49-4f13-8207-c1b0a4e6d3f5)
```

```bash
ft serve --model ... --gpu 1             # by nvidia-smi index -- the 5090
ft serve --model ... --gpu GPU-9e8d7c6b  # the same card by UUID (a unique prefix is enough)
```

### Multiple GPUs (tensor parallelism)

`--tensor-parallel-size N` (`--tp-size`) shards one model across N GPUs: it starts one
engine process per rank, entry `i` of `--gpu` is rank `i`, and the ranks exchange partial
results with NCCL (FreeToken's own PyNCCL path by default; `--disable-pynccl` switches to
torch's). Attention heads, the KV pool, the MLP widths, the embedding and the LM head split;
norms, the MoE router and a sparse attention's indexer stay replicated.

```bash
ft serve --model Qwen/Qwen3.8-27B-FP8 --tp-size 2 --gpu 0,1   # 27 GB of weights over two 24 GB cards
```

What is TP-sharded today:

* Readers of `llama`, `qwen2`, `qwen3`, `qwen3_moe`, `minimax_m2`, `mistral`, `gpt_oss`, the
  dense `Qwen3_5ForConditionalGeneration` family (Qwen3.6/3.8-27B), and `Qwen4Exp`
  (Qwen3.8-Flash-Next) with its dense projections and its vision tower. Anything else reports
  `does not shard its checkpoint for tensor parallelism yet` at startup, before any rank is spawned.
* A tower whose reader still emits full-width weights (Qwen3-VL, Qwen3.6 / Qwen3.8-27B) is
  refused under TP with `the vision encoder's weights are not tensor-parallel sharded yet`: pass
  `--text-model-only` to serve those checkpoints across ranks, or keep `--tp-size 1` to keep the
  images.
* **Experts**: the offload banks shard for NVFP4 experts (`--moe-strategy offload`, which `auto`
  picks): the Triton kernel declares its banks at the rank's intermediate width and packs the
  matching slice, so host RAM, the GPU slot cache and the PCIe stream halve per rank (Qwen3.8-Flash-Next's
  63.5 GiB of expert banks become 31.8 GiB per rank at TP=2). The `cpu` and `hybrid` executors run
  on those per-rank banks too, and bf16 experts with `--moe-strategy fused` shard the same way.
  Refused under TP: an offload-family strategy (`offload` / `cpu` / `hybrid`, and `auto` resolving to
  it) over experts that are not NVFP4 -- the MXFP4, block-FP8 and bf16 offload readers still emit full
  width -- so serve those resident with `--moe-strategy fused`. The Marlin / b12x packs need a wider
  slice than the shard leaves, so the kernel selector falls back to Triton instead.
* That NVFP4 split is weighted by each rank's host->device bandwidth instead of cut evenly, by
  default for `--moe-strategy offload` at `--tp-size > 1`: every rank probes its own link before the
  weights load and the intermediate widths follow `bandwidth ** 0.4`, so a rank on a gen4 x4 slot next
  to a gen5 x16 one takes the smaller slice and both ranks finish their streamed experts at the same
  time (a pure bandwidth-proportional split would starve the fast rank's now-smaller slot cache). One
  failed probe keeps the even split everywhere, and a non-zero `--moe-cpu-layers` (including `auto`)
  disables the weighting because the CPU executor sizes its buffers off the even split.
  `FREETOKEN_EXPERT_SHARD_FRAC=0.375,0.625` pins the per-rank fractions and skips the probe,
  `FREETOKEN_EXPERT_SHARD_GAMMA=0` restores the plain even split, and `FREETOKEN_DENSE_SHARD_FRAC`
  tilts the dense head split (attention / GatedDeltaNet heads) the same way.
* `fi` and `qsa_sparse` attention. The other backends read global head counts and are refused
  (auto selection skips them).
* An FTW directory cannot be sharded: it stores the tensors already fused at full width, so
  point `--model` at the HF checkpoint when running `--tp-size > 1`.
* The runtime cache sliders (`ft ctl cache`) are refused at TP > 1; restart with new sizes.

Sizing has to divide: query heads, KV heads (no replication for the fused QKV projections),
GatedDeltaNet key/value heads, and `moe_intermediate_size / tp` staying on the expert format's
scale block. `--tp-size 2` and any power of two up to the head count works for the models above.

Each rank holds its own share of the weights, so the per-GPU memory roughly halves while the
layers all run on every rank. The price is one or two collectives per layer over the interconnect:
on two PCIe-attached cards measured at ~8 us each, so a small dense model actually gets slower
(Qwen3-0.6B: 374 tok/s at TP=1 against 320 at TP=2 on 2x RTX PRO 4000 Blackwell, with byte-identical
greedy output). Sharding pays where the weights or the expert stream do not fit one card:
`nvidia/Qwen3.8-27B-NVFP4` (21 GiB of weights) OOMs a single 24 GB card and runs at TP=2.

Note that greedy output is only byte-reproducible on some checkpoints: a quantized dense model
can shift a near-tie between two runs of the *same* configuration (the prefix-cache path and the
fp8 / nvfp4 dense GEMMs both reorder accumulation), so compare TP=1 and TP=N on answer quality,
not on identical text, unless you have verified repeatability at TP=1 first.

### KV cache & memory

| Flag | Default | Meaning |
|---|---|---|
| `--memory-ratio` | 0.9 | Fraction of the GPU's total VRAM the engine's whole per-rank footprint may use (weights + MoE cache + KV + CUDA context); the remainder is runtime headroom for CUDA graphs and activations |
| `--num-pages` / `--num-tokens` | auto | KV capacity override in pages / tokens (mutually exclusive; auto sizes from VRAM left after weights and MoE cache) |
| `--page-size` | 1 | KV page size; DSV4 forces 128, the TRTLLM backend needs 16/32/64, SWA models require 1 |
| `--cache-type` | radix | `radix` (prefix reuse; SWA/GDN-aware variants picked automatically) or `naive` |
| `--kv-cache-dtype` | auto | `fp8_e4m3` (`fp8`) stores K/V as fp8 e4m3: half the bytes per cached token, so roughly twice the prefix reuse for the same VRAM. Needs sm_89+ and an MHA, SWA-hybrid or Qwen3.8-Flash-Next (QSA) pool; on QSA only the paged K/V quantizes, its compressed index keys keep the model dtype. The latent-MLA, DSA, DSV4 and MiniMax-M3 block-sparse pools reject it at startup, so drop the flag for those models. `auto` keeps the model dtype |
| `--kv-cache-quant-scale` | 1.0 | For a quantized cache: the static scale K and V divide by on store and multiply back on read. Raise it if attention states exceed the e4m3 range (+/- 448 * scale), which would otherwise clamp |
| `--attention-backend`, `--attn` | auto | `trtllm`/`fi`/`fa`/`triton`/`dsv4_sparse`/`dsa`; `prefill,decode` pair allowed; auto picks per model + GPU. A `--kv-cache-dtype` cache needs a backend that applies its descales (`triton`, `fi` and the in-tree `qsa_sparse` today; `fa`/`trtllm` are refused with it) |

### MoE offload

See [models.md](models.md#moe-strategies) for what each strategy does.

| Flag | Default | Meaning |
|---|---|---|
| `--moe-strategy` | auto | `fused`/`offload`/`cpu`/`hybrid`; auto → offload, or hybrid with a `ft bench bw` profile; fused on unified-memory GPUs (GB10), see [models.md](models.md#unified-memory-gpus-gb10--dgx-spark). `--moe-backend` is the deprecated old spelling |
| `--quant-backend` | auto | Kernel per quantized layer type, `layer[.kind]=name` entries: `linear=marlin,moe=b12x` or `moe.nvfp4=triton`. A layer-level entry applies to every kind whose table lists the name |
| `--nvfp4-backend` | — | Deprecated: stands in for `--quant-backend moe.nvfp4=<marlin\|b12x\|triton>` (`flashinfer` means b12x); cannot be combined with `--quant-backend` |
| `--moe-cache-size` / `--moe-cache-rate` / `--moe-cache-auto` | auto | GPU expert-cache size as slots / fraction of all experts / sized from free VRAM (mutually exclusive; auto is enabled by default for offload-family strategies) |
| `--kv-reserve-tokens` | 8192 | KV token floor reserved before `--moe-cache-auto` fills experts |
| `--moe-cpu-threads` | physical cores | CPU worker threads for the cpu/hybrid executor |
| `--moe-cpu-layers` | all on GPU | With `offload`: which MoE layers decode on CPU (`3,7,11`, a count, a fraction, or `auto`). `auto` is for Windows/WSL only, where CUDA pinned memory is capped; every value needs an expert format the CPU executor serves (bf16, nvfp4, mxfp4), so fp8 experts cannot use it |
| `--moe-hybrid-max-fetch` | auto | With `hybrid`: max experts fetched over PCIe per layer per step; rest computed on CPU |
| `--moe-prefill-hit-d2d` | off | Prefill: copy cache-hit experts device-side, stream only misses (CUDA >= 13) |
| `--disable-moe-prefill-overlap` | overlap on | Disable the two-buffer prefill copy overlap |

### Speculative decoding (MTP)

`--speculative mtp` builds the checkpoint's MTP head, drafts the next token with it and verifies the
draft inside the same forward, so one step costs two target rows and emits one or two tokens. Only a
checkpoint that ships an MTP head can serve it.

Drafting is opportunistic: a request that does not qualify decodes normally, at its own cost. The
rules are what a log with no MTP activity means, so they are printed at start-up too. A request must
reach a decode batch of one (a spec step is a two-row forward for a single request), and the head
drafts over the prompt's residual rows: those exist only for rows the target forward actually ran,
so a prefix-cache hit never produced them and a prompt longer than the stash budget cannot be
covered in full. Both of those declines disappear with `FREETOKEN_MTP_COLD_START`: the head then
seeds its first draft from the row it is decoding and pays for the missing catch-up with a few
rejected verifies. Any number of prefill chunks is fine -- each chunk adds its rows to the stash --
and sharing a decode batch with another request is no longer terminal either: the staged draft is
dropped because it aims at a position that already committed, and drafting resumes the moment the
request runs alone. Greedy requests verify by comparing token ids
and can capture a CUDA graph; sampled requests verify by exact rejection sampling (output
distribution unchanged) and always run eager, so a sampled-only workload never prints a
captured-graph line. Only the drafting steps leave the overlap pipeline (the next batch's positions
and pages are built from the accept count), so a request that never drafts pays nothing for
somebody else's draft; `FREETOKEN_DISABLE_OVERLAP_SCHEDULING=1` restores the old serialized loop.
Enabling the head is not free even for traffic that never drafts: it adds one full-attention layer
to the pool sizing (KV plus its index slab, so the KV/expert pools are a layer smaller), and one
GDN scratch slot per running request is held back for verify rollbacks.

Acceptance is a property of the text as much as of the engine: how often a draft lands is how
predictable the next token is. The head's own ceiling is a separate, measurable quantity -- the
wiring probe (`_scratch/mtp_probe.py`, teacher-forced prose and code) put the shipping wiring at
0.560, and a live greedy session measured 0.578 -- while random word-salad prompts in the same
build measured 0.42-0.50. So compare acceptance only between runs of the same text and the same
sampling mode, and read a lone number with that in mind. Sampling mode matters on its own: a
sampled request lands below its greedy equivalent by construction, because rejection sampling
accepts a draft with probability `min(1, q(y)/p(y))` -- its ceiling is `sum(min(q, p))`, not an
argmax match -- and the flatter the request's temperature/top_p, the lower that sum. `[mtp-s]`
prints the ceiling next to the ratio for exactly this comparison, and
`FREETOKEN_MTP_MIN_ACCEPTANCE` is the floor that stops paying a second target row for a draft that
keeps missing; a sampled-only workload sits permanently closer to it than a greedy one.

What a draft costs, measured on one RTX-class card with this checkpoint and three identical cold
642-token sampled prompts (256 tokens out, every gate open so the head drafted 99.6% of steps):
21.9 tok/s with drafting off, 16.0 with it on, at acceptance 0.453 -- so a spec step costs about two
decode steps and break-even would need acceptance at 1.0, which no head reaches. Shrinking the GPU
expert cache from 4204 to 1200 slots slowed both arms by about a third (21.9 -> 14.8 without
drafting, 16.0 -> 10.6 with) and left the ratio unchanged: the second row is compute, not an expert
fetch, so no batching of drafts amortizes it away. The same step on a 10k-token prompt costs nearer
1.5 decode steps, which is why a long answer can break even at the acceptance it actually reaches
while a short one cannot -- the whole reason `streak` and `low_acceptance` stand down instead of
paying that price for a draft that lands ~0.15 of the time.

| Env var | Default | Meaning |
|---|---|---|
| `FREETOKEN_MTP_ACCEPT_WINDOW` | 64 | Verifies per acceptance report and per auto-off window (floor 8) |
| `FREETOKEN_MTP_MIN_ACCEPTANCE` | 0.35 | Below this acceptance within one window the request stops drafting; `0` disables the check |
| `FREETOKEN_MTP_REJECT_STREAK` | 3 | Consecutive rejected verifies that trigger a short stand-down (the per-step half of the floor); `0` disables |
| `FREETOKEN_MTP_SKIP_STEPS` | 8 | Decode steps that stand-down lasts before drafting resumes |
| `FREETOKEN_MTP_RESUME_AFTER` | one window | Decode steps a suspended request waits before re-probing the floor; `0` makes the trip terminal |
| `FREETOKEN_MTP_COLD_START` | off | Draft even when the prompt rows are gone (prefix hit) or over the stash budget: the head seeds its first draft from its own row instead of declining. Costs a few rejected verifies, buys back the multi-turn traffic that always hits the cache |
| `FREETOKEN_MTP_RESYNC_LAG` | 8 | Rows a request may decode unseen by the head before its stash is written off, and before a stand-down's resume re-seeds the draft instead of verifying it (`suspend_hole`) |
| `FREETOKEN_MTP_MAX_STASH_TOKENS` | 16384 | Prompt rows whose residual is held for the head's catch-up pass (~20 KiB/row at this model's width, so ~320 MiB per draftable request; also how long the catch-up pass is) |
| `FREETOKEN_MTP_SAMPLING` | on | `0` declines every sampled request (the pre-rejection-sampling behaviour) -- the A/B switch |
| `FREETOKEN_MTP_REPORT_INTERVAL_S` | 60 | Shortest gap between acceptance lines when verifies are too sparse to drive one |
| `FREETOKEN_MTP_DECLINE_LOG` | 5 | Decline lines printed per reason; the counters keep counting past it |
| `FREETOKEN_MTP_DEBUG` | off | Per-step trace: `[mtp]` decisions, `[mtp-s]` rejection ratios, `[mtp-t]` step timings. Reads the value for truth, so `0` is off; the timing line synchronizes the device every spec step, so never leave it on for a throughput run |

What to read when MTP appears to do nothing:

- `MTP spec: enabled ...` at start-up names the caps and which sampling modes may draft. No such line
  means `--speculative mtp` was not passed.
- `MTP spec: declined <reason> for req <uid>: ...` prints with the numbers the gate used (a capped
  number of lines per reason); the `declined: <reason>=<count>` breakdown rides on every tally
  line, and counts per request for the terminal reasons, per episode for `resync`, `streak` and
  `suspend_hole`.

| Decline reason | Means |
|---|---|
| `prefix_hit` | The prompt's first rows came from the prefix cache, so no residual exists for them and there is nothing to catch the head up on. Only a decline with `FREETOKEN_MTP_COLD_START=0`: with it on, the head drafts from where it is |
| `over_budget` | The prompt is longer than `FREETOKEN_MTP_MAX_STASH_TOKENS` rows (the line prints `stashed=` how far it got); cold start drafts these without the catch-up pass instead |
| `nongreedy` | Only with `FREETOKEN_MTP_SAMPLING=0`: the request samples and sampled drafting was switched off |
| `mm` | The prompt carries image rows; those keep the regular decode path |
| `no_gdn_pool` | No GDN state pool to clone for a rejection rollback (non-hybrid model, or the pool is off) |
| `exhausted` | The GDN pool had no scratch slot to spare for this request |
| `resync` | The decode batch grew past one request, so the head missed rows. Not terminal: the staged draft is dropped (it aims at a position that already committed) and drafting continues, one rejected verify later |
| `no_stash` | The prefill forward produced no residual rows for this request, so the head cannot be caught up. The forward gate declines whole batches and names no request, which is why this reason exists at all: it is the visible form of "the prefill never offered me rows". Cold start removes it by drafting anyway |
| `lag_overflow` | The request decoded more than `FREETOKEN_MTP_RESYNC_LAG` rows unseen by the head before ever drafting, so the stash (prompt-sized, otherwise pinned for the request's whole life) is written off. Terminal only with cold start off |
| `low_acceptance` | This request's acceptance fell under `FREETOKEN_MTP_MIN_ACCEPTANCE` within one window of verifies; drafting stands down for `FREETOKEN_MTP_RESUME_AFTER` decode steps and probes again (`0` ends it for the request) |
| `suspend_hole` | A stand-down lasted longer than `FREETOKEN_MTP_RESYNC_LAG` rows, so the staged draft (and the sampled path's carried density) was computed too many committed tokens ago to verify: both are dropped and the next single-request step cold-seeds a fresh draft. Not terminal |
| `streak` | `FREETOKEN_MTP_REJECT_STREAK` verifies in a row missed, so the next `FREETOKEN_MTP_SKIP_STEPS` are spent decoding: after three misses the next draft lands ~0.15 while a verify costs ~2.1 decode steps. Not terminal, and the one number to read before blaming the head |
| `no_room` | One row of output budget left: an accept could not pay out, so drafting is over for this request |

- `MTP spec: acceptance <a>/<n> = <rate>` prints when the verify window fills, when the report
  interval elapses, and as the queue drains; the periodic `Decode batch` line carries `mtp: ...` too,
  so the state is readable even with zero verifies.
- `GET /v1/stats` reports the same counters under `spec` (`verifies`, `accepted`, `acceptance_rate`,
  `declined`, plus the coverage split `steps` / `decode_steps` / `drafted_ratio` and the
  `cold_seeds` that had no catch-up pass behind them), and `spec` is `null` when speculative
  decoding is off -- which is how "not enabled" and "enabled, and nothing ever qualified" tell each
  other apart without reading a log.

### API behaviour

| Flag | Default | Meaning |
|---|---|---|
| `--sampling-defaults` | model | Fill unspecified sampling params from the checkpoint's `generation_config.json` (`none` = framework defaults) |
| `--tool-call-parser` | auto | Tool-call format; auto-inferred from the model family |
| `--reasoning-parser` | auto | Splits chain-of-thought into `reasoning_content`; auto-inferred; `off` disables |
| `--enable-cache-report` | off | Report prefix-cache hits in each response's usage block |
| `--anthropic-inline-system` | auto | Placement of late Anthropic system instructions: `auto`, `preserve`, or `fold` |

### Image input

Experimental. Needs a checkpoint whose family registers a vision encoder ([models.md](models.md#image-input) lists them and how each one
maps the flags below); a request carrying images is rejected otherwise. Images are accepted on all three protocols (OpenAI `image_url`,
Anthropic `image` blocks, Responses `input_image`) as an http(s) URL or base64. Images inside a tool
result (an Anthropic `tool_result` block from Claude Code's Read, a Responses `function_call_output`
from Codex's view_image) are moved to the user turn that follows the tool message, as vLLM does,
because chat templates render tool messages as plain text.
`GET /v1/stats` reports what the server accepts as `model.input_modalities` (`["text"]` or `["text", "image"]`),
so a client can gate its attachment controls without reading the checkpoint config.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--text-model-only` | off | Serve a multimodal checkpoint text-only: no encoder tower is built (its VRAM goes to the KV/expert pools) and every multimodal input is rejected. Same as `--mm-disable` with every encoder kind |
| `--mm-disable` | none | Encoder towers to leave unbuilt (`vision`, `audio`); every input they would serve is rejected |
| `--mm-encoder-weights` | host | Where the encoder tower's block weights live. `host` streams them from pinned host banks two blocks at a time behind the compute, so the GPU holds two blocks instead of the whole tower; small images pay the copy time, large ones hide it behind the compute. `gpu` keeps them resident. An encoder without a block stack stays resident either way |
| `--image-min-tokens`, `--image-max-tokens` | processor defaults | Per-image token budget: the image processor resizes every image to take between these many tokens, converted to the family's own units by its processor. A family with fixed budgets honors the maximum only and refuses one below its smallest budget at start-up |
| `--mm-processor-kwargs` | none | JSON object of extra keyword arguments for the checkpoint's image processor call, for knobs the token budget does not cover; applied after the budget, so an explicit key wins |
| `--mm-embed-cache-device` | cpu | Where encoded image embeddings live between prefill chunks. `cpu` keeps them out of the VRAM budget; `cuda` skips the copy back |
| `--allowed-media-domains` | any | Comma-separated hostname allowlist for image URLs; requests for other domains are rejected with a 400. Empty allows any domain |
| `--allowed-local-media-path` | off | Directory `file://` image refs may be read from; unset rejects local files |

## ft shell

```bash
ft shell                                    # attach to a running server
ft shell --model ~/models/Qwen3.6-35B-A3B   # serve + chat in one process
```

- Attach mode talks to `--server URL` (default `http://127.0.0.1:1919`)
- `/help` inside the shell lists the commands (`/think`, `/cache`, `/reset`).

## ft ctl

```bash
ft ctl [--base-url http://127.0.0.1:1919] [--timeout 10] [--json] <subcommand>
```

| Subcommand | Endpoint | Purpose |
|---|---|---|
| `health` | `GET /health` | Server status, model, load progress |
| `stats` | `GET /v1/stats` | Throughput, latency, VRAM, pool occupancy, MTP draft tallies, accepted input modalities |
| `generate [prompt] [--max-tokens N] [--ignore-eos]` | `POST /generate` | Raw completion smoke test (no chat template) |
| `cache` | `GET /v1/cache/status` | Cache pool table |
| `cache --moe N \| --kv N \| --mamba N \| --swa N [--wait 300]` | `POST /v1/cache/rebuild` | Live pool resizing without a restart (`k`/`m` suffixes; `--kv`/`--swa` in tokens) |
| `requests [--since N] [--limit N]` | `GET /v1/requests` | Recent request ring |

## ft launch

```bash
ft launch {claude,codex,dsh,hermes,openclaw,opencode} [options] [-- <agent args>]
```

Discovers the served model via `/v1/models`, writes the agent's provider
config, installs the agent CLI if missing, then launches it. Cloud API keys
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, …) are cleared from the child
environment so the agent cannot silently fall back to a paid endpoint.
When `/v1/stats` reports `image` among `model.input_modalities`, the written
config declares the model image-capable, which Codex, OpenCode, OpenClaw and
dsh require before their image tools and attachments send anything; Claude
Code and Hermes need no declaration.

| Flag | Meaning |
|---|---|
| `--server URL` | Server to point the agent at (default `http://127.0.0.1:1919`) |
| `--dry-run` | Print the planned config changes and command, touch nothing |
| `-y`, `--yes` | Approve install/config prompts |
| `--config` | Configure without launching |
| `--install-only` | Just install the agent CLI (needs no server) |
| `--force-reinstall` | Re-run the agent installer |
| `-- <args>` | Forwarded verbatim to the agent |

## ft checkpoint

```bash
ft checkpoint --model <hf_dir> --out <ftw_dir> [--dtype bfloat16] [--moe-backend offload] [--quant-backend moe.nvfp4=b12x] [--shard-gib 8] [--gpu <uuid-or-index>]
```

Converts an HF safetensors checkpoint to FTW, FreeToken's self-contained
fast-load format; point `ft serve --model` at the output dir. MoE experts are
always packed into expert banks, which serve the fused and the offload
strategies alike (GGUF and DeepSeek-V4 experts serve offload only);
`--moe-backend` is kept for compatibility. See the FTW caveats in
[models.md](models.md#notes); FTW files from older builds can be repaired with
[scripts/ftw_hotfix.py](ftw-hotfix.md) instead of reconverting.

## ft bench bw

```bash
ft bench bw                       # once per GPU
ft bench bw --dtype nvfp4,bf16    # only the formats you serve
ft bench bw --gpu 1               # a specific GPU (UUID or nvidia-smi index, as for ft serve)
```

Measures host-RAM vs PCIe bandwidth with the real cpu/offload MoE kernels and writes a
profile that `ft serve --moe-strategy auto` and `--moe-hybrid-max-fetch -1` then read.

- One profile per GPU, at `~/.cache/freetoken/benchbw/<gpu-uuid>.json`.
- Keyed on expert format + GPU, so a profile from other hardware is ignored rather than
  misapplied. An older single `benchbw.json` still counts if its GPU name matches.
- What to measure: `--dtype`, `--model`, `--formats`, `--isa`.
- `--threshold` (default 2.0) sets the call: recommend hybrid when CPU bandwidth beats PCIe
  by that factor.
