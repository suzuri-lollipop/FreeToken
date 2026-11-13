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
| `ft bench bw` | Benchmark CPU vs PCIe bandwidth to calibrate hybrid MoE decode |

`ft --version` prints the installed version (torch-free; nightly wheels carry a
`+g<sha>` build stamp, tagged releases a bare version). Every command supports
`--help`.

## ft serve

```bash
ft serve --model <path-or-hf-id> [options]
```

`--model` is the only required flag - dtype, attention backend, MoE backend,
MoE cache size, KV capacity, CUDA-graph sizes and the tool-call/reasoning
parsers all resolve automatically from the checkpoint and the GPU.

### Model

| Flag | Default | Meaning |
|---|---|---|
| `--model-path`, `--model` | required | Local dir, HF repo id, a `.gguf` file, or an FTW dir (auto-detected) |
| `--served-model-name` | basename of `--model` | Model id reported by `/v1/models` |
| `--hf-overrides` | - | JSON object applied to the checkpoint's config as vLLM's `--hf-overrides`: a nested config section updates key by key, any other value is replaced whole. A YaRN `rope_parameters` override serves `original_max_position_embeddings * factor` positions |

### Server & runtime

| Flag | Default | Meaning |
|---|---|---|
| `--host` | 127.0.0.1 | Bind address |
| `--port` | 1919 | Bind port |
| `--gpu` | GPU 0 | GPU to run on, one per tensor-parallel rank: a UUID from `nvidia-smi -L` or an `nvidia-smi` index; see [below](#choosing-a-gpu) |
| `--api-key` | `$FREETOKEN_API_KEY`, else off | Require the key on every route except `/health`, as `Authorization: Bearer <key>` or `x-api-key: <key>`; other requests get a 401 |
| `--max-running-requests` | 4 | Max concurrently running requests |
| `--max-output-tokens` | 32768 | Default output budget for requests that omit one |
| `--max-seq-len-override` | from checkpoint | Max sequence length |
| `--max-prefill-length` | 8192 | Chunked-prefill chunk size in tokens; with offloaded experts see [performance/prefill.md](performance/prefill.md) |
| `--cuda-graph-max-bs`, `--graph` | = max running requests | Max batch size captured as CUDA graphs |
| `--decode-log-interval` | 40 | Scheduler status line every N decode steps |

### Diagnostics

| Env | Default | Meaning |
|---|---|---|
| `FREETOKEN_LATENCY` | off | Scheduler-loop latency profiler: one `[latency]` line per window splitting wall time into `recv` (idle/message wait), `sched` (batch build), `fwd_issue` (launch), `drain` (waiting on the previous batch's GPU work) plus queue depth and the spec-verify tally. Host timings only, never a device sync, CUDA-graph safe |
| `FREETOKEN_LATENCY_EVERY_S` | 10 | Window between `[latency]` dumps |
| `FREETOKEN_LATENCY_OUT` | off | Append each window as one JSON line to this path (what the launcher's calibration reads) |

### Prefill / decode scheduling

| Env | Default | Meaning |
|---|---|---|
| `FREETOKEN_PREFILL_DEBT_S` | 2.0 | Wall-clock ITL guard: once prefill has held the engine this many seconds while a decode was runnable, the next step goes to decode (the chunk in flight already counts; the count-based `--prefill-decode-interval` cannot bound ITL when chunk times swing by 10x). `0` keeps the pure count-based guard |
| `FREETOKEN_ADAPTIVE_CHUNK_TARGET_S` | off | While a decode waits, shrink the prefill chunk so one chunk's forward time tracks this target (the waiting user's stall is one chunk, so the target is the stall budget); nobody waiting keeps the full `--max-prefill-length`. Chunks stay 64-aligned, floored at 128 tokens |
| `FREETOKEN_MIXED_PREFILL_TOKENS` | off | With N > 0 a new prompt's chunk runs WITH every running request in one batch (their single-token rows ride the chunk's forward), so decodes advance on every forward instead of waiting for a dedicated decode step; N caps the chunk. While a decode runs this replaces the interleave guard above. The mixed forward stays eager (no CUDA graph, no spec verify until the prompt queue drains), and riding GDN rows take the chunked-scan kernel: same math, different rounding |
| `FREETOKEN_PIPED_PREFILL` | off | Strata's `STRATA_PIPED_CONCURRENT` port: prefill chunks run on a dedicated thread and a default-priority side stream while the scheduler loop keeps issuing decode steps on the high-priority engine stream, so running decodes never wait for a prompt (measured: worst inter-token stall 3.8s -> 0 behind a 16k-token queue; the joiners pay ~+40% TTFT, Strata's trade curve). **Under TP** the pipe's collectives run on their own torch NCCL group (built at engine init and probed there, so a rank that cannot build it turns piping off on every rank; `NCCL_MAX_CTAS` caps a spinning pipe kernel so it cannot starve the decode kernels, and `FREETOKEN_SHM_AR` is forced to 0 because the shm one-shot AR bakes a resident spin kernel into the decode graphs that `NCCL_MAX_CTAS` cannot cap - it holds the SMs until the peer's decode arrives and starves the pipe's collective the other way, which wedged both ranks on a cold 16k prompt), and the drain is rank-locked: rank0 broadcasts its readiness with the message count and every rank drains on that same iteration - never on its own earlier `chunk_done`, because a one-iteration skew composes the next decode batch differently per rank and wedges a collective. A forced wait past `FREETOKEN_PIPED_FORCED_WAIT_S` (30) crashes+restarts instead of wedging, and `FREETOKEN_PIPED_TIMEOUT_S` (600) bounds an in-flight chunk. `FREETOKEN_ADAPTIVE_CHUNK_TARGET_S` is ignored under TP for the same rank-consistency reason. Needs overlap scheduling and no speculative decoding (`--speculative-algorithm` off) |
| `FREETOKEN_ADMIT_SHORT_FIRST` | off | With N > 0 a pending prompt of at most N tokens takes this step's admission before longer ones queued ahead of it (arrival order within each class; chunk continuations always stay first). The chat turn stops queueing behind the 30k-token agent prompt |
| `FREETOKEN_ADMIT_MAX_WAIT_S` | 30 | A long prompt waiting this long in the admit queue counts as short (aging guard against starvation) |

### Sampling guard at high temperature

| Env | Default | Meaning |
|---|---|---|
| `FREETOKEN_SAMPLING_GUARD` | on | `0`/`off` disables the tail guard: a request that samples at `temperature >= 1.0` (the regime this checkpoint's `generation_config.json` recommends for thinking mode) without setting `min_p` gets `min_p=0.05`. Quantized flash-class checkpoints drift into particle-drop fragmentation in exactly that regime (measured on Qwen3.8-Flash-Next NVFP4 at temp 1.0; greedy and temp < 1.0 are untouched, and any explicit `min_p` from the client wins). The guard adds NO frequency penalty on purpose: a count-linear penalty strips every 。、and newline from a long generation (they repeat hundreds of times legitimately), which measured worse than the decay it aimed at. The `min_p` request field itself (OpenAI + Anthropic wires, `0..1`) filters tokens below `min_p x p_max` before top-k/top-p |
| `FREETOKEN_THINK_WRAP` | on | Thinking-budget wrap-up (llama.cpp reasoning-budget idea, keyed to max_tokens): while a request is inside a think block and its remaining output budget drops to the reserve, the next token(s) are forced to the end-of-thinking tag so the model spends what is left on the final answer or tool call instead of being cut off mid-thought. Only acts inside the think block, so post-thinking content and tool calls are untouched; skipped for `ignore_eos` requests and while speculative decoding is armed. `0`/`off` disables |
| `FREETOKEN_THINK_WRAP_RESERVE` | 0.1 | Fraction of max_tokens reserved for the final answer: the wrap-up fires when the remaining budget falls to this fraction (clamped to at least 1 token). `FREETOKEN_THINK_WRAP_START` / `FREETOKEN_THINK_WRAP_END` override the think tags (default `<think>` / `</think>`); a template that pre-opens the think block in the prompt is detected too |

### Engine recovery

The engine runs in worker processes that the serve supervises. When one exits after readiness, or
hangs without exiting, the serve does not keep answering requests over a dead model: it closes
admission (new requests get a retryable `503` with a `Retry-After` and a message naming the
reload), re-spawns the workers and reloads the weights in place, and opens admission again when
the new engine reports ready. In-flight requests of the lost engine end with an `engine_stopped`
error rather than hanging. `GET /health` answers 200 throughout, because the server does answer,
and says what is behind it: `loaded` (is there an engine to serve a request), `generation` (how
many engines this serve process has started), `auto_load`, and `restarting` with `reason`,
`attempt`/`tries`, and `state` (`pending`, `restarting`, `waiting`, `give_up`, `disabled`). After
the tries run out the serve stays up and idle, and the next request starts another round.

| Env | Default | Meaning |
|---|---|---|
| `FREETOKEN_BACKEND_RESTART` | on | `0`/`off` restores the old behaviour: a worker death after readiness takes the serve down (10 s grace) instead of restarting the engine |
| `FREETOKEN_BACKEND_RESTART_TRIES` | 3 | Consecutive failed revive rounds before the serve gives up and waits to be asked again |
| `FREETOKEN_BACKEND_RESTART_RETRY_S` | 15 | Seconds between revive attempts, and how long an engine has to stay up before the failure count starts over |
| `FREETOKEN_BACKEND_RESTART_JOIN_S` | 60 | Seconds a revive waits for the old workers to exit before SIGKILLing them. A rank that is merely hung holds the rendezvous port and the VRAM its replacement needs, so nothing is spawned until every one of them is gone |
| `FREETOKEN_ENGINE_STALL_S` | 90 | How long an engine may owe an answer, use no CPU and no disk, and say nothing, before it is written off as frozen and revived. `0` disables the watch (a host whose GPU work leaves no CPU trace may want that) |

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
  dense `Qwen3_5ForConditionalGeneration` family (Qwen3.6/3.8-27B) with its vision tower, and `Qwen4Exp`
  (Qwen3.8-Flash-Next) with its dense projections and its vision tower. Anything else reports
  `does not shard its checkpoint for tensor parallelism yet` at startup, before any rank is spawned.
* A tower whose reader still emits full-width weights (`Glm5NextForConditionalGeneration`) is
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
| `--embed-weights` | gpu | Where the input token embedding tables live. `host` keeps them in pinned host RAM and each lookup reads its rows over PCIe, so their VRAM goes to the expert cache and KV. A table tied to the LM head stays on the GPU, and `host` is refused when there is no other table; Gemma-4 E2B / E4B's per-layer table is the one that moves, the way those models fit an 8 GB GPU |
| `--online-quant` | - | Convert the Linear layers and input embedding tables that the checkpoint stores in bf16 at load. `df11` compresses them losslessly to about 11 bits per weight and decodes them bit-for-bit on use: about 30% less VRAM for those weights, at the cost of a decode per use. Layers the checkpoint quantizes, the LM head, a table tied to it, vision encoders and layers whose weight the model reads directly are left as they are, and so are tables under TP > 1; refused when no layer qualifies |
| `--online-quant-modules` | all | Limit `--online-quant` to these modules: comma-separated checkpoint module names or glob patterns, e.g. `model.layers.*.self_attn,model.embed_tokens` (`model.language_model.layers.*` where the checkpoint nests the text model); a name also covers the modules under it, and one that selects no convertible layer is refused. A fused layer (`qkv_proj`) must be covered in all of its checkpoint parts |
| `--page-size` | 1 | KV page size; DSV4 forces 128, the TRTLLM backend needs 16/32/64, SWA models require 1 |
| `--cache-type` | radix | `radix` (prefix reuse; SWA/GDN-aware variants picked automatically) or `naive` |
| `--attention-backend`, `--attn` | auto | `trtllm`/`fi`/`fa`/`triton`/`dsv4_sparse`/`dsa`; `prefill,decode` pair allowed; auto picks per model + GPU |
| `--kv-cache-dtype` | auto | How K and V are stored. `auto` keeps the compute dtype. `fp8_per_token_head` (e4m3) and `int8_per_token_head` store 8 bits per element plus one fp32 scale per token and head (two per MLA latent row, for ckv and kpe), about half the KV memory. The `triton`, `qsa_sparse`, `m3_sparse` and `dsa` backends read them; `auto` picks one of these. They need `--dtype bfloat16` and no tensor parallelism. An `mtp` draft stores its KV in the same format |

### MoE offload

See [models.md](models.md#moe-strategies) for what each strategy does, [performance/prefill.md](performance/prefill.md) for prefill speed, and [performance/decode.md](performance/decode.md) for decode speed.

| Flag | Default | Meaning |
|---|---|---|
| `--moe-strategy` | auto | `fused`/`offload`/`cpu`/`hybrid`; auto -> offload; fused on unified-memory GPUs (GB10), see [models.md](models.md#unified-memory-gpus-gb10--dgx-spark). `--moe-backend` is the deprecated old spelling |
| `--quant-backend` | auto | Kernel per quantized layer type, `layer[.kind]=name` entries: `linear=marlin,moe=b12x` or `moe.nvfp4=triton`. A layer-level entry applies to every kind whose table lists the name |
| `--nvfp4-backend` | - | Deprecated: stands in for `--quant-backend moe.nvfp4=<marlin\|b12x\|triton>` (`flashinfer` means b12x); cannot be combined with `--quant-backend` |
| `--moe-cache-size` / `--moe-cache-rate` / `--moe-cache-mem` / `--moe-cache-auto` | auto | GPU expert-cache size as slots / fraction of all experts / memory such as `12GiB` / sized from free VRAM (mutually exclusive; auto is enabled by default for offload-family strategies) |
| `--moe-cache-scope` | unified | `unified`: one LRU pool of slots per expert quant (quant type and bank layout), so all MoE layers share one pool when they share a quant format; `per-layer`: each layer has its own pool. `per-layer` does not combine with `--moe-strategy cpu` or `--moe-cpu-layers` |
| `--moe-cache-split` | equal | How the pools share the slots: `equal` (in proportion to their layer counts), or `profile:<path>`: every pool keeps `top_k` slots and the rest go to the most used experts in the profile's per-layer expert counts. With `offload`, a pool smaller than the expert count lowers `--max-running-requests` to its slots / top_k |
| `--kv-reserve-tokens` | 8192 | KV token floor reserved before `--moe-cache-auto` fills experts |
| `--moe-cpu-threads` | physical cores | CPU worker threads for the cpu/hybrid executor |
| `--moe-cpu-layers` | all on GPU | With `offload`/`hybrid`: which MoE layers decode on CPU (`3,7,11`, a count, a fraction, or `auto`). `auto` is for hosts that cap CUDA pinned memory (Windows/WSL): it pins layers from the middle out until CUDA refuses one and OS-locks the rest (`FREETOKEN_PIN_BUDGET_GB` and FTW checkpoints lock from a size estimate instead), see [models.md](models.md#hosts-that-cap-pinned-memory-wsl); every value needs an expert format the CPU executor serves (bf16, fp8_block, nvfp4, mxfp4, GGUF on x86 CPUs, except gpt-oss GGUF files whose per-expert biases decode on the GPU only) |
| `--moe-ram-budget` | 0 (auto) | Machine-wide host RAM ceiling in GiB the expert banks may pin; `0` sizes it from MemAvailable minus headroom. Raise it only with room for the dense staging, KV host caches and page cache that ride on top; a tensor-parallel boot divides the figure across its ranks |
| `--moe-disk-tier` | auto | `auto`: when the expert banks exceed host RAM, map the head+tail MoE layers' banks from a repacked FTW sidecar on NVMe (built one-time on first serve, next to the model; `FREETOKEN_BANKS_DIR` overrides), pin what still fits; disk layers decode on CPU and prefill with readahead. `all` = every layer on disk; `off` = never (OOMs when banks exceed RAM) |
| `--moe-bank-profile` | none | FTXP expert profile (build with `ft profile write-round-robin`, inspect with `ft profile inspect`); with the disk tier, spend `--moe-hot-frac` of the host RAM budget on the profile's hottest experts of the disk layers (compact pinned banks the CPU executor prefers over the mmap) instead of more whole layers. A profile measured on your workload's routing is what pays; round-robin is a placeholder |
| `--moe-hot-frac` | 0.25 | Fraction (0..1) of the host RAM budget given to the profile's hot experts |
| `--moe-skip-minw` | 0 | Hybrid decode drops CPU-computed routes whose weight is under this fraction of the token's routed total (`--moe-skip-keep`, default 2, always stay). 0 = every route computed exactly; 0.1 is the Strata-GLM tuned default (about 30 percent faster decode there, KL ~0.02) |
| `--moe-skip-keep` | 2 | Routes per token the expert skip never drops |
| `--moe-hybrid-max-fetch` | auto | With `hybrid`: max experts fetched over PCIe per layer per step; rest computed on CPU |
| `--moe-locked-fetch` | auto | With `hybrid` and locked layers: share of a locked layer's misses fetched per step into its GPU slots through pinned staging. Auto takes p / (2p + c) from the `ft bench bw` row of the layer's kind (at most 1 per step without a row); 0 = locked layers decode on CPU only |
| `--moe-locked-max-fetch` | 8 | Most experts a locked layer fetches per step; sizes the staging (that many rows of the widest expert, per bank) |
| `--moe-collect-stats` | off | Log decode MoE stats with each decode log line: hit rate, and per layer-step the misses, experts fetched and experts computed on CPU, for pinned, locked and CPU-only layers |
| `--moe-prefill-hit-d2d` | off | Prefill: copy cache-hit experts device-side, stream only misses (CUDA >= 13) |
| `--disable-moe-prefill-overlap` | overlap on | Disable the two-buffer prefill copy overlap |

### Speculative decoding

| Flag | Default | Meaning |
|---|---|---|
| `--speculative-algorithm`, `--spec-algo` | off | Draft method. `mtp` drafts with the checkpoint's multi-token-prediction layer: Qwen3.5 / Qwen3.6 HF or GGUF checkpoints, and Qwen3.8-Flash-Next HF or GGUF. `dflash` drafts a block of tokens per step with a DFlash 2 draft checkpoint over the target's layer outputs, for the Qwen3.5 / Qwen3.6 / Qwen3.8 family and Qwen3.8-Flash-Next: `incoai/Qwen3.8-27B-DFlash2` for Qwen3.8-27B, `incoai/Qwen3.6-35B-A3B-DFlash2` for Qwen3.6-35B-A3B, `incoai/Qwen3.8-Flash-Next-DFlash2` for Qwen3.8-Flash-Next |
| `--speculative-num-steps`, `--spec-num-steps` | the method's default (3 for `mtp`, the draft's `block_size` - 1 for `dflash`) | Draft tokens per decode step, 1 to 8 for `mtp`; 1 to the draft's `block_size` - 1 for `dflash` |
| `--speculative-draft-model-path`, `--spec-draft-model` | the target checkpoint | Where the draft weights load from. A Qwen3.8-Flash-Next target also takes a Strata MTP pack (a GGUF that keeps the checkpoint's `mtp.*` tensors under their HF names), and a Qwen3.8-Flash-Next GGUF target needs one; a Qwen3.5 / Qwen3.6 GGUF target drafts from its own `nextn` MTP block (the block llama.cpp keeps after the last decoder layer), or from an `mtp-*.gguf` sidecar of the same model (unsloth's `MTP/` releases) given here; for `mtp`, a separate checkpoint folder is an HF-target option, read with the target's quantization config, and a GGUF target refuses a folder draft. `dflash` requires it: the DFlash 2 draft checkpoint (bf16 safetensors with its `config.json`), whose `target_layer_ids`, `num_target_layers`, hidden size and vocabulary must match the target |
| `--speculative-token-map`, `--spec-token-map` | off | `mtp` only. A file of distinct little-endian int32 token ids (`python scripts/make_token_map.py MODEL` writes one): the draft computes its logits over those LM head rows alone, from a copy made at startup whose size is logged. That VRAM comes out of the expert cache when `--moe-cache-auto` sizes it (the default for hybrid and offload without a cache size), and out of the KV cache otherwise. The target still verifies over the whole vocabulary, so the output distribution does not depend on the file, only the acceptance rate does. A tied or block-FP8 LM head is refused |

Speculative decoding needs `--tp 1` and runs without overlap scheduling; targets with sliding-window or DSV4 window-tier KV are refused at startup. `mtp` refuses an FTW checkpoint as the draft source, since FTW drops the `mtp.*` tensors, and a Qwen3.5 / Qwen3.6 GGUF converted without its MTP layer (llama.cpp `--no-mtp`) when no `mtp-*.gguf` sidecar is given: such a trunk has no draft of its own to read, while a `--no-mtp` trunk plus a sidecar works (the trunk config's shapes, the sidecar's weights).

`dflash` keeps the draft's context in a per-request ring of the draft's sliding window (2048 positions for the released drafts) and, on a hybrid GDN target, in a window on the GDN state slots, so a prefix-cache hit restores it with the GDN snapshot; the window adds `(window + 64) x hidden` activation-dtype bytes to every GDN slot (`/v1/cache/status` counts them). A target without GDN layers needs `--cache-type naive`. Quantized drafts and DFlash v1 drafts are refused. With sampling, the drafts are drawn from the draft's own distribution over its top-16 candidates and verified by rejection sampling against the target's distribution, so the output distribution is the target's.

### API behaviour

| Flag | Default | Meaning |
|---|---|---|
| `--sampling-defaults` | model | Fill unspecified sampling params from the checkpoint's `generation_config.json` (`none` = framework defaults) |
| `--tool-call-parser` | auto | Tool-call format; auto-inferred from the model family |
| `--reasoning-parser` | auto | Splits chain-of-thought into `reasoning_content`; auto-inferred; `off` disables |
| `--enable-cache-report` | off | Report prefix-cache hits in each response's usage block |
| `--enable-timings-report` | off | Attach a llama.cpp-shaped per-request `timings` object to responses |
| `--anthropic-inline-system` | auto | Placement of late Anthropic system instructions: `auto`, `preserve`, or `fold` |

With `--enable-timings-report` every chat completion, completion and Responses response
also carries a llama.cpp-shaped `timings` object next to `usage` (`cache_n`, `prompt_n`,
`prompt_ms`, `prompt_per_token_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`,
`predicted_per_token_ms`, `predicted_per_second`, plus `queue_ms` as a FreeToken
extension and `draft_n` / `draft_n_accepted` under speculative decoding), so frontends
like llama-swap and Open Web UI show per-request prefill/decode rates. All spans are
measured in the scheduler: `queue_ms` covers admission to the first prefill chunk,
`prompt_ms` the first chunk to the first sampled token (chunk gaps included),
`predicted_ms` the first to the last committed token; `prompt_n` counts only the
evaluated tokens and the decode rates divide by `predicted_n - 1`, as in llama.cpp. On
streams the object rides the final chunk; a multi-prompt `/v1/completions` request
reports the first prompt's spans, as llama.cpp does. `/v1/messages` never carries it
(llama.cpp emits no timings there either). A streaming chat completion, completion or
Responses request may set the llama.cpp top-level body field `"timings_per_token": true`
to carry the block with growing values on every chunk that follows an ack: text and
reasoning deltas, tool-argument fragments, and the finish chunk (or the usage chunk
with `include_usage`). Without the server flag the field is ignored.

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
| `--mmproj` | found next to the model | GGUF models only: the llama.cpp vision tower file (`mmproj-*.gguf`), see [Image input](models.md#image-input). Without it, FreeToken takes a `*mmproj*.gguf` that fits the model (output width, tower type, image tokens) from the model file's own folder, BF16 before F32, F16 and quantized files. With none, or with files of several models it cannot tell apart, the model serves text only; a repo that keeps each quant in a subfolder and the mmproj at the top needs `--mmproj`. The log names the file it loads |
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
- `--api-key KEY` (default `$FREETOKEN_API_KEY`) attaches to a server started with `--api-key`; with `--model` the shell uses that server's own key
- `/help` inside the shell lists the commands (`/think`, `/cache`, `/reset`).

## ft ctl

```bash
ft ctl [--base-url http://127.0.0.1:1919] [--api-key KEY] [--timeout 10] [--json] <subcommand>
```

`--api-key` defaults to `$FREETOKEN_API_KEY`.

| Subcommand | Endpoint | Purpose |
|---|---|---|
| `health` | `GET /health` | Server status, model, load progress, and whether an engine is behind the API (`loaded`, [Engine recovery](#engine-recovery)) |
| `stats` | `GET /v1/stats` | Throughput, latency, VRAM, pool occupancy, speculative draft tallies, accepted input modalities |
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
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...) are cleared from the child
environment so the agent cannot silently fall back to a paid endpoint.
When `/v1/stats` reports `image` among `model.input_modalities`, the written
config declares the model image-capable, which Codex, OpenCode, OpenClaw and
dsh require before their image tools and attachments send anything; Claude
Code and Hermes need no declaration.

| Flag | Meaning |
|---|---|
| `--server URL` | Server to point the agent at (default `http://127.0.0.1:1919`) |
| `--api-key KEY` | Key of a server started with `--api-key` (default `$FREETOKEN_API_KEY`). The agent gets it in its environment; Hermes and OpenClaw keep it in their config files |
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
strategies alike (DeepSeek-V4 experts serve offload only);
`--moe-backend` is kept for compatibility. See the FTW caveats in
[models.md](models.md#notes); FTW files from older builds can be repaired with
[scripts/ftw_hotfix.py](ftw-hotfix.md) instead of reconverting.

## ft bench bw

```bash
ft bench bw --model /path/to/model.gguf   # every MoE layer kind of a checkpoint (GGUF or HF)
ft bench bw                               # canonical geometry per expert format
ft bench bw --gpu 1                       # a specific GPU (UUID or nvidia-smi index, as for ft serve)
```

Measures host-RAM vs PCIe bandwidth with the real cpu/offload MoE kernels and writes one
row per MoE layer kind, which `ft serve --moe-strategy hybrid --moe-hybrid-max-fetch -1` reads
to split each layer's misses.

- One profile per GPU, at `~/.cache/freetoken/benchbw/<gpu-uuid>.json`; a profile from other
  hardware or another profile version is ignored.
- A row's key is the layer's CPU format and options, expert shape, top_k and CPU worker
  threads; a layer without an exact row keeps a fetch cap of 1. Bench with the same
  `--cpu-threads` as the server's `--moe-cpu-threads`.
- Each run replaces the rows it benched and keeps the others.
- What to measure: `--model` (checkpoint paths or presets), `--dtype`, `--formats`, `--isa`.
- `--threshold` (default 2.0) sets each row's `recommended`: hybrid when CPU bandwidth beats
  PCIe by that factor.

On NVLink-C2C hosts, see [models.md](models.md#nvlink-c2c-hosts-gh200--gb200)
for `FREETOKEN_H2D_BLOCKS_PER_BANK` if `PCIe-gather` sits well below the linear H2D ceiling.
