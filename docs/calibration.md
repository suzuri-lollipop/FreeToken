# Calibrating prompt lookup and parallel requests on this PC

`--speculative lookup` pays when a verify of `1 + K` rows costs less than the `K` decode steps it
replaces. That ratio is a property of the machine and the model: a fast box with experts in VRAM
can afford four or five drafts, a slower one where a wide extend reads more experts may only break
even at two, and a box whose decode step is already cheap may lose at every width. The defaults in
[cli.md](cli.md#prompt-lookup) were measured on one machine; `ft calibrate` measures the server on
this one and writes the result where a start script can use it.

The same run measures the other launch choice that changes with the PC: `--max-running-requests`,
the number of requests the scheduler may run at once. A batch of N reads N prompts and decodes N
rows together, so throughput usually rises with N until the KV cache or the experts' memory
bandwidth saturates. Prompt lookup only drafts a batch of one, so the two choices interact: the
calibration measures the parallel ceiling separately and leaves lookup enabled for the moments
when concurrency drops back to one.

## Run it

Stop any FreeToken server first. The calibration starts and stops the model once per candidate and
needs the VRAM to itself; its own default port is 1920 so it cannot silently reuse a running 1919.

```bash
cd module/fork/freetoken
.venv/bin/ft calibrate \
  --model-path ~/.cache/huggingface/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --served-model-name RadixArk/Qwen3.8-Flash-Next \
  --server-args "--tp-size 2 --max-running-requests 4 --memory-ratio 0.88 \
    --num-tokens 131072 --moe-strategy offload --moe-cache-auto --moe-cpu-layers 0 \
    --host 127.0.0.1 --kv-cache-dtype fp8 --prefill-decode-interval 4" \
  --out-dir ~/calibration/qwen38-lookup --tag qwen38-lookup
```

This is the full sweep: 6 widths, 3 minimum matches, 4 context floors, 3 acceptance floors and
the parallel counts 1/2/4, plus the default and the no-spec baseline. With 192-token answers and
two repeats per prompt it restarts the model about 19 times; expect roughly 20-45 minutes depending
on load time.
`--quick` uses widths 1/4/6, matches 3/5, contexts 128/256 and the same parallel counts with one
repeat (about 12 restarts), and `--smoke` runs widths 4 and 5 with short answers to check the
plumbing -- smoke is not a calibration and its numbers should not be used. `--reps N` changes the
measured repeats per prompt in a full run, and `--parallel 1,2,4,8` changes the concurrency
candidates (each value is also the server's `--max-running-requests` for that arm).

`--out-dir` gets two files:

- `freetoken-lookup-<tag>.json` -- the launch config plus every measurement;
- `run-freetoken-lookup-<tag>.sh` -- executable; when lookup won, it exports the chosen
  `FREETOKEN_LOOKUP_*` values and adds `--speculative lookup`, otherwise it launches the parallel
  winner without speculation.

`<tag>` comes from `--served-model-name` unless `--tag` overrides it (dots become hyphens; passing
`--tag` keeps the names short). The script is ready to run:

```bash
~/calibration/qwen38-lookup/run-freetoken-lookup-qwen38-lookup.sh
```

## What it measures

Every arm answers the same four prompts after a 16-token warm-up that fills the prefix cache, so
every arm emits the same number of tokens. A parallel arm of N sends them in batches of N (four
one at a time, two batches of two, or one batch of four); past four, variants with a unique tail
extend the set:

| Prompt | What it exercises |
|---|---|
| code edit | a ~1.2K-token Python file, asked back with one class renamed; most output is copied |
| prose edit | a long operations spec, asked back with one timeout changed; copied prose |
| control | five haiku about a GPU fan; almost no repeated suffix |
| code edit 2 | the same file with two names changed; a second copy-heavy request for N=4 |

The control matters: a width that only helps copied text is not a width that helps a chat. The
score is all completion tokens over all wall time (`completion_tokens / elapsed`), so the slow
control is part of the verdict rather than hidden by averaging it away. Per-prompt rates are kept
in the JSON for diagnosis.

Plain decoding is measured both before and after the sweep (one extra start). The first arm after
a model load is the coldest -- the expert tier is still filling and graph capture has just
happened -- so the JSON keeps both and the verdict uses the faster one. Lookup has to beat the
machine at its best, not a cold start.

Each candidate is one server start with these variables:

```
FREETOKEN_LOOKUP_MAX_DRAFT=<width>  FREETOKEN_LOOKUP_MIN_MATCH=<match>
FREETOKEN_LOOKUP_MIN_CONTEXT=<context>  FREETOKEN_LOOKUP_MIN_ACCEPTANCE=<floor>
```

and `--speculative lookup` in its argv. The baseline arm has neither, so it measures plain greedy
decoding with the same prompt cache state. The environment has to change per arm because
`LookupState` reads it when a request starts; that is also why the model is restarted rather than
the values being flipped in a live server.

The search is coordinate-wise: width first, then minimum match at the winning width, then the
context floor at the winning pair, then the acceptance floor at that shape, then the parallel
ceiling with the winning policy. Repeating a key is cached, so the default arm is measured once.

The width/match/context search runs with `FREETOKEN_LOOKUP_MIN_ACCEPTANCE=0`, so the gate does
not hide the cost of a shape before the floor itself has been chosen. The floor is then measured
at `0.6`, `0.7` and `0.8` at the winning shape, and the default `(4, 3, 256, 0.8)` is measured as
its own candidate. This matters on a mixed suite: a shape can accept 0.75 per draft and still be
much faster, but the default 0.8 gate would turn it off after one window. The JSON's `chosen.env`
therefore carries all four `FREETOKEN_LOOKUP_*` values, not just the first three.

A candidate is kept only when it beats **both** the no-spec baseline and the default arm by
`MIN_GAIN` (3%); a result inside that band keeps the default, and a result that does not beat the
baseline writes a script without `--speculative lookup` (the JSON says
`"chosen": {"speculative": "none"}`). This is deliberate: a 1-2% difference is inside run-to-run
noise, and a calibration that always "wins" is worse than one that can say no.

The same comparison is applied again at the winning parallel ceiling. Lookup drafts up to two
requests by default (`FREETOKEN_LOOKUP_MAX_BATCH`), so the two-way arm measures the multi-request
verify itself; a larger arm measures the fallback cost. That cost used to be a GDN scratch reserve
of `max_running_req` slots: at four running requests it grew the pool from 22 to 26 slots (four
scratch slots where one is enough) and cost 10-12% batch throughput while `/v1/stats` showed zero
verifies. `_linear_spec_reserve` now reserves **no** slot for lookup: its scratch is taken from the
pool's spare snapshots at verify time, and a mixed decode batch releases it, which brings the
batch path back to the no-spec slot count. A two-request verify takes two slots; a larger batch
falls back to regular decoding. The
calibration keeps lookup when it is within `MAX_PARALLEL_LOSS` (2%) of plain batch decoding, so a
neutral result still keeps the single-stream win; only a larger loss sets
`chosen.enabled_at_parallel=false` and `lookup_enabled=false`, and the generated script launches
the parallel winner **without** lookup. `chosen.gain` is the single-stream gain and
`gain_at_parallel` is the number that decided it. The parallel arms are measured with at least
five batches because one batch on this rig ranges from 88 to 103 tok/s.

If the workload is one request at a time, pass `--parallel 1`: the parallel sweep then has only
N=1, the comparison is single-stream against single-stream, and a winning lookup stays in the
script. The default sweep is deliberately batch-heavy because a server started with
`--max-running-requests 4` spends its time in batches.

The parallel sweep sends N requests through a thread pool and times the whole batch; its score is
still completion tokens over wall time. `parallel.gain_vs_single` is the throughput gain of the
winning N over the same policy at N=1, while `lookup_gain_at_chosen` is what lookup still adds at
that N (usually ~0 for N > 1, because a batch does not draft). `--max-running-requests` in the
server argv is set to N
for that arm, and the winning N goes into the launch JSON and the start script. A parallel value
that cannot start (for example more requests than the KV cache admits) is reported and skipped
rather than ending the run. At N > 1 lookup declines the batch and the arm measures plain batch
throughput; that is why the JSON keeps the single-request gain and the at-parallel gain apart.

A run that starts and stops the same model a dozen times is hard on the distributed setup, so the
calibrator adds `--distributed-timeout 180` when the caller did not choose one, gives every arm a
fresh server port (the engine's Gloo store listens on `port + 1`), and retries an arm twice on a
fresh port if a rank dies during startup. It also fails a start as soon as the log shows the
scheduler's Gloo timeout instead of waiting out the full health budget. The timeout is preserved
in the generated JSON and script.

Acceptance and drafted ratio for the winning arm come from `/v1/stats` after its requests. They
are recorded, not used directly to pick the width: the width search runs with the gate off and the
separate floor sweep measures what the gate costs (or saves) at the winning shape. A request that
trips the floor then stops drafting for the rest of its life, and that cost shows in the arm's
time.

## The JSON

```json
{
 "schema": 1,
 "generated_at": "2026-10-23T13:20:00",
 "hardware": {"gpu": ["NVIDIA RTX PRO 4000 Blackwell", "..."], "cpu": "...", "ram_gb": 121.0},
 "model": {"name": "RadixArk/Qwen3.8-Flash-Next", "path": "/.../snapshots/..."},
 "baseline": {"tok_s": 59.6, "tokens": 768, "seconds": 12.9,
              "per_prompt": {"code-edit": 57.1, "prose-edit": 53.8, "control": 62.8,
                             "code-edit-2": 60.1}},
 "baseline_arms": {"start": {"tok_s": 57.2, "...": "..."},
                   "end": {"tok_s": 59.6, "...": "..."}},
 "lookup_enabled": true,
 "chosen": {"speculative": "lookup", "env": {"FREETOKEN_LOOKUP_MAX_DRAFT": "4",
             "FREETOKEN_LOOKUP_MIN_MATCH": "3", "FREETOKEN_LOOKUP_MIN_CONTEXT": "256",
             "FREETOKEN_LOOKUP_MIN_ACCEPTANCE": "0.7"},
             "tok_s": 84.0, "gain": 0.409, "acceptance": 0.864, "drafted_ratio": 0.577,
             "key": {"max_draft": 4, "min_match": 3, "min_context": 256, "min_acceptance": 0.7},
             "parallel": 4, "enabled_at_parallel": true,
             "tok_s_at_parallel": 132.0, "gain_at_parallel": 0.05,
             "single_stream_gain": 0.409},
 "parallel": {"default": 4, "chosen": 4,
              "sweep": {"1": {"tok_s": 84.0}, "2": {"tok_s": 110.0}, "4": {"tok_s": 132.0}},
              "baseline_at_chosen": {"tok_s": 129.4},
              "gain_vs_single": 0.571, "lookup_gain_at_chosen": 0.05},
 "user_params": {"max_running_requests": 4, "kv_cache_dtype": "fp8",
                 "num_tokens": "131072", "memory_ratio": "0.88"},
 "measurements": {"k=4 match=3 context=256 a=0.7": {"tok_s": 84.0, "...": "..."}},
 "launch": {"python": "/.../.venv/bin/python", "argv": ["-m", "freetoken", "..."],
            "cwd": "...", "env": {"PYTHONPATH": "..."}, "port": 1920,
            "model_name": "RadixArk/Qwen3.8-Flash-Next"},
 "elapsed_s": 1234.5,
 "tag": "radixark-qwen3.8-flash-next"
}
```

`launch.argv` already carries the chosen `--max-running-requests`; it has no `--speculative` (the
script adds that when `lookup_enabled` is true), and its port is the one measured on. `user_params`
records the KV-cache and memory choices that were given rather than tuned, so a later run can tell
what the numbers belong to. `chosen.env` is what the script adds when lookup is enabled. To keep
the numbers comparable, the JSON keeps the port that was measured on and the exact argv; the start
script normalizes `--port` again, so a launcher can rewrite it without touching `argv`.

## Using an existing launch config

If a profile already has a full argv, pass it instead of rebuilding one:

```json
{
 "python": "/home/suzuri/projects/llm-launcher/module/fork/freetoken/.venv/bin/python",
 "argv": ["-m", "freetoken", "--model-path", "...", "--served-model-name", "...",
          "--tp-size", "2", "--max-running-requests", "4", "--memory-ratio", "0.88",
          "--num-tokens", "131072", "--moe-strategy", "offload", "--moe-cache-auto",
          "--moe-cpu-layers", "0", "--host", "127.0.0.1", "--kv-cache-dtype", "fp8",
          "--prefill-decode-interval", "4"],
 "cwd": "/home/suzuri/projects/llm-launcher/module/fork/freetoken",
 "env": {},
 "port": 1920,
 "model_name": "RadixArk/Qwen3.8-Flash-Next"
}
```

```bash
.venv/bin/ft calibrate --config launch.json --out-dir ~/calibration/qwen38-lookup
```

`--port` overrides the config's port. Any `--speculative` already in `argv`, and any
`FREETOKEN_LOOKUP_*` already in its `env`, is removed: the baseline arm must be plain decoding, and
every candidate sets its own three values.

## Notes

- Up to `FREETOKEN_LOOKUP_MAX_BATCH` (default 2) greedy requests draft together; a larger
  batch falls back to plain decoding. The four-way arm therefore measures pool parity for the
  fallback, while the two-way arm measures the multi-request verify (see
  `experiments/freetoken-lookup/multi-request-ab.md`).
- `--parallel` is a ceiling, not a promise: requests still queue when more than N arrive. Measure
  the counts your workload actually reaches; 1/2/4 is the default because FreeToken's default
  `--max-running-requests` is 4.
- The width is a machine property, not a model property, but the model and its quantization decide
  expert placement and therefore the verify cost: keep one JSON per (machine, model).
- The tool writes no launcher profile. Point the profile's env at `chosen.env` and add
  `--speculative lookup`, or run the generated script directly.
- Logs for every arm are under `<out-dir>/logs/`; if an arm fails, its last lines are in the error.
