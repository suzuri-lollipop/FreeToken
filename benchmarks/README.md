# benchmarks

Run from the repo root with `PYTHONPATH=python:.`, pinned to one GPU
(`CUDA_VISIBLE_DEVICES=0`). Each script's `--help` / docstring has the details.

**`bench_decode_moe.py`** — bs=1 decode tok/s of a served MoE model. Spawns `ft serve`
per backend and times token arrivals over streamed `/v1/chat/completions`, so numbers
include the full serving path. AIME-25 prompt, checkpoint-recommended sampling.

```bash
python benchmarks/bench_decode_moe.py --model /path/to/model --backend offload,cpu,hybrid
```

**`bench_token_speed.py`** — token generation speed of a server that is already running: TTFT,
per-stream decode tok/s and aggregate tok/s over streamed requests. Needs no GPU on the client
side. Attaches to `--server` (default `http://127.0.0.1:1919`), or spawns and stops its own
`ft serve` with `--model`. Requests run in waves of `--concurrency` released by a barrier, or are
swept with `--concurrency-sweep 1,2,4,8`; `/v1/stats` is polled while measuring, so traffic from
another client shows up as `active` above your own concurrency. Checkpoint-recommended sampling,
filler prompt rather than AIME, so routing is generic prose, not a reasoning workload.

```bash
python benchmarks/bench_token_speed.py                                # bs=1 against the local server
python benchmarks/bench_token_speed.py --concurrency-sweep 1,2,4,8 --json speed.jsonl
```

**`bench_load_weight_generic.py`** — expert-bank load time: serial vs parallel O_DIRECT
vs pre-repacked FTW, each mode in its own subprocess. Linux-only; stages the FTW under
`/var/tmp` (`--ftw-dir` overrides; roughly checkpoint-sized).

```bash
python benchmarks/bench_load_weight_generic.py --model /path/to/model
```

**`bench_offload_cache_copy.py`** — synthetic (no checkpoint): per-layer decode expert
copy cost (`ensure_experts` + `copy_missing`), swept over bank layout x cache slots x
batch size x miss rate.

```bash
python benchmarks/bench_offload_cache_copy.py
```

For host RAM vs PCIe bandwidth and the offload/hybrid backend pick, use `ft bench bw`
instead — it writes the JSON profile the engine reads.
