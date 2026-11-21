# Layer-split (pipeline) parallelism for Hy4-flash -- plan

Goal: match Maya's 22.8 tok/s decode and keep per-stream speed under concurrent
requests ("parallel inference that does not slow down"). Maya proves the approach:
layer split across GPUs (GPU0 layers 0-19, GPU1 20-41), NOT tensor parallel.

## Why layer split beats TP here (measured 2026-11-21)

FreeToken NVFP4 TP=2 on DL380 Gen9 (gen3 x8, 5.6 GB/s): offload single ~15-16 tok/s,
conc-4 aggregate ~20 (per-stream ~5.1). Two TP costs dominate under load:
1. **Per-layer all-reduce**: 96 AR/step; under PCIe contention the NCCL RING_LL AR
   inflates to ~9-12 ms/step (`_scratch/parallel-perf/RESULTS.md`). Layer split has
   ZERO per-layer AR -- only one activation hand-off per stage boundary.
2. **Expert cache spread thin**: TP shards every layer's experts across both GPUs,
   so each GPU caches half-experts of all 42 layers. Layer split gives each GPU the
   FULL experts of its 21 layers -> concentrated cache, higher hit, less PCIe fetch.

The stage hand-off tensor is the iHC stream `[T, hc_mult=4, dim=3072]` bf16 = 24 KB
at decode bs=1 (~4 us over gen3 x8) -- negligible vs the ~9-12 ms AR it replaces.

## Design (decode-first, no microbatching)

For single-token decode the pipeline is just two sequential stages; no microbatch
schedule is needed. Prefill can chunk through the same stages.

- **PP info**: extend `distributed/info.py` with `pp_rank/pp_size` and a per-rank
  `layer_range` (e.g. rank0 [0,21), rank1 [21,42)). `tp_size=1, pp_size=2`.
- **Model build**: `Transformer` builds only its local layer range; rank0 owns
  `embed_tokens`, the last rank owns `hc_head/norm/lm_head`. Weights load only the
  local layers (the loader already filters by layer id; MoE experts load FULL per
  local layer, no TP shard).
- **Hand-off**: rank0 forward = embed + local blocks -> `send(streams, engram_mask)`
  to rank1; rank1 = `recv` -> local blocks + hc_head + norm + head -> logits. Use
  `torch.distributed.send/recv` (NCCL, graph-capturable) on a dedicated PP group.
- **KV cache**: each rank's pool holds only its local layers (the pool already keys
  by layer id; size `num_layers` to the local count). The page table / radix cache
  stay scheduler-global (shared token addressing).
- **MoE offload**: each rank caches its local layers' full experts; the slot-cache
  budget doubles per layer (no TP halving) -> higher hit.
- **Scheduler**: dispatch the batch to rank0; rank1 waits on recv. Logits/sample on
  the last rank; broadcast sampled tokens back for the next step's input. The
  existing TP broadcast paths become PP send/recv.
- **CUDA graphs**: capture each rank's stage forward (incl. the NCCL send/recv) per
  batch size; replay restages the hand-off buffers.

## Phases

1. PP info + layer-range partition + model build/load for a local range (no comm yet,
   single-rank smoke: rank0-only forward of layers 0-20 returns partial streams).
2. PP send/recv hand-off + last-rank head; greedy decode parity vs TP on a short
   prompt (correctness gate).
3. KV pool / MoE offload per-stage sizing; scheduler dispatch + token broadcast.
4. CUDA graph capture per stage; throughput bench single + conc-4 vs TP offload.
5. Prefill chunking through stages; engram fill per stage; acceptance/quality check.

## Risks / open questions

- The engine, graph runner, scheduler, and MoE offload assume TP throughout; PP
  touches all of them. This is a multi-session engine feature.
- iHC streams (4 residual lanes) must cross the boundary intact; engram layers 1/11
  (rank0) and 21/31 (rank1) need the engram_mask + host fill on the owning rank.
- The DFly draft (5 layers reusing target KV at layers 25-41) lives entirely on
  rank1 under a [0,21)/[21,42) split -- spec decode composes naturally later.
- Hardware alternative: an x16 riser (719076-B21 primary / 719073-B21 secondary)
  ~2x PCIe bandwidth, orthogonal to this software work.
