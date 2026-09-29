# Windows (native, no WSL)

The engine builds and runs natively on Windows: the C++ extensions compile with MSVC,
Triton comes from `triton-windows`, the CUDA kernels JIT with nvcc + MSVC, and
`uv pip install -e ".[accel]"` resolves on both platforms because the two native
accelerator packages in `accel` are Linux-only wheels.

## Requirements

- Windows 10/11 x64 with an NVIDIA GPU (RTX 30/40/50) and driver r580+ (CUDA 13)
- CUDA 13.0 toolkit, found through `CUDA_PATH`, with `nvcc` matching torch's CUDA major
- Visual Studio 2022 Build Tools with the *Desktop development with C++* workload
  (MSVC toolset + Windows SDK). No developer shell is needed: `setup.py` finds the toolset
  through `vswhere` and activates it for the install-time extensions, and tvm-ffi activates
  the dev prompt for its JIT builds.
- The *C++ Clang Compiler for Windows* component of that same install
  (`VC\Tools\Llvm\x64\bin\clang-cl.exe`). Only the `_cpu_moe` extension needs it; without it
  that one extension is skipped and `--moe-cpu-layers auto` becomes unavailable.
  `setup.py` finds it through `vswhere`, or through `FREETOKEN_CLANG_CL=<path>`.
- [uv](https://docs.astral.sh/uv/), Python 3.12 (what `install.sh` and the release wheels use)

## Install from source

```powershell
git clone https://github.com/FlashML-org/FreeToken.git
cd FreeToken
uv venv --python 3.12
.venv\Scripts\Activate.ps1
uv pip install -e ".[accel]" --index-strategy unsafe-best-match `
  --extra-index-url https://download.pytorch.org/whl/cu130
```

`--extra-index-url` pins torch to the cu130 build, the same pin `install.sh` passes on
Linux; PyPI's `torch 2.11` is that identical cu130 build, so the flag can be dropped if you
are happy with whatever PyPI serves.

Verify:

```powershell
ft --version
python -c "import torch, triton; print(torch.__version__, torch.version.cuda, triton.__version__)"
python -c "from freetoken.kernel import _pinned_tensor, _radix_tree; print('extensions OK')"
```

Expected on this path: `torch 2.11.0+cu130 ... 13.0`, `triton 3.6.0` (the version string is
the upstream one; `triton-windows` installs as `triton`), and both extensions import.

## What `[accel]` does and does not do here

`sglang-kernel` and `flashinfer-python` publish `manylinux` wheels only, so on Windows the
`fi`/`sgl` extras are marker-disabled and `[accel]` installs nothing extra. Nothing is lost
for the models whose hot path is Triton: every call site probes
`is_flashinfer_installed()` / `is_sgl_kernel_installed()` in `freetoken.kernel.backend` and
falls back to `freetoken.kernel.triton`, and the type x backend matrix in `engine.engine`
only offers `fi`/`fa`/`trtllm` when those packages exist.

Qwen3.8-Flash-Next (`model_type: qwen4_exp`) is exactly such a model: its 12 sparse
attention layers resolve to `qsa_sparse`, an in-tree Triton kernel that requires neither
package, and its NVFP4 experts pick `TritonNvfp4MoEKernel` (the `b12x` flashinfer kernel is
never auto-selected, and `marlin` needs the separate `vllm` wheel). The GDN layers,
`causal_conv1d`, norms, RoPE, activations and sampling are Triton on both platforms.

## What is different on Windows in the engine itself

`freetoken-kernel-cache` ships Linux `.so` files, so on Windows the kernels under
`python/freetoken/kernel/csrc` are JIT-compiled with nvcc + MSVC on first use, and all five
of them (`radix`, `batch_memcpy`, `store`, `index`, `fast_index_copy`) build and load. The
branches below carry that; none of them changes anything on Linux:

- `utils.mp.zmq_endpoint`: the Windows `libzmq` wheels are built without the `ipc://` transport
  (binding one answers `Protocol not supported` for a `/tmp` path and for a native one alike), so
  the five control queues take `tcp://127.0.0.1` with ports allocated by the process that builds
  the config and carried to every worker with it; POSIX keeps its `/tmp` unix sockets.
- The frontend also needs `tornado`, which pyproject requires on Windows only. `zmq.asyncio`
  waits through `loop.add_reader`, and the proactor loop Windows' asyncio picks by default has no
  such method; pyzmq's own fallback registers the reader on a selector thread, but only when
  tornado can be imported. Without it the engine prefills a request and the frontend never reads
  the reply -- the client hangs, and one `RuntimeError` in the server log is the only trace. A
  `WindowsSelectorEventLoopPolicy` is the other way in, but the swap has to reach the loop uvicorn
  builds on its own thread, which it did not do here, so the dependency is the seam that holds.
- `kernel.utils._msvc_ninja_compat`: tvm-ffi (<= 0.1.14) writes its Windows CUDA rule as
  `-Xcompiler /std:c++17 /O2`. nvcc takes one comma-joined `-Xcompiler` argument, so that
  `/O2` reaches it as a second input file, and the forced `/std:c++17` makes MSVC's STL hide
  `std::source_location` and `std::integral` -- which is what shattered the kernels, not any
  CUDA code. `build.ninja` is repaired before ninja runs it. The same branch maps GCC's
  `__always_inline` (no MSVC spelling) to CUDA's `__forceinline__` and links `cudart.lib`,
  which tvm-ffi's Windows link line omits and whose `/LIBPATH` needs quotes for `Program Files`.
- `jit/store.cu`, `jit/index.cu`, `jit/fast_index_copy.cuh`, `jit/batch_memcpy.cuh`:
  `with_dtype<int32_t, int64_t>(x)` is not parseable by nvcc's front-end in a dependent
  context. It never meant what it looks like: the arg-taking overload discards its template
  pack and only rebinds the ref, so the ignored arguments are dropped here. The
  `with_device<Codes>(x)` and the zero-argument `with_device<Codes...>()` forms are kept.
- `setup.py` builds `_cpu_moe` (the CPU MoE executor behind `--moe-cpu-layers`) with clang-cl,
  the component VS ships beside the toolset. Its SIMD tiers are per-function
  `__attribute__((target))` with `__builtin_cpu_supports` selecting between them; cl has neither
  spelling and would build the scalar executor while the viability probe still reported it usable.
  torch's Windows compile rule hardcodes `cl /showIncludes`, so the generated `build.ninja` is
  retargeted (`_retarget_ninja`), and the msvc dep scan plus MSVC's `/GL` whole-program codegen
  leave with the compiler -- `deps = msvc` would be handed the other front-end's output and
  `link.exe` would LTCG over clang objects. Only the compiler moves: `link.exe`, `/MD` and the
  CRT stay MSVC's, and `clang_rt.builtins-x86_64.lib` comes in as an explicit library because
  `__builtin_cpu_supports` lowers into it. That extension also builds at `/std:c++20`: MSVC's
  `<atomic>` reaches the Interlocked intrinsics through the `winnt.h` prototypes that header
  aliases onto those names, and clang calls the C++17 instantiation ambiguous
  (`call to '_InterlockedAnd' is ambiguous`) where cl merges the two.
  Thread pinning follows through `pin_thread_to_cpu`, whose Win32 branch is
  `SetThreadAffinityMask`; a CPU id outside the thread's 64-bit group mask is left to the
  scheduler.
  `_ple_store` (the default `--ple-backend disk`) also builds here: its three seams (`TableFile`,
  the `cumemop_*` driver lookup, the aligned allocation) carry Win32 bodies, and the `pread`
  thread pool -- not io_uring -- is the reader, which is also the portable shape on Linux.
- `kernel.gguf._host_compiler`: on POSIX it points nvcc's `-ccbin` (and `CXX`/`CC`, which it
  writes into the environment) at clang++ or an older gcc, because a too-new gcc trips torch's
  headers. Here that search matched the `clang++.EXE` inside VS -- a developer prompt has it on
  `PATH` -- and nvcc then handed MSVC's `-nologo`/`/EHsc` spellings to a front-end that rejects
  them; because the choice lands in `os.environ`, one GGUF build poisoned every later JIT in the
  same process (the tvm-ffi kernels failed next, not the GGUF one alone). Windows keeps nvcc's
  default host.
- `moe/cpu_executor.physical_core_cpus` has no sysfs to read here, so it returns every logical
  CPU rather than one per physical core, and the pool is as wide as `os.cpu_count()`. The ids it
  returns do reach the workers: `pin_thread_to_cpu` binds each to one logical CPU through
  `SetThreadAffinityMask`. Narrow the pool with `--moe-cpu-threads` if SMT siblings turn out to
  contend for bandwidth on a given machine.
- `moe/host_banks.py` owns the platform seam for direct I/O (`open_direct`, `pread_into`,
  `close_direct`, `drop_read_cache`): `os.O_DIRECT`/`os.preadv`/`os.posix_fadvise` on POSIX,
  `FILE_FLAG_NO_BUFFERING` + `ReadFile` at an explicit offset on Windows. `models/weight.py`'s
  parallel expert reader -- the one qwen3_5_moe, qwen3_moe, qwen3_vl and the NVFP4 bank loader
  share -- goes through it too, so it keeps bypassing the page cache here instead of degrading.
  `checkpoint/ftw.py` needs nothing: it already probes `getattr(os, "O_DIRECT", 0)` and takes
  its mmap fallback when the flag is absent -- that fallback itself needed one fix, since
  `mmap.mmap(fd, 0, prot=mmap.PROT_READ)` has no Windows spelling (`mmap.PROT_READ` doesn't
  exist); `access=mmap.ACCESS_READ` is equivalent on POSIX.

## Building the extensions again after editing them

`setup.py build_ext --inplace` works from a plain PowerShell as well as
`uv pip install -e ".[accel]"` does: when the shell has no compiler environment, `setup.py`
runs `vcvars64.bat` from the VS install `vswhere` reports and takes its environment over
(including the `ninja` that VS ships under `CommonExtensions\Microsoft\CMake`). Doing it by
hand stays available and is what to do if you want the prompt for your own reasons:

```powershell
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$env:DISTUTILS_USE_SDK = "1"   # without it torch's ABI check refuses the activated env
.venv\Scripts\python.exe setup.py build_ext --inplace
```

`_cpu_moe` needs ninja because the clang-cl swap works by editing torch's generated ninja
compile rule; where ninja is missing, `setup.py` drops the extension rather than quietly
building the scalar executor with cl.

## The limit that actually binds: page-locked host memory

`--moe-strategy offload` keeps every expert in host RAM and page-locks it (`cudaHostRegister`,
pin-after-fill), because `PageableMemoryAccess=0` on this class of GPU -- the gather kernel
cannot read pageable banks. Windows caps that locked total well below the Linux one. Measured
here (126.9 GiB RAM), pinning through `HostBank` itself in 1 GiB steps:

| | |
|---|---|
| last total the driver accepted | **62 GiB** (63 GiB -> `cudaErrorMemoryAllocation`) |
| same through `cudaHostAlloc` instead of `mmap`+register | 62 GiB too, so the backing is irrelevant |
| Qwen3.8-Flash-Next NVFP4 expert banks | **63.5 GiB** |
| PLE n-gram table when `--ple-backend pinned` | **47.7 GiB more** |
| PLE with the default `disk` backend | 0 (it pins ~22 MiB of staging; the 512 MiB row LRU is ordinary RAM) |

`_pin_budget_bytes` therefore caps Windows too (45% of RAM -> 57.1 GiB here, beside the 40%
it already used for WSL), which turns the old mid-load crash plus "free host RAM" guess into a
preflight that names both numbers:

```
expert banks need 63.5 GiB of pinned host RAM but the pin budget is 57.1 GiB
(WDDM caps CUDA pinning; FREETOKEN_PIN_BUDGET_GB overrides); pass --moe-cpu-layers auto ...
```

So the disk backend removes the PLE's 47.7 GiB entirely, and the expert banks alone still sit
about 1.5 GiB over what this machine can lock. The last mile is `--moe-cpu-layers auto`, which
settles head+tail layers as LOCKED -- page-resident in ordinary RAM, no CUDA pin quota spent --
and decodes them on the CPU executor. It resolves to five of the 48 MoE layers here:

```
--moe-cpu-layers auto: banks 63.46 GiB > pin budget 57.09 GiB; locking 5 head+tail MoE layers
for CPU decode ([0, 1, 2, 46, 47])
```

`HostBank.lock()` had no Windows spelling until now: `_os_lock` opened with `import resource`,
`lock()` caught that ImportError next to the OSError it expected, and every LOCKED layer settled
PAGEABLE under an `ulimit -l` message. The Win32 branch raises the process working-set maximum --
a raise to 96 GiB succeeds -- and calls `VirtualLock`. That is not the ceiling that binds: page
locks have their own per-process quota, lifted by the "Lock pages in memory" right, which an
Administrator token here does not hold (`AdjustTokenPrivileges` answers
`ERROR_NOT_ALL_ASSIGNED`). Measured: a 16 MiB bank locks, a 5.3 GiB one comes back
`ERROR_QUOTA_EXCEEDED` (1453), and the running server says so and carries on:

```
WARNING bank lock failed; leaving this and later banks pageable: [Errno 1453] VirtualLock(0.8 GiB):
WinError 1453 (ERROR_QUOTA_EXCEEDED unless the account holds the 'Lock pages in memory' right
(secpol.msc, then re-login); without it --moe-cpu-layers layers stay pageable)
```

The existing downgrade then takes over -- the echoed residency reports PAGEABLE and the CPU
executor reads those layers from pageable RAM exactly as it does on a WSL2 host whose
`RLIMIT_MEMLOCK` refused the lock. To get them genuinely resident, grant the right (secpol.msc ->
Local Policies -> User Rights Assignment -> Lock pages in memory) and start a new login session.

- Tensor parallelism (`shm_ar`/`p2p_ar`) is untested on Windows; keep `--tp-size 1`. A
  `--tp-size 2` run of a model with `FULL` attention layers is refused at config time anyway,
  because the two TP-capable FULL backends (`fi`, `fa`) are the packages gated to Linux.
- The vendored GGUF dequant kernels (`csrc/gguf/gguf_kernel.cu`, the q4_0 path) do not compile
  here: nvcc hosts on cl, and `torch/csrc/dynamo/compiled_autograd.h` -- reached through the
  `pybind11` block at the end of that file -- trips `error C2872: 'std': ambiguous symbol`. Moving
  that include up next to `torch/all.h` changes nothing (measured), so the binding needs its own
  host-only translation unit before a GGUF checkpoint can serve. `_cpu_moe`'s own q4_0 GEMV builds
  and runs; the GPU reference kernel behind
  `tests/moe/test_cpu_moe_q4_0.py::test_cpu_decode_q4_0_matches_ggml_mmvq` is what is missing.

## Verified on Windows (RTX PRO 6000 Blackwell, sm_120, driver 616.92, CUDA 13.0)

- `uv pip install -e ".[accel]"` completes; `torch 2.11.0+cu130`, `triton 3.6.0`
  (`triton-windows 3.6.0.post26`), and all four extensions import: `_pinned_tensor`,
  `_ple_store`, `_radix_tree` from cl, `_cpu_moe` from clang-cl. The clang-cl build really is
  compiling the SIMD tiers -- `dumpbin /DISASM` over its object file counts 305 uses of `zmm`
  registers and 7 `vpdpbusd`/`vdpbf16ps`, and the running executor reports
  `isa=avx512bf16+avx512vnni(nvfp4-w4a8)`, which is `__builtin_cpu_supports` answering through
  the compiler-rt library the link pulls in.
- `uv pip compile pyproject.toml --extra accel --python-platform linux` still resolves
  `flashinfer-python`, `sglang-kernel` and `triton==3.6.0`: the Linux set is unchanged.
- The five tvm-ffi kernels compile and load; Triton paths run (`tests/kernels/test_rotary.py`).
- `pytest tests/models/qwen4_exp/test_ple_disk.py`: 10/10. The row store reads bitwise-identical
  rows over `FILE_FLAG_NO_BUFFERING`, the 512 MiB row LRU and its eviction work, and the C++
  deferred fill (`wait-sync, cpp fill`) is the path taken. Its gate-mode half now builds the
  second table in its own directory: Windows refuses to rebuild a checkpoint in place while a
  reader still holds it open (POSIX allows that), and the fixture is seeded, so both directories
  hold identical bytes.
- `tests/models/qwen4_exp` is green, and a synthetic two-shard checkpoint read back through
  `iter_expert_tensors_parallel` returns the expert tensors byte-identical -- that reader is what
  qwen3_5_moe, qwen3_moe, qwen3_vl and the NVFP4 bank loader share.
- `pytest tests/engine tests/models/qwen4_exp tests/kernels tests/moe tests/checkpoint tests/layers`:
  905 passed, 67 skipped, 15 failed (from 878 / 67 / 41 before this section's `_cpu_moe`,
  `VirtualLock` and event-loop work). The 15 are fixtures naming the `fi` backend (10),
  `torch._scaled_mm` rowwise scaling unsupported in this torch build (4), and the GGUF kernel
  compile named above -- each needs a component Windows does not ship, and none is a Linux
  regression.
- `_pin_budget_bytes` gives 57.1 GiB here (45% of 126.9 GiB), so a 63.5 GiB bank set is refused
  before any disk read; `FREETOKEN_PIN_BUDGET_GB=64` lets the preflight pass when your machine's
  real ceiling is above that conservative default.
- `ft serve --model nvidia/Qwen3.8-Flash-Next-NVFP4 --dummy-weight --ple-backend disk
  --moe-strategy offload --moe-cpu-layers auto --attention-backend qsa_sparse
  --quant-backend moe.nvfp4=triton --max-seq-len-override 4096` loads: it resolves
  `attention_backend=qsa_sparse`, `page_size=64`, `moe_strategy=offload`, logs
  `MoE experts: nvfp4 via triton`, splits the residency, sizes the slot cache off the pinned
  layers only (`--moe-cache-auto resolved moe_cache_size=22528`), builds the CPU executor
  (`CPU MoE executor ready: threads=15 ... isa=avx512bf16+avx512vnni(nvfp4-w4a8) fmt=nvfp4`),
  warms prefill, captures the four decode graphs, logs
  `API server is ready to serve on 127.0.0.1:1919`, and answers a chat completion:

  ```
  latency 2.8 s
  finish: length
  usage: {'prompt_tokens': 53, 'completion_tokens': 7, 'total_tokens': 60}
  ```

  What comes back is empty text with seven tokens accounted for, which is what a `--dummy-weight`
  engine produces; the claim here is the round trip -- frontend, tokenizer, scheduler, Triton
  prefill, CUDA-graph decode across both the pinned and the CPU-served expert layers, and the
  reply returning over the TCP queues. Before the event-loop seam above, the same request prefilled
  and then hung forever.


