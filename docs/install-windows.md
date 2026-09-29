# Windows (native, no WSL)

The engine builds and runs natively on Windows: the C++ extensions compile with MSVC,
Triton comes from `triton-windows`, the CUDA kernels JIT with nvcc + MSVC, and
`uv pip install -e ".[accel]"` resolves on both platforms because the two native
accelerator packages in `accel` are Linux-only wheels.

## Requirements

- Windows 10/11 x64 with an NVIDIA GPU (RTX 30/40/50) and driver r580+ (CUDA 13)
- CUDA 13.0 toolkit, found through `CUDA_PATH`, with `nvcc` matching torch's CUDA major
- Visual Studio 2022 Build Tools with the *Desktop development with C++* workload
  (MSVC toolset + Windows SDK). No developer shell is needed: `setuptools` locates MSVC
  for the install-time extensions, and tvm-ffi activates the dev prompt for its JIT builds.
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
of them (`radix`, `batch_memcpy`, `store`, `index`, `fast_index_copy`) build and load. Four
platform branches carry that, none of which changes anything on Linux:

- `utils.mp.zmq_endpoint`: the Windows `libzmq` wheels are built without the `ipc://`
  transport (binding one fails with `Protocol not supported`), so the five control queues
  take `tcp://127.0.0.1` with ports allocated by the process that builds the config and
  carried to every worker with it; POSIX keeps its `/tmp` unix sockets.
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
- `setup.py` skips `_cpu_moe` on Windows: its SIMD dispatch is GCC/Clang-only
  (`__builtin_cpu_supports`, `__attribute__((target))`), and MSVC would build a scalar
  executor that the viability probe still reports usable. The engine notices the missing
  module and stays on the GPU executor. `_ple_store` (the default `--ple-backend disk`)
  does build here: its three seams (`TableFile`, the `cumemop_*` driver lookup, the
  aligned allocation) carry Win32 bodies, and the `pread` thread pool -- not io_uring --
  is the reader, which is also the portable shape on Linux.
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

`uv pip install -e ".[accel]"` rebuilds from scratch in an isolated environment and needs
nothing special. The documented dev loop (`python setup.py build_ext --inplace`) runs in your
shell, so start from a *Developer Prompt* and let distutils trust it:

```powershell
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$env:DISTUTILS_USE_SDK = "1"   # without it torch's ABI check refuses the activated env
uv pip install ninja           # optional: otherwise distutils compiles instead of ninja
.venv\Scripts\python.exe setup.py build_ext --inplace
```

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
about 1.5 GiB over what this machine can lock. The last mile is `--moe-cpu-layers` (mlock those
layers instead of pinning them), which needs the `_cpu_moe` module Windows does not build.
`fused` is refused for NVFP4 experts, so it is not a way around the cap.

- Tensor parallelism (`shm_ar`/`p2p_ar`) is untested on Windows; keep `--tp-size 1`. A
  `--tp-size 2` run of a model with `FULL` attention layers is refused at config time anyway,
  because the two TP-capable FULL backends (`fi`, `fa`) are the packages gated to Linux.

## Verified on Windows (RTX PRO 6000 Blackwell, sm_120, driver 616.92, CUDA 13.0)

- `uv pip install -e ".[accel]"` completes; `torch 2.11.0+cu130`, `triton 3.6.0`
  (`triton-windows 3.6.0.post26`), `_pinned_tensor`, `_ple_store` and `_radix_tree` import.
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
  878 passed, 67 skipped, 41 failed. Those 41 are only the CPU MoE executor (`_cpu_moe`, 27),
  fixtures naming the `fi` backend (10) and `torch._scaled_mm` rowwise scaling unsupported in
  this torch build (4) -- each needs a component Windows does not ship, and none is a Linux
  regression.
- `_pin_budget_bytes` gives 57.1 GiB here (45% of 126.9 GiB), so a 63.5 GiB bank set is refused
  before any disk read; `FREETOKEN_PIN_BUDGET_GB=64` lets the preflight pass when your machine's
  real ceiling is above that conservative default.
- `ft serve --model nvidia/Qwen3.8-Flash-Next-NVFP4 --dummy-weight` resolves
  `attention_backend=qsa_sparse`, `page_size=64`, `moe_strategy=offload`, logs
  `MoE experts: nvfp4 via triton`, and now stops at the pin-budget preflight quoted above
  instead of crashing mid-load with the old "free host RAM" guess.


