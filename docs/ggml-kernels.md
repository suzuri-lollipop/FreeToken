# Maintaining the ggml kernels

FreeToken runs GGUF weights on kernels copied from [llama.cpp](https://github.com/ggml-org/llama.cpp), and we edit the copy in place. The first line of each `VERSION` file names the llama.cpp commit the copy last merged: the pin.

## What is copied

| Here, under `python/freetoken/` | In llama.cpp | What it is |
|---|---|---|
| `kernel/csrc/ggml/ggml-cuda/` | `ggml/src/ggml-cuda/` | GPU kernels, built by `kernel/ggml.py` |
| `kernel/csrc/ggml/ggml-cpu/` | `ggml/src/ggml-cpu/` | CPU MoE row kernels, built by `setup.py` |
| `kernel/csrc/ggml/src/` | `ggml/src/` | shared headers and `ggml-quants.c` |
| `kernel/csrc/ggml/include/` | `ggml/include/` | shared headers |
| `kernel/csrc/ggml-vk/` | `ggml/src/` + `ggml/src/ggml-vulkan/` | self-contained ggml core + Vulkan backend (own pin), built by `kernel/ggml_vk.py` |
| `third_party/gguf/` | `gguf-py/gguf/` | GGUF file reader |

Only the files FreeToken compiles or imports are here. `kernel/csrc/ggml/freetoken/` is our own code: the GPU ops (`ggml_ops.cu`), the host symbols the kernels need (`ggml_shim.cu`), and the CPU entry points. `kernel/csrc/ggml-vk/freetoken/ggml_vk_ops.cpp` is the Vulkan op wrapper: it runs one-op ggml graphs on the Vulkan backend and keeps the packed weights in resident device buffers.

## Vulkan backend

`FREETOKEN_GGML_BACKEND=vulkan` routes the packed-weight ops (`mul_mat`, `mul_mat_id`, `mul_mat_id_glu`, `dequantize`, `get_rows`, `moe_weighted_sum`) through `ggml-vk` instead of `ggml-cuda`; the op surface and dtype contracts are the same, so GGUF models serve unchanged. Activations stage through the host as fp32 (the NVIDIA Vulkan driver exposes no bf16 storage), and the internal q8_1 activation quantization rounds its fp16 scales differently from CUDA, so parity is at quantization-noise level, not bit-exact. The rest of the engine (attention, convs, activations) still runs on its CUDA/Triton kernels.

`mul_mat_id` / `mul_mat_id_glu` never feed the vendored `mul_mat_id` shader: on the NVIDIA Blackwell driver it miscomputes leading tokens (token 0 for tiny batches, the first six tokens from T=9) and `-1` routes corrupt their neighbors. Instead the wrapper groups the routes by expert and runs one `ggml_mul_mat` per expert in a single graph. Up to 8 tokens per expert (the DMMV `mul_mat_vec_max_cols` bound) ride the vec path, which dot-products fp32 activations against in-shader dequantized weights; larger groups dequantize the routed rows with ggml-quants on the host (parallel over rows) and run an fp32 GEMM, because the MMQ path's internal q8_1 activation rounding differs from CUDA's and the looped decoder amplifies that noise into incoherent text. The gated activation runs on the host. With this split, greedy output is token-identical to the CUDA backend. Only the routed rows cross the host per call and nothing is weight-cached, because an offload slot pool's contents change under the same pointer.

Serving notes: decode CUDA graphs are disabled (the host staging is not capturable), `memory_ratio` is clamped to 0.8 so the KV pool leaves the Vulkan backend device memory, and a raw GGUF file skips the HF banks sidecar. Decode runs at a few tokens/s on an RTX PRO 4000 (offload, fp32 staging); resident Vulkan weight buffers and a capture-safe design are the next step.

Build dependencies: `glslc` (shaderc), Vulkan headers + SPIRV-Headers, and `libvulkan`. Discovery: `FREETOKEN_GLSLC_EXECUTABLE` / `glslc` on PATH / `~/.local/vulkan-deps/prefix/bin/glslc`; headers from `FREETOKEN_VULKAN_INCLUDE` / the system / `~/.local/vulkan-deps/Vulkan-Headers-*/include`. Shaders compile on first use into `~/.cache/freetoken/ggml-vk` (override: `FREETOKEN_GGML_VK_BUILD_DIR`); the C/C++ module then builds through tvm-ffi like the other kernel modules. shaderc 2025.2 is the verified glslc; 2026.4 miscompiles the coopmat per-element overloads.

## Changing a kernel

- Edit the copied file directly, but do not rename, move or split it. An update matches files by path.
- Put new kernels in new files under `kernel/csrc/ggml/freetoken/`, so updates conflict less. Add a new `.cu` to `sources` in `kernel/ggml.py` and a new CPU source to `_ggml_cpu_targets` in `setup.py`.
- Add a `// FreeToken:` comment when the code does not show why it differs from llama.cpp, so a later conflict explains itself.
- The same sources build for ROCm with `-DGGML_USE_HIP`, so use only CUDA features that HIP also has.
- Cover behavior changes in `tests/kernels/test_ggml.py`.

## Building a change

- GPU kernels compile on first use, for this machine's GPU only.
  - An installed `freetoken-kernel-cache` wins over your edit. Set `FREETOKEN_DISABLE_KERNEL_CACHE=1` to run the edit.
  - On ROCm, header edits do not trigger a rebuild. After editing a `.cuh`, delete `~/.cache/tvm-ffi/freetoken__ggml_*`.
  - Before a PR, compile for every architecture the wheel ships with `FREETOKEN_KERNEL_CACHE_SPECS=freetoken__ggml scripts/build-release-wheels.sh`.
- CPU kernels need `python setup.py build_ext --inplace` after a change to `ggml-cpu/`, `src/`, `include/` or the CPU files in `freetoken/`.

## Seeing what differs from llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
LLAMA_CPP=~/llama.cpp scripts/sync-llama-cpp.sh diff
```

## Updating to a newer llama.cpp

Start with no uncommitted changes in the copied directories:

```bash
git -C ~/llama.cpp fetch origin
LLAMA_CPP=~/llama.cpp scripts/sync-llama-cpp.sh merge <llama.cpp commit>
```

The script merges every copied file three ways, with the pin as the base, and moves the pin in both `VERSION` files. To start over, run `git checkout -- python/freetoken/kernel/csrc/ggml python/freetoken/third_party/gguf`.

1. Resolve each `conflict:` file. Where we rewrote a function, keep ours and read llama.cpp's diff for fixes to port.
2. Go through the files the script lists as added, renamed or removed. Copy a new file only if FreeToken compiles it. The script skips a renamed file and prints it as `gone from llama.cpp:`, so `git mv` ours and merge it by hand with `git merge-file`.
3. Read the diffs the script lists as not copied, because our code follows them:
   - `ggml-cuda.cu`: `ggml_ops.cu`, `ggml_shim.cu` and `kernel/ggml.py`
   - `ggml.c`: `ggml_shim.cu`
   - `ggml-cpu/ggml-cpu.c`: `freetoken/cpu_kernels.c`
   - the `CMakeLists.txt` files: `setup.py`
4. Rebuild and test. A symbol missing from the CPU kernels still builds and fails only on import, which these tests do not reach, so `select()` has its own line.

   ```bash
   python setup.py build_ext --inplace
   python -c "from freetoken.kernel import ggml_cpu; ggml_cpu.select()"
   FREETOKEN_DISABLE_KERNEL_CACHE=1 pytest tests/kernels/test_ggml.py tests/moe/test_offload.py tests/models/test_qwen3_5_moe_weight.py -m "not slow"
   ```

5. Commit it all at once, so the pin holds for every file.

## Taking one fix from llama.cpp

Copy the change by hand and leave the pin alone. The next update merges it like any other local edit.
