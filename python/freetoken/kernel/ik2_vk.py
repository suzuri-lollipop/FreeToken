"""ik2-vk: High-throughput Vulkan 1.2 compute ops (BDA + subgroupAdd + zero-copy
CUDA<->Vulkan external memory & external semaphores) for Inkling-2 and GGUF ops.
Shaders compile with glslc on first use; the C++ orchestrator builds through tvm-ffi.
"""

from __future__ import annotations

import functools
import os
import pathlib
import subprocess

from .ggml_vk import _cuda_flags, _glslc, _vulkan_include, _vulkan_ldflags

_ROOT = pathlib.Path(__file__).parent / "csrc" / "ik2-vk"

# Quantization / float ggml types supported natively by the BDA warp-coalesced shaders:
# F32(0), Q4_0(2), Q8_0(8), Q6_K(14), BF16(30), NVFP4(40)
SUPPORTED_TYPES = frozenset((0, 2, 8, 14, 30, 40))


def supports_type(ggml_type: int) -> bool:
    return int(ggml_type) in SUPPORTED_TYPES


def _generate_header(build_dir: pathlib.Path) -> pathlib.Path:
    shader_dir = _ROOT / "shaders"
    header = build_dir / "ik2_shaders.h"
    glslc = _glslc()
    parts = ["#pragma once", "#include <cstdint>", "#include <cstddef>"]
    all_srcs = list(shader_dir.glob("*.comp")) + list(shader_dir.glob("*.glsl"))
    newest = max(p.stat().st_mtime for p in all_srcs)
    if header.exists() and header.stat().st_mtime >= newest:
        return header
    for comp in sorted(shader_dir.glob("*.comp")):
        spv = build_dir / (comp.stem + ".spv")
        proc = subprocess.run(
            [glslc, "-fshader-stage=compute", "--target-env=vulkan1.2", "-O", str(comp), "-o", str(spv)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ik2_vk: glslc {comp.name} failed:\n{proc.stderr}")
        data = spv.read_bytes()
        name = comp.stem.replace(".", "_")
        words = ", ".join(str(int.from_bytes(data[i:i + 4], "little")) for i in range(0, len(data), 4))
        parts.append(f"static const uint32_t {name}_spv[] = {{ {words} }};")
        parts.append(f"static const size_t {name}_spv_len = sizeof({name}_spv);")
    header.write_text("\n".join(parts) + "\n")
    return header


def load_module(build_directory: str | None = None):
    from tvm_ffi.cpp import load

    from .utils import _ensure_ninja_on_path

    _ensure_ninja_on_path()
    build_dir = pathlib.Path(build_directory) if build_directory else pathlib.Path(
        os.environ.get("FREETOKEN_IK2_VK_BUILD_DIR", str(pathlib.Path.home() / ".cache" / "freetoken" / "ik2-vk"))
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    _generate_header(build_dir)
    cuda_defs, cuda_incs, cuda_lds = _cuda_flags()
    return load(
        "freetoken_ik2_vk",
        cpp_files=[str(_ROOT / "ik2_vk.cpp")],
        extra_cflags=["-DFREETOKEN_IK2_VK", *cuda_defs],
        extra_ldflags=[*_vulkan_ldflags(), *cuda_lds],
        extra_include_paths=[str(build_dir), _vulkan_include(), *cuda_incs],
        build_directory=str(build_dir / "module"),
    )


@functools.cache
def _module():
    return load_module()


def set_bank_mode(resident: bool) -> None:
    _module().ik2_vk_set_bank_mode(resident)


def begin_graph() -> None:
    _module().ik2_vk_begin_graph()


def end_graph() -> None:
    _module().ik2_vk_end_graph()


def pre_submit_graph() -> None:
    _module().ik2_vk_pre_submit_graph()


def mul_mat(w, ggml_type: int, x, *, out=None):
    import torch

    if out is None:
        out = torch.empty(x.shape[0], w.shape[0], dtype=x.dtype, device=x.device)
    _module().ik2_vk_mul_mat(w, int(ggml_type), x.contiguous(), out)
    return out


def mul_mat_batch(x, ws, tys, outs):
    assert 1 <= len(ws) <= 6
    args = [x.contiguous()]
    for i in range(6):
        if i < len(ws):
            args.extend((ws[i], int(tys[i]), outs[i]))
        else:
            args.extend((None, 0, None))
    _module().ik2_vk_mul_mat_batch(*args)
    return outs


def moe(wgu, gu_ty: int, wdn, dn_ty: int, x, ids, weights, scales=None, *, gcap: float = 4.0, ucap: float = 25.0):
    """Fused Gate+Up+SITU -> Down -> Weighted Sum in one command buffer & one handoff."""
    import torch

    out = torch.empty(x.shape[0], wdn.shape[1], dtype=x.dtype, device=x.device)
    _module().ik2_vk_moe(
        wgu, int(gu_ty), wdn, int(dn_ty), x.contiguous(), ids.contiguous(),
        weights.float().contiguous(), scales, out, float(gcap), float(ucap),
    )
    return out


def mul_mat_id(w, ggml_type: int, x, ids, scale=None):
    import torch

    t, top_k = ids.shape
    out = torch.empty(t, top_k, w.shape[1], dtype=x.dtype, device=x.device)
    x3 = (x.unsqueeze(1) if x.dim() == 2 else x).contiguous()
    _module().ik2_vk_mul_mat_id(w, int(ggml_type), x3, ids.contiguous(), out, scale)
    return out


def moe_weighted_sum(y, weights):
    import torch

    out = torch.empty(y.shape[0], y.shape[2], dtype=y.dtype, device=y.device)
    _module().ik2_vk_wsum(y.contiguous(), weights.float().contiguous(), out)
    return out


def get_rows(w, ggml_type: int, k: int, ids, dtype):
    import torch

    out = torch.empty(ids.shape[0], k, dtype=dtype, device=ids.device)
    _module().ik2_vk_get_rows(w, int(ggml_type), int(k), ids.contiguous(), out)
    return out


def dequantize(w, ggml_type: int, k: int, dtype):
    import torch

    rows = w.numel() // w.shape[-1]
    out = torch.empty(rows, k, dtype=dtype, device=w.device)
    _module().ik2_vk_dequantize(w, int(ggml_type), int(k), out)
    return out


__all__ = [
    "SUPPORTED_TYPES",
    "dequantize",
    "get_rows",
    "load_module",
    "moe",
    "moe_weighted_sum",
    "mul_mat",
    "mul_mat_batch",
    "mul_mat_id",
    "set_bank_mode",
    "supports_type",
]
