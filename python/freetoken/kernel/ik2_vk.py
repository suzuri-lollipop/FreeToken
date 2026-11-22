"""ik2-vk: Inkling-2's routed-expert FFN on dedicated Vulkan compute shaders (maya-ik2
style: one command buffer per call, fixed-order F32 arithmetic, in-shader weight decode).
Shaders compile with glslc on first use; the C++ orchestrator builds through tvm-ffi.
"""

from __future__ import annotations

import functools
import os
import pathlib
import subprocess

from .ggml_vk import _cuda_flags, _glslc, _vulkan_include, _vulkan_ldflags

_ROOT = pathlib.Path(__file__).parent / "csrc" / "ik2-vk"


def _generate_header(build_dir: pathlib.Path) -> pathlib.Path:
    shader_dir = _ROOT / "shaders"
    header = build_dir / "ik2_shaders.h"
    glslc = _glslc()
    parts = ["#pragma once", "#include <cstdint>", "#include <cstddef>"]
    newest = max(p.stat().st_mtime for p in shader_dir.glob("*.comp"))
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


def moe(wgu, gu_ty: int, wdn, dn_ty: int, x, ids, weights, scales, *, gcap: float = 4.0, ucap: float = 25.0):
    """One call = one command buffer: gate_up GEMV -> SITU -> down GEMV per routed expert,
    then the weighted sum. The banks must be host tensors (fused serving); they upload
    once into resident device buffers."""
    import torch

    out = torch.empty(x.shape[0], wdn.shape[1], dtype=x.dtype, device=x.device)
    _module().ik2_vk_moe(wgu, gu_ty, wdn, dn_ty, x, ids, weights, scales, out, gcap, ucap)
    return out


__all__ = ["load_module", "moe"]
