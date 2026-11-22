"""The vendored ggml-vulkan backend as a tvm-ffi module, exposing the same op surface as
:mod:`freetoken.kernel.ggml`'s CUDA module. Shaders are compiled with glslc on first use
and cached; the C/C++ sources then build through tvm-ffi like the other kernel modules.

Dependency discovery (no root needed): ``FREETOKEN_GLSLC_EXECUTABLE`` / ``glslc`` on PATH /
``~/.local/vulkan-deps/prefix/bin/glslc``; Vulkan headers from ``FREETOKEN_VULKAN_INCLUDE`` /
the system / ``~/.local/vulkan-deps/Vulkan-Headers-*/include``; the loader from
``FREETOKEN_VULKAN_LIB`` / ``-lvulkan`` / ``~/.local/vulkan-deps/prefix/lib/libvulkan.so``.
"""

from __future__ import annotations

import functools
import os
import pathlib
import shutil
import subprocess
import sys

_ROOT = pathlib.Path(__file__).parent / "csrc" / "ggml-vk"
_DEPS = pathlib.Path.home() / ".local" / "vulkan-deps"


def _glslc() -> str:
    env = os.environ.get("FREETOKEN_GLSLC_EXECUTABLE")
    if env:
        return env
    found = shutil.which("glslc")
    if found:
        return found
    local = _DEPS / "prefix" / "bin" / "glslc"
    if local.exists():
        return str(local)
    raise RuntimeError("ggml vk: glslc not found (set FREETOKEN_GLSLC_EXECUTABLE)")


def _vulkan_include() -> str:
    env = os.environ.get("FREETOKEN_VULKAN_INCLUDE")
    if env:
        return env
    if pathlib.Path("/usr/include/vulkan/vulkan.hpp").exists():
        return "/usr/include"
    for cand in sorted(_DEPS.glob("Vulkan-Headers-*/include")):
        if (cand / "vulkan" / "vulkan.hpp").exists():
            return str(cand)
    raise RuntimeError("ggml vk: Vulkan headers not found (set FREETOKEN_VULKAN_INCLUDE)")


def _vulkan_ldflags() -> list[str]:
    env = os.environ.get("FREETOKEN_VULKAN_LIB")
    if env:
        return [env]
    local = _DEPS / "prefix" / "lib" / "libvulkan.so"
    if local.exists():
        return [str(local)]
    return ["-lvulkan"]


def _cuda_flags() -> tuple[list[str], list[str], list[str]]:
    """(defines, includes, ldflags) for staging torch CUDA tensors across the host."""
    try:
        import torch

        if not torch.cuda.is_available():
            return [], [], []
        from torch.utils.cpp_extension import CUDA_HOME

        if not CUDA_HOME:
            return [], [], []
        home = pathlib.Path(CUDA_HOME)
        if not (home / "include" / "cuda_runtime.h").exists():
            return [], [], []
        return (
            ["-DFREETOKEN_VK_CUDA"],
            [str(home / "include")],
            [f"-L{home / 'lib64'}", "-lcudart"],
        )
    except Exception:
        return [], [], []


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ggml vk: {' '.join(cmd[:3])}... failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")


def _generate_shaders(build_dir: pathlib.Path) -> pathlib.Path:
    """Compile every shader variant with glslc and embed the SPIR-V as C arrays."""
    vk_dir = _ROOT / "ggml-vulkan"
    shader_dir = vk_dir / "vulkan-shaders"
    gen_src = shader_dir / "vulkan-shaders-gen.cpp"
    gen_bin = build_dir / "vulkan-shaders-gen"
    spv_dir = build_dir / "vulkan-shaders.spv"
    header = spv_dir / "ggml-vulkan-shaders.hpp"
    spv_dir.mkdir(parents=True, exist_ok=True)

    if not gen_bin.exists() or gen_bin.stat().st_mtime < gen_src.stat().st_mtime:
        cxx = os.environ.get("CXX", "c++")
        _run([cxx, "-std=c++17", "-O2", "-pthread", str(gen_src), "-o", str(gen_bin)])

    glslc = _glslc()
    if not header.exists():
        _run([str(gen_bin), "--output-dir", str(spv_dir), "--target-hpp", str(header)])

    for comp in sorted(shader_dir.glob("*.comp")):
        target_cpp = build_dir / (comp.name + ".cpp")
        if target_cpp.exists() and target_cpp.stat().st_mtime >= comp.stat().st_mtime:
            continue
        _run([
            str(gen_bin),
            "--glslc", glslc,
            "--source", str(comp),
            "--output-dir", str(spv_dir),
            "--target-hpp", str(header),
            "--target-cpp", str(target_cpp),
        ])
    return spv_dir


def load_module(build_directory: str | None = None):
    from tvm_ffi.cpp import load

    from .utils import _ensure_ninja_on_path

    _ensure_ninja_on_path()
    build_dir = pathlib.Path(build_directory) if build_directory else pathlib.Path(
        os.environ.get("FREETOKEN_GGML_VK_BUILD_DIR", str(pathlib.Path.home() / ".cache" / "freetoken" / "ggml-vk"))
    )
    build_dir.mkdir(parents=True, exist_ok=True)
    spv_dir = _generate_shaders(build_dir)

    cuda_defs, cuda_incs, cuda_lds = _cuda_flags()
    sources = [
        _ROOT / "src" / f for f in ("ggml.c", "ggml-alloc.c", "ggml-backend.cpp", "ggml-quants.c", "ggml-threading.cpp", "gguf.cpp", "ggml-backend-meta.cpp")
    ]
    sources += sorted((_ROOT / "ggml-vulkan").glob("*.cpp"))
    sources += sorted(build_dir.glob("*.comp.cpp"))
    sources += [_ROOT / "freetoken" / "ggml_vk_ops.cpp"]
    return load(
        "freetoken_ggml_vk",
        cpp_files=[str(s) for s in sources],
        extra_cflags=["-DGGML_USE_VULKAN", *cuda_defs],
        extra_ldflags=[*_vulkan_ldflags(), *cuda_lds],
        extra_include_paths=[
            str(_ROOT / "include"),
            str(_ROOT / "src"),
            str(_ROOT / "ggml-vulkan"),
            str(spv_dir),
            _vulkan_include(),
            str(_DEPS / "prefix" / "include"),
            *cuda_incs,
        ],
        build_directory=str(build_dir / "module"),
    )


@functools.cache
def _module():
    return load_module()


__all__ = ["load_module"]
