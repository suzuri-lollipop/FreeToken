from __future__ import annotations

import importlib.util
from pathlib import Path

import sys

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension


ROOT = Path(__file__).parent

# Windows has no lib64 and no GCC flag spelling; MSVC is the only host toolchain there.
MSVC = sys.platform == "win32"


def _check_toolchain() -> None:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _cuda_runtime_paths() -> tuple[list[str], list[str]]:
    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA_HOME is required to build freetoken.kernel._pinned_tensor "
            "because it links against the CUDA runtime API."
        )
    cuda_home = Path(CUDA_HOME)
    library_dirs = [str(cuda_home / "lib64")]
    if (cuda_home / "lib").exists():
        library_dirs.append(str(cuda_home / "lib"))
    if MSVC:
        # the Windows toolkit keeps cudart.lib in lib/x64; lib64 does not exist there
        library_dirs.append(str(cuda_home / "lib" / "x64"))
    return [str(cuda_home / "include")], library_dirs


def _cxx_flags(*, pthread: bool = False) -> list[str]:
    """Host compile flags: GCC/Clang spellings on Linux, the MSVC equivalents on Windows."""
    if MSVC:
        return ["/O2", "/std:c++17", "/EHsc"]
    flags = ["-O3", "-std=c++17"]
    if pthread:
        flags.append("-pthread")
    return flags


cuda_include_dirs, cuda_library_dirs = _cuda_runtime_paths()
_check_toolchain()


setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=cuda_include_dirs,
            library_dirs=cuda_library_dirs,
            libraries=["cudart"],
            extra_compile_args=_cxx_flags(),
        ),
        # CPU-compute MoE executor for --moe-backend cpu. Links cudart for the
        # cudaLaunchHostFunc submit/sync graph nodes; the bf16 GEMV microkernels
        # use per-function target attributes (avx512bf16/avx512f) + a runtime
        # __builtin_cpu_supports dispatch, so the single binary stays portable
        # (scalar fallback) -- no global -march is set.
        #
        # GCC/Clang only: MSVC has neither __builtin_cpu_supports nor __x86_64__, so it
        # would build a scalar executor the viability probe still calls usable.
        *([] if MSVC else [
            CppExtension(
                name="freetoken.kernel._cpu_moe",
                sources=[
                    "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
                ],
                include_dirs=cuda_include_dirs,
                library_dirs=cuda_library_dirs,
                libraries=["cudart"],
                extra_compile_args=_cxx_flags(pthread=True),
            )
        ]),
        # --ple-backend disk row store; the TableFile/BatchReader/cumemop seams carry Win32
        # bodies, so this builds on Linux and Windows alike.
        CppExtension(
            name="freetoken.kernel._ple_store",
            sources=[
                "python/freetoken/kernel/csrc/ple_store/ple_store_ext.cpp",
            ],
            extra_compile_args=_cxx_flags(),
        ),
        # C++ radix prefix-tree core for the three prefix caches (plain/swa/hybrid);
        # pure host code (values stay caller-owned torch tensors), so no cudart link.
        # Missing build falls back to the Python trees (FREETOKEN_RADIX_BACKEND).
        CppExtension(
            name="freetoken.kernel._radix_tree",
            sources=[
                "python/freetoken/kernel/csrc/radix_tree/radix_tree_ext.cpp",
            ],
            extra_compile_args=_cxx_flags(),
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
