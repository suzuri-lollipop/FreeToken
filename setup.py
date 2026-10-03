from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import sys

from setuptools import setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDA_HOME, CppExtension, ROCM_HOME


ROOT = Path(__file__).parent
KERNEL_INCLUDE = str(ROOT / "python" / "freetoken" / "kernel" / "csrc" / "include")

# Windows has no lib64 and no GCC flag spelling; MSVC is the only host toolchain there.
MSVC = sys.platform == "win32"

def _vs_install_dirs() -> list[Path]:
    """VS installations on this machine, from the environment or from vswhere."""
    dirs = []
    vc = os.environ.get("VCINSTALLDIR")
    if vc:
        dirs.append(Path(vc).parent)
    vswhere = (Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
               / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    if vswhere.is_file():
        listing = subprocess.run([str(vswhere), "-products", "*", "-property", "installationPath"],
                                 capture_output=True, text=True, check=False).stdout
        dirs += [Path(line.strip()) for line in listing.splitlines() if line.strip()]
    return dirs


# _cpu_moe's SIMD tiers are per-function __attribute__((target)) with __builtin_cpu_supports
# choosing between them, spellings cl does not have. VS ships clang-cl beside the toolset; only
# the compiler moves, the link stays MSVC's (see _retarget_ninja and _clang_rt_ldflags).
def _find_clang_cl() -> str | None:
    override = os.environ.get("FREETOKEN_CLANG_CL", "")
    if override:
        return override if Path(override).is_file() else None
    found = shutil.which("clang-cl") or shutil.which("clang-cl.exe")
    if found:
        return found
    for install in _vs_install_dirs():
        exe = install / "VC" / "Tools" / "Llvm" / "x64" / "bin" / "clang-cl.exe"
        if exe.is_file():
            return str(exe)
    return None


def _activate_msvc_env() -> None:
    """Set up the developer environment the extensions build needs, if the shell has none.

    setuptools' own Visual Studio lookup fails in uv's isolated build environment on this box
    ("Unable to find a compatible Visual Studio installation"), while vcvars64 from the same
    install works, so run it and take its environment over. DISTUTILS_USE_SDK then stops
    distutils from activating a prompt of its own; torch warns when a prompt is active without it.
    """
    if not MSVC or shutil.which("cl") or os.environ.get("INCLUDE"):
        return
    for install in _vs_install_dirs():
        bat = install / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if not bat.is_file():
            continue
        env = subprocess.run(f'call "{bat}" >nul && set', shell=True, capture_output=True,
                             text=True, check=False).stdout
        imported = 0
        for line in env.splitlines():
            name, sep, value = line.partition("=")
            if sep and name:
                os.environ[name] = value
                imported += 1
        if imported:
            os.environ.setdefault("DISTUTILS_USE_SDK", "1")
            return


def _clang_rt_ldflags(clang_cl: str) -> list[str]:
    """link.exe is not driven by clang here, so __builtin_cpu_supports' compiler-rt symbols
    have to come in as an explicit library; torch quotes the path itself, so hand it raw."""
    base = Path(clang_cl).resolve().parent.parent / "lib" / "clang"
    found = sorted(base.glob("*/lib/windows/clang_rt.builtins-x86_64.lib"))
    if not found:
        raise RuntimeError(
            f"clang-cl at {clang_cl} ships no clang_rt.builtins-x86_64.lib under {base}; "
            "_cpu_moe cannot link (set FREETOKEN_CLANG_CL to a complete LLVM, or unset it to "
            "skip the extension)."
        )
    return [str(found[-1])]


# torch's Windows ninja rule is what _retarget_ninja edits; without ninja the build falls back
# to distutils, which would compile this one extension with cl and quietly produce the scalar
# executor, so clang-cl alone is not enough to ask for it. Both questions are answered after
# _activate_msvc_env(), which is what puts cl and ninja on PATH in a plain shell.
CLANG_CL: str | None = None


def _retarget_ninja(ninja_file: Path, driver: str) -> None:
    """Point a torch-generated ninja file at another C++ compiler.

    The Windows compile rule hardcodes `cl /showIncludes`, so CXX never reaches it. The dep
    scan and MSVC's /GL whole-program codegen leave with the compiler: ninja's `deps = msvc`
    parser would be fed the other front-end's output, and link.exe would be asked to LTCG over
    clang objects. Both are unneeded here, and dropping them keeps one variable in play.
    """
    lines = ninja_file.read_text(encoding="utf-8").splitlines()
    retargeted = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("command = cl "):
            lines[index] = line.replace("command = cl ", f"command = {driver} ").replace(
                "/showIncludes ", ""
            )
            retargeted = True
        elif stripped.startswith("cflags = "):
            lines[index] = line.replace(" /GL ", " ")
        elif stripped == "deps = msvc":
            lines[index] = ""
    if not retargeted:
        raise RuntimeError(
            f"{ninja_file} has no `command = cl` compile rule; _cpu_moe cannot be built "
            "with clang-cl -- drop FREETOKEN_CLANG_CL to skip the extension instead."
        )
    ninja_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check_toolchain() -> None:
    path = ROOT / "python" / "freetoken" / "kernel" / "_toolchain.py"
    spec = importlib.util.spec_from_file_location("_freetoken_toolchain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_nvcc_matches_torch()


def _is_rocm() -> bool:
    import torch
    return getattr(torch.version, "hip", None) is not None


def _rocm_paths() -> tuple[list[str], list[str], str]:
    candidates: list[Path] = []
    if os.getenv("ROCM_HOME"):
        candidates.append(Path(os.environ["ROCM_HOME"]))
    if ROCM_HOME:
        candidates.append(Path(ROCM_HOME))

    # ROCm 7.14 PyTorch images ship the SDK as a Python package instead of
    # installing it at /opt/rocm.
    spec = importlib.util.find_spec("_rocm_sdk_core")
    if spec and spec.submodule_search_locations:
        candidates.append(Path(next(iter(spec.submodule_search_locations))))
    candidates.append(Path("/opt/rocm"))

    for rocm_home in dict.fromkeys(candidates):
        include_dir = rocm_home / "include"
        library_dir = rocm_home / "lib"
        if not (include_dir / "hip" / "hip_runtime.h").exists():
            continue
        if (library_dir / "libamdhip64.so").exists():
            return [str(include_dir)], [str(library_dir)], "amdhip64"
        versioned = sorted(library_dir.glob("libamdhip64.so.*"))
        if versioned:
            return [str(include_dir)], [str(library_dir)], f":{versioned[-1].name}"

    searched = ", ".join(str(path) for path in dict.fromkeys(candidates))
    raise RuntimeError(
        "A ROCm SDK with HIP headers and libamdhip64 is required to build on ROCm; "
        f"searched: {searched}. Set ROCM_HOME to override."
    )


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


def _cxx_flags(*, pthread: bool = False, std: str = "c++17") -> list[str]:
    """Host compile flags: GCC/Clang spellings on Linux, the MSVC equivalents on Windows."""
    if MSVC:
        return ["/O2", f"/std:{std}", "/EHsc"]
    flags = ["-O3", f"-std={std}"]
    if pthread:
        flags.append("-pthread")
    return flags


def _build_ext_class():
    base = BuildExtension.with_options(use_ninja=True)

    class _BuildExt(base):
        """Build _cpu_moe with clang-cl and every other extension with cl.

        PATH gets the clang bin dir so the bare driver name resolves inside ninja, and the
        generated compile rule is retargeted because torch hardcodes cl (see _retarget_ninja).
        Linking stays with MSVC's link.exe, so the objects remain MSVC-ABI and /MD.
        """

        def build_extension(self, ext):
            if not (MSVC and CLANG_CL and ext.name.endswith("_cpu_moe")):
                return super().build_extension(ext)
            driver = Path(CLANG_CL).name
            saved_path = os.environ.get("PATH")
            os.environ["PATH"] = (
                str(Path(CLANG_CL).parent) + os.pathsep + (saved_path or "")
            )
            ext.extra_link_args = list(ext.extra_link_args or []) + _clang_rt_ldflags(CLANG_CL)
            original_run = cpp_extension._run_ninja_build

            def _run(build_directory, *args, **kwargs):
                _retarget_ninja(Path(build_directory) / "build.ninja", driver)
                return original_run(build_directory, *args, **kwargs)

            cpp_extension._run_ninja_build = _run
            try:
                return super().build_extension(ext)
            finally:
                cpp_extension._run_ninja_build = original_run
                os.environ["PATH"] = saved_path or ""

    return _BuildExt


IS_ROCM = _is_rocm()

if IS_ROCM:
    runtime_include_dirs, runtime_library_dirs, runtime_lib = _rocm_paths()
    runtime_link_args = [f"-Wl,-rpath,{runtime_library_dirs[0]}"]
    # These extensions contain host code only. BuildExtension supplies the ROCm
    # platform defines to the C++ compiler; offload architecture flags belong on
    # HIP device sources and would be rejected by the host compiler here.
else:
    runtime_include_dirs, runtime_library_dirs = _cuda_runtime_paths()
    runtime_lib = "cudart"
    runtime_link_args = []

_activate_msvc_env()
_check_toolchain()
CLANG_CL = _find_clang_cl() if MSVC and cpp_extension.is_ninja_available() else None


setup(
    ext_modules=[
        CppExtension(
            name="freetoken.kernel._pinned_tensor",
            sources=[
                "python/freetoken/kernel/csrc/pinned_tensor.cpp",
            ],
            include_dirs=[KERNEL_INCLUDE, *runtime_include_dirs],
            library_dirs=runtime_library_dirs,
            libraries=[runtime_lib],
            extra_compile_args=_cxx_flags(),
            extra_link_args=runtime_link_args,
        ),
        # CPU-compute MoE executor for --moe-backend cpu. Links cudart/amdhip64 for the
        # cudaLaunchHostFunc submit/sync graph nodes; the bf16 GEMV microkernels
        # use per-function target attributes (avx512bf16/avx512f) + a runtime
        # __builtin_cpu_supports dispatch, so the single binary stays portable
        # (scalar fallback) -- no global -march is set.
        #
        # On Windows this one extension is compiled by clang-cl instead of cl (see
        # _find_clang_cl); without it, skip -- MSVC would build a scalar executor that the
        # viability probe still reports usable.
        *([] if MSVC and CLANG_CL is None else [
            CppExtension(
                name="freetoken.kernel._cpu_moe",
                sources=[
                    "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
                ],
                include_dirs=[KERNEL_INCLUDE, *runtime_include_dirs],
                library_dirs=runtime_library_dirs,
                libraries=[runtime_lib],
                # c++20 is what makes clang-cl accept MSVC's <atomic> here (see _cxx_flags).
                extra_compile_args=_cxx_flags(pthread=True, std="c++20" if MSVC else "c++17"),
                extra_link_args=runtime_link_args,
            )
        ]),
        # disk-backed row store (PLE / Engram tables); the TableFile/BatchReader/cumemop
        # seams carry Win32 bodies, so this builds on Linux and Windows alike.
        CppExtension(
            name="freetoken.kernel._row_store",
            sources=[
                "python/freetoken/kernel/csrc/row_store/row_store_ext.cpp",
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
    cmdclass={"build_ext": _build_ext_class()},
)
