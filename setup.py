from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
from pathlib import Path

import sys

from setuptools import Extension, setup
from setuptools.command.build_clib import build_clib
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


def _cxx_flags(*, pthread: bool = False) -> list[str]:
    """Host compile flags: GCC/Clang spellings on Linux, the MSVC equivalents on Windows.

    c++20 is not a choice here: every one of these sources includes torch/extension.h, and
    torch 2.14's headers #error out below C++20. It is also what makes clang-cl accept
    MSVC's <atomic> in the cpu_moe sources.
    """
    if MSVC:
        return ["/O2", "/std:c++20", "/EHsc"]
    flags = ["-O3", "-std=c++20"]
    if pthread:
        flags.append("-pthread")
    return flags


IS_ROCM = _is_rocm()

if IS_ROCM:
    runtime_include_dirs, runtime_library_dirs, runtime_lib = _rocm_paths()
    runtime_link_args = [f"-Wl,-rpath,{runtime_library_dirs[0]}"]
    # Host sources. BuildExtension supplies the ROCm
    # platform defines to the C++ compiler; offload architecture flags belong on
    # HIP device sources and would be rejected by the host compiler here.
    # Device sources (.cu, built by hipcc); BuildExtension adds the offload architectures.
    device_compile = ["-O3", "-std=c++20"]
else:
    runtime_include_dirs, runtime_library_dirs = _cuda_runtime_paths()
    runtime_lib = "cudart"
    runtime_link_args = []
    # Device sources (.cu, built by nvcc): SASS for each major arch, which also runs on the newer minors of that major,
    # plus PTX for the oldest arch nvcc 13 supports, which the driver builds for any other GPU at load.
    # Naming archs also stops torch from picking them, which fails on a build machine without a GPU.
    device_compile = ["-O3", "-std=c++20"] + [f"-gencode=arch=compute_{a},code=sm_{a}" for a in (75, 80, 90, 100, 120)]
    device_compile.append("-gencode=arch=compute_75,code=compute_75")

_activate_msvc_env()
_check_toolchain()
CLANG_CL = _find_clang_cl() if MSVC and cpp_extension.is_ninja_available() else None

# ggml-cpu's row kernels for GGUF experts on the CPU MoE executor: one module per x86 variant of llama.cpp's GGML_CPU_ALL_VARIANTS (ggml/src/CMakeLists.txt), freetoken.kernel.ggml_cpu._<variant>.
# freetoken.kernel.ggml_cpu.select() imports the best one this CPU runs; each module's entry is built without the variant's ISA flags so that check is safe on any CPU.
GGML_DIR = "python/freetoken/kernel/csrc/ggml"
GGML_CPU = platform.machine().lower() in ("x86_64", "amd64")
_HASWELL = ("SSE42", "AVX", "F16C", "FMA", "AVX2", "BMI2")
GGML_CPU_VARIANTS = {
    "x64": (),
    "sse42": ("SSE42",),
    "sandybridge": ("SSE42", "AVX"),
    "ivybridge": ("SSE42", "AVX", "F16C"),
    "piledriver": ("SSE42", "AVX", "F16C", "FMA"),
    "haswell": _HASWELL,
    "skylakex": (*_HASWELL, "AVX512"),
    "cannonlake": (*_HASWELL, "AVX512", "AVX512_VBMI"),
    "cascadelake": (*_HASWELL, "AVX512", "AVX512_VNNI"),
    "icelake": (*_HASWELL, "AVX512", "AVX512_VBMI", "AVX512_VNNI"),
    "cooperlake": (*_HASWELL, "AVX512", "AVX512_VNNI", "AVX512_BF16"),
    "zen4": (*_HASWELL, "AVX512", "AVX512_VBMI", "AVX512_VNNI", "AVX512_BF16"),
    "alderlake": (*_HASWELL, "AVX_VNNI"),
    "sapphirerapids": (*_HASWELL, "AVX512", "AVX512_VBMI", "AVX512_VNNI", "AVX512_BF16", "AMX_TILE", "AMX_INT8"),
}
_GGML_GCC_FLAGS = {
    "SSE42": ["-msse4.2"], "F16C": ["-mf16c"], "FMA": ["-mfma"], "BMI2": ["-mbmi2"], "AVX": ["-mavx"], "AVX2": ["-mavx2"],
    "AVX_VNNI": ["-mavxvnni"], "AVX512": ["-mavx512f", "-mavx512cd", "-mavx512vl", "-mavx512dq", "-mavx512bw"],
    "AVX512_VBMI": ["-mavx512vbmi"], "AVX512_VNNI": ["-mavx512vnni"], "AVX512_BF16": ["-mavx512bf16"],
    "AMX_TILE": ["-mamx-tile"], "AMX_INT8": ["-mamx-int8"],
}


def _ggml_variant_flags(feats: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """(ISA flags, macros) of one variant, as upstream's ggml-cpu/CMakeLists.txt sets them; the macros also tell cpu-feats.cpp what to check."""
    if sys.platform != "win32":
        return [f for feat in feats for f in _GGML_GCC_FLAGS[feat]], [f"GGML_{feat}" for feat in feats]
    # cl takes one /arch: level and does not define the macros of the extensions above it
    if "AVX512" in feats:
        flags, macros = ["/arch:AVX512"], ["GGML_AVX512"]
        if "AVX512_VBMI" in feats:
            macros.append("__AVX512VBMI__")
        if "AVX512_VNNI" in feats:
            macros += ["__AVX512VNNI__", "GGML_AVX512_VNNI"]
    elif "AVX2" in feats:
        flags, macros = ["/arch:AVX2"], ["GGML_AVX2", "GGML_FMA", "GGML_F16C"]
    elif "AVX" in feats:
        flags, macros = ["/arch:AVX"], ["GGML_AVX"]
    elif "SSE42" in feats:
        flags, macros = ["/arch:SSE4.2"], ["GGML_SSE42"]
    else:
        flags, macros = [], []
    if "AVX_VNNI" in feats:
        macros += ["__AVXVNNI__", "GGML_AVX_VNNI"]
    if "BMI2" in feats:
        macros += ["__BMI2__", "GGML_BMI2"]
    return flags, macros


def _ggml_cpu_targets() -> tuple[list, list, dict[str, list[str]]]:
    """(build_clib libraries, variant modules, the static libraries each module links in link order)."""
    if not GGML_CPU:
        return [], [], {}
    if sys.platform == "win32":
        cflags, c_std, cxx_std, define = ["/O2", "/DNDEBUG", "/utf-8", "/D_CRT_SECURE_NO_WARNINGS"], "/std:c11", "/std:c++20", "/D"
        # upstream builds these variants only with GCC or Clang
        variants = {k: v for k, v in GGML_CPU_VARIANTS.items() if k not in ("ivybridge", "piledriver", "cooperlake", "zen4", "sapphirerapids")}
    else:
        cflags, c_std, cxx_std, define = ["-O3", "-fPIC", "-fvisibility=hidden", "-DNDEBUG"], "-std=c11", "-std=c++20", "-D"
        variants = GGML_CPU_VARIANTS
    build = {
        "include_dirs": [f"{GGML_DIR}/include", f"{GGML_DIR}/src", f"{GGML_DIR}/ggml-cpu", GGML_DIR],
        "obj_deps": {"": sorted(str(p.relative_to(ROOT)) for d in ("include", "src", "ggml-cpu", "freetoken") for p in (ROOT / GGML_DIR / d).rglob("*.h"))},
    }
    # ggml-base, built once without ISA flags like upstream's; arch/x86/quants.c calls its quantize_row_*_ref
    libraries = [("freetoken_ggml_base", {"sources": [f"{GGML_DIR}/src/ggml-quants.c"], "cflags": [*cflags, c_std], **build})]
    modules, module_libs = [], {}
    for variant, feats in variants.items():
        flags, macros = _ggml_variant_flags(feats)
        defines = [define + m for m in macros]
        libraries += [
            (f"freetoken_ggml_cpu_{variant}", {
                "sources": [f"{GGML_DIR}/ggml-cpu/quants.c", f"{GGML_DIR}/ggml-cpu/arch/x86/quants.c", f"{GGML_DIR}/freetoken/cpu_kernels.c"],
                "cflags": [*cflags, *flags, *defines, c_std],
                **build,
            }),
            # the float row kernels; C++, so a library of their own
            (f"freetoken_ggml_cpu_vec_{variant}", {"sources": [f"{GGML_DIR}/ggml-cpu/vec.cpp"], "cflags": [*cflags, *flags, *defines, cxx_std], **build}),
        ]
        name = f"freetoken.kernel.ggml_cpu._{variant}"
        module_libs[name] = [f"freetoken_ggml_cpu_{variant}", f"freetoken_ggml_cpu_vec_{variant}", "freetoken_ggml_base"]
        modules.append(Extension(
            name=name,
            sources=[f"{GGML_DIR}/freetoken/variant_module.cpp"],
            # absolute: ninja compiles from the build directory
            include_dirs=[str(ROOT / d) for d in (*build["include_dirs"], f"{GGML_DIR}/freetoken")],
            extra_compile_args=[*cflags, *defines, f"{define}FT_GGML_VARIANT={variant}", cxx_std],
        ))
    return libraries, modules, module_libs


ggml_cpu_libraries, ggml_cpu_modules, ggml_cpu_module_libs = _ggml_cpu_targets()


class BuildClib(build_clib):
    def build_libraries(self, libraries):
        # the variants compile the same sources with different flags, so each library gets its own object directory
        build_temp = self.build_temp
        try:
            for lib in libraries:
                self.build_temp = os.path.join(build_temp, lib[0])
                super().build_libraries([lib])
        finally:
            self.build_temp = build_temp


class BuildExt(BuildExtension.with_options(use_ninja=True)):
    """Builds the ggml-cpu variant libraries, then either the clang-cl or the cl form of an extension."""

    def run(self):
        # build_ext alone (`setup.py build_ext --inplace`) does not run build_clib
        if self.distribution.has_c_libraries():
            self.run_command("build_clib")
        # build_ext would link every build_clib library into every extension; build_extension adds each module's own
        libraries, self.distribution.libraries = self.distribution.libraries, None
        try:
            super().run()
        finally:
            self.distribution.libraries = libraries

    def build_extension(self, ext):
        if ext.name in ggml_cpu_module_libs:
            build_clib_dir = self.get_finalized_command("build_clib").build_clib
            if build_clib_dir not in ext.library_dirs:
                ext.library_dirs.append(build_clib_dir)
            for name in ggml_cpu_module_libs[ext.name]:
                if name not in ext.libraries:
                    ext.libraries.append(name)
                lib = os.path.join(build_clib_dir, self.compiler.library_filename(name))
                # relink when only a static library changed
                if lib not in ext.depends:
                    ext.depends.append(lib)
            # every variant module compiles the same source, so each gets its own object directory
            build_temp, self.build_temp = self.build_temp, os.path.join(self.build_temp, ext.name)
            try:
                super().build_extension(ext)
            finally:
                self.build_temp = build_temp
            return
        if not (MSVC and CLANG_CL and ext.name.endswith("_cpu_moe")):
            super().build_extension(ext)
            return
        # _cpu_moe only: PATH gets the clang bin dir so the bare driver name resolves inside
        # ninja, and the generated compile rule is retargeted because torch hardcodes cl (see
        # _retarget_ninja). Linking stays with MSVC's link.exe, so the objects remain
        # MSVC-ABI and /MD.
        driver = Path(CLANG_CL).name
        saved_path = os.environ.get("PATH")
        os.environ["PATH"] = str(Path(CLANG_CL).parent) + os.pathsep + (saved_path or "")
        ext.extra_link_args = list(ext.extra_link_args or []) + _clang_rt_ldflags(CLANG_CL)
        original_run = cpp_extension._run_ninja_build

        def _run(build_directory, *args, **kwargs):
            _retarget_ninja(Path(build_directory) / "build.ninja", driver)
            return original_run(build_directory, *args, **kwargs)

        cpp_extension._run_ninja_build = _run
        try:
            super().build_extension(ext)
        finally:
            cpp_extension._run_ninja_build = original_run
            os.environ["PATH"] = saved_path or ""


setup(
    libraries=ggml_cpu_libraries,
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
# On Windows this one extension is compiled by clang-cl instead of cl (see
        # _find_clang_cl); without it, skip -- MSVC would build a scalar executor that the
        # viability probe still reports usable.
        *([] if MSVC and CLANG_CL is None else [
            CppExtension(
                name="freetoken.kernel._cpu_moe",
                sources=[
                    "python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp",
                    "python/freetoken/kernel/csrc/cpu_moe/locked_fetch.cpp",
                    "python/freetoken/kernel/csrc/cpu_moe/locked_fetch_io.cu",
                    f"python/freetoken/kernel/csrc/cpu_moe/gpu_sync_{'hip' if IS_ROCM else 'cuda'}.cpp",
                    "python/freetoken/kernel/csrc/cpu_moe/gpu_sync_io.cu",
                    *sorted(
                        str(p.relative_to(ROOT))
                        for p in (ROOT / "python/freetoken/kernel/csrc/cpu_moe/kernels").glob("*.cpp")
                    ),
                ],
                include_dirs=[KERNEL_INCLUDE, *runtime_include_dirs, str(ROOT / GGML_DIR / "freetoken")],
                library_dirs=runtime_library_dirs,
                libraries=[runtime_lib],
                extra_compile_args={
                    "cxx": _cxx_flags(pthread=True),
                    "nvcc": device_compile,
                },
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
        *ggml_cpu_modules,
    ],
    cmdclass={"build_clib": BuildClib, "build_ext": BuildExt},
)
