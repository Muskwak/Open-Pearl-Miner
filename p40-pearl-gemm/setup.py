import os
import sys
from pathlib import Path

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = Path(__file__).absolute().parent
CSRC_DIR = ROOT_DIR / "csrc"

TARGET_ARCH = os.environ.get("TARGET_ARCH", "sm_61")
if TARGET_ARCH.startswith("sm_"):
    cc = TARGET_ARCH.split("_")[1]
else:
    cc = "61"
COMPUTE_CAP = f"arch=compute_{cc},code={TARGET_ARCH}"

# Additional architectures to enable broad compatibility
# Pascal: sm_61 (P40, GTX 1080), sm_60 (P100)
# Volta+: sm_70, sm_75, sm_80, sm_86, sm_89 (for fallback testing on newer GPUs)
# Default to the target arch only (Pascal sm_61). Building the extra Volta+
# arches multiplies nvcc compile time ~5x; set ADDITIONAL_ARCHS explicitly to
# re-enable them for broad-compat builds.
ADDITIONAL_ARCHS = os.environ.get("ADDITIONAL_ARCHS", "")

_GENCODE_FLAGS = []
for a in [TARGET_ARCH] + [a.strip() for a in ADDITIONAL_ARCHS.split(",") if a.strip()]:
    if a.startswith("sm_"):
        cc = a.split("_")[1]
    else:
        cc = a
    _GENCODE_FLAGS.extend(["-gencode", f"arch=compute_{cc},code={a}"])

sources = [
    # Python bindings (pybind11 module registration).
    # Built as a host .cpp (MSVC/gcc), NOT nvcc: the torch headers do not parse
    # under nvcc + C++20, and the upstream CuTe kernels require C++20.
    "csrc/gemm/bindings.cpp",

    # Pascal-specific kernels (DP4A-based)
    "csrc/gemm/dp4a_gemm_sm61.cu",
    "csrc/gemm/noising_sm61.cu",
    "csrc/gemm/api_sm61.cu",
    "csrc/gemm/pearl_pow_sm61.cu",
    "csrc/gemm/pearl_pow_fused_sm61.cu",

    # Architecture-independent kernels from upstream pearl-gemm
    "csrc/blake3/blake3.cu",
    "csrc/gemm/noise_generation.cu",
    "csrc/gemm/quantize_kernel.cu",
    "csrc/gemm/inner_hash_kernel.cu",
    "csrc/gemm/denoise_converter.cu",

    # tensor_hash.cu now builds for Pascal: it includes tensor_hash_host_sm61.hpp,
    # which replaces the SM90 TMA/warpgroup leaf kernel with a direct-global-load
    # Pascal kernel (merkle_tree_roots_kernel_sm61.hpp) and reuses the stock CuTe
    # ComputeBlakeMT/ReduceRoots stages.
    "csrc/tensor_hash/tensor_hash.cu",
]

nvcc_flags = [
    "-O3",
    # Upstream CuTe code uses C++20 features (designated initializers in
    # blake3.cuh, `requires` clauses in utils.h), so C++20 is required.
    "-std=c++20",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--use_fast_math",
    "-lineinfo",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--ptxas-options=--verbose,--warn-on-local-memory-usage",
    "-DNDEBUG",
]

# Host-compiler flags for the .cpp bindings. On Windows these go to MSVC (cl),
# which uses different flag syntax; torch's BuildExtension supplies /std:c++17.
if sys.platform == "win32":
    gcc_flags = ["/O2"]
else:
    gcc_flags = ["-O3", "-std=c++17", "-fvisibility=hidden"]

include_dirs = [
    CSRC_DIR,
    CSRC_DIR / "gemm",
    CSRC_DIR / "blake3",
    CSRC_DIR / "tensor_hash",
]

# CUTLASS include path. Override with the CUTLASS_DIR env var (pointing at the
# directory that contains `cutlass/` and `cute/`); falls back to the local
# aphrodite-engine checkout for backwards compatibility.
_DEFAULT_CUTLASS = r"C:\Users\ADMIN\audits\aphrodite-engine\.deps\cutlass-src\include"
cutlass_dir = os.environ.get("CUTLASS_DIR", _DEFAULT_CUTLASS)
if not (Path(cutlass_dir) / "cutlass").exists():
    print(
        f"WARNING: CUTLASS headers not found at {cutlass_dir!r}. "
        "Set CUTLASS_DIR to the include dir that contains cutlass/ and cute/.",
        file=sys.stderr,
    )
include_dirs.append(cutlass_dir)

# Find CUB (comes with CUDA toolkit)
cuda_home = torch.utils.cpp_extension.CUDA_HOME
if cuda_home:
    cub_path = Path(cuda_home) / "include"
    if cub_path.exists():
        include_dirs.append(cub_path)

ext_modules = [
    CUDAExtension(
        name="p40_pearl_gemm_cuda",
        sources=[str(s) for s in sources],
        extra_compile_args={
            "cxx": gcc_flags,
            "nvcc": nvcc_flags + _GENCODE_FLAGS,
        },
        include_dirs=[str(d) for d in include_dirs],
        libraries=["cuda"],
    ),
]

setup(
    name="p40-pearl-gemm",
    version="0.1.0",
    description="Pascal P40-optimized CUDA kernels for Pearl (PRL) mining",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    packages=["p40_pearl_gemm"],
    package_dir={"p40_pearl_gemm": "python"},
    python_requires=">=3.12",
    install_requires=["torch>=2.0.0"],
)
