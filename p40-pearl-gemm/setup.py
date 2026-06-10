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
ADDITIONAL_ARCHS = os.environ.get(
    "ADDITIONAL_ARCHS",
    "sm_70,sm_75,sm_80,sm_86"
)

_GENCODE_FLAGS = [f"-gencode {COMPUTE_CAP}"]
for arch in ADDITIONAL_ARCHS.split(","):
    arch = arch.strip()
    if arch and arch != TARGET_ARCH:
        _GENCODE_FLAGS.append(
            f"-gencode arch=compute_61,code={arch}"
        )

sources = [
    # Pascal-specific kernels (DP4A-based)
    "csrc/gemm/dp4a_gemm_sm61.cu",
    "csrc/gemm/noising_sm61.cu",
    "csrc/gemm/api_sm61.cu",

    # Architecture-independent kernels from upstream pearl-gemm
    "csrc/blake3/blake3.cu",
    "csrc/tensor_hash/tensor_hash.cu",
    "csrc/gemm/noise_generation.cu",
    "csrc/gemm/quantize_kernel.cu",
    "csrc/gemm/inner_hash_kernel.cu",
    "csrc/gemm/denoise_converter.cu",
]

nvcc_flags = [
    "-O3",
    "-std=c++17",
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

gcc_flags = [
    "-O3",
    "-std=c++17",
    "-fvisibility=hidden",
]

include_dirs = [
    CSRC_DIR,
    CSRC_DIR / "gemm",
    CSRC_DIR / "blake3",
    CSRC_DIR / "tensor_hash",
]

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
