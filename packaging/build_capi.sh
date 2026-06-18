#!/usr/bin/env bash
# Build the torch-free CUDA shared library (p40cuda.dll / libp40cuda.so).
# Links only the CUDA runtime — no torch, no pybind. Driven from Python via ctypes.
#
# Run from p40-pearl-gemm/ :  bash packaging/build_capi.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CUTLASS_DIR:="$(pwd)/.deps/cutlass/include"}"
NVCC="${NVCC:-nvcc}"

# Default arch: Pascal (sm_61) + Ampere (sm_80, sm_86) + Ada (sm_89).
# The DP4A kernels run on sm_61, tensor-core kernels on sm_80+.
# Override GENCODE to add/remove arches, e.g. for a pure-Ampere farm:
#   GENCODE="-gencode arch=compute_86,code=sm_86 \
#            -gencode arch=compute_89,code=sm_89 \
#            -gencode arch=compute_86,code=compute_86"
if [ -z "${GENCODE:-}" ]; then
  GENCODE="-gencode arch=compute_61,code=sm_61 \
           -gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_86,code=sm_86 \
           -gencode arch=compute_89,code=sm_89"
fi
case "$(uname -s)" in
  *NT*|*MINGW*|*MSYS*) OUT="p40cuda.dll" ;;
  *) OUT="libp40cuda.so" ;;
esac

SRC=(
  csrc/capi/p40_capi.cu
  csrc/gemm/pearl_gemm_only_sm61.cu
  csrc/gemm/pearl_blake3_sm61.cu
  csrc/gemm/noising_sm61.cu
  csrc/gemm/noise_generation.cu
  csrc/blake3/blake3.cu
  csrc/gemm/rng_fill_sm61.cu
  csrc/tensor_hash/tensor_hash.cu
  csrc/gemm/noise_gemm_sm61.cu
  csrc/gemm/pearl_ampere_tc.cu
)

# -Xcompiler -fPIC is required for a Linux shared library; -allow-unsupported-
# compiler keeps newer host GCC (e.g. 13.3 on Ubuntu 24.04) from being rejected.
"$NVCC" -shared -o "$OUT" "${SRC[@]}" \
  -I csrc -I csrc/gemm -I csrc/blake3 -I csrc/tensor_hash -I "$CUTLASS_DIR" \
  -Xcompiler -fPIC -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda \
  --use_fast_math $GENCODE -O3 -DNDEBUG -DP40_NO_TORCH \
  -allow-unsupported-compiler ${EXTRA_NVCC_FLAGS:-}

echo "built $OUT"
