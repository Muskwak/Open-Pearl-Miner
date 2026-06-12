#!/usr/bin/env bash
# Build the torch-free CUDA shared library (p40cuda.dll / libp40cuda.so).
# Links only the CUDA runtime — no torch, no pybind. Driven from Python via ctypes.
#
# Run from p40-pearl-gemm/ :  bash packaging/build_capi.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CUTLASS_DIR:=C:/Users/ADMIN/audits/aphrodite-engine/.deps/cutlass-src/include}"
NVCC="${NVCC:-nvcc}"

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
)

"$NVCC" -shared -o "$OUT" "${SRC[@]}" \
  -I csrc -I csrc/gemm -I csrc/blake3 -I csrc/tensor_hash -I "$CUTLASS_DIR" \
  -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math \
  -gencode arch=compute_61,code=sm_61 -O3 -DNDEBUG -DP40_NO_TORCH

echo "built $OUT"
