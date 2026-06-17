// C0 — prove CUTLASS compiles for sm_89 AND its int8 IMMA GEMM is bit-exact with an
// exact int32 reference. This validates the toolchain + the "bit-exact is free"
// assumption (IMMA s8*s8->s32 == exact int8 dot product, order-independent) BEFORE any
// of the Pearl-specific fold/pipeline work (C1+). No Pearl transcript here — just the
// raw GEMM accumulator C = A @ Bt^T.
//
// Build (from p40-pearl-gemm/):
//   nvcc -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_89 -cudart static -Xcompiler /MT \
//        -I "<cutlass>/include" -o tests\c0_cutlass.exe tests\c0_cutlass.cu
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1; } } while(0)

// Same deterministic fill the bench uses (int8 in [-128,127]).
static int8_t det8(long long idx, unsigned long long seed) {
    unsigned long long s = seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL;
    s = s * 0xD2511F53CD9E8D57ULL; s ^= s >> 31; s *= 0x9E3779B9ull;
    return (int8_t)((int)(s & 0xFF) - 128);
}

int main() {
    const int M = 128, N = 256, K = 1024;   // one 128x256 CTA tile; K multiple of 32
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("Device: %s (sm_%d%d)\n", p.name, p.major, p.minor);

    std::vector<int8_t> hA((size_t)M * K), hBt((size_t)N * K);
    for (size_t i = 0; i < hA.size();  ++i) hA[i]  = det8((long long)i, 0x1111);
    for (size_t i = 0; i < hBt.size(); ++i) hBt[i] = det8((long long)i, 0x2222);

    // exact int32 reference: Cref[m][n] = sum_k A[m][k] * Bt[n][k]
    std::vector<int32_t> hCref((size_t)M * N, 0);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k)
                acc += (int32_t)hA[(size_t)m * K + k] * (int32_t)hBt[(size_t)n * K + k];
            hCref[(size_t)m * N + n] = acc;
        }

    int8_t *dA = nullptr, *dB = nullptr; int32_t *dD = nullptr;
    CK(cudaMalloc(&dA, hA.size()));
    CK(cudaMalloc(&dB, hBt.size()));
    CK(cudaMalloc(&dD, (size_t)M * N * sizeof(int32_t)));
    CK(cudaMemcpy(dA, hA.data(),  hA.size(),  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hBt.data(), hBt.size(), cudaMemcpyHostToDevice));

    // D = A[M,K] (row-major) @ B[K,N] (col-major == Bt[N,K] row-major) -> int32, row-major.
    using Gemm = cutlass::gemm::device::Gemm<
        int8_t,  cutlass::layout::RowMajor,
        int8_t,  cutlass::layout::ColumnMajor,
        int32_t, cutlass::layout::RowMajor,
        int32_t,                              // accumulator
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80>;

    Gemm gemm_op;
    Gemm::Arguments args(
        {M, N, K},
        {dA, cutlass::layout::RowMajor(K)},
        {dB, cutlass::layout::ColumnMajor(K)},
        {dD, cutlass::layout::RowMajor(N)},
        {dD, cutlass::layout::RowMajor(N)},
        {1, 0});                              // alpha=1, beta=0 (int32)

    cutlass::Status st = gemm_op(args);
    if (st != cutlass::Status::kSuccess) {
        printf("C0: CUTLASS launch FAILED, status=%d (%s)\n", (int)st,
               cutlassGetStatusString(st));
        return 1;
    }
    CK(cudaDeviceSynchronize());

    std::vector<int32_t> hD((size_t)M * N);
    CK(cudaMemcpy(hD.data(), dD, (size_t)M * N * sizeof(int32_t), cudaMemcpyDeviceToHost));

    long long diff = 0; size_t firsti = 0;
    for (size_t i = 0; i < hD.size(); ++i)
        if (hD[i] != hCref[i]) { if (diff == 0) firsti = i; ++diff; }

    printf("C0: %s (%lld/%d words differ) -- CUTLASS int8 IMMA vs exact int32 reference @ %dx%dx%d\n",
           diff == 0 ? "BIT-EXACT PASS" : "FAIL", diff, M * N, M, N, K);
    if (diff) printf("    first mismatch idx %zu: cutlass=%d ref=%d\n",
                     firsti, hD[firsti], hCref[firsti]);
    cudaFree(dA); cudaFree(dB); cudaFree(dD);
    return diff == 0 ? 0 : 1;
}
