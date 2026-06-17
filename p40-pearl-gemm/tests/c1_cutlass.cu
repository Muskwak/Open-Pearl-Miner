// C1 (correctness milestone) — prove CUTLASS can produce the Pearl TRANSCRIPT bit-exact.
//
// Strategy: the Pearl transcript word t for a 16x16 output tile is the XOR-fold of the
// CUMULATIVE int32 accumulator after R-block t (continuous accumulation, R=256 k each,
// T = k/R R-blocks). Here we get the 16 cumulative accumulators the *slow* but obviously
// correct way: 16 accumulating int8 GEMMs (each a k=256 slice, beta=1 into a persistent
// C), folding C -> transcript[t] after each. This is NOT the perf path (full C in global,
// 16 launches) — it validates the fold semantics + the CUTLASS<->Pearl bridge so the
// fused in-mainloop kernel (next) only has to get the CuTe mainloop right, not the fold.
//
// Gate: transcript must match the DP4A reference (pearl_gemm_only) bit-exactly.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1; } } while(0)

#include "../csrc/gemm/pearl_gemm_only_sm61.cu"   // launch_pearl_gemm_only (DP4A reference)

static const int HASH_ROT = 13;

__global__ void fill_det(int8_t* buf, long long numel, unsigned long long seed) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    unsigned long long s = seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL;
    s = s * 0xD2511F53CD9E8D57ULL; s ^= s >> 31; s *= 0x9E3779B9u;
    buf[idx] = (int8_t)((int)(s & 0xFF) - 128);
}

// One thread per 16x16 output tile: lx = XOR of the tile's 256 cumulative int32 acc values;
// transcript[tile*T_LEN + t] = rotl(prev,13) ^ lx. (T_LEN=16; for k=4096 each slot is
// written once so rotl acts on 0 — implemented generally anyway.)
__global__ void pearl_fold(const int32_t* C, int M, int N, int t, uint32_t* transcript) {
    int tiles_w = N / 16, tiles_h = M / 16;
    int tile = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile >= tiles_w * tiles_h) return;
    int ti = tile / tiles_w, tj = tile % tiles_w;
    uint32_t lx = 0;
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
            lx ^= (uint32_t)C[(size_t)(ti * 16 + i) * N + (tj * 16 + j)];
    uint32_t* s = &transcript[(size_t)tile * 16 + t];
    *s = ((*s << HASH_ROT) | (*s >> (32 - HASH_ROT))) ^ lx;
}

int main() {
    const int M = 256, N = 256, K = 4096, R = 256;
    const int T = K / R;                       // 16 R-blocks
    const int tiles = (M / 16) * (N / 16);
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("Device: %s (sm_%d%d)  M=%d N=%d K=%d R=%d\n", p.name, p.major, p.minor, M, N, K, R);

    int8_t *dA, *dB; CK(cudaMalloc(&dA, (size_t)M * K)); CK(cudaMalloc(&dB, (size_t)N * K));
    fill_det<<<(unsigned)(((size_t)M * K + 255) / 256), 256>>>(dA, (long long)M * K, 0x1111);
    fill_det<<<(unsigned)(((size_t)N * K + 255) / 256), 256>>>(dB, (long long)N * K, 0x2222);
    CK(cudaDeviceSynchronize());

    // DP4A reference transcript
    uint32_t *dTref; CK(cudaMalloc(&dTref, (size_t)tiles * 16 * 4)); CK(cudaMemset(dTref, 0, (size_t)tiles * 16 * 4));
    launch_pearl_gemm_only(dA, dB, M, N, K, R, dTref, 1, 0);
    CK(cudaDeviceSynchronize());

    // CUTLASS path: persistent C, 16 accumulating GEMMs + fold after each
    int32_t *dC; CK(cudaMalloc(&dC, (size_t)M * N * 4)); CK(cudaMemset(dC, 0, (size_t)M * N * 4));
    uint32_t *dTcut; CK(cudaMalloc(&dTcut, (size_t)tiles * 16 * 4)); CK(cudaMemset(dTcut, 0, (size_t)tiles * 16 * 4));

    using Gemm = cutlass::gemm::device::Gemm<
        int8_t,  cutlass::layout::RowMajor,
        int8_t,  cutlass::layout::ColumnMajor,
        int32_t, cutlass::layout::RowMajor,
        int32_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80>;
    Gemm gemm_op;

    for (int t = 0; t < T; ++t) {
        // k-slice [t*R, t*R+R); A[:,slice] stride K, Bt[:,slice] stride K
        Gemm::Arguments args(
            {M, N, R},
            {dA + (size_t)t * R, cutlass::layout::RowMajor(K)},
            {dB + (size_t)t * R, cutlass::layout::ColumnMajor(K)},
            {dC, cutlass::layout::RowMajor(N)},
            {dC, cutlass::layout::RowMajor(N)},
            {1, 1});                              // alpha=1, beta=1 -> C += A_slice @ B_slice
        cutlass::Status st = gemm_op(args);
        if (st != cutlass::Status::kSuccess) {
            printf("C1: CUTLASS GEMM r=%d FAILED status=%d (%s)\n", t, (int)st, cutlassGetStatusString(st));
            return 1;
        }
        pearl_fold<<<(unsigned)((tiles + 127) / 128), 128>>>(dC, M, N, t, dTcut);
    }
    CK(cudaDeviceSynchronize());

    std::vector<uint32_t> hRef((size_t)tiles * 16), hCut((size_t)tiles * 16);
    CK(cudaMemcpy(hRef.data(), dTref, (size_t)tiles * 16 * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hCut.data(), dTcut, (size_t)tiles * 16 * 4, cudaMemcpyDeviceToHost));
    long long diff = 0; size_t firsti = 0;
    for (size_t i = 0; i < hRef.size(); ++i)
        if (hRef[i] != hCut[i]) { if (diff == 0) firsti = i; ++diff; }
    printf("C1: %s (%lld/%zu transcript words differ) -- CUTLASS 16xGEMM+fold vs DP4A reference\n",
           diff == 0 ? "BIT-EXACT PASS" : "FAIL", diff, hRef.size());
    if (diff) printf("    first mismatch idx %zu: cutlass=%08x ref=%08x\n", firsti, hCut[firsti], hRef[firsti]);
    cudaFree(dA); cudaFree(dB); cudaFree(dTref); cudaFree(dC); cudaFree(dTcut);
    return diff == 0 ? 0 : 1;
}
