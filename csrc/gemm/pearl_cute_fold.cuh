#pragma once
// Fused CuTe Pearl kernel: int8 tensor-core GEMM + in-mainloop R-block transcript fold.
// Bit-exact with the DP4A reference; ~32.75 TH/s on RTX 4050 (Ada sm_89), vs the hand
// `ldm` kernel's 24.0. Multistage cp.async.cg + ldmatrix; <8,1,1> TiledMMA so each warp
// owns a 16-row band (16 contiguous 16x16 tiles), making the fold a direct in-register
// shfl_xor (no atomics/extra syncs). Namespaced so `using namespace cute` stays local.
//
// Needs the CUTLASS/CuTe include dir on the compile line (-I <cutlass>/include), which
// the p40cuda build (packaging/build_capi.*) and the bench already pass.
#include <cuda_runtime.h>
#include <cstdint>
#include <cute/tensor.hpp>

namespace pcute {
using namespace cute;

static const int HASH_ROT = 13, TLEN = 16;

template <class TiledMMA_, class GCopyA_, class GCopyB_,
          class SmemLayoutA_, class SmemLayoutB_, int bM, int bN, int bK, int Stages>
__global__ static void
cute_pearl_fold(int M, int N, int K, int R,
                const int8_t* __restrict__ Aptr, const int8_t* __restrict__ Bptr,
                uint32_t* __restrict__ transcript)
{
    Tensor mA = make_tensor(make_gmem_ptr(Aptr), make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr(Bptr), make_shape(N, K), make_stride(K, Int<1>{}));

    auto cta_tiler = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<Int<1>, X, Int<1>>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, Int<1>, Int<1>>{});
    Tensor cC = make_identity_tensor(make_shape(Int<bM>{}, Int<bN>{}));

    extern __shared__ char smem_raw[];
    SmemLayoutA_ sA_layout; SmemLayoutB_ sB_layout;
    int8_t* sAp = reinterpret_cast<int8_t*>(smem_raw);
    int8_t* sBp = sAp + cosize(sA_layout);
    Tensor sA = make_tensor(make_smem_ptr(sAp), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(sBp), sB_layout);

    constexpr int TILES = (bM/16) * (bN/16);
    __shared__ uint32_t sT[TILES * TLEN];
    for (int i = threadIdx.x; i < TILES*TLEN; i += blockDim.x) sT[i] = 0;

    GCopyA_ copyA; GCopyB_ copyB;
    auto thrA = copyA.get_slice(threadIdx.x);
    auto thrB = copyB.get_slice(threadIdx.x);
    Tensor tAgA = thrA.partition_S(gA);  Tensor tAsA = thrA.partition_D(sA);
    Tensor tBgB = thrB.partition_S(gB);  Tensor tBsB = thrB.partition_D(sB);

    TiledMMA_ tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA(_,_,_,0));
    Tensor tCrB = thr_mma.make_fragment_B(tCsB(_,_,_,0));
    Tensor tCrC = partition_fragment_C(tiled_mma, make_shape(Int<bM>{}, Int<bN>{}));
    Tensor tCcC = thr_mma.partition_C(cC);
    clear(tCrC);

    auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, int8_t>{}, tiled_mma);
    auto s2r_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, int8_t>{}, tiled_mma);
    auto s2rA = s2r_A.get_slice(threadIdx.x);
    auto s2rB = s2r_B.get_slice(threadIdx.x);
    Tensor tXsA = s2rA.partition_S(sA); Tensor tXrA = s2rA.retile_D(tCrA);
    Tensor tXsB = s2rB.partition_S(sB); Tensor tXrB = s2rB.retile_D(tCrB);

    const int k_tiles  = size<3>(tAgA);
    const int kt_per_R = R / bK;
    const int tiles_w  = N / 16;

    int wr = 0, ww = 0;
    CUTE_UNROLL
    for (int s = 0; s < Stages-1; ++s) {
        copy(copyA, tAgA(_,_,_,s), tAsA(_,_,_,s));
        copy(copyB, tBgB(_,_,_,s), tBsB(_,_,_,s));
        cp_async_fence(); ++ww;
    }

    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_async_wait<Stages-2>(); __syncthreads();
        copy(s2r_A, tXsA(_,_,_,wr), tXrA);
        copy(s2r_B, tXsB(_,_,_,wr), tXrB);
        int knext = kt + (Stages-1);
        if (knext < k_tiles) {
            copy(copyA, tAgA(_,_,_,knext), tAsA(_,_,_,ww));
            copy(copyB, tBgB(_,_,_,knext), tBsB(_,_,_,ww));
        }
        cp_async_fence();
        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
        ww = (ww+1) % Stages; wr = (wr+1) % Stages;

        if ((kt+1) % kt_per_R == 0) {     // R-block boundary: warp-owned-tile fold
            int rb = kt / kt_per_R;
            const int lane = threadIdx.x & 31;
            CUTE_UNROLL
            for (int tm = 0; tm < size<1>(tCrC); ++tm)
            CUTE_UNROLL
            for (int k = 0; k < size<2>(tCrC)/2; ++k) {
                uint32_t pv = 0;
                CUTE_UNROLL
                for (int v = 0; v < size<0>(tCrC); ++v) {
                    pv ^= (uint32_t)tCrC(v,tm,2*k); pv ^= (uint32_t)tCrC(v,tm,2*k+1);
                }
                CUTE_UNROLL
                for (int off = 16; off > 0; off >>= 1) pv ^= __shfl_xor_sync(0xffffffffu, pv, off);
                if (lane == 0) {
                    auto mn = tCcC(0, tm, 2*k);
                    int lt = (get<0>(mn) >> 4) * (bN/16) + (get<1>(mn) >> 4);
                    uint32_t prev = sT[lt*TLEN + rb];
                    sT[lt*TLEN + rb] = ((prev << HASH_ROT) | (prev >> (32-HASH_ROT))) ^ pv;
                }
            }
        }
    }

    __syncthreads();
    const int trow0 = blockIdx.x * (bM/16);
    const int tcol0 = blockIdx.y * (bN/16);
    for (int lt = threadIdx.x; lt < TILES; lt += blockDim.x) {
        int lr = lt / (bN/16), lc = lt % (bN/16);
        int gtile = (trow0 + lr) * tiles_w + (tcol0 + lc);
        CUTE_UNROLL
        for (int t = 0; t < TLEN; ++t) transcript[(size_t)gtile*TLEN + t] = sT[lt*TLEN + t];
    }
}

// Dispatch entry: A[m,k] row-major, Bt[n,k] row-major (= B col-major), transcript[(m/16)*(n/16)][16].
// Requires m % bM == 0 and n % bN == 0 (128x256). bit-exact with DP4A.
template <int bM = 128, int bN = 256, int bK = 32, int Stages = 3>
cudaError_t launch_cute_fold(const int8_t* A, const int8_t* Bt, int m, int n,
                             int k, int R, uint32_t* T, cudaStream_t stream) {
    using MMA_Atom_ = MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>;
    using TiledMMA_ = decltype(make_tiled_mma(MMA_Atom_{}, Layout<Shape<_8,_1,_1>>{}));
    using SwzA = decltype(composition(Swizzle<2,4,3>{}, Layout<Shape<Int<bM>,Int<bK>>, Stride<Int<bK>,Int<1>>>{}));
    using SmemLayoutA_ = decltype(tile_to_shape(SwzA{}, make_shape(Int<bM>{}, Int<bK>{}, Int<Stages>{})));
    using SwzB = decltype(composition(Swizzle<2,4,3>{}, Layout<Shape<Int<bN>,Int<bK>>, Stride<Int<bK>,Int<1>>>{}));
    using SmemLayoutB_ = decltype(tile_to_shape(SwzB{}, make_shape(Int<bN>{}, Int<bK>{}, Int<Stages>{})));
    using GCopy_ = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int8_t>{},
                              Layout<Shape<_128,_2>, Stride<_2,_1>>{}, Layout<Shape<_1,_16>>{}));
    auto kern = cute_pearl_fold<TiledMMA_,GCopy_,GCopy_,SmemLayoutA_,SmemLayoutB_,bM,bN,bK,Stages>;
    const int smem_bytes = (cosize(SmemLayoutA_{}) + cosize(SmemLayoutB_{})) * (int)sizeof(int8_t);
    static bool s_set = false;
    if (!s_set) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes); s_set = true; }
    dim3 grid(m/bM, n/bN), block(size(TiledMMA_{}));
    kern<<<grid, block, smem_bytes, stream>>>(m, n, k, R, A, Bt, T);
    return cudaGetLastError();
}

}  // namespace pcute
