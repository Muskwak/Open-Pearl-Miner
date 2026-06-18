#pragma once
// Tensor-core (IMMA) noise-apply GEMM for Ampere/Ada: out[M,K] = clamp_i8(Z + X @ Y^T over R).
// The DP4A version (noise_gemm_sm61.cu) runs on the CUDA cores and is ~3 ms/region on the 4050 —
// slower than the main tensor-core GEMM itself, despite 1/16th the FLOPs. This is the same TN int8
// GEMM (contract R, A=X[M,R], B=Y[K,R]) so IMMA s8*s8->s32 == DP4A int32 bit-exactly; the only extra
// work is the +Z bias and the int8 clamp in the epilogue. Multistage cp.async + ldmatrix, same
// scaffolding as pearl_cute_fold. Namespaced so `using namespace cute` stays local.
#include <cuda_runtime.h>
#include <cstdint>
#include <cute/tensor.hpp>

namespace pcute {
using namespace cute;

template <class TiledMMA_, class GCopyA_, class GCopyB_,
          class SmemLayoutA_, class SmemLayoutB_, int bM, int bN, int bK, int Stages>
__global__ static void
cute_noise_gemm(int M, int N, int K,
                const int8_t* __restrict__ Xptr, const int8_t* __restrict__ Yptr,
                const int8_t* __restrict__ Zptr, int8_t* __restrict__ Optr)
{
    // GEMM dims: M rows, N output-cols (the reference's K), K contraction (the reference's R).
    Tensor mA = make_tensor(make_gmem_ptr(Xptr), make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr(Yptr), make_shape(N, K), make_stride(K, Int<1>{}));
    Tensor mZ = make_tensor(make_gmem_ptr(Zptr), make_shape(M, N), make_stride(N, Int<1>{}));
    Tensor mO = make_tensor(make_gmem_ptr(Optr), make_shape(M, N), make_stride(N, Int<1>{}));

    auto cta_tiler = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<Int<1>, X, Int<1>>{});  // (bM,bK,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, Int<1>, Int<1>>{}); // (bN,bK,k)
    Tensor gZ = local_tile(mZ, make_shape(Int<bM>{}, Int<bN>{}), make_coord(blockIdx.x, blockIdx.y));
    Tensor gO = local_tile(mO, make_shape(Int<bM>{}, Int<bN>{}), make_coord(blockIdx.x, blockIdx.y));

    extern __shared__ char smem_raw[];
    SmemLayoutA_ sA_layout; SmemLayoutB_ sB_layout;
    int8_t* sAp = reinterpret_cast<int8_t*>(smem_raw);
    int8_t* sBp = sAp + cosize(sA_layout);
    Tensor sA = make_tensor(make_smem_ptr(sAp), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(sBp), sB_layout);

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
    Tensor tCgZ = thr_mma.partition_C(gZ);   // bias, same layout as the accumulator
    Tensor tCgO = thr_mma.partition_C(gO);   // int8 output
    clear(tCrC);

    auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, int8_t>{}, tiled_mma);
    auto s2r_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, int8_t>{}, tiled_mma);
    auto s2rA = s2r_A.get_slice(threadIdx.x);
    auto s2rB = s2r_B.get_slice(threadIdx.x);
    Tensor tXsA = s2rA.partition_S(sA); Tensor tXrA = s2rA.retile_D(tCrA);
    Tensor tXsB = s2rB.partition_S(sB); Tensor tXrB = s2rB.retile_D(tCrB);

    const int k_tiles = size<3>(tAgA);
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
    }

    // epilogue: out = clamp_i8(Z + acc), elementwise over the per-thread C fragment
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        int v = (int)tCgZ(i) + tCrC(i);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        tCgO(i) = (int8_t)v;
    }
}

// out[M,K] = clamp(Z + X @ Y^T over R), tensor-core. M%bM==0, K%bN==0, R%bK==0. bit-exact with DP4A.
template <int bM = 128, int bN = 256, int bK = 32, int Stages = 3>
cudaError_t launch_noise_gemm_tc(const int8_t* X, const int8_t* Y, const int8_t* Z,
                                 int8_t* out, int M, int K, int R, cudaStream_t stream) {
    using MMA_Atom_ = MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>;
    using TiledMMA_ = decltype(make_tiled_mma(MMA_Atom_{}, Layout<Shape<_8,_1,_1>>{}));
    using SwzA = decltype(composition(Swizzle<2,4,3>{}, Layout<Shape<Int<bM>,Int<bK>>, Stride<Int<bK>,Int<1>>>{}));
    using SmemLayoutA_ = decltype(tile_to_shape(SwzA{}, make_shape(Int<bM>{}, Int<bK>{}, Int<Stages>{})));
    using SwzB = decltype(composition(Swizzle<2,4,3>{}, Layout<Shape<Int<bN>,Int<bK>>, Stride<Int<bK>,Int<1>>>{}));
    using SmemLayoutB_ = decltype(tile_to_shape(SwzB{}, make_shape(Int<bN>{}, Int<bK>{}, Int<Stages>{})));
    using GCopy_ = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int8_t>{},
                              Layout<Shape<_128,_2>, Stride<_2,_1>>{}, Layout<Shape<_1,_16>>{}));
    auto kern = cute_noise_gemm<TiledMMA_,GCopy_,GCopy_,SmemLayoutA_,SmemLayoutB_,bM,bN,bK,Stages>;
    const int smem_bytes = (cosize(SmemLayoutA_{}) + cosize(SmemLayoutB_{})) * (int)sizeof(int8_t);
    static bool s_set = false;
    if (!s_set) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes); s_set = true; }
    dim3 grid(M/bM, K/bN), block(size(TiledMMA_{}));
    kern<<<grid, block, smem_bytes, stream>>>(M, K, R, X, Y, Z, out);
    return cudaGetLastError();
}

}  // namespace pcute
