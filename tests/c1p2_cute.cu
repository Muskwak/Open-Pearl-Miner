// C1p-2 — fused CuTe Pearl kernel: int8 GEMM + in-mainloop R-block transcript fold.
// Continuous int32 accumulation; at each R-boundary (every R/bK k-tiles) XOR-fold the
// running accumulator into a per-16x16-tile transcript (rotl-13), no C write. Gate:
// transcript == DP4A reference (pearl_gemm_only), bit-exact.
//
// Fold is coordinate-driven (robust to TiledMMA layout): partition_C on an identity
// tensor gives each accumulator element's (m,n) -> 16x16 tile; XOR into a per-CTA-tile
// smem partial via atomicXor, then rotl-13 into the persistent transcript smem.
//
// Build (from p40-pearl-gemm/):
//   nvcc -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_89 -cudart static -Xcompiler /MT \
//        -I csrc -I "<cutlass>/include" -o tests\c1p2_cute.exe tests\c1p2_cute.cu
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

using namespace cute;

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1; } } while(0)

#include "../csrc/gemm/pearl_gemm_only_sm61.cu"   // launch_pearl_gemm_only (DP4A ref)

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
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<Int<1>, X, Int<1>>{});  // (bM,bK,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, Int<1>, Int<1>>{}); // (bN,bK,k)
    Tensor cC = make_identity_tensor(make_shape(Int<bM>{}, Int<bN>{}));           // (m,n) coords

    extern __shared__ char smem_raw[];
    SmemLayoutA_ sA_layout; SmemLayoutB_ sB_layout;
    int8_t* sAp = reinterpret_cast<int8_t*>(smem_raw);
    int8_t* sBp = sAp + cosize(sA_layout);
    Tensor sA = make_tensor(make_smem_ptr(sAp), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(sBp), sB_layout);

    // persistent per-CTA transcript (bM/16 * bN/16 tiles, TLEN words) + per-fold partial
    constexpr int TILES = (bM/16) * (bN/16);   // 8*16 = 128
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
    Tensor tCcC = thr_mma.partition_C(cC);   // each elem -> (m,n) in the CTA tile
    clear(tCrC);

    auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, int8_t>{}, tiled_mma);
    auto s2r_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, int8_t>{}, tiled_mma);
    auto s2rA = s2r_A.get_slice(threadIdx.x);
    auto s2rB = s2r_B.get_slice(threadIdx.x);
    Tensor tXsA = s2rA.partition_S(sA); Tensor tXrA = s2rA.retile_D(tCrA);
    Tensor tXsB = s2rB.partition_S(sB); Tensor tXrB = s2rB.retile_D(tCrB);

    const int k_tiles  = size<3>(tAgA);
    const int kt_per_R = R / bK;                 // k-tiles per R-block (8 for R=256,bK=32)
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

        // ---- R-block boundary: warp-owned-tile fold (direct in-register, no atomics/syncs) ----
        // With <8,1,1> each warp owns a full 16-row band = 16 contiguous 16x16 tiles (N never
        // split), and atom-N (2k,2k+1) are the two 8-col halves of tile k. XOR the lane's 8
        // int32, shfl across the warp, lane0 rotl-13 folds into the tile's transcript slot.
        if ((kt+1) % kt_per_R == 0) {
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

    __syncthreads();   // all warps' folds visible before the cross-warp transcript read
    // ---- epilogue: write transcript to global ----
    const int trow0 = blockIdx.x * (bM/16);
    const int tcol0 = blockIdx.y * (bN/16);
    for (int lt = threadIdx.x; lt < TILES; lt += blockDim.x) {
        int lr = lt / (bN/16), lc = lt % (bN/16);
        int gtile = (trow0 + lr) * tiles_w + (tcol0 + lc);
        CUTE_UNROLL
        for (int t = 0; t < TLEN; ++t) transcript[(size_t)gtile*TLEN + t] = sT[lt*TLEN + t];
    }
}

int main() {
    const int M=256, N=256, K=4096, R=256;
    constexpr int bM=128, bN=256, bK=32, Stages=3;
    const int tiles = (M/16)*(N/16);
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("Device: %s (sm_%d%d)  %dx%dx%d R=%d  bK=%d st=%d\n", p.name,p.major,p.minor,M,N,K,R,bK,Stages);

    int8_t *dA,*dB; CK(cudaMalloc(&dA,(size_t)M*K)); CK(cudaMalloc(&dB,(size_t)N*K));
    { std::vector<int8_t> hA((size_t)M*K), hB((size_t)N*K);
      auto det=[](long long i,unsigned long long s){ s+= (unsigned long long)i*0x9E3779B97F4A7C15ULL; s*=0xD2511F53CD9E8D57ULL; s^=s>>31; s*=0x9E3779B9ull; return (int8_t)((int)(s&0xFF)-128); };
      for(size_t i=0;i<hA.size();++i) hA[i]=det((long long)i,0x1111);
      for(size_t i=0;i<hB.size();++i) hB[i]=det((long long)i,0x2222);
      CK(cudaMemcpy(dA,hA.data(),hA.size(),cudaMemcpyHostToDevice));
      CK(cudaMemcpy(dB,hB.data(),hB.size(),cudaMemcpyHostToDevice)); }

    uint32_t *dTref,*dTcut;
    CK(cudaMalloc(&dTref,(size_t)tiles*TLEN*4)); CK(cudaMalloc(&dTcut,(size_t)tiles*TLEN*4));
    CK(cudaMemset(dTref,0,(size_t)tiles*TLEN*4)); CK(cudaMemset(dTcut,0,(size_t)tiles*TLEN*4));
    launch_pearl_gemm_only(dA,dB,M,N,K,R,dTref,1,0); CK(cudaDeviceSynchronize());

    using MMA_Atom_ = MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>;
    using TiledMMA_ = decltype(make_tiled_mma(MMA_Atom_{}, Layout<Shape<_8,_1,_1>>{}));
    using SwzA = decltype(composition(Swizzle<2,4,3>{}, Layout<Shape<Int<bM>,Int<bK>>, Stride<Int<bK>,Int<1>>>{}));
    using SmemLayoutA_ = decltype(tile_to_shape(SwzA{}, make_shape(Int<bM>{}, Int<bK>{}, Int<Stages>{})));
    using SwzB = decltype(composition(Swizzle<2,4,3>{}, Layout<Shape<Int<bN>,Int<bK>>, Stride<Int<bK>,Int<1>>>{}));
    using SmemLayoutB_ = decltype(tile_to_shape(SwzB{}, make_shape(Int<bN>{}, Int<bK>{}, Int<Stages>{})));
    using GCopyA_ = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int8_t>{},
                              Layout<Shape<_128,_2>, Stride<_2,_1>>{}, Layout<Shape<_1,_16>>{}));
    using GCopyB_ = GCopyA_;

    int smem_bytes = (cosize(SmemLayoutA_{}) + cosize(SmemLayoutB_{})) * (int)sizeof(int8_t);
    auto kern = cute_pearl_fold<TiledMMA_,GCopyA_,GCopyB_,SmemLayoutA_,SmemLayoutB_,bM,bN,bK,Stages>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    dim3 grid(M/bM, N/bN), block(size(TiledMMA_{}));
    kern<<<grid, block, smem_bytes>>>(M,N,K,R,dA,dB,dTcut);
    CK(cudaDeviceSynchronize());

    std::vector<uint32_t> hRef((size_t)tiles*TLEN), hCut((size_t)tiles*TLEN);
    CK(cudaMemcpy(hRef.data(),dTref,(size_t)tiles*TLEN*4,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hCut.data(),dTcut,(size_t)tiles*TLEN*4,cudaMemcpyDeviceToHost));
    long long diff=0; size_t fi=0;
    for(size_t i=0;i<hRef.size();++i) if(hRef[i]!=hCut[i]){ if(!diff)fi=i; ++diff; }
    printf("C1p-2: %s (%lld/%zu transcript words differ)\n", diff==0?"BIT-EXACT PASS":"FAIL", diff, hRef.size());
    if(diff) printf("  first @ %zu: cute=%08x ref=%08x\n", fi, hCut[fi], hRef[fi]);

    // ---- real fused throughput at the region (4096^3 R256): GEMM+fold, no C-write ----
    {
        const int SM=4096, SN=4096, SK=4096, SR=256;
        int8_t *bA,*bB; uint32_t *bT;
        CK(cudaMalloc(&bA,(size_t)SM*SK)); CK(cudaMalloc(&bB,(size_t)SN*SK));
        CK(cudaMalloc(&bT,(size_t)(SM/16)*(SN/16)*TLEN*4));
        dim3 g(SM/bM, SN/bN), b(size(TiledMMA_{}));
        auto go=[&]{ kern<<<g,b,smem_bytes>>>(SM,SN,SK,SR,bA,bB,bT); };
        for(int i=0;i<40;i++) go(); CK(cudaDeviceSynchronize());
        cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0); for(int i=0;i<100;i++) go(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=100;
        double tiles=(double)(SM/16)*(SN/16);
        printf("C1p-2 fused: %.3f ms/region -> %.2f TH/s  (BK=32 untuned, GEMM+fold, NO C-write; hand=24.0)\n",
               ms, tiles*1048576.0/(ms/1000.0)/1e12);
        cudaFree(bA);cudaFree(bB);cudaFree(bT);
    }
    return diff==0?0:1;
}
