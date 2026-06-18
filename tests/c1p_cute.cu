// C1p-1 — hand-written CuTe int8 GEMM, one step toward the fused Pearl kernel.
// Goal of THIS step: prove we can drive a CuTe TiledMMA (int8 m16n8k32) + swizzled
// smem + cp.async + ldmatrix and get a BIT-EXACT int32 C = A @ Bt^T vs a host int32
// reference. No Pearl fold yet (that's C1p-2) — this nails the TiledMMA + accumulator
// fragment layout we need for the per-16x16-tile fold.
//
// Build (from p40-pearl-gemm/):
//   nvcc -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_89 -cudart static -Xcompiler /MT \
//        -I "<cutlass>/include" -o tests\c1p_cute.exe tests\c1p_cute.cu
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>

using namespace cute;

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1; } } while(0)

static int8_t det8(long long idx, unsigned long long seed) {
    unsigned long long s = seed + (unsigned long long)idx * 0x9E3779B97F4A7C15ULL;
    s = s * 0xD2511F53CD9E8D57ULL; s ^= s >> 31; s *= 0x9E3779B9ull;
    return (int8_t)((int)(s & 0xFF) - 128);
}

// One CTA computes a bM x bN tile of C = A(M,K) @ B(N,K)^T  (A,B both K-major / row-major).
template <class TiledMMA_, class GmemTiledCopyA_, class GmemTiledCopyB_,
          class SmemLayoutA_, class SmemLayoutB_, int bM, int bN, int bK, int Stages>
__global__ static void
cute_i8gemm(int M, int N, int K,
            const int8_t* __restrict__ Aptr,
            const int8_t* __restrict__ Bptr,
            int32_t* __restrict__ Cptr)
{
    // ---- global tensors ----
    Tensor mA = make_tensor(make_gmem_ptr(Aptr), make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr(Bptr), make_shape(N, K), make_stride(K, Int<1>{}));
    Tensor mC = make_tensor(make_gmem_ptr(Cptr), make_shape(M, N), make_stride(N, Int<1>{}));

    auto cta_tiler = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<Int<1>, X, Int<1>>{});  // (bM,bK,k)
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, Int<1>, Int<1>>{}); // (bN,bK,k)
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<Int<1>, Int<1>, X>{});  // (bM,bN)

    // ---- shared memory (multistage ring) ----
    extern __shared__ char smem_raw[];
    SmemLayoutA_ sA_layout;
    SmemLayoutB_ sB_layout;
    int8_t* sAp = reinterpret_cast<int8_t*>(smem_raw);
    int8_t* sBp = sAp + cosize(sA_layout);
    Tensor sA = make_tensor(make_smem_ptr(sAp), sA_layout);  // (bM,bK,Stages)
    Tensor sB = make_tensor(make_smem_ptr(sBp), sB_layout);  // (bN,bK,Stages)

    // ---- gmem->smem tiled copy (cp.async) ----
    GmemTiledCopyA_ copyA;
    GmemTiledCopyB_ copyB;
    auto thrA = copyA.get_slice(threadIdx.x);
    auto thrB = copyB.get_slice(threadIdx.x);
    Tensor tAgA = thrA.partition_S(gA);   // (CPY,CPY_M,CPY_K,k)
    Tensor tAsA = thrA.partition_D(sA);   // (CPY,CPY_M,CPY_K,Stages)
    Tensor tBgB = thrB.partition_S(gB);
    Tensor tBsB = thrB.partition_D(sB);

    // ---- TiledMMA / register fragments ----
    TiledMMA_ tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCgC = thr_mma.partition_C(gC);                // (MMA,MMA_M,MMA_N)
    Tensor tCrC = thr_mma.make_fragment_C(tCgC);          // accumulator in regs
    Tensor tCsA = thr_mma.partition_A(sA);                // (MMA,MMA_M,MMA_K,Stages)
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA(_,_,_,0)); // reg frag A
    Tensor tCrB = thr_mma.make_fragment_B(tCsB(_,_,_,0));
    clear(tCrC);

    // smem->reg ldmatrix copies
    auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, int8_t>{}, tiled_mma);
    auto s2r_B = make_tiled_copy_B(Copy_Atom<SM75_U32x2_LDSM_N, int8_t>{}, tiled_mma);
    auto s2rA  = s2r_A.get_slice(threadIdx.x);
    auto s2rB  = s2r_B.get_slice(threadIdx.x);
    Tensor tXsA = s2rA.partition_S(sA);   // (CPY,MMA_M,MMA_K,Stages)
    Tensor tXrA = s2rA.retile_D(tCrA);
    Tensor tXsB = s2rB.partition_S(sB);
    Tensor tXrB = s2rB.retile_D(tCrB);

    const int k_tiles = size<3>(tAgA);

    // ---- prologue: fill Stages-1 stages ----
    int smem_pipe_write = 0, smem_pipe_read = 0;
    CUTE_UNROLL
    for (int s = 0; s < Stages - 1; ++s) {
        copy(copyA, tAgA(_,_,_,s), tAsA(_,_,_,s));
        copy(copyB, tBgB(_,_,_,s), tBsB(_,_,_,s));
        cp_async_fence();
        ++smem_pipe_write;
    }

    // ---- mainloop ----
    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_async_wait<Stages - 2>();
        __syncthreads();
        // load fragments for this stage
        copy(s2r_A, tXsA(_,_,_,smem_pipe_read), tXrA);
        copy(s2r_B, tXsB(_,_,_,smem_pipe_read), tXrB);
        // prefetch next global tile into the write stage
        int knext = kt + (Stages - 1);
        if (knext < k_tiles) {
            copy(copyA, tAgA(_,_,_,knext), tAsA(_,_,_,smem_pipe_write));
            copy(copyB, tBgB(_,_,_,knext), tBsB(_,_,_,smem_pipe_write));
        }
        cp_async_fence();
        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
        smem_pipe_write = (smem_pipe_write + 1) % Stages;
        smem_pipe_read  = (smem_pipe_read  + 1) % Stages;
    }

    // ---- epilogue: write C ----
    copy(tCrC, tCgC);
}

int main() {
    const int M = 128, N = 256, K = 1024;
    constexpr int bM = 128, bN = 256, bK = 32, Stages = 3;
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    printf("Device: %s (sm_%d%d)  %dx%dx%d  bK=%d stages=%d\n", p.name, p.major, p.minor, M,N,K, bK, Stages);

    std::vector<int8_t> hA((size_t)M*K), hBt((size_t)N*K);
    for (size_t i=0;i<hA.size();++i)  hA[i]  = det8((long long)i, 0x1111);
    for (size_t i=0;i<hBt.size();++i) hBt[i] = det8((long long)i, 0x2222);
    std::vector<int32_t> hRef((size_t)M*N, 0);
    for (int m=0;m<M;++m) for (int n=0;n<N;++n){ int32_t a=0; for(int k=0;k<K;++k) a += (int32_t)hA[(size_t)m*K+k]*(int32_t)hBt[(size_t)n*K+k]; hRef[(size_t)m*N+n]=a; }

    int8_t *dA,*dB; int32_t *dC;
    CK(cudaMalloc(&dA,hA.size())); CK(cudaMalloc(&dB,hBt.size())); CK(cudaMalloc(&dC,(size_t)M*N*4));
    CK(cudaMemcpy(dA,hA.data(),hA.size(),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB,hBt.data(),hBt.size(),cudaMemcpyHostToDevice));

    // TiledMMA: int8 m16n8k32 atom, 4x2 warps (=8) -> covers 64x16 per atom-pass; loop tiles the rest.
    using MMA_Atom_ = MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>;
    using TiledMMA_ = decltype(make_tiled_mma(MMA_Atom_{}, Layout<Shape<_4,_2,_1>>{}));

    // Swizzled smem layouts (bM/bN x bK, Stages), atom swizzle for conflict-free ldmatrix.
    using SwzA = decltype(composition(Swizzle<2,4,3>{}, Layout<Shape<Int<bM>,Int<bK>>, Stride<Int<bK>,Int<1>>>{}));
    using SmemLayoutA_ = decltype(tile_to_shape(SwzA{}, make_shape(Int<bM>{}, Int<bK>{}, Int<Stages>{})));
    using SwzB = decltype(composition(Swizzle<2,4,3>{}, Layout<Shape<Int<bN>,Int<bK>>, Stride<Int<bK>,Int<1>>>{}));
    using SmemLayoutB_ = decltype(tile_to_shape(SwzB{}, make_shape(Int<bN>{}, Int<bK>{}, Int<Stages>{})));

    // gmem->smem cp.async tiled copies (16B / 16 int8 per thread).
    using GCopyA_ = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int8_t>{},
                              Layout<Shape<_128,_2>, Stride<_2,_1>>{}, Layout<Shape<_1,_16>>{}));
    using GCopyB_ = GCopyA_;

    int smem_bytes = (cosize(SmemLayoutA_{}) + cosize(SmemLayoutB_{})) * (int)sizeof(int8_t);
    dim3 grid(M/bM, N/bN);
    dim3 block(size(TiledMMA_{}));
    cudaFuncSetAttribute(cute_i8gemm<TiledMMA_,GCopyA_,GCopyB_,SmemLayoutA_,SmemLayoutB_,bM,bN,bK,Stages>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cute_i8gemm<TiledMMA_,GCopyA_,GCopyB_,SmemLayoutA_,SmemLayoutB_,bM,bN,bK,Stages>
        <<<grid, block, smem_bytes>>>(M,N,K,dA,dB,dC);
    CK(cudaDeviceSynchronize());

    std::vector<int32_t> hC((size_t)M*N);
    CK(cudaMemcpy(hC.data(),dC,(size_t)M*N*4,cudaMemcpyDeviceToHost));
    long long diff=0; size_t fi=0;
    for (size_t i=0;i<hC.size();++i) if (hC[i]!=hRef[i]) { if(!diff) fi=i; ++diff; }
    printf("C1p-1: %s (%lld/%d differ)\n", diff==0?"BIT-EXACT PASS":"FAIL", diff, M*N);
    if (diff) printf("  first @ %zu: cute=%d ref=%d\n", fi, hC[fi], hRef[fi]);

    // ---- rough speed at the real region (4096^3). CAVEATS: UNTUNED (BK=32, no register
    // pipelining yet = C1p-3) and includes a full raw-C global write (~+7%) the fused
    // kernel will NOT do. Lower bound, not the real number. Hand kernel = 24.0 TH/s. ----
    {
        const int SM=4096, SN=4096, SK=4096;
        int8_t *bA,*bB; int32_t *bC;
        CK(cudaMalloc(&bA,(size_t)SM*SK)); CK(cudaMalloc(&bB,(size_t)SN*SK)); CK(cudaMalloc(&bC,(size_t)SM*SN*4));
        dim3 g(SM/bM, SN/bN), b(size(TiledMMA_{}));
        auto go=[&]{ cute_i8gemm<TiledMMA_,GCopyA_,GCopyB_,SmemLayoutA_,SmemLayoutB_,bM,bN,bK,Stages>
                     <<<g,b,smem_bytes>>>(SM,SN,SK,bA,bB,bC); };
        for(int i=0;i<40;i++) go(); CK(cudaDeviceSynchronize());
        cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0); for(int i=0;i<100;i++) go(); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float ms=0; cudaEventElapsedTime(&ms,e0,e1); ms/=100;
        double tiles=(double)(SM/16)*(SN/16);
        printf("C1p-1 speed: %.3f ms/region -> %.2f TH/s  (UNTUNED BK=32 + raw-C write; hand=24.0)\n",
               ms, tiles*1048576.0/(ms/1000.0)/1e12);
        cudaFree(bA);cudaFree(bB);cudaFree(bC);
    }
    return diff==0?0:1;
}
