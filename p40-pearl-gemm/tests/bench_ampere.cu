// bench_ampere.cu — benchmark + bit-exact validate the Ampere/Ada TC Pearl kernel.
//
// Build (Windows, from p40-pearl-gemm/):
//   nvcc -O3 -std=c++17 -arch=sm_89 -cudart static -o tests\bench_ampere.exe tests\bench_ampere.cu
// Run:
//   bench_ampere.exe [m n k R iters]    (defaults: 4096 4096 4096 256 50)
//
// Reports TC GEMM throughput in TH/s (Pearl difficulty: 1 tile = 16*16*4096
// = 2^20 "hashes"), the DP4A reference throughput, effective INT8 TOPS, and a
// bit-exact transcript check (TC vs DP4A) at a small config.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err), err); \
        exit(1); \
    } \
} while(0)

#include "../csrc/gemm/pearl_gemm_only_sm61.cu"
#include "../csrc/gemm/pearl_ampere_tc.cu"
#include "../csrc/gemm/pearl_blake3_sm61.cu"   // launch_pearl_blake3 (needs -I csrc)

__global__ void fill_det(int8_t* buf, int64_t numel, uint64_t seed) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    uint64_t s = seed + idx * 0x9E3779B97F4A7C15ULL;
    s = s * 0xD2511F53CD9E8D57ULL; s ^= s >> 31; s *= 0x9E3779B9;
    buf[idx] = (int8_t)((s & 0xFF) - 128);
}

static double ths_from(double tiles, double ms) {
    // hashes = tiles * 16*16*4096 = tiles * 2^20 ;  TH/s = hashes/s / 1e12
    return tiles * 1048576.0 / (ms / 1000.0) / 1e12;
}

// Run launch_pearl_ampere `iters` times, return avg ms.
static double time_tc(const int8_t* A, const int8_t* Bt, int m, int n, int k,
                      int R, uint32_t* T, int iters) {
    cudaEvent_t a, b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    for (int i = 0; i < 60; ++i) launch_pearl_ampere(A, Bt, m, n, k, R, T, 0); // warmup (ramp GPU clocks)
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch_pearl_ampere(A, Bt, m, n, k, R, T, 0);
    CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / iters;
}

// Launch a specific compile-time config directly (bypassing the dispatcher), so
// we can sweep block/warp/stage/minblocks without editing the kernel.
template<int BM,int BN,int WM,int WN,int STG,int MNB>
static double time_cfg(const int8_t* A, const int8_t* Bt, int m, int n, int k,
                       int R, uint32_t* T, int iters) {
    dim3 block(WM*WN*32);
    dim3 grid((unsigned)((m/BM)*(n/BN)));
    auto go = [&]{ pearl_ampere_fused_kernel<BM,BN,32,WM,WN,STG,MNB>
                   <<<grid, block, 0, 0>>>(A, Bt, n, k, R, T); };
    for (int i=0;i<5;i++) go();
    if (cudaDeviceSynchronize()!=cudaSuccess) return -1.0;
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i=0;i<iters;i++) go();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    if (cudaGetLastError()!=cudaSuccess) return -1.0;
    return (double)ms/iters;
}

static double time_dp4a(const int8_t* A, const int8_t* Bt, int m, int n, int k,
                        int R, uint32_t* T, int iters) {
    cudaEvent_t a, b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    for (int i = 0; i < 3; ++i) launch_pearl_gemm_only(A, Bt, m, n, k, R, T, 1, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) launch_pearl_gemm_only(A, Bt, m, n, k, R, T, 1, 0);
    CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / iters;
}

int main(int argc, char** argv) {
    int m = 4096, n = 4096, k = 4096, R = 256, iters = 50;
    if (argc >= 6) { m=atoi(argv[1]); n=atoi(argv[2]); k=atoi(argv[3]); R=atoi(argv[4]); iters=atoi(argv[5]); }

    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (sm_%d%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // ---------- prof mode: one clean dispatcher launch for ncu ----------
    // Usage: bench_ampere.exe prof [m] [n]   (defaults 4096x4096x4096 R256)
    // The ONLY tensor-core kernel launched here is the real-config wide kernel,
    // so `ncu -k pearl_ampere_wide_kernel -c 1` profiles exactly it (1024 blocks).
    if (argc >= 2 && (strcmp(argv[1], "prof") == 0 || strcmp(argv[1], "profldm") == 0)) {
        const bool use_ldm = (strcmp(argv[1], "profldm") == 0);
        int pm = (argc>=3?atoi(argv[2]):4096), pn = (argc>=4?atoi(argv[3]):4096), pk = 4096, pR = 256;
        size_t szA=(size_t)pm*pk, szBt=(size_t)pn*pk, szT=(size_t)(pm/16)*(pn/16)*16*4;
        int8_t *A,*Bt; uint32_t *T;
        CUDA_CHECK(cudaMalloc(&A,szA)); CUDA_CHECK(cudaMalloc(&Bt,szBt)); CUDA_CHECK(cudaMalloc(&T,szT));
        int thr=256;
        fill_det<<<(unsigned)((szA+thr-1)/thr),thr>>>(A,szA,0x1111);
        fill_det<<<(unsigned)((szBt+thr-1)/thr),thr>>>(Bt,szBt,0x2222);
        CUDA_CHECK(cudaMemset(T,0,szT)); CUDA_CHECK(cudaDeviceSynchronize());
        if (use_ldm) launch_ldm<64,256,4,1,16,2,1>(A,Bt,pm,pn,pk,pR,T,0);
        else         launch_pearl_ampere(A,Bt,pm,pn,pk,pR,T,0);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("prof%s: launched @ %dx%dx%d R=%d\n", use_ldm?"ldm":"", pm,pn,pk,pR);
        cudaFree(A);cudaFree(Bt);cudaFree(T);
        return 0;
    }

    // ---------- bit-exact correctness (small grid, real R) ----------
    {
        const int cm = 256, cn = 256, ck = (k >= 4096 ? 4096 : k), cR = R;
        const int tiles = (cm/16)*(cn/16);
        size_t szA=(size_t)cm*ck, szBt=(size_t)cn*ck, szT=(size_t)tiles*16*4;
        int8_t *A,*Bt; uint32_t *Tp,*Ta;
        CUDA_CHECK(cudaMalloc(&A,szA)); CUDA_CHECK(cudaMalloc(&Bt,szBt));
        CUDA_CHECK(cudaMalloc(&Tp,szT)); CUDA_CHECK(cudaMalloc(&Ta,szT));
        int thr=256;
        fill_det<<<(szA+thr-1)/thr,thr>>>(A,szA,0x12345678);
        fill_det<<<(szBt+thr-1)/thr,thr>>>(Bt,szBt,0x87654321);
        CUDA_CHECK(cudaMemset(Tp,0,szT)); CUDA_CHECK(cudaMemset(Ta,0,szT));
        CUDA_CHECK(cudaDeviceSynchronize());
        launch_pearl_gemm_only(A,Bt,cm,cn,ck,cR,Tp,1,0);
        cudaError_t ae=launch_pearl_ampere(A,Bt,cm,cn,ck,cR,Ta,0);
        if (ae!=cudaSuccess){fprintf(stderr,"TC launch failed: %s\n",cudaGetErrorString(ae));return 1;}
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t *hp=(uint32_t*)malloc(szT),*ha=(uint32_t*)malloc(szT);
        CUDA_CHECK(cudaMemcpy(hp,Tp,szT,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ha,Ta,szT,cudaMemcpyDeviceToHost));
        int diff=0; for(int i=0;i<tiles*16;i++) if(hp[i]!=ha[i]) diff++;
        printf("Correctness @ %dx%dx%d R=%d: %s (%d/%d words differ)\n",
               cm,cn,ck,cR, diff==0?"BIT-EXACT PASS":"FAIL", diff, tiles*16);
        cudaFree(A);cudaFree(Bt);cudaFree(Tp);cudaFree(Ta);free(hp);free(ha);
    }

    // ---------- throughput @ real region config ----------
    const double tiles = (double)(m/16) * (n/16);
    size_t szA=(size_t)m*k, szBt=(size_t)n*k, szT=(size_t)(m/16)*(n/16)*16*4;
    int8_t *A,*Bt; uint32_t *T;
    CUDA_CHECK(cudaMalloc(&A,szA)); CUDA_CHECK(cudaMalloc(&Bt,szBt)); CUDA_CHECK(cudaMalloc(&T,szT));
    int thr=256;
    fill_det<<<(unsigned)((szA+thr-1)/thr),thr>>>(A,szA,0x1111);
    fill_det<<<(unsigned)((szBt+thr-1)/thr),thr>>>(Bt,szBt,0x2222);
    CUDA_CHECK(cudaMemset(T,0,szT)); CUDA_CHECK(cudaDeviceSynchronize());

    double tc_ms = time_tc(A,Bt,m,n,k,R,T,iters);
    double dp_ms = time_dp4a(A,Bt,m,n,k,R,T,iters);
    double tops = tiles * 16.0*16.0*4096.0 * 2.0 / (tc_ms/1000.0) / 1e12;

    printf("\nConfig: m=%d n=%d k=%d R=%d  tiles/region=%.0f  iters=%d\n", m,n,k,R,tiles,iters);
    printf("  TC   : %.3f ms/region  ->  %.2f TH/s   (%.1f INT8 TOPS)\n", tc_ms, ths_from(tiles,tc_ms), tops);
    printf("  DP4A : %.3f ms/region  ->  %.2f TH/s\n", dp_ms, ths_from(tiles,dp_ms));
    printf("  TC speedup vs DP4A: %.2fx\n", dp_ms/tc_ms);

    // ---------- BLAKE3 pass (the other half of pearl_pow_split) ----------
    {
        const int num_tiles = (m/16)*(n/16);
        uint32_t *dkey,*dtgt; uint8_t* digests; int *found,*coord;
        CUDA_CHECK(cudaMalloc(&dkey,32)); CUDA_CHECK(cudaMalloc(&dtgt,32));
        CUDA_CHECK(cudaMalloc(&digests,(size_t)num_tiles*32));
        CUDA_CHECK(cudaMalloc(&found,4)); CUDA_CHECK(cudaMalloc(&coord,8));
        CUDA_CHECK(cudaMemset(dkey,0x5a,32)); CUDA_CHECK(cudaMemset(dtgt,0,32)); // target 0 -> no hits (clean timing)
        CUDA_CHECK(cudaMemset(found,0,4));
        cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
        for(int i=0;i<5;i++) launch_pearl_blake3(T,num_tiles,n,dkey,dtgt,digests,found,coord,0);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEventRecord(a);
        for(int i=0;i<iters;i++) launch_pearl_blake3(T,num_tiles,n,dkey,dtgt,digests,found,coord,0);
        cudaEventRecord(b); cudaEventSynchronize(b);
        float bms=0; cudaEventElapsedTime(&bms,a,b); bms/=iters;
        double full = tc_ms + bms;
        printf("  BLAKE3: %.3f ms/region   GEMM+BLAKE3: %.3f ms -> %.2f TH/s (end-to-end est, GEMM is %.0f%%)\n",
               bms, full, ths_from(tiles, full), 100.0*tc_ms/full);
        cudaEventDestroy(a);cudaEventDestroy(b);
        cudaFree(dkey);cudaFree(dtgt);cudaFree(digests);cudaFree(found);cudaFree(coord);
    }

    printf("\n--- config sweep (BMxBN warpsMxN stages minb) ---\n");
    #define SWEEP(BM,BN,WM,WN,STG,MNB) do { \
        double _ms = time_cfg<BM,BN,WM,WN,STG,MNB>(A,Bt,m,n,k,R,T,iters); \
        if (_ms>0) printf("  %3dx%-3d %dx%d s%d b%d : %.3f ms  %.2f TH/s\n", \
                          BM,BN,WM,WN,STG,MNB,_ms,ths_from(tiles,_ms)); \
        else       printf("  %3dx%-3d %dx%d s%d b%d : (launch failed)\n",BM,BN,WM,WN,STG,MNB); \
    } while(0)
    SWEEP(32,64,2,4,4,2);
    SWEEP(32,64,2,4,6,2);
    SWEEP(32,64,2,4,3,4);
    SWEEP(64,64,4,4,4,3);
    SWEEP(64,64,4,4,6,2);
    SWEEP(64,64,4,4,3,4);
    SWEEP(64,128,4,8,4,1);
    SWEEP(128,64,8,4,4,1);
    SWEEP(32,128,2,8,4,2);
    #undef SWEEP

    printf("\n--- R-block-staged kernel ---\n");
    {   // correctness vs DP4A at a small grid
        const int cm=256, cn=256, ck=(k>=4096?4096:k), cR=R;
        const int ctiles=(cm/16)*(cn/16);
        size_t szA=(size_t)cm*ck, szBt=(size_t)cn*ck, szTc=(size_t)ctiles*16*4;
        int8_t *cA,*cBt; uint32_t *cTp,*cTa;
        cudaMalloc(&cA,szA);cudaMalloc(&cBt,szBt);cudaMalloc(&cTp,szTc);cudaMalloc(&cTa,szTc);
        int t2=256;
        fill_det<<<(unsigned)((szA+t2-1)/t2),t2>>>(cA,szA,0x12345678);
        fill_det<<<(unsigned)((szBt+t2-1)/t2),t2>>>(cBt,szBt,0x87654321);
        cudaMemset(cTp,0,szTc);cudaMemset(cTa,0,szTc);cudaDeviceSynchronize();
        launch_pearl_gemm_only(cA,cBt,cm,cn,ck,cR,cTp,1,0);
        cudaError_t re=launch_rblock<64,64,4,4,2>(cA,cBt,cm,cn,ck,cR,cTa,0);
        cudaDeviceSynchronize();
        if(re!=cudaSuccess) printf("  launch err: %s\n",cudaGetErrorString(re));
        else {
            uint32_t *hp=(uint32_t*)malloc(szTc),*ha=(uint32_t*)malloc(szTc);
            cudaMemcpy(hp,cTp,szTc,cudaMemcpyDeviceToHost);
            cudaMemcpy(ha,cTa,szTc,cudaMemcpyDeviceToHost);
            int d=0;for(int i=0;i<ctiles*16;i++)if(hp[i]!=ha[i])d++;
            printf("  correctness: %s (%d/%d differ)\n", d==0?"BIT-EXACT PASS":"FAIL", d, ctiles*16);
            free(hp);free(ha);
        }
        cudaFree(cA);cudaFree(cBt);cudaFree(cTp);cudaFree(cTa);
    }
    #define RBLK(BM,BN,WM,WN,STG) do { \
        cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b); \
        for(int i=0;i<5;i++) launch_rblock<BM,BN,WM,WN,STG>(A,Bt,m,n,k,R,T,0); \
        if(cudaDeviceSynchronize()!=cudaSuccess) printf("  rblock %dx%d s%d : launch failed\n",BM,BN,STG); \
        else { cudaEventRecord(a); \
          for(int i=0;i<iters;i++) launch_rblock<BM,BN,WM,WN,STG>(A,Bt,m,n,k,R,T,0); \
          cudaEventRecord(b);cudaEventSynchronize(b);float ms=0;cudaEventElapsedTime(&ms,a,b);ms/=iters; \
          printf("  rblock %dx%d s%d : %.3f ms  %.2f TH/s\n",BM,BN,STG,ms,ths_from(tiles,ms)); } \
        cudaEventDestroy(a);cudaEventDestroy(b); \
    } while(0)
    RBLK(64,64,4,4,2);
    RBLK(32,64,2,4,2);
    #undef RBLK

    printf("\n--- wide kernel (NT tiles/warp -> NT*2 acc chains) ---\n");
    {   // correctness vs DP4A
        const int cm=256, cn=256, ck=(k>=4096?4096:k), cR=R;
        const int ctiles=(cm/16)*(cn/16);
        size_t szA=(size_t)cm*ck, szBt=(size_t)cn*ck, szTc=(size_t)ctiles*16*4;
        int8_t *cA,*cBt; uint32_t *cTp,*cTa;
        cudaMalloc(&cA,szA);cudaMalloc(&cBt,szBt);cudaMalloc(&cTp,szTc);cudaMalloc(&cTa,szTc);
        int t2=256;
        fill_det<<<(unsigned)((szA+t2-1)/t2),t2>>>(cA,szA,0x12345678);
        fill_det<<<(unsigned)((szBt+t2-1)/t2),t2>>>(cBt,szBt,0x87654321);
        cudaMemset(cTp,0,szTc);cudaMemset(cTa,0,szTc);cudaDeviceSynchronize();
        launch_pearl_gemm_only(cA,cBt,cm,cn,ck,cR,cTp,1,0);
        cudaError_t we=launch_wide<64,64,4,2,2,4,2>(cA,cBt,cm,cn,ck,cR,cTa,0);
        cudaDeviceSynchronize();
        if(we!=cudaSuccess) printf("  launch err: %s\n",cudaGetErrorString(we));
        else {
            uint32_t *hp=(uint32_t*)malloc(szTc),*ha=(uint32_t*)malloc(szTc);
            cudaMemcpy(hp,cTp,szTc,cudaMemcpyDeviceToHost);
            cudaMemcpy(ha,cTa,szTc,cudaMemcpyDeviceToHost);
            int d=0;for(int i=0;i<ctiles*16;i++)if(hp[i]!=ha[i])d++;
            printf("  correctness (NT=2): %s (%d/%d differ)\n", d==0?"BIT-EXACT PASS":"FAIL", d, ctiles*16);
            free(hp);free(ha);
        }
        cudaFree(cA);cudaFree(cBt);cudaFree(cTp);cudaFree(cTa);
    }
    // honest tiles = tiles ACTUALLY covered (grid may not tile m,n if BM|m or BN|n
    // fails). A '*' marks configs that don't evenly divide the region (not usable
    // in the miner without bounds-checking; shown for reference only).
    #define WIDE(BM,BN,WM,WN,NT,STG,MNB) do { \
        double _at = (double)((m/BM)*(BM/16)) * (double)((n/BN)*(BN/16)); \
        const char* _bad = ((m%BM)||(n%BN)) ? "*" : " "; \
        cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b); \
        for(int i=0;i<5;i++) launch_wide<BM,BN,WM,WN,NT,STG,MNB>(A,Bt,m,n,k,R,T,0); \
        if(cudaDeviceSynchronize()!=cudaSuccess) printf("  wide %dx%d WN%d NT%d s%d : launch failed\n",BM,BN,WN,NT,STG); \
        else { cudaEventRecord(a); \
          for(int i=0;i<iters;i++) launch_wide<BM,BN,WM,WN,NT,STG,MNB>(A,Bt,m,n,k,R,T,0); \
          cudaEventRecord(b);cudaEventSynchronize(b);float ms=0;cudaEventElapsedTime(&ms,a,b);ms/=iters; \
          printf("  wide%s%3dx%-3d WN%d NT%-2d s%d : %.3f ms  %.2f TH/s\n",_bad,BM,BN,WN,NT,STG,ms,ths_from(_at,ms)); } \
        cudaEventDestroy(a);cudaEventDestroy(b); \
    } while(0)
    WIDE(64,128,4,1,8,3,2);
    WIDE(64,128,4,1,8,4,1);
    WIDE(64,256,4,1,16,2,1);
    WIDE(64,256,4,1,16,3,1);
    WIDE(128,128,8,1,8,2,1);
    WIDE(128,256,8,1,16,2,1);
    WIDE(32,128,2,1,8,4,2);
    WIDE(64,192,4,1,12,2,1);
    WIDE(96,192,6,1,12,2,1);
    #undef WIDE

    printf("\n--- ldm kernel (ldmatrix loads) ---\n");
    {   // correctness vs DP4A
        const int cm=256, cn=256, ck=(k>=4096?4096:k), cR=R;
        const int ctiles=(cm/16)*(cn/16);
        size_t szA=(size_t)cm*ck, szBt=(size_t)cn*ck, szTc=(size_t)ctiles*16*4;
        int8_t *cA,*cBt; uint32_t *cTp,*cTa;
        cudaMalloc(&cA,szA);cudaMalloc(&cBt,szBt);cudaMalloc(&cTp,szTc);cudaMalloc(&cTa,szTc);
        int t2=256;
        fill_det<<<(unsigned)((szA+t2-1)/t2),t2>>>(cA,szA,0x12345678);
        fill_det<<<(unsigned)((szBt+t2-1)/t2),t2>>>(cBt,szBt,0x87654321);
        cudaMemset(cTp,0,szTc);cudaMemset(cTa,0,szTc);cudaDeviceSynchronize();
        launch_pearl_gemm_only(cA,cBt,cm,cn,ck,cR,cTp,1,0);
        cudaError_t we=launch_ldm<64,256,4,1,16,2,1>(cA,cBt,cm,cn,ck,cR,cTa,0);
        cudaDeviceSynchronize();
        if(we!=cudaSuccess) printf("  launch err: %s\n",cudaGetErrorString(we));
        else {
            uint32_t *hp=(uint32_t*)malloc(szTc),*ha=(uint32_t*)malloc(szTc);
            cudaMemcpy(hp,cTp,szTc,cudaMemcpyDeviceToHost);
            cudaMemcpy(ha,cTa,szTc,cudaMemcpyDeviceToHost);
            int d=0;for(int i=0;i<ctiles*16;i++)if(hp[i]!=ha[i])d++;
            printf("  correctness (NT=16): %s (%d/%d differ)\n", d==0?"BIT-EXACT PASS":"FAIL", d, ctiles*16);
            free(hp);free(ha);
        }
        cudaFree(cA);cudaFree(cBt);cudaFree(cTp);cudaFree(cTa);
    }
    #define LDM(BM,BN,WM,WN,NT,STG,MNB) do { \
        double _at=(double)((m/BM)*(BM/16))*(double)((n/BN)*(BN/16)); \
        const char* _bad=((m%BM)||(n%BN))?"*":" "; \
        cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b); \
        for(int i=0;i<5;i++) launch_ldm<BM,BN,WM,WN,NT,STG,MNB>(A,Bt,m,n,k,R,T,0); \
        if(cudaDeviceSynchronize()!=cudaSuccess) printf("  ldm %dx%d NT%d s%d : launch failed\n",BM,BN,NT,STG); \
        else { cudaEventRecord(a); \
          for(int i=0;i<iters;i++) launch_ldm<BM,BN,WM,WN,NT,STG,MNB>(A,Bt,m,n,k,R,T,0); \
          cudaEventRecord(b);cudaEventSynchronize(b);float ms=0;cudaEventElapsedTime(&ms,a,b);ms/=iters; \
          printf("  ldm%s%3dx%-3d NT%-2d s%d : %.3f ms  %.2f TH/s\n",_bad,BM,BN,NT,STG,ms,ths_from(_at,ms)); } \
        cudaEventDestroy(a);cudaEventDestroy(b); \
    } while(0)
    LDM(64,256,4,1,16,2,1);
    LDM(64,256,4,1,16,3,1);
    LDM(64,256,4,1,16,4,1);   // s5/s6 @64x256 exceed the 48KB static smem cap
    LDM(64,128,4,1,8,3,2);
    LDM(64,128,4,1,8,4,2);
    LDM(64,128,4,1,8,6,2);
    LDM(128,256,8,1,16,2,1);
    LDM(128,256,8,1,16,3,1);
    LDM(256,256,16,1,16,2,1);   // biggest block: 16 warps share B (max amortization)
    LDM(256,128,16,1,8,3,1);
    #undef LDM

    printf("\n--- wide1 kernel (1 sync/k-tile) ---\n");
    {   // correctness vs DP4A
        const int cm=256, cn=256, ck=(k>=4096?4096:k), cR=R;
        const int ctiles=(cm/16)*(cn/16);
        size_t szA=(size_t)cm*ck, szBt=(size_t)cn*ck, szTc=(size_t)ctiles*16*4;
        int8_t *cA,*cBt; uint32_t *cTp,*cTa;
        cudaMalloc(&cA,szA);cudaMalloc(&cBt,szBt);cudaMalloc(&cTp,szTc);cudaMalloc(&cTa,szTc);
        int t2=256;
        fill_det<<<(unsigned)((szA+t2-1)/t2),t2>>>(cA,szA,0x12345678);
        fill_det<<<(unsigned)((szBt+t2-1)/t2),t2>>>(cBt,szBt,0x87654321);
        cudaMemset(cTp,0,szTc);cudaMemset(cTa,0,szTc);cudaDeviceSynchronize();
        launch_pearl_gemm_only(cA,cBt,cm,cn,ck,cR,cTp,1,0);
        cudaError_t we=launch_wide1<64,128,4,1,8,3,2>(cA,cBt,cm,cn,ck,cR,cTa,0);
        cudaDeviceSynchronize();
        if(we!=cudaSuccess) printf("  launch err: %s\n",cudaGetErrorString(we));
        else { uint32_t *hp=(uint32_t*)malloc(szTc),*ha=(uint32_t*)malloc(szTc);
            cudaMemcpy(hp,cTp,szTc,cudaMemcpyDeviceToHost);cudaMemcpy(ha,cTa,szTc,cudaMemcpyDeviceToHost);
            int d=0;for(int i=0;i<ctiles*16;i++)if(hp[i]!=ha[i])d++;
            printf("  correctness (NT=8): %s (%d/%d differ)\n", d==0?"BIT-EXACT PASS":"FAIL", d, ctiles*16);
            free(hp);free(ha);}
        cudaFree(cA);cudaFree(cBt);cudaFree(cTp);cudaFree(cTa);
    }
    #define WIDE1(BM,BN,WM,WN,NT,STG,MNB) do { \
        double _at=(double)((m/BM)*(BM/16))*(double)((n/BN)*(BN/16)); \
        const char* _bad=((m%BM)||(n%BN))?"*":" "; \
        cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b); \
        for(int i=0;i<5;i++) launch_wide1<BM,BN,WM,WN,NT,STG,MNB>(A,Bt,m,n,k,R,T,0); \
        if(cudaDeviceSynchronize()!=cudaSuccess) printf("  wide1 %dx%d NT%d s%d : launch failed\n",BM,BN,NT,STG); \
        else { cudaEventRecord(a); \
          for(int i=0;i<iters;i++) launch_wide1<BM,BN,WM,WN,NT,STG,MNB>(A,Bt,m,n,k,R,T,0); \
          cudaEventRecord(b);cudaEventSynchronize(b);float ms=0;cudaEventElapsedTime(&ms,a,b);ms/=iters; \
          printf("  wide1%s%3dx%-3d NT%-2d s%d : %.3f ms  %.2f TH/s\n",_bad,BM,BN,NT,STG,ms,ths_from(_at,ms)); } \
        cudaEventDestroy(a);cudaEventDestroy(b); \
    } while(0)
    WIDE1(64,128,4,1,8,3,2);
    WIDE1(64,128,4,1,8,4,2);
    WIDE1(64,256,4,1,16,3,1);
    WIDE1(64,256,4,1,16,4,1);
    WIDE1(128,256,8,1,16,3,1);
    WIDE1(64,192,4,1,12,3,1);
    #undef WIDE1

    cudaFree(A);cudaFree(Bt);cudaFree(T);
    return 0;
}
