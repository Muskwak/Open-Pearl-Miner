// Validate the tensor-core noise GEMM (pcute::launch_noise_gemm_tc) bit-exact vs the
// DP4A reference (launch_noise_gemm), and time both at the real region dims.
//   out[M,K] = clamp_i8(Z + X @ Y^T over R),  X[M,R] Y[K,R] Z[M,K] out[M,K]
//
// Build (from p40-pearl-gemm/):
//   nvcc -O3 -std=c++17 --expt-relaxed-constexpr -arch=sm_89 -cudart static -Xcompiler /MT \
//        -I csrc -I "<cutlass>/include" -o tests\cnoise_cute.exe tests\cnoise_cute.cu
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
    printf("CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); return 1; } } while(0)

#include "../csrc/gemm/noise_gemm_sm61.cu"     // launch_noise_gemm (DP4A reference)
#include "../csrc/gemm/pearl_cute_noise.cuh"   // pcute::launch_noise_gemm_tc

static int8_t det(long long i, unsigned long long s) {
    s += (unsigned long long)i * 0x9E3779B97F4A7C15ULL; s *= 0xD2511F53CD9E8D57ULL;
    s ^= s >> 31; s *= 0x9E3779B9ull; return (int8_t)((int)(s & 0xFF) - 128);
}

static double time_it(void(*fn)(), int iters) {
    for (int i = 0; i < 20; i++) fn(); cudaDeviceSynchronize();
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a); for (int i = 0; i < iters; i++) fn(); cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms,a,b); cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)ms / iters;
}

static int8_t *gX,*gY,*gZ,*gOref,*gOtc;
static int gM,gK,gR;
static void run_ref() { launch_noise_gemm(gX,gY,gZ,gOref,gM,gK,gR,0); }
static void run_tc()  { pcute::launch_noise_gemm_tc<128,256,32,3>(gX,gY,gZ,gOtc,gM,gK,gR,0); }

int main() {
    const int M=4096, K=4096, R=256;
    gM=M; gK=K; gR=R;
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("Device: %s (sm_%d%d)  noise out[%d,%d] R=%d\n", p.name,p.major,p.minor,M,K,R);

    CK(cudaMalloc(&gX,(size_t)M*R)); CK(cudaMalloc(&gY,(size_t)K*R));
    CK(cudaMalloc(&gZ,(size_t)M*K)); CK(cudaMalloc(&gOref,(size_t)M*K)); CK(cudaMalloc(&gOtc,(size_t)M*K));
    { std::vector<int8_t> hX((size_t)M*R), hY((size_t)K*R), hZ((size_t)M*K);
      for(size_t i=0;i<hX.size();++i) hX[i]=det((long long)i,0x1111);
      for(size_t i=0;i<hY.size();++i) hY[i]=det((long long)i,0x2222);
      for(size_t i=0;i<hZ.size();++i) hZ[i]=det((long long)i,0x3333);
      CK(cudaMemcpy(gX,hX.data(),hX.size(),cudaMemcpyHostToDevice));
      CK(cudaMemcpy(gY,hY.data(),hY.size(),cudaMemcpyHostToDevice));
      CK(cudaMemcpy(gZ,hZ.data(),hZ.size(),cudaMemcpyHostToDevice)); }

    run_ref(); CK(cudaDeviceSynchronize());
    run_tc();  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    std::vector<int8_t> hRef((size_t)M*K), hTc((size_t)M*K);
    CK(cudaMemcpy(hRef.data(),gOref,hRef.size(),cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hTc.data(),gOtc,hTc.size(),cudaMemcpyDeviceToHost));
    long long diff=0; size_t fi=0;
    for(size_t i=0;i<hRef.size();++i) if(hRef[i]!=hTc[i]){ if(!diff)fi=i; ++diff; }
    printf("noise TC: %s (%lld/%zu int8 differ)\n", diff==0?"BIT-EXACT PASS":"FAIL", diff, hRef.size());
    if(diff) printf("  first @ %zu: tc=%d ref=%d\n", fi, (int)hTc[fi], (int)hRef[fi]);

    double dp = time_it(run_ref, 100), tc = time_it(run_tc, 100);
    printf("noise DP4A : %.3f ms/region\n", dp);
    printf("noise TC   : %.3f ms/region   (%.2fx faster)\n", tc, dp/tc);
    return diff==0?0:1;
}
