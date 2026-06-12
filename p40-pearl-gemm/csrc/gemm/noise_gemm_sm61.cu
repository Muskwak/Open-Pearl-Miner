// Fast DP4A noise-apply for Pascal (sm_61). Replaces the naive noise_A/noise_B
// scalar kernels (~147 ms each at the mining dims) with a tiled DP4A GEMM:
//
//   out[m,k] = clamp_i8( Z[m,k] + sum_{r<R} X[m,r] * Y[k,r] )
//
//   noise_A:  X = EAL[M,R],  Y = EAR_t[K,R] (= EAR^T),  Z = A[M,K]
//   noise_B:  X = EBR[N,R],  Y = EBL[K,R],              Z = Bt[N,K]
//
// Both operands are contracted over R (contiguous), i.e. the A@B^T form, so DP4A
// applies directly. Bit-exact with the naive kernels (same integer products and
// the same int8 clamp); the int32 denoise side-products are intentionally
// dropped (mining never uses them).

#include <cuda_runtime.h>
#include <cstdint>

static __device__ __forceinline__ int dp4a_ng(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  int r = c;
  for (int i = 0; i < 4; ++i)
    r += (int)(int8_t)((a >> (i * 8)) & 0xFF) * (int)(int8_t)((b >> (i * 8)) & 0xFF);
  return r;
#endif
}

static __device__ __forceinline__ int8_t clamp8_ng(int v) {
  if (v > 127) v = 127;
  if (v < -128) v = -128;
  return (int8_t)v;
}

template <int BM, int BK>
__global__ void noise_gemm_kernel(const int8_t* __restrict__ X,
                                  const int8_t* __restrict__ Y,
                                  const int8_t* __restrict__ Z,
                                  int8_t* __restrict__ out, int M, int K, int R) {
  const int RW = R / 4;  // dp4a words per row
  extern __shared__ int smem[];
  int* sX = smem;            // BM rows x RW
  int* sY = smem + BM * RW;  // BK rows x RW
  const int bm = blockIdx.y * BM;
  const int bk = blockIdx.x * BK;
  const int* Xi = reinterpret_cast<const int*>(X);
  const int* Yi = reinterpret_cast<const int*>(Y);

  for (int i = threadIdx.x; i < BM * RW; i += blockDim.x) {
    const int r = i / RW, c = i % RW;
    sX[i] = Xi[(size_t)(bm + r) * RW + c];
  }
  for (int i = threadIdx.x; i < BK * RW; i += blockDim.x) {
    const int r = i / RW, c = i % RW;
    sY[i] = Yi[(size_t)(bk + r) * RW + c];
  }
  __syncthreads();

  // 16x16 threads, each computes a 4x4 micro-tile (BM=BK=64). Register-blocked:
  // per contraction step load 4 X + 4 Y from shared and do 16 dp4a (0.5 loads/dp4a).
  const int tm = (threadIdx.x >> 4) * 4;   // 0,4,..,60
  const int tn = (threadIdx.x & 15) * 4;
  int acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0;
#pragma unroll 4
  for (int c = 0; c < RW; ++c) {
    int xv[4], yv[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) xv[i] = sX[(tm + i) * RW + c];
#pragma unroll
    for (int j = 0; j < 4; ++j) yv[j] = sY[(tn + j) * RW + c];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j) acc[i][j] = dp4a_ng(xv[i], yv[j], acc[i][j]);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const size_t o = (size_t)(bm + tm + i) * K + (bk + tn + j);
      out[o] = clamp8_ng((int)Z[o] + acc[i][j]);
    }
}

// out[M,K] = clamp(Z + X @ Y^T over R). Requires M%64==0, K%64==0, R%4==0.
void launch_noise_gemm(const int8_t* X, const int8_t* Y, const int8_t* Z,
                       int8_t* out, int M, int K, int R, cudaStream_t stream) {
  constexpr int BM = 64, BK = 64;
  dim3 grid(K / BK, M / BM);
  const int RW = R / 4;
  size_t smem = (size_t)(BM + BK) * RW * sizeof(int);
  noise_gemm_kernel<BM, BK><<<grid, 256, smem, stream>>>(X, Y, Z, out, M, K, R);
}
