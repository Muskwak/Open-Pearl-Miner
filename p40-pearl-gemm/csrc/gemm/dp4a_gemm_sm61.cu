#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define DP4A_TILE_M 64
#define DP4A_TILE_N 64
#define DP4A_TILE_K 64
#define DP4A_WARPS_M 2
#define DP4A_WARPS_N 2
#define DP4A_THREADS_PER_WARP 32
#define DP4A_TX (DP4A_WARPS_M * 16)
#define DP4A_TY (DP4A_WARPS_N * 2)

__device__ __forceinline__ int dp4a(int a, int b, int c) {
  int result;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
               : "=r"(result)
               : "r"(a), "r"(b), "r"(c));
#else
  result = c;
  for (int i = 0; i < 4; ++i) {
    int8_t ba = (a >> (i * 8)) & 0xFF;
    int8_t bb = (b >> (i * 8)) & 0xFF;
    result += int(ba) * int(bb);
  }
#endif
  return result;
}

__global__ void dp4a_gemm_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    const float* __restrict__ A_scales,
    const float* __restrict__ B_scales,
    half* __restrict__ C,
    int M, int N, int K) {

  __shared__ int8_t smem_A[DP4A_TILE_M * DP4A_TILE_K];
  __shared__ int8_t smem_B[DP4A_TILE_N * DP4A_TILE_K];

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int m_start = bx * DP4A_TILE_M;
  int n_start = by * DP4A_TILE_N;

  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int warp_x = warp_id % DP4A_WARPS_M;
  int warp_y = warp_id / DP4A_WARPS_N;
  int lane = tid % 32;

  int a_col = lane % 16;
  int a_row = (lane / 16) + warp_x * 16;
  int b_col = lane % 2;
  int b_row_base = (lane / 2) * 2;

  int acc[DP4A_TILE_N / (DP4A_WARPS_N * 2)] = {0};

  for (int k_tile = 0; k_tile < K; k_tile += DP4A_TILE_K) {
    __syncthreads();

    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
         i < DP4A_TILE_M * DP4A_TILE_K;
         i += blockDim.y * blockDim.x) {
      int mk = i;
      int mi = mk / DP4A_TILE_K;
      int ki = mk % DP4A_TILE_K;
      int gm = m_start + mi;
      int gk = k_tile + ki;
      if (gm < M && gk < K) {
        smem_A[mi * DP4A_TILE_K + ki] = A[gm * K + gk];
      } else {
        smem_A[mi * DP4A_TILE_K + ki] = 0;
      }
    }

    for (int i = threadIdx.y * blockDim.x + threadIdx.x;
         i < DP4A_TILE_N * DP4A_TILE_K;
         i += blockDim.y * blockDim.x) {
      int nk = i;
      int ni = nk / DP4A_TILE_K;
      int ki = nk % DP4A_TILE_K;
      int gn = n_start + ni;
      int gk = k_tile + ki;
      if (gn < N && gk < K) {
        smem_B[ni * DP4A_TILE_K + ki] = B[gn * K + gk];
      } else {
        smem_B[ni * DP4A_TILE_K + ki] = 0;
      }
    }

    __syncthreads();

    constexpr int TILE_N_PER_WARP = DP4A_TILE_N / DP4A_WARPS_N;
    constexpr int TILE_M_PER_WARP = DP4A_TILE_M / DP4A_WARPS_M;
    constexpr int THREADS_PER_TILE_N = TILE_N_PER_WARP / 2;

    for (int kk = 0; kk < DP4A_TILE_K; kk += 4) {
      int pk = kk / 4;

      const int8_t* A_warp = &smem_A[warp_x * TILE_M_PER_WARP * DP4A_TILE_K];
      const int8_t* B_warp = &smem_B[warp_y * TILE_N_PER_WARP * DP4A_TILE_K];

      int a_val = *(reinterpret_cast<const int*>(
          &A_warp[a_row * DP4A_TILE_K + kk]));

      for (int n_sub = 0; n_sub < TILE_N_PER_WARP / 2; ++n_sub) {
        int n_idx = n_sub * 2 + b_col;
        int b_idx = b_row_base + n_sub * 2;

        int b_val = *(reinterpret_cast<const int*>(
            &B_warp[b_idx * DP4A_TILE_K + kk]));

        acc[n_sub] = dp4a(a_val, b_val, acc[n_sub]);
      }
    }

    __syncthreads();
  }

  for (int n_sub = 0; n_sub < DP4A_TILE_N / (DP4A_WARPS_N * 2); ++n_sub) {
    int global_n = n_start + warp_y * (DP4A_TILE_N / DP4A_WARPS_N) +
                   n_sub * 2 + b_col;
    int global_m = m_start + a_row;

    if (global_m < M && global_n < N) {
      float scale = A_scales[global_m] * B_scales[global_n];
      float result_f32 = acc[n_sub] * scale;
      result_f32 = fmaxf(-65504.0f, fminf(65504.0f, result_f32));
      C[global_m * N + global_n] = __float2half(result_f32);
    }
  }
}

void launch_dp4a_gemm(
    const int8_t* A, const int8_t* B,
    const float* A_scales, const float* B_scales,
    half* C, int M, int N, int K,
    cudaStream_t stream) {

  dim3 block(DP4A_TX, DP4A_TY);
  dim3 grid(
      (M + DP4A_TILE_M - 1) / DP4A_TILE_M,
      (N + DP4A_TILE_N - 1) / DP4A_TILE_N);

  dp4a_gemm_kernel<<<grid, block, 0, stream>>>(
      A, B, A_scales, B_scales, C, M, N, K);
}
