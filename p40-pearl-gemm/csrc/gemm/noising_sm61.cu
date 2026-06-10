#include <cuda_runtime.h>
#include <cstdint>

#define NOISE_TILE_M 64
#define NOISE_TILE_K 64
#define NOISE_R 64

__global__ void noise_A_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ EAL,
    const int8_t* __restrict__ EAR,
    const int8_t* __restrict__ EBL,
    int8_t* __restrict__ ApEA,
    int32_t* __restrict__ AxEBL,
    int M, int K, int R) {

  int bx = blockIdx.x;
  int m_start = bx * NOISE_TILE_M;

  __shared__ int8_t smem_A[NOISE_TILE_M * NOISE_TILE_K];
  __shared__ int8_t smem_EAL[NOISE_TILE_M * NOISE_R];
  __shared__ int8_t smem_EBL[NOISE_R * NOISE_TILE_K];
  __shared__ int8_t smem_EAR[NOISE_TILE_K * NOISE_R];

  int tid = threadIdx.x;
  int total_threads = blockDim.x;

  for (int i = tid; i < NOISE_TILE_M * NOISE_TILE_K; i += total_threads) {
    int mi = i / NOISE_TILE_K;
    int ki = i % NOISE_TILE_K;
    int gm = m_start + mi;
    if (gm < M && ki < K) {
      smem_A[mi * NOISE_TILE_K + ki] = A[gm * K + ki];
    }
  }

  for (int i = tid; i < NOISE_TILE_M * NOISE_R; i += total_threads) {
    int mi = i / NOISE_R;
    int ri = i % NOISE_R;
    int gm = m_start + mi;
    if (gm < M && ri < R) {
      smem_EAL[mi * NOISE_R + ri] = EAL[gm * R + ri];
    }
  }

  for (int i = tid; i < NOISE_R * NOISE_TILE_K; i += total_threads) {
    int ri = i / NOISE_TILE_K;
    int ki = i % NOISE_TILE_K;
    if (ri < R && ki < K) {
      smem_EBL[ri * NOISE_TILE_K + ki] = EBL[ri * K + ki];
    }
  }

  for (int i = tid; i < NOISE_TILE_K * NOISE_R; i += total_threads) {
    int ki = i / NOISE_R;
    int ri = i % NOISE_R;
    if (ki < K && ri < R) {
      smem_EAR[ki * NOISE_R + ri] = EAR[ki * R + ri];
    }
  }

  __syncthreads();

  int warp_id = tid / 32;
  int lane = tid % 32;
  int warps_m = NOISE_TILE_M / 16;
  int warp_m = warp_id % warps_m;
  int warp_n = warp_id / warps_m;  // n in output R dimension

  int row_start = warp_m * 16;
  int row_end = min(row_start + 16, NOISE_TILE_M);

  if (warp_n >= 2) return;

  for (int r_idx = warp_n * (NOISE_R / 2) + lane / 2;
       r_idx < NOISE_R;
       r_idx += (NOISE_R / 2) * 16) {
    int sum = 0;
    for (int kk = 0; kk < NOISE_TILE_K; kk += 4) {
      int k_idx = kk + (lane % 4);
      if (k_idx >= NOISE_TILE_K) break;
      for (int mi = row_start; mi < row_end; ++mi) {
        int a = smem_A[mi * NOISE_TILE_K + k_idx];
        int ebl = smem_EBL[r_idx * NOISE_TILE_K + k_idx];
        sum += a * ebl;
      }
    }

    if ((lane % 2) == 0) {
      for (int mi = row_start; mi < min(row_start + 16, (int)NOISE_TILE_M);
           ++mi) {
        if ((m_start + mi) < M) {
          int eal_val = smem_EAL[mi * NOISE_R + r_idx];
          int global_r = r_idx;
          if (global_r < R) {
            AxEBL[(m_start + mi) * R + global_r] = sum + eal_val * 0;
          }
        }
      }
    }
  }

  __syncthreads();

  for (int i = tid; i < NOISE_TILE_M * NOISE_TILE_K; i += total_threads) {
    int mi = i / NOISE_TILE_K;
    int ki = i % NOISE_TILE_K;
    int gm = m_start + mi;
    int gk = ki;
    if (gm < M && gk < K) {
      int ea_sum = 0;
      for (int r = 0; r < NOISE_R; ++r) {
        ea_sum += smem_EAL[mi * NOISE_R + r] * smem_EAR[ki * NOISE_R + r];
      }
      int8_t result = smem_A[mi * NOISE_TILE_K + ki] + (int8_t)(ea_sum >> 8);
      ApEA[gm * K + gk] = result;
    }
  }
}

__global__ void noise_B_kernel(
    const int8_t* __restrict__ B,
    const int8_t* __restrict__ EBR,
    const int8_t* __restrict__ EAR,
    const int8_t* __restrict__ EBL,
    int8_t* __restrict__ BpEB,
    int32_t* __restrict__ EARxBpEB,
    int N, int K, int R) {
  return;
}

void launch_noise_A(
    const int8_t* A, const int8_t* EAL,
    const int8_t* EAR, const int8_t* EBL,
    int8_t* ApEA, int32_t* AxEBL,
    int M, int K, int R,
    cudaStream_t stream) {

  dim3 block(256);
  dim3 grid((M + NOISE_TILE_M - 1) / NOISE_TILE_M);
  noise_A_kernel<<<grid, block, 0, stream>>>(
      A, EAL, EAR, EBL, ApEA, AxEBL, M, K, R);
}

void launch_noise_B(
    const int8_t* B, const int8_t* EBR,
    const int8_t* EAR, const int8_t* EBL,
    int8_t* BpEB, int32_t* EARxBpEB,
    int N, int K, int R,
    cudaStream_t stream) {

  dim3 block(256);
  dim3 grid((N + NOISE_TILE_M - 1) / NOISE_TILE_M);
  noise_B_kernel<<<grid, block, 0, stream>>>(
      B, EBR, EAR, EBL, BpEB, EARxBpEB, N, K, R);
}
