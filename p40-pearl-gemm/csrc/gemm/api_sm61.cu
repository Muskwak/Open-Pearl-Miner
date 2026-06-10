#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

extern "C" {

void launch_dp4a_gemm(
    const int8_t* A, const int8_t* B,
    const float* A_scales, const float* B_scales,
    half* C, int M, int N, int K,
    cudaStream_t stream);

void launch_noise_A(
    const int8_t* A, const int8_t* EAL,
    const int8_t* EAR, const int8_t* EBL,
    int8_t* ApEA, int32_t* AxEBL,
    int M, int K, int R,
    cudaStream_t stream);

void launch_noise_B(
    const int8_t* B, const int8_t* EBR,
    const int8_t* EAR, const int8_t* EBL,
    int8_t* BpEB, int32_t* EARxBpEB,
    int N, int K, int R,
    cudaStream_t stream);

void launch_denoise_converter(
    const int32_t* EARxBpEB_in,
    const int32_t* AxEBL_in,
    half* EARxBpEB_out,
    half* AxEBL_out,
    int M, int N, int R,
    cudaStream_t stream) {

  int total = 0;
  if (EARxBpEB_in && EARxBpEB_out) {
    total += N * R;
  }
  if (AxEBL_in && AxEBL_out) {
    total += M * R;
  }

  if (total == 0) return;

  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  auto convert_kernel = [=] __device__ (int i) {
    if (EARxBpEB_in && EARxBpEB_out) {
      if (i < N * R) {
        int val = EARxBpEB_in[i];
        float fval = (float)val / 4096.0f;
        fval = fmaxf(-65504.0f, fminf(65504.0f, fval));
        EARxBpEB_out[i] = __float2half(fval);
      }
      i -= N * R;
    }
    if (AxEBL_in && AxEBL_out) {
      if (i < M * R) {
        int val = AxEBL_in[i];
        float fval = (float)val / 4096.0f;
        fval = fmaxf(-65504.0f, fminf(65504.0f, fval));
        AxEBL_out[i] = __float2half(fval);
      }
    }
  };

  // Simple grid-stride loop kernel
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  for (int i = idx; i < total; i += stride) {
    convert_kernel(i);
  }
}

}  // extern "C"
