// Torch-free C ABI over the Pascal (sm_61) and Ampere+ (sm_80+) Pearl kernels.
//
// Auto-detects GPU architecture and dispatches to DP4A (Pascal) or tensor-core
// (Ampere/Ada) kernel. Multi-GPU: each device uses its optimal kernel path.
//
// Compiles to a standalone shared library (p40cuda.dll / libp40cuda.so) that
// links ONLY the CUDA runtime (cudart, ~0.5 MB) — no torch, no pybind. Callers
// drive it from Python via stdlib ctypes, managing device memory with
// cuda-python (cuMemAlloc returns a device pointer that these functions accept).
//
// Every function returns a cudaError_t as int (0 == cudaSuccess).

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

#include "tensor_hash/tensor_hash_decl.hpp"  // tensor_hash, commitment_hash_from_merkle_roots

// GPU job setup kernels (rng_fill_sm61.cu).
extern "C" void launch_fill_rand_i8(int8_t* out, int64_t numel, uint64_t seed, cudaStream_t stream);
extern "C" void launch_transpose_i8(const int8_t* src, int8_t* dst, int rows, int cols, cudaStream_t stream);

#ifdef _WIN32
#define P40_API extern "C" __declspec(dllexport)
#else
#define P40_API extern "C" __attribute__((visibility("default")))
#endif

// --- kernel launchers (defined in the kernel .cu translation units) ---
void launch_noise_gen(int8_t*, int8_t*, int8_t*, int8_t*, const uint8_t*,
                      const uint8_t*, int, int, int, int, cudaStream_t);
void launch_noise_A(const int8_t*, const int8_t*, const int8_t*, const int8_t*,
                    int8_t*, int32_t*, int, int, int, cudaStream_t);
void launch_noise_B(const int8_t*, const int8_t*, const int8_t*, const int8_t*,
                    int8_t*, int32_t*, int, int, int, cudaStream_t);
void launch_noise_gemm(const int8_t* X, const int8_t* Y, const int8_t* Z,
                       int8_t* out, int M, int K, int R, cudaStream_t);
void launch_pearl_gemm_only(const int8_t*, const int8_t*, int, int, int, int,
                            uint32_t*, int, cudaStream_t);
void launch_pearl_blake3(const uint32_t*, int, int, const uint32_t*,
                         const uint32_t*, uint8_t*, int*, int*, cudaStream_t);

// Ampere+ tensor-core GEMM+transcript kernel (definition in pearl_ampere_tc.cu).
cudaError_t launch_pearl_ampere(const int8_t*, const int8_t*, int, int, int, int,
                                uint32_t*, cudaStream_t);

// Tensor-core (IMMA) noise-apply GEMM for sm_80+ (pcute::launch_noise_gemm_tc). ~5.6x
// the DP4A noise kernel on Ada, bit-exact; closes most of the kernel-vs-miner gap.
#include "pearl_cute_noise.cuh"

// =========================== transpose =====================================
// Logical src is [rows, cols] with src[r,c] = src_base[r*src_ld + col_off + c]
// (src_ld/col_off let this transpose a column-slice of a wider matrix without a
// separate copy). Writes dst[cols, rows] with dst[c,r] = src[r,c].
__global__ void transpose_i8_kernel(const int8_t* __restrict__ src,
                                    int8_t* __restrict__ dst, int rows, int cols,
                                    int src_ld, int col_off) {
  const long total = (long)rows * cols;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += (long)gridDim.x * blockDim.x) {
    const int r = (int)(idx / cols);
    const int c = (int)(idx % cols);
    dst[(size_t)c * rows + r] = src[(size_t)r * src_ld + col_off + c];
  }
}

// =========================== init / device ================================
// Block (sleep) the calling CPU thread while waiting on the GPU instead of
// spinning, so N single-GPU miner processes on a multi-GPU rig don't each burn a
// full CPU core. Must be called before this process creates its device context
// (i.e. at startup, before any malloc/kernel).
P40_API int p40_init(void) {
  return (int)cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
}
P40_API int p40_device_count(void) {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
  return n;
}

// =========================== memory helpers ================================
P40_API int p40_malloc(void** p, size_t n) { return (int)cudaMalloc(p, n); }
P40_API int p40_free(void* p) { return (int)cudaFree(p); }
P40_API int p40_memcpy_htod(void* d, const void* h, size_t n) {
  return (int)cudaMemcpy(d, h, n, cudaMemcpyHostToDevice);
}
P40_API int p40_memcpy_dtoh(void* h, const void* d, size_t n) {
  return (int)cudaMemcpy(h, d, n, cudaMemcpyDeviceToHost);
}
P40_API int p40_memset(void* d, int v, size_t n) { return (int)cudaMemset(d, v, n); }
P40_API int p40_sync(void) { return (int)cudaDeviceSynchronize(); }

// dst[cols,rows] = transpose of the [rows,cols] logical view of src (a column
// slice of a row-major matrix with leading dim src_ld, starting at col_off).
P40_API int p40_transpose_i8(const void* src, void* dst, int rows, int cols,
                              int src_ld, int col_off) {
  const long total = (long)rows * cols;
  int tpb = 256;
  long blocks = (total + tpb - 1) / tpb;
  if (blocks > 65535) blocks = 65535;
  transpose_i8_kernel<<<(unsigned)blocks, tpb>>>((const int8_t*)src,
                                                  (int8_t*)dst, rows, cols,
                                                  src_ld, col_off);
  return (int)cudaGetLastError();
}

// =========================== kernels =======================================

// GPU job setup (mirrors the torch setup_job): Philox-fill A[M,K] and B[K,N],
// transpose B->Bt[N,K], tensor_hash A and Bt (keyed by `key`), then derive the
// noise seeds. A, B, Bt, key, nsA, nsB are caller-allocated device buffers
// (nsA/nsB are 32 bytes). Replaces the ~7s host RNG+commit with ~0.1s on-device.
P40_API int p40_setup_job(void* A, void* B, void* Bt, const void* key,
                           void* nsA, void* nsB, int M, int N, int K, int R,
                           unsigned long long seed) {
  cudaStream_t s = 0;
  launch_fill_rand_i8((int8_t*)A, (int64_t)M * K, seed, s);
  launch_fill_rand_i8((int8_t*)B, (int64_t)K * N, seed + 1, s);
  launch_transpose_i8((const int8_t*)B, (int8_t*)Bt, K, N, s);  // B[K,N] -> Bt[N,K]

  const unsigned chunk = 1024u, tpb = 128u;
  unsigned ncA = ((unsigned)((int64_t)M * K) + chunk - 1) / chunk, nbA = (ncA + tpb - 1) / tpb;
  unsigned ncB = ((unsigned)((int64_t)N * K) + chunk - 1) / chunk, nbB = (ncB + tpb - 1) / tpb;
  unsigned scratch_bytes = (nbA > nbB ? nbA : nbB) * 32u;
  uint8_t *A_root = nullptr, *B_root = nullptr, *scratch = nullptr;
  cudaMalloc((void**)&A_root, 32);
  cudaMalloc((void**)&B_root, 32);
  cudaMalloc((void**)&scratch, scratch_bytes);
  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  tensor_hash((const uint8_t*)A, (uint32_t)((int64_t)M * K), A_root,
              (const uint8_t*)key, nbA, tpb, 2, 512, scratch, prop, s);
  tensor_hash((const uint8_t*)Bt, (uint32_t)((int64_t)N * K), B_root,
              (const uint8_t*)key, nbB, tpb, 2, 512, scratch, prop, s);
  commitment_hash_from_merkle_roots(A_root, B_root, (const uint8_t*)key,
                                     (uint8_t*)nsA, (uint8_t*)nsB, prop, s);
  cudaError_t e = cudaGetLastError();
  cudaFree(A_root);
  cudaFree(B_root);
  cudaFree(scratch);
  return (int)e;
}

// Generate the four noise operands. EAR is K-major [R,k], EBL is R-major [k,R]
// (the corrected layout mapping, matching the torch noise_gen binding).
P40_API int p40_noise_gen(void* EAL, void* EAR, void* EBL, void* EBR,
                           const void* key_A, const void* key_B,
                           int m, int n, int k, int R) {
  launch_noise_gen((int8_t*)EAL, (int8_t*)EAR, (int8_t*)EBL, (int8_t*)EBR,
                    (const uint8_t*)key_A, (const uint8_t*)key_B, m, n, k, R, 0);
  return (int)cudaGetLastError();
}

// A_ns = A + round(EAL @ EAR), int8 (ApEA). AxEBL is the int32 side-product.
P40_API int p40_noise_apply_A(const void* A, const void* EAL, const void* EAR,
                               const void* EBL, void* ApEA, void* AxEBL,
                               int M, int K, int R) {
  launch_noise_A((const int8_t*)A, (const int8_t*)EAL, (const int8_t*)EAR,
                  (const int8_t*)EBL, (int8_t*)ApEA, (int32_t*)AxEBL, M, K, R, 0);
  return (int)cudaGetLastError();
}

P40_API int p40_noise_apply_B(const void* B, const void* EBR, const void* EAR,
                               const void* EBL, void* BpEB, void* EARxBpEB,
                               int N, int K, int R) {
  launch_noise_B((const int8_t*)B, (const int8_t*)EBR, (const int8_t*)EAR,
                  (const int8_t*)EBL, (int8_t*)BpEB, (int32_t*)EARxBpEB, N, K, R, 0);
  return (int)cudaGetLastError();
}

// Fast noise-apply: out[M,K] = clamp(Z + X @ Y^T over R). For noise_A pass
// X=EAL, Y=EAR_t, Z=A_slice; for noise_B pass X=EBR, Y=EBL, Z=Bt.
P40_API int p40_noise_gemm(const void* X, const void* Y, const void* Z, void* out,
                            int M, int K, int R) {
  // sm_80+: IMMA tensor-core path (bit-exact, ~5.6x DP4A). Pascal / odd shapes: DP4A.
  static int s_major = -1;
  if (s_major < 0) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) s_major = prop.major;
    else s_major = 0;
  }
  if (s_major >= 8 && (M % 128 == 0) && (K % 256 == 0) && (R % 32 == 0)) {
    cudaError_t e = pcute::launch_noise_gemm_tc<128, 256, 32, 3>(
        (const int8_t*)X, (const int8_t*)Y, (const int8_t*)Z, (int8_t*)out, M, K, R, 0);
    if (e == cudaSuccess) return 0;   // else fall back to DP4A
  }
  launch_noise_gemm((const int8_t*)X, (const int8_t*)Y, (const int8_t*)Z,
                     (int8_t*)out, M, K, R, 0);
  return (int)cudaGetLastError();
}

// Two-step search: GEMM-only -> transcript buffer -> BLAKE3. The caller provides
// a reusable transcript buffer (>= (m/16)*(n/16)*16 uint32) so the mining hot
// loop does NOT cudaMalloc/Free per region (that serializes the device). All of
// transcript, digests[(m/16)*(n/16),32], found[1], coord[2] are caller-allocated.
//
// Auto-detects GPU architecture:
//   sm_80+ (Ampere/Ada): tensor-core kernel (launch_pearl_ampere) + BLAKE3
//   pre-sm_80 (Pascal/Volta): DP4A kernel (launch_pearl_gemm_only) + BLAKE3
P40_API int p40_pearl_pow_split(const void* A, const void* Bt, int m, int n,
                                 int k, int R, const void* key, const void* target,
                                 void* transcript, void* digests, void* found,
                                 void* coord, int variant) {
  cudaDeviceProp prop;
  int dev = 0;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);

  const int num_tiles = (m / 16) * (n / 16);
  uint32_t* tb = (uint32_t*)transcript;

  if (prop.major >= 8) {
    // Ampere+ tensor-core path
    cudaError_t err = launch_pearl_ampere(
        (const int8_t*)A, (const int8_t*)Bt, m, n, k, R, tb, 0);
    if (err != cudaSuccess) return (int)err;
  } else {
    // Pascal DP4A path
    launch_pearl_gemm_only((const int8_t*)A, (const int8_t*)Bt, m, n, k, R, tb,
                            variant, 0);
  }

  launch_pearl_blake3(tb, num_tiles, n, (const uint32_t*)key,
                       (const uint32_t*)target, (uint8_t*)digests, (int*)found,
                       (int*)coord, 0);
  return (int)cudaGetLastError();
}
