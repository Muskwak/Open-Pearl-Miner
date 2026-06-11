// Python bindings for the Pascal (sm_61) Pearl GEMM kernels.
//
// Registers the C++/CUDA launchers as functions on the `p40_pearl_gemm_cuda`
// extension module so they are callable from `p40_gemm_bindings.py`.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cstdint>

// --- Launcher declarations (definitions live in the *_sm61.cu / *.cu files) --
// Pascal-native launchers have plain C++ linkage; the denoise converter is
// exported with C linkage from api_sm61.cu.
void launch_dp4a_gemm(
    const int8_t* A, const int8_t* B,
    const float* A_scales, const float* B_scales,
    half* C, int M, int N, int K, cudaStream_t stream);

void launch_noise_A(
    const int8_t* A, const int8_t* EAL, const int8_t* EAR, const int8_t* EBL,
    int8_t* ApEA, int32_t* AxEBL, int M, int K, int R, cudaStream_t stream);

// GPU noise generator (definition in noise_generation.cu)
void launch_noise_gen(
    int8_t* EAL, int8_t* EAR_R_major, int8_t* EBL_K_major, int8_t* EBR,
    const uint8_t* key_A, const uint8_t* key_B,
    int m, int n, int k, int R, cudaStream_t stream);

void launch_noise_B(
    const int8_t* B, const int8_t* EBR, const int8_t* EAR, const int8_t* EBL,
    int8_t* BpEB, int32_t* EARxBpEB, int N, int K, int R, cudaStream_t stream);

extern "C" void launch_denoise_converter(
    const int32_t* EARxBpEB_in, const int32_t* AxEBL_in,
    half* EARxBpEB_out, half* AxEBL_out,
    int M, int N, int R, cudaStream_t stream);

void launch_inner_hash_kernel(
    uint32_t* input_buffer, int input_size,
    uint32_t* output_hash, int64_t iterations, cudaStream_t stream);

// Pascal Pearl PoW kernel (definition in pearl_pow_sm61.cu)
void launch_pearl_pow(
    const int8_t* A, const int8_t* Bt, int m, int n, int k, int R,
    const uint32_t* pow_key, const uint32_t* pow_target,
    uint8_t* out_digests, int* found_flag, int* found_coord, cudaStream_t stream);

// Fused high-throughput variant (definition in pearl_pow_fused_sm61.cu)
void launch_pearl_pow_fused(
    const int8_t* A, const int8_t* Bt, int m, int n, int k, int R,
    const uint32_t* pow_key, const uint32_t* pow_target,
    uint8_t* out_digests, int* found_flag, int* found_coord, cudaStream_t stream);

void launch_pearl_pow_fused_v(
    const int8_t* A, const int8_t* Bt, int m, int n, int k, int R,
    const uint32_t* pow_key, const uint32_t* pow_target,
    uint8_t* out_digests, int* found_flag, int* found_coord,
    int variant, cudaStream_t stream);

// tensor_hash host entry (definition in tensor_hash.cu -> tensor_hash_host_sm61.hpp)
void tensor_hash(
    const uint8_t* data, uint32_t data_size, uint8_t* out,
    const uint8_t key[32], uint32_t num_blocks, uint32_t threads_per_block,
    uint32_t num_stages, uint32_t leaves_per_mt_block, uint8_t* roots,
    cudaDeviceProp& deviceProp, cudaStream_t stream);

namespace {

inline cudaStream_t cur_stream() {
  return at::cuda::getCurrentCUDAStream().stream();
}

inline half* half_ptr(at::Tensor& t) {
  return reinterpret_cast<half*>(t.data_ptr<at::Half>());
}

}  // namespace

void dp4a_gemm(at::Tensor A, at::Tensor B, at::Tensor A_scales,
               at::Tensor B_scales, at::Tensor C,
               int64_t M, int64_t N, int64_t K) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(),
              "dp4a_gemm: all tensors must be CUDA");
  TORCH_CHECK(A.scalar_type() == at::kChar && B.scalar_type() == at::kChar,
              "dp4a_gemm: A and B must be int8");
  TORCH_CHECK(C.scalar_type() == at::kHalf, "dp4a_gemm: C must be float16");
  TORCH_CHECK(A_scales.scalar_type() == at::kFloat &&
                  B_scales.scalar_type() == at::kFloat,
              "dp4a_gemm: scales must be float32");
  // Run on the tensors' device, and use that device's current stream, so the
  // kernel does not dereference pointers from another device's address space.
  const c10::cuda::CUDAGuard device_guard(A.device());
  launch_dp4a_gemm(
      A.data_ptr<int8_t>(), B.data_ptr<int8_t>(),
      A_scales.data_ptr<float>(), B_scales.data_ptr<float>(),
      half_ptr(C), (int)M, (int)N, (int)K, cur_stream());
}

void noise_A(at::Tensor A, at::Tensor EAL, at::Tensor EAR, at::Tensor EBL,
             at::Tensor ApEA, at::Tensor AxEBL,
             int64_t M, int64_t K, int64_t R) {
  TORCH_CHECK(A.is_cuda(), "noise_A: tensors must be CUDA");
  const c10::cuda::CUDAGuard device_guard(A.device());
  launch_noise_A(
      A.data_ptr<int8_t>(), EAL.data_ptr<int8_t>(), EAR.data_ptr<int8_t>(),
      EBL.data_ptr<int8_t>(), ApEA.data_ptr<int8_t>(),
      AxEBL.data_ptr<int32_t>(), (int)M, (int)K, (int)R, cur_stream());
}

void noise_B(at::Tensor B, at::Tensor EBR, at::Tensor EAR, at::Tensor EBL,
             at::Tensor BpEB, at::Tensor EARxBpEB,
             int64_t N, int64_t K, int64_t R) {
  TORCH_CHECK(B.is_cuda(), "noise_B: tensors must be CUDA");
  const c10::cuda::CUDAGuard device_guard(B.device());
  launch_noise_B(
      B.data_ptr<int8_t>(), EBR.data_ptr<int8_t>(), EAR.data_ptr<int8_t>(),
      EBL.data_ptr<int8_t>(), BpEB.data_ptr<int8_t>(),
      EARxBpEB.data_ptr<int32_t>(), (int)N, (int)K, (int)R, cur_stream());
}

void denoise_converter(c10::optional<at::Tensor> EARxBpEB_in,
                       c10::optional<at::Tensor> AxEBL_in,
                       c10::optional<at::Tensor> EARxBpEB_out,
                       c10::optional<at::Tensor> AxEBL_out,
                       int64_t M, int64_t N, int64_t R) {
  c10::optional<c10::cuda::CUDAGuard> device_guard;
  if (AxEBL_in) device_guard.emplace(AxEBL_in->device());
  else if (EARxBpEB_in) device_guard.emplace(EARxBpEB_in->device());
  const int32_t* ear_in =
      EARxBpEB_in ? EARxBpEB_in->data_ptr<int32_t>() : nullptr;
  const int32_t* axebl_in =
      AxEBL_in ? AxEBL_in->data_ptr<int32_t>() : nullptr;
  half* ear_out = EARxBpEB_out ? half_ptr(*EARxBpEB_out) : nullptr;
  half* axebl_out = AxEBL_out ? half_ptr(*AxEBL_out) : nullptr;
  launch_denoise_converter(
      ear_in, axebl_in, ear_out, axebl_out,
      (int)M, (int)N, (int)R, cur_stream());
}

at::Tensor inner_hash(at::Tensor input_buffer, int64_t iterations) {
  TORCH_CHECK(input_buffer.is_cuda(), "inner_hash: input must be CUDA");
  const c10::cuda::CUDAGuard device_guard(input_buffer.device());
  auto out = at::empty({1}, input_buffer.options());
  launch_inner_hash_kernel(
      reinterpret_cast<uint32_t*>(input_buffer.data_ptr()),
      (int)input_buffer.numel(),
      reinterpret_cast<uint32_t*>(out.data_ptr()),
      iterations, cur_stream());
  return out;
}

void tensor_hash_py(at::Tensor data, at::Tensor key, at::Tensor out,
                    at::Tensor roots, int64_t threads_per_block,
                    int64_t num_stages, int64_t leaves_per_mt_block) {
  TORCH_CHECK(data.is_cuda() && key.is_cuda() && out.is_cuda() && roots.is_cuda(),
              "tensor_hash: all tensors must be CUDA");
  TORCH_CHECK(data.is_contiguous(), "tensor_hash: data must be contiguous");
  TORCH_CHECK(key.scalar_type() == at::kByte && out.scalar_type() == at::kByte &&
                  roots.scalar_type() == at::kByte,
              "tensor_hash: key/out/roots must be uint8");
  TORCH_CHECK(key.numel() == 32, "tensor_hash: key must be 32 bytes");
  TORCH_CHECK(out.numel() == 32, "tensor_hash: out must be 32 bytes");

  const c10::cuda::CUDAGuard device_guard(data.device());

  const uint32_t data_size = static_cast<uint32_t>(data.nbytes());
  const uint32_t chunk_size = 1024u;
  const uint32_t num_chunks = (data_size + chunk_size - 1) / chunk_size;
  const uint32_t num_blocks =
      (num_chunks + (uint32_t)threads_per_block - 1) / (uint32_t)threads_per_block;
  TORCH_CHECK((uint32_t)(roots.nbytes()) >= num_blocks * 32u,
              "tensor_hash: roots scratch too small; need ", num_blocks * 32u,
              " bytes");

  cudaDeviceProp* dprops = at::cuda::getCurrentDeviceProperties();
  tensor_hash(reinterpret_cast<const uint8_t*>(data.data_ptr()), data_size,
              out.data_ptr<uint8_t>(), key.data_ptr<uint8_t>(), num_blocks,
              (uint32_t)threads_per_block, (uint32_t)num_stages,
              (uint32_t)leaves_per_mt_block, roots.data_ptr<uint8_t>(), *dprops,
              cur_stream());
}

// Returns {digests[num_tiles,32] uint8, found[1] int32, coord[2] int32}.
// A: [m,k] int8 noised; Bt: [n,k] int8 noised (B transposed). pow_key/pow_target:
// 32-byte uint8 tensors (pow_target little-endian uint256).
std::vector<at::Tensor> pearl_pow(at::Tensor A, at::Tensor Bt,
                                  at::Tensor pow_key, at::Tensor pow_target,
                                  int64_t R) {
  TORCH_CHECK(A.is_cuda() && Bt.is_cuda(), "pearl_pow: A/Bt must be CUDA");
  TORCH_CHECK(A.scalar_type() == at::kChar && Bt.scalar_type() == at::kChar,
              "pearl_pow: A/Bt must be int8");
  TORCH_CHECK(A.is_contiguous() && Bt.is_contiguous(), "pearl_pow: A/Bt must be contiguous");
  TORCH_CHECK(pow_key.numel() == 32 && pow_target.numel() == 32,
              "pearl_pow: pow_key/pow_target must be 32 bytes");
  const int m = (int)A.size(0), k = (int)A.size(1), n = (int)Bt.size(0);
  TORCH_CHECK(Bt.size(1) == k, "pearl_pow: A[k] must match Bt[k]");
  TORCH_CHECK(m % 16 == 0 && n % 16 == 0 && k % R == 0,
              "pearl_pow: require m%16==0, n%16==0, k%R==0");
  TORCH_CHECK(R == 256 || R == 128 || R == 64, "pearl_pow: R must be 64, 128, or 256");

  const c10::cuda::CUDAGuard device_guard(A.device());
  const int num_tiles = (m / 16) * (n / 16);
  auto u8 = at::TensorOptions().dtype(at::kByte).device(A.device());
  auto i32 = at::TensorOptions().dtype(at::kInt).device(A.device());
  auto digests = at::empty({num_tiles, 32}, u8);
  auto found = at::zeros({1}, i32);
  auto coord = at::full({2}, -1, i32);

  launch_pearl_pow(
      A.data_ptr<int8_t>(), Bt.data_ptr<int8_t>(), m, n, k, (int)R,
      reinterpret_cast<const uint32_t*>(pow_key.data_ptr()),
      reinterpret_cast<const uint32_t*>(pow_target.data_ptr()),
      digests.data_ptr<uint8_t>(), found.data_ptr<int>(), coord.data_ptr<int>(),
      cur_stream());
  return {digests, found, coord};
}

// Fused variant: same outputs/semantics as pearl_pow, but requires the block
// region to divide m/n (WM=WN=4 -> m%64==0, n%64==0).
std::vector<at::Tensor> pearl_pow_fused(at::Tensor A, at::Tensor Bt,
                                        at::Tensor pow_key, at::Tensor pow_target,
                                        int64_t R, int64_t variant) {
  TORCH_CHECK(A.is_cuda() && Bt.is_cuda(), "pearl_pow_fused: A/Bt must be CUDA");
  TORCH_CHECK(A.scalar_type() == at::kChar && Bt.scalar_type() == at::kChar,
              "pearl_pow_fused: A/Bt must be int8");
  TORCH_CHECK(A.is_contiguous() && Bt.is_contiguous(),
              "pearl_pow_fused: A/Bt must be contiguous");
  TORCH_CHECK(pow_key.numel() == 32 && pow_target.numel() == 32,
              "pearl_pow_fused: pow_key/pow_target must be 32 bytes");
  const int m = (int)A.size(0), k = (int)A.size(1), n = (int)Bt.size(0);
  TORCH_CHECK(Bt.size(1) == k, "pearl_pow_fused: A[k] must match Bt[k]");
  TORCH_CHECK(m % 64 == 0 && n % 64 == 0 && k % R == 0,
              "pearl_pow_fused: require m%64==0, n%64==0, k%R==0");
  TORCH_CHECK(R == 256 || R == 128 || R == 64,
              "pearl_pow_fused: R must be 64, 128, or 256");

  const c10::cuda::CUDAGuard device_guard(A.device());
  const int num_tiles = (m / 16) * (n / 16);
  auto u8 = at::TensorOptions().dtype(at::kByte).device(A.device());
  auto i32 = at::TensorOptions().dtype(at::kInt).device(A.device());
  auto digests = at::empty({num_tiles, 32}, u8);
  auto found = at::zeros({1}, i32);
  auto coord = at::full({2}, -1, i32);

  launch_pearl_pow_fused_v(
      A.data_ptr<int8_t>(), Bt.data_ptr<int8_t>(), m, n, k, (int)R,
      reinterpret_cast<const uint32_t*>(pow_key.data_ptr()),
      reinterpret_cast<const uint32_t*>(pow_target.data_ptr()),
      digests.data_ptr<uint8_t>(), found.data_ptr<int>(), coord.data_ptr<int>(),
      (int)variant, cur_stream());
  return {digests, found, coord};
}

// Generate the Pearl noise tensors on the GPU from the commitment keys.
// Returns {EAL[m,R], EAR[R,k], EBL[k,R], EBR[n,R]} int8 (matches Python
// generate_noise: EAL/EBR uniform, EAR/EBL sparse-perm; seeds "A_tensor"/"B_tensor").
std::vector<at::Tensor> noise_gen(at::Tensor key_A, at::Tensor key_B,
                                  int64_t m, int64_t n, int64_t k, int64_t R) {
  TORCH_CHECK(key_A.is_cuda() && key_B.is_cuda(), "noise_gen: keys must be CUDA");
  TORCH_CHECK(key_A.scalar_type() == at::kByte && key_B.scalar_type() == at::kByte,
              "noise_gen: keys must be uint8");
  TORCH_CHECK(key_A.numel() == 32 && key_B.numel() == 32,
              "noise_gen: keys must be 32 bytes");
  const c10::cuda::CUDAGuard device_guard(key_A.device());
  auto i8 = at::TensorOptions().dtype(at::kChar).device(key_A.device());
  auto EAL = at::empty({m, R}, i8);
  auto EAR = at::empty({R, k}, i8);
  auto EBL = at::empty({k, R}, i8);
  auto EBR = at::empty({n, R}, i8);
  launch_noise_gen(
      EAL.data_ptr<int8_t>(), EAR.data_ptr<int8_t>(), EBL.data_ptr<int8_t>(),
      EBR.data_ptr<int8_t>(),
      reinterpret_cast<const uint8_t*>(key_A.data_ptr()),
      reinterpret_cast<const uint8_t*>(key_B.data_ptr()),
      (int)m, (int)n, (int)k, (int)R, cur_stream());
  return {EAL, EAR, EBL, EBR};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Pascal (sm_61) Pearl GEMM CUDA kernels";
  m.def("dp4a_gemm", &dp4a_gemm, "INT8 DP4A GEMM (C = A @ B^T, dequantized)");
  m.def("noise_A", &noise_A, "Pearl noise A kernel");
  m.def("noise_gen", &noise_gen,
        "Generate Pearl noise EAL/EAR/EBL/EBR on GPU from commitment keys",
        pybind11::arg("key_A"), pybind11::arg("key_B"),
        pybind11::arg("m"), pybind11::arg("n"), pybind11::arg("k"),
        pybind11::arg("R"));
  m.def("noise_B", &noise_B, "Pearl noise B kernel");
  m.def("denoise_converter", &denoise_converter,
        "int32 -> fp16 denoise conversion");
  m.def("inner_hash", &inner_hash, "PoW inner hash (XOR reduction)");
  m.def("pearl_pow", &pearl_pow,
        "Pascal Pearl PoW: per-16x16-tile transcript + keyed BLAKE3 vs target",
        pybind11::arg("A"), pybind11::arg("Bt"), pybind11::arg("pow_key"),
        pybind11::arg("pow_target"), pybind11::arg("R") = 128);
  m.def("pearl_pow_fused", &pearl_pow_fused,
        "Fused high-throughput Pearl PoW (warp-per-tile, shared-mem reuse)",
        pybind11::arg("A"), pybind11::arg("Bt"), pybind11::arg("pow_key"),
        pybind11::arg("pow_target"), pybind11::arg("R") = 256,
        pybind11::arg("variant") = 0);
  m.def("tensor_hash", &tensor_hash_py,
        "BLAKE3 keyed Merkle hash of a tensor (Pascal)",
        pybind11::arg("data"), pybind11::arg("key"), pybind11::arg("out"),
        pybind11::arg("roots"), pybind11::arg("threads_per_block") = 128,
        pybind11::arg("num_stages") = 2,
        pybind11::arg("leaves_per_mt_block") = 512);
}
