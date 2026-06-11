#include "noise_generation_host.h"
#include "pearl_api_params.h"

template void run_noise_generation<128, 32>(Noise_gen_params&, cudaStream_t);
template void run_noise_generation<128, 64>(Noise_gen_params&, cudaStream_t);
template void run_noise_generation<128, 128>(Noise_gen_params&, cudaStream_t);
template void run_noise_generation<64, 32>(Noise_gen_params&, cudaStream_t);
template void run_noise_generation<64, 64>(Noise_gen_params&, cudaStream_t);
template void run_noise_generation<64, 128>(Noise_gen_params&, cudaStream_t);
template void run_noise_generation<256, 128>(Noise_gen_params&, cudaStream_t);

// Host launcher (plain C++ linkage) so bindings.cpp can drive the generator.
// Fills the noise tensors the miner needs:
//   EAL [m,R], EAR [R,k] (R-major), EBL [k,R] (K-major), EBR [n,R]
// from the keyed-BLAKE3 seeds (key_A, key_B = the commitment hashes).
void launch_noise_gen(
    int8_t* EAL, int8_t* EAR, int8_t* EBL, int8_t* EBR,
    const uint8_t* key_A, const uint8_t* key_B,
    int m, int n, int k, int R, cudaStream_t stream) {
  Noise_gen_params p{};
  p.ptr_EAL = EAL;
  // EAR wanted as [R,k] (k contiguous) = the kernel's K_major output.
  // EBL wanted as [k,R] (R contiguous) = the kernel's R_major output.
  p.ptr_EAR_K_major = EAR;
  p.ptr_EBL_R_major = EBL;
  p.ptr_EBR = EBR;
  p.m = m;
  p.n = n;
  p.k = k;
  p.r = R;
  p.ptr_key_A = const_cast<uint8_t*>(key_A);
  p.ptr_key_B = const_cast<uint8_t*>(key_B);
  switch (R) {
    case 256: run_noise_generation<256, 128>(p, stream); break;
    case 128: run_noise_generation<128, 128>(p, stream); break;
    case 64:  run_noise_generation<64, 128>(p, stream); break;
    default: break;
  }
}
