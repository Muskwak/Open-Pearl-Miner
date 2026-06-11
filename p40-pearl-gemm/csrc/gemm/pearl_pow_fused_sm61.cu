// Pascal (sm_61) FUSED Pearl PoW kernel — high-throughput variant.
//
// Same math as pearl_pow_sm61.cu (per-16x16-tile cumulative DP4A contraction,
// per-k-chunk XOR of the cumulative tile folded into a 16-word transcript, keyed
// BLAKE3 of the transcript, compare <= target) but restructured for throughput:
//
//   * ONE WARP per 16x16 hash-tile (not one block) — the 256 tile cells are held
//     in registers, 8 per lane.
//   * A block computes a (16*WM) x (16*WN) output region = WM*WN hash-tiles, and
//     stages A[16*WM, R] and B[16*WN, R] in shared memory ONCE per k-chunk, so
//     every tile in the block reuses the loaded operands (WN-/WM-fold reuse).
//   * The per-chunk XOR reduction of the cumulative tile uses warp __shfl_xor
//     (5 shuffles, warp-synchronous) instead of a 128-barrier shared-mem tree.
//
// Bit-exact with the naive kernel: accumulation is identical, XOR is
// associative/commutative so the per-lane-then-warp reduction order is
// irrelevant, and the transcript / rotl / keyed-BLAKE3 are unchanged.
//
// Requires m % (16*WM) == 0, n % (16*WN) == 0, k % R == 0, R % 4 == 0.

#include <cuda_runtime.h>
#include <cstdint>

#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"

using namespace cute;

static __device__ __forceinline__ int dp4a_f(int a, int b, int c) {
  int r;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
  asm volatile("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
#else
  r = c;
  for (int i = 0; i < 4; ++i)
    r += int((int8_t)((a >> (i * 8)) & 0xFF)) * int((int8_t)((b >> (i * 8)) & 0xFF));
#endif
  return r;
}

static __device__ __forceinline__ uint32_t rotl32_f(uint32_t x, int n) {
  return (x << n) | (x >> (32 - n));
}

static constexpr int HT = 16;            // hash tile dimension
static constexpr int HASH_ROT = 13;      // HASH_ACCUMULATE_ROTATION
static constexpr int TRANSCRIPT_U32 = 16;
static constexpr int ELT_PER_LANE = (HT * HT) / 32;  // 256/32 = 8

template <int R, int WM, int WN, int MINB>
__global__ void __launch_bounds__(WM* WN * 32, MINB) pearl_pow_fused_kernel(
    const int8_t* __restrict__ A,    // [m, k] noised
    const int8_t* __restrict__ Bt,   // [n, k] noised (B transposed)
    int n, int k,
    const uint32_t* __restrict__ pow_key,     // 8 words
    const uint32_t* __restrict__ pow_target,  // 8 words, little-endian
    uint8_t* __restrict__ out_digests,        // [num_tiles, 32] or nullptr
    int* __restrict__ found_flag,             // atomic, may be nullptr
    int* __restrict__ found_coord) {          // [2] = (row,col) of winning tile

  constexpr int ROWS_A = HT * WM;   // A rows staged per block
  constexpr int ROWS_B = HT * WN;   // Bt rows (= output cols) staged per block
  constexpr int RW = R / 4;         // ints (dp4a words) per row
  // Pad the shared row stride so it is NOT a multiple of 32: with stride RW (=64
  // for R=256) every lane of a warp indexes the same bank (row*RW % 32 == 0),
  // a 32-way bank conflict on every load. RW+1 is coprime to 32 -> conflict-free.
  constexpr int SAW = RW + 1;

  const int tiles_w = n / HT;
  const int blocks_n = tiles_w / WN;
  const int block_row = blockIdx.x / blocks_n;
  const int block_col = blockIdx.x % blocks_n;
  const int row_base = block_row * ROWS_A;   // first A row of this block
  const int col_base = block_col * ROWS_B;   // first Bt row of this block

  const int tid = threadIdx.x;
  const int warp = tid >> 5;          // 0..WM*WN-1
  const int lane = tid & 31;
  const int wm = warp / WN;           // tile row within block
  const int wn = warp % WN;           // tile col within block
  const int aRow0 = wm * HT;          // base row in sA for this warp's tile
  const int bRow0 = wn * HT;          // base row in sB for this warp's tile

  int acc[ELT_PER_LANE];
#pragma unroll
  for (int e = 0; e < ELT_PER_LANE; ++e) acc[e] = 0;
  uint32_t transcript[TRANSCRIPT_U32];
#pragma unroll
  for (int e = 0; e < TRANSCRIPT_U32; ++e) transcript[e] = 0u;

  __shared__ int sAi[ROWS_A * SAW];
  __shared__ int sBi[ROWS_B * SAW];
  const int* Ai = reinterpret_cast<const int*>(A);
  const int* Bi = reinterpret_cast<const int*>(Bt);

  const int T = k / R;
  for (int t = 0; t < T; ++t) {
    const int koff4 = (t * R) / 4;            // int offset of this chunk within a row
    __syncthreads();
    // stage A[row_base.., chunk] and Bt[col_base.., chunk] as ints (coalesced),
    // into padded shared rows (stride SAW) to avoid bank conflicts on read.
    for (int i = tid; i < ROWS_A * RW; i += blockDim.x) {
      const int r = i / RW, c4 = i % RW;
      sAi[r * SAW + c4] = Ai[(size_t)(row_base + r) * (k / 4) + koff4 + c4];
    }
    for (int i = tid; i < ROWS_B * RW; i += blockDim.x) {
      const int r = i / RW, c4 = i % RW;
      sBi[r * SAW + c4] = Bi[(size_t)(col_base + r) * (k / 4) + koff4 + c4];
    }
    __syncthreads();

    // Register-blocked RM x RN micro-tile per lane: per kk load RM A-ints + RN
    // B-ints and do RM*RN dp4a (outer product). Shared loads per dp4a drop from
    // 2 to (RM+RN)/(RM*RN)=0.75, with RM*RN independent accumulators for ILP.
    // The 32 lanes tile the 16x16 hash-tile as (16/RM) x (16/RN) = 4 x 8 blocks.
    constexpr int RM = 4, RN = 2;            // RM*RN == ELT_PER_LANE (8)
    const int mtr = lane >> 3;               // micro-tile row 0..3
    const int mtc = lane & 7;                // micro-tile col 0..7
    const int* ar[RM];
    const int* br[RN];
#pragma unroll
    for (int i = 0; i < RM; ++i) ar[i] = &sAi[(aRow0 + mtr * RM + i) * SAW];
#pragma unroll
    for (int j = 0; j < RN; ++j) br[j] = &sBi[(bRow0 + mtc * RN + j) * SAW];
#pragma unroll
    for (int kk = 0; kk < RW; ++kk) {
      int a[RM], b[RN];
#pragma unroll
      for (int i = 0; i < RM; ++i) a[i] = ar[i][kk];
#pragma unroll
      for (int j = 0; j < RN; ++j) b[j] = br[j][kk];
#pragma unroll
      for (int i = 0; i < RM; ++i)
#pragma unroll
        for (int j = 0; j < RN; ++j)
          acc[i * RN + j] = dp4a_f(a[i], b[j], acc[i * RN + j]);  // cumulative
    }
    uint32_t lx = 0u;
#pragma unroll
    for (int e = 0; e < ELT_PER_LANE; ++e) lx ^= (uint32_t)acc[e];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      lx ^= __shfl_xor_sync(0xffffffffu, lx, off);
    if (lane == 0) {
      const int idx = t % TRANSCRIPT_U32;
      transcript[idx] = rotl32_f(transcript[idx], HASH_ROT) ^ lx;
    }
  }

  if (lane != 0) return;

  Tensor block = make_tensor<uint32_t>(Int<TRANSCRIPT_U32>{});
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < TRANSCRIPT_U32; ++i) block(i) = transcript[i];
  Tensor cv = make_tensor<uint32_t>(Int<blake3::CHAINING_VALUE_SIZE_U32>{});
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) cv(i) = pow_key[i];

  blake3::compress_msg_block_u32(block, cv,
                                 blake3::COMPRESS_PARAMS_SINGLE_BLOCK_KEYED);

  const int gi = row_base + aRow0;   // global tile row
  const int gj = col_base + bRow0;   // global tile col
  if (out_digests) {
    const int tile_id = (gi / HT) * (n / HT) + (gj / HT);
    uint32_t* od = reinterpret_cast<uint32_t*>(out_digests + (size_t)tile_id * 32);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < blake3::CHAINING_VALUE_SIZE_U32; ++i) od[i] = cv(i);
  }
  if (found_flag) {
    bool le = true;
    CUTLASS_PRAGMA_UNROLL
    for (int i = blake3::CHAINING_VALUE_SIZE_U32 - 1; i >= 0; --i) {
      uint32_t h = cv(i), tg = pow_target[i];
      if (h > tg) { le = false; break; }
      if (h < tg) break;
    }
    if (le && atomicCAS(found_flag, 0, 1) == 0 && found_coord) {
      found_coord[0] = gi;
      found_coord[1] = gj;
    }
  }
}

template <int R, int WM, int WN, int MINB>
static void launch_cfg(const int8_t* A, const int8_t* Bt, int m, int n, int k,
                       const uint32_t* pow_key, const uint32_t* pow_target,
                       uint8_t* out_digests, int* found_flag, int* found_coord,
                       cudaStream_t stream) {
  const int num_block_tiles = (m / (HT * WM)) * (n / (HT * WN));
  dim3 grid(num_block_tiles);
  dim3 block(WM * WN * 32);
  pearl_pow_fused_kernel<R, WM, WN, MINB><<<grid, block, 0, stream>>>(
      A, Bt, n, k, pow_key, pow_target, out_digests, found_flag, found_coord);
}

// `variant` selects the block-shape / occupancy config (for tuning).
// 0 = 4x4 MINB1 (baseline), 1 = 4x4 MINB2, 2 = 2x2 MINB4, 3 = 2x4 MINB3.
void launch_pearl_pow_fused_v(
    const int8_t* A, const int8_t* Bt, int m, int n, int k, int R,
    const uint32_t* pow_key, const uint32_t* pow_target,
    uint8_t* out_digests, int* found_flag, int* found_coord,
    int variant, cudaStream_t stream) {
  if (R != 256) {  // small-R kept only at the baseline shape
    if (R == 128)
      launch_cfg<128, 4, 4, 1>(A, Bt, m, n, k, pow_key, pow_target, out_digests, found_flag, found_coord, stream);
    else if (R == 64)
      launch_cfg<64, 4, 4, 1>(A, Bt, m, n, k, pow_key, pow_target, out_digests, found_flag, found_coord, stream);
    return;
  }
  switch (variant) {
    case 1:
      launch_cfg<256, 4, 4, 2>(A, Bt, m, n, k, pow_key, pow_target, out_digests, found_flag, found_coord, stream); break;
    case 2:
      launch_cfg<256, 2, 2, 4>(A, Bt, m, n, k, pow_key, pow_target, out_digests, found_flag, found_coord, stream); break;
    case 3:
      launch_cfg<256, 2, 4, 3>(A, Bt, m, n, k, pow_key, pow_target, out_digests, found_flag, found_coord, stream); break;
    case 4:
      launch_cfg<256, 4, 4, 3>(A, Bt, m, n, k, pow_key, pow_target, out_digests, found_flag, found_coord, stream); break;
    case 5:
      launch_cfg<256, 4, 4, 4>(A, Bt, m, n, k, pow_key, pow_target, out_digests, found_flag, found_coord, stream); break;
    case 6:
      launch_cfg<256, 2, 4, 4>(A, Bt, m, n, k, pow_key, pow_target, out_digests, found_flag, found_coord, stream); break;
    default:
      launch_cfg<256, 4, 4, 1>(A, Bt, m, n, k, pow_key, pow_target, out_digests, found_flag, found_coord, stream); break;
  }
}

void launch_pearl_pow_fused(
    const int8_t* A, const int8_t* Bt, int m, int n, int k, int R,
    const uint32_t* pow_key, const uint32_t* pow_target,
    uint8_t* out_digests, int* found_flag, int* found_coord,
    cudaStream_t stream) {
  launch_pearl_pow_fused_v(A, Bt, m, n, k, R, pow_key, pow_target,
                           out_digests, found_flag, found_coord, 0, stream);
}
