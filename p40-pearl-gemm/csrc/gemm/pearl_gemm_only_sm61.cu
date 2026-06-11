// Pascal (sm_61) GEMM-ONLY Pearl kernel — high-throughput, no BLAKE3.
//
// Same fused warp/tile structure as pearl_pow_fused_sm61.cu (shared-mem operand
// reuse, register-blocked micro-tiles, __shfl_xor reduction, rotl-xor transcript
// accumulation) but DOES NOT compute BLAKE3. Instead it writes the 16-word
// transcript per hash tile to a global buffer (transcript_buffer[num_tiles, 16]).
//
// Without the keyed-BLAKE3 ~60 registers the thread register footprint drops to
// ~40, letting us raise occupancy from MINB=2 (50 %) to MINB=3 (75 %).
//
// A companion kernel (pearl_blake3_sm61.cu) consumes the transcript buffer.
//
// Bit-exact transcript: the GEMM loop, XOR reduction and rotl-xor accumulation
// are identical to pearl_pow_fused_sm61.cu, so the 16-word transcript written
// to global memory is byte-identical.

#include <cuda_runtime.h>
#include <cstdint>

#include <cute/tensor.hpp>
#include "blake3/blake3.cuh"

using namespace cute;

static __device__ __forceinline__ int dp4a_go(int a, int b, int c) {
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

static __device__ __forceinline__ uint32_t rotl32_go(uint32_t x, int n) {
  return (x << n) | (x >> (32 - n));
}

static constexpr int HT_GO = 16;
static constexpr int HASH_ROT_GO = 13;
static constexpr int TRANSCRIPT_U32_GO = 16;
static constexpr int ELT_PER_LANE_GO = (HT_GO * HT_GO) / 32;  // 8

// S = shared-memory staging width (a divisor of R). The transcript still samples
// the running accumulator at every R-column boundary (bit-exact), but operands
// are staged S columns at a time, so the shared footprint scales with S, not R.
// At R=256 a full-width stage is ~33 KB/block -> only 2 blocks/SM (shared-mem
// limited, NOT register limited). S=128 halves that to ~17 KB -> 3-4 blocks/SM.
template <int R, int S, int WM, int WN, int MINB>
__global__ void __launch_bounds__(WM* WN * 32, MINB) pearl_gemm_only_kernel(
    const int8_t* __restrict__ A,     // [m, k] noised
    const int8_t* __restrict__ Bt,    // [n, k] noised (B transposed)
    int n, int k,
    uint32_t* __restrict__ transcript_buffer)  // [num_tiles, 16] output

{
  constexpr int ROWS_A = HT_GO * WM;
  constexpr int ROWS_B = HT_GO * WN;
  constexpr int SW = S / 4;          // dp4a words per staged sub-chunk
  constexpr int SUB = R / S;         // sub-chunks per R-wide reduction window
  static_assert(R % S == 0, "S must divide R");
  static_assert(SW % 2 == 0, "SW must be even for int2 shared loads");
  // Even stride so int2 (LDS.64) loads are 8-byte aligned at every even kk.
  // Bank check (32 banks): inner-loop bank = (2*row + kk) mod 32. A-reads hit 4
  // distinct rows R0,R0+4,R0+8,R0+12 -> banks 2R0+{0,8,16,24} (distinct);
  // B-reads step rows by 2 -> banks {0,4,..,28} (distinct). Conflict-free, and
  // int2 halves the inner-loop LDS instruction count vs scalar LDS.32.
  constexpr int SAW = SW + 2;

  const int tiles_w = n / HT_GO;
  const int blocks_n = tiles_w / WN;
  const int block_row = blockIdx.x / blocks_n;
  const int block_col = blockIdx.x % blocks_n;
  const int row_base = block_row * ROWS_A;
  const int col_base = block_col * ROWS_B;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wm = warp / WN;
  const int wn = warp % WN;
  const int aRow0 = wm * HT_GO;
  const int bRow0 = wn * HT_GO;

  int acc[ELT_PER_LANE_GO];
#pragma unroll
  for (int e = 0; e < ELT_PER_LANE_GO; ++e) acc[e] = 0;

  __shared__ __align__(16) int sAi[ROWS_A * SAW];
  __shared__ __align__(16) int sBi[ROWS_B * SAW];
  // Per-warp transcript in shared memory. Only lane 0 of each warp touches its
  // row, so keeping the 16-word transcript here (instead of a per-thread
  // register array the other 31 lanes carry dead) frees ~16 regs/thread —
  // enough to run MINB4 (100% occupancy) without register spills.
  __shared__ uint32_t sT[WM * WN][TRANSCRIPT_U32_GO];
  if (lane == 0) {
#pragma unroll
    for (int e = 0; e < TRANSCRIPT_U32_GO; ++e) sT[warp][e] = 0u;
  }
  const int* Ai = reinterpret_cast<const int*>(A);
  const int* Bi = reinterpret_cast<const int*>(Bt);

  // Per-thread micro-tile shared pointers (constant across all chunks).
  constexpr int RM = 4, RN = 2;
  const int mtr = lane >> 3;
  const int mtc = lane & 7;
  const int* ar[RM];
  const int* br[RN];
#pragma unroll
  for (int i = 0; i < RM; ++i) ar[i] = &sAi[(aRow0 + mtr * RM + i) * SAW];
#pragma unroll
  for (int j = 0; j < RN; ++j) br[j] = &sBi[(bRow0 + mtc * RN + j) * SAW];

  const int T = k / R;
  int ts = 0;  // global sub-chunk index
  for (int t = 0; t < T; ++t) {
#pragma unroll
    for (int sub = 0; sub < SUB; ++sub, ++ts) {
      const int koff4 = ts * SW;
      __syncthreads();
      for (int i = tid; i < ROWS_A * SW; i += blockDim.x) {
        const int r = i / SW, c4 = i % SW;
        sAi[r * SAW + c4] = Ai[(size_t)(row_base + r) * (k / 4) + koff4 + c4];
      }
      for (int i = tid; i < ROWS_B * SW; i += blockDim.x) {
        const int r = i / SW, c4 = i % SW;
        sBi[r * SAW + c4] = Bi[(size_t)(col_base + r) * (k / 4) + koff4 + c4];
      }
      __syncthreads();

      // int2 (LDS.64) loads: fetch 2 dp4a-words per operand per instruction,
      // halving the inner-loop shared-load instruction count. SAW=SW+2 keeps
      // these 8-byte aligned and bank-conflict-free (see SAW note above).
#pragma unroll
      for (int kk = 0; kk < SW; kk += 2) {
        int2 a2[RM], b2[RN];
#pragma unroll
        for (int i = 0; i < RM; ++i)
          a2[i] = *reinterpret_cast<const int2*>(&ar[i][kk]);
#pragma unroll
        for (int j = 0; j < RN; ++j)
          b2[j] = *reinterpret_cast<const int2*>(&br[j][kk]);
#pragma unroll
        for (int i = 0; i < RM; ++i)
#pragma unroll
          for (int j = 0; j < RN; ++j) {
            int acumm = dp4a_go(a2[i].x, b2[j].x, acc[i * RN + j]);
            acc[i * RN + j] = dp4a_go(a2[i].y, b2[j].y, acumm);
          }
      }
    }
    // R-wide reduction window complete: sample the running accumulator.
    uint32_t lx = 0u;
#pragma unroll
    for (int e = 0; e < ELT_PER_LANE_GO; ++e) lx ^= (uint32_t)acc[e];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      lx ^= __shfl_xor_sync(0xffffffffu, lx, off);
    if (lane == 0) {
      const int idx = t % TRANSCRIPT_U32_GO;
      sT[warp][idx] = rotl32_go(sT[warp][idx], HASH_ROT_GO) ^ lx;
    }
  }

  if (lane != 0) return;

  const int gi = row_base + aRow0;
  const int gj = col_base + bRow0;
  const int tile_id = (gi / HT_GO) * tiles_w + (gj / HT_GO);
  uint32_t* tb = &transcript_buffer[(size_t)tile_id * TRANSCRIPT_U32_GO];
#pragma unroll
  for (int i = 0; i < TRANSCRIPT_U32_GO; ++i) tb[i] = sT[warp][i];
}

template <int R, int S, int WM, int WN, int MINB>
static void launch_go_cfg(const int8_t* A, const int8_t* Bt, int m, int n, int k,
                          uint32_t* transcript_buffer, cudaStream_t stream) {
  const int num_block_tiles = (m / (HT_GO * WM)) * (n / (HT_GO * WN));
  dim3 grid(num_block_tiles);
  dim3 block(WM * WN * 32);
  pearl_gemm_only_kernel<R, S, WM, WN, MINB><<<grid, block, 0, stream>>>(
      A, Bt, n, k, transcript_buffer);
}

// Variant dispatch for the GEMM-only kernel. S = staging width (divides R); the
// shared footprint scales with S, so a narrower S unlocks higher occupancy at
// R=256 (which is shared-mem-bound at full S=256: ~33 KB -> 2 blocks/SM).
//   v=0 -> S=128 4×4 MINB3   (half shared mem -> aim 3 blocks/SM)
//   v=1 -> S=128 4×4 MINB4   (push 4 blocks/SM if registers allow)
//   v=2 -> S=64  4×4 MINB4   (quarter shared mem)
//   v=3 -> S=256 4×4 MINB3   (full-width baseline = old best)
//   v=4 -> S=128 4×4 MINB2
//   v=5 -> S=64  4×4 MINB3
//   v=6 -> S=128 2×4 MINB4
void launch_pearl_gemm_only(
    const int8_t* A, const int8_t* Bt, int m, int n, int k, int R,
    uint32_t* transcript_buffer, int variant, cudaStream_t stream) {
  if (R == 256) {
    switch (variant) {
      case 0:
        launch_go_cfg<256, 128, 4, 4, 3>(A, Bt, m, n, k, transcript_buffer, stream); break;
      case 1:
        launch_go_cfg<256, 128, 4, 4, 4>(A, Bt, m, n, k, transcript_buffer, stream); break;
      case 2:
        launch_go_cfg<256, 64, 4, 4, 4>(A, Bt, m, n, k, transcript_buffer, stream); break;
      case 3:
        launch_go_cfg<256, 256, 4, 4, 3>(A, Bt, m, n, k, transcript_buffer, stream); break;
      case 4:
        launch_go_cfg<256, 128, 4, 4, 2>(A, Bt, m, n, k, transcript_buffer, stream); break;
      case 5:
        launch_go_cfg<256, 64, 4, 4, 3>(A, Bt, m, n, k, transcript_buffer, stream); break;
      case 6:
        launch_go_cfg<256, 128, 2, 4, 4>(A, Bt, m, n, k, transcript_buffer, stream); break;
      case 7:  // 8x2 (512 thr): WN=2 less A-reuse, WM=8 more B-reuse
        launch_go_cfg<256, 128, 8, 2, 4>(A, Bt, m, n, k, transcript_buffer, stream); break;
      case 8:  // 8x4 (1024 thr): big reuse, ~25KB smem -> 2 blocks = 100%
        launch_go_cfg<256, 128, 8, 4, 2>(A, Bt, m, n, k, transcript_buffer, stream); break;
      case 9:  // 4x8 (1024 thr)
        launch_go_cfg<256, 128, 4, 8, 2>(A, Bt, m, n, k, transcript_buffer, stream); break;
      default:
        launch_go_cfg<256, 128, 4, 4, 3>(A, Bt, m, n, k, transcript_buffer, stream); break;
    }
  } else if (R == 128) {
    launch_go_cfg<128, 128, 4, 4, 3>(A, Bt, m, n, k, transcript_buffer, stream);
  } else if (R == 64) {
    launch_go_cfg<64, 64, 4, 4, 3>(A, Bt, m, n, k, transcript_buffer, stream);
  }
}
