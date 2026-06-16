// pearl_ampere_tc.cu — Production Ampere (sm_80+) Pearl GEMM kernel
// Bit-exact transcript with Pascal sm61 dp4a kernel.
// Uses direct register packing (NO ldmatrix) — CuTe ALayout/BLayout bit
// decomposition for m16n8k32 int8 tensor-core.
//
// Target any sm_80+ GPU (compile with -arch=sm_86, sm_89, sm_90, etc.).
// Compile: nvcc -arch=sm_89 -O3 -std=c++17 -c pearl_ampere_tc.cu

#include <cuda_runtime.h>
#include <cstdint>

// ==================================================================
// PTX helper functions — device-only (use asm which requires sm_80+)
// ==================================================================
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

__device__ __forceinline__ void mma_m16n8k32(
    int32_t d[4], const uint32_t a[4], const uint32_t b[2], const int32_t c[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
}

__device__ __forceinline__ void cp_async_16B(void* smem, const void* gmem) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)), "l"(gmem)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

template <int n>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(n));
}

// ====== HARDWARE REGISTER-TO-MATRIX MAPPING FOR m16n8k32.s8 ======
//
// Each warp runs 32 threads, split into 8 groups of 4 by tid_y:
//   tid_x = lane & 3   (0..3)
//   tid_y = lane >> 2  (0..7)
//
// --- A matrix (16x32 int8, row-major smem_A[row][k]) ---
// Each thread holds 16 int8 values (2 rows x 8 k) in 4 uint32 regs:
//   a[0]: row=tid_y,       k=tid_x*4 .. tid_x*4+3   (4 consecutive k)
//   a[1]: row=tid_y+8,     k=tid_x*4 .. tid_x*4+3
//   a[2]: row=tid_y,       k=tid_x*4+16 .. tid_x*4+19
//   a[3]: row=tid_y+8,     k=tid_x*4+16 .. tid_x*4+19
// 4 threads (tid_x=0..3) cover all 32 k per row.
// 8 groups (tid_y=0..7) cover all 16 rows.
//
// --- B matrix (32x8 int8, column-major smem_B[col][k]) ---
// Each thread holds 8 int8 values (1 col x 8 k) in 2 uint32 regs:
//   b[0]: col=tid_y,     k=tid_x*4 .. tid_x*4+3
//   b[1]: col=tid_y,     k=tid_x*4+16 .. tid_x*4+19
// 4 threads (tid_x=0..3) cover all 32 k per column.
// 8 groups (tid_y=0..7) cover all 8 columns.
//
// --- D accumulator (16x8 int32) ---
//   d[0] = D[tid_y][tid_x*2]
//   d[1] = D[tid_y][tid_x*2+1]
//   d[2] = D[tid_y+8][tid_x*2]
//   d[3] = D[tid_y+8][tid_x*2+1]

__device__ __forceinline__ void load_A_frag_m16n8k32(
    uint32_t a[4], const int8_t* smem_A, int BLOCK_K)
{
    const int lane = threadIdx.x & 31;
    const int tid_x = lane & 3;
    const int tid_y = lane >> 2;

    const int base_k = tid_x * 4;            // 4-aligned

    // Each fragment register is 4 consecutive k-bytes of one row, so it is a
    // single 32-bit shared load (bit-identical to the byte-wise pack, since
    // smem is little-endian: byte0 | byte1<<8 | byte2<<16 | byte3<<24).
    a[0] = *(const uint32_t*)&smem_A[ tid_y      * BLOCK_K + base_k];
    a[1] = *(const uint32_t*)&smem_A[(tid_y + 8) * BLOCK_K + base_k];
    a[2] = *(const uint32_t*)&smem_A[ tid_y      * BLOCK_K + base_k + 16];
    a[3] = *(const uint32_t*)&smem_A[(tid_y + 8) * BLOCK_K + base_k + 16];
}

__device__ __forceinline__ void load_B_frag_m16n8k32(
    uint32_t b[2], const int8_t* smem_B, int BLOCK_K)
{
    const int lane = threadIdx.x & 31;
    const int tid_x = lane & 3;
    const int tid_y = lane >> 2;

    // B is column-major in shared memory: smem_B[col * BLOCK_K + row]
    const int col = tid_y;
    const int base_k = tid_x * 4;            // 4-aligned

    b[0] = *(const uint32_t*)&smem_B[col * BLOCK_K + base_k];
    b[1] = *(const uint32_t*)&smem_B[col * BLOCK_K + base_k + 16];
}

__device__ __forceinline__ void extract_D_m16n8k32(
    int32_t acc[4], int32_t* smem_D, int ldm)
{
    const int lane = threadIdx.x & 31;
    const int tid_x = lane & 3;
    const int tid_y = lane >> 2;

    smem_D[tid_y * ldm + tid_x * 2]         = acc[0];
    smem_D[tid_y * ldm + tid_x * 2 + 1]     = acc[1];
    smem_D[(tid_y + 8) * ldm + tid_x * 2]   = acc[2];
    smem_D[(tid_y + 8) * ldm + tid_x * 2 + 1] = acc[3];
}

#endif // __CUDA_ARCH__ >= 800

// ==================================================================
// Constants (identical to sm61 kernel)
// ==================================================================
static constexpr int HT              = 16;
static constexpr int HASH_ROT        = 13;
static constexpr int TRANSCRIPT_LEN  = 16;
static constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 32;

// ==================================================================
// Kernel template — visible to host for <<<>>> launch syntax
// Body uses #if __CUDA_ARCH__ to guard PTX calls.
// Guarded by PEARL_UNIT_TEST — unit tests only need the helpers above.
// ==================================================================

#ifndef PEARL_UNIT_TEST

template <int BLOCK_M, int BLOCK_N, int BLOCK_K,
          int WARPS_M, int WARPS_N, int STAGES, int MINB>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32, MINB)
pearl_ampere_fused_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ Bt,
    int n, int k, int R,
    uint32_t* __restrict__ transcript_buffer)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(BLOCK_M == WARPS_M * 16, "BLOCK_M must equal WARPS_M * 16");
    static_assert(BLOCK_N == WARPS_N * 16, "BLOCK_N must equal WARPS_N * 16");
    static_assert(BLOCK_K == 32, "BLOCK_K must equal 32 for m16n8k32 with current loaders");
    (void)R;

    const int tid      = threadIdx.x;
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int warp_m   = warp / WARPS_N;
    const int warp_n   = warp % WARPS_N;

    const int tiles_w  = n / HT;
    const int blocks_n = tiles_w / WARPS_N;
    const int block_row = blockIdx.x / blocks_n;
    const int block_col = blockIdx.x % blocks_n;
    const int row_base  = block_row * BLOCK_M;
    const int col_base  = block_col * BLOCK_N;
    const int warp_row0 = row_base + warp_m * HT;
    const int warp_col0 = col_base + warp_n * HT;

    constexpr int SMEM_A = BLOCK_M * BLOCK_K;
    constexpr int SMEM_B = BLOCK_N * BLOCK_K;
    constexpr int SMEM_STAGE = SMEM_A + SMEM_B;

    __shared__ __align__(16) int8_t smem_pipe[STAGES * SMEM_STAGE];
    __shared__ __align__(16) uint32_t sT[WARPS_M * WARPS_N][TRANSCRIPT_LEN];

    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < TRANSCRIPT_LEN; ++i) sT[warp][i] = 0;
    }

    int32_t accL[4] = {0,0,0,0};
    int32_t accR[4] = {0,0,0,0};

    const int T       = k / R;
    const int INNER_K = R / BLOCK_K;

    for (int t = 0; t < T; ++t) {

        for (int step = 0; step < INNER_K + STAGES - 1; ++step) {

            if (step < INNER_K) {
                const int k_off = t * R + step * BLOCK_K;
                const int stg   = step % STAGES;
                int8_t* smem_A_stg = &smem_pipe[stg * SMEM_STAGE];
                int8_t* smem_B_stg = &smem_pipe[stg * SMEM_STAGE + SMEM_A];

                for (int i = tid * 16; i < SMEM_A; i += blockDim.x * 16) {
                    const int row = i / BLOCK_K;
                    const int col = i % BLOCK_K;
                    cp_async_16B(&smem_A_stg[i],
                                 &A[(size_t)(row_base + row) * k + k_off + col]);
                }
                for (int i = tid * 16; i < SMEM_B; i += blockDim.x * 16) {
                    const int col = i / BLOCK_K;
                    const int row = i % BLOCK_K;
                    cp_async_16B(&smem_B_stg[i],
                                 &Bt[(size_t)(col_base + col) * k + k_off + row]);
                }
                cp_async_commit();
            }

            if (step >= STAGES - 1) {
                const int comp_stage = (step - (STAGES - 1)) % STAGES;
                cp_async_wait_group<STAGES - 2>();
                __syncthreads();

                const int8_t* smem_A_stage = &smem_pipe[comp_stage * SMEM_STAGE];
                const int8_t* smem_B_stage = &smem_pipe[comp_stage * SMEM_STAGE + SMEM_A];

                // Load A once per warp — shared between left/right halves
                uint32_t a_frag[4];
                load_A_frag_m16n8k32(a_frag,
                    &smem_A_stage[warp_m * 16 * BLOCK_K], BLOCK_K);

                // Left half (cols 0-7)
                {
                    uint32_t b_frag[2];
                    load_B_frag_m16n8k32(b_frag,
                        &smem_B_stage[warp_n * 16 * BLOCK_K], BLOCK_K);
                    mma_m16n8k32(accL, a_frag, b_frag, accL);
                }

                // Right half (cols 8-15)
                {
                    uint32_t b_frag[2];
                    load_B_frag_m16n8k32(b_frag,
                        &smem_B_stage[(warp_n * 16 + 8) * BLOCK_K], BLOCK_K);
                    mma_m16n8k32(accR, a_frag, b_frag, accR);
                }

                __syncthreads();
            }
        }

        // Hash the tile directly from accumulator registers — no tile_buf needed.
        // XOR is commutative: each thread XORs its 8 acc values, then shuffle
        // across the warp to fold the full 16×16 tile into one 32-bit word.
        // This eliminates a 256-int32 shared-memory write + read + syncthreads.
        uint32_t lx = 0;
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            lx ^= (uint32_t)accL[e];
            lx ^= (uint32_t)accR[e];
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            lx ^= __shfl_xor_sync(0xffffffffu, lx, off);

        if (lane == 0) {
            const int idx = t % TRANSCRIPT_LEN;
            sT[warp][idx] = ((sT[warp][idx] << HASH_ROT) |
                             (sT[warp][idx] >> (32 - HASH_ROT))) ^ lx;
        }
        __syncthreads();
    }

    if (lane == 0) {
        const int gi = warp_row0;
        const int gj = warp_col0;
        const int tile_id = (gi / HT) * tiles_w + (gj / HT);
        uint32_t* tb = &transcript_buffer[(size_t)tile_id * TRANSCRIPT_LEN];

        #pragma unroll
        for (int i = 0; i < TRANSCRIPT_LEN; i += 4) {
            *((int4*)&tb[i]) = *((int4*)&sT[warp][i]);
        }
    }
#else
    // On pre-sm_80, this kernel should never be launched — return.
    (void)A; (void)Bt; (void)n; (void)k; (void)R;
    (void)transcript_buffer;
#endif
}

// ==================================================================
// R-block-staged kernel: stage the full R-wide k-slice into shared memory per
// transcript step, then fire all R/32 MMA substeps back-to-back with NO
// inter-substep __syncthreads. Cuts barriers from ~2 per 32-k to ~2 per R-k and
// keeps the tensor pipe fed. Dynamic shared memory (R is a runtime value).
// Bit-exact with the fused kernel / DP4A reference.
// ==================================================================
template <int BLOCK_M, int BLOCK_N, int WARPS_M, int WARPS_N, int STAGES>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32, 1)
pearl_ampere_rblock_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ Bt,
    int n, int k, int R,
    uint32_t* __restrict__ transcript_buffer)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(BLOCK_M == WARPS_M * 16, "BLOCK_M must equal WARPS_M*16");
    static_assert(BLOCK_N == WARPS_N * 16, "BLOCK_N must equal WARPS_N*16");

    const int tid    = threadIdx.x;
    const int nthr   = WARPS_M * WARPS_N * 32;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;

    const int tiles_w   = n / HT;
    const int blocks_n  = tiles_w / WARPS_N;
    const int block_row = blockIdx.x / blocks_n;
    const int block_col = blockIdx.x % blocks_n;
    const int row_base  = block_row * BLOCK_M;
    const int col_base  = block_col * BLOCK_N;

    extern __shared__ int8_t smem[];
    const int ASZ = BLOCK_M * R;
    const int BSZ = BLOCK_N * R;
    const int STG = ASZ + BSZ;

    __shared__ uint32_t sT[WARPS_M * WARPS_N][TRANSCRIPT_LEN];
    if (lane == 0)
        #pragma unroll
        for (int i = 0; i < TRANSCRIPT_LEN; ++i) sT[warp][i] = 0;

    int32_t accL[4] = {0,0,0,0};
    int32_t accR[4] = {0,0,0,0};
    const int T     = k / R;
    const int INNER = R / 32;

    auto load_rblock = [&](int slot, int t) {
        int8_t* As = smem + slot * STG;
        int8_t* Bs = As + ASZ;
        for (int i = tid * 16; i < ASZ; i += nthr * 16) {
            int row = i / R, c = i % R;
            cp_async_16B(&As[i], &A[(size_t)(row_base + row) * k + (size_t)t * R + c]);
        }
        for (int i = tid * 16; i < BSZ; i += nthr * 16) {
            int col = i / R, c = i % R;
            cp_async_16B(&Bs[i], &Bt[(size_t)(col_base + col) * k + (size_t)t * R + c]);
        }
        cp_async_commit();
    };

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
        if (s < T) load_rblock(s, s);

    for (int t = 0; t < T; ++t) {
        const int cur = t % STAGES;
        const int pf  = t + STAGES - 1;
        if (pf < T) load_rblock(pf % STAGES, pf);
        cp_async_wait_group<STAGES - 1>();
        __syncthreads();

        const int8_t* As = smem + cur * STG + warp_m * 16 * R;
        const int8_t* Bs = smem + cur * STG + ASZ + warp_n * 16 * R;
        #pragma unroll
        for (int kk = 0; kk < INNER; ++kk) {
            const int koff = kk * 32;
            uint32_t a_frag[4];
            load_A_frag_m16n8k32(a_frag, As + koff, R);
            uint32_t bL[2], bR[2];
            load_B_frag_m16n8k32(bL, Bs + koff, R);
            load_B_frag_m16n8k32(bR, Bs + 8 * R + koff, R);
            mma_m16n8k32(accL, a_frag, bL, accL);
            mma_m16n8k32(accR, a_frag, bR, accR);
        }
        __syncthreads();

        uint32_t lx = 0;
        #pragma unroll
        for (int e = 0; e < 4; ++e) { lx ^= (uint32_t)accL[e]; lx ^= (uint32_t)accR[e]; }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            lx ^= __shfl_xor_sync(0xffffffffu, lx, off);
        if (lane == 0) {
            const int idx = t % TRANSCRIPT_LEN;
            sT[warp][idx] = ((sT[warp][idx] << HASH_ROT) |
                             (sT[warp][idx] >> (32 - HASH_ROT))) ^ lx;
        }
    }

    if (lane == 0) {
        const int gi = row_base + warp_m * HT;
        const int gj = col_base + warp_n * HT;
        const int tile_id = (gi / HT) * tiles_w + (gj / HT);
        uint32_t* tb = &transcript_buffer[(size_t)tile_id * TRANSCRIPT_LEN];
        #pragma unroll
        for (int i = 0; i < TRANSCRIPT_LEN; i += 4)
            *((int4*)&tb[i]) = *((int4*)&sT[warp][i]);
    }
#else
    (void)A;(void)Bt;(void)n;(void)k;(void)R;(void)transcript_buffer;
#endif
}

template <int BM, int BN, int WM, int WN, int STAGES>
cudaError_t launch_rblock(const int8_t* A, const int8_t* Bt, int m, int n,
                          int k, int R, uint32_t* T, cudaStream_t stream) {
#if defined(__CUDACC__)
    const int smem = STAGES * (BM + BN) * R;
    auto kern = pearl_ampere_rblock_kernel<BM, BN, WM, WN, STAGES>;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    dim3 block(WM * WN * 32);
    dim3 grid((unsigned)((m / BM) * (n / BN)));
    kern<<<grid, block, smem, stream>>>(A, Bt, n, k, R, T);
    return cudaGetLastError();
#else
    return cudaErrorNotSupported;
#endif
}

// ==================================================================
// Wide kernel: each warp computes NT adjacent 16×16 hash tiles (16×(NT*16)),
// giving NT*2 INDEPENDENT accumulator chains per warp -> NT*2 MMAs in flight to
// hide the mma.sync latency (the serial acc-chain dependency is the real bottleneck,
// not loads or syncs). Small 32-k smem stages keep occupancy high. NT=1 == fused.
// Bit-exact with the fused kernel / DP4A.
// ==================================================================
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARPS_M, int WARPS_N,
          int NT, int STAGES, int MINB>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32, MINB)
pearl_ampere_wide_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ Bt,
    int n, int k, int R, uint32_t* __restrict__ transcript_buffer)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(BLOCK_M == WARPS_M * 16, "BLOCK_M must equal WARPS_M*16");
    static_assert(BLOCK_N == WARPS_N * NT * 16, "BLOCK_N must equal WARPS_N*NT*16");
    static_assert(BLOCK_K == 32, "BLOCK_K must equal 32");

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;

    const int tiles_w   = n / HT;
    const int blocks_n  = tiles_w / (WARPS_N * NT);
    const int block_row = blockIdx.x / blocks_n;
    const int block_col = blockIdx.x % blocks_n;
    const int row_base  = block_row * BLOCK_M;
    const int col_base  = block_col * BLOCK_N;
    const int warp_row0 = row_base + warp_m * 16;
    const int warp_col0 = col_base + warp_n * NT * 16;

    constexpr int SMEM_A = BLOCK_M * BLOCK_K;
    constexpr int SMEM_B = BLOCK_N * BLOCK_K;
    constexpr int SMEM_STAGE = SMEM_A + SMEM_B;

    __shared__ __align__(16) int8_t smem_pipe[STAGES * SMEM_STAGE];
    __shared__ __align__(16) uint32_t sT[WARPS_M * WARPS_N * NT][TRANSCRIPT_LEN];

    if (lane == 0)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int i = 0; i < TRANSCRIPT_LEN; ++i) sT[warp * NT + nt][i] = 0;

    int32_t accL[NT][4];
    int32_t accR[NT][4];
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) { accL[nt][e] = 0; accR[nt][e] = 0; }

    const int T       = k / R;
    const int INNER_K = R / BLOCK_K;

    for (int t = 0; t < T; ++t) {
        for (int step = 0; step < INNER_K + STAGES - 1; ++step) {
            if (step < INNER_K) {
                const int k_off = t * R + step * BLOCK_K;
                const int stg   = step % STAGES;
                int8_t* sA = &smem_pipe[stg * SMEM_STAGE];
                int8_t* sB = &smem_pipe[stg * SMEM_STAGE + SMEM_A];
                for (int i = tid * 16; i < SMEM_A; i += blockDim.x * 16) {
                    const int row = i / BLOCK_K, col = i % BLOCK_K;
                    cp_async_16B(&sA[i], &A[(size_t)(row_base + row) * k + k_off + col]);
                }
                for (int i = tid * 16; i < SMEM_B; i += blockDim.x * 16) {
                    const int col = i / BLOCK_K, row = i % BLOCK_K;
                    cp_async_16B(&sB[i], &Bt[(size_t)(col_base + col) * k + k_off + row]);
                }
                cp_async_commit();
            }
            if (step >= STAGES - 1) {
                const int comp = (step - (STAGES - 1)) % STAGES;
                cp_async_wait_group<STAGES - 2>();
                __syncthreads();
                const int8_t* sA = &smem_pipe[comp * SMEM_STAGE];
                const int8_t* sB = &smem_pipe[comp * SMEM_STAGE + SMEM_A];
                uint32_t a_frag[4];
                load_A_frag_m16n8k32(a_frag, &sA[warp_m * 16 * BLOCK_K], BLOCK_K);
                #pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    uint32_t bL[2], bR[2];
                    load_B_frag_m16n8k32(bL, &sB[(warp_n * NT * 16 + nt * 16) * BLOCK_K], BLOCK_K);
                    load_B_frag_m16n8k32(bR, &sB[(warp_n * NT * 16 + nt * 16 + 8) * BLOCK_K], BLOCK_K);
                    mma_m16n8k32(accL[nt], a_frag, bL, accL[nt]);
                    mma_m16n8k32(accR[nt], a_frag, bR, accR[nt]);
                }
                __syncthreads();
            }
        }
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            uint32_t lx = 0;
            #pragma unroll
            for (int e = 0; e < 4; ++e) { lx ^= (uint32_t)accL[nt][e]; lx ^= (uint32_t)accR[nt][e]; }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) lx ^= __shfl_xor_sync(0xffffffffu, lx, off);
            if (lane == 0) {
                const int idx = t % TRANSCRIPT_LEN;
                uint32_t* s = sT[warp * NT + nt];
                s[idx] = ((s[idx] << HASH_ROT) | (s[idx] >> (32 - HASH_ROT))) ^ lx;
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int gi = warp_row0;
            const int gj = warp_col0 + nt * 16;
            const int tile_id = (gi / HT) * tiles_w + (gj / HT);
            uint32_t* tb = &transcript_buffer[(size_t)tile_id * TRANSCRIPT_LEN];
            uint32_t* s = sT[warp * NT + nt];
            #pragma unroll
            for (int i = 0; i < TRANSCRIPT_LEN; i += 4) *((int4*)&tb[i]) = *((int4*)&s[i]);
        }
    }
#else
    (void)A;(void)Bt;(void)n;(void)k;(void)R;(void)transcript_buffer;
#endif
}

template <int BM, int BN, int WM, int WN, int NT, int STG, int MNB>
cudaError_t launch_wide(const int8_t* A, const int8_t* Bt, int m, int n,
                        int k, int R, uint32_t* T, cudaStream_t stream) {
    dim3 block(WM * WN * 32);
    dim3 grid((unsigned)((m / BM) * (n / BN)));
    pearl_ampere_wide_kernel<BM, BN, 32, WM, WN, NT, STG, MNB>
        <<<grid, block, 0, stream>>>(A, Bt, n, k, R, T);
    return cudaGetLastError();
}

// ==================================================================
// wide1: like wide but a proper software pipeline with ONE __syncthreads per
// k-tile (prefetch the FAR stage AFTER compute, so the single sync separates the
// read of a stage from its next write). Flat k-tile loop; fold at R boundaries.
// ==================================================================
template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int WARPS_M, int WARPS_N,
          int NT, int STAGES, int MINB>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32, MINB)
pearl_ampere_wide1_kernel(
    const int8_t* __restrict__ A, const int8_t* __restrict__ Bt,
    int n, int k, int R, uint32_t* __restrict__ transcript_buffer)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(BLOCK_M == WARPS_M * 16, "");
    static_assert(BLOCK_N == WARPS_N * NT * 16, "");
    static_assert(BLOCK_K == 32, "");

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;

    const int tiles_w   = n / HT;
    const int blocks_n  = tiles_w / (WARPS_N * NT);
    const int block_row = blockIdx.x / blocks_n;
    const int block_col = blockIdx.x % blocks_n;
    const int row_base  = block_row * BLOCK_M;
    const int col_base  = block_col * BLOCK_N;
    const int warp_row0 = row_base + warp_m * 16;
    const int warp_col0 = col_base + warp_n * NT * 16;

    constexpr int SMEM_A = BLOCK_M * BLOCK_K;
    constexpr int SMEM_B = BLOCK_N * BLOCK_K;
    constexpr int SMEM_STAGE = SMEM_A + SMEM_B;

    __shared__ __align__(16) int8_t smem_pipe[STAGES * SMEM_STAGE];
    __shared__ __align__(16) uint32_t sT[WARPS_M * WARPS_N * NT][TRANSCRIPT_LEN];
    if (lane == 0)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int i = 0; i < TRANSCRIPT_LEN; ++i) sT[warp * NT + nt][i] = 0;

    int32_t accL[NT][4];
    int32_t accR[NT][4];
    #pragma unroll
    for (int nt = 0; nt < NT; ++nt)
        #pragma unroll
        for (int e = 0; e < 4; ++e) { accL[nt][e] = 0; accR[nt][e] = 0; }

    const int INNER_K = R / BLOCK_K;
    const int KT = (k / R) * INNER_K;   // total k-tiles (contiguous in k)

    auto issue = [&](int kt) {
        const int stg = kt % STAGES;
        const int k_off = kt * BLOCK_K;
        int8_t* sA = &smem_pipe[stg * SMEM_STAGE];
        int8_t* sB = &smem_pipe[stg * SMEM_STAGE + SMEM_A];
        for (int i = tid * 16; i < SMEM_A; i += blockDim.x * 16) {
            const int row = i / BLOCK_K, col = i % BLOCK_K;
            cp_async_16B(&sA[i], &A[(size_t)(row_base + row) * k + k_off + col]);
        }
        for (int i = tid * 16; i < SMEM_B; i += blockDim.x * 16) {
            const int col = i / BLOCK_K, row = i % BLOCK_K;
            cp_async_16B(&sB[i], &Bt[(size_t)(col_base + col) * k + k_off + row]);
        }
        cp_async_commit();
    };

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) if (s < KT) issue(s);

    for (int kt = 0; kt < KT; ++kt) {
        const int stg = kt % STAGES;
        cp_async_wait_group<STAGES - 2>();
        __syncthreads();

        const int8_t* sA = &smem_pipe[stg * SMEM_STAGE] + warp_m * 16 * BLOCK_K;
        const int8_t* sB = &smem_pipe[stg * SMEM_STAGE + SMEM_A];
        uint32_t a_frag[4];
        load_A_frag_m16n8k32(a_frag, sA, BLOCK_K);
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            uint32_t bL[2], bR[2];
            load_B_frag_m16n8k32(bL, &sB[(warp_n * NT * 16 + nt * 16) * BLOCK_K], BLOCK_K);
            load_B_frag_m16n8k32(bR, &sB[(warp_n * NT * 16 + nt * 16 + 8) * BLOCK_K], BLOCK_K);
            mma_m16n8k32(accL[nt], a_frag, bL, accL[nt]);
            mma_m16n8k32(accR[nt], a_frag, bR, accR[nt]);
        }

        const int pf = kt + STAGES - 1;
        if (pf < KT) issue(pf);

        if ((kt + 1) % INNER_K == 0) {
            const int t = kt / INNER_K;
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                uint32_t lx = 0;
                #pragma unroll
                for (int e = 0; e < 4; ++e) { lx ^= (uint32_t)accL[nt][e]; lx ^= (uint32_t)accR[nt][e]; }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) lx ^= __shfl_xor_sync(0xffffffffu, lx, off);
                if (lane == 0) {
                    const int idx = t % TRANSCRIPT_LEN;
                    uint32_t* s = sT[warp * NT + nt];
                    s[idx] = ((s[idx] << HASH_ROT) | (s[idx] >> (32 - HASH_ROT))) ^ lx;
                }
            }
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int gi = warp_row0;
            const int gj = warp_col0 + nt * 16;
            const int tile_id = (gi / HT) * tiles_w + (gj / HT);
            uint32_t* tb = &transcript_buffer[(size_t)tile_id * TRANSCRIPT_LEN];
            uint32_t* s = sT[warp * NT + nt];
            #pragma unroll
            for (int i = 0; i < TRANSCRIPT_LEN; i += 4) *((int4*)&tb[i]) = *((int4*)&s[i]);
        }
    }
#else
    (void)A;(void)Bt;(void)n;(void)k;(void)R;(void)transcript_buffer;
#endif
}

template <int BM, int BN, int WM, int WN, int NT, int STG, int MNB>
cudaError_t launch_wide1(const int8_t* A, const int8_t* Bt, int m, int n,
                         int k, int R, uint32_t* T, cudaStream_t stream) {
    dim3 block(WM * WN * 32);
    dim3 grid((unsigned)((m / BM) * (n / BN)));
    pearl_ampere_wide1_kernel<BM, BN, 32, WM, WN, NT, STG, MNB>
        <<<grid, block, 0, stream>>>(A, Bt, n, k, R, T);
    return cudaGetLastError();
}

// ==================================================================
// Host dispatcher
// ==================================================================
cudaError_t launch_pearl_ampere(
    const int8_t* A, const int8_t* Bt,
    int m, int n, int k, int R,
    uint32_t* transcript_buffer,
    cudaStream_t stream)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return err;

    if (prop.major < 8) {
        return cudaErrorNotSupported;
    }

    constexpr int block_k = 32;
    if (k % block_k != 0) return cudaErrorInvalidValue;

    dim3 block;
    int grids_m, grids_n;

    // Best on Ada (AD107): wide kernel, each warp computes NT 16×16 tiles -> NT*2
    // independent accumulator chains -> many MMAs in flight (the serial acc-chain
    // dependency, not loads/syncs, was the bottleneck). NT=16 (16×256/warp) ~2x
    // the old fused kernel on the RTX 4050. Needs n%256, m%64.
    if (m % 64 == 0 && n % 256 == 0) {
        return launch_wide<64, 256, 4, 1, 16, 2, 1>(A, Bt, m, n, k, R,
                                                    transcript_buffer, stream);
    }
    // wide NT=8 (n%128).
    if (m % 64 == 0 && n % 128 == 0) {
        return launch_wide<64, 128, 4, 1, 8, 3, 2>(A, Bt, m, n, k, R,
                                                   transcript_buffer, stream);
    }

    // Fallback: 64×64 fused 4-stage (n only a multiple of 64).
    if (m % 64 == 0 && n % 64 == 0) {
        block = dim3(4 * 4 * 32);
        grids_m = m / 64;
        grids_n = n / 64;
        pearl_ampere_fused_kernel<64,64,32, 4,4,4,3>
            <<<dim3(grids_m * grids_n), block, 0, stream>>>(A,Bt,n,k,R,transcript_buffer);
        return cudaGetLastError();
    }

    // Fallback: 32×64 4-stage (n multiple of 64, m only multiple of 32).
    if (m % 32 == 0 && n % 64 == 0) {
        block = dim3(2 * 4 * 32);
        grids_m = m / 32;
        grids_n = n / 64;
        pearl_ampere_fused_kernel<32,64,32, 2,4,4,2>
            <<<dim3(grids_m * grids_n), block, 0, stream>>>(A,Bt,n,k,R,transcript_buffer);
        return cudaGetLastError();
    }

    return cudaErrorInvalidValue;
}

#endif // !defined(PEARL_UNIT_TEST)
