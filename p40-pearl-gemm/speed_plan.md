# P40 Pearl Miner — Speed Plan

Throughput roadmap for the Pascal (sm_61) Pearl PoW search kernel.

## Unit
Pool hashrate = (16×16 hash-tiles evaluated/s) × difficulty_adjustment_factor,
factor = tile_size × dot_product_length = 256 × 4096 = 2^20.
So **1 Mtile/s ≈ 1.0486 TH/s**. Each tile = 16×16×4096 = 2^20 INT8 MACs = 2^18 dp4a.

## Hardware ceiling
- **Tesla P40** (sm_61): 47 INT8 TOPS via DP4A → ~11.7 Tdp4a/s → **~22.4 Mtiles/s = ~23.5 TH/s theoretical**, ~16 TH/s realistic (70%).
- **GTX 1070** (sm_61): ~weaker; useful as a 2nd device for aggregate.
- User target **25 TH/s** is above one P40's theoretical peak → needs P40 + 1070 + an efficient kernel.

## Progress (region 16384, full scan, P40)
| Stage | Mtiles/s | TH/s | ×naive | Note |
|------|---------|------|--------|------|
| naive `pearl_pow` | 0.46 | 0.48 | 1× | one block/tile, 128 __syncthreads/tile |
| fused warp/tile + `__shfl_xor` | 1.98 | 2.08 | 4.3× | shared-mem operand reuse |
| + bank-conflict padding (stride 65) | 2.34 | 2.46 | 5.1× | row stride coprime to 32 banks |
| + 4×2 register blocking | 3.13 | 3.28 | 6.8× | 0.75 shared-loads/dp4a, ILP |
| **+ MINB2 occupancy (variant 1)** | **5.64** | **5.91** | **12.3×** | 2 blocks/SM = 50% occupancy |

Current best: **variant 1 (4×4 region, `__launch_bounds__(512,2)`)** — the miner uses it.
At 5.64 Mtiles/s we are at ~25% of dp4a peak; occupancy-limited at 50%.

## Diagnosis
- Bottleneck history: barriers → shared-load ratio → **occupancy**.
- The cute keyed-BLAKE3 (per-tile, lane 0) needs ~100 regs and dominates the kernel's
  register footprint; capping regs (MINB2) to reach 2 blocks/SM is the current win, but
  pushing to MINB3/4 over-spills BLAKE3 and regresses. So registers (BLAKE3) cap occupancy.

## Remaining levers (priority order)
1. **Split BLAKE3 into a 2nd kernel** *(highest value, ~1.5–2×)*
   - GEMM kernel computes only the 16-word transcript per tile → writes to a global
     buffer (num_tiles × 64 B). Without BLAKE3 its reg footprint drops to ~40 → 3–4
     blocks/SM (75–100% occupancy) without spills.
   - A 2nd 1-thread/tile kernel does keyed BLAKE3 + target compare over the transcript
     buffer (cheap; ~64 MB for 1M tiles, sub-ms).
   - Must stay bit-exact (transcript bytes identical).
2. **Vectorized `int4` shared loads** *(~1.2–1.4×)*
   - Load 4 dp4a-words at once (16 B) per operand fragment to cut load-instruction count;
     keeps the 4×2 register micro-tile.
3. **Noised-matmul on GPU INT path** *(removes per-region fp32 matmul overhead)*
   - Replace the `_imatmul_i8` fp32 GEMM (E_AL@E_AR) with the existing `noise_A`/`noise_B`
     INT kernels, or fuse the noise add into the search kernel.
4. **Multi-GPU dispatch (P40 + 1070)** *(aggregate)*
   - Split the output-tile space across both devices; each runs the same kernel on its
     own CUDA stream/process. Aggregate ≈ P40 + 1070 hashrate.
5. **Larger micro-tiles / bigger blocks** once occupancy is unlocked by (1).

## Realistic outlook
- P40 alone, after (1)+(2): ~9–13 TH/s plausible (toward the ~16 TH/s realistic ceiling).
- + GTX 1070 via (4): aggregate target ~12–18 TH/s.
- **25 TH/s is at the edge of this hardware** even fully optimized; it likely needs a
  faster/3rd card. Everything above keeps the search loop 100% on-GPU.

## Invariant
Every kernel change MUST stay **bit-exact** with the naive `pearl_pow` transcript
(validate via all-tile digest compare) — a divergence silently invalidates shares.
