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

## Progress (real config: region 4096, k=4096, R=256, P40)
| Stage | Mtiles/s | TH/s | ×naive | Note |
|------|---------|------|--------|------|
| naive `pearl_pow` | 0.46 | 0.48 | 1× | one block/tile, 128 __syncthreads/tile |
| fused warp/tile + `__shfl_xor` | 1.98 | 2.08 | 4.3× | shared-mem operand reuse |
| + bank-conflict padding (stride 65) | 2.34 | 2.46 | 5.1× | row stride coprime to 32 banks |
| + 4×2 register blocking | 3.13 | 3.28 | 6.8× | 0.75 shared-loads/dp4a, ILP |
| + MINB2 occupancy (fused variant 1) | 5.25 | 5.5 | 11.4× | 2 blocks/SM = 50% occupancy |
| + split BLAKE3 → GEMM-only (split v3, S=256) | 5.68 | 5.95 | 12.4× | still 2 blocks/SM (shared-mem bound!) |
| + decouple staging width S=128 (split v1) | 6.92 | 7.25 | 15.1× | 4 blocks/SM = **100% occupancy** |
| **+ per-warp transcript in shared (split v1)** | **7.15** | **7.5** | **15.6×** | frees regs, trims MINB4 spill |

Current best: **`pearl_pow_split` variant 1 (GEMM-only S=128 4×4 MINB4 + BLAKE3 pass)** — the miner uses it.
At ~7.15 Mtiles/s we are at ~36% of dp4a peak.

### The shared-memory occupancy unlock (the big one)
The split GEMM-only kernel reported MINB3, but at full staging width S=R=256 it stages
`sAi`+`sBi` = 64×65×4 ×2 ≈ **33 KB/block** → `floor(96/33) = 2` blocks/SM. So it was
**shared-memory-bound at 2 blocks**, and the MINB3 register cap did nothing. Decoupling
the staging width **S** from the R=256 reduction window (stage S=128 columns, keep
accumulating `acc`, XOR-reduce only at R boundaries — bit-exact) halves the shared
footprint to ~17 KB → 4 blocks/SM = **100% thread occupancy**. That single change is
+21%. Moving the per-warp transcript into shared (the 31 non-zero lanes were carrying
it dead in registers) trims the residual MINB4 spill for another ~+3%.

> **Measurement note.** All numbers above are at the *real* mining config (k=4096).
> An earlier split benchmark at k=16384 (4× the real k) overstated the split win at
> +23–28%; at real k the GEMM compute is smaller so the transcript round-trip costs
> relatively more. The honest win is ~+3–8% **and** much lower run-to-run variance
> (the lean GEMM-only kernel has no BLAKE3 register pressure). The `int4` vectorized
> loads were a **regression**: their 16-B-aligned stride (RW+4)&~3 = 68 has 68 mod 32
> = 4, reintroducing bank conflicts that outweigh the fewer load instructions. Reverted
> to conflict-free scalar loads at stride RW+1 = 65.

## Diagnosis
- Bottleneck history: barriers → shared-load ratio → **occupancy**.
- The cute keyed-BLAKE3 (per-tile, lane 0) needs ~100 regs and dominated the fused
  kernel's register footprint, capping it at MINB2 (50%). Splitting BLAKE3 into a 2nd
  1-thread/tile kernel drops the GEMM kernel to ~40 regs → MINB3 (75%) with no spills.
- Within one warp the micro-tile is **already optimal**: a warp owns one 16×16 tile =
  256 elts / 32 lanes = 8 elts/lane (fixed), so RM·RN = 8 is forced and (4,2) at 0.75
  loads/dp4a is the best single-warp ratio. Going below 0.75 needs operand reuse across
  *multiple* tiles per warp (lever 5).

## Remaining levers (priority order)
1. ~~**Split BLAKE3 into a 2nd kernel**~~ *(DONE — split v0, +3–8% and steadier)*
2. ~~**Vectorized `int4` shared loads**~~ *(TRIED — net regression from bank conflicts; reverted)*
3. **Noised-matmul on GPU INT path** *(removes per-region fp32 matmul overhead)*
   - Replace the `_imatmul_i8` fp32 GEMM (E_AL@E_AR) with the existing `noise_A`/`noise_B`
     INT kernels, or fuse the noise add into the search kernel.
4. **Multi-GPU dispatch (P40 + 1070)** *(aggregate, highest remaining value)*
   - Split the output-tile space across both devices; each runs the same kernel on its
     own CUDA stream/process. Aggregate ≈ P40 + 1070 hashrate.
5. **Multi-tile-per-warp operand reuse** *(the only way below 0.75 loads/dp4a)*
   - Have each warp compute 2 horizontally-adjacent 16×16 tiles that share the same A
     rows: load A once, reuse for both tiles' B → 8 loads / 16 dp4a = 0.5 ratio. Needs
     2 transcript accumulators + 2 XOR reductions per warp (still bit-exact). Bigger
     restructure; attempt after multi-GPU.

## Realistic outlook
- P40 alone, after (1)+(2): ~9–13 TH/s plausible (toward the ~16 TH/s realistic ceiling).
- + GTX 1070 via (4): aggregate target ~12–18 TH/s.
- **25 TH/s is at the edge of this hardware** even fully optimized; it likely needs a
  faster/3rd card. Everything above keeps the search loop 100% on-GPU.

## Invariant
Every kernel change MUST stay **bit-exact** with the naive `pearl_pow` transcript
(validate via all-tile digest compare) — a divergence silently invalidates shares.
