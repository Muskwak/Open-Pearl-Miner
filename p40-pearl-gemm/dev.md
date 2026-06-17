# Dev log — Ada (sm_89) tensor-core kernel optimization

Goal: close the gap on the RTX 4050 (Ada, AD107, 20 SM) between our Ampere/Ada
tensor-core GEMM and the reference "ipminer" — **8 TH/s → ~25 TH/s**, staying
**bit-exact** with the DP4A transcript.

1 tile = 16×16×4096 = 2^20 "hashes"; **1 Mtile/s ≈ 1.05 TH/s**. One mining region
= 4096×4096×4096 (R=256) = 65536 tiles. 25 TH/s ≈ 23.8 Mtile/s ≈ 2.75 ms/region.

## Test rig & loop
- An RTX 4050 (Ada, sm_89) test box reached over SSH. Windows, CUDA 12.8, **no Nsight
  Compute** installed (so profiling is empirical, via kernel timings).
- Cross-compile here (CUDA 12.8 + VS2022, `-arch=sm_89 -cudart static -Xcompiler /MT`),
  `scp` the ~0.6 MB exe, run on the box. One cycle ≈ 1 min.
- `tests/bench_ampere.cu` — includes the DP4A ref (`pearl_gemm_only_sm61.cu`) and the
  TC kernel; does a **bit-exact transcript check** (256×256×4096 R256 vs DP4A) and times
  the TC GEMM + DP4A at the real region config + a config sweep. `tests/iter.ps1` =
  build→scp→run.

## Baseline (fused kernel, 32×64 default)
```
TC   : 7.725 ms/region -> 8.90 TH/s  (17.8 INT8 TOPS, ~18% of peak)
DP4A : 10.31 ms/region -> 6.66 TH/s
TC speedup vs DP4A: 1.34x   <-- abysmal for tensor cores (should be 3-5x)
```
GEMM-only 8.9 ≈ the end-to-end 8 TH/s the user measured → **the GEMM is the
bottleneck** (BLAKE3 + noise + host overhead ≈ 10%). So optimize the GEMM.

## Iterations

| # | Change | Result | Verdict |
|---|--------|--------|---------|
| 1 | Vectorize smem→reg packing: 24 byte-loads+shifts → 6 `uint32` loads per MMA pair | 8.90 → 8.92 | no-op — nvcc already vectorized; loads aren't the bottleneck |
| 2 | Config sweep (block/warp/stage/minb) | **64×64 s4 b3 = 10.67** vs 32×64 = 8.9; 64×128/128×64/32×128 ≈ 9.3 | **64×64 is +20%**; dispatcher was picking the worse 32×64 |
| 3 | Reorder dispatcher to prefer 64×64 | TC 8.9 → **10.3** (shipped) | ✅ free +20% in the real miner |
| 4 | **R-block-staged kernel** (stage full R=256 k-slice, ~10× fewer syncs, dynamic smem) | bit-exact but **5.2 TH/s (2× slower)** | ❌ 64 KB smem → 1 block/SM → occupancy collapse |
| 5 | **Wide kernel** — each warp computes **NT** adjacent 16×16 tiles → **NT·2 independent accumulator chains** (small 32-k smem kept) | NT2=13.0, NT4=14.6, NT8=16.7, **NT16=17.6** | ✅ **the real fix** — ILP, not occupancy/sync |
| 6 | **wide1** — same but 1 `__syncthreads`/k-tile (sw-pipeline, prefetch far stage) | NT8=16.8, NT16=16.8 | ❌ no gain — syncs were never the bottleneck |
| 7 | Wire **wide NT16 (64×256)** into the dispatcher (n%256,m%64), NT8 fallback | TC path **8.9 → 16.2 TH/s (2.41× DP4A)** | ✅ shipped in `launch_pearl_ampere` |
| 8 | **XOR-swizzle smem** (`kk ^ (((row>>2)&1)<<4)`) → conflict-free A/B frag loads, zero extra smem, bit-exact | NT16 17.6 → **18.0**; bank conflicts **35.7M → 0**, shared-ld wavefronts **72.6M → 37.0M** | ✅ shipped (`swz32`/`load_*_swz`) — small TH/s gain but unblocks ldmatrix |
| 9 | **Cache the dispatcher arch check** (was calling `cudaGetDeviceProperties` per region, ~0.3 ms) | TC path 16.3 → **16.7** (~8% host tax removed; real-miner per-region) | ✅ shipped |
| 10 | **ldmatrix loads, plain layout** (A `x4`→4 regs, B `x2`→2 regs, no `.trans`) | bit-exact; LSU pipe **61%→41%** but conflicts **back** (plain) → L1 72% → **17.7** (no net gain) | ❌ alone — each fix solves only half |
| 11 | **ldmatrix + swizzle** (swizzle the per-lane ldmatrix addr — 16-byte-row granularity matches) | conflicts **0**, LSU 47%, L1 60%, **tensor 37.6%→43%** → **19.9** (64×256) | ✅ the combo unblocks the tensor pipe |
| 12 | **Bigger block 128×256 s3** (8 warps share the B tile → amortize cp.async, cut `long_scoreboard`) | **21.1** (256×256 regresses: register spill, 1 blk) | ✅ shipped — dispatcher routes m%128,n%256 → `launch_ldm<128,256,8,1,16,3,1>` |
| 16 | **`cp.async.cg` (L1 bypass)** — `.cg` (L2-only) instead of `.ca`; ncu showed L1 hit = 0.59% (useless) yet cp.async still floods the L1/MIO pipe | **21.4 → 23.4 (+10%)** | ✅ shipped — single biggest post-iter-12 win; frees the MIO pipe for ldmatrix |
| 17 | **`ldmatrix.x4` B-load** (`ldm_B2_frag`) — one `ldmatrix.x4` returns both bL+bR k-halves of a B tile, replacing two `ldmatrix.x2` calls → ½ the B-load instrs | **23.4 → 24.0 (+2.4%)** | ✅ shipped — bit-exact; tensor pipe 43%→48.7% |
| 18 | **"cutlass" kernel-name ptxas hack** — rename `pearl_ampere_ldm_kernel`→`cutlass_pearl_ampere_ldm_kernel`; ptxas gates a more aggressive load↔math scheduling pass on the `cutlass` substring | **+1.0%** (24.085 vs 23.838 mean, n=6 each) | ⚠️ real but marginal — **NOT adopted** (name-hack fragility); recorded only |

### Key insight — it was ILP all along
`mma(acc,a,b,acc)` makes each accumulator a **serial dependency chain**. The fused kernel
had only **2 chains/warp** (accL,accR) → the tensor pipe stalls on that 2-deep latency.
Giving each warp **NT·2** independent chains (NT=16 → 32) fills the pipe. Confirmed:
- Wide NT16 ≈ 2× fused, and wins **even at low occupancy** (ILP > occupancy).
- R-block (more k-staging, same 2 chains) and wide1 (fewer syncs) gave **nothing** → the
  bottleneck was never loads, syncs, or occupancy.

### Benchmark gotcha
TH/s must be computed from tiles **actually covered**: BM and BN must divide the region
(4096). NT∈{2,4,8,16} give BN∈{32,64,128,256} (valid); NT=10/12 (BN=160/192) silently
cover fewer tiles and *inflate* TH/s. The bench now flags non-dividing configs with `*`.

## Current best
**`ldm` kernel, 128×256 NT16 3-stage = 21.4 TH/s GEMM (42.8 INT8 TOPS, 3.17× DP4A)** —
bit-exact, shipped via the dispatcher. **2.67× the 8 TH/s baseline**, ~86% of ipminer's 25.
(Steady-state with warmed clocks; `time_tc` needs ~60 warmup iters or it reads ~18 cold —
the GPU boost clock ramps over the run, *not* a dispatcher overhead.)
The climb: 8.0 → 16.3 (ILP/wide) → 18.0 (swizzle) → 19.9 (ldmatrix+swizzle) → 21.4 (128×256).

## End-to-end split (measured)
BLAKE3 over the transcript = **0.025 ms/region**; GEMM = 4.22 ms. So GEMM is **99%**
of `pearl_pow_split`, and noise (amortized per row/col block) + host loop are the only
other costs. **The GEMM is essentially the whole end-to-end cost** — optimizing it is
exactly right, and end-to-end ≈ GEMM (minus ~10% fixed overhead).

## Profiled with Nsight Compute (ncu 2025.1) — the real bottleneck
`tests/prof.ps1` builds + scp's + runs `ncu` against a single clean launch (the bench's
`prof` mode launches the dispatcher once at 4096² → 1024 blocks). Wide NT16 **before**
the swizzle:
```
L1/TEX throughput 71%  (Memory 70%, "L1 bottleneck")   DRAM 2%  (L2 hit 97.6%)
Tensor pipe 36.5%  ("well-utilized, should NOT be a bottleneck")  — 2× headroom
Scheduler No-Eligible 80%   active warps/sched 1.98   Occupancy 16.7% (reg+smem limited, 2 blk/SM)
Stalls (per-issue): wait 2.51 | mio_throttle 2.45 | long_scoreboard 1.80 | math 0.91 | barrier 0.76
Bank conflicts 35.7M of 72.6M shared-ld wavefronts  (~49% are conflict replays!)
```
→ **L1/shared-load-bound**, tensor idle. The bank conflicts (iter 8) and the dispatcher
tax (iter 9) were the two findings. **After** the swizzle: conflicts **0**, wavefronts
**37.0M**, L1/TEX **61%**, mio_throttle **2.04**, tensor **37.6%** — but only +2.4% TH/s.

### Two co-limiters remain (the wall to 25 TH/s)
1. **LSU instruction count** — LSU pipe still **61%**, `mio_throttle` 2.04. ~68 `LDS.32`/k-step
   (1 A-frag = 4, + NT·2 B-frags × 2). **Next lever: `ldmatrix.sync`** — 1 instr fills a whole
   fragment (A: x4 → 4 regs; B: x2 → 2 regs), ~2× fewer load instrs. The swizzle is a
   prerequisite (ldmatrix also wants conflict-free smem). Risk: must reproduce the exact
   mma s8 fragment layout (.b16 reinterpret + `.trans` for col-major B) — bit-exact or revert.
2. **Latency under low occupancy** — `wait` 2.31 + `long_scoreboard` 1.62 + `barrier` 0.71 = 4.6,
   with only ~2 warps/sched (16.5% occ, NT=16 accumulators = 128 int32 regs → 2 blk/SM).
   Empirically *can't* be fixed by lowering NT: NT8 (½ the regs) = 17.96 ≈ NT16 18.0, so the
   ILP gain from big NT cancels the occupancy gain from small NT. ILP ≥ occupancy here.

So 25 TH/s likely needs **ldmatrix AND** an occupancy/latency win; ldmatrix alone ≈ 20–22 est.

### Outcome (confirmed) + what's left to 25
ldmatrix+swizzle landed at **21.4** (iter 10–12) — both predictions held: ldmatrix alone was a
wash (conflicts returned), the combo unblocked the tensor pipe (37.6→43%), and the bigger
128×256 block's B-amortization cut `long_scoreboard`. After iter 12 the kernel is still
**latency-bound under 16.5% occupancy** (`wait` ~2.2 + `long_scoreboard` ~2.0, tensor ~43%).
The remaining ~16% to 25 needs an **occupancy win that doesn't cost ILP** — the only real lever
left is **dynamic shared memory** (`cudaFuncAttributeMaxDynamicSharedMemorySize`, up to 100 KB on
Ada vs the 48 KB static cap) to fit more stages/blocks per SM.

### Dynamic smem + carveout sweep (iter 13) — explored, does NOT beat 21.1
Built `pearl_ampere_ldm_dyn_kernel` (single `extern __shared__`, manual pipe+transcript offsets)
+ a `carveout` knob. Findings on the 4050 (Ada = 128 KB unified L1/shared, shared ≤100 KB):
- **Deeper pipelines** (128×256 s4/s5/s6, 1 block) — no gain (20.4–20.9 < 21.1); s3 already hides it.
- **Occupancy 2nd block** needs a bigger shared carveout; the L1↔shared **knee** is real:
  `64×256 s2` co0/32 = 1 blk = **15.0**, co50/64 = 2 blk (+64 KB L1) = **19.9**, co100 (28 KB L1) = 19.1.
  So mid-carveout (your "somewhere in the middle") *is* best among 2-block configs — but it tops out
  at **19.9 < 21.1**, because `128×256 s3` already reaches the same **8 warps/SM** AND all 8 warps
  share one B tile (amortizing cp.async), which the 2×`64×256` blocks don't.
### WARPS_N occupancy sweep (iter 14) — falsifies the register-frugal redesign
The kernel already has a `WARPS_N` param: WN>1 makes WN warps cooperate on the same 256-wide
block, each holding **NT/WN** accumulators — i.e. the exact "register-frugal" split (fewer regs/warp
→ more warps/SM) with **no rewrite**. Swept it (all bit-exact):
- 128×256 **WN1** NT16 (8 warps/SM) = **21.1**  ← still best
- 128×256 WN2 NT8 (16 warps/SM) = 19.4 ;  WN4 NT4 (32 warps/SM) = 18.4
- 64×256 WN2 NT8 = 16.5
**More occupancy consistently HURTS** — each warp loses ILP, and this kernel is **ILP-bound, not
occupancy-bound**. So the register-frugal redesign would land *below* 21.1; its premise is
empirically false. (Note: target is Ada/sm_89 RTX 4050 — AD107, 20 SM, 128 KB L1+shared/SM. The
`ampere`/`sm_80+` naming is just the tensor-core ISA baseline Ada inherits.)

**Conclusion: 21.4 TH/s (static 128×256 NT16 s3) is the firm structural ceiling.** Every
occupancy/register lever (dynamic smem, carveout, WARPS_N) makes it worse; every ILP/load lever
(ILP, swizzle, ldmatrix, B-amortization) is spent.

### Clock-scaling check (iter 15) — the gap to 27 is NOT clock
The 4050 (Laptop, 75 W cap, max boost 3105 MHz) was suspected power/clock-throttled — but ncu's
"1.78 GHz" was just ncu locking to base. Live under load it runs **2430→2800 MHz at only 41 W / 51 °C**
(power & thermal headroom). Pinned clock-scaling (`nvidia-smi -lgc f,f`):
- 1500 MHz → **13.3 TH/s**;  2800 MHz → **21.0 TH/s**  (1.58× for a 1.87× clock → **sub-linear**)
So it's part clock-bound, part memory-latency-bound. The card already boosts to ~2800 under load
(unlocked 21.1 ≈ pinned-2800), max 3105 ⇒ only **~5 % clock headroom** (~22 at max). The tempting
`21.1×3105/2430≈27` was a coincidence — **clock does not explain ipminer's 27**.

**Final verdict: ~21 TH/s (2.67× baseline, 3.2× DP4A, bit-exact) is our ceiling for this design on
this card.** ipminer's ~27 (~28 % more) is a genuine kernel/algorithm advantage (memory-latency
hiding we haven't cracked) or a higher-TGP card — not reachable by tuning this structure. Shipping
kernel = static `ldm` 128×256 NT16 s3. Levers tried & exhausted: ILP/wide, smem swizzle, dispatcher
cache, ldmatrix, block/stage sweep, dynamic smem, carveout, WARPS_N occupancy, clock. **Stop here.**

### Ultrathink (iter 16–17) — the "21.4 ceiling" was wrong; two real wins remained
The iter-15 "stop here" verdict was **premature**. Re-profiling found two levers it had missed,
both bit-exact, both in the *load/MIO* path (not occupancy):
- **iter 16 — `cp.async.cg`**: ncu showed the cp.async traffic was hammering L1/MIO for a **0.59 %**
  L1 hit rate (the data is streamed once, never reused before eviction). Switching `.ca`→`.cg`
  (bypass L1, land in L2) frees the entire L1/MIO pipe. **+10 % (21.4 → 23.4)** — the single biggest
  post-iter-12 win, and exactly the "memory-latency hiding we haven't cracked" gap iter-15 hand-waved.
- **iter 17 — `ldmatrix.x4` B-load**: collapse the two `ldmatrix.x2` B-frag loads into one
  `ldmatrix.x4` (`ldm_B2_frag`). Half the B-load instructions → tensor pipe **43 % → 48.7 %**.
  **+2.4 % (23.4 → 24.0)**.

**New best: 24.0 TH/s** (3.0× the 8.0 baseline, 3.57× DP4A, bit-exact). Climb in full:
8.0 → 16.3 (ILP/wide) → 18.0 (swizzle) → 19.9 (ldmatrix+swizzle) → 21.4 (128×256) →
**23.4 (cp.async.cg)** → **24.0 (ldmatrix.x4)**.

Falsified dead-ends from this round (recorded so we don't re-try them): `cp.async` `.L2::256B`
prefetch hint (no gain over `.cg`), flat 1-sync pipeline (`ldm_flat`, 21.7 < 24.0 — the per-R-block
pipeline issues its prefetch at the top of the loop = better load lead), B-prefetch, fold-interleave.

### Why lpminer hits ~27 and we cap at 24 — from disassembling the competitor
Disassembled `E:\lpminer-0.1.10\lpminer.exe` (cuobjdump, CUDA 12.8). Its sm_89 mining GEMM is
`pearl::ampere_dedicated_mining_ws< Sm80KernelTraits<int8, half, half, float, tuple<C<128>,C<256>>, …>,
StaticPersistentTileScheduler >` — **REG:255, dynamic smem, CUTLASS/CuTe**. The edge over our
hand-written kernel is two things we structurally cannot express in this single-loop design:
1. **Warp specialization** (`_ws`): dedicated *producer* (cp.async) warps + *consumer* (MMA) warps.
   The consumer warps get the **full 255-reg budget for accumulator chains** → many more MMAs in
   flight → hides the dominant `wait` (MMA-dependency) stall that caps us. Our kernel makes every
   warp do both load and compute, so registers are split and ILP is bounded.
2. **Persistent scheduling** (`StaticPersistentTileScheduler`): grid-resident blocks loop over tiles
   → no wave-quantization bubbles + a hot L2 working set (cuts `long_scoreboard`).
The CTA tile is **128×256 — identical to what we found best**. So we reached **89 % of lpminer
(24.0 / ~27) with lean, bit-exact, hand-written PTX**; the last ~11 % is *not* a tweak — it needs a
CUTLASS-style warp-specialized persistent mainloop with a custom Pearl-transcript epilogue (multi-day
effort + large dependency). See memory `lpminer-technique.md`. The pearl-research-labs reference is
CUTLASS too, but sm_90-only; lpminer wrote its own `Sm80KernelTraits` warp-spec path for Ampere/Ada.

### Settled: the "cutlass" kernel-name ptxas hack (+1%, NOT adopted)
The viral "FP8 is 150 TFLOPS faster when the kernel name contains `cutlass`" claim — ptxas keys a
more aggressive load↔math scheduling pass off the `cutlass` substring in the symbol. Tested it
properly: built two binaries identical except the kernel symbol (`pearl_ampere_ldm_kernel` vs
`cutlass_pearl_ampere_ldm_kernel`), ran a controlled back-to-back A/B (`tcths` mode, 300 iters,
n=6 each, both orderings to cancel thermal drift):
```
            run1   run2   run3  | run4   run5   run6  | mean
cutlass :  24.12  24.06  24.03  | 24.25  24.04  24.01 | 24.085
orig    :  24.08  23.84  23.78  | 23.82  23.81  23.70 | 23.838
```
**Real and reproducible (+1.0%), but NOT the FP8 magnitude.** cutlass-named wins every adjacent pair
in BOTH orderings (so it's the name, not run-order). Why only +1% for us: that pass recovers
*load-scheduling* slack; our int8 `mma.sync` kernel has **no spills** and is bound on the
**MMA-dependency `wait` stall**, not load scheduling, so it only tightens the `ldmatrix`/IMMA
interleave a hair. **Not adopted** — a published miner shouldn't depend on an undocumented,
driver-version-fragile name heuristic for 0.25 TH/s; recorded as a curiosity. (Bench `tcths` mode +
the throwaway `bench_orig`/`bench_cutlass` binaries were the test rig.)

## Roadmap to 27 TH/s — the warp-specialized persistent rewrite
24.1 is the ceiling of the **single-loop design** (every warp loads AND computes). lpminer's ~27 comes
from a structure we haven't built: **warp specialization + persistence** ([[lpminer-technique]]). This
is the plan to get there, bit-exact.

**Core insight — why warp-spec wins on Ada (no `setmaxnreg`!).** Ampere/Ada have **no per-warp
register reallocation** (`setmaxnreg` is sm_90+). So the win is NOT "donate producer regs to
consumers." It is **instruction-stream decoupling**: today each consumer warp interleaves `cp.async`
(global→smem) + address math between its IMMAs, so the warp scheduler can't issue MMA back-to-back
(`mio_throttle` 2.04, tensor pipe only 48.7%). Move ALL `cp.async` onto a few dedicated **producer**
warps; the **consumer** warps then issue a dense `ldmatrix→IMMA→IMMA…` stream with no global-load
contention. Producers run ahead through a named-barrier ring so consumers never stall on DRAM/L2
(kills `long_scoreboard`). Net: tensor pipe rises from 48.7% toward ~60–80%.

**The gain math (why 27 is realistic, not aspirational):** throughput ∝ tensor-pipe utilization.
24.1 TH/s @ 48.7% ⇒ each +1pt util ≈ +0.49 TH/s. **27 needs only ~54.5% util** — a modest lift that
WS directly targets; ~60% util ≈ 29.7 TH/s (headroom above 27). lpminer proves ~55%+ is reachable on
this exact silicon. So the structure, not the clock or the card, is the gap.

**Design (hand-rolled, NOT CUTLASS).** CUTLASS does *not* give Ampere warp-spec for free — stock CUTLASS
Sm80 collectives are multistage (not `_ws`); lpminer wrote custom `Sm80KernelTraits`. And our
**in-mainloop R-block transcript fold** (16× per region, rotate-13 XOR, continuous accumulation) is
not a standard end-of-loop epilogue a CUTLASS EpilogueVisitor can express. We already own every
bit-exact piece (`swz32`, `ldm_A_frag`, `ldm_B2_frag`, `.cg cp.async`, the fold). So **extend our
kernel**, using CUTLASS only as a reference for mbarrier-pipeline patterns.
- **CTA tile: keep 128×256** (proven best; matches lpminer).
- **Warps: 8 consumers (→ 8 row-tiles ×16, 32 frags = 128 accum regs each, ~our current consumer body)
  + 2 producers** = 10 warps / 320 threads. Reg budget 65536/320 = 204/thread ⇒ still 1 block/SM
  (consumers ~180, producers ~40 "wasted" but we're not at the 65536 wall, so it's free).
- **Sync: HW named-barrier ring** (`bar.arrive`/`bar.sync`, 16 IDs/CTA), NOT mbarrier. Named barriers
  are stateless rendezvous → no phase-bit to mismanage (mbarrier's #1 deadlock source) and native
  subset-sync for the producer/consumer groups. They do **control** handoff only; keep
  `cp.async.commit_group`/`wait_group` for **data-landed** (we already use these). Per stage: a
  `full[S]` barrier (producer `bar.arrive` after `wait_group` confirms the copy landed; consumer
  `bar.sync` before reading) and an `empty[S]` barrier (consumer `bar.arrive` after consuming;
  producer `bar.sync` before reusing buffer S). 2·STAGES ≤ 16 IDs ⇒ STAGES ≤ 8 (we run 3–5).
  CUTLASS uses `arch::NamedBarrier` for the same Ampere cross-warp handoff — battle-tested.
  *Tradeoff:* slightly coarser producer look-ahead than `cp.async.mbarrier.arrive`; negligible at
  3–5 stages. mbarrier is a **conditional P3 upgrade** only if residual `long_scoreboard` proves
  look-ahead is the limiter. *Gotcha:* exact thread counts + no participant early-exit/divergence
  before its arrive/sync or it hangs — near-zero risk for us (full 128×256 tiles, no ragged boundary).
- **Only consumers fold the transcript**; intra-warp ordering already guarantees the fold reads
  completed R-block accumulators (no cross-warp accumulator sharing — low bit-exact risk).

**Phased plan (bit-exact gate after EVERY phase = revert on regression):**
- **P0 scaffold (½d):** add `pearl_ampere_ws_kernel` + `ws`/`profws` bench modes + DP4A bit-exact check,
  beside the shipping kernel (don't touch the live path). Reuse all existing bit-exact device fns.
- **P1 named-barrier ring, NO split (1d):** swap the `__syncthreads` double-buffer for a named-barrier
  (`bar.arrive`/`bar.sync`) multi-stage ring with ALL warps still homogeneous (load+compute). Isolates
  the **ring-buffer mechanics** — stage indexing, smem-offset rotation, cp.async group depth, staged
  data flow — decoupled from warp roles, with dead-simple sync (no phase bits). Gate: BIT-EXACT, ≈24.
- **P2 warp specialization (2–3d, the core):** flip warps into roles — wrap the produce path in
  `if (warp<2)`, the consume path in `if (warp>=2)`, pointing at the *same* barrier IDs (the P1 ring is
  unchanged; only who calls `arrive` vs `sync` changes). 8 consumer + 2 producer. Gate: BIT-EXACT, then
  measure — **the jump toward 27 lands here.** Tune STAGES{3,4,5}, producers{2,3,4}.
- **P3 ILP/SASS tune (1d):** with cp.async off consumers, densify the IMMA issue (back-to-back, A-operand
  `.reuse`); verify in `nvdisasm` the consumer loop is MMA-dense. Maybe push NT within 255 regs.
  *Conditional:* if residual `long_scoreboard` shows producer look-ahead is the limiter, upgrade the
  named-barrier ring → **mbarrier** (`cp.async.mbarrier.arrive` unifies data+control) for deeper overlap.
- **P4 persistence (1–2d):** launch `num_SM × blk/SM` (~20–40) persistent blocks; static-partition the
  512 region-tiles (block b ⇒ tiles b, b+G, …); order tiles for L2 reuse (n fastest within an m-band so
  A-rows stay hot). Overlap tile-T epilogue with T+1 prologue only if needed. Gate: BIT-EXACT, +2–4%.
- **P5 integrate (½d):** dispatcher routes 128×256 region → WS kernel (old kernel = fallback); validate
  bit-exact at real 131072² multi-region + sustained TH/s + thermals; A/B vs lpminer at the pool.

**Total ≈ 6–8 focused days.** Risks: (1) named-barrier hang from miscounted thread totals or a
participant diverging/exiting before its arrive/sync — *now the main risk, but low*: full 128×256
tiles → no ragged-boundary divergence, and P1 isolation + diffing against the known-good 24.0 kernel
at small grids catches indexing bugs early. (2) producers can't sustain cp.async rate → consumer
starvation → no gain; mitigated by tunable producer count. (3) WS may land at 25–26 not full 27 —
still beats 24.1 and is 93–96% of lpminer. Intermediate success = any bit-exact result >24.1 with
tensor pipe >55%.

## Invariant
Every change must keep `bench_ampere.exe` reporting **BIT-EXACT PASS** (TC transcript ==
DP4A transcript). A correctness regression = revert.
