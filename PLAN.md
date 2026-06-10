# P40-Pearl-GEMM — Architecture & Plan

## Goal

Fork [AlphaMine-Tech/alpha-miner](https://github.com/AlphaMine-Tech/alpha-miner) to enable Pearl (PRL) proof-of-work mining on **NVIDIA Tesla P40 (sm_61)** using:

- **INT8 DP4A** (via `dp4a.s32.s32` PTX) for the GEMM hot loop
- **FP32** CUDA cores for noise generation and auxiliary kernels
- No Tensor Core dependency (P40 has no Tensor Cores)

## Hardware

| Device | Architecture | Key Feature |
|--------|-------------|-------------|
| Tesla P40 | sm_61 (Pascal) | DP4A INT8 intrinsic, no Tensor Cores |
| GTX 1070 | sm_61 (Pascal) | Same feature set for dev/test |

## Software Stack

- **CUDA 12.8** toolkit at `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`
- **PyTorch 2.6.0+cu124** (CUDA 12.4 runtime)
- **Python 3.13.12** (conda env `pytorch`)
- **MSVC 2022** (cl.exe)
- **CUTLASS** headers at `aphrodite-engine\.deps\cutlass-src\include\`
- **Windows x64**

## Repository Structure

```
p40-alpha-miner/
├── p40-pearl-gemm/                    # Buildable package
│   ├── setup.py                       # Build config (sm_61 primary target)
│   ├── pyproject.toml
│   ├── tests/
│   │   └── test_build.py              # Smoke tests
│   ├── python/
│   │   ├── __init__.py                # Re-exports from bindings
│   │   └── p40_gemm_bindings.py       # PyTorch -> C API bindings
│   └── csrc/
│       ├── blake3/                     # Upstream BLAKE3 (no mods needed)
│       ├── tensor_hash/                # Upstream Merkle tree (no mods needed)
│       └── gemm/
│           ├── dp4a_gemm_sm61.cu       # ★ NEW: Pascal INT8 GEMM via dp4a.s32.s32
│           ├── noising_sm61.cu         # ★ NEW: Pascal noise A/B kernels
│           ├── api_sm61.cu             # ★ NEW: C API + inline denoise converter
│           ├── convert_util.h          # Upstream FlashAttention-derived converter
│           ├── denoise_converter.cu    # Upstream template instantiation
│           ├── denoise_converter_host.h # Upstream launch wrapper (patched: no cluster_launch)
│           ├── denoise_converter_kernel.h # Upstream CuTe kernel (uses CUTLASS types)
│           ├── error_check.hpp         # Upstream CUDA error checking
│           ├── host_signal_header.hpp   # Upstream host signal structs
│           ├── inner_hash_kernel.cu    # Upstream inner hash kernel (uses CuTe)
│           ├── inner_hash_kernel.h     # Upstream inner hash host header
│           ├── noise_generation.cu     # Upstream template instantiation
│           ├── noise_generation_host.h # Upstream launch wrapper (patched: no cluster_launch)
│           ├── noise_generation_kernel.h # Upstream CuTe noise gen kernel
│           ├── pearl_api_params.h      # Upstream param structs
│           ├── pearl_gemm_constants.hpp # Upstream scaling constants
│           ├── pow_utils.hpp           # Upstream PoW helpers (uses CuTe)
│           ├── quantization_util.cuh   # Upstream quant util from vLLM
│           ├── quantize_kernel.cu      # Upstream quant kernel
│           ├── quantize_kernel.hpp     # Upstream quant kernel header
│           └── utils.h                 # Upstream CuTe/CUTLASS utilities
└── PLAN.md                            # This file
```

## Legend

| Marker | Meaning |
|--------|---------|
| ★ NEW  | Written from scratch for Pascal P40 (no upstream equivalent) |
| ★ PATCHED | Upstream file with minimal modifications to remove SM90 dependency |
| (no marker) | Copied verbatim from upstream alpha-miner |

## Kernel Inventory

### Pascal-Specific (pure CUDA, no CUTLASS)

| Kernel | File | Description |
|--------|------|-------------|
| `dp4a_gemm_kernel` | `dp4a_gemm_sm61.cu` | INT8 GEMM via `dp4a.s32.s32`, 64×64 tiles, 4 warps, shared memory tiling, dequant to fp16 |
| `noise_A_kernel` | `noising_sm61.cu` | INT8 noise A: loads A/EAL/EBL/EAR → shared mem → computes ApEA + AxEBL |
| `noise_B_kernel` | `noising_sm61.cu` | INT8 noise B: same structure for B (stub implementation) |
| `launch_denoise_converter` | `api_sm61.cu` | Simple int32→fp16 conversion inline (no CuTe/CUTLASS) |

### Upstream (use CuTe/CUTLASS via included headers)

| Kernel | File | Description |
|--------|------|-------------|
| `NoiseGenerationKernel<R,N>` | `noise_generation_kernel.h` | BLAKE3-based random noise matrix generation |
| `DenoiseConverterKernel<R,N>` | `denoise_converter_kernel.h` | CuTe tiled int32→fp16+scale conversion |
| `inner_hash_kernel<NUM_ITER>` | `inner_hash_kernel.cu` | XOR-reduction inner hash with CuTe tensors |
| quantize kernel | `quantize_kernel.cu` | Dynamic per-token int8 quantization from vLLM |

### Architecture-Independent Headers

| File | Source |
|------|--------|
| `blake3/` | Upstream (standalone CUDA BLAKE3) |
| `tensor_hash/` | Upstream (CuTe-based Merkle tree) |

## Build Configuration

**Target architecture**: `sm_61` (Pascal P40/GTX 1070) by default

**Additional archs** (via `ADDITIONAL_ARCHS` env): `sm_70,sm_75,sm_80,sm_86`

**CUTLASS include path**: `aphrodite-engine\.deps\cutlass-src\include` (hardcoded absolute)

**Key nvcc flags**:
- `-std=c++17`, `--expt-relaxed-constexpr`, `--expt-extended-lambda`
- `--use_fast_math`, `-lineinfo`
- Half enablement: `-U__CUDA_NO_HALF_OPERATORS__` etc.

**Build command**: `python setup.py build_ext --inplace`

## Patches Applied

### 1. `setup.py` — CUTLASS include path
Add CUTLASS headers for CuTe/CUTLASS template resolution. Now configurable via
the `CUTLASS_DIR` env var (falls back to the local aphrodite-engine checkout).
Also adds `csrc/gemm/bindings.cpp` to the source list (and drops the SM90-only
`csrc/tensor_hash/tensor_hash.cu`; see "Remaining work").

### 2. `noise_generation_host.h` — Remove SM90 cluster_launch
Remove `#include "cutlass/cluster_launch.hpp"`. Keep `cutlass/device_kernel.h` which provides `cutlass::device_kernel<Op>` — a simple `__global__` wrapper compatible with sm_61.

### 3. `denoise_converter_host.h` — Remove SM90 cluster_launch
Same patch: replace `cluster_launch.hpp` with `device_kernel.h`.

### 4. `pow_utils.hpp` — Remove SM90 MMA trait include
Remove `#include <cute/atom/mma_traits_sm90_gmma.hpp>`. The `xor_reduction` function only needs `cute/array.hpp` and `cute/fold.hpp` which come transitively via `cute/tensor.hpp` (included from `utils.h`).

### 5. `bindings.cpp` — NEW pybind11 module (was entirely missing)
The Python layer (`p40_gemm_bindings.py`) called `_C.dp4a_gemm(...)` etc., but
**no `PYBIND11_MODULE` existed anywhere** — the built extension exported zero
callable functions. Added `csrc/gemm/bindings.cpp` registering `dp4a_gemm`,
`noise_A`, `noise_B`, `denoise_converter`, and `inner_hash` as torch-tensor
wrappers. Built as a **host `.cpp`** (MSVC/gcc), not nvcc: the torch headers do
not parse under nvcc + C++20 (the upstream CuTe code requires C++20). Each
wrapper installs a `c10::cuda::CUDAGuard` so launches target the tensors'
device (a multi-GPU box otherwise dereferences another device's pointers).

### 6. `dp4a_gemm_sm61.cu` — Rewrote the GEMM thread→tile mapping
The original kernel's `a_row = lane/16 + warp_x*16` only ever produced output
rows {0,1,16,17}, so it wrote 4 of every 64 rows and left the rest garbage.
Replaced with a 16×16-thread / 4×4-micro-tile scheme that covers the full
64×64 output tile. **Validated on Tesla P40 and GTX 1070** (max relative error
0.0005 vs. an int reference). See `tests/test_gemm_native.cu` and
`tests/test_module_e2e.py`.

### 7. `noising_sm61.cu` — Rewrote noise_A and implemented noise_B
The original `noise_A` was incorrect (partial K-reduction, cross-lane races,
no-op `+ eal_val*0`) and `noise_B` was an empty stub. Both rewritten to the
exact algebra recovered from `noise_generation_kernel.h`:
`ApEA = A + EAL·EAR`, `AxEBL = A·EBLᵀ` (clean A); `BpEB = B + EBR·EBL`,
`EARxBpEB = BpEB·EARᵀ` (noised B). Inputs are 7-bit quantized so the noised
int8 matrices fit without any bit-shift. **Validated bit-exact vs. a CPU
reference on P40/GTX 1070** (`tests/test_noise_native.cu`, `test_module_e2e.py`).

### 8. `api_sm61.cu` — Fixed denoise converter (did not compile)
`launch_denoise_converter` was a `__host__` function that referenced
`blockIdx`/`threadIdx`/`gridDim` and invoked a `__device__` lambda from host —
it could never compile. Rewrote as a proper `__global__` kernel + host
launcher, using the correct **separate** scale factors confirmed against
`DenoiseConverterKernel`: AxEBL ÷ `kAxEBLScaleFactor` (1<<14), EARxBpEB ÷
`kEARxBpEBScaleFactor` (1<<12).

### 9. Build flags & portability (`setup.py`, headers)
- `-std=c++20` for nvcc (upstream uses designated initializers in `blake3.cuh`
  and `requires` clauses in `utils.h`).
- Host (`cxx`) flags made MSVC-compatible on Windows (`/O2`).
- `CUTLASS_DIR` env var for the CUTLASS include path.
- `merkle_tree_utils.hpp`: replaced host-only `ceil(log2(num_leaves))` with a
  device `__clz`-based `ceil_log2_u32` (the `<cmath>` integral `log2` overload
  is not callable from device code).
- `quantize_kernel.cu` / `.hpp`: removed unused `<torch/all.h>` / `<ATen/ATen.h>`
  (they don't parse under nvcc + C++20) and replaced `cuda::maximum<>` (shadowed
  by torch's bundled CCCL) with a local `fmaxf` functor.

## Build & validation status (CUDA 12.8, MSVC 2022, torch 2.6.0+cu124, sm_61)

`python setup.py build_ext --inplace` **succeeds** and produces
`p40_pearl_gemm_cuda*.pyd`. `tests/test_module_e2e.py` passes on **both** the
Tesla P40 and the GTX 1070 for every registered op:

| Op | Status |
|----|--------|
| `dp4a_gemm` | ✅ build + numerically validated (rel err 5e-4) |
| `noise_A` / `noise_B` | ✅ build + bit-exact vs CPU reference |
| `denoise_converter` | ✅ build + bit-exact (1<<14 / 1<<12) |
| `inner_hash` | ✅ build + runs (CuTe kernel works on sm_61) |
| `pearl_pow` | ✅ **Pascal PoW core — bit-exact vs the reference transcript + keyed-BLAKE3** on the P40 (see below) |
| `tensor_hash` | ✅ **Pascal rewrite, bit-exact vs stock BLAKE3 keyed hash** on the P40 for all realistic configs (see below) |
| `noise_generation.cu`, `denoise_converter.cu` | ✅ compile on sm_61 (not yet wired into bindings) |
| `quantize_kernel.cu` | ✅ compiles (not yet registered; quantization is also trivially doable in torch) |

### `tensor_hash` Pascal rewrite (DONE)
The SM90 `merkle_tree_roots_kernel` (TMA + warpgroup pipelines) was replaced by
`merkle_tree_roots_kernel_sm61.hpp` — one thread per 1024-byte chunk, BLAKE3
keyed leaf via direct global loads, then the stock per-CTA Merkle reduction. The
downstream `ComputeBlakeMTKernel`/`ReduceRootsKernel` and the commitment-hash
kernel are plain CuTe and reused unchanged via `tensor_hash_host_sm61.hpp`.
`tests/test_tensor_hash.py` confirms the output equals
`blake3.blake3(data, key=key).digest()` for every config with `num_roots ≥ 2`
(`num_roots = ceil(num_chunks / threads_per_block)`), which is **every config
the miner actually uses** — `run_tensor_hash` requires data > 2¹⁷ bytes and the
default `threads_per_block = 128`. The degenerate `num_roots == 1` case skips
BLAKE3's ROOT finalization (matching the upstream pipeline's structure) and is
never hit in production.

## Reference source (the earlier "hard ceiling" is lifted)

The canonical Pearl source is public at **github.com/pearl-research-labs/pearl**
(cloned to `C:\Users\ADMIN\audits\pearl-ref`). It contains the full GPU source
(`miner/pearl-gemm`, incl. the fused `pearl_gemm_kernel.h` + collective
mainloop/epilogue that were missing here) **and** readable Python reference
implementations in `miner/miner-base/src/miner_base/` (`noisy_gemm.py`,
`noise_generation.py`, `inner_hash.py`, `matrix_merkle_tree.py`) plus a
validation oracle (`miner/vllm-miner/tests/test_reference_vs_kernels.py`).

The PoW, recovered from `noisy_gemm.py`: tile m,n,k by `R = noise_rank` (128);
per 16×16 hash tile, for each full k-tile `t`, XOR-reduce the **cumulative**
int32 partial sum and `transcript[t%16] = rotl32(transcript[t%16],13) ^ h`; the
tile wins if `blake3(transcript_LE, key=noise_seed_A) ≤ target`. The denoised
result (`C − A·E_BL·E_BR − E_AL·E_AR·B_noised`) is the *inference* output and is
**not** part of the PoW hash.

### `pearl_pow` Pascal PoW (DONE, validated)
`pearl_pow_sm61.cu` computes the PoW with one CUDA block per 16×16 tile, DP4A for
the int8 contraction, and the validated `blake3.cuh` for the keyed compression.
`tests/test_pearl_pow.py` confirms the per-tile digests are **bit-exact** vs a
faithful CPU reimplementation of `noisy_gemm.py` (shapes 128–512, R=64/128) and
the ≤target found-flag behaves correctly. The noising kernels (`noise_A/noise_B`)
are algebraically consistent with `noisy_gemm.py`'s `noise_A/noise_B`.

Option A's compute pipeline is delivered and validated on your P40.

### What now works (validated on the Tesla P40)

`python/pearl_miner.py` — a complete Pascal mining pipeline:

`mine_once(header, target, A, B)`:
  `key      = blake3(header + mining_config_bytes)`
  `seeds    = commit(A, B, key)`        # CUDA tensor_hash Merkle roots + BLAKE3
  `E_*      = generate_noise(seeds)`    # faithful port of noise_generation.py
  `A_n,B_n  = noised_operands(...)`     # noisy_gemm.py noise_A/noise_B
  `result   = pearl_pow(A_n, B_nᵀ, pow_key=seed_A, target)`   # validated CUDA kernel

`tests/test_pipeline_e2e.py` passes: it finds a block at an easy target,
independently recomputes the winning 16×16 tile's transcript and confirms
`blake3(transcript, key=seed_A) ≤ target`, finds nothing at the hardest target,
and `mine_once` agrees. The PoW core (`pearl_pow_sm61.cu`) is bit-exact vs the
reference `noisy_gemm.py`.

So the entire consensus computation — commit → noise → noised GEMM → per-tile
keyed-BLAKE3 PoW — runs correctly on a P40. That's the part that was impossible
before we had the reference.

### What I learned about the work model (the key unlock)

A job is only `(incomplete_header_bytes, target)` — the miner picks A and B
itself and commits to them. Pearl is proof-of-*useful*-work, but consensus just
needs validly-committed operands, so a standalone miner with synthetic (or
real-inference) matrices is protocol-valid. `MatrixMerkleTree == flat keyed
BLAKE3`, so the `tensor_hash` commitment is network-faithful for full-size
matrices.

### AlphaPool protocol (recovered from live probing)

AlphaPool (`stratum+tcp://eu2.alphapool.tech:5566` / `us2.alphapool.tech:5566`)
uses a custom Stratum-like protocol over line-delimited JSON-RPC:

| Direction | Message | Description |
|-----------|---------|-------------|
| Pool → Miner | `{"id":null,"method":"pearl.challenge","params":{"seed":"<64-hex>","difficulty":32}}` | Sent on connect and in response to any message |
| Miner → Pool | `{"id":N,"method":"mining.submit","params":["wallet.worker","<seed-hex>","<proof-base64>"]}` | Submit a share |

Key observations from probing:

- **No standard Stratum handshake.** `mining.subscribe` / `mining.authorize`
  trigger another `pearl.challenge` but no subscription response — the protocol
  is effectively stateless: connect → receive challenge → mine → submit.
- **Silent drop of invalid shares.** The pool sends no error/result response
  for invalid `mining.submit` requests (tested with dummy data across ~10
  format variants). Only a valid `PlainProof` will elicit a response.
- **Connection abort on wrong format.** Sending `[wallet, worker, proof]` (with
  wallet and worker as separate params) causes the pool to abort the connection;
  the correct format is `[wallet.worker, seed_hex, proof_base64]` (combined
  wallet.worker as first param).
- **`seed` = 32 bytes** — treated as the `incomplete_header_bytes` directly in
  our pipeline (the pool doesn't send a full 76-byte `IncompleteBlockHeader`).
- **`difficulty` = number of leading zero bits required** in the jackpot hash.
  The effective target for the CUDA kernel is
  `2^(256-difficulty) × tile_size × dot_product_length` (matching the Rust
  `extract_difficulty_bound` logic). For the default config (k=1024, rank=128,
  tile_h=16, tile_w=16) with difficulty=32, the effective target has ~22
  leading zero bits.
- **EU pool** (`eu2.alphapool.tech`) is currently the most responsive server;
  the US server (`us2.alphapool.tech`) rate-limits after ~5 connections.

The reference Pearl protocol (in `pearl-ref`) uses a different flow
(`getMiningInfo`/`submitPlainProof` with a local pearl-gateway), but
AlphaPool's custom `pearl.challenge`/`mining.submit` protocol is simpler.

### What's implemented

| Component | File | Status |
|-----------|------|--------|
| `MiningConfiguration.to_bytes()` (52-byte, Rust-compatible) | `python/mining_config.py` | ✅ validated |
| `PlainProof` + `MatrixMerkleProof` + bincode serialization | `python/gateway_client.py` | ✅ matches Rust bincode format |
| `AlphaPoolClient` (pool connect, challenge recv, share submit) | `python/gateway_client.py` | ✅ tested live |
| `pool_target()` (difficulty → U256 with `extract_difficulty_bound` adjustment) | `python/pearl_miner.py` | ✅ correct |
| `generate_matrices()` (random int8 A/B) | `python/pearl_miner.py` | ✅ |
| `pool_miner.py` (full mining loop) | `python/pool_miner.py` | ✅ ready (needs GPU to mine valid shares) |

### To make A a live miner on the P40

1. **Run `pool_miner.py` on the P40 machine** with a real wallet:
   ```
   uv run python pool_miner.py --wallet prl1... --worker my_rig \
       --pool eu2.alphapool.tech:5566
   ```
   The pool's difficulty=32 should be trivially satisfiable — expect a valid
   share within seconds.

2. **If accepted,** the `mining.submit` format is confirmed and A is a live
   miner.  **If not,** the pool may require additional fields (the exact 76-byte
   `IncompleteBlockHeader` instead of the raw 32-byte seed, or a different
   `job_id` encoding).  Revise `pool_miner.py` accordingly.

3. **Then B (llama.cpp):** route a model's linear-layer GEMMs through this
   same pipeline for real useful-work mining.

## Work model (recovered from pearl-gateway / miner-base)

A `MiningJob` is just `(incomplete_header_bytes, target)`. **The miner chooses
A and B** — Pearl is proof-of-*useful*-work, but consensus only requires validly
committed operands, not a specific model. Per attempt:
`key = blake3(incomplete_header_bytes + MiningConfiguration.to_bytes())`; the
matrices are committed via keyed-BLAKE3 Merkle roots of `A` and `Bᵀ`
(`MatrixMerkleTree`, which equals flat keyed BLAKE3 of the bytes), then
`commitment_B = blake3(key+root_B)`, `commitment_A = blake3(commitment_B+root_A)`;
`(noise_seed_A, noise_seed_B) = (commitment_A, commitment_B)`. Noise is generated
from the seeds; `pow_key = noise_seed_A`. A win is submitted as `OpenedBlockInfo`
→ `PlainProof` (Merkle proofs of the winning tile's A rows / B cols) → the gateway
wraps it in a zk-pow proof and submits to the node.

### End-to-end Pascal pipeline (DONE, validated — `python/pearl_miner.py`)
`mine_once(header, target, A, B)` chains: derive key → commit (CUDA `tensor_hash`
Merkle roots + BLAKE3) → generate noise → noised operands → `pearl_pow`.
`tests/test_pipeline_e2e.py` runs it on the P40: finds blocks at an easy target,
**independently re-verifies the winning tile's transcript ≤ target**, and finds
nothing at the hardest target. The cheap steps (noise gen, noised operands) are
faithful torch/BLAKE3 ports of `noise_generation.py` / `noisy_gemm.py`; the hot
path is the validated `pearl_pow` CUDA kernel.

## Remaining work for live mining

1. **Pool miner validation on P40.** Run `pool_miner.py` on the P40 machine
   where `p40_pearl_gemm_cuda` is built. The pool's `difficulty=32` should yield
   a valid share quickly. If accepted, the protocol format is confirmed.
2. **If share submission is silently rejected**, the pool may require a full
   76-byte `IncompleteBlockHeader` (with `prev_block=seed`, `merkle_root=0`,
    `timestamp=now`, `nbits` from difficulty) instead of the raw 32-byte seed,
    or a different `job_id` format.  Revise `pool_miner.py` accordingly.
3. **Perf:** move noise gen / noised-operand matmuls from torch onto the existing
   CUDA kernels (`noise_gen`, `noise_A`, `noise_B`) and fuse; tune `pearl_pow`.
4. **Optional Pascal denoise-subtract** for the actual inference output (only the
   "useful work" result, not the PoW share).
5. **Host (option B): llama.cpp** — route a model's linear-layer GEMMs through the
   noised pipeline for genuine useful-work mining (better Pascal inference than
   vLLM/aphrodite; the README anticipates non-vLLM plugins).

## Residual notes
- The build still targets sm_61 plus sm_70/75/80/86 (from torch's default arch
  list); set `TORCH_CUDA_ARCH_LIST=6.1` to build P40-only and cut build time.
- `pow_utils.hpp` still contains unused SM90 template code (`TileHashAccumulator`)
  that parses but is never instantiated on the Pascal path.
