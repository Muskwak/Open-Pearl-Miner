# p40-alpha-miner

**Fork of [AlphaMine-Tech/alpha-miner](https://github.com/AlphaMine-Tech/alpha-miner) with Pascal P40 (sm_61) support.**

Mines the **Pearl (PRL)** network using **FP32 CUDA cores** and **INT8 DP4A** instructions on Pascal-generation GPUs (Tesla P40, GTX 1080, GTX 1080 Ti, etc.).

## Architecture support

| Architecture | Compute Capability | Tensor Cores | DP4A | Status |
|---|---|---|---|---|
| Pascal (GP10x) | sm_61 | None | Yes (INT8) | **Supported** |
| Volta+ (original) | sm_70+ | Yes | Yes | See alpha-miner upstream |

P40 has no Tensor Cores — all INT8 matrix math uses `__dp4a` (dot-product-4-accumulate) on CUDA cores.

## Why fork?

The original alpha-miner only ships precompiled binaries targeting Volta (sm_70) and newer, requiring Tensor Cores. Pascal GPUs (like the P40) have 24 GB VRAM and 3840 CUDA cores but no Tensor Cores, making them an excellent low-cost mining option if the kernels are adapted.

## Key changes from upstream

| Component | Upstream (sm_90) | This fork (sm_61) |
|---|---|---|
| INT8 GEMM | WGMMA on Tensor Cores | `__dp4a` on CUDA cores |
| Memory fabric | TMA (Tensor Memory Accelerator) | `ldg/stg` + shared mem |
| Pipeline | `PipelineTmaAsync` | Manual double-buffering |
| Cluster sync | `NamedBarrier` / warp-group | `__syncthreads` / cooperative groups |
| Compilation target | `sm_90a` | `sm_61` + `sm_70+` |

## Performance expectations

On P40 (1417 MHz boost, 3840 CUDA cores, 24 GB HBM2):
- **INT8 DP4A path:** ~2-4 TOPS (estimated 15-30 TH/s for Pearl)
- **FP32 path:** ~0.5-1 TFLOPS (estimated 5-10 TH/s)

Actual numbers depend on kernel tile sizes, memory bandwidth (346 GB/s), and PCIe generation.

## Building

### Prerequisites

- CUDA Toolkit 12.x (last version supporting sm_61)
- Python 3.12+
- PyTorch 2.x (matching CUDA version)
- NVIDIA driver 545+ (R550 UDA driver supports P40)
- `uv` package manager

### Build from source

```bash
# Clone this fork
git clone --recursive https://github.com/YOUR_USER/p40-alpha-miner
cd p40-alpha-miner

# Build with Pascal target
uv sync --package p40-pearl-gemm

# The build system auto-selects sm_61 for P40.
# Override with:
#   TARGET_ARCH=sm_70 uv sync --package p40-pearl-gemm
```

## Usage

```bash
# List devices
./alpha-miner --list-devices

# Mine (single GPU)
./alpha-miner --pool stratum+tcp://us2.alphapool.tech:5566 \
  --address prl1pYOURPEARLADDRESS \
  --worker p40-rig

# Static difficulty for P40 (start with 4096)
./alpha-miner --pool stratum+tcp://us2.alphapool.tech:5566 \
  --address prl1p... \
  --worker p40-rig \
  --password 'x;d=4096'
```

## Technical approach

### INT8 DP4A GEMM (`sm_61` path)

Pascal's `__dp4a` instruction computes a signed 8-bit dot product in a single cycle:

```cuda
int __dp4a(int a, int b, int c);
// c = c + sum_{i=0..3} byte(a, i) * byte(b, i)
```

Matrix multiplication is tiled: each thread block loads 64×64 INT8 tiles into shared memory, then threads compute partial dot products using `__dp4a`, accumulating into `int32`. Shared memory is double-buffered to overlap compute with global memory loads.

### FP32 fallback

For noising and denoising steps (non-GEMM operations), standard FP32 CUDA core operations are used.

## Project structure

```
p40-alpha-miner/
├── README.md                 # This file
├── install.sh                # Installer (adapted)
├── p40-pearl-gemm/           # Pascal CUDA extension
│   ├── setup.py              # Build config (sm_61 target)
│   ├── csrc/
│   │   ├── gemm/             # CUDA kernels for Pascal
│   │   │   ├── dp4a_gemm.cu  # INT8 DP4A GEMM mainloop
│   │   │   ├── fp32_gemm.cu  # FP32 fallback kernels
│   │   │   ├── noising.cu    # Noising (Pascal-optimized)
│   │   │   └── denoise.cu    # Denoising (Pascal-optimized)
│   │   ├── blake3/           # BLAKE3 hash (arch-independent)
│   │   ├── noise_gen/        # Noise generation (arch-independent)
│   │   └── quantize/         # INT8 quantization (arch-independent)
│   └── python/               # Python bindings
└── alpha-miner               # Symlink to p40-pearl-gemm's wrapper binary
```

## License

Binary redistribution as permitted by the original alpha-miner license. Custom CUDA kernel source is available under MIT license.
