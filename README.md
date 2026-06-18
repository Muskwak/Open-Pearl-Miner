# Open Pearl Miner

A high-performance **Pearl (PRL)** proof-of-work miner for **NVIDIA GPUs** — Pascal
(Tesla P40, GTX 1070 / 1080, other `sm_61` DP4A cards) **and Ampere / Ada (`sm_80+`)
via a fused int8 tensor-core GEMM**.

No Python, CUDA toolkit, or PyTorch required **to run** — the CUDA runtime is bundled
in pre-built releases. Just an NVIDIA driver and the standalone binary.
Build it yourself or grab a pre-built release. See [License](#license) for dev-fee terms.

> Mixed-GPU rigs are supported. Ampere/Ada cards run a fused int8 tensor-core kernel
> (GEMM + in-mainloop transcript fold + tensor-core noise generation), bit-exact with
> the Pascal DP4A path; Pascal cards run DP4A.
> 
## Features

- **~7.0 TH/s** sustained on a single Tesla P40 (Pascal DP4A).
- **~28 TH/s** on a single RTX 4050 mobile (Ada) — fused int8 **tensor-core** GEMM +
  in-mainloop transcript fold + tensor-core noise generation, all bit-exact with DP4A.
- **Multi-GPU** — auto-detects every GPU and runs one worker per card, near-linear scaling.
- **Continuous mining** — no idle time waiting between pool jobs.
- **Background proof submission** — finding a share never stalls the search.
- **Solo mining** — mine to your local pearl-gateway node.
- **Pool mining** — built for LuckyPool's Pearl stratum (default) with more pool protocols planned.
- **HiveOS** — full HiveOS custom-miner package.

## Roadmap

- **AMD RDNA (HIP / ROCm)** — int8 WMMA GEMM + transcript-fold path for RDNA2/RDNA3
  (`gfx10xx` / `gfx11xx`). Experimental kernels in progress.
- **Apple Silicon (Metal)** — Metal `simdgroup_matrix` int8 path for M-series GPUs.
- **More pools** — additional Pearl stratum protocols beyond LuckyPool.
- **Per-kernel autotuning** — pick tile/stage parameters per GPU at startup.

## Pre-built Releases

Grab the latest binary from the [Releases](https://github.com/Muskwak/Open-Pearl-Miner/releases) page:

| File | Platform |
|------|----------|
| `p40-miner-windows-x64.zip` | Windows x64 |
| `p40-miner-linux-x64.tar.gz` | Linux (glibc >= 2.31) |
| `p40-miner-hiveos-<ver>.tar.gz` | HiveOS custom miner |

**Requirements:** an NVIDIA Pascal GPU (Tesla P40, GTX 1070/1080, or other `sm_61`), an
up-to-date NVIDIA driver, and **Windows x64**, **Linux x86-64** (glibc >= 2.31), or **HiveOS**.

### Quick Start

```bat
:: Windows
p40-miner.exe --wallet prl1YOURWALLET --worker rig1
```

```bash
# Linux
./p40-miner --wallet prl1YOURWALLET --worker rig1
```

That's it — it auto-detects all GPUs and starts mining.

## Build from Source

### Prerequisites

| Dependency | Windows | Linux |
|---|---|---|
| **CUDA Toolkit** | 12.x ([NVIDIA](https://developer.nvidia.com/cuda-downloads)) | 12.x |
| **CUTLASS headers** | `git clone --depth 1 https://github.com/NVIDIA/cutlass` | same |
| **Python** | >= 3.12 | >= 3.12 |
| **MSVC** | Visual Studio 2022 (C++ workload) | — |
| **pip packages** | `numpy`, `blake3`, `py-pearl-mining` | same |

Set `CUTLASS_DIR` to the directory containing `cutlass/` and `cute/` headers.

### Windows

```bat
git clone https://github.com/Muskwak/Pascal-Pearl-Miner.git
cd Pascal-Pearl-Miner

:: 1. Build the CUDA library (p40cuda.dll)
packaging\build_capi.bat

:: 2. Install Python deps
pip install numpy blake3 py-pearl-mining

:: 3. Run directly from source
python packaging\p40_miner_lite_main.py --wallet prl1YOURWALLET

:: Or freeze a standalone binary with PyInstaller:
pip install pyinstaller
pyinstaller packaging\p40-miner-lite.spec --noconfirm --distpath dist --workpath build_pyi
dist\p40-miner\p40-miner.exe --wallet prl1YOURWALLET
```

To rebuild the full CUDA extension (includes torch bindings):

```bat
pip install -e .   # requires torch + CUTLASS_DIR
```

### Linux

```bash
git clone https://github.com/Muskwak/Pascal-Pearl-Miner.git
cd Pascal-Pearl-Miner

# 1. Build the CUDA library (libp40cuda.so)
CUTLASS_DIR=/path/to/cutlass/include bash packaging/build_capi.sh

# 2. Install Python deps
pip install numpy blake3 py-pearl-mining

# 3. Run directly from source
python packaging/p40_miner_lite_main.py --wallet prl1YOURWALLET

# Or freeze a standalone binary:
pip install pyinstaller
pyinstaller packaging/p40-miner-lite.spec --noconfirm --distpath dist --workpath build_pyi
./dist/p40-miner/p40-miner --wallet prl1YOURWALLET
```

### HiveOS Package

Build the Linux binary first, then:

```bash
bash packaging/hiveos/build_hiveos_package.sh  # -> p40-miner-hiveos-<ver>.tar.gz
```

Upload to a URL or `scp` to the rig, then add as a Custom miner in HiveOS.

### Development Install

```bash
pip install -e .   # editable install of the torch-based extension
CUTLASS_DIR=... python -c "import p40_pearl_gemm"  # smoke test
```

## Usage

### Options

| Flag | Default | Description |
|---|---|---|
| `--wallet` | _(required)_ | Your Pearl payout address |
| `--worker` | `p40` | Worker name shown on the pool |
| `--pool` | _(auto by GPU)_ | Stratum `host:port`. Auto: Pascal→`pearl-cpu-eu1…:3370`, sm_80+→`pearl-eu2…:3360` |
| `--devices` | _(auto-detect all)_ | GPU selection, e.g. `0,1,2` or `all` |
| `--region` | `4096` | Sub-output search size |
| `--solo` | _(off)_ | Solo mine to local pearl-gateway `HOST:PORT` |

### Multi-GPU

Auto-detects every GPU and runs one worker per card (workers named `<worker>-gpu0`,
`-gpu1`, ...) with a combined-hashrate summary. Each GPU runs as an independent pinned
process with blocking-sync CUDA, giving near-linear scaling on 4–8 GPU rigs. Just run it:

```
p40-miner.exe --wallet prl1YOURWALLET
```

Use `--devices` to select specific cards:

```
p40-miner.exe --wallet prl1YOURWALLET --devices 0,2
```

### Pool Mining

The default pool is picked automatically by GPU class: **tensor-core cards (`sm_80+`)**
use `pearl-eu2.luckypool.io:3360` (GPU difficulty), while **Pascal / DP4A cards** (P40,
GTX 10-series) use `pearl-cpu-eu1.luckypool.io:3370`. On a mixed rig each card picks its
own. Override for any card with `--pool`:

```
p40-miner.exe --wallet prl1YOURWALLET --pool pearl-cpu-eu1.luckypool.io:3370
```

(The CPU pool's low difficulty produces frequent shares whose ~1 GB proof snapshot
stalls a fast GPU between grids — hence the GPU-difficulty default for tensor-core
cards.) Currently only LuckyPool's stratum protocol is supported; more planned.

### Solo Mining

```
p40-miner.exe --solo GATEWAY_HOST:PORT
```

No wallet flag needed — the node's configured address receives block rewards.
The same 2% dev fee applies (GPU mines to the dev's pool wallet during the 2% window).

## HiveOS

> **The Custom miner name must be exactly `p40-miner`.** HiveOS derives the miner name
> from the package (`p40-miner-<version>.tar.gz` → folder `p40-miner/` → `CUSTOM_NAME`).
> If the name doesn't match, the agent can't find `h-run.sh` and **the miner never
> launches** from the flight sheet. Keep the asset filename as `p40-miner-<version>.tar.gz`.

Install as a **Custom** miner:

1. **Flight Sheet → Miner → Custom → Setup Miner Config.**
2. **Miner name:** `p40-miner` (exactly — this becomes `/hive/miners/custom/p40-miner/`).
3. **Installation URL:**
   `https://github.com/Muskwak/Open-Pearl-Miner/releases/download/v1.6.0/p40-miner-1.6.0.tar.gz`
4. **Wallet and worker template:** your Pearl wallet `prl1...` (worker auto-appended).
5. **Pool URL:** Pascal rigs (P40 / GTX 10-series) → `pearl-cpu-eu1.luckypool.io:3370`;
   tensor-core cards (sm_80+) → `pearl-eu2.luckypool.io:3360`.
6. **Extra config arguments** (optional): `--devices 0,1`, `--region 4096`, etc.

Or install manually instead of via the URL:

```bash
cd /tmp && wget -O p40-miner-1.6.0.tar.gz \
  https://github.com/Muskwak/Open-Pearl-Miner/releases/download/v1.6.0/p40-miner-1.6.0.tar.gz
tar -C /hive/miners/custom -xzf p40-miner-1.6.0.tar.gz   # -> /hive/miners/custom/p40-miner/
```

The miner reports per-GPU TH/s and accepted shares to the HiveOS dashboard. Built against
glibc 2.31, so it runs on focal (20.04), jammy (22.04), and newer HiveOS images.
**Requires NVIDIA driver ≥ 525** (the CUDA 12 runtime) — update the rig's driver if older,
or the miner can't initialise the GPU.

## Dev Fee

A transparent **2%** dev fee is included: for 2% of cumulative mining time the miner
mines to the developer's address. This is disclosed at startup and logged on every
switch, so you can always see exactly when it is active. The 2% applies to both pool
and solo mining — in solo mode the GPU mines to the dev's pool wallet during the 2%
window; the rest of the time solo block rewards go to your node's configured address.

Thank you for supporting development!

## Tip / Donate

If this miner helped you save time or earn PRL, consider donating to the developer:

```
prl1pfu7yr6u6mfkku3mh2deyuwegcnpaunjz4vlsvaj2shg2qjkaux2q76uyud
```

Every tip is appreciated and helps fund continued development.

## License

**Pascal Pearl Miner License** — see [LICENSE](LICENSE).

- Free to use, modify, and distribute (including commercially), **provided the 2% dev fee is retained**.
- Removal or bypass of the dev fee is a license violation in any distributed or commercial deployment.
- **Personal-use exemption**: you may disable the dev fee for strictly personal, non-commercial mining on your own hardware — but the modified version must not be distributed.
