# Pascal Pearl Miner

A high-performance **Pearl (PRL)** proof-of-work miner for **NVIDIA Pascal GPUs** —
Tesla P40, GTX 1070 / 1080 (and other `sm_61`, DP4A-capable cards).

No Python, CUDA toolkit, or PyTorch required **to run** — the CUDA runtime is bundled
in pre-built releases. Just an NVIDIA driver and the standalone binary.
**Source is open** — build it yourself or grab a pre-built release. See [License](#license) for dev-fee terms.

> Mixed-GPU rigs are supported: native code is compiled for Pascal/Turing/Ampere/Ada
> (`sm_61/75/86/89`) plus a PTX fallback that JIT-loads on any newer NVIDIA card.
> Pascal is the optimized target; newer cards run functionally via the DP4A path.

## Features

- **~7.0 TH/s** sustained on a single Tesla P40.
- **Multi-GPU** — auto-detects every GPU and runs one worker per card, near-linear scaling.
- **Continuous mining** — no idle time waiting between pool jobs.
- **Background proof submission** — finding a share never stalls the search.
- **Solo mining** — mine to your local pearl-gateway node.
- **Pool mining** — built for LuckyPool's Pearl stratum (default) with more pool protocols planned.
- **HiveOS** — full HiveOS custom-miner package.

## Pre-built Releases

Grab the latest binary from the [Releases](https://github.com/Muskwak/Pascal-Pearl-Miner/releases) page:

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
| `--pool` | `pearl-cpu-eu1.luckypool.io:3370` | Stratum `host:port` |
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

Default: `pearl-cpu-eu1.luckypool.io:3370` (LuckyPool). Override with `--pool`:

```
p40-miner.exe --wallet prl1YOURWALLET --pool pearl-cpu-eu2.luckypool.io:3370
```

Currently only LuckyPool's stratum protocol is supported. More pools planned.

### Solo Mining

```
p40-miner.exe --solo GATEWAY_HOST:PORT
```

No wallet flag needed — the node's configured address receives block rewards.
The same 2% dev fee applies (GPU mines to the dev's pool wallet during the 2% window).

## HiveOS

Download `p40-miner-hiveos-1.2.1.tar.gz` and install it as a **Custom** miner:

1. **Flight Sheet → Miner = Custom.**
2. Set the **Installation URL** to the tarball, *or* `scp` it to the rig and run
   `tar -C /hive/miners/custom -xzf p40-miner-hiveos-1.2.1.tar.gz`.
3. **Wallet and worker:** your Pearl wallet `prl1...` (worker name auto-appended).
4. **Pool URL:** `pearl-cpu-eu1.luckypool.io:3370` (default LuckyPool).
5. **Extra config arguments** (optional): `--devices 0,1`, `--region 4096`, etc.

The miner reports per-GPU TH/s and accepted shares to the HiveOS dashboard.
Built against glibc 2.31, so it runs on both the *focal* (20.04) and *jammy* (22.04)
HiveOS images and newer.

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
