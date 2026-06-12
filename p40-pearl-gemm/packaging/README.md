# Building shareable binaries (closed source)

These scripts freeze the miner into a self-contained **onedir bundle** with
PyInstaller — torch, the CUDA extension, `pearl_mining`, blake3 and numpy are all
included, and only compiled `.pyc` (not `.py`) ships, so you can distribute
without source.

## Windows (.exe)

From the `p40-pearl-gemm/` directory:

```bat
packaging\build_windows.bat
```

Output: `dist\p40-miner\` (zip and share the whole folder). Users run:

```bat
p40-miner.exe --wallet prl1THEIRWALLET --worker rig1
```

## Linux (binary)

PyInstaller **does not cross-compile**, so the Linux binary must be built **on a
Linux machine with an NVIDIA (Pascal) GPU**. The torch-free build only needs the
CUDA toolkit (nvcc) + a few small Python packages — NOT PyTorch.

```bash
# 1. Prerequisites (Ubuntu/Debian example):
sudo apt-get install -y nvidia-cuda-toolkit            # provides nvcc + cudart
pip install numpy blake3 py-pearl-mining pyinstaller   # small pure deps + Rust proof builder

# 2. CUTLASS headers (header-only; clone once):
git clone --depth 1 https://github.com/NVIDIA/cutlass
export CUTLASS_DIR=$PWD/cutlass/include

# 3. Build (from p40-pearl-gemm/):
bash packaging/build_linux.sh
```

This compiles `libp40cuda.so` (`packaging/build_capi.sh`) then freezes the miner
into `dist/p40-miner/`. Users run `./p40-miner --wallet prl1... --worker rig1`
(needs only an NVIDIA driver). If `nvcc` isn't found, install the matching CUDA
toolkit from NVIDIA and ensure `nvcc` and `cudart` are on PATH.

## Notes

- **Size**: the bundle is large (~3–5 GB) because it embeds CUDA PyTorch. That's
  unavoidable with a torch-based miner. Distribute as a compressed archive.
- **Driver**: end users still need an NVIDIA driver installed (the CUDA *runtime*
  is bundled; the kernel driver is not).
- **GPU**: the extension is compiled for `sm_61` only (Pascal). Build with
  `ADDITIONAL_ARCHS=sm_70,sm_75,...` before freezing for broader GPU support.
- **Dev fee**: the 2% dev fee (`DEV_FEE`/`DEV_ADDRESS` in `luckypool_miner.py`) is
  compiled into the binary. Because the source isn't shipped, users can't trivially
  remove it. For stronger protection you can Cython-compile `luckypool_miner.py`
  before freezing, but the onedir `.pyc` bundle is the standard approach.
- **Python version**: the bundle is tied to the Python that built it (cp313 here).
