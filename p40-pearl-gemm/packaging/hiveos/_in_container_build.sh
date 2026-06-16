#!/usr/bin/env bash
# Runs INSIDE the Ubuntu-20.04 build container. Expects a staging tree at /work:
#   /work/p40/        <- p40-pearl-gemm source (clean copy)
#   /work/pearl-ref/  <- pearl-ref (py-pearl-mining + deps, no target/)
#   /work/cutlass/    <- cutlass include dir (header-only)
#   /work/out/        <- artifacts land here (the HiveOS tarball + dist)
# No GPU is required to build; the kernels are compiled, not run.
set -euo pipefail

P40=/work/p40
OUT=/work/out
mkdir -p "$OUT"

echo "=== [1/4] multi-arch libp40cuda.so (sm_61 SASS + 75/86/89 + PTX fallback) ==="
cd "$P40"
export CUTLASS_DIR=/work/cutlass
# gcc-10 host compiler (understands -std=c++20 for CuTe); statically fold in
# libstdc++/libgcc so the .so needs only glibc + the NVIDIA driver on the rig.
export EXTRA_NVCC_FLAGS="-ccbin g++-10 -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc"
export GENCODE="-gencode arch=compute_61,code=sm_61 \
-gencode arch=compute_75,code=sm_75 \
-gencode arch=compute_86,code=sm_86 \
-gencode arch=compute_89,code=sm_89 \
-gencode arch=compute_61,code=compute_61"
rm -f libp40cuda.so
bash packaging/build_capi.sh
ls -la libp40cuda.so

echo "=== [2/4] pearl_mining wheel (maturin, abi3-py312) ==="
cd /work/pearl-ref/py-pearl-mining
rm -rf /work/wheels && mkdir -p /work/wheels
maturin build --release --out /work/wheels
WHEEL=$(ls /work/wheels/py_pearl_mining-*.whl | head -1)
echo "wheel: $WHEEL"
uv pip install --python /opt/venv/bin/python --reinstall --no-deps "$WHEEL"

echo "=== [3/4] freeze with PyInstaller ==="
cd "$P40"
rm -rf dist_linux build_pyi_linux
/opt/venv/bin/pyinstaller packaging/p40-miner-lite.spec --noconfirm \
    --distpath dist_linux --workpath build_pyi_linux
test -x dist_linux/p40-miner/p40-miner

echo "=== [4/4] assemble HiveOS tarball ==="
cd "$P40"
DIST=dist_linux/p40-miner bash packaging/hiveos/build_hiveos_package.sh
cp -v p40-miner-hiveos-*.tar.gz "$OUT"/
cp -v "$WHEEL" "$OUT"/
cp -rv dist_linux/p40-miner "$OUT"/p40-miner-linux-dist
# world-readable so the host user can read root-owned outputs
chmod -R a+rX "$OUT"
echo "=== BUILD COMPLETE; artifacts in /work/out ==="
ls -la "$OUT"
