#!/usr/bin/env bash
# HOST side (WSL Ubuntu-24.04). Stages clean source onto ext4, then runs the
# Ubuntu-20.04 container to build the glibc-2.31 binary + HiveOS tarball.
# Paths are literal in this FILE so Git-Bash MSYS path-mangling can't corrupt
# them. Invoke via PowerShell:  wsl -d Ubuntu-24.04 bash <this file>
set -euo pipefail

SRC=/mnt/c/Users/ADMIN/audits
HB="$HOME/hivebuild"

echo "=== staging sources -> $HB (ext4) ==="
mkdir -p "$HB"
# A prior container build (root) leaves root-owned files under $HB (cargo target/,
# dist_linux/, etc.) that the host user can't delete. Scrub them with a throwaway
# root container before re-staging.
docker run --rm -v "$HB":/work p40-hiveos-build bash -c 'rm -rf /work/* /work/.[!.]* 2>/dev/null' || true
mkdir -p "$HB/out"
rsync -a --exclude dist --exclude dist_linux \
  --exclude /build --exclude /build_pyi --exclude /build_pyi_linux --exclude .venv \
  --exclude '*.whl' --exclude '*.pyd' --exclude '*.dll' --exclude '*.so' \
  --exclude '*.lib' --exclude '*.exp' --exclude '*.err' --exclude '*.out' \
  --exclude '*.log' --exclude .git \
  "$SRC/p40-alpha-miner/p40-pearl-gemm/" "$HB/p40/"
rsync -a --exclude target --exclude .git "$SRC/pearl-ref/" "$HB/pearl-ref/"
rsync -a "$SRC/aphrodite-engine/.deps/cutlass-src/include/" "$HB/cutlass/"
echo "staged:"; du -sh "$HB/p40" "$HB/pearl-ref" "$HB/cutlass"

echo "=== running build container ==="
docker run --rm -v "$HB":/work -w /work/p40 p40-hiveos-build \
  bash /work/p40/packaging/hiveos/_in_container_build.sh

echo "=== HOST BUILD DONE; artifacts in $HB/out ==="
ls -la "$HB/out"
