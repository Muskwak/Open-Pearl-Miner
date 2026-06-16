#!/usr/bin/env bash
# Assemble the HiveOS custom-miner tarball:  p40-miner-hiveos-<ver>.tar.gz
# Layout inside the tarball (HiveOS unpacks into /hive/miners/custom/):
#   p40-miner/
#     h-manifest.conf  h-config.sh  h-run.sh  h-stats.sh
#     bin/             <- the PyInstaller onedir (p40-miner + _internal/)
#
# Run from p40-pearl-gemm/ AFTER building the Linux binary (dist/p40-miner/):
#   bash packaging/hiveos/build_hiveos_package.sh
set -euo pipefail
cd "$(dirname "$0")/../.."          # -> p40-pearl-gemm/

HIVE=packaging/hiveos
DIST="${DIST:-dist/p40-miner}"      # PyInstaller onedir (override for dist_linux)
VER=$(grep -oE 'CUSTOM_VERSION=[0-9.]+' "$HIVE/h-manifest.conf" | cut -d= -f2)

[[ -x "$DIST/p40-miner" ]] || { echo "ERROR: $DIST/p40-miner not found. Run packaging/build_linux.sh first."; exit 1; }

STAGE=$(mktemp -d)
PKG="$STAGE/p40-miner"
mkdir -p "$PKG/bin"

cp "$HIVE/h-manifest.conf" "$HIVE/h-config.sh" "$HIVE/h-run.sh" "$HIVE/h-stats.sh" "$PKG/"
cp -r "$DIST/." "$PKG/bin/"

chmod +x "$PKG"/h-*.sh "$PKG/bin/p40-miner"

OUT="$(pwd)/p40-miner-hiveos-${VER}.tar.gz"
tar -C "$STAGE" -czf "$OUT" p40-miner
rm -rf "$STAGE"

echo "built $OUT"
echo "size:  $(du -h "$OUT" | cut -f1)"
echo
echo "Install on the rig: HiveOS -> Miners -> add 'Custom' miner, point its"
echo "Installation URL at this tarball (host it), or scp it to the rig and:"
echo "  tar -C /hive/miners/custom -xzf p40-miner-hiveos-${VER}.tar.gz"
