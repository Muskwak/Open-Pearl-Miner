#!/usr/bin/env bash
# p40-alpha-miner installer
# Usage: curl -sSL https://raw.githubusercontent.com/YOUR_USER/p40-alpha-miner/main/install.sh | bash
set -euo pipefail

REPO="YOUR_USER/p40-alpha-miner"
INSTALL_DIR="${INSTALL_DIR:-$HOME/p40-alpha-miner}"
BUILD_DIR="${INSTALL_DIR}/build"

echo "==> Cloning p40-alpha-miner from ${REPO}"
git clone --recursive "https://github.com/${REPO}.git" "$INSTALL_DIR"

cd "$INSTALL_DIR"

echo "==> Installing Python dependencies"
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "==> Building Pascal-optimized CUDA kernels"
cd p40-pearl-gemm
pip install -e .

echo "==> Installing alpha-miner wrapper"
cd "$INSTALL_DIR"
chmod +x alpha-miner-wrapper.sh 2>/dev/null || true

echo
echo "==> Installed at ${INSTALL_DIR}"
echo
echo "To mine:"
echo "  cd ${INSTALL_DIR}"
echo "  python -m p40_pearl_gemm.mine --pool stratum+tcp://us2.alphapool.tech:5566 \\"
echo "    --address prl1pYOURPEARLADDRESS --worker p40-rig"
echo
echo "Or with the alpha-miner wrapper:"
echo "  ${INSTALL_DIR}/alpha-miner-wrapper.sh --pool ..."
echo
echo "Note: P40 has no Tensor Cores. The DP4A kernel runs at ~15-30 TH/s."
echo "Use --password 'x;d=4096' for static difficulty."
