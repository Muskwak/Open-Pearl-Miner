#!/usr/bin/env bash
# alpha-miner wrapper for p40-alpha-miner
# Forks the original alpha-miner CLI interface but uses p40-pearl-gemm kernels
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Build if not already built
if ! python -c "import p40_pearl_gemm" 2>/dev/null; then
    echo "Building p40-pearl-gemm kernels..." >&2
    cd "$SCRIPT_DIR/p40-pearl-gemm"
    pip install -e . 2>&1 | tail -1
    cd "$SCRIPT_DIR"
fi

exec python -m p40_pearl_gemm "$@"
