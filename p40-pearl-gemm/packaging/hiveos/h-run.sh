#!/usr/bin/env bash
# Launch p40-miner under the HiveOS agent. The agent runs this inside a screen
# session; we also tee to ${CUSTOM_LOG_BASENAME}.log so h-stats.sh can parse it.
cd "$(dirname "$0")" || exit 1
. h-manifest.conf

# Rebuild args from the current flight sheet, then load ARGS="...".
./h-config.sh
# shellcheck disable=SC1090
. "./$CUSTOM_CONFIG_FILENAME"

mkdir -p "$(dirname "$CUSTOM_LOG_BASENAME")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "[h-run] $(date '+%F %T') launching: ./bin/p40-miner $ARGS"
# Unbuffered passthrough so per-line hashrate updates reach the log promptly.
exec ./bin/p40-miner $ARGS 2>&1 | tee "${CUSTOM_LOG_BASENAME}.log"
