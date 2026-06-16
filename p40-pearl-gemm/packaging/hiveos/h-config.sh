#!/usr/bin/env bash
# Translate the HiveOS flight-sheet fields into p40-miner CLI args.
# Inputs (set by the HiveOS agent):
#   CUSTOM_URL          pool host:port (may carry a stratum+tcp:// scheme)
#   CUSTOM_TEMPLATE     wallet template, usually  prl1wallet.%WORKER%  (already
#                       expanded to e.g.  prl1wallet.rig01)
#   CUSTOM_PASS         pool password (unused by LuckyPool; ignored)
#   CUSTOM_USER_CONFIG  extra raw args, e.g.  --devices 0,1  --region 4096
# Output: writes  ARGS="..."  to $CUSTOM_CONFIG_FILENAME.

cd "$(dirname "$0")" || exit 1
. h-manifest.conf

# Strip any URL scheme (stratum+tcp://, tcp://, http://) -> host:port
POOL=$(echo "$CUSTOM_URL" | sed -E 's~^[a-zA-Z0-9+]+://~~' | sed -E 's~/+$~~')

# Split  wallet.worker  on the LAST dot (Pearl addresses contain no dots).
WALLET="${CUSTOM_TEMPLATE%.*}"
WORKER="${CUSTOM_TEMPLATE##*.}"
if [[ -z "$WALLET" || "$WALLET" == "$WORKER" ]]; then
  # No dot in the template -> whole thing is the wallet; name the worker by rig.
  WALLET="$CUSTOM_TEMPLATE"
  WORKER="${WORKER_NAME:-hive}"
fi

ARGS="--wallet $WALLET --worker $WORKER --pool $POOL"

# Append any extra user flags verbatim (--devices, --region, --solo, ...).
if [[ -n "$CUSTOM_USER_CONFIG" ]]; then
  ARGS="$ARGS $CUSTOM_USER_CONFIG"
fi

echo "ARGS=\"$ARGS\"" > "$CUSTOM_CONFIG_FILENAME"
echo "[h-config] wrote $CUSTOM_CONFIG_FILENAME: $ARGS"
