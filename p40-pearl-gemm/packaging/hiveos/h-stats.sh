#!/usr/bin/env bash
# Report p40-miner speed + shares to the HiveOS agent.
# The agent sources this script and reads two variables: $khs and $stats.
#   khs   : total speed in kH/s
#   stats : JSON {hs[], hs_units, temp[], fan[], uptime, ar[acc,rej], algo, bus_numbers[]}
#
# The miner prints, per completed grid:
#   single-GPU :  "<ts>   grid N done: H hits over R regions (X.XX TH/s)"
#   multi-GPU  :  "[gpuD] <ts>   grid N done: ... (X.XX TH/s)"
# and "  *** SHARE ACCEPTED" on each accepted share.

cd "$(dirname "$0")" || exit 1
. h-manifest.conf
LOG="${CUSTOM_LOG_BASENAME}.log"

khs=0
stats=""
[[ -f "$LOG" ]] || { echo "khs=0"; exit 0; }

# Per-GPU NVIDIA inventory (index -> bus number / temp / fan), PCI-bus order to
# match the miner's CUDA_DEVICE_ORDER=PCI_BUS_ID indexing.
mapfile -t SMI < <(nvidia-smi --query-gpu=index,pci.bus_id,temperature.gpu,fan.speed \
                   --format=csv,noheader,nounits 2>/dev/null)

hs="[]"; temp="[]"; fan="[]"; bus="[]"
total_hs=0
have_gpu_prefix=$(grep -c '\[gpu' "$LOG")

for row in "${SMI[@]}"; do
  idx=$(echo "$row"  | awk -F',' '{gsub(/ /,"",$1);print $1}')
  busid=$(echo "$row"| awk -F',' '{gsub(/ /,"",$2);print $2}')
  t=$(echo "$row"    | awk -F',' '{gsub(/ /,"",$3);print $3}')
  f=$(echo "$row"    | awk -F',' '{gsub(/ /,"",$4);print $4}')
  # PCI bus id "00000000:01:00.0" -> decimal bus number (the "01" field).
  bn=$(echo "$busid" | awk -F':' '{print $2}')
  bn=$((16#${bn:-0}))
  [[ "$t" =~ ^[0-9]+$ ]] || t=0
  [[ "$f" =~ ^[0-9]+$ ]] || f=0

  # Latest TH/s for this GPU.
  if [[ "$have_gpu_prefix" -gt 0 ]]; then
    line=$(grep "\[gpu${idx}\]" "$LOG" | grep -oE '[0-9]+\.[0-9]+ TH/s' | tail -1)
  else
    line=$(grep -E 'grid [0-9]+ done:' "$LOG" | grep -oE '[0-9]+\.[0-9]+ TH/s' | tail -1)
  fi
  ths=$(echo "$line" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  [[ -n "$ths" ]] || ths=0
  ghs=$(awk -v t="$ths" 'BEGIN{printf "%.0f", t*1000000000000}')   # TH/s -> H/s
  total_hs=$(awk -v a="$total_hs" -v b="$ghs" 'BEGIN{printf "%.0f", a+b}')

  hs=$(echo "$hs"     | jq -c ". + [$ghs]")
  temp=$(echo "$temp" | jq -c ". + [$t]")
  fan=$(echo "$fan"   | jq -c ". + [$f]")
  bus=$(echo "$bus"   | jq -c ". + [$bn]")
done

khs=$(awk -v h="$total_hs" 'BEGIN{printf "%.0f", h/1000}')          # H/s -> kH/s

acc=$(grep -c 'SHARE ACCEPTED' "$LOG")
rej=$(grep -ciE 'reject|stale' "$LOG")

# Uptime from the longest-running miner process.
pid=$(pgrep -f 'bin/p40-miner' | head -1)
uptime=0
[[ -n "$pid" ]] && uptime=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ')
[[ "$uptime" =~ ^[0-9]+$ ]] || uptime=0

stats=$(jq -nc \
  --argjson hs   "$hs" \
  --argjson temp "$temp" \
  --argjson fan  "$fan" \
  --argjson bus  "$bus" \
  --argjson up   "$uptime" \
  --argjson acc  "${acc:-0}" \
  --argjson rej  "${rej:-0}" \
  '{hs: $hs, hs_units: "hs", temp: $temp, fan: $fan, uptime: $up,
    ar: [$acc, $rej], algo: "pearl", bus_numbers: $bus}')

echo "khs=$khs"
