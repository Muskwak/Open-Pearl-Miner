#!/usr/bin/env bash
# Audit the max GLIBC / GLIBCXX symbol version required across the frozen bundle.
# Anything > 2.31 (glibc) would fail on HiveOS focal. Run after the container build:
#   wsl -d Ubuntu-24.04 bash <this file>
set -uo pipefail
D="${1:-$HOME/hivebuild/out/p40-miner-linux-dist}"

echo "=== scanning ELF files under $D ==="
maxglibc=""
worst_file=""
while IFS= read -r f; do
  # ELF only
  head -c4 "$f" 2>/dev/null | grep -q $'\x7fELF' || continue
  g=$(objdump -T "$f" 2>/dev/null | grep -oE 'GLIBC_[0-9]+\.[0-9]+' \
        | sort -uV | tail -1)
  x=$(objdump -T "$f" 2>/dev/null | grep -oE 'GLIBCXX_[0-9]+\.[0-9]+\.[0-9]+' \
        | sort -uV | tail -1)
  if [[ -n "$g$x" ]]; then
    printf '  %-14s %-22s %s\n' "${g:-—}" "${x:-—}" "${f#$D/}"
    if [[ -n "$g" ]]; then
      top=$(printf '%s\n%s\n' "$maxglibc" "$g" | sort -uV | tail -1)
      [[ "$top" != "$maxglibc" ]] && { maxglibc="$top"; worst_file="${f#$D/}"; }
    fi
  fi
done < <(find "$D" -type f \( -name '*.so' -o -name '*.so.*' -o -name 'p40-miner' \) )

echo
echo "=== HIGHEST glibc required: ${maxglibc:-none}  (from: ${worst_file:-n/a}) ==="
echo "    HiveOS focal provides glibc 2.31, jammy 2.35."
echo "=== libp40cuda.so dynamic deps (should be only glibc + libdl/libpthread/libm; NO libstdc++/libcudart) ==="
ldd "$D/_internal/libp40cuda.so" 2>/dev/null || objdump -p "$D/_internal/libp40cuda.so" 2>/dev/null | grep NEEDED
