"""Throughput benchmark: fused vs split (GEMM-only + BLAKE3) at miner scale.

Reports Mtiles/s and TH/s (1 Mtile/s ~= 1.0486 TH/s). Sweeps split variants.
Run:  python tests/bench_split.py
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import p40_pearl_gemm_cuda as _C

dev = torch.device("cuda", 0)
print(f"Device: {torch.cuda.get_device_name(dev)}")

R = 256
# miner region scale: 4096x4096 output, full k=4096
M = N = int(os.environ.get("BENCH_MN", "4096"))
K = 4096
HT = 16
num_tiles = (M // HT) * (N // HT)
TH_PER_MTILE = (1 << 20) / 1e6  # 2^20 work-units per tile, /1e6 for Mtiles, /1e6 for TH

g = torch.Generator().manual_seed(7)
A = torch.randint(-110, 110, (M, K), dtype=torch.int8, generator=g).to(dev)
Bt = torch.randint(-110, 110, (N, K), dtype=torch.int8, generator=g).to(dev)
key = torch.zeros(32, dtype=torch.uint8, device=dev)
tgt = torch.zeros(32, dtype=torch.uint8, device=dev)  # hardest -> no early-out


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    mtiles = num_tiles / 1e6 / dt
    return dt * 1e3, mtiles, mtiles * TH_PER_MTILE


print(f"\nM={M} N={N} K={K} R={R}  ({num_tiles} tiles/call)\n")

ms, mt, th = bench(lambda: _C.pearl_pow_fused(A, Bt, key, tgt, R, 1))
print(f"  fused v1 (4x4 MINB2):     {ms:6.2f} ms  {mt:6.3f} Mtiles/s  {th:5.2f} TH/s")

for v, name in [(0, "S128 4x4 MINB3"), (1, "S128 4x4 MINB4"), (2, "S64 4x4 MINB4"),
                (3, "S256 4x4 MINB3"), (4, "S128 4x4 MINB2"), (5, "S64 4x4 MINB3"),
                (6, "S128 2x4 MINB4"), (7, "S128 8x2 MINB4"), (8, "S128 8x4 MINB2"),
                (9, "S128 4x8 MINB2")]:
    ms, mt, th = bench(lambda v=v: _C.pearl_pow_split(A, Bt, key, tgt, R, v))
    print(f"  split v{v} ({name:16s}): {ms:6.2f} ms  {mt:6.3f} Mtiles/s  {th:5.2f} TH/s")
