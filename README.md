# p40-alpha-miner

**Pascal P40-optimized Pearl (PRL) miner** — 7.5 TH/s on a single P40.

Mines Pearl using INT8 DP4A instructions on Pascal GPUs (Tesla P40, GTX 1070, GTX 1080, etc.) via luckypool.io stratum.

## Performance

| GPU | Hashrate | Config | Bottleneck |
|---|---|---|---|
| Tesla P40 | **7.5 TH/s** | v1 (S=128, 4×4, MINB4) | 32% of 23.5 TOPS dp4a peak |
| GTX 1070 | ~4 TH/s (est.) | same kernel | 15 SM vs P40's 30 |

## Quick start

```bash
# Install deps
pip install py-pearl-mining torch blake3 numpy

# Build CUDA extension
cd p40-pearl-gemm
CUTLASS_DIR=/path/to/cutlass/include pip install -e .

# Mine
p40-mine --wallet prl1YOURWALLET --worker p40
```

On Windows:

```
run_luckypool.bat --wallet prl1YOURWALLET --worker p40
```

## Standalone binary (no Python/torch needed)

A self-contained **~74 MB** binary (torch-free, ~6.9 TH/s sustained) is built with:

```
packaging\build_windows.bat       # Windows -> dist\p40-miner\
bash packaging/build_linux.sh     # Linux   -> dist/p40-miner/   (build on Linux)
```

It bundles its own Python, the 655 KB CUDA library, and `cudart` — users only
need an NVIDIA driver. Share the whole `dist/p40-miner/` folder. (The Python
entry point is `python/miner_capi.py`; the legacy torch miner is
`python/luckypool_miner.py`.)

## Dev fee

This miner includes a **2% dev fee**: 2% of cumulative mining time mines to the
developer's address to fund continued development. This is standard for public
miners (T-Rex 1%, lolMiner 0.7%, TeamRedMiner 0.75–2.5%, Gminer 1–3%).

It is **fully transparent**:
- The rate and dev address are printed at startup.
- Every dev-fee round is logged (`[dev fee] mining to the dev address …`), and
  accepted shares are tagged `(DEV FEE)` vs `(you)`.
- The fee is realized in short contiguous rounds (~30 s+) so it converges to 2%
  over a session; very short sessions (< ~25 min) pay proportionally less.

The rate and address are the `DEV_FEE` / `DEV_ADDRESS` constants at the top of
`p40-pearl-gemm/python/luckypool_miner.py` — inspect or change them as you wish.

### Required dependencies

- CUDA Toolkit 12.x
- PyTorch 2.x (matching CUDA)
- py-pearl-mining (Rust proof builder: `pip install py-pearl-mining`)
- CUTLASS headers (set `CUTLASS_DIR` or place at `~/.cache/cutlass/include`)

## Architecture

Split pipeline: **GEMM-only kernel** → transcript buffer → **BLAKE3-only kernel**.

Decouples the shared-memory-bound GEMM step (S=128 staging width for 4 blocks/SM = 100% warp occupancy) from the compute-bound BLAKE3 step. Bit-exact with the fused kernel at +37% throughput.

## Supported GPUs

- Tesla P40 (GP102) — 30 SM, 24 GB, TCC or WDDM
- Tesla P4 (GP104) — 20 SM, 8 GB
- GTX 1070/1080/1080 Ti — Pascal consumer cards
- **Not supported**: Volta+, Maxwell, AMD

## License

MIT (custom CUDA kernels). py-pearl-mining and alpha-miner upstream components under their respective licenses.
