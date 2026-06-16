# p40-miner on HiveOS

A HiveOS *custom miner* package for **p40-miner** (Pearl / PRL, NVIDIA Pascal —
Tesla P40, GTX 10xx). Runs as a normal flight-sheet miner: HiveOS supplies the
wallet/pool, this package launches the miner and reports hashrate + accepted
shares back to the dashboard.

## What's in the package

```
p40-miner/
  h-manifest.conf   name/version + log path
  h-config.sh       flight-sheet fields  ->  p40-miner CLI args
  h-run.sh          launches ./bin/p40-miner and tees the log
  h-stats.sh        parses TH/s + accepted shares  ->  HiveOS JSON
  bin/              the Linux binary (p40-miner + _internal/)
```

The binary is the torch-free onedir build from `packaging/build_linux.sh`. The
CUDA kernels are compiled with `sm_61` SASS **plus a `compute_61` PTX fallback**,
so the `.so` also loads (via driver JIT) on newer NVIDIA cards in a mixed rig —
Pascal runs native, newer cards run the same DP4A path (functional, not
tensor-core-optimal).

## Build the package

On a Linux box with the CUDA toolkit (or in WSL2):

```bash
cd p40-pearl-gemm
bash packaging/build_linux.sh                 # -> dist/p40-miner/
bash packaging/hiveos/build_hiveos_package.sh # -> p40-miner-hiveos-<ver>.tar.gz
```

## Install on the rig

**Option A — local file (scp):**
```bash
scp p40-miner-hiveos-<ver>.tar.gz user@rig:/tmp/
ssh user@rig 'tar -C /hive/miners/custom -xzf /tmp/p40-miner-hiveos-<ver>.tar.gz'
```

**Option B — Installation URL:** host the tarball somewhere HTTP-reachable and
paste its URL into the flight sheet's *Custom miner → Installation URL* field;
the agent downloads + unpacks it automatically.

## Flight sheet setup

Create a flight sheet with **Miner = Custom** and:

| Field                 | Value                                                      |
|-----------------------|------------------------------------------------------------|
| Miner name            | `p40-miner`                                                |
| Installation URL      | (Option B only) the tarball URL                            |
| Wallet and worker     | your Pearl wallet, e.g. `prl1...`  (worker auto-appended)  |
| Pool URL              | `pearl-cpu-eu1.luckypool.io:3370` (default LuckyPool)      |
| Pass                  | `x` (unused by LuckyPool)                                  |
| Extra config arguments| optional, e.g. `--devices 0,1` or `--region 4096`          |

- **Multi-GPU is automatic** — every detected GPU gets its own pinned worker
  (`<worker>-gpu0`, `-gpu1`, …). Use `--devices 0,2` to pick specific cards.
- **Other pools:** set Pool URL to any `host:port`. LuckyPool is the only fully
  supported protocol today; more pools land in a later update.
- **Solo / local node:** put `--solo NODE_HOST:PORT` in Extra config arguments
  (leave the wallet blank — the node assembles the block). The same transparent 2%
  dev fee as pool mining applies in solo, disclosed at startup.

## Hashrate reporting

HiveOS shows per-GPU TH/s (the miner's `grid … done: … (X.XX TH/s)` lines) and
the accepted-share count (`SHARE ACCEPTED`). A 2% dev fee is included and
disclosed at miner startup (see the main project README).
