"""Standalone (torch-free) Pearl miner for luckypool on Pascal GPUs.

Same pipeline as luckypool_miner.py but with NO torch: device memory + kernels via
cuda_capi (ctypes -> p40cuda.dll), commitments/proofs via pearl_host (numpy +
pearl_mining). Bundles to a <100 MB binary.

    python miner_capi.py --wallet prl1... --worker p40
"""
from __future__ import annotations

import argparse
import json
import socket
import time

import numpy as np

import cuda_capi as cc
import pearl_host
from pool_common import (DEV_ADDRESS, DEV_FEE, K, M, N, R, DevFeeScheduler,
                         LuckyPool, real_config)

VARIANT = 1  # pearl_pow_split S=128 4x4 MINB4


class Bufs:
    """Device buffers reused across jobs (allocated once for the mandated dims)."""

    def __init__(self, region):
        RS = region
        self.dA = cc.DBuf(M * K)
        self.dB = cc.DBuf(K * N)
        self.dEAL = cc.DBuf(M * R)
        self.dEAR = cc.DBuf(R * K)
        self.dEBL = cc.DBuf(K * R)
        self.dEBR = cc.DBuf(N * R)
        self.dEAR_t = cc.DBuf(K * R)     # EAR^T = Y for the noise_A gemm
        self.dBt_full = cc.DBuf(N * K)   # B^T for the GPU commitment
        self.dkeyjob = cc.DBuf(32)       # job key (keys the commitment + proof)
        self.dnsA = cc.DBuf(32)          # noise_seed_A (== commitment_A, the pow_key)
        self.dnsB = cc.DBuf(32)          # noise_seed_B
        self.dtgt = cc.DBuf(32)
        self.dApEA = cc.DBuf(RS * K)
        self.dBt_tmp = cc.DBuf(RS * K)
        ntiles = (RS // 16) ** 2
        self.dtb = cc.DBuf(ntiles * 16 * 4)   # reusable transcript buffer
        self.dfound = cc.DBuf(4)
        self.dcoord = cc.DBuf(8)
        # Persistent per-column Bt_ns buffers (one per column block), reused every
        # job — recomputed per job but never re-malloc'd.
        self.dBpEB = [cc.DBuf(RS * K) for _ in range(N // RS)]


def mine_job(pool, cfg, header, target_int, job_id, region, max_regions, sched, bufs, log):
    factor = cfg.hash_tile_h * cfg.hash_tile_w * cfg.rounded_common_dim
    bound = min(target_int * factor, (1 << 256) - 1)
    log(f"job {job_id} target=2^{target_int.bit_length()-1} "
        f"factor={factor} bound=2^{bound.bit_length()-1}")

    key = pearl_host.derive_key(header, cfg)

    t0 = time.time()
    # All-GPU setup: fill A,B (Philox) + keyed commitment -> noise seeds, ~0.1s.
    bufs.dkeyjob.from_host(np.frombuffer(key, np.uint8).copy())
    cc.setup_job(bufs.dA, bufs.dB, bufs.dBt_full, bufs.dkeyjob, bufs.dnsA, bufs.dnsB,
                 M, N, K, R, time.time_ns() & 0xFFFFFFFFFFFFFFFF)
    cc.noise_gen(bufs.dEAL, bufs.dEAR, bufs.dEBL, bufs.dEBR, bufs.dnsA, bufs.dnsB, M, N, K, R)
    cc.transpose_i8(bufs.dEAR, bufs.dEAR_t, R, K, K, 0)   # [R,K]->[K,R] (Y for noise_A)
    bufs.dtgt.from_host(np.frombuffer(int(bound).to_bytes(32, "little"), np.uint8).copy())
    cc.sync()
    log(f"  committed A,B + noise ({time.time()-t0:.1f}s)")

    RS = region
    tiles_per_region = (RS // 16) ** 2
    searched = 0
    search_t0 = time.time()
    last_print = search_t0
    found = np.empty(1, np.int32)
    coord = np.empty(2, np.int32)
    computed: set[int] = set()  # which column blocks have Bt_ns ready this job

    def bt_noised(c0):
        idx = c0 // RS
        d = bufs.dBpEB[idx]
        if idx not in computed:
            cc.transpose_i8(bufs.dB, bufs.dBt_tmp, K, RS, N, c0)  # B[:,c0:c0+RS].t()
            # Bt_ns = clamp(Bt + EBR @ EBL^T over R)
            cc.noise_gemm(bufs.dEBR.offset(c0 * R), bufs.dEBL, bufs.dBt_tmp, d, RS, K, R)
            computed.add(idx)
        return d

    if True:
        for r0 in range(0, M, RS):
            # A_ns = clamp(A + EAL @ EAR_t^T over R)
            cc.noise_gemm(bufs.dEAL.offset(r0 * R), bufs.dEAR_t, bufs.dA.offset(r0 * K),
                          bufs.dApEA, RS, K, R)
            for c0 in range(0, N, RS):
                if max_regions and searched >= max_regions:
                    ths = searched * tiles_per_region / max(time.time() - search_t0, 1e-9) / 1e6
                    log(f"  {searched} regions ({ths:.2f} TH/s); no hit, next job")
                    return None
                newer = pool.check_newer_job(job_id)
                if newer is not None:
                    log(f"  job superseded after {searched} regions; abandoning")
                    return ("NEWJOB", newer)
                searched += 1
                if time.time() - last_print >= 5:
                    ths = searched * tiles_per_region / max(time.time() - search_t0, 1e-9) / 1e6
                    log(f"  {searched} regions searched ({ths:.2f} TH/s)")
                    last_print = time.time()

                dBpEB = bt_noised(c0)
                bufs.dfound.memset(0)
                # digests=None: mining only needs found/coord (skips the per-tile
                # digest write); transcript is the reusable buffer (no per-region malloc).
                # pow_key MUST be noise_seed_A (dnsA), not the job key.
                cc.pearl_pow_split(bufs.dApEA, dBpEB, RS, RS, K, R, bufs.dnsA, bufs.dtgt,
                                   bufs.dtb, None, bufs.dfound, bufs.dcoord, VARIANT)
                cc.sync()
                bufs.dfound.to_host(found)
                if int(found[0]) != 1:
                    continue

                bufs.dcoord.to_host(coord)
                gr, gc = r0 + int(coord[0]), c0 + int(coord[1])
                ths = searched * tiles_per_region / max(time.time() - search_t0, 1e-9) / 1e6
                log(f"  HIT tile (row={gr}, col={gc}) after {searched} regions "
                    f"({ths:.2f} TH/s); building proof...")
                # A,B live on the GPU; copy them to host only on a hit (rare).
                A = np.empty((M, K), np.int8); bufs.dA.to_host(A)
                B = np.empty((K, N), np.int8); bufs.dB.to_host(B)
                proof = pearl_host.build_proof(A, B, gr, gc, key, R)
                try:
                    v, vmsg = pearl_host.verify_proof_local(header, proof)
                    log(f"  local verify (informational): {v} ({vmsg})")
                except Exception as e:
                    log(f"  local verify error: {e}")
                return proof.to_base64()
        ths = searched * tiles_per_region / max(time.time() - search_t0, 1e-9) / 1e6
        log(f"  {searched} regions searched ({ths:.2f} TH/s); no hit in this job")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallet", required=True)
    ap.add_argument("--worker", default="p40")
    ap.add_argument("--pool", default="pearl-cpu-eu1.luckypool.io:3370")
    ap.add_argument("--region", type=int, default=4096)
    ap.add_argument("--max-regions", type=int, default=0)
    ap.add_argument("--max-jobs", type=int, default=0)
    args = ap.parse_args()
    host, port = args.pool.rsplit(":", 1)

    def log(m):
        print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)

    log(f"p40 miner (torch-free) | pool {args.pool} | region {args.region}")
    cfg = real_config()
    sched = DevFeeScheduler(DEV_FEE, args.wallet, DEV_ADDRESS, log)
    if sched.fee > 0:
        log(f"dev fee: {sched.fee * 100:.1f}% of mining time -> dev address "
            f"{DEV_ADDRESS[:14]}...{DEV_ADDRESS[-6:]} (transparent; see README). Thank you!")
    bufs = Bufs(args.region)
    accepted = {"user": 0, "dev": 0}
    jobs = 0
    while True:
        try:
            pool = LuckyPool(host, int(port), sched.wallet, args.worker)
            pool.connect()
            log(f"authorized ({'DEV FEE round' if sched.mode == 'dev' else 'your wallet'}); "
                f"waiting for job...")
            job = pool.next_job()
            switching = False
            while job is not None:
                header, target_int, job_id = job
                jobs += 1
                t0 = time.time()
                result = mine_job(pool, cfg, header, target_int, job_id,
                                  args.region, args.max_regions, sched, bufs, log)
                sched.note(time.time() - t0)
                if isinstance(result, tuple) and result[0] == "NEWJOB":
                    job = result[1]
                    continue
                if result:
                    log(f"  submitting share ({len(result)} B) for job {job_id}...")
                    resp = pool.submit(job_id, result)
                    log(f"  POOL RESPONSE: {json.dumps(resp)[:400]}")
                    if resp and resp.get("result") is True:
                        accepted[sched.mode] += 1
                        tag = "DEV FEE" if sched.mode == "dev" else "you"
                        log(f"  *** SHARE ACCEPTED ({tag}) *** "
                            f"you={accepted['user']} dev={accepted['dev']}")
                if args.max_jobs and jobs >= args.max_jobs:
                    log(f"done ({jobs} jobs; you={accepted['user']} dev={accepted['dev']}; "
                        f"realized dev fee {sched.realized_pct():.2f}%)")
                    return
                if sched.maybe_switch():
                    switching = True
                    break
                job = pool.next_job()
            if not switching:
                log("no job (timeout); reconnecting")
        except (ConnectionError, OSError, socket.timeout) as e:
            log(f"connection issue: {e}; reconnecting in 5s")
            time.sleep(5)
        except KeyboardInterrupt:
            log("stopping"); return


if __name__ == "__main__":
    main()
