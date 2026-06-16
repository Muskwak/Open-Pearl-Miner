"""Standalone (torch-free) Pearl miner for luckypool on Pascal GPUs.

Same pipeline as luckypool_miner.py but with NO torch: device memory + kernels via
cuda_capi (ctypes -> p40cuda.dll), commitments/proofs via pearl_host (numpy +
pearl_mining). Bundles to a <100 MB binary.

    python miner_capi.py --wallet prl1... --worker p40
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import socket
import subprocess
import sys
import threading
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


def _submit_share(pool, job_id, proof, mode, accepted, log):
    """Submit one tile proof and account an accepted share to `mode`."""
    b64 = proof.to_base64()
    log(f"  submitting share ({len(b64)} B) for job {job_id}...")
    resp = pool.submit(job_id, b64)
    log(f"  POOL RESPONSE: {json.dumps(resp)[:300]}")
    if resp and resp.get("result") is True:
        accepted[mode] += 1
        tag = "DEV FEE" if mode == "dev" else "you"
        log(f"  *** SHARE ACCEPTED ({tag}) *** "
            f"you={accepted['user']} dev={accepted['dev']}")


def _proof_worker(proof_q, accepted, log):
    """Background thread: build the (expensive, ~6s CPU) Merkle proof and submit it
    while the GPU keeps mining the next grid. Each task carries its own pool, key
    and dev-fee mode so it stays correct across job/wallet changes."""
    while True:
        task = proof_q.get()
        try:
            if task is None:
                return
            pool, job_id, A, Bt, gr, gc, key, noise_rank, mode, header, verify = task
            proof = pearl_host.build_proof_bt(A, Bt, gr, gc, key, noise_rank)
            if verify:
                try:
                    v, vmsg = pearl_host.verify_proof_local(header, proof)
                    log(f"  local verify (informational): {v} ({vmsg})")
                except Exception as e:
                    log(f"  local verify error: {e}")
            _submit_share(pool, job_id, proof, mode, accepted, log)
        except Exception as e:
            log(f"  proof/submit error: {e}")
        finally:
            proof_q.task_done()


def mine_job(pool, cfg, header, target_int, job_id, region, max_regions,
             sched, bufs, accepted, proof_q, log, verify=False):
    """Mine a pool job CONTINUOUSLY: the job only fixes the key (from the header)
    and the target -- the A,B matrices are miner-chosen (random Philox seed). So we
    generate an endless stream of fresh grids on the SAME job_id, sweeping each in
    full and submitting EVERY qualifying tile, with zero idle time, until a newer
    job arrives or it is time to switch wallets for the dev fee.

    Returns ('NEWJOB', newjob) or ('SWITCH', None)."""
    factor = cfg.hash_tile_h * cfg.hash_tile_w * cfg.rounded_common_dim
    bound = min(target_int * factor, (1 << 256) - 1)
    log(f"job {job_id} target=2^{target_int.bit_length()-1} "
        f"factor={factor} bound=2^{bound.bit_length()-1}")

    key = pearl_host.derive_key(header, cfg)
    bufs.dtgt.from_host(np.frombuffer(int(bound).to_bytes(32, "little"), np.uint8).copy())

    RS = region
    tiles_per_region = (RS // 16) ** 2
    found = np.empty(1, np.int32)
    coord = np.empty(2, np.int32)

    grid = 0
    t_acct = time.time()      # wall time charged to the dev-fee scheduler so far
    while True:
        grid += 1
        # ---- fresh random grid (new commitment) for this same job ----
        bufs.dkeyjob.from_host(np.frombuffer(key, np.uint8).copy())
        cc.setup_job(bufs.dA, bufs.dB, bufs.dBt_full, bufs.dkeyjob, bufs.dnsA, bufs.dnsB,
                     M, N, K, R, time.time_ns() & 0xFFFFFFFFFFFFFFFF)
        cc.noise_gen(bufs.dEAL, bufs.dEAR, bufs.dEBL, bufs.dEBR, bufs.dnsA, bufs.dnsB, M, N, K, R)
        cc.transpose_i8(bufs.dEAR, bufs.dEAR_t, R, K, K, 0)   # [R,K]->[K,R] (Y for noise_A)
        cc.sync()

        searched = 0
        hits = 0
        search_t0 = time.time()
        last_print = search_t0
        computed: set[int] = set()  # which column blocks have Bt_ns ready this grid

        def bt_noised(c0):
            idx = c0 // RS
            d = bufs.dBpEB[idx]
            if idx not in computed:
                cc.transpose_i8(bufs.dB, bufs.dBt_tmp, K, RS, N, c0)  # B[:,c0:c0+RS].t()
                cc.noise_gemm(bufs.dEBR.offset(c0 * R), bufs.dEBL, bufs.dBt_tmp, d, RS, K, R)
                computed.add(idx)
            return d

        for r0 in range(0, M, RS):
            cc.noise_gemm(bufs.dEAL.offset(r0 * R), bufs.dEAR_t, bufs.dA.offset(r0 * K),
                          bufs.dApEA, RS, K, R)
            for c0 in range(0, N, RS):
                if max_regions and searched >= max_regions:
                    break
                newer = pool.check_newer_job(job_id)
                if newer is not None:
                    log(f"  newer job (grid {grid}, {searched} regions, {hits} hits this grid)")
                    sched.note(time.time() - t_acct)
                    return ("NEWJOB", newer)
                searched += 1
                if time.time() - last_print >= 5:
                    ths = searched * tiles_per_region / max(time.time() - search_t0, 1e-9) / 1e6
                    log(f"  grid {grid}: {searched} regions ({ths:.2f} TH/s, {hits} hits)")
                    last_print = time.time()

                dBpEB = bt_noised(c0)
                bufs.dfound.memset(0)
                # digests=None: mining only needs found/coord. pow_key MUST be dnsA.
                cc.pearl_pow_split(bufs.dApEA, dBpEB, RS, RS, K, R, bufs.dnsA, bufs.dtgt,
                                   bufs.dtb, None, bufs.dfound, bufs.dcoord, VARIANT)
                cc.sync()
                bufs.dfound.to_host(found)
                if int(found[0]) != 1:
                    continue

                bufs.dcoord.to_host(coord)
                gr, gc = r0 + int(coord[0]), c0 + int(coord[1])
                log(f"  HIT tile (row={gr}, col={gc}) grid {grid} region {searched}; proof async")
                # Snapshot the winning grid's A and B^T to host (~160 ms D2H), then
                # hand the expensive (~6s CPU) Merkle proof+submit to a worker thread
                # so the GPU keeps mining the next grid instead of idling. B^T was
                # already produced on the GPU by setup_job -> no host transpose.
                A = np.empty((M, K), np.int8); bufs.dA.to_host(A)
                Bt = np.empty((N, K), np.int8); bufs.dBt_full.to_host(Bt)
                proof_q.put((pool, job_id, A, Bt, gr, gc, key, R, sched.mode, header, verify))
                hits += 1
                # keep scanning THIS grid for more qualifying tiles (free shares)

        ths = searched * tiles_per_region / max(time.time() - search_t0, 1e-9) / 1e6
        log(f"  grid {grid} done: {hits} hits over {searched} regions ({ths:.2f} TH/s)")

        # charge elapsed wall time to the dev fee and switch wallets if it's time
        now = time.time()
        sched.note(now - t_acct)
        t_acct = now
        if sched.maybe_switch():
            return ("SWITCH", None)
        # otherwise immediately mine another fresh grid on the SAME job -- no idle


def _list_gpu_indices():
    """Enumerate GPU indices, nvidia-smi first (PCI-bus order to match CUDA), then
    fall back to the CUDA device count."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL)
        idxs = [int(x.strip()) for x in out.splitlines() if x.strip()]
        if idxs:
            return idxs
    except Exception:
        pass
    try:
        return list(range(cc.device_count()))
    except Exception:
        return []


def run_supervisor(devices, args, log):
    """Spawn one single-GPU child process per device (pinned via CUDA_VISIBLE_DEVICES),
    prefix its output with [gpuN], and print a combined-hashrate summary. This is how
    multi-GPU rigs run: independent workers, one per card, no shared CUDA state."""
    exe = sys.executable
    base = [] if getattr(sys, "frozen", False) else [os.path.abspath(sys.argv[0])]
    ths, acc = {}, {}
    lock = threading.Lock()
    # Track per-GPU rate from "grid N done" lines only (per-grid average), not the
    # mid-grid snapshots which start low each grid and read noisy.
    done_re = re.compile(r"grid \d+ done:.*\(([\d.]+) TH/s")

    def pump(dev, proc):
        for line in proc.stdout:
            line = line.rstrip("\n")
            print(f"[gpu{dev}] {line}", flush=True)
            m = done_re.search(line)
            if m:
                with lock:
                    ths[dev] = float(m.group(1))
            if "SHARE ACCEPTED" in line:
                with lock:
                    acc[dev] = acc.get(dev, 0) + 1

    children = []
    for d in devices:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(d)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["PYTHONUNBUFFERED"] = "1"
        cmd = [exe] + base + [
            "--single",
            "--wallet", args.wallet,
            "--worker", f"{args.worker}-gpu{d}",
            "--pool", args.pool,
            "--region", str(args.region),
        ]
        if args.verify:
            cmd.append("--verify")
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        children.append((d, p))
        threading.Thread(target=pump, args=(d, p), daemon=True).start()
        log(f"launched GPU {d} -> worker {args.worker}-gpu{d} (pid {p.pid})")

    try:
        while True:
            time.sleep(20)
            alive = [d for d, p in children if p.poll() is None]
            with lock:
                total = sum(ths.get(d, 0.0) for d in alive)
                tacc = sum(acc.values())
            log(f"[ALL] {len(alive)}/{len(children)} GPUs up | "
                f"combined ~{total:.2f} TH/s | accepted {tacc}")
            if not alive:
                log("all GPU workers exited")
                break
    except KeyboardInterrupt:
        log("stopping all GPU workers...")
    finally:
        for d, p in children:
            try:
                p.terminate()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallet", default=None)  # required for pool mining; unused in --solo
    ap.add_argument("--worker", default="p40")
    ap.add_argument("--pool", default="pearl-cpu-eu1.luckypool.io:3370")
    ap.add_argument("--region", type=int, default=4096)
    ap.add_argument("--max-regions", type=int, default=0)
    ap.add_argument("--max-jobs", type=int, default=0)
    ap.add_argument("--devices", default=None,
                    help="GPU selection: comma-separated indices (e.g. 0,1,2,3) or 'all'. "
                         "Default: auto-detect and run one worker per GPU.")
    ap.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--solo", default=None, metavar="HOST:PORT",
                    help="solo mining to a local pearl-gateway (JSON-RPC). Block "
                         "rewards go to the node's configured address. Includes the "
                         "same transparent 2%% dev fee as pool mining. Single-GPU.")
    ap.add_argument("--verify", action="store_true",
                    help="run the (always-False, ~3s) informational local verify per hit")
    args = ap.parse_args()

    def log(m):
        print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)

    # Solo mining to a local node's pearl-gateway (pull protocol, no pool/wallet).
    if args.solo:
        run_solo(args, log)
        return
    if not args.wallet:
        ap.error("--wallet is required (omit it only when using --solo)")

    # A supervisor-spawned child: pinned to one GPU (via CUDA_VISIBLE_DEVICES),
    # runs the mining loop directly. The --single flag prevents it re-supervising.
    if args.single:
        run_single_gpu(args, log)
        return

    # Explicit --devices: always supervise so each chosen card is correctly pinned
    # (even a single non-zero index needs the per-child CUDA_VISIBLE_DEVICES).
    if args.devices is not None:
        if args.devices.strip().lower() == "all":
            devs = _list_gpu_indices()
        else:
            devs = [int(x) for x in args.devices.split(",") if x.strip() != ""]
        if not devs:
            log("no GPUs selected via --devices")
            return
        log(f"GPUs {devs} | one worker per card | pool {args.pool}")
        run_supervisor(devs, args, log)
        return

    # No flag: auto-detect. Multiple GPUs -> one worker each; single GPU -> run here.
    devs = _list_gpu_indices()
    if len(devs) > 1:
        log(f"multi-GPU: {len(devs)} GPUs {devs} | one worker per card | pool {args.pool}")
        run_supervisor(devs, args, log)
        return
    if devs:
        log(f"single GPU detected (index {devs[0]})")
    run_single_gpu(args, log)


def run_single_gpu(args, log):
    """Mine on this process's one visible GPU (device 0, or the card pinned by the
    supervisor via CUDA_VISIBLE_DEVICES)."""
    host, port = args.pool.rsplit(":", 1)
    log(f"p40 miner (torch-free) | pool {args.pool} | region {args.region}")
    cfg = real_config()
    sched = DevFeeScheduler(DEV_FEE, args.wallet, DEV_ADDRESS, log)
    if sched.fee > 0:
        log(f"dev fee: {sched.fee * 100:.1f}% of mining time "
            f"(transparent; logged on every switch). Thank you!")
    bufs = Bufs(args.region)
    accepted = {"user": 0, "dev": 0}
    jobs = 0
    # Background proof builder/submitter: overlaps the ~6s host Merkle proof with
    # GPU mining of the next grid. Bounded so a slow CPU can't pile up 1 GB snapshots.
    proof_q: queue.Queue = queue.Queue(maxsize=3)
    threading.Thread(target=_proof_worker, args=(proof_q, accepted, log),
                     daemon=True).start()
    while True:
        try:
            pool = LuckyPool(host, int(port), sched.wallet, args.worker)
            pool.connect()
            log(f"authorized ({'DEV FEE round' if sched.mode == 'dev' else 'your wallet'}); "
                f"waiting for job...")
            job = pool.next_job()
            status = None
            while job is not None:
                header, target_int, job_id = job
                jobs += 1
                # mine_job runs CONTINUOUSLY (fresh grids, no idle) on this job and
                # only returns when a newer job arrives or a dev-fee switch is due.
                status, val = mine_job(pool, cfg, header, target_int, job_id,
                                       args.region, args.max_regions, sched, bufs,
                                       accepted, proof_q, log, verify=args.verify)
                if args.max_jobs and jobs >= args.max_jobs:
                    proof_q.join()  # finish submitting in-flight proofs before exit
                    log(f"done ({jobs} jobs; you={accepted['user']} dev={accepted['dev']}; "
                        f"realized dev fee {sched.realized_pct():.2f}%)")
                    return
                if status == "NEWJOB":
                    job = val
                    continue
                if status == "SWITCH":
                    break  # reconnect with the other wallet
                job = None
            # Drain pending proofs on THIS connection before tearing the socket down
            # (a switch reconnects with the other wallet; a timeout reconnects fresh).
            proof_q.join()
            if status != "SWITCH":
                log("no job (timeout); reconnecting")
        except (ConnectionError, OSError, socket.timeout) as e:
            log(f"connection issue: {e}; reconnecting in 5s")
            time.sleep(5)
        except KeyboardInterrupt:
            log("stopping"); return


def run_solo(args, log):
    """Solo mining against a local pearl-gateway over its JSON-RPC miner protocol
    (getMiningInfo / submitPlainProof). Same GPU pipeline and proof construction as
    the pool path -- only the wire protocol differs. The gateway assembles and
    submits the full block, so block rewards go to the node's configured address.

    A 2% dev fee applies here too, the same transparent way as the pool path: for 2%
    of cumulative mining time the GPU mines to the dev's pool wallet instead of your
    node, logged on every switch. Single-GPU.

    NOTE: untested end-to-end -- needs a running, synced Pearl node + pearl-gateway.
    """
    cfg = real_config()
    bufs = Bufs(args.region)
    sched = DevFeeScheduler(DEV_FEE, "your-node", DEV_ADDRESS, log)
    accepted = {"user": 0, "dev": 0}
    # Background proof builder/submitter, used only during dev-fee (pool) rounds.
    proof_q: queue.Queue = queue.Queue(maxsize=3)
    threading.Thread(target=_proof_worker, args=(proof_q, accepted, log),
                     daemon=True).start()

    log(f"SOLO mode | gateway {args.solo} | single-GPU")
    if sched.fee > 0:
        log(f"dev fee: {sched.fee * 100:.1f}% of mining time -- during a dev round the "
            f"GPU mines to the dev's pool wallet instead of your node (transparent; "
            f"logged on every switch). Thank you!")
    dh, dp = args.pool.rsplit(":", 1)
    try:
        while True:
            if sched.mode == "dev":
                _solo_dev_round(dh, int(dp), sched.wallet, args.worker + "-solo", cfg,
                                bufs, sched, accepted, proof_q, args.region,
                                args.max_regions, log, args.verify)
            else:
                _solo_gateway_round(args, cfg, bufs, sched, log)
    except KeyboardInterrupt:
        log("solo: stopping")
        return


def _solo_dev_round(host, port, wallet, worker, cfg, bufs, sched, accepted, proof_q,
                    region, max_regions, log, verify):
    """Dev-fee round of solo mining: mine LuckyPool with the dev wallet until the
    scheduler switches back to your node. Reuses the exact pool mining loop."""
    while sched.mode == "dev":
        try:
            pool = LuckyPool(host, port, wallet, worker)
            pool.connect()
            log(f"solo: DEV FEE round -- mining dev pool wallet "
                f"(realized {sched.realized_pct():.2f}%)")
            job = pool.next_job()
            while job is not None:
                header, target_int, job_id = job
                status, val = mine_job(pool, cfg, header, target_int, job_id, region,
                                       max_regions, sched, bufs, accepted, proof_q, log,
                                       verify=verify)
                if status == "NEWJOB":
                    job = val
                    continue
                if status == "SWITCH":
                    proof_q.join()
                    return
                job = None
            proof_q.join()
        except (ConnectionError, OSError, socket.timeout) as e:
            # Don't strand the user if the dev pool is unreachable: charge the outage
            # to the dev's owed time so the scheduler returns to your node.
            log(f"solo: dev-pool connection issue: {e}; retrying in 5s")
            sched.note(5.0)
            time.sleep(5)
            if sched.maybe_switch():
                return


def _solo_gateway_round(args, cfg, bufs, sched, log):
    """Mine the local gateway (rewards -> your node) until the dev-fee scheduler
    switches to a dev round (or forever, if the fee is 0)."""
    from gateway_client import MiningClient  # stdlib-only JSON-RPC client

    host, port = args.solo.rsplit(":", 1)
    port = int(port)
    factor = cfg.hash_tile_h * cfg.hash_tile_w * cfg.rounded_common_dim
    RS = args.region
    tiles_per_region = (RS // 16) ** 2
    found = np.empty(1, np.int32)
    coord = np.empty(2, np.int32)
    submitted = 0
    while True:
        try:
            client = MiningClient(host, port)
            log(f"solo: connected to gateway {args.solo}")
            cur_header = None
            key = None
            grid = 0
            t_acct = time.time()      # wall time not yet charged to the dev scheduler
            while True:
                # Pull the current template; re-key only when the header changes.
                job = client.get_mining_info()
                header = job.incomplete_header_bytes
                target_int = job.target
                if header != cur_header:
                    cur_header = header
                    key = pearl_host.derive_key(header, cfg)
                    bound = min(target_int * factor, (1 << 256) - 1)
                    bufs.dtgt.from_host(
                        np.frombuffer(int(bound).to_bytes(32, "little"), np.uint8).copy())
                    log(f"solo: new template target=2^{target_int.bit_length()-1} "
                        f"bound=2^{bound.bit_length()-1}")

                grid += 1
                # ---- fresh random grid (new commitment) for this template ----
                bufs.dkeyjob.from_host(np.frombuffer(key, np.uint8).copy())
                cc.setup_job(bufs.dA, bufs.dB, bufs.dBt_full, bufs.dkeyjob, bufs.dnsA,
                             bufs.dnsB, M, N, K, R, time.time_ns() & 0xFFFFFFFFFFFFFFFF)
                cc.noise_gen(bufs.dEAL, bufs.dEAR, bufs.dEBL, bufs.dEBR,
                             bufs.dnsA, bufs.dnsB, M, N, K, R)
                cc.transpose_i8(bufs.dEAR, bufs.dEAR_t, R, K, K, 0)
                cc.sync()

                searched = 0
                search_t0 = time.time()
                last_print = search_t0
                computed: set[int] = set()

                def bt_noised(c0):
                    idx = c0 // RS
                    d = bufs.dBpEB[idx]
                    if idx not in computed:
                        cc.transpose_i8(bufs.dB, bufs.dBt_tmp, K, RS, N, c0)
                        cc.noise_gemm(bufs.dEBR.offset(c0 * R), bufs.dEBL,
                                      bufs.dBt_tmp, d, RS, K, R)
                        computed.add(idx)
                    return d

                hit = False
                for r0 in range(0, M, RS):
                    cc.noise_gemm(bufs.dEAL.offset(r0 * R), bufs.dEAR_t,
                                  bufs.dA.offset(r0 * K), bufs.dApEA, RS, K, R)
                    for c0 in range(0, N, RS):
                        searched += 1
                        if time.time() - last_print >= 5:
                            ths = searched * tiles_per_region / max(time.time() - search_t0, 1e-9) / 1e6
                            log(f"  solo grid {grid}: {searched} regions ({ths:.2f} TH/s)")
                            last_print = time.time()
                        dBpEB = bt_noised(c0)
                        bufs.dfound.memset(0)
                        cc.pearl_pow_split(bufs.dApEA, dBpEB, RS, RS, K, R, bufs.dnsA,
                                           bufs.dtgt, bufs.dtb, None, bufs.dfound,
                                           bufs.dcoord, VARIANT)
                        cc.sync()
                        bufs.dfound.to_host(found)
                        if int(found[0]) != 1:
                            continue
                        bufs.dcoord.to_host(coord)
                        gr, gc = r0 + int(coord[0]), c0 + int(coord[1])
                        log(f"  solo: BLOCK CANDIDATE (row={gr}, col={gc})! building proof...")
                        A = np.empty((M, K), np.int8); bufs.dA.to_host(A)
                        Bt = np.empty((N, K), np.int8); bufs.dBt_full.to_host(Bt)
                        proof = pearl_host.build_proof_bt(A, Bt, gr, gc, key, R)
                        try:
                            client.submit_plain_proof(proof, job)
                            submitted += 1
                            log(f"  solo: *** SUBMITTED BLOCK PROOF to gateway *** (total {submitted})")
                        except Exception as e:
                            log(f"  solo: submit error: {e}")
                        hit = True
                        break
                    if hit:
                        break
                    # charge elapsed time per row-block; hand over if a dev round is due
                    now = time.time()
                    sched.note(now - t_acct)
                    t_acct = now
                    if sched.maybe_switch():
                        return
                if not hit:
                    ths = searched * tiles_per_region / max(time.time() - search_t0, 1e-9) / 1e6
                    log(f"  solo grid {grid} swept ({ths:.2f} TH/s); no candidate")
                now = time.time()
                sched.note(now - t_acct)
                t_acct = now
                if sched.maybe_switch():
                    return
        except (ConnectionError, OSError, socket.timeout) as e:
            log(f"solo: gateway connection issue: {e}; retrying in 5s")
            time.sleep(5)


if __name__ == "__main__":
    main()
