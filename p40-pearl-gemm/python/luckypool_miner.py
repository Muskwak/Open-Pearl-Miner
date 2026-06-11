"""Live Pearl miner for luckypool on the Tesla P40.

Implements the (reverse-engineered) luckypool stratum:
  -> mining.authorize {"wallet","worker"}
  <- mining.notify {"header"(76B hex), "height", "job_id", "target"(32B hex)}
  -> mining.submit  {"wallet","worker","job_id","plain_proof":<base64>}

Mandated matmul config: m=n=131072, k=4096, rank=256, hash tile 16x16.
We commit full A[m,k] and B^T[n,k], then search a sub-region of the output with
the P40 `pearl_pow` kernel (R=256) for jackpot<=bound, build the PlainProof with
the real Rust `pearl_mining`, verify it locally, and submit.

Use the CPU stratum (pearl-cpu-eu1.luckypool.io:3370) whose starting vardiff suits
CPU-class hashrates (a naive P40 search is in that band).

    python luckypool_miner.py --wallet prl1... --worker p40 \
        --pool pearl-cpu-eu1.luckypool.io:3370 --region 4096
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from math import ceil

import blake3
import numpy as np
import torch

import pearl_mining as pm
import pearl_miner as miner

# ---- mandated config ----
M = N = 131072
K = 4096
R = 256
HT = 16
RNG_RANGE = 128
SEED_A = b"A_tensor" + b"\x00" * 24
SEED_B = b"B_tensor" + b"\x00" * 24


def real_config():
    p = pm.PeriodicPattern.from_list(list(range(HT)))
    return pm.MiningConfiguration(
        common_dim=K, rank=R, mma_type=pm.MMAType.Int7xInt7ToInt32,
        rows_pattern=p, cols_pattern=pm.PeriodicPattern.from_list(list(range(HT))),
        reserved=pm.MiningConfiguration.RESERVED,
    )


# ---- windowed noise (only generate the rows/cols we search) ----
HASHES_PER_ROW = R // 32  # 256/32 = 8


def _uniform_rows(seed: bytes, key: bytes, row_start: int, row_count: int) -> torch.Tensor:
    """Rows [row_start, row_start+row_count) of the keyed-BLAKE3 dense noise [*,R]."""
    _r = RNG_RANGE // 2
    zero_point = _r // 2
    mask = _r - 1
    j0 = row_start * HASHES_PER_ROW
    n = row_count * HASHES_PER_ROW
    rb = b"".join(miner._random_hash(j0 + j, seed, key, 0) for j in range(n))
    rt = torch.frombuffer(bytearray(rb), dtype=torch.uint8)[: row_count * R]
    return (((rt & mask).int() - zero_point).to(torch.int8)).view(row_count, R)


def _imatmul_i8(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return (X.float() @ Y.float()).round().to(torch.int32).to(torch.int8)


class LuckyPool:
    def __init__(self, host, port, wallet, worker):
        self.host, self.port, self.wallet, self.worker = host, port, wallet, worker
        self.s = None
        self.buf = b""
        self.difficulty = None  # last vardiff value seen (informational)

    def connect(self):
        self.s = socket.create_connection((self.host, self.port), timeout=30)
        self.s.settimeout(60)
        self._send("mining.authorize", {"wallet": self.wallet, "worker": self.worker})

    def _send(self, method, params, mid=1):
        self.s.sendall((json.dumps({"id": mid, "method": method, "params": params}) + "\n").encode())

    def _readline(self, timeout):
        self.s.settimeout(timeout)
        while b"\n" not in self.buf:
            try:
                d = self.s.recv(65536)
            except socket.timeout:
                return None
            if not d:
                return None
            self.buf += d
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line) if line.strip() else None

    def next_job(self, timeout=70):
        """Block until a mining.notify; returns (header_bytes, target_int, job_id)."""
        end = time.time() + timeout
        while time.time() < end:
            msg = self._readline(timeout=end - time.time())
            if msg is None:
                continue
            if msg.get("method") == "mining.notify":
                p = msg["params"]
                return bytes.fromhex(p["header"]), int(p["target"], 16), p["job_id"]
            if msg.get("method") == "mining.set_difficulty":
                self.difficulty = msg["params"]
                print(f"  [pool set_difficulty] {msg['params']}")
        return None

    def check_newer_job(self, current_job_id):
        """Non-blocking: drain the socket; return a newer (header,target,job_id) if the
        pool pushed a fresh job (so we can abandon a stale search), else None."""
        import select
        while True:
            r, _, _ = select.select([self.s], [], [], 0)
            if not r:
                break
            try:
                d = self.s.recv(65536)
            except OSError:
                break
            if not d:
                break
            self.buf += d
        newer = None
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if msg.get("method") == "mining.notify":
                p = msg["params"]
                if p["job_id"] != current_job_id:
                    newer = (bytes.fromhex(p["header"]), int(p["target"], 16), p["job_id"])
            elif msg.get("method") == "mining.set_difficulty":
                self.difficulty = msg["params"]
        return newer

    def submit(self, job_id, plain_proof_b64):
        self._send("mining.submit",
                   {"wallet": self.wallet, "worker": self.worker,
                    "job_id": job_id, "plain_proof": plain_proof_b64}, mid=99)
        for _ in range(40):
            msg = self._readline(timeout=10)
            if msg is None:
                return None
            if msg.get("id") == 99:
                return msg
            if msg.get("method") == "mining.notify":
                # stash a fresh job arriving during submit
                self.buf = (json.dumps(msg) + "\n").encode() + self.buf
                return {"pending_job": True}


def mine_job(pool, cfg, header, target_int, job_id, region, max_regions, dev, log):
    # difficulty-adjustment factor = tile_size * rounded_common_dim (extract_difficulty_bound)
    factor = cfg.hash_tile_h * cfg.hash_tile_w * cfg.rounded_common_dim
    bound = min(target_int * factor, (1 << 256) - 1)
    log(f"job {job_id} target=2^{target_int.bit_length()-1} "
        f"factor={factor} bound=2^{bound.bit_length()-1}")

    # job key = BLAKE3(header + config) — validated derivation
    key = miner.derive_key(header, cfg)

    # full operands at the mandated dims; commit via the VALIDATED path
    t0 = time.time()
    A = torch.randint(-64, 63, (M, K), dtype=torch.int8, device=dev)   # [m,k]
    B = torch.randint(-64, 63, (K, N), dtype=torch.int8, device=dev)   # [k,n]
    a_seed, b_seed = miner.commitment_hashes(A, B, key)                # commits A and B^T
    log(f"  committed A,B ({(time.time()-t0):.1f}s)")

    # GPU noise generation (all on-device; replaces the Python BLAKE3 path).
    # EAL[M,R], EAR[R,K], EBL[K,R], EBR[N,R] — bit-exact with generate_noise.
    key_t = torch.frombuffer(bytearray(a_seed), dtype=torch.uint8).to(dev)
    keyB_t = torch.frombuffer(bytearray(b_seed), dtype=torch.uint8).to(dev)
    EAL, EAR, EBL, EBR = miner._C.noise_gen(key_t, keyB_t, M, N, K, R)
    E_BLt = EBL.t().contiguous()                                                  # [R,K]

    RS = region
    tgt_t = torch.frombuffer(bytearray(int(bound).to_bytes(32, "little")), dtype=torch.uint8).to(dev)
    searched = 0
    for r0 in range(0, M, RS):
        A_ns = (A[r0:r0+RS].int() + _imatmul_i8(EAL[r0:r0+RS], EAR).int()).to(torch.int8)
        for c0 in range(0, N, RS):
            if max_regions and searched >= max_regions:
                log(f"  searched {searched} regions, no hit; next job"); return None
            newer = pool.check_newer_job(job_id)
            if newer is not None:
                log(f"  job superseded after {searched} regions; abandoning for fresh job")
                return ("NEWJOB", newer)
            searched += 1
            Bt_cols = B[:, c0:c0+RS].t().contiguous()                    # [RS,K] = B^T[cols]
            Bt_ns = (Bt_cols.int() + _imatmul_i8(EBR[c0:c0+RS], E_BLt).int()).to(torch.int8)

            # fused kernel (warp-per-tile, ~5x the naive pearl_pow); bit-exact transcript
            _, found, coord = miner._C.pearl_pow_fused(A_ns.contiguous(), Bt_ns.contiguous(), key_t, tgt_t, R, 0)
            torch.cuda.synchronize()
            if int(found[0]) != 1:
                continue

            gr, gc = r0 + int(coord[0]), c0 + int(coord[1])
            log(f"  HIT tile (row={gr}, col={gc}) after {searched} regions; building proof...")
            proof = miner.build_proof(A, B, gr, gc, key, R)             # validated full-matrix proof
            try:
                v, vmsg = miner.verify_proof_local(header, proof)
                log(f"  local verify (block diff, informational): {v} ({vmsg})")
            except Exception as e:
                log(f"  local verify error: {e}")
            return proof.to_base64()
    log("  no hit in searched regions for this job")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wallet", required=True)
    ap.add_argument("--worker", default="p40")
    ap.add_argument("--pool", default="pearl-cpu-eu1.luckypool.io:3370")
    ap.add_argument("--region", type=int, default=4096, help="sub-output search size (mult of 16)")
    ap.add_argument("--max-regions", type=int, default=0, help="cap regions searched per job (0 = full output)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-jobs", type=int, default=0, help="0 = run forever")
    args = ap.parse_args()
    host, port = args.pool.rsplit(":", 1)
    dev = torch.device(args.device)

    def log(m):
        print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)

    log(f"luckypool miner | {torch.cuda.get_device_name(dev)} | pool {args.pool} | region {args.region}")
    cfg = real_config()
    accepted = 0
    jobs = 0
    while True:
        try:
            pool = LuckyPool(host, int(port), args.wallet, args.worker)
            pool.connect()
            log("authorized; waiting for job...")
            job = pool.next_job()
            while job is not None:
                header, target_int, job_id = job
                jobs += 1
                result = mine_job(pool, cfg, header, target_int, job_id,
                                  args.region, args.max_regions, dev, log)
                if isinstance(result, tuple) and result[0] == "NEWJOB":
                    job = result[1]          # mine the fresher job immediately
                    continue
                if result:                   # base64 PlainProof for a winning tile
                    log(f"  submitting share ({len(result)} B) for job {job_id}...")
                    resp = pool.submit(job_id, result)
                    log(f"  POOL RESPONSE: {json.dumps(resp)[:400]}")
                    if resp and resp.get("result") is True:
                        accepted += 1
                        log(f"  *** SHARE ACCEPTED *** total={accepted}")
                if args.max_jobs and jobs >= args.max_jobs:
                    log(f"done ({jobs} jobs, {accepted} accepted)")
                    return
                job = pool.next_job()
            log("no job (timeout); reconnecting")
        except (ConnectionError, OSError, socket.timeout) as e:
            log(f"connection issue: {e}; reconnecting in 5s")
            time.sleep(5)
        except KeyboardInterrupt:
            log("stopping"); return


if __name__ == "__main__":
    main()
