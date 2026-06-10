"""Pool miner loop for AlphaPool.

Usage:
    uv run python pool_miner.py \\
        --wallet prl1... \\
        --worker my_rig \\
        --pool eu2.alphapool.tech:5566
"""
from __future__ import annotations

import argparse
import logging
import time

import torch

from gateway_client import AlphaPoolClient
from mining_config import MiningConfiguration
from pearl_miner import (
    NOISE_RANK,
    build_proof,
    default_mining_config,
    derive_key,
    generate_matrices,
    mine_once,
    pool_target,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

M = 1024
K = 1024
N = 1024


def mining_round(
    client: AlphaPoolClient,
    config: MiningConfiguration,
    device: torch.device,
) -> int:
    """Receive a challenge, mine one round, submit any found shares."""
    challenge = client.recv_challenge(timeout=15)
    if challenge is None:
        log.warning("No challenge received")
        return 0

    seed = challenge.seed
    log.info(
        "Challenge seed=%s... difficulty=%s",
        seed.hex()[:16], challenge.difficulty,
    )

    target = pool_target(challenge.difficulty, config)
    log.info("Target (LE U256) = %s", hex(target)[:20])

    A, B = generate_matrices(M, K, N).to(device)

    result = mine_once(seed, target, A, B, config, rank=NOISE_RANK)
    if result.found:
        log.info("HIT at tile (%s, %s)! Building proof...", result.row, result.col)
        key = derive_key(seed, config)
        proof = build_proof(A, B, result.row, result.col, key, NOISE_RANK)
        proof_b64 = proof.to_base64()
        log.info("Proof base64: %s...", proof_b64[:48])

        accepted = client.submit_share(proof_b64, seed.hex())
        if accepted is True:
            log.info("Share ACCEPTED")
            return 1
        elif accepted is False:
            log.warning("Share REJECTED")
        else:
            log.warning("No response from pool (share may be invalid or pool dropped)")
    else:
        log.debug("No hit in this round")

    return 0


def main():
    parser = argparse.ArgumentParser(description="AlphaPool Pearl miner")
    parser.add_argument("--wallet", required=True, help="PRL wallet address")
    parser.add_argument("--worker", default="default", help="Worker name")
    parser.add_argument("--pool", default="eu2.alphapool.tech:5566", help="Pool host:port")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    args = parser.parse_args()

    host, port_str = args.pool.rsplit(":", 1)
    port = int(port_str)

    device = torch.device(args.device)
    log.info("device=%s", device)
    if device.type == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(device))

    config = default_mining_config(M, K, NOISE_RANK)
    log.info("MiningConfiguration: k=%s rank=%s", config.common_dim, config.rank)

    shares_found = 0
    while True:
        try:
            client = AlphaPoolClient(host, port, args.wallet, args.worker)
            log.info("Connected to %s:%s", host, port)

            while True:
                shares_found += mining_round(client, config, device)
                if shares_found > 0:
                    log.info("Total shares: %s", shares_found)

        except (ConnectionError, TimeoutError, OSError) as exc:
            log.warning("Connection lost: %s. Reconnecting in 5s...", exc)
            time.sleep(5)
        except KeyboardInterrupt:
            log.info("Shutting down")
            break
        except Exception:
            log.exception("Unexpected error, reconnecting in 10s...")
            time.sleep(10)


if __name__ == "__main__":
    main()
