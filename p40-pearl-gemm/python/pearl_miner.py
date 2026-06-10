"""End-to-end Pearl mining pipeline for Pascal (P40), option A: standalone.

Pipeline (per the reference miner-base):
    1. A job is just (incomplete_header_bytes, target). The miner CHOOSES A,B.
    2. key            = blake3(incomplete_header_bytes + mining_config_bytes)
    3. A/B merkle roots (tensor_hash) -> commitment hashes -> noise_seed_A/B
    4. noise E_AL,E_AR,E_BL,E_BR generated from the seeds
    5. A_noised = A + E_AL@E_AR ; B_noised = B + E_BL@E_BR
    6. pearl_pow(A_noised, B_noised^T, pow_key=noise_seed_A, target)  [Pascal CUDA]
       -> per-16x16-tile transcript + keyed BLAKE3 <= target  (a "block")
    7. on a hit, build OpenedBlockInfo and submit through the gateway.

The hot path (step 6, the noised GEMM + per-tile transcript + keyed BLAKE3) is the
validated Pascal kernel `p40_pearl_gemm_cuda.pearl_pow`. Steps 3-5 are faithful
torch/BLAKE3 ports of `miner_base.noise_generation` / `noisy_gemm` and are cheap
(rank R << k). They can be moved to the existing CUDA kernels later for speed.

NOTE on network compatibility: `mining_config_bytes` must equal
`MiningConfiguration.to_bytes()` from the Rust `pearl_mining` crate for the pool to
accept shares. This module now uses the real `MiningConfiguration` serialization
(52 bytes, via `mining_config.py`). Everything downstream of the key is protocol-exact.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import blake3
import numpy as np
import torch

import p40_pearl_gemm_cuda as _C

from gateway_client import BLAKE3_CHUNK_LEN, PlainProof, MatrixMerkleProof, MerkleProofData, pad_to_chunk_boundary
from mining_config import MiningConfiguration, PeriodicPattern, patterns_for_tile

NOISE_RANK = 128
NOISE_RANGE = 128
HASH_TILE = 16


# --------------------------------------------------------------------------- #
# Noise generation (faithful port of miner_base/noise_generation.py)
# --------------------------------------------------------------------------- #
def _mul_hi_u32(a: np.ndarray, b: np.uint32) -> np.ndarray:
    return ((a.astype(np.uint64) * np.uint64(b)) >> np.uint64(32)).astype(np.uint32)


def _random_hash(index: int, seed: bytes, key: bytes, prepend_index: int) -> bytes:
    msg = np.zeros(8, dtype=np.int32)
    msg[prepend_index] = 1 + index
    return blake3.blake3(msg.tobytes() + seed, key=key).digest()


def _uniform_matrix(seed: bytes, key: bytes, rows: int, rank: int, rng_range: int) -> torch.Tensor:
    _r = rng_range // 2
    zero_point = _r // 2
    mask = _r - 1
    cols = rank
    draws = ceil(rows * cols / 32)
    rb = b"".join(_random_hash(i, seed, key, 0) for i in range(draws))
    rt = torch.frombuffer(bytearray(rb), dtype=torch.uint8)[: rows * cols]
    return (((rt & mask).int() - zero_point).to(torch.int8)).view(rows, cols)


def _perm_matrix(seed: bytes, key: bytes, rows: int, cols: int, rank: int, assign_cols: bool) -> torch.Tensor:
    rank_mask = rank - 1
    mat = torch.zeros(rows, cols, dtype=torch.int8)
    required = cols if assign_cols else rows
    draws = ceil(required * 4 / 32)
    for i in range(draws):
        words = np.frombuffer(_random_hash(i, seed, key, 1), dtype=np.uint32)
        for kk in range(8):
            u = words[kk]
            first = int(u & np.uint32(rank_mask))
            second = int(first ^ (1 + int(_mul_hi_u32(np.array([rank - 1], np.uint32), u)[0])))
            ai = i * 8 + kk
            if ai >= required:
                break
            perm = torch.zeros(rank, dtype=torch.int8)
            perm[first] = 1
            perm[second] = -1
            if assign_cols:
                mat[:, ai] = perm
            else:
                mat[ai, :] = perm
    return mat


def generate_noise(key_A: bytes, key_B: bytes, m: int, k: int, n: int,
                   rank: int = NOISE_RANK, rng_range: int = NOISE_RANGE):
    """Returns E_AL [m,r], E_AR [r,k], E_BL [k,r], E_BR [r,n] (all int8)."""
    seed_A = b"A_tensor" + b"\x00" * 24
    seed_B = b"B_tensor" + b"\x00" * 24
    E_AL = _uniform_matrix(seed_A, key_A, m, rank, rng_range)               # [m,r]
    E_AR = _perm_matrix(seed_A, key_A, rank, k, rank, assign_cols=True)     # [r,k]
    E_BL = _perm_matrix(seed_B, key_B, k, rank, rank, assign_cols=False)    # [k,r]
    E_BR = _uniform_matrix(seed_B, key_B, n, rank, rng_range).T.contiguous()  # [r,n]
    return E_AL, E_AR, E_BL, E_BR


# --------------------------------------------------------------------------- #
# Commitment hashing (uses validated CUDA tensor_hash for the matrix Merkle roots)
# --------------------------------------------------------------------------- #
def matrix_merkle_root(matrix_int8: torch.Tensor, key: bytes) -> bytes:
    """BLAKE3-keyed Merkle root of an int8 matrix, via the validated Pascal tensor_hash.

    For matrices below tensor_hash's 2^17-byte design point this falls back to the
    flat keyed BLAKE3 (which the kernel equals for num_roots>=2 anyway)."""
    dev = matrix_int8.device
    data = matrix_int8.to(torch.uint8).contiguous()
    nbytes = data.numel()
    key_t = torch.frombuffer(bytearray(key), dtype=torch.uint8).to(dev)
    out = torch.empty(32, dtype=torch.uint8, device=dev)
    tpb = 128
    num_chunks = (nbytes + 1024 - 1) // 1024
    num_blocks = (num_chunks + tpb - 1) // tpb
    if num_blocks >= 2:
        roots = torch.empty(num_blocks * 32, dtype=torch.uint8, device=dev)
        _C.tensor_hash(data.view(-1), key_t, out, roots, tpb, 2, 512)
        torch.cuda.synchronize()
        return bytes(out.cpu().numpy().tobytes())
    # tiny-matrix fallback: plain keyed BLAKE3 of the bytes
    return blake3.blake3(bytes(matrix_int8.to(torch.uint8).cpu().numpy().tobytes()), key=key).digest()


def commitment_hashes(A: torch.Tensor, B: torch.Tensor, key: bytes) -> tuple[bytes, bytes]:
    """Returns (noise_seed_A, noise_seed_B) per miner_base/commitment_hash.py."""
    root_A = matrix_merkle_root(A, key)
    root_B = matrix_merkle_root(B.T.contiguous(), key)  # commit B^T
    commitment_B = blake3.blake3(key + root_B).digest()
    commitment_A = blake3.blake3(commitment_B + root_A).digest()
    return commitment_A, commitment_B  # (noise_seed_A, noise_seed_B)


def derive_key(incomplete_header_bytes: bytes, mining_config: MiningConfiguration) -> bytes:
    """Key = BLAKE3(header_bytes + MiningConfiguration.to_bytes())."""
    return blake3.blake3(incomplete_header_bytes + mining_config.to_bytes()).digest()


def default_mining_config(m: int, k: int, rank: int = NOISE_RANK) -> MiningConfiguration:
    rows_pattern, cols_pattern = patterns_for_tile(16, 16)  # single hash tile
    return MiningConfiguration(
        common_dim=k,
        rank=rank,
        mma_type=0,
        rows_pattern=rows_pattern,
        cols_pattern=cols_pattern,
    )


def pool_target(difficulty: int, config: MiningConfiguration) -> int:
    """Convert pool difficulty (leading-zero-bits count) to U256 target.

    The pool's difficulty is the number of leading zero bits required in the
    final jackpot hash.  This is multiplied by the difficulty-adjustment factor
    (tile_size × dot_product_length) per the Rust ``extract_difficulty_bound``,
    so the actual target checked against the hash is easier than the raw
    bit count would imply.  Result is clamped to U256::MAX.
    """
    raw = 2 ** (256 - difficulty)
    h = config.hash_tile_h()
    w = config.hash_tile_w()
    adj = h * w * config.dot_product_length()
    result = raw * adj
    u256_max = (1 << 256) - 1
    return result if result <= u256_max else u256_max


def generate_matrices(
    m: int, k: int, n: int,
    signal_min: int = -64, signal_max: int = 64,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate random int8 matrices A (m×k) and B (k×n) for mining."""
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    A = torch.randint(signal_min, signal_max + 1, (m, k), dtype=torch.int8, generator=rng)
    B = torch.randint(signal_min, signal_max + 1, (k, n), dtype=torch.int8, generator=rng)
    return A, B


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
@dataclass
class PoWResult:
    found: bool
    row: int
    col: int


def _imatmul(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Exact int matmul via fp32 (CUDA has no int matmul). Operands here are tiny
    (noise: |E_AL|<32, E_AR/E_BL are +-1 sparse, rank R), so fp32 is exact."""
    return (X.float() @ Y.float()).round().to(torch.int32)


def noised_operands(A: torch.Tensor, B: torch.Tensor, E_AL, E_AR, E_BL, E_BR):
    """A_noised [m,k] int8, B_noised [k,n] int8 (matches noisy_gemm.noise_A/noise_B)."""
    E_A = _imatmul(E_AL, E_AR).to(torch.int8)
    E_B = _imatmul(E_BL, E_BR).to(torch.int8)
    A_noised = (A.int() + E_A.int()).to(torch.int8)
    B_noised = (B.int() + E_B.int()).to(torch.int8)
    return A_noised, B_noised


def run_pow(A_noised: torch.Tensor, B_noised: torch.Tensor, noise_seed_A: bytes,
            target: int, rank: int = NOISE_RANK) -> PoWResult:
    dev = A_noised.device
    Bt = B_noised.T.contiguous()
    key_t = torch.frombuffer(bytearray(noise_seed_A), dtype=torch.uint8).to(dev)
    target_t = torch.frombuffer(bytearray(int(target).to_bytes(32, "little")), dtype=torch.uint8).to(dev)
    _, found, coord = _C.pearl_pow(A_noised.contiguous(), Bt, key_t, target_t, rank)
    torch.cuda.synchronize()
    return PoWResult(bool(int(found[0])), int(coord[0]), int(coord[1]))


def mine_once(
    incomplete_header_bytes: bytes,
    target: int,
    A: torch.Tensor,
    B: torch.Tensor,
    mining_config: MiningConfiguration | None = None,
    rank: int = NOISE_RANK,
) -> PoWResult:
    """One attempt: commit (A,B) under this header, derive noise, noised GEMM + PoW."""
    m, k = A.shape
    n = B.shape[1]
    if mining_config is None:
        mining_config = default_mining_config(m, k, rank)
    key = derive_key(incomplete_header_bytes, mining_config)
    seed_A, seed_B = commitment_hashes(A, B, key)
    E_AL, E_AR, E_BL, E_BR = generate_noise(seed_A, seed_B, m, k, n, rank)
    E_AL, E_AR = E_AL.to(A.device), E_AR.to(A.device)
    E_BL, E_BR = E_BL.to(A.device), E_BR.to(A.device)
    A_noised, B_noised = noised_operands(A, B, E_AL, E_AR, E_BL, E_BR)
    return run_pow(A_noised, B_noised, seed_A, target, rank)


def _build_merkle_proof_for_rows(
    matrix: torch.Tensor,
    row_indices: list[int],
    key: bytes,
) -> MatrixMerkleProof:
    """Build a MatrixMerkleProof for specific rows of an int8 matrix.
    
    Extracts the given rows, builds a BLAKE3-keyed Merkle tree over the
    padded chunk data, and returns the proof (matching the Rust structure
    so that bincode serialization is byte-compatible).
    """
    import blake3 as _blake3

    # Extract rows and flatten to bytes
    rows_data = matrix[row_indices].contiguous()
    flat_bytes = rows_data.to(torch.uint8).cpu().numpy().tobytes()

    padded = pad_to_chunk_boundary(flat_bytes)
    n = len(padded) // BLAKE3_CHUNK_LEN
    leaves = [padded[i * BLAKE3_CHUNK_LEN:(i + 1) * BLAKE3_CHUNK_LEN] for i in range(n)]

    leaf_hashes = [_blake3.blake3(leaf, key=key).digest() for leaf in leaves]

    if n == 1:
        proof_data = MerkleProofData(
            leaf_data=leaves,
            leaf_indices=[0],
            total_leaves=n,
            root=leaf_hashes[0],
            siblings=[],
        )
    else:
        level = leaf_hashes
        all_siblings: list[bytes] = []
        while len(level) > 1:
            next_level = []
            for i in range(0, len(level), 2):
                if i + 1 < len(level):
                    combined = _blake3.blake3(level[i] + level[i + 1], key=key).digest()
                    next_level.append(combined)
                    all_siblings.append(level[i ^ 1])
                else:
                    next_level.append(level[i])
            level = next_level
        proof_data = MerkleProofData(
            leaf_data=leaves,
            leaf_indices=list(range(n)),
            total_leaves=n,
            root=level[0],
            siblings=all_siblings,
        )

    return MatrixMerkleProof(proof=proof_data, row_indices=row_indices)


def build_proof(
    A: torch.Tensor,
    B: torch.Tensor,
    winning_row: int,
    winning_col: int,
    key: bytes,
    noise_rank: int = NOISE_RANK,
) -> PlainProof:
    """Build a PlainProof from a winning 16×16 hash tile.

    `key` is the job key = BLAKE3(header + MiningConfiguration.to_bytes())
    used during commitment. The winning tile covers A rows
    [winning_row*16, (winning_row+1)*16) and B cols [winning_col*16,
    (winning_col+1)*16).
    """
    m, k = A.shape
    n = B.shape[1]
    tile_size = 16

    row_start = winning_row * tile_size
    row_end = min(row_start + tile_size, m)
    col_start = winning_col * tile_size
    col_end = min(col_start + tile_size, n)

    a_rows = list(range(row_start, row_end))
    b_cols = list(range(col_start, col_end))

    a_proof = _build_merkle_proof_for_rows(A, a_rows, key)
    bt_proof = _build_merkle_proof_for_rows(B.T.contiguous(), b_cols, key)

    return PlainProof(
        m=m,
        n=n,
        k=k,
        noise_rank=noise_rank,
        a=a_proof,
        bt=bt_proof,
    )
