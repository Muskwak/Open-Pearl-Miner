"""Lightweight JSON-RPC gateway client for Pearl mining.

Connects to a pearl-gateway or pool stratum bridge, fetches MiningJobs,
and submits PlainProofs on hits. No external dependencies beyond stdlib.
"""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MinerRpcConfig:
    transport: str = "tcp"
    host: str = "127.0.0.1"
    port: int = 8337
    socket_path: str | None = None


@dataclass
class MiningJob:
    incomplete_header_bytes: bytes
    target: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MiningJob:
        from base64 import b64decode
        return cls(
            incomplete_header_bytes=b64decode(data["incomplete_header_bytes"]),
            target=data["target"],
        )

    def to_dict(self) -> dict[str, Any]:
        from base64 import b64encode
        return {
            "incomplete_header_bytes": b64encode(self.incomplete_header_bytes).decode("ascii"),
            "target": self.target,
        }


class BincodeWriter:
    """Minimal bincode serializer (bincode 1.x, LE, fixed-size)."""

    def __init__(self):
        self.buf = bytearray()

    def write_u64(self, v: int):
        self.buf.extend(struct.pack("<Q", v))

    def write_u32(self, v: int):
        self.buf.extend(struct.pack("<I", v))

    def write_u16(self, v: int):
        self.buf.extend(struct.pack("<H", v))

    def write_u8(self, v: int):
        self.buf.append(v & 0xFF)

    def write_bytes(self, data: bytes):
        self.buf.extend(data)

    def write_vec_u64(self, items: list[int]):
        self.write_u64(len(items))
        for v in items:
            self.write_u64(v)

    def write_vec_bytes32(self, items: list[bytes]):
        self.write_u64(len(items))
        for item in items:
            assert len(item) == 32
            self.buf.extend(item)

    def write_vec_chunks(self, items: list[bytes]):
        self.write_u64(len(items))
        for item in items:
            self.buf.extend(item)

    def getvalue(self) -> bytes:
        return bytes(self.buf)


BLAKE3_CHUNK_LEN = 1024


def pad_to_chunk_boundary(data: bytes) -> bytes:
    """Pad data to the next BLAKE3 chunk boundary (1024 bytes)."""
    remainder = len(data) % BLAKE3_CHUNK_LEN
    if remainder:
        data += b"\x00" * (BLAKE3_CHUNK_LEN - remainder)
    return data


@dataclass
class MerkleProofData:
    leaf_data: list[bytes]
    leaf_indices: list[int]
    total_leaves: int
    root: bytes
    siblings: list[bytes]


@dataclass
class MatrixMerkleProof:
    proof: MerkleProofData | None
    row_indices: list[int]


@dataclass
class PlainProof:
    m: int
    n: int
    k: int
    noise_rank: int
    a: MatrixMerkleProof
    bt: MatrixMerkleProof

    def to_base64(self) -> str:
        import base64
        return base64.b64encode(self._bincode()).decode("ascii")

    def _bincode(self) -> bytes:
        w = BincodeWriter()
        w.write_u64(self.m)
        w.write_u64(self.n)
        w.write_u64(self.k)
        w.write_u64(self.noise_rank)
        self._write_matrix_proof(w, self.a)
        self._write_matrix_proof(w, self.bt)
        return w.getvalue()

    @staticmethod
    def _write_matrix_proof(w: BincodeWriter, mp: MatrixMerkleProof):
        if mp.proof is None:
            w.write_u64(0)
            w.write_u64(0)
            w.write_u64(0)
            w.write_bytes(b"\x00" * 32)
            w.write_u64(0)
        else:
            p = mp.proof
            w.write_u64(len(p.leaf_data))
            for leaf in p.leaf_data:
                w.write_bytes(leaf)
            w.write_u64(len(p.leaf_indices))
            for idx in p.leaf_indices:
                w.write_u64(idx)
            w.write_u64(p.total_leaves)
            assert len(p.root) == 32
            w.write_bytes(p.root)
            w.write_u64(len(p.siblings))
            for sib in p.siblings:
                assert len(sib) == 32
                w.write_bytes(sib)
        w.write_u64(len(mp.row_indices))
        for ri in mp.row_indices:
            w.write_u64(ri)


class JSONRPCClient:
    """Simple JSON-RPC v2.0 client over TCP."""

    def __init__(self, host: str, port: int):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((host, port))
        self._reader = self._sock.makefile("r", encoding="utf-8")
        self._writer = self._sock.makefile("w", encoding="utf-8")
        self._request_id = 0

    def close(self):
        self._reader.close()
        self._writer.close()
        self._sock.close()

    def call(self, method: str, params: Any = None) -> Any:
        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._request_id,
            "params": params if params is not None else {},
        }
        json_str = json.dumps(request)
        self._writer.write(json_str + "\n")
        self._writer.flush()
        line = self._reader.readline()
        if not line:
            raise ConnectionError("Connection closed by remote host")
        response = json.loads(line.strip())
        if "error" in response:
            err = response["error"]
            raise Exception(f"JSON-RPC error {err.get('code')}: {err.get('message')}")
        if response.get("id") != self._request_id:
            raise Exception(f"Response ID mismatch")
        return response.get("result")


class MiningClient:
    """Mining-specific wrapper around JSONRPCClient (pearl-gateway protocol)."""

    def __init__(self, host: str, port: int):
        self.client = JSONRPCClient(host, port)

    def get_mining_info(self) -> MiningJob:
        result = self.client.call("getMiningInfo")
        return MiningJob.from_dict(result)

    def submit_plain_proof(self, plain_proof: PlainProof, mining_job: MiningJob):
        self.client.call(
            "submitPlainProof",
            {"plain_proof": plain_proof.to_base64(), "mining_job": mining_job.to_dict()},
        )

    def close(self):
        self.client.close()


class PoolChallenge:
    """Represents a challenge received from the pool."""

    def __init__(self, seed: bytes, difficulty: int):
        self.seed = seed
        self.difficulty = difficulty

    @classmethod
    def from_json(cls, data: dict) -> PoolChallenge:
        return cls(
            seed=bytes.fromhex(data["seed"]),
            difficulty=data["difficulty"],
        )


class AlphaPoolClient:
    """Client for the AlphaPool custom Stratum protocol.

    Protocol: pool sends ``pearl.challenge`` immediately on connect and in
    response to any message.  The miner submits shares via ``mining.submit``.
    """

    def __init__(self, host: str, port: int, wallet: str, worker: str = "default"):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(30)
        self._sock.connect((host, port))
        self._reader = self._sock.makefile("r", encoding="utf-8")
        self._writer = self._sock.makefile("w", encoding="utf-8")
        self._request_id = 0
        self.wallet = wallet
        self.worker = worker

    def close(self):
        self._reader.close()
        self._writer.close()
        self._sock.close()

    def _send(self, method: str, params: Any = None) -> int:
        self._request_id += 1
        req = {"id": self._request_id, "method": method, "params": params or []}
        self._writer.write(json.dumps(req) + "\n")
        self._writer.flush()
        return self._request_id

    def _read_line(self, timeout: float = 5) -> dict | None:
        self._sock.settimeout(timeout)
        try:
            line = self._reader.readline()
            if not line:
                return None
            return json.loads(line.strip())
        except socket.timeout:
            return None

    def _drain_notifications(self, timeout: float = 0.5) -> list[dict]:
        msgs: list[dict] = []
        while True:
            msg = self._read_line(timeout)
            if msg is None:
                break
            msgs.append(msg)
        return msgs

    def recv_challenge(self, timeout: float = 5) -> PoolChallenge | None:
        msg = self._read_line(timeout)
        if msg is None:
            return None
        if msg.get("method") == "pearl.challenge":
            return PoolChallenge.from_json(msg["params"])
        return None

    def submit_share(self, proof_base64: str, seed_hex: str) -> bool | None:
        """Submit a share to the pool.

        Returns True if accepted, False if rejected, None if no response.
        """
        params = [f"{self.wallet}.{self.worker}", seed_hex, proof_base64]
        req_id = self._send("mining.submit", params)
        self._sock.settimeout(10)
        while True:
            line = self._reader.readline()
            if not line:
                return None
            resp = json.loads(line.strip())
            if resp.get("id") == req_id:
                if resp.get("result") is True:
                    return True
                return False
            # ignore notifications (new challenges etc.)

    def subscribe(self):
        self._send("mining.subscribe", ["pearl-miner/1.0.0"])

    def authorize(self, password: str = "x"):
        self._send("mining.authorize", [self.wallet, password])
