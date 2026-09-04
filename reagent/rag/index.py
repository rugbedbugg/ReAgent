"""A small Tanimoto-similarity index over reaction fingerprints.

Brute-force search is fine at this scale (tens of thousands of templates): the
whole matrix is a few tens of MB and one query is a single vectorized pass. The
index caches to a compressed .npz so it is built once and loaded thereafter.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FingerprintIndex:
    def __init__(self, matrix: np.ndarray, records: list[dict]):
        self.matrix = matrix  # (N, n_bits) uint8, 0/1
        self.records = records
        self._popcount = matrix.sum(axis=1).astype(np.int32)  # bits set per row

    def __len__(self) -> int:
        return len(self.records)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[dict, float]]:
        """Top-k records by Tanimoto similarity to the query fingerprint."""
        q = query.astype(np.int32)
        inter = self.matrix.astype(np.int32) @ q
        union = self._popcount + int(q.sum()) - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            tanimoto = np.where(union > 0, inter / union, 0.0)
        top = np.argsort(-tanimoto)[:k]
        return [(self.records[i], float(tanimoto[i])) for i in top]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.savez_compressed(
            path,
            matrix=np.packbits(self.matrix, axis=1),
            n_bits=np.array([self.matrix.shape[1]]),
            records=np.array([json.dumps(self.records)]),
        )

    @classmethod
    def load(cls, path: str | Path) -> FingerprintIndex:
        data = np.load(path, allow_pickle=False)
        n_bits = int(data["n_bits"][0])
        matrix = np.unpackbits(data["matrix"], axis=1)[:, :n_bits].astype(np.uint8)
        records = json.loads(str(data["records"][0]))
        return cls(matrix, records)
