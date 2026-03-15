"""Riemann zeta zeros source (Odlyzko dataset)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from atft.core.types import PointCloud, PointCloudBatch


class ZetaZerosSource:
    """Loads non-trivial zeta zero imaginary parts from a text file.

    Expects one zero per line. Comments (lines starting with #) are skipped.
    """

    def __init__(self, data_path: Path | str):
        self._data_path = Path(data_path)
        self._zeros: np.ndarray | None = None

    def _load(self) -> np.ndarray:
        if self._zeros is None:
            lines = []
            with open(self._data_path) as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        lines.append(float(stripped))
            self._zeros = np.array(lines, dtype=np.float64)
        return self._zeros

    def generate(self, n_points: int, **kwargs) -> PointCloud:
        all_zeros = self._load()
        if n_points > len(all_zeros):
            raise ValueError(
                f"Requested {n_points} zeros but only {len(all_zeros)} available "
                f"in {self._data_path}"
            )
        selected = all_zeros[:n_points]
        return PointCloud(
            points=selected.reshape(-1, 1),
            metadata={
                "source": "zeta_zeros",
                "n_points": n_points,
                "data_path": str(self._data_path),
            },
        )

    def generate_batch(self, n_points: int, batch_size: int, **kwargs) -> PointCloudBatch:
        cloud = self.generate(n_points)
        return PointCloudBatch(clouds=[cloud] * batch_size)
