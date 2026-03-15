"""Poisson point process source (negative control baseline)."""
from __future__ import annotations

import numpy as np

from atft.core.types import PointCloud, PointCloudBatch


class PoissonSource:
    """Generates 1D point clouds from a Poisson process.

    Points are cumulative sums of i.i.d. Exp(1) gaps.
    Already unfolded by construction (mean gap = 1).
    """

    def __init__(self, seed: int = 42):
        self._seed = seed
        self._rng = np.random.default_rng(np.random.SeedSequence(seed))

    def generate(self, n_points: int, **kwargs) -> PointCloud:
        gaps = self._rng.exponential(scale=1.0, size=n_points - 1)
        positions = np.concatenate([[0.0], np.cumsum(gaps)])
        positions += self._rng.exponential(scale=1.0)
        return PointCloud(
            points=positions.reshape(-1, 1).astype(np.float64),
            metadata={"source": "poisson", "n_points": n_points, "seed": self._seed},
        )

    def generate_batch(self, n_points: int, batch_size: int, **kwargs) -> PointCloudBatch:
        clouds = [self.generate(n_points) for _ in range(batch_size)]
        return PointCloudBatch(clouds=clouds)
