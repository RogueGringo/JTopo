"""Spectral unfolding feature map."""
from __future__ import annotations

import numpy as np

from atft.core.types import PointCloud, PointCloudBatch


class SpectralUnfolding:
    """Normalizes spectra to mean gap = 1.

    Methods:
      - "rank": Rank-based unfolding via empirical CDF (for GUE).
      - "zeta": Analytic smooth staircase for Riemann zeta zeros.
    """

    def __init__(self, method: str = "rank"):
        if method not in ("rank", "zeta"):
            raise ValueError(f"Unknown method: {method!r}. Use 'rank' or 'zeta'.")
        self._method = method

    def transform(self, cloud: PointCloud) -> PointCloud:
        pts = cloud.points[:, 0].copy()

        if self._method == "rank":
            unfolded = self._rank_unfold(pts)
        elif self._method == "zeta":
            unfolded = self._zeta_unfold(pts)

        return PointCloud(
            points=unfolded.reshape(-1, 1),
            metadata={**cloud.metadata, "unfolding": self._method},
        )

    def transform_batch(self, batch: PointCloudBatch) -> PointCloudBatch:
        return PointCloudBatch(
            clouds=[self.transform(c) for c in batch.clouds]
        )

    @staticmethod
    def _rank_unfold(pts: np.ndarray) -> np.ndarray:
        """Rank-based unfolding: positions become ranks 0 to N-1 (mean gap = 1)."""
        n = len(pts)
        sorted_idx = np.argsort(pts)
        ranks = np.empty(n, dtype=np.float64)
        ranks[sorted_idx] = np.arange(n, dtype=np.float64)
        return ranks

    @staticmethod
    def _zeta_unfold(gamma: np.ndarray) -> np.ndarray:
        """Unfold zeta zeros via the smooth staircase function.

        N_smooth(T) = (T/(2*pi)) * ln(T/(2*pi*e)) + 7/8
        """
        two_pi = 2.0 * np.pi
        n_smooth = (gamma / two_pi) * np.log(gamma / (two_pi * np.e)) + 7.0 / 8.0
        return n_smooth
