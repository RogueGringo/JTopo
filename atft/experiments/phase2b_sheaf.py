"""Phase 2b: Full non-abelian sigma-sweep experiment.

The core falsifiable experiment: sweep sigma across the critical strip
and test whether the sheaf Betti number peaks uniquely at sigma=1/2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from atft.core.types import SheafValidationResult
from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.sheaf_ph import SheafPH
from atft.topology.transport_maps import TransportMapBuilder


@dataclass
class Phase2bConfig:
    """Configuration for the Phase 2b sigma-sweep experiment."""

    n_points: int = 1000
    K: int = 20
    sigma_grid: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70])
    )
    n_epsilon_steps: int = 200
    epsilon_max: float = 3.0
    m: int = 20
    zeta_data_path: Path = Path("data/odlyzko_zeros.txt")
    seed: int = 42


class Phase2bExperiment:
    """Orchestrates the full sigma-sweep experiment."""

    def __init__(self, config: Phase2bConfig) -> None:
        self.config = config

    def run(self) -> SheafValidationResult:
        """Execute the sigma-sweep and return validation result."""
        cfg = self.config

        # Load and unfold zeta zeros
        print(f"Loading {cfg.n_points} zeta zeros...")
        source = ZetaZerosSource(cfg.zeta_data_path)
        cloud = source.generate(cfg.n_points)
        unfolded = SpectralUnfolding(method="zeta").transform(cloud)
        zeros = unfolded.points[:, 0]

        eps_grid = np.linspace(0, cfg.epsilon_max, cfg.n_epsilon_steps)

        # Build initial builder just to pass K to SheafPH
        builder = TransportMapBuilder(K=cfg.K, sigma=0.5)
        ph = SheafPH(builder, zeros)

        print(f"Running sigma-sweep: {len(cfg.sigma_grid)} sigma values x {cfg.n_epsilon_steps} epsilon steps")
        heatmap = ph.sigma_sweep(eps_grid, cfg.sigma_grid, m=cfg.m)

        # Find peak
        max_per_sigma = np.max(heatmap[:, 1:], axis=1)  # skip eps=0 (trivial)
        peak_idx = int(np.argmax(max_per_sigma))
        peak_sigma = float(cfg.sigma_grid[peak_idx])
        peak_kernel_dim = int(max_per_sigma[peak_idx])

        # Check uniqueness: is peak_sigma the only maximum?
        is_unique = int(np.sum(max_per_sigma == peak_kernel_dim)) == 1

        return SheafValidationResult(
            sigma_grid=cfg.sigma_grid,
            epsilon_grid=eps_grid,
            betti_heatmap=heatmap,
            peak_sigma=peak_sigma,
            peak_kernel_dim=peak_kernel_dim,
            is_unique_peak=is_unique,
            metadata={
                "n_points": cfg.n_points,
                "K": cfg.K,
                "m": cfg.m,
            },
        )
