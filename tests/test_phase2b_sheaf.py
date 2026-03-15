"""Tests for phase2b_sheaf.py — full sigma-sweep experiment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from atft.core.types import SheafValidationResult
from atft.experiments.phase2b_sheaf import Phase2bConfig, Phase2bExperiment


class TestPhase2bExperiment:
    """Integration tests at tiny scale."""

    def test_run_returns_validation_result(self):
        """Full run at tiny scale should complete and return SheafValidationResult."""
        config = Phase2bConfig(
            n_points=10,
            K=2,
            sigma_grid=np.array([0.3, 0.5, 0.7]),
            n_epsilon_steps=5,
            epsilon_max=3.0,
            m=3,
            zeta_data_path=Path("data/odlyzko_zeros.txt"),
        )
        experiment = Phase2bExperiment(config)
        result = experiment.run()
        assert isinstance(result, SheafValidationResult)
        assert result.betti_heatmap.shape == (3, 5)
        assert result.peak_sigma in [0.3, 0.5, 0.7]

    def test_epsilon_zero_column(self):
        """First column (epsilon=0) should always be N*K^2."""
        config = Phase2bConfig(
            n_points=8,
            K=2,
            sigma_grid=np.array([0.4, 0.5, 0.6]),
            n_epsilon_steps=5,
            epsilon_max=3.0,
            m=3,
            zeta_data_path=Path("data/odlyzko_zeros.txt"),
        )
        experiment = Phase2bExperiment(config)
        result = experiment.run()
        expected_full = 8 * 2 * 2  # N * K^2
        assert np.all(result.betti_heatmap[:, 0] == expected_full)

    def test_config_defaults(self):
        """Config should have sensible defaults."""
        config = Phase2bConfig()
        assert config.K == 20
        assert config.n_points == 1000
        assert len(config.sigma_grid) == 9
        assert config.sigma_grid[4] == 0.5  # middle value
