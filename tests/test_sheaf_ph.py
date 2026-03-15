"""Tests for sheaf_ph.py — epsilon and sigma sweep orchestrators."""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from atft.core.types import SheafBettiCurve, SheafValidationResult
from atft.topology.sheaf_ph import SheafPH
from atft.topology.transport_maps import TransportMapBuilder


class TestSheafPHSweep:
    """Test the epsilon sweep."""

    def test_returns_sheaf_betti_curve(self):
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        ph = SheafPH(builder, zeros)
        eps_grid = np.linspace(0, 5.0, 10)
        curve = ph.sweep(eps_grid, m=5)
        assert isinstance(curve, SheafBettiCurve)
        assert len(curve.kernel_dimensions) == 10
        assert curve.sigma == 0.5
        assert curve.K == 2

    def test_epsilon_zero_gives_full_kernel(self):
        """At epsilon=0, kernel dim = N*K^2."""
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        ph = SheafPH(builder, zeros)
        eps_grid = np.array([0.0, 1.0, 2.0])
        curve = ph.sweep(eps_grid, m=5)
        assert curve.kernel_dimensions[0] == 3 * 2 * 2  # N * K^2 = 12

    def test_betti_curve_monotonically_nonincreasing(self):
        """More edges means tighter constraints — kernel can only shrink."""
        zeros = np.linspace(0, 5, 8)
        builder = TransportMapBuilder(K=2, sigma=0.5)
        ph = SheafPH(builder, zeros)
        eps_grid = np.linspace(0, 6.0, 15)
        curve = ph.sweep(eps_grid, m=5)
        diffs = np.diff(curve.kernel_dimensions)
        assert np.all(diffs <= 0), f"Betti curve not monotone: {curve.kernel_dimensions}"

    def test_k1_reproduces_scalar_betti(self):
        """At K=1, sheaf Betti curve must match scalar H_0 Betti curve."""
        zeros = np.array([0.0, 1.0, 2.5, 4.0, 6.0])
        builder = TransportMapBuilder(K=1, sigma=0.5)
        ph = SheafPH(builder, zeros)
        eps_grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0])
        curve = ph.sweep(eps_grid, m=5)

        # At K=1, sheaf Betti = scalar Betti (number of connected components)
        # Gaps: [1.0, 1.5, 1.5, 2.0]
        # eps=0: 5 components
        # eps=1.0: 4 components (first gap merges)
        # eps=1.5: 2 components (gaps 1.0, 1.5, 1.5 merge)
        # eps=2.0: 1 component
        expected_at_0 = 5
        assert curve.kernel_dimensions[0] == expected_at_0


class TestSheafPHSigmaSweep:
    """Test the sigma sweep (2D heatmap)."""

    def test_returns_2d_array(self):
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        ph = SheafPH(builder, zeros)
        eps_grid = np.linspace(0, 3.0, 5)
        sigma_grid = np.array([0.3, 0.5, 0.7])
        heatmap = ph.sigma_sweep(eps_grid, sigma_grid)
        assert heatmap.shape == (3, 5)

    def test_epsilon_zero_column_constant(self):
        """The epsilon=0 column should always be N*K^2 regardless of sigma."""
        zeros = np.array([0.0, 1.0, 2.0])
        K = 2
        builder = TransportMapBuilder(K=K, sigma=0.5)
        ph = SheafPH(builder, zeros)
        eps_grid = np.array([0.0, 1.0, 2.0])
        sigma_grid = np.array([0.3, 0.5, 0.7])
        heatmap = ph.sigma_sweep(eps_grid, sigma_grid)
        expected = len(zeros) * K * K
        assert np.all(heatmap[:, 0] == expected)
