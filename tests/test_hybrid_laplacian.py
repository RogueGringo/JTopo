"""Tests for HybridSheafLaplacian (CPU transport + GPU Lanczos).

Validates:
  1. K=200 result matches known spectral sum 11.784063 (atol=0.01)
  2. K=400 completes without OOM

All heavy tests run with device="cpu" so they work without a GPU.
The CPU device also avoids VRAM budget issues during CI.
"""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from atft.topology.hybrid_sheaf_laplacian import HybridSheafLaplacian
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.zeta_zeros import ZetaZerosSource
from atft.topology.transport_maps import TransportMapBuilder

pytestmark = [
    pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed"),
    pytest.mark.skipif(not HYBRID_AVAILABLE, reason="HybridSheafLaplacian not available"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def load_zeta_zeros(n: int) -> np.ndarray:
    """Load N unfolded zeta zeros (same pipeline as the production runs)."""
    import os
    # Try relative paths common in both CI and local dev
    candidates = [
        "data/odlyzko_zeros.txt",
        "../data/odlyzko_zeros.txt",
        "tests/data/odlyzko_zeros.txt",
    ]
    for path in candidates:
        if os.path.exists(path):
            source = ZetaZerosSource(path)
            cloud = source.generate(n)
            return SpectralUnfolding(method="zeta").transform(cloud).points[:, 0]

    pytest.skip("odlyzko_zeros.txt not found — skipping production-data tests")


# ---------------------------------------------------------------------------
# Smoke test: small K, small N
# ---------------------------------------------------------------------------

class TestHybridSmoke:
    """Fast smoke tests at small K/N that run in seconds."""

    def test_construction(self):
        builder = TransportMapBuilder(K=4, sigma=0.5)
        zeros = np.linspace(0, 5, 8)
        lap = HybridSheafLaplacian(builder, zeros, device="cpu")
        assert lap.K == 4
        assert lap.N == 8

    def test_eigenvalues_shape_and_nonneg(self):
        builder = TransportMapBuilder(K=4, sigma=0.5)
        zeros = np.linspace(0, 5, 10)
        lap = HybridSheafLaplacian(builder, zeros, device="cpu")
        eigs = lap.smallest_eigenvalues(1.0, k=10)
        assert eigs.shape == (10,)
        assert np.all(eigs >= -1e-8)
        assert np.all(eigs[:-1] <= eigs[1:] + 1e-10)  # sorted

    def test_spectral_sum_nonneg(self):
        builder = TransportMapBuilder(K=4, sigma=0.5)
        zeros = np.linspace(0, 5, 10)
        lap = HybridSheafLaplacian(builder, zeros, device="cpu")
        s = lap.spectral_sum(1.0, k=10)
        assert s >= 0.0

    def test_eps_zero_returns_zeros(self):
        builder = TransportMapBuilder(K=4, sigma=0.5)
        zeros = np.linspace(0, 5, 8)
        lap = HybridSheafLaplacian(builder, zeros, device="cpu")
        eigs = lap.smallest_eigenvalues(0.0, k=5)
        npt.assert_allclose(eigs, np.zeros(5), atol=1e-14)

    def test_build_matrix_returns_none(self):
        """HybridSheafLaplacian is matrix-free — build_matrix returns None."""
        builder = TransportMapBuilder(K=4, sigma=0.5)
        zeros = np.linspace(0, 5, 8)
        lap = HybridSheafLaplacian(builder, zeros, device="cpu")
        assert lap.build_matrix(1.0) is None

    def test_matvec_output_shape(self):
        builder = TransportMapBuilder(K=4, sigma=0.5)
        zeros = np.linspace(0, 5, 8)
        lap = HybridSheafLaplacian(builder, zeros, device="cpu")
        lap._prepare(1.0)
        dim = lap._dim
        v = torch.zeros(dim, dtype=torch.cdouble)
        result = lap.matvec(v)
        assert result.shape == (dim,)

    def test_matvec_zero_input(self):
        """L @ 0 = 0 for any Laplacian."""
        builder = TransportMapBuilder(K=4, sigma=0.5)
        zeros = np.linspace(0, 5, 8)
        lap = HybridSheafLaplacian(builder, zeros, device="cpu")
        lap._prepare(1.0)
        dim = lap._dim
        v = torch.zeros(dim, dtype=torch.cdouble)
        result = lap.matvec(v)
        npt.assert_allclose(result.numpy(), np.zeros(dim), atol=1e-15)

    def test_prepare_caches_epsilon(self):
        builder = TransportMapBuilder(K=4, sigma=0.5)
        zeros = np.linspace(0, 5, 8)
        lap = HybridSheafLaplacian(builder, zeros, device="cpu")
        lap._prepare(1.0)
        eps = lap._cached_epsilon
        lap._prepare(1.0)  # second call should be instant
        assert lap._cached_epsilon == eps

    def test_different_sigma_different_result(self):
        zeros = np.linspace(0, 5, 12)
        b1 = TransportMapBuilder(K=6, sigma=0.3)
        b2 = TransportMapBuilder(K=6, sigma=0.7)
        l1 = HybridSheafLaplacian(b1, zeros, device="cpu")
        l2 = HybridSheafLaplacian(b2, zeros, device="cpu")
        s1 = l1.spectral_sum(1.0, k=8)
        s2 = l2.spectral_sum(1.0, k=8)
        assert abs(s1 - s2) > 1e-6, "Different sigma should produce different S"


# ---------------------------------------------------------------------------
# Cross-validation against MatFreeSheafLaplacian at small K
# ---------------------------------------------------------------------------

class TestHybridVsMatFree:
    """Hybrid and MatFree should produce consistent spectral sums at small K."""

    def test_matches_matfree_K10(self):
        try:
            from atft.topology.matfree_sheaf_laplacian import MatFreeSheafLaplacian
        except ImportError:
            pytest.skip("MatFreeSheafLaplacian not available")

        builder = TransportMapBuilder(K=10, sigma=0.5)
        zeros = np.linspace(0, 5, 20)
        epsilon = 0.8

        lap_hybrid = HybridSheafLaplacian(builder, zeros, device="cpu")
        lap_matfree = MatFreeSheafLaplacian(builder, zeros, device="cpu")

        k_eig = 15
        s_hybrid = lap_hybrid.spectral_sum(epsilon, k=k_eig)
        s_matfree = lap_matfree.spectral_sum(epsilon, k=k_eig)

        # Both should agree within Lanczos numerical tolerance
        npt.assert_allclose(s_hybrid, s_matfree, rtol=0.02,
                            err_msg=(f"Hybrid S={s_hybrid:.6f} vs "
                                     f"MatFree S={s_matfree:.6f}"))


# ---------------------------------------------------------------------------
# K=200 validation: must match known S=11.784063 (atol=0.01)
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestHybridK200Validation:
    """Regression test: K=200 spectral sum must match the known result."""

    KNOWN_S_ZETA_K200 = 11.784063  # from phase3d_torch_k200 run
    ATOL = 0.01

    def test_hybrid_matches_known_k200(self):
        """K=200 hybrid run should match known S=11.784063 to atol=0.01."""
        zeros = load_zeta_zeros(1000)
        builder = TransportMapBuilder(K=200, sigma=0.5)
        lap = HybridSheafLaplacian(builder, zeros, device="cpu")

        # Use k=20 eigenvalues (same as production)
        s = lap.spectral_sum(epsilon=3.0, k=20)

        print(f"\n  K=200 hybrid: S={s:.6f}, known={self.KNOWN_S_ZETA_K200:.6f}, "
              f"diff={abs(s - self.KNOWN_S_ZETA_K200):.6f} (atol={self.ATOL})")

        npt.assert_allclose(
            s, self.KNOWN_S_ZETA_K200, atol=self.ATOL,
            err_msg=(f"K=200 hybrid S={s:.6f} deviates from known "
                     f"{self.KNOWN_S_ZETA_K200:.6f} by "
                     f"{abs(s - self.KNOWN_S_ZETA_K200):.6f} > atol={self.ATOL}")
        )


# ---------------------------------------------------------------------------
# K=400 OOM guard: must complete without crashing
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestHybridK400NoOOM:
    """Verify K=400 completes without OOM on CPU (bypasses the GPU 12GB limit)."""

    def test_hybrid_handles_k400(self):
        """K=400 run should complete and return a finite spectral sum."""
        zeros = load_zeta_zeros(1000)
        builder = TransportMapBuilder(K=400, sigma=0.5)
        lap = HybridSheafLaplacian(builder, zeros, device="cpu")

        # Run with k=20 eigenvalues
        s = lap.spectral_sum(epsilon=3.0, k=20)

        print(f"\n  K=400 hybrid: S={s:.6f}")

        assert np.isfinite(s), f"K=400 spectral sum is not finite: {s}"
        assert s >= 0.0, f"K=400 spectral sum is negative: {s}"
        # Known value from phase3f is ~11.440; allow ±1% for Lanczos variance
        npt.assert_allclose(s, 11.440, atol=0.2,
                            err_msg=f"K=400 S={s:.4f} is far from expected ~11.44")
