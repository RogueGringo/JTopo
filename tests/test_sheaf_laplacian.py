"""Tests for sheaf_laplacian.py — matrix-free sheaf Laplacian.

Covers:
  - Dense equivalence via explicit coboundary assembly
  - Hermiticity and positive semi-definiteness
  - K=1 reduction to standard graph Laplacian
  - ε=0 degenerate case
  - LinearOperator wrapper
  - Eigenvalue computation (LOBPCG / eigsh)
  - Kernel dimension
  - Global section extraction
"""
from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from atft.topology.transport_maps import TransportMapBuilder
from atft.topology.sheaf_laplacian import SheafLaplacian


# ── Helper ──────────────────────────────────────────────────────────────────


def _build_explicit_laplacian_via_coboundary(zeros, builder, epsilon, transport_mode="global"):
    """Build L_F = delta_0^dagger delta_0 via explicit coboundary assembly.

    Only for tiny N, K — used as ground-truth in dense-equivalence tests.
    """
    N = len(zeros)
    K = builder.K
    vertex_dim = N * K * K
    edges = [
        (i, j)
        for i in range(N)
        for j in range(i + 1, N)
        if abs(zeros[j] - zeros[i]) <= epsilon
    ]
    if not edges:
        return np.zeros((vertex_dim, vertex_dim), dtype=np.complex128)
    edge_dim = len(edges) * K * K
    delta = np.zeros((edge_dim, vertex_dim), dtype=np.complex128)
    for e_idx, (i, j) in enumerate(edges):
        dg = zeros[j] - zeros[i]
        if transport_mode == "resonant":
            U = builder.transport_resonant(dg)
        else:
            U = builder.transport(dg)
        Uh = U.conj().T
        for a in range(K):
            for b in range(K):
                row = (e_idx * K + a) * K + b
                col_j = (j * K + a) * K + b
                delta[row, col_j] += 1.0
                for c in range(K):
                    for d in range(K):
                        col_i = (i * K + c) * K + d
                        delta[row, col_i] -= U[a, c] * Uh[d, b]
    return delta.conj().T @ delta


# ── Dense equivalence ──────────────────────────────────────────────────────


class TestSheafLaplacianMatvec:
    """Verify matvec matches explicit coboundary construction."""

    @pytest.mark.parametrize("transport_mode", ["global", "resonant"])
    def test_dense_equivalence_K2_N3(self, transport_mode):
        """K=2, N=3 with all edges present."""
        zeros = np.array([0.0, 0.5, 1.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros, transport_mode=transport_mode)
        epsilon = 2.0  # all pairs connected

        L_dense = _build_explicit_laplacian_via_coboundary(zeros, builder, epsilon, transport_mode)
        dim = len(zeros) * builder.K * builder.K

        rng = np.random.default_rng(42)
        for _ in range(5):
            x = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            y_matvec = lap.matvec(x, epsilon)
            y_dense = L_dense @ x
            assert_allclose(y_matvec, y_dense, atol=1e-12)

    @pytest.mark.parametrize("transport_mode", ["global", "resonant"])
    def test_dense_equivalence_K3_N4(self, transport_mode):
        """K=3, N=4 with all edges present."""
        zeros = np.array([0.0, 0.3, 0.7, 1.2])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros, transport_mode=transport_mode)
        epsilon = 5.0  # all pairs connected

        L_dense = _build_explicit_laplacian_via_coboundary(zeros, builder, epsilon, transport_mode)
        dim = len(zeros) * builder.K * builder.K

        rng = np.random.default_rng(99)
        for _ in range(5):
            x = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            y_matvec = lap.matvec(x, epsilon)
            y_dense = L_dense @ x
            assert_allclose(y_matvec, y_dense, atol=1e-12)

    @pytest.mark.parametrize("transport_mode", ["global", "resonant"])
    def test_dense_equivalence_partial_edges(self, transport_mode):
        """Only some edges present (small epsilon)."""
        zeros = np.array([0.0, 0.5, 2.0, 2.3])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros, transport_mode=transport_mode)
        epsilon = 0.6  # only (0,1) and (2,3) connected

        L_dense = _build_explicit_laplacian_via_coboundary(zeros, builder, epsilon, transport_mode)
        dim = len(zeros) * builder.K * builder.K

        rng = np.random.default_rng(7)
        for _ in range(5):
            x = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            y_matvec = lap.matvec(x, epsilon)
            y_dense = L_dense @ x
            assert_allclose(y_matvec, y_dense, atol=1e-12)


# ── Properties ──────────────────────────────────────────────────────────────


class TestSheafLaplacianProperties:
    """Hermiticity, PSD, kernel at small eps, output shape."""

    def test_hermiticity(self):
        """L_F must be Hermitian: <x, Ly> = <Lx, y>."""
        zeros = np.array([0.0, 0.5, 1.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        epsilon = 2.0

        rng = np.random.default_rng(10)
        dim = len(zeros) * builder.K * builder.K
        for _ in range(5):
            x = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            y = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            lx = lap.matvec(x, epsilon)
            ly = lap.matvec(y, epsilon)
            inner1 = np.vdot(x, ly)
            inner2 = np.vdot(lx, y)
            assert_allclose(inner1, inner2, atol=1e-12)

    def test_positive_semi_definite(self):
        """<x, Lx> >= 0 for any x."""
        zeros = np.array([0.0, 0.4, 0.9, 1.5])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        epsilon = 2.0

        rng = np.random.default_rng(20)
        dim = len(zeros) * builder.K * builder.K
        for _ in range(10):
            x = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            val = np.real(np.vdot(x, lap.matvec(x, epsilon)))
            assert val >= -1e-12, f"Got negative Rayleigh quotient: {val}"

    def test_kernel_at_small_epsilon(self):
        """At small enough epsilon, many vertices are isolated => large kernel."""
        zeros = np.array([0.0, 10.0, 20.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        epsilon = 0.1  # no edges

        dim = len(zeros) * builder.K * builder.K
        # With no edges, L = 0
        rng = np.random.default_rng(30)
        x = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        y = lap.matvec(x, epsilon)
        assert_allclose(y, np.zeros(dim), atol=1e-15)

    def test_output_shape(self):
        """matvec output must have same shape as input."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        dim = len(zeros) * builder.K * builder.K
        x = np.ones(dim, dtype=np.complex128)
        y = lap.matvec(x, 2.0)
        assert y.shape == (dim,)
        assert y.dtype == np.complex128


# ── K=1 reduces to graph Laplacian ─────────────────────────────────────────


class TestSheafLaplacianK1:
    """K=1: sheaf Laplacian should reduce to standard graph Laplacian."""

    def test_k1_matvec_equals_graph_laplacian(self):
        """K=1 => U=1 (identity). L_F should equal scalar graph Laplacian."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=1, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        epsilon = 1.5  # edges: (0,1), (1,2), (2,3)

        # Build standard graph Laplacian
        N = len(zeros)
        L_graph = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if abs(zeros[j] - zeros[i]) <= epsilon:
                    L_graph[i, i] += 1
                    L_graph[j, j] += 1
                    L_graph[i, j] -= 1
                    L_graph[j, i] -= 1

        rng = np.random.default_rng(50)
        for _ in range(5):
            x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            y_sheaf = lap.matvec(x, epsilon)
            y_graph = L_graph @ x
            assert_allclose(y_sheaf, y_graph, atol=1e-13)

    def test_k1_eigenvalues(self):
        """K=1 eigenvalues should match standard graph Laplacian eigenvalues."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=1, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        epsilon = 1.5

        # Standard graph Laplacian eigenvalues
        N = len(zeros)
        L_graph = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                if abs(zeros[j] - zeros[i]) <= epsilon:
                    L_graph[i, i] += 1
                    L_graph[j, j] += 1
                    L_graph[i, j] -= 1
                    L_graph[j, i] -= 1
        expected_eigs = np.sort(np.linalg.eigvalsh(L_graph))

        computed_eigs = lap.smallest_eigenvalues(epsilon, m=N - 1)
        # Should have same eigenvalues (possibly fewer returned)
        assert_allclose(computed_eigs[:len(expected_eigs)], expected_eigs[:len(computed_eigs)], atol=1e-10)


# ── ε=0 degenerate case ────────────────────────────────────────────────────


class TestSheafLaplacianEpsilonZero:
    """ε=0 (or negative) => no edges => zero Laplacian."""

    def test_eps_zero_matvec_returns_zero(self):
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        dim = len(zeros) * builder.K * builder.K
        x = np.ones(dim, dtype=np.complex128)
        y = lap.matvec(x, 0.0)
        assert_allclose(y, np.zeros(dim), atol=1e-15)

    def test_eps_zero_kernel_dimension(self):
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        dim = len(zeros) * builder.K * builder.K
        kdim = lap.kernel_dimension(0.0)
        assert kdim == dim

    def test_eps_negative_kernel_dimension(self):
        zeros = np.array([0.0, 1.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        dim = len(zeros) * builder.K * builder.K
        kdim = lap.kernel_dimension(-1.0)
        assert kdim == dim


# ── LinearOperator ──────────────────────────────────────────────────────────


class TestAsLinearOperator:
    """as_linear_operator should return a scipy LinearOperator."""

    def test_returns_linear_operator(self):
        from scipy.sparse.linalg import LinearOperator

        zeros = np.array([0.0, 0.5, 1.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        op = lap.as_linear_operator(epsilon=2.0)
        assert isinstance(op, LinearOperator)
        dim = len(zeros) * builder.K * builder.K
        assert op.shape == (dim, dim)

    def test_matches_direct_matvec(self):
        zeros = np.array([0.0, 0.5, 1.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        epsilon = 2.0
        op = lap.as_linear_operator(epsilon)

        rng = np.random.default_rng(60)
        dim = len(zeros) * builder.K * builder.K
        for _ in range(3):
            x = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
            y_op = op @ x
            y_direct = lap.matvec(x, epsilon)
            assert_allclose(y_op, y_direct, atol=1e-15)


# ── Eigenvalue computation ──────────────────────────────────────────────────


class TestSmallestEigenvalues:
    """Test eigenvalue computation."""

    def test_sorted_nonnegative(self):
        zeros = np.array([0.0, 0.5, 1.0, 1.5])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        eigs = lap.smallest_eigenvalues(epsilon=2.0, m=10)
        # Sorted
        assert np.all(np.diff(eigs) >= -1e-12)
        # Non-negative
        assert np.all(eigs >= -1e-10)

    def test_eigsh_solver(self):
        zeros = np.array([0.0, 0.5, 1.0, 1.5])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        eigs = lap.smallest_eigenvalues(epsilon=2.0, m=10, solver="eigsh")
        assert len(eigs) > 0
        assert np.all(eigs >= -1e-10)

    def test_auto_solver(self):
        zeros = np.array([0.0, 0.5, 1.0, 1.5])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        eigs = lap.smallest_eigenvalues(epsilon=2.0, m=10, solver="auto")
        assert len(eigs) > 0
        assert np.all(eigs >= -1e-10)

    def test_eps_zero_returns_zeros(self):
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        eigs = lap.smallest_eigenvalues(epsilon=0.0, m=5)
        assert_allclose(eigs, np.zeros(5), atol=1e-15)

    def test_no_edges_returns_zeros(self):
        zeros = np.array([0.0, 100.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        eigs = lap.smallest_eigenvalues(epsilon=0.5, m=5)
        assert_allclose(eigs, np.zeros(5), atol=1e-15)


# ── Kernel dimension ───────────────────────────────────────────────────────


class TestKernelDimension:
    """Test kernel_dimension computation."""

    def test_full_kernel_no_edges(self):
        zeros = np.array([0.0, 100.0, 200.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        dim = len(zeros) * builder.K * builder.K
        kdim = lap.kernel_dimension(epsilon=0.5)
        assert kdim == dim

    def test_kernel_shrinks_with_more_edges(self):
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        # Small epsilon => few edges => large kernel
        kdim_small = lap.kernel_dimension(epsilon=0.6)
        # Large epsilon => more edges => smaller kernel
        kdim_large = lap.kernel_dimension(epsilon=5.0)
        assert kdim_small >= kdim_large


# ── Global sections ─────────────────────────────────────────────────────────


class TestExtractGlobalSections:
    """Test extract_global_sections."""

    def test_returns_correct_shape(self):
        zeros = np.array([0.0, 0.5, 1.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        sections = lap.extract_global_sections(epsilon=0.1)  # no edges => all in kernel
        # Should be (num_sections, N, K, K) arrays
        assert sections.ndim == 4
        N = len(zeros)
        K = builder.K
        assert sections.shape[1] == N
        assert sections.shape[2] == K
        assert sections.shape[3] == K

    def test_sections_are_in_kernel(self):
        """Each global section should satisfy Lx ≈ 0."""
        zeros = np.array([0.0, 0.5, 1.0])
        builder = TransportMapBuilder(K=2, sigma=0.5)
        builder.build_generator_sum()
        lap = SheafLaplacian(builder, zeros)
        epsilon = 2.0
        sections = lap.extract_global_sections(epsilon=epsilon)
        for i in range(sections.shape[0]):
            sec = sections[i].ravel()
            y = lap.matvec(sec, epsilon)
            assert np.linalg.norm(y) < 1e-8, f"Section {i} not in kernel"
