"""Tests for SparseSheafLaplacian (vector fibers, BSR sparse)."""
from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from numpy.typing import NDArray

from atft.topology.sparse_sheaf_laplacian import SparseSheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder


def _build_dense_vector_laplacian(
    zeros: NDArray, builder: TransportMapBuilder, epsilon: float,
    transport_mode: str = "superposition", normalize: bool = True,
) -> NDArray[np.complex128]:
    """Build explicit dense (N*K, N*K) vector-fiber sheaf Laplacian.

    Reference implementation for validating the sparse version.
    """
    sorted_zeros = np.sort(zeros.ravel())
    N = len(sorted_zeros)
    K = builder.K
    L = np.zeros((N * K, N * K), dtype=np.complex128)
    I_K = np.eye(K, dtype=np.complex128)

    for i in range(N):
        for j in range(i + 1, N):
            gap = sorted_zeros[j] - sorted_zeros[i]
            if gap > epsilon:
                break
            # Compute transport for this edge
            if transport_mode == "superposition":
                U = builder.batch_transport_superposition(
                    np.array([gap]), normalize=normalize
                )[0]
            elif transport_mode == "fe":
                U = builder.batch_transport_fe(np.array([gap]))[0]
            else:
                U = builder.batch_transport_resonant(np.array([gap]))[0]

            Uh = U.conj().T

            # Off-diagonal: L[i,j] = -U†, L[j,i] = -U
            L[i*K:(i+1)*K, j*K:(j+1)*K] = -Uh
            L[j*K:(j+1)*K, i*K:(i+1)*K] = -U

            # Diagonal: L[i,i] += U†U, L[j,j] += I
            L[i*K:(i+1)*K, i*K:(i+1)*K] += Uh @ U
            L[j*K:(j+1)*K, j*K:(j+1)*K] += I_K

    return L


class TestEdgeDiscovery:
    """Tests for build_edge_list()."""

    def test_shape(self):
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, gaps = lap.build_edge_list(1.5)
        assert i_idx.shape == j_idx.shape == gaps.shape
        assert i_idx.ndim == 1

    def test_edges_within_epsilon(self):
        """All discovered gaps should be <= epsilon."""
        zeros = np.array([0.0, 0.8, 1.5, 3.0, 3.2, 5.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, gaps = lap.build_edge_list(1.0)
        assert np.all(gaps <= 1.0 + 1e-12)
        assert np.all(gaps > 0)

    def test_edges_complete(self):
        """Should find ALL pairs within epsilon."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, gaps = lap.build_edge_list(0.5)
        # Pairs within 0.5: (0,1), (1,2), (2,3), (3,4)
        assert len(i_idx) == 4

    def test_oriented_i_less_j(self):
        """All edges should be oriented i < j."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, _ = lap.build_edge_list(2.5)
        assert np.all(i_idx < j_idx)

    def test_eps_zero_no_edges(self):
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, gaps = lap.build_edge_list(0.0)
        assert len(i_idx) == 0

    def test_large_eps_complete_graph(self):
        """Large epsilon should connect all pairs."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        i_idx, j_idx, _ = lap.build_edge_list(100.0)
        # Complete graph on 4 vertices: 4*3/2 = 6 edges
        assert len(i_idx) == 6

    def test_unsorted_input_gets_sorted(self):
        """Input zeros in any order should produce correct edges."""
        zeros_unsorted = np.array([3.0, 1.0, 0.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros_unsorted)
        i_idx, j_idx, gaps = lap.build_edge_list(1.5)
        # After sorting: [0, 1, 2, 3]. Edges within 1.5: (0,1), (1,2), (2,3)
        assert len(i_idx) == 3
        assert np.all(gaps <= 1.5)


class TestBuildMatrix:
    """Tests for build_matrix() BSR assembly."""

    def test_dense_equivalence_K3_N5(self):
        """Sparse Laplacian matches dense reference at small scale."""
        zeros = np.array([0.0, 0.8, 1.5, 2.1, 3.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros, normalize=True)
        L_sparse = lap.build_matrix(1.0)
        L_dense = _build_dense_vector_laplacian(
            zeros, builder, 1.0, "superposition", True
        )
        npt.assert_allclose(L_sparse.toarray(), L_dense, atol=1e-12)

    def test_dense_equivalence_K6_N4_unnormalized(self):
        """Unnormalized superposition matches dense reference."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.4)
        lap = SparseSheafLaplacian(builder, zeros, normalize=False)
        L_sparse = lap.build_matrix(1.5)
        L_dense = _build_dense_vector_laplacian(
            zeros, builder, 1.5, "superposition", False
        )
        npt.assert_allclose(L_sparse.toarray(), L_dense, atol=1e-12)

    def test_hermitian(self):
        """L should equal L†."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros, normalize=True)
        L = lap.build_matrix(1.5).toarray()
        npt.assert_allclose(L, L.conj().T, atol=1e-12)

    def test_positive_semi_definite(self):
        """All eigenvalues should be >= 0."""
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        L = lap.build_matrix(1.0).toarray()
        eigenvals = np.linalg.eigvalsh(L)
        assert np.all(eigenvals > -1e-10)

    def test_shape(self):
        N, K = 5, 4
        zeros = np.arange(N, dtype=np.float64)
        builder = TransportMapBuilder(K=K, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        L = lap.build_matrix(1.5)
        assert L.shape == (N * K, N * K)

    def test_eps_zero_is_zero_matrix(self):
        """No edges => zero Laplacian."""
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        L = lap.build_matrix(0.0)
        npt.assert_allclose(L.toarray(), np.zeros((9, 9)), atol=1e-14)

    def test_fe_transport_mode(self):
        """FE transport mode also produces valid Laplacian."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros, transport_mode="fe")
        L_sparse = lap.build_matrix(1.5)
        L_dense = _build_dense_vector_laplacian(
            zeros, builder, 1.5, "fe", True
        )
        npt.assert_allclose(L_sparse.toarray(), L_dense, atol=1e-12)


class TestEigensolver:
    """Tests for smallest_eigenvalues() and spectral_sum()."""

    def test_eigenvalues_sorted_nonneg(self):
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        eigs = lap.smallest_eigenvalues(1.5, k=10)
        assert len(eigs) == 10
        assert np.all(eigs[:-1] <= eigs[1:] + 1e-10)  # sorted
        assert np.all(eigs > -1e-10)  # nonneg

    def test_matches_dense_eigenvalues(self):
        """Sparse eigsh should match dense eigh for small problem."""
        zeros = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        eigs_sparse = lap.smallest_eigenvalues(1.5, k=10)
        L_dense = lap.build_matrix(1.5).toarray()
        eigs_dense = np.sort(np.linalg.eigvalsh(L_dense))[:10]
        npt.assert_allclose(eigs_sparse, eigs_dense, atol=1e-8)

    def test_eps_zero_returns_zeros(self):
        """At epsilon=0, no edges => all eigenvalues are 0."""
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        eigs = lap.smallest_eigenvalues(0.0, k=5)
        npt.assert_allclose(eigs, np.zeros(5), atol=1e-14)

    def test_spectral_sum(self):
        zeros = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        builder = TransportMapBuilder(K=6, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        eigs = lap.smallest_eigenvalues(1.0, k=10)
        s = lap.spectral_sum(1.0, k=10)
        npt.assert_allclose(s, float(np.sum(eigs)), atol=1e-14)

    def test_spectral_sum_eps_zero(self):
        zeros = np.array([0.0, 1.0, 2.0])
        builder = TransportMapBuilder(K=3, sigma=0.5)
        lap = SparseSheafLaplacian(builder, zeros)
        s = lap.spectral_sum(0.0, k=5)
        assert s == 0.0
