"""Sparse sheaf Laplacian with C^K vector fibers.

Implements the Phase 3 vector-valued sheaf Laplacian using BSR sparse
matrices and shift-invert eigsh. Designed for N=10,000 scale with
K=50 fiber dimension.

Key differences from SheafLaplacian (Phase 2):
  - Vector fibers C^K instead of matrix fibers C^{K x K}
  - Explicit sparse matrix (BSR) instead of matrix-free LinearOperator
  - Coboundary: (delta s)_e = U_e s_i - s_j (left multiply, not conjugation)
  - Supports "superposition" transport mode (multi-prime phase interference)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import eigsh

from atft.topology.transport_maps import TransportMapBuilder


class SparseSheafLaplacian:
    """BSR sparse sheaf Laplacian with C^K vector fibers.

    The sheaf Laplacian L = delta_0^dagger delta_0 where
    (delta_0 s)_e = U_e s_i - s_j for oriented edge e = (i -> j).

    Block structure for each edge (i -> j) with transport U:
      L[i,i] += U^dagger U    (diagonal)
      L[j,j] += I_K           (diagonal)
      L[i,j] = -U^dagger      (off-diagonal)
      L[j,i] = -U             (off-diagonal)

    Args:
        builder: TransportMapBuilder providing K, sigma, transport methods.
        zeros: 1D array of (possibly unsorted) unfolded zeta zeros.
        transport_mode: "superposition" (default), "fe", or "resonant".
        normalize: Frobenius-normalize superposition generators (only for
            transport_mode="superposition").
    """

    def __init__(
        self,
        builder: TransportMapBuilder,
        zeros: NDArray[np.float64],
        transport_mode: str = "superposition",
        normalize: bool = True,
    ) -> None:
        self._builder = builder
        self._zeros = np.sort(zeros.ravel())
        self._N = len(self._zeros)
        self._K = builder.K
        self._transport_mode = transport_mode
        self._normalize = normalize

    @property
    def N(self) -> int:
        return self._N

    @property
    def K(self) -> int:
        return self._K

    def build_edge_list(
        self, epsilon: float
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
        """Discover all edges in the 1D Rips complex at scale epsilon.

        Uses binary search on sorted zeros for O(N log N + |E|) complexity.

        Returns:
            (i_idx, j_idx, gaps) where each is a 1D array of length |E|.
            All edges satisfy i < j and gaps[e] = zeros[j] - zeros[i] <= epsilon.
        """
        zeros = self._zeros
        N = self._N

        if epsilon <= 0 or N < 2:
            return (
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
            )

        i_parts: list[NDArray[np.int64]] = []
        j_parts: list[NDArray[np.int64]] = []

        for i in range(N - 1):
            max_val = zeros[i] + epsilon
            # Binary search for rightmost j where zeros[j] <= max_val
            j_right = int(np.searchsorted(zeros, max_val, side='right'))
            j_right = min(j_right, N)
            if i + 1 < j_right:
                js = np.arange(i + 1, j_right, dtype=np.int64)
                i_parts.append(np.full(len(js), i, dtype=np.int64))
                j_parts.append(js)

        if i_parts:
            i_idx = np.concatenate(i_parts)
            j_idx = np.concatenate(j_parts)
        else:
            i_idx = np.array([], dtype=np.int64)
            j_idx = np.array([], dtype=np.int64)

        gaps = (
            self._zeros[j_idx] - self._zeros[i_idx]
            if len(i_idx) > 0
            else np.array([], dtype=np.float64)
        )
        return i_idx, j_idx, gaps

    def _compute_transport(
        self, gaps: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """Compute transport matrices for all edges."""
        if self._transport_mode == "superposition":
            return self._builder.batch_transport_superposition(
                gaps, normalize=self._normalize
            )
        elif self._transport_mode == "fe":
            return self._builder.batch_transport_fe(gaps)
        elif self._transport_mode == "resonant":
            return self._builder.batch_transport_resonant(gaps)
        else:
            raise ValueError(
                f"Unknown transport_mode {self._transport_mode!r}. "
                "Must be 'superposition', 'fe', or 'resonant'."
            )

    def build_matrix(self, epsilon: float) -> sp.csr_matrix:
        """Assemble the N*K x N*K sparse sheaf Laplacian.

        Block structure per edge (i -> j) with transport U:
          L[i,j] = -U†,  L[j,i] = -U   (off-diagonal K x K blocks)
          L[i,i] += U†U,  L[j,j] += I   (diagonal K x K blocks)

        Returns a Hermitian PSD sparse matrix in CSR format.
        """
        N = self._N
        K = self._K
        dim = N * K

        i_idx, j_idx, gaps = self.build_edge_list(epsilon)
        M = len(i_idx)

        if M == 0:
            return sp.csr_matrix((dim, dim), dtype=np.complex128)

        # Compute all transport matrices: (M, K, K)
        U_all = self._compute_transport(gaps)
        Uh_all = np.conj(np.transpose(U_all, (0, 2, 1)))  # U†: (M, K, K)

        # --- Build diagonal blocks: (N, K, K) ---
        diag_blocks = np.zeros((N, K, K), dtype=np.complex128)
        I_K = np.eye(K, dtype=np.complex128)

        # Accumulate U†U at tail vertices (i)
        UhU = Uh_all @ U_all  # (M, K, K) batched matmul
        np.add.at(diag_blocks, i_idx, UhU)

        # Accumulate I at head vertices (j)
        head_degrees = np.bincount(j_idx, minlength=N)
        for v in range(N):
            if head_degrees[v] > 0:
                diag_blocks[v] += head_degrees[v] * I_K

        # --- Collect all blocks ---
        n_blocks = N + 2 * M
        all_rows = np.empty(n_blocks, dtype=np.int64)
        all_cols = np.empty(n_blocks, dtype=np.int64)
        all_data = np.empty((n_blocks, K, K), dtype=np.complex128)

        # Diagonal blocks
        all_rows[:N] = np.arange(N)
        all_cols[:N] = np.arange(N)
        all_data[:N] = diag_blocks

        # Off-diagonal: -U† at (i, j)
        all_rows[N:N+M] = i_idx
        all_cols[N:N+M] = j_idx
        all_data[N:N+M] = -Uh_all

        # Off-diagonal: -U at (j, i)
        all_rows[N+M:] = j_idx
        all_cols[N+M:] = i_idx
        all_data[N+M:] = -U_all

        # --- Expand blocks to element-level COO ---
        rr, cc = np.meshgrid(np.arange(K), np.arange(K), indexing='ij')
        row_exp = (all_rows[:, None, None] * K + rr[None, :, :]).ravel()
        col_exp = (all_cols[:, None, None] * K + cc[None, :, :]).ravel()
        data_exp = all_data.ravel()

        L = sp.coo_matrix((data_exp, (row_exp, col_exp)), shape=(dim, dim))
        return L.tocsr()

    def smallest_eigenvalues(
        self, epsilon: float, k: int = 100
    ) -> NDArray[np.float64]:
        """Compute the k smallest eigenvalues of the sheaf Laplacian.

        Uses shift-invert eigsh (targeting eigenvalues near 0) with
        fallback to standard eigsh if shift-invert fails.

        Args:
            epsilon: Rips complex scale parameter.
            k: Number of eigenvalues to compute.

        Returns:
            Sorted 1D array of k smallest eigenvalues (float64).
        """
        N = self._N
        K = self._K
        dim = N * K

        # Degenerate: no edges
        if epsilon <= 0:
            return np.zeros(k, dtype=np.float64)

        L = self.build_matrix(epsilon)

        # Don't request more eigenvalues than matrix dimension allows
        # eigsh requires k < dim for sparse matrices
        k_actual = min(k, dim - 2) if dim > 2 else dim

        if k_actual <= 0:
            return np.zeros(k, dtype=np.float64)

        # If matrix is small enough, use dense eigensolver
        if dim <= 500:
            eigs = np.sort(np.linalg.eigvalsh(L.toarray()).real)
            eigs = np.maximum(eigs[:k], 0.0)
            if len(eigs) < k:
                eigs = np.concatenate([eigs, np.zeros(k - len(eigs))])
            return eigs

        # Try shift-invert (targets eigenvalues near sigma)
        try:
            eigs, _ = eigsh(L, k=k_actual, sigma=1e-8, which='LM', tol=1e-6)
            eigs = np.sort(eigs.real)
            # Clamp tiny negatives from numerical noise
            eigs = np.maximum(eigs, 0.0)
        except Exception:
            # Fallback 1: LOBPCG — iterative, no LU factorization, low memory
            try:
                from scipy.sparse.linalg import lobpcg
                rng = np.random.default_rng(42)
                X0 = rng.standard_normal((dim, k_actual)) + 1j * rng.standard_normal((dim, k_actual))
                eigs_raw, _ = lobpcg(L, X0, largest=False, tol=1e-6, maxiter=500, verbosityLevel=0)
                eigs = np.sort(eigs_raw.real)
                eigs = np.maximum(eigs, 0.0)
            except Exception:
                # Fallback 2: standard eigsh targeting smallest eigenvalues
                try:
                    eigs, _ = eigsh(L, k=k_actual, which='SM', tol=1e-6)
                    eigs = np.sort(eigs.real)
                    eigs = np.maximum(eigs, 0.0)
                except Exception:
                    # Last resort: dense
                    eigs = np.sort(np.linalg.eigvalsh(L.toarray()).real)
                    eigs = np.maximum(eigs[:k], 0.0)
                    if len(eigs) < k:
                        eigs = np.concatenate([eigs, np.zeros(k - len(eigs))])
                    return eigs

        # Pad with zeros if we got fewer than k
        if len(eigs) < k:
            eigs = np.concatenate([eigs, np.zeros(k - len(eigs))])
        return eigs[:k]

    def spectral_sum(self, epsilon: float, k: int = 100) -> float:
        """Sum of the k smallest eigenvalues (primary metric).

        This is the total "constraint energy" in the near-kernel of the
        sheaf Laplacian. Lower values indicate more globally consistent
        sections (stronger topological signal).
        """
        return float(np.sum(self.smallest_eigenvalues(epsilon, k)))
