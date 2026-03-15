"""Matrix-free sheaf Laplacian for ATFT Phase 2.

Implements L_F = delta_0^dagger delta_0 without assembling the full matrix.
Uses the TransportMapBuilder's O(K^2) transport shortcut for edge-local
computations, yielding an efficient matvec for iterative eigensolvers.

The sheaf assigns a K x K matrix fiber to each vertex. Given an edge (i,j)
with transport U_{ij}, the coboundary difference is:

    (delta_0 x)_{ij} = x_j - U_{ij} x_i U_{ij}^dagger

The Laplacian matvec accumulates:
    (L_F x)_j += diff
    (L_F x)_i -= U^dagger diff U
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg

if TYPE_CHECKING:
    from atft.topology.transport_maps import TransportMapBuilder


class SheafLaplacian:
    """Matrix-free sheaf Laplacian on a Vietoris-Rips graph.

    Args:
        transport_builder: A TransportMapBuilder with cached eigendecomp.
        unfolded_zeros: 1-D array of unfolded zero positions (imaginary parts).
    """

    def __init__(
        self,
        transport_builder: TransportMapBuilder,
        unfolded_zeros: NDArray[np.float64],
    ) -> None:
        self._builder = transport_builder
        self._zeros = np.asarray(unfolded_zeros, dtype=np.float64).ravel()
        self._N = len(self._zeros)
        self._K = transport_builder.K
        self._dim = self._N * self._K * self._K

        # Ensure eigendecomp is cached
        transport_builder.build_generator_sum()

        # Sort indices for early-exit edge enumeration
        self._sorted_idx = np.argsort(self._zeros)
        self._sorted_zeros = self._zeros[self._sorted_idx]

    @property
    def dim(self) -> int:
        """Total dimension of the vertex vector space: N * K * K."""
        return self._dim

    def _edges(self, epsilon: float) -> list[tuple[int, int]]:
        """Return edges [(i,j)] with i<j, |gamma_j - gamma_i| <= epsilon.

        Uses sorted zeros for early-exit optimization.
        """
        if epsilon <= 0:
            return []

        edges: list[tuple[int, int]] = []
        N = self._N
        sorted_z = self._sorted_zeros
        sorted_idx = self._sorted_idx

        for a in range(N):
            for b in range(a + 1, N):
                gap = sorted_z[b] - sorted_z[a]
                if gap > epsilon:
                    break  # sorted, so all subsequent b have larger gap
                # Map back to original indices, ensure i < j
                orig_a = int(sorted_idx[a])
                orig_b = int(sorted_idx[b])
                i, j = min(orig_a, orig_b), max(orig_a, orig_b)
                edges.append((i, j))

        return edges

    def matvec(self, x: NDArray[np.complex128], epsilon: float) -> NDArray[np.complex128]:
        """Apply L_F to vector x.

        x is a flattened (N*K*K,) complex vector, interpreted as (N, K, K) matrices.
        Returns (N*K*K,) complex vector.
        """
        K = self._K
        N = self._N
        x = np.asarray(x, dtype=np.complex128).ravel()
        X = x.reshape(N, K, K)
        Y = np.zeros_like(X)

        edges = self._edges(epsilon)

        for i, j in edges:
            U = self._builder.transport(self._zeros[j] - self._zeros[i])
            Uh = U.conj().T

            # diff = x_j - U @ x_i @ U^dagger
            diff = X[j] - U @ X[i] @ Uh

            Y[j] += diff
            # y_i -= U^dagger @ diff @ U
            Y[i] -= Uh @ diff @ U

        return Y.ravel()

    def as_linear_operator(self, epsilon: float) -> LinearOperator:
        """Wrap matvec as a scipy LinearOperator."""
        dim = self._dim

        def mv(x: NDArray) -> NDArray:
            return self.matvec(x, epsilon)

        def rmv(x: NDArray) -> NDArray:
            # L_F is Hermitian, so rmatvec = matvec
            return self.matvec(x, epsilon)

        return LinearOperator(
            shape=(dim, dim),
            matvec=mv,
            rmatvec=rmv,
            dtype=np.complex128,
        )

    def frobenius_norm_estimate(self, epsilon: float) -> float:
        """Estimate Frobenius norm from edge count: sqrt(2 * K * |E| * K)."""
        edges = self._edges(epsilon)
        K = self._K
        n_edges = len(edges)
        if n_edges == 0:
            return 0.0
        return np.sqrt(2.0 * K * n_edges * K)

    def smallest_eigenvalues(
        self,
        epsilon: float,
        m: int = 20,
        solver: str = "auto",
    ) -> NDArray[np.float64]:
        """Compute m smallest eigenvalues of L_F(epsilon).

        Args:
            epsilon: Rips radius.
            m: Number of eigenvalues to compute.
            solver: "auto" (try LOBPCG then eigsh), "lobpcg", or "eigsh".

        Returns:
            Sorted real eigenvalues, length min(m, dim-1) or m.
        """
        dim = self._dim
        edges = self._edges(epsilon)

        # Degenerate cases: no edges or eps <= 0
        if epsilon <= 0 or len(edges) == 0:
            return np.zeros(m, dtype=np.float64)

        # Clamp m to dim - 1 (eigsh requirement for sparse)
        m_clamped = min(m, dim - 1)
        if m_clamped < 1:
            return np.zeros(min(m, dim), dtype=np.float64)

        op = self.as_linear_operator(epsilon)

        if solver == "auto":
            # Try LOBPCG first, fall back to eigsh
            try:
                eigs = self._solve_lobpcg(op, m_clamped)
            except Exception:
                eigs = self._solve_eigsh(op, m_clamped)
        elif solver == "lobpcg":
            eigs = self._solve_lobpcg(op, m_clamped)
        elif solver == "eigsh":
            eigs = self._solve_eigsh(op, m_clamped)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        eigs = np.real(eigs)
        eigs = np.sort(eigs)
        # Clamp tiny negatives to zero
        eigs = np.maximum(eigs, 0.0)
        return eigs

    def _solve_lobpcg(
        self, op: LinearOperator, m: int
    ) -> NDArray[np.float64]:
        """Solve using LOBPCG."""
        rng = np.random.default_rng(0)
        X0 = rng.standard_normal((op.shape[0], m)) + 1j * rng.standard_normal(
            (op.shape[0], m)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            eigenvalues, _ = lobpcg(op, X0, largest=False, maxiter=500, tol=1e-10)
        return eigenvalues

    def _solve_eigsh(
        self, op: LinearOperator, m: int
    ) -> NDArray[np.float64]:
        """Solve using eigsh with shift-invert via SM."""
        eigenvalues, _ = eigsh(op, k=m, which="SM")
        return eigenvalues

    def kernel_dimension(
        self,
        epsilon: float,
        tol: float | None = None,
        m: int = 20,
    ) -> int:
        """Count eigenvalues below tolerance (approximate kernel dimension).

        Args:
            epsilon: Rips radius.
            tol: Tolerance for "zero" eigenvalue. Default: 1e-6 * frobenius_norm_estimate.
            m: Number of eigenvalues to compute.

        Returns:
            Number of eigenvalues below tol.
        """
        dim = self._dim
        edges = self._edges(epsilon)

        # No edges or eps <= 0 => full kernel
        if epsilon <= 0 or len(edges) == 0:
            return dim

        if tol is None:
            fnorm = self.frobenius_norm_estimate(epsilon)
            tol = 1e-6 * fnorm if fnorm > 0 else 1e-10

        eigs = self.smallest_eigenvalues(epsilon, m=m)
        return int(np.sum(eigs < tol))

    def extract_global_sections(
        self,
        epsilon: float,
        tol: float | None = None,
    ) -> NDArray[np.complex128]:
        """Extract approximate global sections (kernel eigenvectors).

        Returns:
            Array of shape (num_sections, N, K, K) containing reshaped
            eigenvectors with eigenvalue below tol.
        """
        dim = self._dim
        N = self._N
        K = self._K
        edges = self._edges(epsilon)

        if tol is None:
            fnorm = self.frobenius_norm_estimate(epsilon)
            tol = 1e-6 * fnorm if fnorm > 0 else 1e-10

        # No edges => everything is in the kernel; return identity-like basis
        if epsilon <= 0 or len(edges) == 0:
            sections = np.zeros((dim, N, K, K), dtype=np.complex128)
            for idx in range(dim):
                vec = np.zeros(dim, dtype=np.complex128)
                vec[idx] = 1.0
                sections[idx] = vec.reshape(N, K, K)
            return sections

        # Compute eigenpairs
        m = min(20, dim - 1)
        if m < 1:
            return np.zeros((0, N, K, K), dtype=np.complex128)

        op = self.as_linear_operator(epsilon)
        try:
            eigenvalues, eigenvectors = eigsh(op, k=m, which="SM")
        except Exception:
            return np.zeros((0, N, K, K), dtype=np.complex128)

        eigenvalues = np.real(eigenvalues)
        # Select kernel vectors
        mask = eigenvalues < tol
        kernel_vecs = eigenvectors[:, mask]

        n_sections = kernel_vecs.shape[1]
        sections = np.zeros((n_sections, N, K, K), dtype=np.complex128)
        for i in range(n_sections):
            sections[i] = kernel_vecs[:, i].reshape(N, K, K)

        return sections
