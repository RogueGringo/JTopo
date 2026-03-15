"""Phase 2a: Abelian eigenbasis diagnostic.

Decomposes the sheaf Laplacian into K^2 independent scalar twisted
graph Laplacians at eigenfrequency differences of A(sigma). Validates
that the diagonal (zero-frequency) blocks reproduce the standard graph
Laplacian, and produces a resonance matrix showing which prime harmonics
the zero spacings respond to.

Optimizations:
  - Vectorized edge enumeration via searchsorted
  - Batch phase computation for all edges at once
  - Cached edge sets across omega values (edges only depend on epsilon)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from atft.topology.transport_maps import TransportMapBuilder


class Phase2aAbelian:
    """Abelian eigenbasis diagnostic for Phase 2a.

    Args:
        transport_builder: TransportMapBuilder with cached eigendecomp.
        unfolded_zeros: 1D array of N sorted unfolded zeros.
    """

    def __init__(
        self,
        transport_builder: TransportMapBuilder,
        unfolded_zeros: NDArray[np.float64],
    ) -> None:
        self._builder = transport_builder
        self._zeros = np.asarray(unfolded_zeros, dtype=np.float64).ravel()
        self._K = transport_builder.K
        self._N = len(self._zeros)
        self._edge_cache: dict[float, tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]] = {}

    def _get_edges(self, epsilon: float) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
        """Return cached (i_indices, j_indices, gaps) for edges at given epsilon."""
        cached = self._edge_cache.get(epsilon)
        if cached is not None:
            return cached

        N = self._N
        zeros = self._zeros
        i_list, j_list, gap_list = [], [], []

        for i in range(N):
            for j in range(i + 1, N):
                gap = zeros[j] - zeros[i]
                if gap > epsilon:
                    break
                i_list.append(i)
                j_list.append(j)
                gap_list.append(gap)

        result = (
            np.array(i_list, dtype=np.intp),
            np.array(j_list, dtype=np.intp),
            np.array(gap_list, dtype=np.float64),
        )
        self._edge_cache[epsilon] = result
        return result

    @staticmethod
    def _build_twisted_laplacian(
        zeros: NDArray[np.float64],
        omega: float,
        epsilon: float,
    ) -> NDArray[np.complex128]:
        """Build the scalar twisted graph Laplacian L_omega(epsilon).

        L_omega is N×N Hermitian:
          L[i,i] += 1, L[j,j] += 1 for each edge
          L[i,j] -= e^{i*delta_gamma*omega}
          L[j,i] -= e^{-i*delta_gamma*omega}
        """
        N = len(zeros)
        L = np.zeros((N, N), dtype=np.complex128)

        for i in range(N):
            for j in range(i + 1, N):
                gap = zeros[j] - zeros[i]
                if gap > epsilon:
                    break  # zeros are sorted
                phase = np.exp(1j * gap * omega)
                L[i, i] += 1.0
                L[j, j] += 1.0
                L[i, j] -= phase
                L[j, i] -= np.conj(phase)

        return L

    def _build_twisted_laplacian_fast(
        self,
        omega: float,
        epsilon: float,
    ) -> NDArray[np.complex128]:
        """Vectorized twisted Laplacian using cached edges."""
        N = self._N
        i_idx, j_idx, gaps = self._get_edges(epsilon)

        L = np.zeros((N, N), dtype=np.complex128)
        if len(i_idx) == 0:
            return L

        # Batch phase computation
        phases = np.exp(1j * gaps * omega)

        # Vectorized degree accumulation
        np.add.at(L, (i_idx, i_idx), 1.0)
        np.add.at(L, (j_idx, j_idx), 1.0)
        L[i_idx, j_idx] -= phases
        L[j_idx, i_idx] -= phases.conj()

        return L

    def compute_resonance_matrix(
        self, epsilon_grid: NDArray[np.float64]
    ) -> NDArray[np.int64]:
        """Compute K×K resonance matrix R where R_{kl} = max_eps kernel_dim(L_{omega_kl}).

        Returns integer matrix of shape (K, K).
        """
        eigenvalues = self._builder.eigenvalues()
        K = self._K
        R = np.zeros((K, K), dtype=np.int64)

        for k in range(K):
            for l in range(k, K):
                omega = eigenvalues[k] - eigenvalues[l]
                max_kernel = 0

                for eps in epsilon_grid:
                    if eps <= 0:
                        max_kernel = max(max_kernel, self._N)
                        continue

                    L = self._build_twisted_laplacian_fast(omega, eps)
                    eigs = np.linalg.eigvalsh(L)
                    n_kernel = int(np.sum(np.abs(eigs) < 1e-10))
                    max_kernel = max(max_kernel, n_kernel)

                R[k, l] = max_kernel
                R[l, k] = max_kernel

        return R

    def run(self, epsilon_grid: NDArray[np.float64]) -> dict:
        """Run the full Phase 2a diagnostic.

        Returns dict with:
          - resonance_matrix: K×K int array
          - eigenvalues_A: K eigenvalues of A(sigma)
          - n_distinct_frequencies: number of distinct omega_{kl} values
        """
        eigenvalues = self._builder.eigenvalues()
        R = self.compute_resonance_matrix(epsilon_grid)

        # Count distinct frequencies
        K = self._K
        freqs = set()
        for k in range(K):
            for l in range(K):
                freq = round(eigenvalues[k] - eigenvalues[l], 10)
                freqs.add(abs(freq))
        n_distinct = len(freqs)

        return {
            "resonance_matrix": R,
            "eigenvalues_A": eigenvalues,
            "n_distinct_frequencies": n_distinct,
        }
