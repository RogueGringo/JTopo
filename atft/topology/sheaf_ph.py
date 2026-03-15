"""Sheaf persistent homology: epsilon and sigma sweep orchestrators.

Produces SheafBettiCurve (single sigma) and 2D heatmaps (sigma sweep)
by wrapping the matrix-free SheafLaplacian kernel computation.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from atft.core.types import SheafBettiCurve
from atft.topology.sheaf_laplacian import SheafLaplacian
from atft.topology.transport_maps import TransportMapBuilder


class SheafPH:
    """Orchestrates epsilon and sigma sweeps for sheaf Betti curves.

    Args:
        transport_builder: TransportMapBuilder for the current (K, sigma).
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

    def sweep(self, epsilon_grid: NDArray[np.float64], m: int = 20) -> SheafBettiCurve:
        """Compute sheaf Betti curve beta_0^F(epsilon) across filtration.

        At epsilon=0, returns N*K^2 without calling the eigensolver.
        """
        lap = SheafLaplacian(self._builder, self._zeros)
        n_steps = len(epsilon_grid)
        kernel_dims = np.zeros(n_steps, dtype=np.int64)
        all_eigs = np.zeros((n_steps, m), dtype=np.float64)

        for idx, eps in enumerate(epsilon_grid):
            if eps <= 0:
                kernel_dims[idx] = self._N * self._K * self._K
                continue

            eigs = lap.smallest_eigenvalues(eps, m=m)
            all_eigs[idx, :len(eigs)] = eigs

            # Derive kernel dim from eigenvalues directly (avoid double eigensolver call)
            tol = lap.frobenius_norm_estimate(eps) * 1e-6
            if tol == 0:
                tol = 1e-12
            kernel_dims[idx] = int(np.sum(np.abs(eigs) < tol))

        return SheafBettiCurve(
            epsilon_grid=np.array(epsilon_grid, dtype=np.float64),
            kernel_dimensions=kernel_dims,
            smallest_eigenvalues=all_eigs,
            sigma=self._builder.sigma,
            K=self._K,
        )

    def sigma_sweep(
        self,
        epsilon_grid: NDArray[np.float64],
        sigma_grid: NDArray[np.float64],
        m: int = 20,
    ) -> NDArray[np.int64]:
        """Compute 2D heatmap beta_0^F(epsilon, sigma).

        Rebuilds TransportMapBuilder at each sigma value.
        Returns array of shape (len(sigma_grid), len(epsilon_grid)).
        """
        heatmap = np.zeros((len(sigma_grid), len(epsilon_grid)), dtype=np.int64)

        for s_idx, sigma in enumerate(sigma_grid):
            builder = TransportMapBuilder(K=self._K, sigma=sigma)
            ph = SheafPH(builder, self._zeros)
            curve = ph.sweep(epsilon_grid, m=m)
            heatmap[s_idx, :] = curve.kernel_dimensions

        return heatmap
