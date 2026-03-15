"""Evolution curve computation: Betti, Gini, and Persistence curves."""
from __future__ import annotations

import numpy as np

from atft.core.types import (
    CurveType,
    EvolutionCurve,
    EvolutionCurveSet,
    PersistenceDiagram,
)


class EvolutionCurveComputer:
    """Computes topological evolution curves from persistence diagrams."""

    def __init__(self, n_steps: int = 1000):
        self.n_steps = n_steps

    def compute(self, pd: PersistenceDiagram, degree: int = 0, epsilon_max: float | None = None) -> EvolutionCurveSet:
        pairs = pd.degree(degree)
        if len(pairs) == 0:
            eps_grid = np.linspace(0, 1, self.n_steps)
            empty = EvolutionCurve(eps_grid, np.zeros(self.n_steps), CurveType.BETTI, degree)
            return EvolutionCurveSet(
                betti={degree: empty},
                gini={degree: EvolutionCurve(eps_grid, np.zeros(self.n_steps), CurveType.GINI, degree)},
                persistence={degree: EvolutionCurve(eps_grid, np.zeros(self.n_steps), CurveType.PERSISTENCE, degree)},
            )

        births = pairs[:, 0]
        deaths = pairs[:, 1].copy()

        finite_deaths = deaths[np.isfinite(deaths)]
        if epsilon_max is None:
            if len(finite_deaths) > 0:
                epsilon_max = 1.1 * np.max(finite_deaths)
            else:
                epsilon_max = 1.0

        # Cap immortal features at epsilon_max for Gini/Persistence
        capped_deaths = deaths.copy()
        capped_deaths[~np.isfinite(capped_deaths)] = epsilon_max
        lifetimes = capped_deaths - births

        eps_grid = np.linspace(0, epsilon_max, self.n_steps)

        betti_vals = np.empty(self.n_steps, dtype=np.float64)
        gini_vals = np.empty(self.n_steps, dtype=np.float64)
        pers_vals = np.empty(self.n_steps, dtype=np.float64)

        for i, eps in enumerate(eps_grid):
            alive = (births <= eps) & (deaths > eps)
            n_alive = np.sum(alive)
            betti_vals[i] = n_alive

            alive_lifetimes = lifetimes[alive]
            gini_vals[i] = self._gini(alive_lifetimes)
            pers_vals[i] = np.sum(alive_lifetimes)

        return EvolutionCurveSet(
            betti={degree: EvolutionCurve(eps_grid, betti_vals, CurveType.BETTI, degree)},
            gini={degree: EvolutionCurve(eps_grid, gini_vals, CurveType.GINI, degree)},
            persistence={degree: EvolutionCurve(eps_grid, pers_vals, CurveType.PERSISTENCE, degree)},
        )

    @staticmethod
    def _gini(values: np.ndarray) -> float:
        """Gini coefficient using the 1-indexed sorted formula."""
        n = len(values)
        if n <= 1:
            return 0.0
        total = np.sum(values)
        if total == 0.0:
            return 0.0
        sorted_v = np.sort(values)
        index = np.arange(1, n + 1, dtype=np.float64)
        return float(
            (2.0 * np.sum(index * sorted_v)) / (n * total) - (n + 1.0) / n
        )
