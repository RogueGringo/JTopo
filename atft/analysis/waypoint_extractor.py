"""Top-K gap-based waypoint extraction."""
from __future__ import annotations

import numpy as np

from atft.core.types import (
    EvolutionCurveSet,
    PersistenceDiagram,
    WaypointSignature,
)


class WaypointExtractor:
    """Extracts the waypoint signature W_0(C) from persistence data.

    Uses gap-based extraction (not numerical derivatives) to avoid
    differentiation artifacts on the step-function Betti curve.
    """

    def __init__(self, k_waypoints: int = 2):
        self.k_waypoints = k_waypoints

    def extract(self, pd: PersistenceDiagram, curves: EvolutionCurveSet, degree: int = 0) -> WaypointSignature:
        pairs = pd.degree(degree)
        finite_mask = np.isfinite(pairs[:, 1])
        finite_deaths = pairs[finite_mask, 1]

        if len(finite_deaths) == 0:
            return self._empty_signature()

        onset_scale = float(np.min(finite_deaths))

        sorted_desc = np.sort(finite_deaths)[::-1]
        k = min(self.k_waypoints, len(sorted_desc))
        top_k = sorted_desc[:k]

        if k < self.k_waypoints:
            top_k = np.concatenate([
                top_k,
                np.zeros(self.k_waypoints - k, dtype=np.float64),
            ])

        top_k = np.sort(top_k)

        # Store gap magnitudes as proxy for topological derivative.
        # In 1D H_0, every merging event has literal delta = -1, which
        # would create zero-variance columns in the covariance matrix.
        # Gap magnitude is a more informative proxy (per spec Section 2.6).
        topo_derivs = -top_k.copy()

        betti_curve = curves.betti[degree]
        gini_curve = curves.gini[degree]
        eps = betti_curve.epsilon_grid

        onset_idx = np.argmin(np.abs(eps - onset_scale))
        gini_at_onset = float(gini_curve.values[onset_idx])

        gini_deriv = np.gradient(gini_curve.values, eps)
        gini_deriv_at_onset = float(gini_deriv[onset_idx])

        return WaypointSignature(
            onset_scale=onset_scale,
            waypoint_scales=top_k,
            topo_derivatives=topo_derivs,
            gini_at_onset=gini_at_onset,
            gini_derivative_at_onset=gini_deriv_at_onset,
        )

    def _empty_signature(self) -> WaypointSignature:
        return WaypointSignature(
            onset_scale=0.0,
            waypoint_scales=np.zeros(self.k_waypoints, dtype=np.float64),
            topo_derivatives=np.zeros(self.k_waypoints, dtype=np.float64),
            gini_at_onset=0.0,
            gini_derivative_at_onset=0.0,
        )
