"""Core data types for the ATFT framework.

All types are immutable (frozen dataclasses). Modules communicate
through these types and never depend on each other directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class CurveType(Enum):
    BETTI = "betti"
    GINI = "gini"
    PERSISTENCE = "persistence"


@dataclass(frozen=True)
class PointCloud:
    """A finite metric space: N points in R^d."""

    points: NDArray[np.float64]
    metadata: dict = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    @property
    def dimension(self) -> int:
        return self.points.shape[1]


@dataclass(frozen=True)
class PointCloudBatch:
    """A collection of point clouds for batch processing."""

    clouds: list[PointCloud]

    @property
    def batch_size(self) -> int:
        return len(self.clouds)

    def uniform_size(self) -> int | None:
        """Returns N if all clouds have the same size, else None."""
        sizes = {c.n_points for c in self.clouds}
        return sizes.pop() if len(sizes) == 1 else None


@dataclass(frozen=True)
class PersistenceDiagram:
    """Birth-death pairs for each homological degree.

    diagrams[k] is an (n_features, 2) array of (birth, death) pairs.
    The immortal H_0 feature is stored with death = np.inf.
    """

    diagrams: dict[int, NDArray[np.float64]]
    metadata: dict = field(default_factory=dict)

    def degree(self, k: int) -> NDArray[np.float64]:
        """Get persistence pairs for degree k. Returns empty (0,2) if missing."""
        return self.diagrams.get(k, np.empty((0, 2), dtype=np.float64))

    def lifetimes(self, k: int) -> NDArray[np.float64]:
        """Persistence = death - birth for degree k."""
        d = self.degree(k)
        if len(d) > 0:
            return d[:, 1] - d[:, 0]
        return np.array([], dtype=np.float64)

    @property
    def max_degree(self) -> int:
        return max(self.diagrams.keys()) if self.diagrams else -1


@dataclass(frozen=True)
class EvolutionCurve:
    """A topological evolution function sampled on an epsilon grid."""

    epsilon_grid: NDArray[np.float64]
    values: NDArray[np.float64]
    curve_type: CurveType
    degree: int

    @property
    def n_steps(self) -> int:
        return len(self.epsilon_grid)


@dataclass(frozen=True)
class EvolutionCurveSet:
    """All evolution curves for a single configuration."""

    betti: dict[int, EvolutionCurve]
    gini: dict[int, EvolutionCurve]
    persistence: dict[int, EvolutionCurve]

    def curve(self, curve_type: CurveType, degree: int) -> EvolutionCurve:
        lookup = {
            CurveType.BETTI: self.betti,
            CurveType.GINI: self.gini,
            CurveType.PERSISTENCE: self.persistence,
        }
        return lookup[curve_type][degree]


@dataclass(frozen=True)
class WaypointSignature:
    """The finite-dimensional topological fingerprint W(C).

    W_0(C) = (eps*, {eps_w,i}, {delta_0(eps_w,i)}, G_0(eps*), dG_0/deps|_eps*)
    Lives in R^(2K+3) where K = len(waypoint_scales).
    """

    onset_scale: float
    waypoint_scales: NDArray[np.float64]
    topo_derivatives: NDArray[np.float64]
    gini_at_onset: float
    gini_derivative_at_onset: float

    def as_vector(self) -> NDArray[np.float64]:
        """Flatten to R^(2K+3) for statistical comparison."""
        return np.concatenate([
            [self.onset_scale],
            self.waypoint_scales,
            self.topo_derivatives,
            [self.gini_at_onset],
            [self.gini_derivative_at_onset],
        ])

    @property
    def n_waypoints(self) -> int:
        return len(self.waypoint_scales)

    @property
    def vector_dimension(self) -> int:
        return 2 * self.n_waypoints + 3


@dataclass(frozen=True)
class ValidationResult:
    """Output of the statistical comparison."""

    mahalanobis_distance: float
    p_value: float
    l2_distance_betti: float
    l2_distance_gini: float
    within_confidence_band: bool
    ensemble_size: int
    metadata: dict = field(default_factory=dict)
