"""Tests for top-K gap-based waypoint extraction."""
import numpy as np
import pytest

from atft.analysis.evolution_curves import EvolutionCurveComputer
from atft.analysis.waypoint_extractor import WaypointExtractor
from atft.core.types import (
    CurveType,
    EvolutionCurve,
    EvolutionCurveSet,
    PersistenceDiagram,
)


def _make_pd(gaps):
    births = np.zeros(len(gaps), dtype=np.float64)
    deaths = np.array(gaps, dtype=np.float64)
    births = np.append(births, 0.0)
    deaths = np.append(deaths, np.inf)
    return PersistenceDiagram(diagrams={0: np.column_stack([births, deaths])})


class TestWaypointExtractor:
    def test_top_k_selects_largest_gaps(self):
        pd = _make_pd([1.0, 2.0, 3.0, 4.0])
        curves = EvolutionCurveComputer(n_steps=100).compute(pd, degree=0)
        ext = WaypointExtractor(k_waypoints=2)
        sig = ext.extract(pd, curves, degree=0)
        assert sig.n_waypoints == 2
        np.testing.assert_array_almost_equal(np.sort(sig.waypoint_scales), [3.0, 4.0])

    def test_vector_dimension(self):
        pd = _make_pd([1.0, 2.0, 3.0])
        curves = EvolutionCurveComputer(n_steps=100).compute(pd, degree=0)
        sig = WaypointExtractor(k_waypoints=2).extract(pd, curves, degree=0)
        assert sig.vector_dimension == 7

    def test_onset_scale_is_smallest_gap(self):
        pd = _make_pd([0.5, 2.0, 3.0])
        curves = EvolutionCurveComputer(n_steps=100).compute(pd, degree=0)
        sig = WaypointExtractor(k_waypoints=2).extract(pd, curves, degree=0)
        assert abs(sig.onset_scale - 0.5) < 1e-10

    def test_zero_padding_when_few_gaps(self):
        pd = _make_pd([3.0])
        curves = EvolutionCurveComputer(n_steps=100).compute(pd, degree=0)
        sig = WaypointExtractor(k_waypoints=2).extract(pd, curves, degree=0)
        assert sig.n_waypoints == 2
        assert sig.waypoint_scales[0] == 0.0

    def test_waypoints_sorted_by_position(self):
        pd = _make_pd([5.0, 1.0, 3.0, 2.0, 4.0])
        curves = EvolutionCurveComputer(n_steps=100).compute(pd, degree=0)
        sig = WaypointExtractor(k_waypoints=3).extract(pd, curves, degree=0)
        assert np.all(np.diff(sig.waypoint_scales) >= 0)

    def test_empty_persistence_returns_zero_signature(self):
        pd = PersistenceDiagram(diagrams={0: np.array([[0.0, np.inf]])})
        eps = np.linspace(0, 1, 50)
        dummy_curves = EvolutionCurveSet(
            betti={0: EvolutionCurve(eps, np.ones(50), CurveType.BETTI, 0)},
            gini={0: EvolutionCurve(eps, np.zeros(50), CurveType.GINI, 0)},
            persistence={0: EvolutionCurve(eps, np.ones(50), CurveType.PERSISTENCE, 0)},
        )
        sig = WaypointExtractor(k_waypoints=2).extract(pd, dummy_curves, degree=0)
        assert sig.onset_scale == 0.0
        np.testing.assert_array_equal(sig.waypoint_scales, [0.0, 0.0])

    def test_gini_at_onset_is_float(self):
        pd = _make_pd([1.0, 2.0, 3.0])
        curves = EvolutionCurveComputer(n_steps=100).compute(pd, degree=0)
        sig = WaypointExtractor(k_waypoints=2).extract(pd, curves, degree=0)
        assert isinstance(sig.gini_at_onset, float)
        assert 0.0 <= sig.gini_at_onset <= 1.0

    def test_as_vector_correct_shape(self):
        pd = _make_pd([1.0, 2.0, 3.0, 4.0])
        curves = EvolutionCurveComputer(n_steps=100).compute(pd, degree=0)
        sig = WaypointExtractor(k_waypoints=2).extract(pd, curves, degree=0)
        vec = sig.as_vector()
        assert vec.shape == (7,)
        assert vec.dtype == np.float64
