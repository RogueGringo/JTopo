"""Tests for evolution curve computation."""
import numpy as np
import pytest

from atft.analysis.evolution_curves import EvolutionCurveComputer
from atft.core.types import CurveType, PersistenceDiagram


def _make_pd(gaps):
    """Helper: create a PersistenceDiagram from gap values."""
    births = np.zeros(len(gaps), dtype=np.float64)
    deaths = np.array(gaps, dtype=np.float64)
    births = np.append(births, 0.0)
    deaths = np.append(deaths, np.inf)
    return PersistenceDiagram(diagrams={0: np.column_stack([births, deaths])})


class TestBettiCurve:
    def test_starts_at_n(self):
        pd = _make_pd([1.0, 2.0, 3.0, 4.0])
        computer = EvolutionCurveComputer(n_steps=100)
        curves = computer.compute(pd, degree=0)
        betti = curves.betti[0]
        assert betti.values[0] == 5.0

    def test_ends_at_one(self):
        pd = _make_pd([1.0, 2.0, 3.0])
        computer = EvolutionCurveComputer(n_steps=100)
        curves = computer.compute(pd, degree=0)
        betti = curves.betti[0]
        assert betti.values[-1] == 1.0

    def test_monotonically_decreasing(self):
        pd = _make_pd([1.0, 2.0, 3.0, 4.0])
        computer = EvolutionCurveComputer(n_steps=200)
        curves = computer.compute(pd, degree=0)
        betti = curves.betti[0]
        assert np.all(np.diff(betti.values) <= 0)

    def test_known_values(self):
        pd = _make_pd([1.0, 3.0])
        computer = EvolutionCurveComputer(n_steps=1000)
        curves = computer.compute(pd, degree=0)
        betti = curves.betti[0]
        eps = curves.betti[0].epsilon_grid
        idx_05 = np.argmin(np.abs(eps - 0.5))
        idx_2 = np.argmin(np.abs(eps - 2.0))
        assert betti.values[idx_05] == 3.0
        assert betti.values[idx_2] == 2.0


class TestGiniCurve:
    def test_uniform_gaps_gini_low(self):
        pd = _make_pd([1.0, 1.0, 1.0, 1.0])
        computer = EvolutionCurveComputer(n_steps=100)
        curves = computer.compute(pd, degree=0)
        gini = curves.gini[0]
        assert gini.values[0] < 0.5
        assert gini.values[0] >= 0.0

    def test_empty_persistence_diagram(self):
        pd = PersistenceDiagram(diagrams={})
        computer = EvolutionCurveComputer(n_steps=50)
        curves = computer.compute(pd, degree=0)
        assert np.all(curves.betti[0].values == 0)
        assert np.all(curves.gini[0].values == 0)

    def test_gini_bounded(self):
        pd = _make_pd([0.1, 0.5, 1.0, 5.0])
        computer = EvolutionCurveComputer(n_steps=100)
        curves = computer.compute(pd, degree=0)
        gini = curves.gini[0]
        assert np.all(gini.values >= 0.0)
        assert np.all(gini.values <= 1.0)

    def test_gini_edge_case_single_feature(self):
        pd = _make_pd([1.0])
        computer = EvolutionCurveComputer(n_steps=100)
        curves = computer.compute(pd, degree=0)
        gini = curves.gini[0]
        assert gini.values[-1] == 0.0


class TestPersistenceCurve:
    def test_starts_positive(self):
        pd = _make_pd([1.0, 2.0, 3.0])
        computer = EvolutionCurveComputer(n_steps=100)
        curves = computer.compute(pd, degree=0)
        pers = curves.persistence[0]
        assert pers.values[0] > 0

    def test_curve_type_set(self):
        pd = _make_pd([1.0])
        curves = EvolutionCurveComputer(n_steps=10).compute(pd, degree=0)
        assert curves.betti[0].curve_type == CurveType.BETTI
        assert curves.gini[0].curve_type == CurveType.GINI
        assert curves.persistence[0].curve_type == CurveType.PERSISTENCE
