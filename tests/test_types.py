"""Tests for ATFT core data types."""
import numpy as np
import pytest

from atft.core.types import (
    CurveType,
    EvolutionCurve,
    EvolutionCurveSet,
    PersistenceDiagram,
    PointCloud,
    PointCloudBatch,
    ValidationResult,
    WaypointSignature,
)


class TestPointCloud:
    def test_creation(self, simple_1d_points):
        pc = PointCloud(points=simple_1d_points)
        assert pc.n_points == 5
        assert pc.dimension == 1

    def test_immutable(self, simple_1d_points):
        pc = PointCloud(points=simple_1d_points)
        with pytest.raises(AttributeError):
            pc.points = np.zeros((3, 1))

    def test_metadata(self, simple_1d_points):
        pc = PointCloud(points=simple_1d_points, metadata={"source": "test"})
        assert pc.metadata["source"] == "test"

    def test_2d_points(self):
        pts = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        pc = PointCloud(points=pts)
        assert pc.n_points == 2
        assert pc.dimension == 2


class TestPointCloudBatch:
    def test_creation(self, simple_1d_points, uniform_1d_points):
        c1 = PointCloud(points=simple_1d_points)
        c2 = PointCloud(points=uniform_1d_points)
        batch = PointCloudBatch(clouds=[c1, c2])
        assert batch.batch_size == 2

    def test_uniform_size_same(self):
        c1 = PointCloud(points=np.zeros((5, 1)))
        c2 = PointCloud(points=np.zeros((5, 1)))
        batch = PointCloudBatch(clouds=[c1, c2])
        assert batch.uniform_size() == 5

    def test_uniform_size_different(self):
        c1 = PointCloud(points=np.zeros((5, 1)))
        c2 = PointCloud(points=np.zeros((3, 1)))
        batch = PointCloudBatch(clouds=[c1, c2])
        assert batch.uniform_size() is None


class TestPersistenceDiagram:
    def test_creation(self):
        diag = np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float64)
        pd = PersistenceDiagram(diagrams={0: diag})
        assert pd.max_degree == 0

    def test_degree_access(self):
        diag = np.array([[0.0, 1.0], [0.0, 2.0]], dtype=np.float64)
        pd = PersistenceDiagram(diagrams={0: diag})
        assert pd.degree(0).shape == (2, 2)

    def test_missing_degree_returns_empty(self):
        pd = PersistenceDiagram(diagrams={0: np.array([[0.0, 1.0]])})
        result = pd.degree(1)
        assert result.shape == (0, 2)
        assert result.dtype == np.float64

    def test_lifetimes(self):
        diag = np.array([[0.0, 1.0], [0.0, 3.0]], dtype=np.float64)
        pd = PersistenceDiagram(diagrams={0: diag})
        lts = pd.lifetimes(0)
        np.testing.assert_array_equal(lts, [1.0, 3.0])

    def test_lifetimes_empty(self):
        pd = PersistenceDiagram(diagrams={})
        lts = pd.lifetimes(0)
        assert len(lts) == 0
        assert lts.dtype == np.float64

    def test_max_degree_empty(self):
        pd = PersistenceDiagram(diagrams={})
        assert pd.max_degree == -1


class TestEvolutionCurve:
    def test_creation(self):
        eps = np.linspace(0, 1, 10)
        vals = np.arange(10, dtype=np.float64)
        ec = EvolutionCurve(
            epsilon_grid=eps, values=vals,
            curve_type=CurveType.BETTI, degree=0
        )
        assert ec.n_steps == 10
        assert ec.curve_type == CurveType.BETTI


class TestEvolutionCurveSet:
    def test_curve_lookup(self):
        eps = np.linspace(0, 1, 5)
        betti = EvolutionCurve(eps, np.ones(5), CurveType.BETTI, 0)
        gini = EvolutionCurve(eps, np.zeros(5), CurveType.GINI, 0)
        pers = EvolutionCurve(eps, np.ones(5) * 2, CurveType.PERSISTENCE, 0)
        cs = EvolutionCurveSet(
            betti={0: betti}, gini={0: gini}, persistence={0: pers}
        )
        assert cs.curve(CurveType.BETTI, 0) is betti
        assert cs.curve(CurveType.GINI, 0) is gini


class TestWaypointSignature:
    def test_as_vector_shape(self):
        ws = WaypointSignature(
            onset_scale=0.5,
            waypoint_scales=np.array([1.0, 2.0]),
            topo_derivatives=np.array([-3.0, -2.0]),
            gini_at_onset=0.3,
            gini_derivative_at_onset=0.01,
        )
        vec = ws.as_vector()
        assert vec.shape == (7,)
        assert ws.n_waypoints == 2
        assert ws.vector_dimension == 7

    def test_as_vector_values(self):
        ws = WaypointSignature(
            onset_scale=0.5,
            waypoint_scales=np.array([1.0, 2.0]),
            topo_derivatives=np.array([-3.0, -2.0]),
            gini_at_onset=0.3,
            gini_derivative_at_onset=0.01,
        )
        vec = ws.as_vector()
        expected = np.array([0.5, 1.0, 2.0, -3.0, -2.0, 0.3, 0.01])
        np.testing.assert_array_almost_equal(vec, expected)


class TestValidationResult:
    def test_creation(self):
        vr = ValidationResult(
            mahalanobis_distance=1.5,
            p_value=0.23,
            l2_distance_betti=0.05,
            l2_distance_gini=0.02,
            within_confidence_band=True,
            ensemble_size=1000,
        )
        assert vr.p_value == 0.23
        assert vr.within_confidence_band is True
