"""Tests for HDF5 cache serialization."""
import numpy as np
import pytest

from atft.io.cache import load_persistence_diagram, save_persistence_diagram
from atft.core.types import PersistenceDiagram


class TestPersistenceDiagramCache:
    def test_round_trip(self, tmp_path):
        diag = np.array([[0.0, 1.0], [0.0, 3.0], [0.0, np.inf]], dtype=np.float64)
        pd = PersistenceDiagram(
            diagrams={0: diag},
            metadata={"method": "analytical_h0", "n_points": 4},
        )
        path = tmp_path / "test.h5"
        save_persistence_diagram(pd, path)
        loaded = load_persistence_diagram(path)

        np.testing.assert_array_equal(loaded.degree(0), pd.degree(0))
        assert loaded.metadata["method"] == "analytical_h0"

    def test_empty_diagram(self, tmp_path):
        pd = PersistenceDiagram(diagrams={}, metadata={})
        path = tmp_path / "empty.h5"
        save_persistence_diagram(pd, path)
        loaded = load_persistence_diagram(path)
        assert loaded.max_degree == -1

    def test_multi_degree(self, tmp_path):
        pd = PersistenceDiagram(diagrams={
            0: np.array([[0.0, 1.0]], dtype=np.float64),
            1: np.array([[0.5, 2.0]], dtype=np.float64),
        })
        path = tmp_path / "multi.h5"
        save_persistence_diagram(pd, path)
        loaded = load_persistence_diagram(path)
        assert loaded.max_degree == 1
        np.testing.assert_array_equal(loaded.degree(1), pd.degree(1))
