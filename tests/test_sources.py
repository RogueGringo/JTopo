"""Tests for configuration sources."""
import numpy as np
import pytest
from pathlib import Path

from atft.core.protocols import ConfigurationSource
from atft.sources.poisson import PoissonSource
from atft.sources.gue import GUESource
from atft.sources.zeta_zeros import ZetaZerosSource

TEST_ZEROS_PATH = Path(__file__).parent / "data" / "test_zeros.txt"


class TestPoissonSource:
    def test_satisfies_protocol(self):
        assert isinstance(PoissonSource(seed=42), ConfigurationSource)

    def test_generate_shape(self):
        src = PoissonSource(seed=42)
        cloud = src.generate(100)
        assert cloud.points.shape == (100, 1)
        assert cloud.points.dtype == np.float64

    def test_generate_sorted_and_positive(self):
        src = PoissonSource(seed=42)
        cloud = src.generate(100)
        pts = cloud.points[:, 0]
        assert np.all(pts > 0)
        assert np.all(np.diff(pts) > 0)

    def test_gaps_are_exponential(self):
        src = PoissonSource(seed=42)
        cloud = src.generate(10_000)
        gaps = np.diff(cloud.points[:, 0])
        assert 0.95 < np.mean(gaps) < 1.05

    def test_reproducibility(self):
        c1 = PoissonSource(seed=42).generate(100)
        c2 = PoissonSource(seed=42).generate(100)
        np.testing.assert_array_equal(c1.points, c2.points)

    def test_generate_batch(self):
        src = PoissonSource(seed=42)
        batch = src.generate_batch(100, batch_size=5)
        assert batch.batch_size == 5
        assert all(c.n_points == 100 for c in batch.clouds)

    def test_batch_members_differ(self):
        src = PoissonSource(seed=42)
        batch = src.generate_batch(100, batch_size=3)
        assert not np.array_equal(batch.clouds[0].points, batch.clouds[1].points)

    def test_metadata(self):
        src = PoissonSource(seed=42)
        cloud = src.generate(100)
        assert cloud.metadata["source"] == "poisson"
        assert cloud.metadata["n_points"] == 100


class TestGUESource:
    def test_satisfies_protocol(self):
        assert isinstance(GUESource(seed=42), ConfigurationSource)

    def test_generate_shape(self):
        src = GUESource(seed=42)
        cloud = src.generate(50)
        assert cloud.points.shape == (50, 1)
        assert cloud.points.dtype == np.float64

    def test_eigenvalues_are_real_and_sorted(self):
        src = GUESource(seed=42)
        cloud = src.generate(50)
        pts = cloud.points[:, 0]
        assert np.all(np.isreal(pts))
        assert np.all(np.diff(pts) >= 0)

    def test_eigenvalues_bounded_by_semicircle(self):
        src = GUESource(seed=42)
        cloud = src.generate(200)
        pts = cloud.points[:, 0]
        assert np.all(pts > -1.5)
        assert np.all(pts < 1.5)

    def test_reproducibility(self):
        c1 = GUESource(seed=42).generate(50)
        c2 = GUESource(seed=42).generate(50)
        np.testing.assert_array_almost_equal(c1.points, c2.points)

    def test_generate_batch(self):
        src = GUESource(seed=42)
        batch = src.generate_batch(50, batch_size=5)
        assert batch.batch_size == 5
        assert all(c.n_points == 50 for c in batch.clouds)

    def test_metadata(self):
        src = GUESource(seed=42)
        cloud = src.generate(50)
        assert cloud.metadata["source"] == "gue"


class TestZetaZerosSource:
    def test_satisfies_protocol(self):
        assert isinstance(ZetaZerosSource(data_path=TEST_ZEROS_PATH), ConfigurationSource)

    def test_generate_shape(self):
        src = ZetaZerosSource(data_path=TEST_ZEROS_PATH)
        cloud = src.generate(5)
        assert cloud.points.shape == (5, 1)
        assert cloud.points.dtype == np.float64

    def test_generate_loads_first_n(self):
        src = ZetaZerosSource(data_path=TEST_ZEROS_PATH)
        cloud = src.generate(3)
        expected = np.array([[14.134725141734693],
                             [21.022039638771555],
                             [25.010857580145688]])
        np.testing.assert_array_almost_equal(cloud.points, expected)

    def test_generate_all(self):
        src = ZetaZerosSource(data_path=TEST_ZEROS_PATH)
        cloud = src.generate(10)
        assert cloud.points.shape == (10, 1)

    def test_sorted_ascending(self):
        src = ZetaZerosSource(data_path=TEST_ZEROS_PATH)
        cloud = src.generate(10)
        assert np.all(np.diff(cloud.points[:, 0]) > 0)

    def test_metadata(self):
        src = ZetaZerosSource(data_path=TEST_ZEROS_PATH)
        cloud = src.generate(5)
        assert cloud.metadata["source"] == "zeta_zeros"

    def test_n_exceeds_available_raises(self):
        src = ZetaZerosSource(data_path=TEST_ZEROS_PATH)
        with pytest.raises(ValueError, match="Requested 20.*only 10"):
            src.generate(20)
