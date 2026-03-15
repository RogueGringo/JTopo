"""Tests for feature maps."""
import numpy as np

from atft.core.protocols import FeatureMap
from atft.core.types import PointCloud, PointCloudBatch
from atft.feature_maps.identity import IdentityMap
from atft.feature_maps.spectral_unfolding import SpectralUnfolding


class TestIdentityMap:
    def test_satisfies_protocol(self):
        assert isinstance(IdentityMap(), FeatureMap)

    def test_transform_passthrough(self, simple_1d_points):
        fm = IdentityMap()
        cloud = PointCloud(points=simple_1d_points)
        result = fm.transform(cloud)
        np.testing.assert_array_equal(result.points, cloud.points)

    def test_transform_batch(self, simple_1d_points, uniform_1d_points):
        fm = IdentityMap()
        batch = PointCloudBatch(clouds=[
            PointCloud(points=simple_1d_points),
            PointCloud(points=uniform_1d_points),
        ])
        result = fm.transform_batch(batch)
        assert result.batch_size == 2


class TestSpectralUnfolding:
    def test_satisfies_protocol(self):
        assert isinstance(SpectralUnfolding(method="rank"), FeatureMap)

    def test_rank_unfolding_mean_gap(self):
        rng = np.random.default_rng(42)
        pts = np.sort(rng.standard_normal(200)).reshape(-1, 1)
        cloud = PointCloud(points=pts.astype(np.float64))
        fm = SpectralUnfolding(method="rank")
        result = fm.transform(cloud)
        gaps = np.diff(result.points[:, 0])
        assert 0.9 < np.mean(gaps) < 1.1

    def test_rank_unfolding_sorted(self):
        rng = np.random.default_rng(42)
        pts = np.sort(rng.standard_normal(100)).reshape(-1, 1)
        cloud = PointCloud(points=pts.astype(np.float64))
        fm = SpectralUnfolding(method="rank")
        result = fm.transform(cloud)
        assert np.all(np.diff(result.points[:, 0]) > 0)

    def test_zeta_unfolding_mean_gap(self):
        zeros = np.array([
            14.134725, 21.022040, 25.010858, 30.424876, 32.935062,
            37.586178, 40.918719, 43.327073, 48.005151, 49.773832,
        ]).reshape(-1, 1)
        cloud = PointCloud(points=zeros.astype(np.float64))
        fm = SpectralUnfolding(method="zeta")
        result = fm.transform(cloud)
        gaps = np.diff(result.points[:, 0])
        assert 0.5 < np.mean(gaps) < 2.0

    def test_transform_batch(self):
        rng = np.random.default_rng(42)
        clouds = [
            PointCloud(points=np.sort(rng.standard_normal(50)).reshape(-1, 1))
            for _ in range(3)
        ]
        batch = PointCloudBatch(clouds=clouds)
        fm = SpectralUnfolding(method="rank")
        result = fm.transform_batch(batch)
        assert result.batch_size == 3
