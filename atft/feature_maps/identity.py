"""Identity feature map (pass-through)."""
from __future__ import annotations

from atft.core.types import PointCloud, PointCloudBatch


class IdentityMap:
    """Pass-through feature map for pre-processed data."""

    def transform(self, cloud: PointCloud) -> PointCloud:
        return cloud

    def transform_batch(self, batch: PointCloudBatch) -> PointCloudBatch:
        return batch
