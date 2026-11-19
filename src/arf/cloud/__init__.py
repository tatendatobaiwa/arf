"""Cloud over the region: a spatially and temporally correlated field, and the mask it gives each observation."""

from arf.cloud.field import LatentCloudField
from arf.cloud.mask import correlated_cloud_mask, independent_cloud_mask

__all__ = ["LatentCloudField", "correlated_cloud_mask", "independent_cloud_mask"]
