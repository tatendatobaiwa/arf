"""A Gaussian random field over the region, the latent variable of the correlated cloud model."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import correlate1d
from scipy.special import ndtri

#: Mean radius of the Earth, km (IUGG).
EARTH_RADIUS_KM = 6371.0088
#: The grid spacing is a sixth of the correlation length, but never more than this.
MAX_GRID_SPACING_KM = 10.0
#: The grid extends beyond the region by this many kernel standard deviations.
PADDING_IN_SIGMAS = 4.5


def gaussian_kernel(sigma_in_grid_cells: float, truncate: float = 4.0) -> np.ndarray:
    """Return a normalised one-dimensional Gaussian kernel, as ``scipy.ndimage.gaussian_filter1d`` builds it."""
    radius = int(truncate * sigma_in_grid_cells + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma_in_grid_cells) ** 2)
    return kernel / kernel.sum()


class LatentCloudField:
    """Unit-variance Gaussian field with correlation ``exp(-d^2 / L^2)`` at distance ``d``.

    The field lives on a local equirectangular plane around the region. White
    noise on a square grid is filtered by a separable Gaussian kernel of standard
    deviation ``L / 2``; the grid is padded and wraps around, so opposite edges
    of the region stay at least ``4.5 L`` apart through the wrap. The value at a
    cell centre is a bilinear interpolation of the four surrounding grid points.
    Interpolation lowers the variance by an amount that depends on where the
    centre falls between grid points; ``thresholds`` corrects for it, so that
    the marginal cloud probability of every cell is exactly the one asked for.

    The length ``L`` is an assumption of the study, varied in its sensitivity
    analysis; see the cloud model in the Methods of the paper.
    """

    def __init__(self, latitude_deg: np.ndarray, longitude_deg: np.ndarray, length_km: float) -> None:
        """Lay out the grid for the cell centres of a region.

        Args:
            latitude_deg: Latitude of each cell centre.
            longitude_deg: Longitude of each cell centre.
            length_km: The e-folding distance ``L`` of the correlation.
        """
        latitude_deg = np.asarray(latitude_deg, dtype=np.float64)
        longitude_deg = np.asarray(longitude_deg, dtype=np.float64)
        mean_latitude, mean_longitude = np.radians(latitude_deg.mean()), np.radians(longitude_deg.mean())
        x_km = EARTH_RADIUS_KM * (np.radians(longitude_deg) - mean_longitude) * np.cos(mean_latitude)
        y_km = EARTH_RADIUS_KM * (np.radians(latitude_deg) - mean_latitude)

        self.length_km = float(length_km)
        sigma_km = self.length_km / 2.0
        self.spacing_km = min(self.length_km / 6.0, MAX_GRID_SPACING_KM)
        padding_km = PADDING_IN_SIGMAS * sigma_km
        x_origin, y_origin = x_km.min() - padding_km, y_km.min() - padding_km
        self.n_columns = int(np.ceil((x_km.max() + padding_km - x_origin) / self.spacing_km)) + 2
        self.n_rows = int(np.ceil((y_km.max() + padding_km - y_origin) / self.spacing_km)) + 2

        # Position of each cell centre in grid units: whole part and fraction.
        x_grid, y_grid = (x_km - x_origin) / self.spacing_km, (y_km - y_origin) / self.spacing_km
        self._column, self._row = np.floor(x_grid).astype(np.int64), np.floor(y_grid).astype(np.int64)
        fraction_x, fraction_y = x_grid - self._column, y_grid - self._row

        self._kernel = gaussian_kernel(sigma_km / self.spacing_km)
        kernel_energy = float((self._kernel**2).sum())
        # The standard deviation of noise filtered along both axes is the kernel energy.
        self._field_sd = kernel_energy
        neighbour_correlation = float((self._kernel[:-1] * self._kernel[1:]).sum() / kernel_energy)
        variance_x = (1 - fraction_x) ** 2 + fraction_x**2 + 2 * (1 - fraction_x) * fraction_x * neighbour_correlation
        variance_y = (1 - fraction_y) ** 2 + fraction_y**2 + 2 * (1 - fraction_y) * fraction_y * neighbour_correlation
        self._interpolated_sd = np.sqrt(variance_x * variance_y)

        self._weight_00 = (1 - fraction_x) * (1 - fraction_y)
        self._weight_10 = fraction_x * (1 - fraction_y)
        self._weight_01 = (1 - fraction_x) * fraction_y
        self._weight_11 = fraction_x * fraction_y

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one independent unit-variance field on the grid."""
        noise = rng.standard_normal((self.n_rows, self.n_columns))
        noise = correlate1d(noise, self._kernel, axis=0, mode="wrap")
        noise = correlate1d(noise, self._kernel, axis=1, mode="wrap")
        return noise / self._field_sd

    def at_cells(self, field: np.ndarray, cell_index: np.ndarray) -> np.ndarray:
        """Interpolate a field drawn by ``draw`` at the centres of the given cells."""
        column, row = self._column[cell_index], self._row[cell_index]
        return (
            self._weight_00[cell_index] * field[row, column]
            + self._weight_10[cell_index] * field[row, column + 1]
            + self._weight_01[cell_index] * field[row + 1, column]
            + self._weight_11[cell_index] * field[row + 1, column + 1]
        )

    def thresholds(self, cloud_probability: np.ndarray) -> np.ndarray:
        """Return, per cell, the field value above which the cell is cloudy with the given probability."""
        return ndtri(1.0 - np.asarray(cloud_probability, dtype=np.float64)) * self._interpolated_sd
