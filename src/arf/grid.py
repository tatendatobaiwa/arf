"""The region as a list of H3 cells.

The region is discretised on the H3 hexagonal grid (Uber Technologies, https://h3geo.org): a cell belongs to the
region when its centre lies inside the region's boundary. This module reproduces the discretisation of Cell 1 of
releases 1.5.3 to 1.5.6.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from arf.errors import ConfigurationError, SpatialDiscretisationError
from arf.regions import Region

#: The H3 resolution of the registered releases: cells of 5.16 km2 on average.
DEFAULT_RESOLUTION = 7


@dataclass(frozen=True)
class RegionGrid:
    """A region turned into H3 cells.

    Attributes:
        cells: The identifiers of the cells, sorted; every per-cell array of the package follows this order.
        resolution: The H3 resolution of the cells.
        site_reports: For a region made of WDPA sites, one report per site (see :func:`arf.boundaries.load_site`);
            ``None`` otherwise.
        outlines: The boundary of the region as GeoJSON-like polygons in (longitude, latitude), for maps. For a
            WDPA region these are the licensed boundaries: they are held in memory only and must never be written
            to an output file.
        outline_source: ``"WDPA"`` or ``"ROI definition"``.
    """

    cells: np.ndarray
    resolution: int
    site_reports: tuple[dict, ...] | None
    outlines: tuple[dict, ...]
    outline_source: str


def sorted_region_cells(cells: Iterable[str]) -> np.ndarray:
    """Return the cells of a region in the order every per-cell array uses: sorted by identifier."""
    return np.array(sorted(set(cells)))


def region_cells_from_run_folder(run_folder: str | Path) -> np.ndarray:
    """Read the cells of the region from ``arf_mosaic_tiles.csv`` of a run, sorted by identifier."""
    table = pd.read_csv(Path(run_folder) / "arf_mosaic_tiles.csv", usecols=["tile"])
    return sorted_region_cells(table["tile"])


def cell_centres(cells: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return the latitude and longitude, in degrees, of the centre of each cell.

    Needs the ``h3`` package, which is imported here so that the rest of the
    package works without it.
    """
    import h3

    centres = np.array([h3.cell_to_latlng(cell) for cell in cells], dtype=np.float64).reshape(-1, 2)
    return centres[:, 0].copy(), centres[:, 1].copy()


def corridor_polygon(waypoints: Sequence[tuple[float, float]], buffer_km: float) -> list[tuple[float, float]]:
    """Return the polygon around a line of waypoints, buffered by a distance on each side.

    The buffer is taken in degrees with a flat-Earth conversion at the mean latitude of the waypoints: one degree of
    latitude is 111 km and one degree of longitude 111 km times the cosine of that latitude, and the smaller of the
    two converts the distance, so the corridor is at least ``buffer_km`` wide in every direction. Needs ``shapely``.

    Args:
        waypoints: The line as (latitude, longitude) pairs in degrees.
        buffer_km: The distance from the line to the edge of the corridor, in kilometres.

    Returns:
        The corners of the corridor as (latitude, longitude) pairs.
    """
    from shapely.geometry import LineString

    mean_latitude = sum(point[0] for point in waypoints) / len(waypoints)
    km_per_degree_latitude = 111.0
    km_per_degree_longitude = 111.0 * math.cos(math.radians(mean_latitude))
    buffer_degrees = buffer_km / min(km_per_degree_latitude, km_per_degree_longitude)
    line = LineString([(longitude, latitude) for latitude, longitude in waypoints])
    corridor = line.buffer(buffer_degrees)
    return [(latitude, longitude) for longitude, latitude in corridor.exterior.coords]


def cells_in_polygon(polygon: Sequence[tuple[float, float]], resolution: int = DEFAULT_RESOLUTION) -> set[str]:
    """Return the H3 cells whose centres lie inside a polygon given as (latitude, longitude) corners. Needs ``h3``."""
    import h3

    return set(h3.polygon_to_cells(h3.LatLngPoly(list(polygon)), resolution))


def _discretise_wdpa_region(region: Region, resolution: int, boundary_dir: str | Path | None) -> RegionGrid:
    """Return the union of the cells of every WDPA site of a region."""
    from arf.boundaries import load_site

    if boundary_dir is None:
        raise ConfigurationError(
            f"region {region.key!r} is defined by WDPA boundaries: give the folder that holds the downloads"
        )
    cells: set[str] = set()
    reports, outlines = [], []
    for site in region.wdpa_sites:
        loaded = load_site(site, boundary_dir, resolution)
        cells |= loaded.cells
        reports.append(loaded.report)
        outlines.extend(loaded.outlines)
    return RegionGrid(sorted_region_cells(cells), resolution, tuple(reports), tuple(outlines), "WDPA")


def discretise_region(
    region: Region, resolution: int = DEFAULT_RESOLUTION, boundary_dir: str | Path | None = None
) -> RegionGrid:
    """Turn a region into H3 cells.

    Args:
        region: The region, defined by WDPA sites, a polygon, or waypoints with a buffer.
        resolution: The H3 resolution.
        boundary_dir: The folder that holds the WDPA downloads; needed only for a region defined by WDPA sites.

    Raises:
        ConfigurationError: If the region has no definition, or a WDPA file is missing or is not the intended site.
        SpatialDiscretisationError: If the region cannot be discretised, has no cell, or the cells of a WDPA site
            do not match the area of its boundary.
    """
    if region.wdpa_sites:
        grid = _discretise_wdpa_region(region, resolution, boundary_dir)
    else:
        if region.polygon is not None:
            polygon = list(region.polygon)
        elif region.waypoints is not None and region.buffer_km is not None:
            polygon = corridor_polygon(region.waypoints, region.buffer_km)
        else:
            raise ConfigurationError(f"region {region.key!r} has no polygon, waypoints or WDPA sites")
        try:
            cells = cells_in_polygon(polygon, resolution)
        except Exception as error:
            raise SpatialDiscretisationError(
                f"could not discretise region {region.key!r} at H3 resolution {resolution}: {error}"
            ) from error
        outline = {"type": "Polygon", "coordinates": [[(longitude, latitude) for latitude, longitude in polygon]]}
        grid = RegionGrid(sorted_region_cells(cells), resolution, None, (outline,), "ROI definition")
    if grid.cells.size == 0:
        raise SpatialDiscretisationError(f"region {region.key!r} has no cell at H3 resolution {resolution}")
    return grid
