"""Protected-area boundaries from the World Database on Protected Areas (WDPA), turned into H3 cells.

The WDPA licence does not allow the data to be redistributed, so no boundary file is part of this package and none
is ever written by it. Download each site from ``https://www.protectedplanet.net/<site_id>`` as a shapefile and put
the download in one folder, as the zip or extracted, at any depth; :func:`load_site` finds it by its site
identifier. Cite the data as: UNEP-WCMC and IUCN (year), Protected Planet: The World Database on Protected Areas
(WDPA) [On-line], month and year of the version downloaded, Cambridge, UK: UNEP-WCMC and IUCN.

This module reproduces the WDPA part of Cell 1 of releases 1.5.3 to 1.5.6. Needs ``h3`` and ``pyshp``.
"""

from __future__ import annotations

import glob
import hashlib
import io
import logging
import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from arf.errors import ConfigurationError, SpatialDiscretisationError
from arf.regions import WdpaSite

logger = logging.getLogger(__name__)

#: The cells' area may differ from the WDPA polygon area (``GIS_AREA``) by this fraction ...
AREA_TOLERANCE = 0.05
#: ... or by this many cells' worth of area, whichever is larger. A small site is quantised to whole cells (one
#: cell of resolution 7 is 5.16 km2, 3.6% of a 143 km2 site), so a fixed 5% rejects correct boundaries. The value is
#: empirical: for 400 random irregular polygons of 143 km2 the largest error was 4.6 cells (release 1.5.x
#: validation, 8 October 2026). Above about 3,600 km2 the 5% term is the larger one.
CELL_TOLERANCE = 5
#: The file's reported area (``REP_AREA``) may differ from the value on the site's page by this fraction.
REPORTED_AREA_TOLERANCE = 0.005

_MONTHS = {
    name: number
    for number, name in enumerate(
        ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"), start=1
    )
}


@dataclass(frozen=True)
class SiteCells:
    """The H3 cells of one WDPA site.

    Attributes:
        cells: The cells whose centres lie inside the site's polygons.
        report: What was read and checked, with the keys the releases write to ``arf_cell1_validation.json``
            under ``wdpa_boundaries``.
        outlines: The boundary polygons as GeoJSON-like mappings, for drawing maps. They are held in memory only
            and must never be written to an output file (WDPA licence).
    """

    cells: frozenset[str]
    report: dict
    outlines: tuple[dict, ...]


def wdpa_release(path: str | Path) -> tuple[int, int]:
    """Return the (year, month) of the WDPA release named in a file name such as ``..._Oct2026_...``.

    Returns (0, 0) when the name holds no release.
    """
    match = re.search(r"_([A-Z][a-z]{2})(\d{4})_", os.path.basename(path))
    if not match:
        return (0, 0)
    return (int(match.group(2)), _MONTHS.get(match.group(1), 0))


def _unpack_site(archive: zipfile.ZipFile, site_id: int, destination: str) -> None:
    """Write the polygon shapefile members of one site from an archive, and from archives inside it."""
    for member in archive.namelist():
        base = os.path.basename(member)
        if member.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(archive.read(member))) as inner:
                _unpack_site(inner, site_id, os.path.join(destination, os.path.splitext(base)[0]))
        elif f"_{site_id}_shp-polygons." in base:
            os.makedirs(destination, exist_ok=True)
            with open(os.path.join(destination, base), "wb") as handle:
                handle.write(archive.read(member))


def find_site_files(site_id: int, boundary_dir: str | Path, unpack_dir: str | Path) -> list[str]:
    """Return every polygon shapefile part of a WDPA site under a folder, at any depth.

    Protected Planet splits a download into numbered parts (``..._shp_0``, ``_shp_1``, ...) inside the download
    folder, and may deliver them as zips inside a zip. Zips whose name holds the site identifier are unpacked (only
    this site's members) into ``unpack_dir``. If several monthly releases are present, only the parts of the newest
    are returned. A part present both extracted and inside its zip is returned once, since reading it twice would
    count the site's area twice.

    Args:
        site_id: The WDPA site identifier.
        boundary_dir: The folder that holds the downloads.
        unpack_dir: A folder for the members taken out of zips. The caller removes it.

    Returns:
        The paths of the ``.shp`` files, sorted; empty if the site is not there.
    """
    boundary_dir = str(boundary_dir)
    target = f"_{site_id}_shp-polygons.shp"
    found = [
        path for path in glob.glob(os.path.join(boundary_dir, "**", "*.shp"), recursive=True) if path.endswith(target)
    ]
    archives = sorted(
        path
        for path in glob.glob(os.path.join(boundary_dir, "**", "*.zip"), recursive=True)
        if f"_{site_id}_" in os.path.basename(path)
    )
    for number, archive_path in enumerate(archives):
        destination = os.path.join(str(unpack_dir), f"wdpa_{site_id}_{number}")
        os.makedirs(destination, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            _unpack_site(archive, site_id, destination)
        found += [
            path
            for path in glob.glob(os.path.join(destination, "**", "*.shp"), recursive=True)
            if path.endswith(target)
        ]
    if not found:
        return []
    newest = max(wdpa_release(path) for path in found)
    unique: dict[str, str] = {}
    for path in sorted(path for path in found if wdpa_release(path) == newest):
        with open(path, "rb") as handle:
            unique.setdefault(hashlib.sha256(handle.read()).hexdigest(), path)
    return sorted(unique.values())


def sites_present(boundary_dir: str | Path) -> list[int]:
    """Return the identifiers of the WDPA sites with an extracted polygon shapefile anywhere under a folder."""
    identifiers = set()
    for path in glob.glob(os.path.join(str(boundary_dir), "**", "*_shp-polygons.shp"), recursive=True):
        match = re.search(r"_(\d+)_shp-polygons\.shp$", path)
        if match:
            identifiers.add(int(match.group(1)))
    return sorted(identifiers)


def _missing_site_error(site: WdpaSite, boundary_dir: str | Path) -> ConfigurationError:
    """Return the error for a site with no polygon file, saying whether the WDPA holds only a point for it."""
    boundary_dir = str(boundary_dir)
    points = glob.glob(os.path.join(boundary_dir, "**", f"*_{site.site_id}_shp-points.shp"), recursive=True)
    if points:
        # WDPA/WDOECM Manual 1.6, p. 15: where a data provider supplied no boundary, the WDPA stores the centremost
        # point of the site instead. A point has no shape, so the site cannot be turned into cells.
        return ConfigurationError(
            f"WDPA site {site.site_id} ({site.name}) has no boundary polygon in the WDPA: the download contains "
            f"only a point record ({os.path.basename(points[0])}). The WDPA stores a point where the data provider "
            f"supplied no boundary (WDPA/WDOECM Manual 1.6, p. 15), so the shape of this site is unknown and it "
            f"cannot be turned into cells. Choose a site whose download contains '_shp-polygons' files."
        )
    present = sites_present(boundary_dir) if os.path.isdir(boundary_dir) else []
    return ConfigurationError(
        f"WDPA boundary for site {site.site_id} ({site.name}) not found anywhere under {boundary_dir}. Download it "
        f"as a shapefile from https://www.protectedplanet.net/{site.site_id} and place the download (zip, or "
        f"extracted folders at any depth) in that folder. WDPA sites found there: {present if present else 'none'}."
    )


@dataclass
class _SiteRecords:
    """What the records of a site's shapefile parts hold, accumulated over the parts."""

    cells: set[str]
    outlines: list[dict]
    gis_area_km2: float
    designations: set[str]
    names: set[str]
    reported_areas_km2: set[float]


def _read_site_records(site: WdpaSite, paths: list[str], resolution: int) -> _SiteRecords:
    """Read every record of every part of a site: its attributes, and the cells inside its polygons."""
    import h3
    import shapefile  # the pyshp package

    read = _SiteRecords(set(), [], 0.0, set(), set(), set())
    for path in paths:
        with shapefile.Reader(path) as reader:
            fields = [description[0] for description in reader.fields[1:]]
            for shape_record in reader.iterShapeRecords():
                attributes = dict(zip(fields, shape_record.record, strict=False))
                record_id = int(attributes.get("SITE_ID", attributes.get("WDPAID", -1)))
                if record_id != int(site.site_id):
                    raise ConfigurationError(
                        f"{os.path.basename(paths[0])} holds WDPA site {record_id}, expected {site.site_id}."
                    )
                geometry = shape_record.shape.__geo_interface__
                read.designations.add(str(attributes.get("DESIG_ENG", "")))
                read.names.add(str(attributes.get("NAME_ENG", attributes.get("NAME", ""))))
                read.gis_area_km2 += float(attributes.get("GIS_AREA", 0.0))
                read.reported_areas_km2.add(round(float(attributes.get("REP_AREA", 0.0)), 4))
                read.cells |= set(h3.geo_to_cells(geometry, resolution))
                read.outlines.append(geometry)
    return read


def _check_identity(site: WdpaSite, read: _SiteRecords) -> None:
    """Check that the file is the intended site: by its reported area, then by its designation, when given."""
    if site.expected_rep_area_km2:
        reported = max(read.reported_areas_km2) if read.reported_areas_km2 else 0.0
        if abs(reported - site.expected_rep_area_km2) / site.expected_rep_area_km2 > REPORTED_AREA_TOLERANCE:
            raise ConfigurationError(
                f"WDPA site {site.site_id}: file reports REP_AREA {reported:,.2f} km2, but "
                f"{site.source_url or 'the site page'} shows {site.expected_rep_area_km2:,.2f} km2. This is "
                f"probably not {site.name}."
            )
    designations = sorted(read.designations)
    if site.designation_contains is None:
        logger.info("WDPA %s: designation %s (recorded, not specified in advance).", site.site_id, designations)
    elif not any(site.designation_contains.lower() in designation.lower() for designation in designations):
        raise ConfigurationError(
            f"WDPA site {site.site_id} is designated {designations}, but the region requires "
            f"'{site.designation_contains}' ({site.name}). Check the site identifier: Protected Planet lists "
            f"several overlapping designations under similar names."
        )


def load_site(site: WdpaSite, boundary_dir: str | Path, resolution: int) -> SiteCells:
    """Read the boundary of one WDPA site, check that it is the intended site, and return its H3 cells.

    A cell belongs to the site when its centre lies inside one of the site's polygons (``h3.geo_to_cells``). The
    summed area of the cells must match the WDPA polygon area within :data:`AREA_TOLERANCE` or
    :data:`CELL_TOLERANCE` cells, whichever is larger.

    Args:
        site: The site and the values that identify it.
        boundary_dir: The folder that holds the WDPA downloads.
        resolution: The H3 resolution of the cells.

    Raises:
        ConfigurationError: If no polygon file is found, or the file is not the intended site.
        SpatialDiscretisationError: If the area of the cells does not match the area of the boundary.
    """
    import h3

    with tempfile.TemporaryDirectory(prefix=f"wdpa_{site.site_id}_") as unpack_dir:
        paths = find_site_files(site.site_id, boundary_dir, unpack_dir)
        if not paths:
            raise _missing_site_error(site, boundary_dir)
        read = _read_site_records(site, paths, resolution)
    _check_identity(site, read)
    gis_area = read.gis_area_km2
    cell_area = float(sum(h3.cell_area(cell, unit="km^2") for cell in read.cells))
    deviation = (cell_area - gis_area) / gis_area if gis_area else float("nan")
    mean_cell_km2 = cell_area / len(read.cells) if read.cells else 0.0
    tolerance = max(AREA_TOLERANCE, CELL_TOLERANCE * mean_cell_km2 / gis_area if gis_area else 0.0)
    report = {
        "site_id": int(site.site_id),
        "name": sorted(read.names),
        "designation": sorted(read.designations),
        "file": os.path.basename(paths[0]),
        "n_parts": len(paths),
        "wdpa_gis_area_km2": gis_area,
        "expected_gis_area_km2": site.expected_gis_area_km2,
        "tiled_area_km2": cell_area,
        "area_deviation_pct": 100 * deviation,
        "n_tiles": len(read.cells),
        "area_tolerance_pct": 100 * tolerance,
        "wdpa_rep_area_km2": max(read.reported_areas_km2) if read.reported_areas_km2 else None,
        "source_url": site.source_url,
        "accessed": site.accessed,
    }
    if not abs(deviation) <= tolerance:  # written so that a deviation that is not a number is rejected
        raise SpatialDiscretisationError(
            f"WDPA site {site.site_id}: area of the cells {cell_area:,.0f} km2 differs from WDPA GIS_AREA "
            f"{gis_area:,.0f} km2 by {100 * deviation:+.1f}% (tolerance {100 * tolerance:.1f}%)."
        )
    expected = site.expected_gis_area_km2
    if expected and abs(gis_area - expected) / expected > AREA_TOLERANCE:
        logger.warning(
            "WDPA site %s: GIS_AREA %.0f km2 differs from the expected %.0f km2; the boundary may have been "
            "revised in this WDPA release.",
            site.site_id,
            gis_area,
            expected,
        )
    return SiteCells(frozenset(read.cells), report, tuple(read.outlines))
