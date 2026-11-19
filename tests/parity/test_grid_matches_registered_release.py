"""arf.regions, arf.boundaries and arf.grid against the registered releases themselves.

The region definitions and the discretisation statements of Cell 1 of releases 1.5.3 and 1.5.6 are executed unchanged
(``registered.py``) and compared with release 2: the regions as data, the cells of a polygon and of a corridor, and the
cells and the report of regions read from shapefiles laid out as Protected Planet delivers them (split parts, zips
inside a zip, a part present twice, two monthly releases).

The shapefiles are written by the tests and hold invented rectangles: no WDPA data is in this repository. The check
on the real boundaries is ``tools/check_grid_on_runs.py``, run where the downloads are.
"""

import dataclasses
import io
import logging
import zipfile

import pytest

from arf import boundaries
from arf.errors import ConfigurationError, SpatialDiscretisationError
from arf.grid import corridor_polygon, discretise_region
from arf.regions import REGIONS, Region, WdpaSite

h3 = pytest.importorskip("h3")
shapefile = pytest.importorskip("shapefile")
pytest.importorskip("shapely")
from registered import run_cell1_grid  # noqa: E402

RELEASE_NAMES = ("1.5.3", "1.5.6")


def rectangle(south, west, north, east):
    """One clockwise ring of (longitude, latitude), as a shapefile stores an outer boundary."""
    return [(west, south), (west, north), (east, north), (east, south), (west, south)]


def cell_area_km2(rings, resolution=7):
    """The area of the cells inside a polygon with the given rings: what GIS_AREA is set to in these files."""
    shape = {"type": "Polygon", "coordinates": [[list(point) for point in ring] for ring in rings]}
    return float(sum(h3.cell_area(cell, unit="km^2") for cell in h3.geo_to_cells(shape, resolution)))


def write_part(folder, release_tag, site_id, polygons, designation, name, gis_areas, reported_area, id_field="SITE_ID"):
    """Write one polygon shapefile part of a site, named and placed as in a Protected Planet download."""
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"WDPA_WDOECM_{release_tag}_Public_{site_id}_shp-polygons"
    with shapefile.Writer(str(stem), shapeType=shapefile.POLYGON) as writer:
        writer.field(id_field, "N", 12, 0)
        writer.field("NAME_ENG", "C", 80)
        writer.field("DESIG_ENG", "C", 80)
        writer.field("GIS_AREA", "F", 20, 6)
        writer.field("REP_AREA", "F", 20, 6)
        for rings, gis_area in zip(polygons, gis_areas, strict=True):
            writer.poly(rings)
            writer.record(site_id, name, designation, gis_area, reported_area)
    return stem.with_suffix(".shp")


def download_folder(root, release_tag, site_id, part):
    return (
        root
        / f"WDPA_WDOECM_{release_tag}_Public_{site_id}_shp"
        / f"WDPA_WDOECM_{release_tag}_Public_{site_id}_shp_{part}"
    )


CKGR_PARTS = [  # three parts, the second with a hole
    [rectangle(-22.6, 23.0, -21.6, 24.0)],
    [rectangle(-21.6, 23.0, -21.0, 24.0), rectangle(-21.4, 23.3, -21.2, 23.6)[::-1]],
    [rectangle(-22.6, 24.0, -21.0, 24.6)],
]
OKAVANGO = [rectangle(-19.8, 22.4, -19.0, 23.4)]


def write_ckgr_okavango(root, release_tag="Oct2026", gis_factor=1.0):
    for part, rings in enumerate(CKGR_PARTS):
        write_part(
            download_folder(root, release_tag, 7510, part),
            release_tag,
            7510,
            [rings],
            "Game Reserve",
            "Central Kalahari",
            [gis_factor * cell_area_km2(rings)],
            52800.0,
        )
    write_part(
        download_folder(root, release_tag, 555577555, 0),
        release_tag,
        555577555,
        [OKAVANGO],
        "World Heritage Site (natural or mixed)",
        "Okavango Delta",
        [gis_factor * cell_area_km2(OKAVANGO)],
        20235.99,
    )


def compare(release, key, boundary_dir, resolution=7):
    """Discretise a region with the registered release and with release 2, and require the same result."""
    registered = run_cell1_grid(release, key, resolution, boundary_dir)
    grid = discretise_region(REGIONS[key], resolution, boundary_dir)
    assert set(grid.cells) == set(registered["target_tiles"])
    assert list(grid.cells) == sorted(registered["target_tiles"])
    assert grid.outline_source == registered["ROI_OUTLINE_SOURCE"]
    assert list(grid.outlines) == registered["ROI_OUTLINE_GEOMS"]
    if registered["roi_boundary_report"] is None:
        assert grid.site_reports is None
    else:
        assert list(grid.site_reports) == registered["roi_boundary_report"]  # every key and value, floats exactly
    return grid


@pytest.mark.parametrize("release", RELEASE_NAMES)
def test_regions_are_the_registered_definitions(release):
    registered = run_cell1_grid(release)["ROI_DEFINITIONS"]
    assert list(REGIONS) == list(registered)
    for key, region in REGIONS.items():
        rebuilt = {"short_name": region.short_name, "description": region.description}
        if region.polygon is not None:
            rebuilt["roi_coords"] = list(region.polygon)
        if region.waypoints is not None:
            rebuilt["waypoints_latlon"] = list(region.waypoints)
            rebuilt["buffer_km"] = region.buffer_km
        if region.historical_epoch is not None:
            rebuilt["historical_epoch"] = region.historical_epoch
        if region.wdpa_sites:
            rebuilt["wdpa_sites"] = {}
            for site in region.wdpa_sites:
                spec = {"name": site.name, "designation_contains": site.designation_contains}
                optional = ("expected_gis_area_km2", "expected_rep_area_km2", "source_url", "accessed")
                spec.update({name: getattr(site, name) for name in optional if getattr(site, name) is not None})
                rebuilt["wdpa_sites"][site.site_id] = spec
        rebuilt["ground_stations"] = {
            name: {"lat": station.latitude_deg, "lng": station.longitude_deg, "alt_m": station.altitude_m}
            for name, station in region.ground_stations.items()
        }
        rebuilt["tle_source"] = region.element_source
        assert rebuilt == registered[key], key
        assert [site.site_id for site in region.wdpa_sites] == list(registered[key].get("wdpa_sites", {}))


@pytest.mark.parametrize("release", RELEASE_NAMES)
def test_tolerances_are_the_registered_ones(release):
    registered = run_cell1_grid(release)
    assert registered["WDPA_AREA_TOLERANCE"] == boundaries.AREA_TOLERANCE
    assert registered["WDPA_TILE_TOLERANCE"] == boundaries.CELL_TOLERANCE
    assert registered["WDPA_REP_AREA_TOLERANCE"] == boundaries.REPORTED_AREA_TOLERANCE


@pytest.mark.parametrize("release", RELEASE_NAMES)
def test_corridor_region(release):
    registered = run_cell1_grid(release)
    region = REGIONS["nepal_trishuli_corridor"]
    assert corridor_polygon(region.waypoints, region.buffer_km) == registered["_buffer_waypoints_to_polygon"](
        list(region.waypoints), region.buffer_km
    )
    for resolution in (7, 8):
        grid = compare(release, "nepal_trishuli_corridor", None, resolution)
    assert grid.cells.size > 500


@pytest.mark.parametrize("release", RELEASE_NAMES)
def test_polygon_region(release):
    grid = compare(release, "botswana_full", None, resolution=5)
    assert grid.cells.size > 3000


@pytest.mark.parametrize("release", RELEASE_NAMES)
def test_wdpa_region_from_extracted_parts(release, tmp_path):
    write_ckgr_okavango(tmp_path / "ARF_boundaries" / "downloads")
    grid = compare(release, "ckgr_okavango", tmp_path / "ARF_boundaries")
    ckgr, okavango = grid.site_reports
    assert (ckgr["n_parts"], okavango["n_parts"]) == (3, 1)
    assert ckgr["n_tiles"] + okavango["n_tiles"] == grid.cells.size > 4000
    hole = set(h3.geo_to_cells({"type": "Polygon", "coordinates": [rectangle(-21.4, 23.3, -21.2, 23.6)]}, 7))
    assert hole and not hole & set(grid.cells)


@pytest.mark.parametrize("release", RELEASE_NAMES)
def test_wdpa_region_from_a_zip_of_zips_with_one_part_also_extracted(release, tmp_path):
    staging, root = tmp_path / "staging", tmp_path / "ARF_boundaries"
    root.mkdir()
    for site_id, parts, designation in (
        (7510, CKGR_PARTS[:2], "Game Reserve"),
        (555577555, [OKAVANGO], "World Heritage Site"),
    ):
        outer_bytes = io.BytesIO()
        with zipfile.ZipFile(outer_bytes, "w") as outer:
            for part, rings in enumerate(parts):
                folder = staging / str(site_id) / str(part)
                write_part(folder, "Oct2026", site_id, [rings], designation, "A site", [cell_area_km2(rings)], 1.0)
                inner_bytes = io.BytesIO()
                with zipfile.ZipFile(inner_bytes, "w") as inner:
                    for file in sorted(folder.iterdir()):
                        inner.write(file, file.name)
                    inner.writestr("WDPA_sources.txt", "not a boundary")
                outer.writestr(f"WDPA_WDOECM_Oct2026_Public_{site_id}_shp_{part}.zip", inner_bytes.getvalue())
        (root / f"WDPA_WDOECM_Oct2026_Public_{site_id}_shp.zip").write_bytes(outer_bytes.getvalue())
    # The first CKGR part is also there extracted: it must be read once, not twice.
    rings = CKGR_PARTS[0]
    write_part(root / "extracted", "Oct2026", 7510, [rings], "Game Reserve", "A site", [cell_area_km2(rings)], 1.0)
    grid = compare(release, "ckgr_okavango", root)
    assert grid.site_reports[0]["n_parts"] == 2
    assert abs(grid.site_reports[0]["area_deviation_pct"]) < 1e-6  # the area is stored to six decimals


@pytest.mark.parametrize("release", RELEASE_NAMES)
def test_newest_wdpa_release_is_used(release, tmp_path):
    root = tmp_path / "ARF_boundaries"
    write_ckgr_okavango(root, "Oct2026")
    older = [rectangle(-22.0, 23.0, -21.8, 23.2)]
    for site_id, designation in ((7510, "Game Reserve"), (555577555, "World Heritage Site")):
        folder = download_folder(root, "Sep2026", site_id, 0)
        write_part(folder, "Sep2026", site_id, [older], designation, "Old", [cell_area_km2(older)], 1.0)
    with_both = compare(release, "ckgr_okavango", root)
    assert with_both.cells.size > 4000
    assert boundaries.wdpa_release("x/WDPA_WDOECM_Sep2026_Public_7510_shp-polygons.shp") == (2026, 9)
    assert boundaries.wdpa_release("no_release.shp") == (0, 0)


@pytest.mark.parametrize("release", RELEASE_NAMES)
@pytest.mark.parametrize(
    ("key", "site_id", "rings", "reported"),
    [
        ("lower_zambezi", 7962, rectangle(-15.9, 29.1, -15.3, 30.2), 4143.0),
        ("lower_zambezi", 7962, rectangle(-15.9, 29.1, -15.3, 30.2), 4160.0),  # inside the 0.5% on the reported area
        ("karoo_nature_reserve", 555563860, rectangle(-32.30, 24.45, -32.20, 24.60), 143.15),  # a few dozen cells
    ],
)
def test_wdpa_region_identified_by_reported_area(release, tmp_path, key, site_id, rings, reported):
    folder = download_folder(tmp_path, "Oct2026", site_id, 0)
    # GIS_AREA 3% below the area of the cells: inside the tolerance for both sites.
    write_part(
        folder, "Oct2026", site_id, [[rings]], "National Park", "A site", [0.97 * cell_area_km2([rings])], reported
    )
    grid = compare(release, key, tmp_path)
    assert grid.site_reports[0]["wdpa_rep_area_km2"] == reported


def test_small_site_is_accepted_on_the_cell_tolerance_only(tmp_path):
    rings = rectangle(-32.30, 24.45, -32.20, 24.60)
    area = cell_area_km2([rings])
    folder = download_folder(tmp_path, "Oct2026", 555563860, 0)
    # GIS_AREA 10% below the area of the cells: outside 5%, inside five cells' worth for a site this small.
    write_part(folder, "Oct2026", 555563860, [[rings]], "Nature Reserve", "Karoo", [area / 1.10], 143.15)
    for release in RELEASE_NAMES:
        report = compare(release, "karoo_nature_reserve", tmp_path).site_reports[0]
        assert report["area_deviation_pct"] > 100 * boundaries.AREA_TOLERANCE
        assert report["area_tolerance_pct"] > report["area_deviation_pct"]


def both_raise(release, key, boundary_dir, error):
    """Require the registered release and release 2 to reject the same input with the same kind of error."""
    with pytest.raises(error):
        run_cell1_grid(release, key, 7, boundary_dir)
    with pytest.raises(error) as raised:
        discretise_region(REGIONS[key], 7, boundary_dir)
    return str(raised.value)


@pytest.mark.parametrize("release", RELEASE_NAMES)
def test_rejected_inputs(release, tmp_path):
    rings = rectangle(-15.9, 29.1, -15.3, 30.2)
    area = cell_area_km2([rings])

    def site_folder(case):
        return download_folder(tmp_path / case, "Oct2026", 7962, 0)

    (tmp_path / "empty").mkdir()
    assert "protectedplanet.net/7962" in both_raise(release, "lower_zambezi", tmp_path / "empty", ConfigurationError)
    assert "not found" in both_raise(release, "lower_zambezi", tmp_path / "no_such_folder", ConfigurationError)

    folder = site_folder("points")
    folder.mkdir(parents=True)
    with shapefile.Writer(
        str(folder / "WDPA_WDOECM_Oct2026_Public_7962_shp-points"), shapeType=shapefile.POINT
    ) as writer:
        writer.field("SITE_ID", "N", 12, 0)
        writer.point(29.5, -15.6)
        writer.record(7962)
    assert "only a point record" in both_raise(release, "lower_zambezi", tmp_path / "points", ConfigurationError)

    write_part(site_folder("reported"), "Oct2026", 7962, [[rings]], "National Park", "A site", [area], 4200.0)
    assert "REP_AREA" in both_raise(release, "lower_zambezi", tmp_path / "reported", ConfigurationError)

    write_part(site_folder("other_site"), "Oct2026", 7962, [[rings]], "National Park", "A site", [area], 4143.0)
    renamed = site_folder("other_site")
    with shapefile.Writer(
        str(renamed / "WDPA_WDOECM_Oct2026_Public_7962_shp-polygons"), shapeType=shapefile.POLYGON
    ) as w:
        w.field("SITE_ID", "N", 12, 0)
        w.poly([rings])
        w.record(1234)
    assert "holds WDPA site 1234" in both_raise(release, "lower_zambezi", tmp_path / "other_site", ConfigurationError)

    write_part(site_folder("area"), "Oct2026", 7962, [[rings]], "National Park", "A site", [1.2 * area], 4143.0)
    assert "differs from WDPA GIS_AREA" in both_raise(
        release, "lower_zambezi", tmp_path / "area", SpatialDiscretisationError
    )

    write_part(site_folder("no_area"), "Oct2026", 7962, [[rings]], "National Park", "A site", [0.0], 4143.0)
    both_raise(release, "lower_zambezi", tmp_path / "no_area", SpatialDiscretisationError)

    # A Ramsar file supplied where the World Heritage Site is required.
    for site_id, designation in (
        (7510, "Game Reserve"),
        (555577555, "Ramsar Site, Wetland of International Importance"),
    ):
        folder = download_folder(tmp_path / "designation", "Oct2026", site_id, 0)
        write_part(folder, "Oct2026", site_id, [OKAVANGO], designation, "A site", [cell_area_km2(OKAVANGO)], 1.0)
    assert "World Heritage" in both_raise(release, "ckgr_okavango", tmp_path / "designation", ConfigurationError)


@pytest.mark.parametrize("release", RELEASE_NAMES)
def test_older_files_with_the_wdpaid_field(release, tmp_path):
    rings = rectangle(-15.9, 29.1, -15.3, 30.2)
    folder = download_folder(tmp_path, "Oct2026", 7962, 0)
    write_part(
        folder, "Oct2026", 7962, [[rings]], "National Park", "A site", [cell_area_km2([rings])], 4143.0, "WDPAID"
    )
    compare(release, "lower_zambezi", tmp_path)


def test_revised_boundary_is_reported_not_rejected(tmp_path, caplog):
    write_ckgr_okavango(tmp_path)  # the invented rectangles are far smaller than the expected GIS areas
    with caplog.at_level(logging.WARNING, logger="arf.boundaries"):
        discretise_region(REGIONS["ckgr_okavango"], 7, tmp_path)
    assert sum("may have been revised" in record.getMessage() for record in caplog.records) == 2


def test_wdpa_region_needs_a_boundary_folder_and_a_region_needs_a_definition():
    with pytest.raises(ConfigurationError, match="folder that holds the downloads"):
        discretise_region(REGIONS["ckgr_okavango"])
    with pytest.raises(ConfigurationError, match="no polygon"):
        discretise_region(Region("nowhere", "Nowhere", "Not defined"))
    tiny = dataclasses.replace(REGIONS["botswana_full"], polygon=((0.0, 0.0), (0.0, 1e-6), (1e-6, 0.0)))
    with pytest.raises(SpatialDiscretisationError, match="no cell"):
        discretise_region(tiny)
    assert WdpaSite(1, "x").designation_contains is None
