"""arf.access against the registered releases themselves, with Orekit.

The fleet construction of Cell 2 and the whole of Cells 4 and 5 of releases 1.5.3 and 1.5.6 are executed unchanged
(``registered.py``) on invented element sets, and release 2 computes the same from the same inputs. Required equal,
to the last bit: the altitude of every satellite at the start of the window, the fleet's median altitude and orbital
period, the search radii, the access events with every strip sample, the downlink contacts, the refined events (the
table every later step reads) and the counts of what was dropped.

Orekit runs with a leap-second table only (``orekit_context.py``), so the numbers are not those of a real run; both
programs run in the same context. The tests are skipped where Orekit is not installed.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest
from orekit_context import start_minimal_orekit
from registered import compare_with_registered, run_cell2_fleet_functions, run_cells_2_4_5
from test_fleet_matches_registered_release import omm, to_alpha5

from arf.access import Imager, RegionIndex, compute_access
from arf.fleet import (
    ACQUISITION_EXCLUSIONS,
    elements_as_of,
    merge_groups,
    mission_table,
    operational_sample,
    pinned_selection,
    select_fleet,
)
from arf.regions import GroundStation

h3 = pytest.importorskip("h3")

BOTSWANA_STATIONS = {
    "BIUST": {"lat": -22.59, "lng": 27.14, "alt_m": 994.0},
    "Kasane": {"lat": -17.80, "lng": 25.15, "alt_m": 930.0},
    "Kebonye": {"lat": -24.84, "lng": 25.84, "alt_m": 994.0},
}
FAR_STATION = {"Svalbard": {"lat": 78.23, "lng": 15.39, "alt_m": 450.0}}  # contacts every orbit, never over the region
# About 330 km by 390 km around the Okavango Delta and the northern Kalahari, in H3 resolution 6 (2,896 cells).
REGION_CELLS = sorted(
    h3.geo_to_cells(
        {
            "type": "Polygon",
            "coordinates": [[(22.0, -22.5), (22.0, -19.0), (25.0, -19.0), (25.0, -22.5), (22.0, -22.5)]],
        },
        6,
    )
)


@pytest.fixture(scope="module", autouse=True)
def orekit():
    start_minimal_orekit()


def invented_real_fleet(epoch, node_shift=0.0, extra_sentinel_2a=0):
    """Sun-synchronous satellites with a morning descending node, as the real fleets fly, with invented phases.

    ``node_shift`` turns every orbital plane, to keep the morning node in another month. ``extra_sentinel_2a`` adds
    objects that share the name SENTINEL-2A at other phases, so that at least one passes over the region in daylight.
    """
    rng = np.random.default_rng(1)
    sentinel = {"mean_motion": 14.30818, "INCLINATION": 98.57, "epoch": epoch}
    landsat = {"mean_motion": 14.571, "INCLINATION": 98.2, "epoch": epoch}
    records = [
        omm(40697, "SENTINEL-2A", RA_OF_ASC_NODE=290.0, MEAN_ANOMALY=10.0, **sentinel),
        omm(42063, "SENTINEL-2B", RA_OF_ASC_NODE=290.0, MEAN_ANOMALY=190.0, **sentinel),
        omm(60989, "SENTINEL-2C", RA_OF_ASC_NODE=291.0, MEAN_ANOMALY=100.0, **sentinel),
        omm(39084, "LANDSAT 8", RA_OF_ASC_NODE=288.0, MEAN_ANOMALY=50.0, **landsat),
        omm(49260, "LANDSAT 9", RA_OF_ASC_NODE=288.0, MEAN_ANOMALY=230.0, **landsat),
        omm(27424, "AQUA", epoch=epoch),  # no mission entry
    ]
    for index in range(14):
        records.append(
            omm(
                50000 + index if index else 123456,  # one catalogue number above 99999
                f"FLOCK 4X-{index}",
                epoch=epoch,
                mean_motion=float(rng.uniform(15.1, 15.3)),
                eccentricity=float(rng.uniform(0, 0.002)),
                INCLINATION=97.4,
                RA_OF_ASC_NODE=float(rng.uniform(285, 300)),
                MEAN_ANOMALY=float(rng.uniform(0, 360)),
                group="planet",
            )
        )
    for index in range(extra_sentinel_2a):
        anomaly = 10.0 + 360.0 * (index + 1) / (extra_sentinel_2a + 1)
        records.append(omm(70000 + index, "SENTINEL-2A", RA_OF_ASC_NODE=290.0, MEAN_ANOMALY=anomaly, **sentinel))
    for record in records:
        record["RA_OF_ASC_NODE"] = (record["RA_OF_ASC_NODE"] + node_shift) % 360.0
    return records


def checksum(line):
    return str(sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line) % 10)


def valid_tle(number, day_of_year, raan, mean_anomaly, mean_motion=15.06, eccentricity=0.0001, inclination=53.05):
    """One invented element set that Orekit's TLE parser accepts: fields in their columns, with checksums."""
    field = to_alpha5(number)
    line1 = f"1 {field}U {'19074A':<8s} 26{day_of_year:012.8f}  .00001234  00000-0  12345-3 0  999"
    ecc = f"{round(eccentricity * 1e7):07d}"
    line2 = (
        f"2 {field} {inclination:8.4f} {raan:8.4f} {ecc} {90.0:8.4f} {mean_anomaly:8.4f} {mean_motion:11.8f}{1234:5d}"
    )
    assert len(line1) == len(line2) == 68
    return (f"STARLINK-{number}", line1 + checksum(line1), line2 + checksum(line2))


def invented_dense_constellation(n=140):
    """Element sets in a 53-degree shell around day 215 of 2026, a fifth with Alpha-5 numbers, some not operational."""
    rng = np.random.default_rng(2)
    records = []
    for index in range(n):
        number = int(rng.integers(44000, 99999)) if index % 5 else int(rng.integers(100000, 339999))
        day = 215.0 - float(rng.uniform(0.0, 6.0)) if index % 11 else 200.0  # every eleventh is stale
        mean_motion = float(rng.uniform(15.0, 15.3)) if index % 13 else 16.2  # every thirteenth is too low
        records.append(valid_tle(number, day, float(rng.uniform(0, 360)), float(rng.uniform(0, 360)), mean_motion))
    return records


def write_real_fleet_cache(cache, records, tag, reference):
    """Put the CelesTrak group lists and the Space-Track history where the registered release looks for them."""
    groups = {"resource": [], "planet": []}
    for record in records:
        groups[record["_GROUP"]].append({key: value for key, value in record.items() if key != "_GROUP"})
    for group, group_records in groups.items():
        content = {"fetched_at": time.time(), "url": "x", "records": group_records}
        (cache / f"arf_gp_{group}.json").write_text(json.dumps(content))
    history = [{key: value for key, value in record.items() if key != "_GROUP"} for record in records]
    stamp = reference[:19].replace("-", "").replace(":", "")
    (cache / f"arf_gp_history_{tag}_{stamp}.json").write_text(json.dumps({"records": history}))
    return groups, history


def stations_of(definitions):
    return {name: GroundStation(d["lat"], d["lng"], d["alt_m"]) for name, d in definitions.items()}


def require_equal(result, registered):
    """Require release 2's access result to equal the namespace of the registered run, exactly."""
    assert compare_with_registered(result, registered) == []
    # The comparison itself must see a difference of one bit, a lost row and a changed count.
    events = registered["refined_events_df"]
    if len(events):
        nudged = result.refinement.events.copy()
        nudged.loc[0, "A_nadir_deg"] = np.nextafter(nudged.loc[0, "A_nadir_deg"], np.inf)
        result.refinement.events, original = nudged, result.refinement.events
        assert any("refined events" in line for line in compare_with_registered(result, registered))
        result.refinement.events = original.iloc[1:]
        assert any("refined events" in line for line in compare_with_registered(result, registered))
        result.refinement.events = original
    result.refinement.counts["rows_dark"] += 1
    assert any("refinement counts" in line for line in compare_with_registered(result, registered))
    result.refinement.counts["rows_dark"] -= 1


REAL_CASES = {
    "primary": {},
    "optimistic_swath": {"planet_swath_case": "optimistic"},
    "without_sentinel_2a": {"exclude_satellites": ("SENTINEL-2A",)},
    "far_station": {"stations": FAR_STATION},
    "legacy_end_caps": {"legacy_end_caps": True},
    # 10 February 2026 lies in the Sentinel-2A pause of 13 January to 8 March. The Sun's right ascension is then about
    # 191 degrees greater than on 3 August, so the planes are turned by that much to keep the morning node.
    "during_an_acquisition_exclusion": {"reference": "2026-02-10T00:00:00.000000", "node_shift": 191.0, "extra": 5},
    "short_window_low_mask": {"window_hours": 13, "min_elevation_deg": 5.0},
}


@pytest.mark.parametrize("release", ["1.5.3", "1.5.6"])
@pytest.mark.parametrize("case", list(REAL_CASES))
def test_real_fleet_access_matches_registered_release(release, case, tmp_path):
    options = dict(REAL_CASES[case])
    reference = options.pop("reference", "2026-08-03T00:00:00.000000")
    stations = options.pop("stations", BOTSWANA_STATIONS)
    swath_case = options.get("planet_swath_case", "conservative")
    excluded = options.get("exclude_satellites", ())
    tag = f"earth-resources_planet-{swath_case}_from{reference[:10].replace('-', '')}"
    epoch = reference[:8] + f"{int(reference[8:10]) - 1:02d}T10:00:00.000000"  # element sets 14 hours old
    fleet = invented_real_fleet(epoch, options.pop("node_shift", 0.0), options.pop("extra", 0))
    groups, history = write_real_fleet_cache(tmp_path, fleet, tag, reference)
    registered = run_cells_2_4_5(release, REGION_CELLS, stations, reference, tmp_path, fleet_tag=tag, **options)

    from arf.access.orekit import OrekitEphemeris, tle_from_omm

    records = elements_as_of(history, merge_groups(groups), reference)
    satellites, counts = select_fleet(records, reference, mission_table(swath_case), exclude_names=excluded)
    assert counts == registered["_op_stats"]
    imagers = [Imager(s.id, s.swath_km, s.min_sun_elev_deg) for s in satellites]
    station_objects = stations_of(stations)
    ephemeris = OrekitEphemeris({s.id: tle_from_omm(s.elements) for s in satellites}, reference, station_objects)
    result = compute_access(
        imagers,
        ephemeris,
        RegionIndex.from_cells(REGION_CELLS),
        station_objects,
        window_hours=options.get("window_hours", 48),
        min_elevation_deg=options.get("min_elevation_deg", 15.0),
        exclusions=ACQUISITION_EXCLUSIONS,
        legacy_end_caps=options.get("legacy_end_caps", False),
    )
    require_equal(result, registered)

    events, dropped = result.refinement.events, result.refinement.counts
    assert len(events) > 500 and events["sat"].nunique() >= 5
    if case == "primary":
        assert dropped["rows_dark"] > 0 and dropped["rows_end_cap_excluded"] > 0  # night passes and strip ends occur
        assert (events["T_wait_min"] == 0).all()  # the Botswana stations see every pass over the region
        assert events["sun_elev_deg"].min() >= 5.0
    if case == "far_station":
        assert events["T_wait_min"].between(1, 100).any() and (events["D_downlink"] < 1).any()
    if case == "legacy_end_caps":
        assert dropped["rows_end_cap_excluded"] == 0
    if case == "during_an_acquisition_exclusion":
        # Every object named SENTINEL-2A is dropped, also those told apart by a catalogue number in brackets.
        assert dropped["passes_acquisition_excluded"] > 0 and dropped["rows_acquisition_excluded"] > 0
        assert not any(str(name).startswith("SENTINEL-2A") for name in set(events["sat"]))
        assert any(imager.id.startswith("SENTINEL-2A [") for imager in imagers)


DENSE_CASES = {
    "swath_and_daylight_as_in_the_study": {"proxy_swath_km": 32.5, "proxy_min_sun_elev_deg": 10.0, "active_n": 60},
    "legacy_half_angle_no_daylight_limit": {"active_n": 12, "window_hours": 24},
    "wide_swath_far_station": {"proxy_swath_km": 120.0, "active_n": 25, "stations": FAR_STATION, "seed": 7},
}


@pytest.mark.parametrize("case", list(DENSE_CASES))
def test_dense_constellation_access_matches_release_156(case, tmp_path):
    options = dict(DENSE_CASES[case])
    stations = options.pop("stations", BOTSWANA_STATIONS)
    reference = "2026-08-03T00:00:00.000000"
    records = invented_dense_constellation()
    registered = run_cells_2_4_5(
        "1.5.6", REGION_CELLS, stations, reference, tmp_path, fleet="starlink", raw_records=records, **options
    )

    from arf.access.orekit import OrekitEphemeris, tle_from_lines

    seed, n = options.get("seed", 42), options["active_n"]
    shuffled, counts = operational_sample(records, reference, seed)
    assert counts == registered["_op_stats"]
    sample = pinned_selection(shuffled, n, tmp_path / "release2_pin.json")
    assert [record[0] for record in sample] == [satellite["id"] for satellite in registered["constellation"]]
    registered_pin = Path(registered["SELECTION_CACHE_PATH"]).read_text()
    assert json.loads((tmp_path / "release2_pin.json").read_text()) == json.loads(registered_pin)
    imagers = [Imager(r[0], options.get("proxy_swath_km"), options.get("proxy_min_sun_elev_deg")) for r in sample]
    station_objects = stations_of(stations)
    ephemeris = OrekitEphemeris({record[0]: tle_from_lines(record) for record in sample}, reference, station_objects)
    result = compute_access(
        imagers,
        ephemeris,
        RegionIndex.from_cells(REGION_CELLS),
        station_objects,
        window_hours=options.get("window_hours", 48),
    )
    require_equal(result, registered)
    assert len(result.refinement.events) > 300
    assert any(number >= 100000 for number in json.loads((tmp_path / "release2_pin.json").read_text())["norad_ids"])
    if case == "legacy_half_angle_no_daylight_limit":
        assert result.refinement.events["sun_elev_deg"].isna().all()
        assert (result.refinement.events["theta_max_deg"] == 9.22).all()


@pytest.mark.parametrize("release", ["1.5.3", "1.5.6"])
def test_element_sets_orekit_refuses_are_malformed_as_in_cell_2(release):
    """Cell 2 builds each TLE while selecting the fleet and counts one that Orekit refuses as malformed."""
    from arf.access.orekit import check_omm_with_orekit

    registered = run_cell2_fleet_functions(release)  # Orekit's own TLE class: no stand-ins here
    reference = "2026-08-03T00:00:00.000000"
    records = invented_real_fleet("2026-08-02T10:00:00.000000")
    records.append(omm(39634, "LANDSAT 8", epoch="2026-08-02 10:00:00"))  # Python reads it; Orekit refuses it
    theirs, their_counts = registered["build_eo_constellation"](records, reference)
    ours, our_counts = select_fleet(records, reference, mission_table(), check_elements=check_omm_with_orekit)
    assert our_counts == their_counts and our_counts["malformed"] == 1
    assert [satellite.id for satellite in ours] == [satellite["id"] for satellite in theirs]
    plain = select_fleet(records, reference, mission_table())[1]  # without Orekit's check the record is taken
    assert plain["malformed"] == 0 and plain["operational"] == our_counts["operational"] + 1
