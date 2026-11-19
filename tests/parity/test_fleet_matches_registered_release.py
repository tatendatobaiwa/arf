"""arf.fleet against the registered releases themselves.

The mission tables of Cell 0 and the fleet functions of Cell 2 are taken from the release files unchanged
(``registered.py``) and called on invented element sets: the parsers, the operational filter with its seeded sample,
the pin file, and the selection of the real Earth-observation fleet with its counts. Orekit's classes are replaced by
stand-ins that record their arguments (see ``orekit_stand_ins``), so the values each TLE would be built from are
compared as well.

The dense-constellation sample is compared with release 1.5.6. Release 1.5.3 gives the same sample when no element set
has an Alpha-5 catalogue number, and counts those that do as malformed, which is the fault release 1.5.6 corrected.
No real element set is in this repository.
"""

import json
import math

import numpy as np
import pytest
from registered import orekit_stand_ins, run_cell2_fleet_functions

from arf.fleet import (
    ACQUISITION_EXCLUSIONS,
    CELESTRAK_GROUPS,
    OPERATIONAL_CRITERIA,
    elements_as_of,
    fleet_file_entries,
    match_mission,
    mission_table,
    operational_sample,
    pinned_selection,
    select_fleet,
    swath_from_fov_km,
)
from arf.fleet import elements as el

RELEASE_NAMES = ("1.5.3", "1.5.6")
ONSET = "2026-08-03T00:00:00.000000"


def to_alpha5(number):
    """Write a catalogue number in the five-character TLE field."""
    if number < 100000:
        return f"{number:05d}"
    return el.ALPHA5_LETTERS[number // 10000 - 10] + f"{number % 10000:04d}"


def tle(number, day_of_year, mean_motion=15.06, eccentricity=0.0001, name=None, year=26):
    """One invented element set with the fields in their TLE columns."""
    field = to_alpha5(number)
    line1 = f"1 {field}U {'19074A':<8s} {year:02d}{day_of_year:012.8f}  .00001234  00000-0  12345-3 0  9991"
    ecc = f"{round(eccentricity * 1e7):07d}"
    line2 = f"2 {field} {53.05:8.4f} {120.5:8.4f} {ecc} {90.0:8.4f} {270.0:8.4f} {mean_motion:11.8f}{1234:5d}0"
    return (name or f"STARLINK-{number}", line1, line2)


def invented_tles(seed, n=400, alpha5=True):
    """Element sets around day 215 of 2026 (3 August): operational ones, and every way of failing the filter."""
    rng = np.random.default_rng(seed)
    records = []
    for index in range(n):
        number = int(rng.integers(44000, 99999))
        if alpha5 and rng.random() < 0.2:
            number = int(rng.integers(100000, 339999))
        day = 215.0 - float(rng.uniform(0.0, 6.5))
        kind = rng.random()
        if kind < 0.08:
            day = 215.0 - float(rng.uniform(7.2, 30.0))  # stale
        elif kind < 0.12:
            day = 215.0 + float(rng.uniform(0.01, 2.0))  # from after the onset
        mean_motion = float(rng.uniform(15.0, 15.6))  # about 550 to 380 km
        eccentricity = float(rng.uniform(0.0, 0.002))
        if 0.12 <= kind < 0.2:
            mean_motion = float(rng.uniform(15.9, 16.3))  # low perigee
        elif 0.2 <= kind < 0.26:
            mean_motion, eccentricity = float(rng.uniform(13.0, 14.3)), float(rng.uniform(0.0, 0.05))  # high apogee
        record = tle(number, day, mean_motion, eccentricity)
        if 0.26 <= kind < 0.29:
            record = (record[0], record[1][:10], record[2])  # cut short
        elif 0.29 <= kind < 0.31:
            record = (record[0], record[1], record[2][:52] + "not a number" + record[2][63:])
        elif 0.31 <= kind < 0.33:
            record = (record[0], "1 ?????U" + record[1][8:], record[2])  # unreadable catalogue number
        records.append(record)
        if index % 97 == 0:
            records.append(tle(number, day - 0.5, mean_motion, eccentricity, name=f"STARLINK-{number} (again)"))
    return records


# --- parsers ---------------------------------------------------------------------------------------------------------


def test_parsers_match_release_156():
    registered = run_cell2_fleet_functions("1.5.6")
    assert registered["_ALPHA5_LETTERS"] == el.ALPHA5_LETTERS
    for field in ("00005", "25544", "99999", "A0000", "Z9999", "H1234", " 1234", "T0001"):
        assert el.alpha5_to_int(field) == registered["_alpha5_to_int"](field)
    assert el.alpha5_to_int("Z9999") == 339999 and el.alpha5_to_int("A0000") == 100000
    for field in ("I0001", "O0001", "a0001", "AB123", "", "12 45"):
        with pytest.raises(ValueError):
            registered["_alpha5_to_int"](field)
        with pytest.raises(ValueError):
            el.alpha5_to_int(field)
    for _, line1, line2 in invented_tles(1):
        assert el.catalogue_number(line1) == registered["_parse_norad_id"](line1)
        for ours, theirs, line in (
            (el.tle_epoch, registered["_elset_epoch_utc"], line1),
            (el.tle_orbit_km, registered["_elset_orbit_km"], line2),
        ):
            try:
                expected = theirs(line)
            except (ValueError, IndexError, ZeroDivisionError) as error:
                with pytest.raises(type(error)):
                    ours(line)
            else:
                assert ours(line) == expected
    assert el.catalogue_number(None) is None and registered["_parse_norad_id"](None) is None
    assert el.tle_epoch(tle(1, 1.5, year=99)[1]).year == 1999  # two-digit years from 57 are in the 1900s
    assert registered["_elset_epoch_utc"](tle(1, 1.5, year=56)[1]) == el.tle_epoch(tle(1, 1.5, year=56)[1])


@pytest.mark.parametrize("release", RELEASE_NAMES)
def test_newest_epoch_and_history_reduction(release):
    registered = run_cell2_fleet_functions(release)
    records = invented_tles(2, alpha5=False)
    assert el.newest_tle_epoch_iso(records) == registered["_newest_elset_epoch_iso"](records)
    with pytest.raises(RuntimeError):
        el.newest_tle_epoch_iso([("x", "bad", "bad")])
    rng = np.random.default_rng(3)
    history = []
    for name, line1, line2 in records[:150]:
        for copy in range(int(rng.integers(1, 4))):
            second = int(rng.integers(0, 60))
            epoch = f"2026-08-0{int(rng.integers(1, 4))}T12:00:{second:02d}" + (".250000" if copy % 2 else "")
            # Space-Track gives the catalogue number as text, so the order is that of the text.
            history.append(
                {
                    "NORAD_CAT_ID": str(el.catalogue_number(line1)),
                    "OBJECT_NAME": f"{name} v{copy}",
                    "EPOCH": epoch,
                    "TLE_LINE1": line1,
                    "TLE_LINE2": line2,
                }
            )
    order = rng.permutation(len(history))
    history = [history[position] for position in order]
    assert el.latest_per_satellite(history) == registered["_dedupe_gp_history_to_latest_per_satellite"](history)


# --- the dense-constellation sample ----------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [42, 43, 7])
@pytest.mark.parametrize("data_seed", [11, 12])
def test_operational_sample_matches_release_156(seed, data_seed):
    registered = run_cell2_fleet_functions("1.5.6")
    records = invented_tles(data_seed)
    ours, our_counts = operational_sample(records, ONSET, seed)
    theirs, their_counts = registered["select_operational_sample"](records, ONSET, seed)
    assert ours == theirs
    assert our_counts == their_counts
    assert registered["OPERATIONAL_CRITERIA"] == OPERATIONAL_CRITERIA
    for key in ("malformed", "stale_or_future", "low_perigee", "high_apogee"):
        assert our_counts[key] > 0, key  # the invented data reach every branch of the filter
    assert sum(el.catalogue_number(record[1]) >= 100000 for record in ours) > 20  # Alpha-5 satellites are sampled
    stricter = dict(OPERATIONAL_CRITERIA, max_apogee_km=560.0, max_elset_age_days=3.0)
    assert operational_sample(records, ONSET, seed, stricter) == registered["select_operational_sample"](
        records, ONSET, seed, stricter
    )
    assert operational_sample(records, "2026-08-03T00:00:00Z", seed)[0] == ours


def test_release_153_differs_only_on_alpha5_catalogue_numbers():
    registered = run_cell2_fleet_functions("1.5.3")
    without = invented_tles(11, alpha5=False)
    assert operational_sample(without, ONSET, 42) == registered["select_operational_sample"](without, ONSET, 42)
    with_alpha5 = invented_tles(11, alpha5=True)
    _, ours = operational_sample(with_alpha5, ONSET, 42)
    _, theirs = registered["select_operational_sample"](with_alpha5, ONSET, 42)
    assert theirs["malformed"] > ours["malformed"] and theirs["operational"] < ours["operational"]


@pytest.mark.parametrize("n", [5, 100, 10_000])
def test_pin_file_matches_release_156(tmp_path, n):
    registered = run_cell2_fleet_functions("1.5.6")
    candidates, _ = operational_sample(invented_tles(13), ONSET, 42)
    our_pin, their_pin = tmp_path / "ours.json", tmp_path / "theirs.json"
    ours = pinned_selection(candidates, n, our_pin)
    theirs = registered["select_pinned_satellites"](candidates, n, str(their_pin))
    assert ours == theirs and len(ours) == min(n, len(candidates))
    assert our_pin.read_text() == their_pin.read_text()
    # A later run with other element sets: the pin decides, in its order; a pinned satellite no longer there drops out.
    later = [record for record in invented_tles(13) if record != candidates[1]][::-1]
    again = pinned_selection(later, n, our_pin)
    assert again == registered["select_pinned_satellites"](later, n, str(their_pin))
    assert [el.catalogue_number(r[1]) for r in again] == [
        number
        for number in json.loads(our_pin.read_text())["norad_ids"]
        if number != el.catalogue_number(candidates[1][1])
    ]


# --- the real Earth-observation fleet --------------------------------------------------------------------------------


@pytest.mark.parametrize("release", RELEASE_NAMES)
@pytest.mark.parametrize(("case", "sun"), [("conservative", 5.0), ("optimistic", 5.0), ("conservative", 10.0)])
def test_mission_tables_are_the_registered_ones(release, case, sun):
    registered = run_cell2_fleet_functions(release, case, sun)
    assert mission_table(case, sun) == registered["MISSION_TABLE"]
    assert registered["CELESTRAK_GROUPS"] == CELESTRAK_GROUPS
    assert registered["ACQUISITION_EXCLUSIONS"] == ACQUISITION_EXCLUSIONS
    for name in (
        "SENTINEL-2A",
        "sentinel 2c",
        "SENTINEL-2",
        "SENTINEL-1A",
        "LANDSAT 9",
        "LANDSAT 7",
        "FLOCK 4X-12",
        " FLOCK 1",
        "DOVE 3",
    ):
        assert match_mission(name, mission_table(case, sun)) == registered["_match_mission"](name)
    for altitude, field in ((475.0, 3.0), (475.0, 4.0), (520.3, 4.0), (786.0, 20.6)):
        assert swath_from_fov_km(altitude, field) == registered["swath_from_fov_km"](altitude, field)
    assert 24.5 < swath_from_fov_km(475.0, 3.0) < 25.5  # the PSScene specification gives 24 km frames at 475 km
    with pytest.raises(ValueError, match="beyond the horizon"):
        swath_from_fov_km(700.0, 170.0)
    with pytest.raises(ValueError, match="beyond the horizon"):
        registered["swath_from_fov_km"](700.0, 170.0)


def omm(
    number, name, epoch="2026-08-02T10:00:00.000000", mean_motion=14.3, eccentricity=0.0001, group="resource", **changes
):
    record = {
        "OBJECT_NAME": name,
        "OBJECT_ID": "2015-028A",
        "EPOCH": epoch,
        "MEAN_MOTION": mean_motion,
        "ECCENTRICITY": eccentricity,
        "INCLINATION": 98.57,
        "RA_OF_ASC_NODE": 290.1,
        "ARG_OF_PERICENTER": 85.2,
        "MEAN_ANOMALY": 274.9,
        "EPHEMERIS_TYPE": 0,
        "CLASSIFICATION_TYPE": "U",
        "NORAD_CAT_ID": number,
        "ELEMENT_SET_NO": 999,
        "REV_AT_EPOCH": 58123,
        "BSTAR": 0.00012,
        "MEAN_MOTION_DOT": 0.0000021,
        "MEAN_MOTION_DDOT": 0,
        "_GROUP": group,
    }
    record.update(changes)
    return {key: value for key, value in record.items() if value is not None}


def invented_omm_records():
    """A fleet with every case the selection distinguishes."""
    records = [
        omm(40697, "SENTINEL-2A"),
        omm(42063, "SENTINEL-2B", epoch="2026-08-02T23:59:59"),
        omm(60989, "SENTINEL-2C", epoch="2026-07-27T00:00:00.000000"),  # exactly 7 days old: kept
        omm(39084, "LANDSAT 8", mean_motion=14.571),
        omm(49260, "LANDSAT 9", mean_motion=14.571, OBJECT_ID="2021-088A"),
        omm(25682, "LANDSAT 7"),  # no mission entry
        omm(27424, "AQUA"),
        omm(43013, "NOAA 20"),
        omm(39634, "SENTINEL-1A"),
        omm(41335, "SENTINEL-3A"),
        omm(40001, "SENTINEL-2A", NORAD_CAT_ID="40001"),  # a second object of the same name; number as text
        omm(50001, "FLOCK 4X-1", mean_motion=15.2, group="planet"),
        omm(50002, "FLOCK 4X-2", mean_motion=15.25, eccentricity=0.0012, group="planet"),
        omm(123456, "FLOCK 4BE-7", mean_motion=15.1, group="planet", OBJECT_ID="2025-200BQ"),  # above 99999
        omm(50004, "FLOCK 4X-4", mean_motion=16.2, group="planet"),  # perigee below 300 km
        omm(50005, "FLOCK 4X-5", epoch="2026-07-20T00:00:00", group="planet"),  # stale
        omm(50006, "FLOCK 4X-6", epoch="2026-08-03T06:00:00", group="planet"),  # from after the reference epoch
        omm(50007, "FLOCK 4X-7", group="planet", MEAN_MOTION="fast"),  # not a number
        omm(50008, "FLOCK 4X-8", group="planet", BSTAR=None),  # a field missing
        omm(50009, "FLOCK 4X-9", group="planet", EPOCH="yesterday"),
        omm(50010, "FLOCK 4X-10", group="planet", MEAN_MOTION=0.0),  # division by zero
        omm(
            50011,
            "FLOCK 4X-11",
            group="planet",
            OBJECT_ID="",
            MEAN_MOTION_DOT="-0.00000321",
            EPOCH="2026-08-01T00:00:00Z",
        ),
        omm(50012, "flock 4x-12 ", mean_motion=15.3, group=None),
    ]
    rng = np.random.default_rng(5)
    return [records[position] for position in rng.permutation(len(records))]


def as_entries(constellation):
    """The registered constellation without its Orekit objects, in the fields of release 2's Satellite."""
    keys = ("id", "norad_id", "group", "mission", "swath_km", "fov_deg", "min_sun_elev_deg")
    return [{key: satellite[key] for key in keys} for satellite in constellation]


@pytest.mark.parametrize("release", RELEASE_NAMES)
@pytest.mark.parametrize(
    ("case", "options"),
    [
        ("conservative", {}),
        ("optimistic", {}),
        ("conservative", {"exclude_names": ("SENTINEL-2A", "sentinel-2b ", "LANDSAT 5")}),
        ("conservative", {"min_age_days": -2.0}),
        ("optimistic", {"min_perigee_km": 450.0, "max_age_days": 1.0}),
    ],
)
def test_real_fleet_matches_registered_release(release, case, options):
    registered = run_cell2_fleet_functions(release, case)
    records = invented_omm_records()
    reference = "2026-08-03T00:00:00.000000"
    with orekit_stand_ins():
        constellation, their_counts = registered["build_eo_constellation"](records, reference, **options)
    satellites, our_counts = select_fleet(records, reference, mission_table(case), **options)
    assert our_counts == their_counts
    keys = ("id", "norad_id", "group", "mission", "swath_km", "fov_deg", "min_sun_elev_deg")
    assert [{key: getattr(satellite, key) for key in keys} for satellite in satellites] == as_entries(constellation)
    # The values each TLE is built from: Orekit's constructor takes them in this order.
    for satellite, theirs in zip(satellites, constellation, strict=True):
        arguments = list(theirs["tle"].arguments)
        assert arguments.pop() == "UTC"
        fields = list(el.omm_tle_fields(satellite.elements).values())
        fields[7] = ("date", fields[7], "UTC")
        assert fields == arguments
    if not options:
        assert our_counts["malformed"] == 4 and our_counts["stale_or_future"] == 2 and our_counts["low_perigee"] == 1
        assert our_counts["unmatched_mission"] == 5 and our_counts["operational"] == len(satellites) == 11
        assert {satellite.id for satellite in satellites} >= {"SENTINEL-2A", "SENTINEL-2A [40697]", "FLOCK 4BE-7"}
    # The fleet file the releases write beside the run.
    expected = [
        {
            "id": s["id"],
            "norad_id": s["norad_id"],
            "mission": s["mission"],
            "swath_km": round(s["swath_km"], 3),
            "fov_deg": s.get("fov_deg"),
            "min_sun_elev_deg": s["min_sun_elev_deg"],
        }
        for s in constellation
    ]
    assert fleet_file_entries(satellites) == expected


@pytest.mark.parametrize("release", RELEASE_NAMES)
def test_elements_as_of_an_epoch_match_registered_release(release, tmp_path):
    tag = "earth-resources_planet-conservative_from20260803"
    registered = run_cell2_fleet_functions(release, fleet_tag=tag)
    live = [
        omm(40697, "SENTINEL-2A"),
        omm(42063, "SENTINEL-2B"),
        omm(50001, "FLOCK 4X-1", group="planet"),
        omm(27424, "AQUA"),
    ]
    history = [
        omm(40697, "SENTINEL 2A", epoch="2026-08-01T00:00:00", _GROUP=None),
        omm(40697, "SENTINEL 2A", epoch="2026-08-02T12:00:00.500000", _GROUP=None, NORAD_CAT_ID="40697"),
        omm(40697, "SENTINEL 2A", epoch="2026-08-03T00:00:01", _GROUP=None),  # one second after the epoch: never used
        omm(42063, "SENTINEL 2B", epoch="2026-08-03T00:00:00", _GROUP=None),  # exactly at the epoch: used
        omm(50001, "FLOCK 4X 1", epoch="2026-07-30T00:00:00", _GROUP=None),
        omm(50001, "FLOCK 4X 1", epoch="2026-07-30T00:00:00", _GROUP=None, BSTAR=0.5),  # same epoch: the first stays
        omm(77777, "NOT IN THE LIVE LIST", epoch="2026-08-01T00:00:00", _GROUP=None),
    ]
    epoch = "2026-08-03T00:00:00.000000"
    (tmp_path / f"arf_gp_history_{tag}_20260803T000000.json").write_text(json.dumps({"records": history}))
    theirs = registered["load_eo_gp_history"](live, epoch, str(tmp_path))  # reads the cache: no network
    ours = elements_as_of(history, live, epoch)
    assert ours == theirs
    assert [record["OBJECT_NAME"] for record in ours] == [
        "SENTINEL-2A",
        "SENTINEL-2B",
        "FLOCK 4X-1",
        "NOT IN THE LIVE LIST",
    ]
    assert ours[0]["EPOCH"] == "2026-08-02T12:00:00.500000" and ours[2]["BSTAR"] == 0.00012


def test_orbit_altitudes_are_physically_sensible():
    perigee, apogee = el.omm_orbit_km({"MEAN_MOTION": 14.30818, "ECCENTRICITY": 0.0001})  # Sentinel-2: 786 km
    assert 770 < perigee < apogee < 800
    assert math.isclose(sum(el.tle_orbit_km(tle(1, 1.0, 15.06, 0.0)[2])) / 2, 550, abs_tol=10)  # a 550 km shell
