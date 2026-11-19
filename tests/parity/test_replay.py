"""Rebuilding a finished run's fleet (arf.replay) and the guard against satellites that share a name."""

import json

import pytest
from test_check_fleet_tool import write_dense_run, write_real_run

from arf.access import Imager, RegionIndex, compute_access
from arf.fleet.elements import catalogue_number, omm_tle_fields
from arf.replay import recorded_fleet


def test_dense_fleet_is_read_from_the_pin_file(tmp_path):
    _, pin = write_dense_run(tmp_path / "run", tmp_path / "cache", 50)
    pinned = json.loads(pin.read_text())["norad_ids"]
    fleet = recorded_fleet(tmp_path / "run")
    assert [catalogue_number(fleet.elements[imager.id][1]) for imager in fleet.imagers] == pinned
    assert fleet.notes == [] and fleet.kind == "dense" and fleet.exclusions == []

    # A pin in another order is followed, as Cell 2 follows it, and the difference from the seeded sample is noted.
    swapped = list(pinned)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    pin.write_text(json.dumps({"active_n": 50, "norad_ids": [*swapped, 1]}))  # 1 is not an operational satellite
    fleet = recorded_fleet(tmp_path / "run")
    assert [catalogue_number(fleet.elements[imager.id][1]) for imager in fleet.imagers] == swapped
    assert fleet.notes == [
        "1 pinned satellites are not among the operational ones",
        "the pin file differs from the first N of the seeded sample",
    ]
    assert fleet.changed_inputs == [pin.name]


def test_real_fleet_counts_a_refused_element_set_as_malformed(tmp_path):
    write_real_run(tmp_path / "run", tmp_path / "cache", "1.5.6")
    plain = recorded_fleet(tmp_path / "run")

    def refuse_flock_4x_1(omm):
        omm_tle_fields(omm)  # as Orekit's check does first: a record with a missing field stays malformed
        if omm["OBJECT_NAME"] == "FLOCK 4X-1":
            raise ValueError("refused")

    checked = recorded_fleet(tmp_path / "run", check_elements=refuse_flock_4x_1)
    assert checked.counts["malformed"] == plain.counts["malformed"] + 1
    assert [imager.id for imager in checked.imagers] == [i.id for i in plain.imagers if i.id != "FLOCK 4X-1"]


def test_satellites_sharing_a_name_are_refused():
    imagers = [Imager("STARLINK-1"), Imager("STARLINK-2"), Imager("STARLINK-1")]
    with pytest.raises(ValueError, match="share an identifier"):
        compute_access(imagers, ephemeris=None, region=RegionIndex([], [], []), stations={})
