"""The real Earth-observation fleet: every operational satellite of a mission with a sourced swath.

No sampling: the fleet is every satellite that matches a mission, has an element set of a usable age and a perigee
above the limit. Reproduces ``build_eo_constellation`` and the selection of element sets in ``load_eo_gp_history`` of
Cell 2 of releases 1.5.3 to 1.5.6, without building the propagators (see ``arf.access``, which needs Orekit).
"""

from __future__ import annotations

import collections
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from arf.fleet.elements import omm_epoch, omm_orbit_km, omm_tle_fields
from arf.fleet.missions import match_mission, swath_from_fov_km


@dataclass(frozen=True)
class Satellite:
    """One satellite of the fleet.

    Attributes:
        id: Its name, made unique: a second satellite of the same name has its catalogue number appended.
        norad_id: Its catalogue number.
        group: The CelesTrak group its record came from.
        mission: The mission it was matched to.
        swath_km: Its ground swath: the mission's stated swath, or the one its field of view gives at the mean of
            its perigee and apogee altitudes.
        fov_deg: The field of view the swath was computed from, or ``None`` when the swath is stated.
        min_sun_elev_deg: The sun elevation below which it does not image.
        elements: Its OMM record.
    """

    id: str
    norad_id: int
    group: str | None
    mission: str
    swath_km: float
    fov_deg: float | None
    min_sun_elev_deg: float
    elements: Mapping


def merge_groups(records_by_group: Mapping[str, Iterable[Mapping]]) -> list[dict]:
    """Merge the OMM records of several CelesTrak groups into one list, one record per satellite.

    A satellite listed in more than one group is kept once, from the first group that lists it. Each record gets
    the name of its group under ``_GROUP``.
    """
    by_number: dict[int, dict] = {}
    for group, records in records_by_group.items():
        for record in records:
            by_number.setdefault(int(record["NORAD_CAT_ID"]), dict(record, _GROUP=group))
    return list(by_number.values())


def elements_as_of(
    history_records: Iterable[Mapping],
    live_records: Iterable[Mapping],
    epoch_iso: str,
) -> list[dict]:
    """Return, per satellite, the newest element set of a Space-Track history that is not later than an epoch.

    The name and the group of each satellite are taken from the current CelesTrak list (``live_records``), so that
    missions are matched as in a live run; a satellite absent from that list keeps the name of its history record.

    Returns:
        One OMM record per satellite, in catalogue-number order, with ``OBJECT_NAME`` and ``_GROUP`` set.
    """
    names = {int(record["NORAD_CAT_ID"]): str(record.get("OBJECT_NAME", "")).strip() for record in live_records}
    groups = {int(record["NORAD_CAT_ID"]): record.get("_GROUP") for record in live_records}
    end = datetime.fromisoformat(epoch_iso.replace("Z", ""))
    latest: dict[int, Mapping] = {}
    for record in history_records:
        number = int(record["NORAD_CAT_ID"])
        epoch = omm_epoch(record)
        if epoch.replace(tzinfo=None) > end:  # never an element set from after the epoch
            continue
        if number not in latest or epoch > omm_epoch(latest[number]):
            latest[number] = record
    return [
        dict(record, OBJECT_NAME=names.get(number, record.get("OBJECT_NAME", "")), _GROUP=groups.get(number))
        for number, record in sorted(latest.items())
    ]


def _swath(mission: Mapping, perigee_km: float, apogee_km: float) -> tuple[float, float | None]:
    """Return the swath of a satellite of a mission and the field of view it came from, if it came from one."""
    if "swath_km" in mission:
        return float(mission["swath_km"]["value"]), None
    field_of_view = float(mission["fov_deg"]["value"])
    return swath_from_fov_km(0.5 * (perigee_km + apogee_km), field_of_view), field_of_view


def select_fleet(
    omm_records: Sequence[Mapping],
    reference_iso: str,
    missions: Sequence[Mapping],
    min_perigee_km: float = 300.0,
    max_age_days: float = 7.0,
    min_age_days: float = 0.0,
    exclude_names: Iterable[str] = (),
    check_elements: Callable[[Mapping], object] = omm_tle_fields,
) -> tuple[list[Satellite], dict]:
    """Select the fleet from OMM records.

    A satellite is taken, in catalogue-number order, when its name matches a mission, it is not excluded by name,
    its record is well formed, the age of its element set at the reference epoch lies in
    ``[min_age_days, max_age_days]``, and its perigee is at least ``min_perigee_km``. Each rejected record is
    counted under the first of these it fails, in that order.

    Args:
        omm_records: The element sets, each with ``OBJECT_NAME`` and, optionally, ``_GROUP``.
        reference_iso: The start of the window, UTC, ISO 8601.
        missions: The mission table (:func:`arf.fleet.missions.mission_table`).
        min_perigee_km: The lowest perigee altitude of an operational satellite.
        max_age_days: The greatest age of an element set at the reference epoch.
        min_age_days: The least age; 0 admits no element set newer than the reference epoch, and the releases use
            -2 when a live window is given a fixed start.
        exclude_names: Object names left out on purpose, for a fleet sensitivity run; matched exactly, case
            ignored.
        check_elements: Called on each record; a ``KeyError``, ``ValueError``, ``TypeError`` or
            ``ZeroDivisionError`` from it marks the record as malformed. By default the conversion of the record
            to the values of a TLE. The registered releases build the Orekit TLE here, which can reject a record
            this conversion accepts; pass the Orekit conversion to reproduce that.

    Returns:
        The satellites and the counts the releases record under ``satellite_selection`` in
        ``arf_cell2_summary.json``.
    """
    reference = datetime.fromisoformat(reference_iso).replace(tzinfo=timezone.utc)
    excluded = {str(name).strip().upper() for name in exclude_names}
    counts: dict = {
        "fetched": len(omm_records),
        "malformed": 0,
        "stale_or_future": 0,
        "low_perigee": 0,
        "high_apogee": 0,
        "unmatched_mission": 0,
        "excluded_by_config": 0,
        "operational": 0,
        "criteria": {
            "min_perigee_km": min_perigee_km,
            "max_elset_age_days": max_age_days,
            "min_elset_age_days": min_age_days,
            "exclude_satellites": sorted(excluded),
        },
        "seed": None,
    }
    unmatched_families: collections.Counter = collections.Counter()
    per_mission: collections.Counter = collections.Counter()
    satellites: list[Satellite] = []
    used_ids: set[str] = set()
    unmatched_names: list[str] = []
    excluded_found: list[str] = []
    for omm in sorted(omm_records, key=lambda record: int(record["NORAD_CAT_ID"])):
        name = str(omm.get("OBJECT_NAME", "")).strip()
        mission = match_mission(name, missions)
        if mission is None:
            counts["unmatched_mission"] += 1
            unmatched_names.append(name)
            unmatched_families[re.sub(r"[\s\-]*\d+[A-Z]?$", "", name) or name] += 1
            continue
        if name.upper() in excluded:
            counts["excluded_by_config"] += 1
            excluded_found.append(name)
            continue
        try:
            age_days = (reference - omm_epoch(omm)).total_seconds() / 86400.0
            perigee_km, apogee_km = omm_orbit_km(omm)
            check_elements(omm)
        except (KeyError, ValueError, TypeError, ZeroDivisionError):
            counts["malformed"] += 1
            continue
        if not min_age_days <= age_days <= max_age_days:
            counts["stale_or_future"] += 1
            continue
        if perigee_km < min_perigee_km:
            counts["low_perigee"] += 1
            continue
        satellite_id = name if name not in used_ids else f"{name} [{int(omm['NORAD_CAT_ID'])}]"
        used_ids.add(satellite_id)
        per_mission[mission["mission"]] += 1
        swath_km, field_of_view = _swath(mission, perigee_km, apogee_km)
        satellites.append(
            Satellite(
                id=satellite_id,
                norad_id=int(omm["NORAD_CAT_ID"]),
                group=omm.get("_GROUP"),
                mission=mission["mission"],
                swath_km=swath_km,
                fov_deg=field_of_view,
                min_sun_elev_deg=float(mission["min_sun_elev_deg"]["value"]),
                elements=omm,
            )
        )
    counts["operational"] = len(satellites)
    counts["per_mission"] = dict(per_mission)
    counts["unmatched_families"] = dict(unmatched_families.most_common(40))
    counts["unmatched_names"] = sorted(set(unmatched_names))[:200]
    counts["excluded_names"] = sorted(excluded_found)
    counts["exclusions_not_found"] = sorted(excluded - {name.upper() for name in excluded_found})
    return satellites, counts


def fleet_file_entries(satellites: Iterable[Satellite]) -> list[dict]:
    """Return the satellites as the releases write them to the fleet file ``arf_fleet_<tag>.json``."""
    return [
        {
            "id": satellite.id,
            "norad_id": satellite.norad_id,
            "mission": satellite.mission,
            "swath_km": round(satellite.swath_km, 3),
            "fov_deg": satellite.fov_deg,
            "min_sun_elev_deg": satellite.min_sun_elev_deg,
        }
        for satellite in satellites
    ]
