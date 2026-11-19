"""Refinement: which cells each strip images, at what off-nadir angle, in daylight, and when it is delivered.

Each event of the access scan is a one-minute strip. A cell is imaged when the ground track passes abeam of it, so
its off-nadir angle comes from its great-circle distance to the track. Satellites with their own swath use the
distance to the arcs between the strip's samples, and each observation belongs to the one strip that contains its
closest approach; the legacy dense-constellation sensor uses the nearest sample. Only sunlit cells are kept, by each
satellite's daylight limit, and passes inside a sourced acquisition exclusion are dropped.

Reproduces ``refine_access_events`` of Cell 5 of releases 1.5.3 to 1.5.6. The output is the table every later step
reads, ``arf_refined_events.csv``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from arf.access.downlink import DownlinkSchedule, downlink_reward
from arf.access.geometry import haversine_km, min_track_distance_km, off_nadir_angle_deg, solar_elevation_deg
from arf.access.scan import AccessScan, Imager, RegionIndex, ScanSettings
from arf.fleet.elements import EARTH_RADIUS_KM

REFINED_COLUMNS = [
    "sat",
    "tile",
    "t_min",
    "A_nadir_deg",
    "D_downlink",
    "T_wait_min",
    "theta_max_deg",
    "sun_elev_deg",
]


@dataclass
class Refinement:
    """The refined events and what was dropped on the way.

    Attributes:
        events: One row per (satellite, minute, cell) observation: off-nadir angle, downlink reward, wait to the
            next contact (``nan`` if none), the satellite's swath-edge angle, and the Sun's elevation (``nan`` for a
            satellite with no daylight limit).
        counts: ``events_dark`` (strips with no sunlit cell), ``rows_dark``, ``rows_lit``,
            ``rows_end_cap_excluded`` (observations the rule of release 1.5.0 would have added),
            ``rows_acquisition_excluded``, ``passes_acquisition_excluded`` and, per exclusion,
            ``acquisition_exclusions_hit``.
    """

    events: pd.DataFrame
    counts: dict


def acquisition_exclusion_windows(
    imagers: Sequence[Imager], exclusions: Sequence[Mapping], start_utc: np.datetime64
) -> dict[str, list[tuple[float, float, int]]]:
    """Return, per satellite, the exclusion periods as (start, end, entry) in minutes from the start of the window.

    A satellite identifier may carry a catalogue number in brackets, added when two satellites share a name; the
    name before it is matched.
    """
    windows: dict[str, list[tuple[float, float, int]]] = {}
    for entry, exclusion in enumerate(exclusions):
        start = float((np.datetime64(exclusion["start_utc"]) - start_utc) / np.timedelta64(1, "s")) / 60.0
        end = float((np.datetime64(exclusion["end_utc"]) - start_utc) / np.timedelta64(1, "s")) / 60.0
        for imager in imagers:
            name = str(imager.id).split(" [")[0].strip()
            if re.match(exclusion["name_pattern"], name, flags=re.IGNORECASE):
                windows.setdefault(imager.id, []).append((start, end, entry))
    return windows


def _own_swath_angles(samples, imager, latitude, longitude, exclude_end_caps):
    """Return the off-nadir angles of candidate cells for a satellite with its own swath, and its end-cap cells.

    The second value marks the cells that only the rule of release 1.5.0 would have kept in this strip (their
    closest approach lies in a neighbouring strip); ``None`` when there is none to consider.
    """
    track_latitude, track_longitude, track_radius = samples
    distance, radius_near = min_track_distance_km(
        track_latitude, track_longitude, track_radius, latitude, longitude, EARTH_RADIUS_KM, exclude_end_caps
    )
    angle = np.full(latitude.size, np.inf)  # inf: closest approach outside this strip
    finite = np.isfinite(distance)
    if finite.any():
        angle[finite] = off_nadir_angle_deg(distance[finite], EARTH_RADIUS_KM, radius_near[finite])
    end_cap = None
    if exclude_end_caps and (~finite).any():
        old_distance, old_radius = min_track_distance_km(
            track_latitude, track_longitude, track_radius, latitude[~finite], longitude[~finite], EARTH_RADIUS_KM, False
        )
        edge = float(off_nadir_angle_deg(imager.swath_km / 2.0, EARTH_RADIUS_KM, float(np.mean(track_radius))))
        end_cap = np.zeros(latitude.size, dtype=bool)
        end_cap[~finite] = off_nadir_angle_deg(old_distance, EARTH_RADIUS_KM, old_radius) <= edge
    return angle, end_cap


def _nearest_sample_angles(samples, latitude, longitude):
    """Return the off-nadir angles of candidate cells from the nearest strip sample (the legacy sensor)."""
    angle = np.full(latitude.size, np.inf)
    for track_latitude, track_longitude, track_radius in zip(*samples, strict=True):
        distance = haversine_km(track_latitude, track_longitude, latitude, longitude)
        angle = np.minimum(angle, off_nadir_angle_deg(distance, EARTH_RADIUS_KM, track_radius))
    return angle


def _new_counts() -> dict:
    return {
        "events_dark": 0,
        "rows_dark": 0,
        "rows_lit": 0,
        "rows_end_cap_excluded": 0,
        "rows_acquisition_excluded": 0,
        "passes_acquisition_excluded": 0,
        "acquisition_exclusions_hit": {},
    }


def _assemble(rows: dict, region: RegionIndex) -> pd.DataFrame:
    """Build the refined table from per-strip chunks, with compact column types."""
    if len(rows["cells"]) == 0:
        return pd.DataFrame(columns=REFINED_COLUMNS)
    sizes = np.array(rows["size"], dtype=np.int64)
    satellites = sorted(set(rows["sat"]))
    code = {satellite: position for position, satellite in enumerate(satellites)}
    satellite_codes = np.repeat(np.array([code[satellite] for satellite in rows["sat"]], dtype=np.int32), sizes)
    return pd.DataFrame(
        {
            "sat": pd.Categorical.from_codes(satellite_codes, categories=satellites),
            "tile": pd.Categorical.from_codes(np.concatenate(rows["cells"]), categories=region.cells),
            "t_min": np.repeat(np.array(rows["t_min"], dtype=np.int32), sizes),
            "A_nadir_deg": np.concatenate(rows["angle"]).astype(np.float64),
            "D_downlink": np.repeat(np.array(rows["reward"], dtype=np.float64), sizes),
            "T_wait_min": np.repeat(np.array(rows["wait"], dtype=np.float64), sizes),
            "theta_max_deg": np.repeat(np.array(rows["edge"], dtype=np.float64), sizes),
            "sun_elev_deg": np.concatenate(rows["sun"]).astype(np.float64),
        }
    )


def refine_access(
    scan: AccessScan,
    imagers: Sequence[Imager],
    region: RegionIndex,
    settings: ScanSettings,
    schedule: DownlinkSchedule,
    start_utc: np.datetime64,
    max_wait_min: float,
    exclusions: Sequence[Mapping] = (),
    legacy_end_caps: bool = False,
) -> Refinement:
    """Apply the exact field-of-view, daylight and acquisition filters to every strip, and attach the downlink wait.

    Args:
        scan: The access scan.
        imagers: The satellites.
        region: The cells of the region, as given to the scan.
        settings: The scan settings, for the search radius and the legacy half-angle.
        schedule: The downlink contacts.
        start_utc: The start of the window.
        max_wait_min: The wait at which the downlink reward reaches zero; the releases use one orbital period.
        exclusions: Sourced periods in which a satellite did not acquire
            (:data:`arf.fleet.missions.ACQUISITION_EXCLUSIONS`); pass none for the dense constellation.
        legacy_end_caps: Use the rule of release 1.5.0 and earlier for strip ends.
    """
    by_id = {imager.id: imager for imager in imagers}
    exclude_end_caps = not legacy_end_caps
    windows = acquisition_exclusion_windows(imagers, exclusions, start_utc)
    counts = _new_counts()
    rows: dict[str, list] = {
        key: [] for key in ("sat", "t_min", "size", "reward", "wait", "edge", "cells", "angle", "sun")
    }
    for row in scan.events.itertuples(index=False):
        satellite_id, minute = row.sat, int(row.t_min)
        samples = scan.strip_samples[(satellite_id, minute)]
        imager = by_id.get(satellite_id, Imager(satellite_id))
        excluded_by = next(
            (entry for start, end, entry in windows.get(satellite_id, ()) if start <= minute < end), None
        )
        candidates = region.cells_near(samples[0], samples[1], settings.footprint_radius_km)
        if candidates.size == 0:
            continue
        latitude, longitude = region.latitude[candidates], region.longitude[candidates]
        end_cap = None
        if imager.swath_km:
            angle, end_cap = _own_swath_angles(samples, imager, latitude, longitude, exclude_end_caps)
            # The off-nadir angle of this satellite's swath edge, at its mean radius over the strip.
            edge = float(off_nadir_angle_deg(imager.swath_km / 2.0, EARTH_RADIUS_KM, float(np.mean(samples[2]))))
        else:
            angle = _nearest_sample_angles(samples, latitude, longitude)
            edge = settings.swath_half_angle_deg
        in_view = angle <= edge
        time_utc = start_utc + np.timedelta64(round(minute * 60), "s")
        if end_cap is not None and end_cap.any() and excluded_by is None:
            # Counted only where the rule of release 1.5.0 would also have kept the row, that is, sunlit.
            if imager.min_sun_elev_deg is not None:
                sun_at_caps = solar_elevation_deg(time_utc, latitude[end_cap], longitude[end_cap])
                counts["rows_end_cap_excluded"] += int((sun_at_caps >= imager.min_sun_elev_deg).sum())
            else:
                counts["rows_end_cap_excluded"] += int(end_cap.sum())
        if not in_view.any():
            continue
        sun = np.full(candidates.size, np.nan)
        if imager.min_sun_elev_deg is not None:
            sun[in_view] = solar_elevation_deg(time_utc, latitude[in_view], longitude[in_view])
            lit = in_view & (sun >= imager.min_sun_elev_deg)
            counts["rows_dark"] += int(in_view.sum() - lit.sum())
            counts["rows_lit"] += int(lit.sum())
            if not lit.any():
                counts["events_dark"] += 1
                continue
            in_view = lit
        if excluded_by is not None:
            dropped = int(in_view.sum())
            counts["rows_acquisition_excluded"] += dropped
            counts["passes_acquisition_excluded"] += 1
            hit = counts["acquisition_exclusions_hit"].setdefault(str(excluded_by), {"passes": 0, "rows": 0})
            hit["passes"] += 1
            hit["rows"] += dropped
            continue
        wait = schedule.wait_min(satellite_id, minute)
        rows["cells"].append(candidates[in_view].astype(np.int32))
        rows["angle"].append(angle[in_view])
        rows["sun"].append(sun[in_view])
        rows["edge"].append(edge)
        rows["sat"].append(satellite_id)
        rows["t_min"].append(minute)
        rows["size"].append(int(in_view.sum()))
        rows["reward"].append(float(downlink_reward(wait, max_wait_min)))
        rows["wait"].append(np.nan if wait is None else float(wait))
    return Refinement(_assemble(rows, region), counts)
