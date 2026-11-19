"""Downlink: when each satellite can be received by a ground station, and how long an image waits for that.

Reproduces the downlink part of Cell 5 of releases 1.5.3 to 1.5.6.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence

import pandas as pd

from arf.access.geometry import haversine_km, visibility_radius_km
from arf.access.scan import Ephemeris, Imager, ScanSettings
from arf.regions import GroundStation

DOWNLINK_COLUMNS = ["sat", "gs", "t_min", "elev"]


def _coarse_windows(
    imager: Imager,
    ephemeris: Ephemeris,
    station_points: Mapping[str, tuple[float, float]],
    coarse_seconds: Sequence[int],
    reach_km: float,
    half_s: int,
    window_s: int,
) -> dict[str, list[tuple[int, int]]]:
    """Return, per station within reach of the satellite at a coarse sample, the spans of seconds to look at closely."""
    windows: dict[str, list[tuple[int, int]]] = {name: [] for name in station_points}
    for second in coarse_seconds:
        latitude, longitude, _ = ephemeris.subsatellite(imager.id, second / 60.0)
        for name, (station_latitude, station_longitude) in station_points.items():
            if haversine_km(latitude, longitude, station_latitude, station_longitude) <= reach_km:
                windows[name].append((max(0, second - half_s), min(window_s, second + half_s)))
    return {name: spans for name, spans in windows.items() if spans}


def detect_downlink_contacts(
    imagers: Sequence[Imager],
    ephemeris: Ephemeris,
    stations: Mapping[str, GroundStation],
    settings: ScanSettings,
    min_elevation_deg: float = 15.0,
    coarse_step_min: int = 5,
    fine_step_s: int = 10,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Find every minute in which a satellite is at or above the elevation mask of a ground station.

    A coarse stage, every ``coarse_step_min``, flags a station when the sub-satellite point lies within the
    visibility radius of the highest satellite (15 km higher) plus the distance a satellite travels in half a coarse
    step; the coarse sample nearest to any contact is then flagged, so no pass falls between samples. A fine stage,
    every ``fine_step_s`` within half a coarse step of a flag, computes the exact elevation. A contact at time t is
    recorded against the next whole minute, so that a wait derived from this table is never too short. A contact
    above the mask for less than ``fine_step_s`` can be missed.

    Returns:
        The contacts, one row per (satellite, station, minute) with the highest elevation seen in that minute,
        sorted, and the counts ``coarse_hits``, ``fine_samples`` and the radius ``reach_km``.
    """
    window_s = settings.window_hours * 60 * 60
    step_s = coarse_step_min * 60
    reach_km = (
        visibility_radius_km(settings.max_altitude_km + 15.0, min_elevation_deg)
        + settings.max_ground_speed_km_min * coarse_step_min / 2.0
    )
    station_points = {name: (float(s.latitude_deg), float(s.longitude_deg)) for name, s in stations.items()}
    counts: dict[str, float] = {"coarse_hits": 0, "fine_samples": 0, "reach_km": reach_km}
    best: dict[tuple[str, str, int], float] = {}
    coarse_seconds = list(range(0, window_s + 1, step_s))
    if coarse_seconds[-1] != window_s:
        coarse_seconds.append(window_s)
    half = step_s // 2 + fine_step_s  # every contact instant lies within half a step of a flagged coarse sample

    for imager in imagers:
        windows = _coarse_windows(imager, ephemeris, station_points, coarse_seconds, reach_km, half, window_s)
        counts["coarse_hits"] += sum(len(spans) for spans in windows.values())
        if not windows:
            continue
        fine_seconds = sorted(
            {
                second
                for spans in windows.values()
                for low, high in spans
                for second in range(low, high + 1, fine_step_s)
            }
        )
        counts["fine_samples"] += len(fine_seconds)
        for second in fine_seconds:
            in_window = [name for name, spans in windows.items() if any(low <= second <= high for low, high in spans)]
            if not in_window:
                continue
            for name, elevation in zip(
                in_window, ephemeris.elevations_deg(imager.id, float(second), in_window), strict=True
            ):
                if elevation >= min_elevation_deg:
                    key = (imager.id, name, -(-second // 60))  # the next whole minute
                    if elevation > best.get(key, -90.0):
                        best[key] = elevation
    if best:
        contacts = pd.DataFrame(
            [
                {"sat": satellite, "gs": station, "t_min": int(minute), "elev": elevation}
                for (satellite, station, minute), elevation in sorted(best.items())
            ]
        )
    else:
        contacts = pd.DataFrame(columns=DOWNLINK_COLUMNS)
    return contacts, counts


class DownlinkSchedule:
    """The contact minutes of each satellite, for looking up how long an image waits."""

    def __init__(self, contacts: pd.DataFrame):
        """Index the contacts by satellite."""
        self._minutes: dict[str, list[int]] = {}
        if len(contacts) > 0:
            self._minutes = contacts.groupby("sat")["t_min"].apply(lambda minutes: sorted(minutes.unique())).to_dict()

    def wait_min(self, satellite_id: str, acquired_min: int) -> int | None:
        """Return the minutes from an acquisition to the satellite's first contact at or after it, or ``None``."""
        minutes = self._minutes.get(satellite_id, [])
        position = bisect.bisect_left(minutes, acquired_min)
        return int(minutes[position] - acquired_min) if position < len(minutes) else None


def downlink_reward(wait_min: int | None, max_wait_min: float) -> float:
    """Return the downlink term of the selection function: 1 for no wait, falling linearly to 0 at ``max_wait_min``.

    An image with no contact in the window has reward 0.
    """
    if wait_min is None:
        return 0.0
    return max(0.0, 1.0 - wait_min / max_wait_min)
