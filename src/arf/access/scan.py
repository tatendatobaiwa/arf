"""The access scan: which satellite passes over the region in which minute.

Checking every satellite against every cell at every instant costs satellites x instants x cells. The scan avoids it
in two stages. A coarse stage, every five minutes, keeps a satellite when its sub-satellite point falls in the
region's box inflated by the footprint radius plus the distance a satellite travels in half a coarse step, so the
coarse sample nearest to any pass falls inside the box and no pass slips between samples. A fine stage then takes each
whole minute near a coarse hit as a candidate strip: the swath swept over [m - 0.5, m + 0.5) minutes, sampled every
two seconds. The search radii are deliberate supersets; the exact filter is applied in :mod:`arf.access.refine`.

Reproduces Cell 4 of releases 1.5.3 to 1.5.6.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from arf.access.geometry import GeoBox, footprint_ground_radius_km, haversine_km, wrap_longitude_difference
from arf.fleet.elements import EARTH_MU_KM3_S2, EARTH_RADIUS_KM

#: Length of a sidereal day, minutes, for the speed of the Earth's surface at the equator.
SIDEREAL_DAY_MIN = 1436.068

ACCESS_COLUMNS = ["sat", "t_min", "lat", "lon", "tiles_hit"]


@dataclass(frozen=True)
class Imager:
    """What the access computation needs to know about a satellite.

    Attributes:
        id: The satellite's identifier, as in the fleet.
        swath_km: Its ground swath. ``None`` for the legacy dense-constellation sensor, which is described by one
            half-angle for the whole fleet.
        min_sun_elev_deg: The sun elevation below which it does not image; ``None`` for no daylight limit.
    """

    id: str
    swath_km: float | None = None
    min_sun_elev_deg: float | None = None


class Ephemeris(Protocol):
    """Where the satellites are. Implemented with Orekit in :mod:`arf.access.orekit`."""

    def subsatellite(self, satellite_id: str, t_min: float) -> tuple[float, float, float]:
        """Return geodetic latitude and longitude in degrees and geocentric radius in km, at minutes from the start."""

    def elevations_deg(self, satellite_id: str, t_s: float, stations: Sequence[str]) -> list[float]:
        """Return the elevation of the satellite above each named ground station, at seconds from the start."""


@dataclass
class ScanSettings:
    """The settings of the access scan and the search radii derived from them.

    Attributes:
        swath_half_angle_deg: Half-angle of the legacy dense-constellation sensor.
        max_altitude_km: Altitude of the highest satellite at the start of the window.
        min_altitude_km: Altitude of the lowest satellite at the start of the window.
        window_hours: Length of the window.
        coarse_step_minutes: Step of the coarse stage.
        strip_sample_seconds: Spacing of the samples along a strip. Two seconds is about 14 km along the track and
            limits the erosion of the swath edge by sampling to about 0.3 km.
        max_swath_km: The widest swath in the fleet, when satellites have their own swaths; the search radius is then
            a superset for every satellite.
        max_ground_speed_km_min: Derived: the fastest speed of a sub-satellite point over the ground, the lowest
            satellite's ground-track speed plus the rotation of the Earth's surface at the equator.
        footprint_radius_km: Derived: the search radius around the track.
        coarse_margin_km: Derived: the inflation of the region's box for the coarse stage.
        strip_margin_km: Derived: the inflation of the region's box for the fine stage.
    """

    swath_half_angle_deg: float
    max_altitude_km: float
    min_altitude_km: float
    window_hours: int = 48
    coarse_step_minutes: int = 5
    strip_sample_seconds: float = 2.0
    max_swath_km: float | None = None
    max_ground_speed_km_min: float = field(init=False)
    footprint_radius_km: float = field(init=False)
    coarse_margin_km: float = field(init=False)
    strip_margin_km: float = field(init=False)

    def __post_init__(self):
        """Derive the speeds and radii."""
        altitude = max(150.0, self.min_altitude_km - 15.0)
        orbital = (
            np.sqrt(EARTH_MU_KM3_S2 / (EARTH_RADIUS_KM + altitude))
            * 60.0
            * EARTH_RADIUS_KM
            / (EARTH_RADIUS_KM + altitude)
        )
        rotation = 2.0 * np.pi * EARTH_RADIUS_KM / SIDEREAL_DAY_MIN
        self.max_ground_speed_km_min = orbital + rotation
        if self.max_swath_km:
            # Half the widest swath plus 5%, plus half the spacing of the samples along the track: a cell between two
            # samples can be that far from both while still on the track.
            self.footprint_radius_km = (
                1.05 * self.max_swath_km / 2.0 + 0.5 * self.max_ground_speed_km_min * self.strip_sample_seconds / 60.0
            )
        else:
            # Footprint of the highest satellite, 15 km higher for the eccentricity of the orbit, plus 5%.
            self.footprint_radius_km = 1.05 * footprint_ground_radius_km(
                self.max_altitude_km + 15.0, self.swath_half_angle_deg
            )
        self.coarse_margin_km = self.footprint_radius_km + self.max_ground_speed_km_min * self.coarse_step_minutes / 2.0
        self.strip_margin_km = self.footprint_radius_km + self.max_ground_speed_km_min * 0.5

    @classmethod
    def for_fleet(
        cls,
        imagers: Iterable[Imager],
        altitudes_km: Sequence[float],
        swath_half_angle_deg: float,
        window_hours: int = 48,
    ) -> ScanSettings:
        """Return the settings for a fleet from the altitudes of its satellites at the start of the window."""
        swaths = [imager.swath_km for imager in imagers if imager.swath_km]
        return cls(
            swath_half_angle_deg=swath_half_angle_deg,
            max_altitude_km=float(np.max(altitudes_km)),
            min_altitude_km=float(np.min(altitudes_km)),
            window_hours=window_hours,
            max_swath_km=float(max(swaths)) if swaths else None,
        )


class RegionIndex:
    """The cells of the region with their centres, ordered by latitude so that cells near a track are found fast.

    Attributes:
        cells: The cell identifiers, ordered by the latitude of their centres (ties in identifier order).
        latitude: The latitudes of the centres, in that order.
        longitude: The longitudes of the centres, in that order.
    """

    def __init__(self, cells: Iterable[str], latitude: Sequence[float], longitude: Sequence[float]):
        """Order the cells, given with their centres, by identifier and then stably by latitude."""
        cells = np.asarray(list(cells))
        latitude, longitude = np.asarray(latitude, dtype=float), np.asarray(longitude, dtype=float)
        by_identifier = np.argsort(cells, kind="stable")
        cells, latitude, longitude = cells[by_identifier], latitude[by_identifier], longitude[by_identifier]
        order = np.argsort(latitude, kind="stable")
        self.cells, self.latitude, self.longitude = cells[order], latitude[order], longitude[order]

    @classmethod
    def from_cells(cls, cells: Iterable[str]) -> RegionIndex:
        """Build the index from H3 cell identifiers. Needs ``h3``."""
        import h3

        cells = sorted(set(cells))
        centres = np.array([h3.cell_to_latlng(cell) for cell in cells])
        return cls(cells, centres[:, 0], centres[:, 1])

    def box(self, margin_km: float) -> GeoBox:
        """Return the box around the cell centres, inflated by a margin."""
        return GeoBox(self.latitude, self.longitude, margin_km)

    def cells_near(self, latitudes, longitudes, radius_km: float) -> np.ndarray:
        """Return the positions of the cells whose centre is within a ground distance of at least one given point."""
        box = GeoBox(latitudes, longitudes, radius_km)
        low = np.searchsorted(self.latitude, box.lat_lo, side="left")
        high = np.searchsorted(self.latitude, box.lat_hi, side="right")
        positions = np.arange(low, high)
        if positions.size and not box.all_longitudes:
            near = np.abs(wrap_longitude_difference(self.longitude[positions], box.lon_centre)) <= box.lon_half
            positions = positions[near]
        if positions.size == 0:
            return positions
        distance = haversine_km(
            np.asarray(latitudes)[:, None],
            np.asarray(longitudes)[:, None],
            self.latitude[positions][None, :],
            self.longitude[positions][None, :],
        ).min(axis=0)
        return positions[distance <= radius_km]


@dataclass
class AccessScan:
    """The result of the access scan.

    Attributes:
        events: One row per strip that comes near the region: satellite, minute, the sub-satellite point at the
            minute, and the number of candidate cells.
        strip_samples: For each (satellite, minute), the latitudes, longitudes and geocentric radii of the strip's
            sub-satellite samples; the refinement reuses them.
        counts: ``coarse_hits``, ``candidate_minutes`` and ``strips_sampled``.
    """

    events: pd.DataFrame
    strip_samples: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, np.ndarray]]
    counts: dict[str, int]


def detect_access(
    imagers: Sequence[Imager], ephemeris: Ephemeris, settings: ScanSettings, region: RegionIndex
) -> AccessScan:
    """Find the one-minute strips of every satellite that come near the region.

    Args:
        imagers: The satellites, in fleet order.
        ephemeris: Where they are.
        settings: The scan settings.
        region: The cells of the region.
    """
    window_min = settings.window_hours * 60
    step = settings.coarse_step_minutes
    coarse_times = list(range(0, window_min + 1, step))
    if coarse_times[-1] != window_min:
        coarse_times.append(window_min)
    offsets = np.arange(-0.5, 0.5, settings.strip_sample_seconds / 60.0)
    if settings.max_swath_km:
        # Include the strip's end point, the next strip's first sample, so that the track arcs of consecutive strips
        # join with no gap.
        offsets = np.append(offsets, 0.5)
    coarse_box, strip_box = region.box(settings.coarse_margin_km), region.box(settings.strip_margin_km)

    events, strip_samples = [], {}
    counts = {"coarse_hits": 0, "candidate_minutes": 0, "strips_sampled": 0}
    for imager in imagers:
        hits = []
        for minute in coarse_times:
            latitude, longitude, _ = ephemeris.subsatellite(imager.id, minute)
            if coarse_box.contains(latitude, longitude):
                hits.append(minute)
        counts["coarse_hits"] += len(hits)
        if not hits:
            continue
        minutes = sorted(
            {minute for hit in hits for minute in range(max(0, hit - step), min(window_min, hit + step) + 1)}
        )
        counts["candidate_minutes"] += len(minutes)
        for minute in minutes:
            mid_latitude, mid_longitude, _ = ephemeris.subsatellite(imager.id, minute)
            if not strip_box.contains(mid_latitude, mid_longitude):
                continue  # no part of this minute's strip can reach the region
            counts["strips_sampled"] += 1
            times = np.clip(minute + offsets, 0.0, float(window_min))
            samples = np.array([ephemeris.subsatellite(imager.id, time) for time in times])
            near = region.cells_near(samples[:, 0], samples[:, 1], settings.footprint_radius_km)
            if near.size:
                strip_samples[(imager.id, minute)] = (samples[:, 0], samples[:, 1], samples[:, 2])
                events.append(
                    {
                        "sat": imager.id,
                        "t_min": minute,
                        "lat": mid_latitude,
                        "lon": mid_longitude,
                        "tiles_hit": int(near.size),
                    }
                )
    return AccessScan(pd.DataFrame(events, columns=ACCESS_COLUMNS), strip_samples, counts)
