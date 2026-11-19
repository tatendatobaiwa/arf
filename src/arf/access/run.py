"""The access computation from end to end: fleet altitudes, scan, downlink contacts, refinement."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from arf.access.downlink import DownlinkSchedule, detect_downlink_contacts
from arf.access.geometry import circular_orbit_period_min
from arf.access.refine import Refinement, refine_access
from arf.access.scan import AccessScan, Ephemeris, Imager, RegionIndex, ScanSettings, detect_access
from arf.regions import GroundStation


class FleetEphemeris(Ephemeris, Protocol):
    """An ephemeris that also knows the start of the window and the altitude of each satellite at that instant."""

    @property
    def start_datetime64(self) -> np.datetime64:
        """The start of the window."""

    def altitudes_at_start_km(self) -> list[float]:
        """Return the altitude of every satellite at the start of the window, in fleet order."""


@dataclass
class AccessResult:
    """Everything the access computation produces.

    Attributes:
        altitudes_km: Altitude of each satellite at the start of the window.
        median_altitude_km: Their median (``median_alt_km`` in ``arf_cell2_summary.json``).
        orbit_period_min: The period of a circular orbit at the median altitude (``t_max_wait_min``): the wait at
            which the downlink reward reaches zero.
        settings: The scan settings with the derived search radii.
        scan: The access scan (``arf_access_events.csv``).
        contacts: The downlink contacts (``arf_downlink_events.csv``).
        downlink_counts: Counts of the downlink scan.
        refinement: The refined events (``arf_refined_events.csv``) and the counts of what was dropped.
    """

    altitudes_km: list[float]
    median_altitude_km: float
    orbit_period_min: float
    settings: ScanSettings
    scan: AccessScan
    contacts: pd.DataFrame
    downlink_counts: dict
    refinement: Refinement


def compute_access(
    imagers: Sequence[Imager],
    ephemeris: FleetEphemeris,
    region: RegionIndex,
    stations: Mapping[str, GroundStation],
    window_hours: int = 48,
    swath_half_angle_deg: float = 9.22,
    min_elevation_deg: float = 15.0,
    exclusions: Sequence[Mapping] = (),
    legacy_end_caps: bool = False,
) -> AccessResult:
    """Run the access scan, the downlink scan and the refinement for a fleet over a region.

    Args:
        imagers: The satellites, in fleet order.
        ephemeris: Their positions, with the ground stations (:class:`arf.access.orekit.OrekitEphemeris`).
        region: The cells of the region.
        stations: The ground stations, by name.
        window_hours: Length of the window.
        swath_half_angle_deg: Half-angle of the legacy dense-constellation sensor; unused when every satellite has
            its own swath.
        min_elevation_deg: Elevation mask of the ground stations.
        exclusions: Sourced periods without acquisition; pass none for the dense constellation.
        legacy_end_caps: Use the rule of release 1.5.0 and earlier for strip ends.

    Raises:
        ValueError: If two satellites share an identifier.
    """
    identifiers = [imager.id for imager in imagers]
    if len(set(identifiers)) != len(identifiers):
        # The ephemeris and the strip samples are keyed by satellite, so a shared name would mix two orbits.
        raise ValueError("two satellites of the fleet share an identifier")
    altitudes = ephemeris.altitudes_at_start_km()
    median_altitude = float(np.median(altitudes))
    period = circular_orbit_period_min(median_altitude)
    settings = ScanSettings.for_fleet(imagers, altitudes, swath_half_angle_deg, window_hours)
    scan = detect_access(imagers, ephemeris, settings, region)
    contacts, downlink_counts = detect_downlink_contacts(imagers, ephemeris, stations, settings, min_elevation_deg)
    refinement = refine_access(
        scan,
        imagers,
        region,
        settings,
        DownlinkSchedule(contacts),
        ephemeris.start_datetime64,
        period,
        exclusions,
        legacy_end_caps,
    )
    return AccessResult(altitudes, median_altitude, period, settings, scan, contacts, downlink_counts, refinement)
