"""The imaging missions of the real fleets: which satellites, what swath, what daylight limit.

Every value carries its status and its source. ``sourced`` means it is stated on the cited page, with the sentence
quoted; ``assumption`` means no source was found, and the value is labelled so. The tables are those of Cell 0 of
releases 1.5.3 to 1.5.6, and ``tests/parity`` checks that they are equal.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from arf.fleet.elements import EARTH_RADIUS_KM

#: CelesTrak GP groups read for each fleet ("resource" is Earth Resources, "planet" is Planet).
CELESTRAK_GROUPS = {
    "earth-resources": ["resource"],
    "planet": ["planet"],
    "earth-resources+planet": ["resource", "planet"],
}

_ACCESSED = "2026-10-08"
_PLANETSCOPE_SPECIFICATION = "https://assets.planet.com/docs/Planet_PSScene_Imagery_Product_Spec_letter_screen.pdf"
_PLANETSCOPE_FIELD_OF_VIEW_QUOTE = (
    "Field of View 3.0° (swath) 1.0° (scene length) 3.0° (swath) 2.0° (scene length) "
    "4.0° (swath) 2.3° (scene length) [PS2, PS2.SD, PSB.SD]"
)


def mission_table(planet_swath_case: str = "conservative", assumed_min_sun_elev_deg: float = 5.0) -> list[dict]:
    """Return the missions of the real fleets.

    Each entry has ``mission``, ``name_pattern`` (a regular expression matched against the CelesTrak object name),
    either ``swath_km`` or ``fov_deg``, and ``min_sun_elev_deg``; each of the last three is a mapping with
    ``value``, ``status`` and the source.

    PlanetScope gives its swath as a field of view across the swath (PSScene specification, June 2026, Table 2-A):
    3.0 degrees for PS2.SD (Dove-R) and 4.0 degrees for PSB.SD (SuperDove). Which Dove carries which instrument
    cannot be told from its name, so the two cases bracket it, and each satellite's swath is computed from the field
    of view at its own altitude (:func:`swath_from_fov_km`).

    Args:
        planet_swath_case: ``"conservative"`` (3.0 degrees) or ``"optimistic"`` (4.0 degrees).
        assumed_min_sun_elev_deg: The daylight limit assumed for Sentinel-2, for which no limit was found in a
            source; the Landsat limit of 5 degrees is borrowed by default.
    """
    if planet_swath_case == "conservative":
        planetscope_field_of_view = {
            "value": 3.0,
            "status": "sourced (lower bound: PS2.SD Dove-R)",
            "accessed": "2026-10-09",
            "source_url": _PLANETSCOPE_SPECIFICATION,
            "quote": _PLANETSCOPE_FIELD_OF_VIEW_QUOTE,
        }
    else:
        planetscope_field_of_view = {
            "value": 4.0,
            "status": "sourced (upper bound: PSB.SD SuperDove)",
            "accessed": "2026-10-09",
            "source_url": _PLANETSCOPE_SPECIFICATION,
            "quote": _PLANETSCOPE_FIELD_OF_VIEW_QUOTE,
        }
    return [
        {
            "mission": "Sentinel-2",
            "name_pattern": r"^SENTINEL[- ]?2[A-Z]?$",
            "swath_km": {
                "value": 290.0,
                "status": "sourced",
                "accessed": _ACCESSED,
                "source_url": "https://sentiwiki.copernicus.eu/web/s2-mission",
                "quote": "The orbital swath width is 290 km.",
            },
            "min_sun_elev_deg": {
                "value": assumed_min_sun_elev_deg,
                "status": "assumption",
                "note": "Source states MSI collects reflected sunlight but gives no sun-elevation "
                "limit; the Landsat 5 deg criterion is borrowed (ASSUMED_MIN_SUN_ELEV_DEG).",
            },
        },
        {
            "mission": "Landsat 8/9",
            "name_pattern": r"^LANDSAT[- ]?[89]$",
            "swath_km": {
                "value": 185.0,
                "status": "sourced",
                "accessed": _ACCESSED,
                "source_url": "https://science.nasa.gov/mission/landsat-9",
                "quote": "Swath width: 185 km (115 mi)",
            },
            "min_sun_elev_deg": {
                "value": 5.0,
                "status": "sourced",
                "accessed": _ACCESSED,
                "source_url": "https://www.usgs.gov/landsat-missions/landsat-9",
                "quote": "including islands at solar elevation angles greater than 5 degrees",
            },
        },
        {
            "mission": "PlanetScope",
            "name_pattern": r"^FLOCK",
            "fov_deg": planetscope_field_of_view,
            "min_sun_elev_deg": {
                "value": 10.0,
                "status": "sourced",
                "accessed": _ACCESSED,
                "source_url": "https://docs.planet.com/data/imagery/planetscope/",
                "quote": "All of the Doves have a system constraint to image regions that have a "
                "sun elevation angle greater than 10°.",
            },
        },
    ]


#: Periods in which a satellite is tracked but, by its operator's statement, does not acquire nominal imagery.
#: Passes in ``[start_utc, end_utc)`` are dropped when the access events are refined.
ACQUISITION_EXCLUSIONS = [
    {
        "name_pattern": r"^SENTINEL[- ]?2A$",
        "start_utc": "2026-01-13T00:00:00",
        "end_utc": "2026-03-09T00:00:00",
        "status": "sourced",
        "accessed": "2026-10-14",
        "source_url": "https://sentinels.copernicus.eu/-/sentinel-2a-and-sentinel-2b-tandem-acquisitions",
        "quote": "From 13 January to 08 March 2026, Sentinel-2A will temporarily pause its nominal acquisition "
        "activities.",
        "note": "Sentinel-2A/2B tandem calibration (page published 18 December 2025). The end date is inclusive, so "
        "the window ends at 9 March 00:00 UTC. Images taken in vicarious (VIC) mode during the drift were "
        "distributed to expert users only and are not modelled.",
    },
]


def match_mission(name: str, missions: Sequence[dict]) -> dict | None:
    """Return the first mission whose name pattern matches a satellite's name (case ignored), or ``None``."""
    for mission in missions:
        if re.match(mission["name_pattern"], str(name).strip(), flags=re.IGNORECASE):
            return mission
    return None


def swath_from_fov_km(altitude_km: float, fov_deg: float, earth_radius_km: float = EARTH_RADIUS_KM) -> float:
    """Return the ground swath of a nadir-pointing sensor from its full field of view.

    Over a sphere, the Earth-central angle from the sub-satellite point to the edge of the field is
    ``arcsin((R + h) / R * sin(fov / 2)) - fov / 2``; the swath is twice that angle times the radius.

    Args:
        altitude_km: The altitude of the satellite.
        fov_deg: The full field of view across the swath, in degrees.
        earth_radius_km: The radius of the sphere.

    Raises:
        ValueError: If the field of view reaches beyond the horizon.
    """
    half_angle = math.radians(fov_deg / 2.0)
    sine = (earth_radius_km + altitude_km) / earth_radius_km * math.sin(half_angle)
    if sine >= 1.0:
        raise ValueError(f"field of view {fov_deg} deg reaches beyond the horizon at {altitude_km} km")
    return 2.0 * earth_radius_km * (math.asin(sine) - half_angle)
