"""The regions the registered releases define, as data.

A region is given in one of three ways: a polygon, a line of waypoints buffered into a corridor, or one or more
protected areas of the World Database on Protected Areas (WDPA), whose boundaries are read from files the user
downloads (see :mod:`arf.boundaries`). The values below are those of ``ROI_DEFINITIONS`` in Cell 1 of releases
1.5.3 to 1.5.6; ``tests/parity`` checks that they are equal.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GroundStation:
    """A receiving station: geodetic latitude and longitude in degrees, altitude in metres."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float


@dataclass(frozen=True)
class WdpaSite:
    """One protected area of the WDPA and the checks that identify its boundary file.

    Attributes:
        site_id: The WDPA site identifier, as in ``https://www.protectedplanet.net/<site_id>``.
        name: The name of the site, used in messages.
        designation_contains: Text the designation in the file must contain (several designations can overlap under
            similar names, for instance a Ramsar site and a World Heritage Site). ``None`` records the designation
            without checking it.
        expected_gis_area_km2: The area of the WDPA polygon (``GIS_AREA``) when the region was defined. A different
            value in a later WDPA release is reported as a warning, since the boundary may have been revised.
        expected_rep_area_km2: The reported area on the site's Protected Planet page. The file's ``REP_AREA`` must
            match it, which identifies the site when no designation is given.
        source_url: The page the expected values were read from.
        accessed: The date that page was read (ISO format).
    """

    site_id: int
    name: str
    designation_contains: str | None = None
    expected_gis_area_km2: float | None = None
    expected_rep_area_km2: float | None = None
    source_url: str | None = None
    accessed: str | None = None


@dataclass(frozen=True)
class Region:
    """A region to be mapped.

    Exactly one of ``polygon``, ``waypoints`` (with ``buffer_km``) and ``wdpa_sites`` is given.

    Attributes:
        key: The name the releases use for the region in their settings (``ACTIVE_ROI``).
        short_name: The label on maps.
        description: One or two sentences on what the region is.
        polygon: The corners of the region as (latitude, longitude) pairs in degrees.
        waypoints: A line of (latitude, longitude) pairs, buffered by ``buffer_km`` on each side.
        buffer_km: The half-width of the corridor around ``waypoints``, in kilometres.
        wdpa_sites: The protected areas whose union is the region.
        ground_stations: The receiving stations, by name.
        element_source: Where the releases take the orbital elements from for this region.
        historical_epoch: The epoch (UTC, ISO format) of a historical event the region is tied to, if any.
    """

    key: str
    short_name: str
    description: str
    polygon: tuple[tuple[float, float], ...] | None = None
    waypoints: tuple[tuple[float, float], ...] | None = None
    buffer_km: float | None = None
    wdpa_sites: tuple[WdpaSite, ...] = ()
    ground_stations: dict[str, GroundStation] = field(default_factory=dict)
    element_source: str = "celestrak_live"
    historical_epoch: str | None = None


def _botswana_stations() -> dict[str, GroundStation]:
    """Return the notional Botswana receiving sites (approximate coordinates).

    These are not the operators' ground networks: Sentinel-2, Landsat and PlanetScope downlink to their own stations.
    The releases use them so that downlink waits reflect a ground segment operated from Botswana.
    """
    return {
        "BIUST": GroundStation(-22.59, 27.14, 994.0),
        "Kasane": GroundStation(-17.80, 25.15, 930.0),
        "Kebonye": GroundStation(-24.84, 25.84, 994.0),
    }


#: The regions of releases 1.5.3 to 1.5.6, by key.
REGIONS: dict[str, Region] = {
    region.key: region
    for region in (
        Region(
            key="botswana_full",
            short_name="Botswana (bounding box)",
            description="Botswana full national boundary (bounding rectangle)",
            polygon=(
                (-17.6618156877, 19.8954577979),
                (-17.6618156877, 29.4321883481),
                (-26.8285429827, 29.4321883481),
                (-26.8285429827, 19.8954577979),
            ),
            ground_stations=_botswana_stations(),
        ),
        Region(
            key="nepal_trishuli_corridor",
            short_name="Trishuli corridor",
            description=(
                "2026 Nepal Trishuli/Lende Khola debris flow and flash flood corridor. "
                "Buffered 7.5km either side of the waypoint polyline."
            ),
            waypoints=(
                (28.27778, 85.37778),  # Rasuwagadhi
                (28.1635, 85.3363),  # Syaphru / Syabrubesi
                (27.9833, 85.1833),  # Betrawati
                (27.8881907, 85.1340051),  # Devighat
            ),
            buffer_km=7.5,
            ground_stations={"NAST_Khumaltar": GroundStation(27.65479, 85.32882, 1335.0)},
            element_source="spacetrack_history",
            historical_epoch="2026-08-26T02:52:00",
        ),
        Region(
            key="ckgr_okavango",
            short_name="CKGR + Okavango Delta",
            description=(
                "Central Kalahari Game Reserve and the Okavango Delta World Heritage Site, "
                "mosaicked together from their official WDPA boundaries."
            ),
            wdpa_sites=(
                WdpaSite(
                    7510,
                    "Central Kalahari Game Reserve",
                    designation_contains="Game Reserve",
                    expected_gis_area_km2=52225.35,
                ),
                WdpaSite(
                    555577555,
                    "Okavango Delta (World Heritage Site)",
                    designation_contains="World Heritage",
                    expected_gis_area_km2=20551.25,
                ),
            ),
            ground_stations=_botswana_stations(),
        ),
        Region(
            key="lower_zambezi",
            short_name="Lower Zambezi NP",
            description="Lower Zambezi protected area (Zambia), from its official WDPA boundary.",
            wdpa_sites=(
                WdpaSite(
                    7962,
                    "Lower Zambezi",
                    expected_rep_area_km2=4143.0,
                    source_url="https://www.protectedplanet.net/7962",
                    accessed="2026-10-08",
                ),
            ),
            ground_stations=_botswana_stations(),
        ),
        Region(
            key="karoo_nature_reserve",
            short_name="Karoo NR",
            description=(
                "Karoo Nature Reserve (South Africa), from its official WDPA boundary. "
                "32 H3 res-7 tiles with the October 2026 WDPA release (167 km2 tiled for a "
                "143 km2 site: small sites are quantised to whole tiles)."
            ),
            wdpa_sites=(
                WdpaSite(
                    555563860,
                    "Karoo Nature Reserve",
                    expected_rep_area_km2=143.15,
                    source_url="https://www.protectedplanet.net/555563860",
                    accessed="2026-10-08",
                ),
            ),
            ground_stations=_botswana_stations(),
        ),
    )
}


def region(key: str) -> Region:
    """Return the region with the given key.

    Raises:
        ConfigurationError: If no region has that key.
    """
    from arf.errors import ConfigurationError

    if key not in REGIONS:
        raise ConfigurationError(f"region {key!r} is not defined; defined regions: {sorted(REGIONS)}")
    return REGIONS[key]
