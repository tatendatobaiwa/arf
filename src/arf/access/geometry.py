"""Geometry on a spherical Earth: distances, boxes, footprints, off-nadir angles, the Sun's elevation.

All of it is arithmetic on latitudes, longitudes and radii; no orbit is propagated here. The sphere has the WGS 84
equatorial radius, the same sphere for the search radii and for the exact filters, so they agree on where a swath edge
is. Reproduces the helpers of Cells 4 and 5 of releases 1.5.3 to 1.5.6, operation for operation.
"""

from __future__ import annotations

import numpy as np

from arf.fleet.elements import EARTH_MU_KM3_S2, EARTH_RADIUS_KM

#: Length of a degree of latitude at the equator, km (WGS 84 meridian). The shortest, so margins in degrees are
#: never too small.
KM_PER_DEGREE_LATITUDE_MIN = 110.574
#: Length of a degree of longitude at the equator, km (WGS 84 parallel).
KM_PER_DEGREE_LONGITUDE_EQUATOR = 111.320


def wrap_longitude_difference(longitude, reference):
    """Return the longitude difference to a reference, wrapped into [-180, 180) degrees."""
    return (np.asarray(longitude, dtype=float) - reference + 180.0) % 360.0 - 180.0


def haversine_km(latitude_1, longitude_1, latitude_2, longitude_2):
    """Return the great-circle distance in km between points given in degrees (haversine formula)."""
    phi_1, phi_2 = np.radians(latitude_1), np.radians(latitude_2)
    a = (
        np.sin((phi_2 - phi_1) / 2) ** 2
        + np.cos(phi_1)
        * np.cos(phi_2)
        * np.sin(np.radians(wrap_longitude_difference(longitude_2, 0) - wrap_longitude_difference(longitude_1, 0)) / 2)
        ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class GeoBox:
    """A latitude and longitude box around a set of points, inflated by a margin in km.

    The longitude margin is scaled by the cosine of the latitude of the box's edge farthest from the equator, where
    a degree of longitude is shortest. Longitudes are handled relative to a centre, so a box may straddle the
    antimeridian, and a box reaching within one degree of a pole accepts every longitude.
    """

    def __init__(self, latitudes, longitudes, margin_km: float):
        """Build the box around the points."""
        latitudes = np.asarray(latitudes, dtype=float)
        latitude_margin = margin_km / KM_PER_DEGREE_LATITUDE_MIN
        self.lat_lo, self.lat_hi = latitudes.min() - latitude_margin, latitudes.max() + latitude_margin
        reference_latitude = max(abs(self.lat_lo), abs(self.lat_hi))
        reference = float(np.asarray(longitudes, dtype=float).flat[0])
        difference = wrap_longitude_difference(longitudes, reference)
        self.lon_centre = reference + (difference.min() + difference.max()) / 2.0
        self.all_longitudes = reference_latitude >= 89.0
        self.lon_half = (difference.max() - difference.min()) / 2.0
        if not self.all_longitudes:
            self.lon_half += margin_km / (KM_PER_DEGREE_LONGITUDE_EQUATOR * np.cos(np.radians(reference_latitude)))
            self.all_longitudes = self.lon_half >= 180.0

    def contains(self, latitude, longitude):
        """Return whether each point lies in the box."""
        latitude = np.asarray(latitude, dtype=float)
        inside = (latitude >= self.lat_lo) & (latitude <= self.lat_hi)
        if self.all_longitudes:
            return inside
        return inside & (np.abs(wrap_longitude_difference(longitude, self.lon_centre)) <= self.lon_half)


def footprint_ground_radius_km(altitude_km: float, half_angle_deg: float) -> float:
    """Return the ground radius of the footprint of a nadir-pointing sensor with a given half-angle.

    ``R * (arcsin((R + h) / R * sin(eta)) - eta)``; the sine is capped at one, the horizon.
    """
    eta = np.radians(half_angle_deg)
    sine = min(1.0, (EARTH_RADIUS_KM + altitude_km) / EARTH_RADIUS_KM * np.sin(eta))
    return EARTH_RADIUS_KM * (np.arcsin(sine) - eta)


def visibility_radius_km(altitude_km: float, min_elevation_deg: float) -> float:
    """Return the ground distance from a station within which a satellite is at or above an elevation.

    ``R * (arccos(R * cos(E) / (R + h)) - E)``; at zero elevation this is the distance to the horizon, about
    2,559 km at 550 km altitude.
    """
    elevation = np.radians(min_elevation_deg)
    return float(
        EARTH_RADIUS_KM * (np.arccos(EARTH_RADIUS_KM * np.cos(elevation) / (EARTH_RADIUS_KM + altitude_km)) - elevation)
    )


def off_nadir_angle_deg(ground_distance_km, earth_radius_km: float, satellite_radius_km) -> np.ndarray:
    """Return the off-nadir angle at which a sensor sees a ground point at a ground distance from its nadir.

    Args:
        ground_distance_km: Great-circle distance from the sub-satellite point.
        earth_radius_km: Radius of the sphere.
        satellite_radius_km: Geocentric radius of the satellite.

    Returns:
        The angle in degrees; ``inf`` for a point beyond the horizon. Without that guard the arcsine turns small
        again on the far side of the Earth and would report a near-nadir angle for a point that cannot be seen.
    """
    gamma = np.asarray(ground_distance_km, dtype=float) / earth_radius_km
    slant_range = np.sqrt(
        satellite_radius_km**2 + earth_radius_km**2 - 2.0 * satellite_radius_km * earth_radius_km * np.cos(gamma)
    )
    sine = np.clip(earth_radius_km * np.sin(gamma) / slant_range, -1.0, 1.0)
    eta = np.degrees(np.arcsin(sine))
    return np.where(gamma < np.arccos(earth_radius_km / satellite_radius_km), eta, np.inf)


def solar_elevation_deg(time_utc, latitude_deg, longitude_deg):
    """Return the Sun's elevation above the horizon, in degrees, geometric and without refraction.

    NOAA Global Monitoring Laboratory, "General Solar Position Calculations"
    (https://gml.noaa.gov/grad/solcalc/solareqns.PDF): fractional year, equation of time, declination, true solar
    time, hour angle, zenith.

    Args:
        time_utc: A UTC time or an array of them (``numpy.datetime64``).
        latitude_deg: Latitudes of the ground points.
        longitude_deg: Longitudes of the ground points, east positive.
    """
    t = np.atleast_1d(np.asarray(time_utc, dtype="datetime64[s]"))
    years = t.astype("datetime64[Y]")
    day_of_year = (t.astype("datetime64[D]") - years).astype(int) + 1
    year = years.astype(int) + 1970
    days_in_year = np.where((year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0)), 366, 365)
    seconds = (t - t.astype("datetime64[D]")).astype(int)
    hour = seconds / 3600.0
    g = 2.0 * np.pi / days_in_year * (day_of_year - 1 + (hour - 12.0) / 24.0)
    equation_of_time = 229.18 * (
        0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g)
    )
    declination = (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g)
        + 0.00148 * np.sin(3 * g)
    )
    true_solar_time = hour * 60.0 + equation_of_time + 4.0 * np.asarray(longitude_deg)  # time zone 0 (UTC)
    hour_angle = np.radians(true_solar_time / 4.0 - 180.0)
    latitude = np.radians(latitude_deg)
    cos_zenith = np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
    return 90.0 - np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))


def _unit_vectors(latitude, longitude):
    """Return the unit vectors of points given in degrees."""
    phi, lam = np.radians(latitude), np.radians(longitude)
    return np.stack([np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)], axis=-1)


_ARC_END_TOLERANCE_RAD = 1e-12  # rounding at the ends of an arc


def _outside_end(start, end, points, before: bool):
    """Return whether each point's closest approach to a great-circle arc lies before its start or after its end."""
    normal = np.cross(start, end)
    length = np.linalg.norm(normal)
    if length < 1e-15:
        return np.zeros(len(points), dtype=bool)
    normal = normal / length
    foot = points - np.outer(points @ normal, normal)
    if before:
        return (np.cross(start, foot) @ normal) < -_ARC_END_TOLERANCE_RAD
    return (np.cross(foot, end) @ normal) < -_ARC_END_TOLERANCE_RAD


def min_track_distance_km(
    track_latitude,
    track_longitude,
    track_radius_km,
    point_latitude,
    point_longitude,
    earth_radius_km: float,
    exclude_end_caps: bool = True,
):
    """Return the least great-circle distance from ground points to a ground track given by samples.

    The distance is to the arcs between consecutive samples, not only to the samples: samples two seconds apart are
    about 15 km apart, which loses whole cells of a narrow swath.

    A push-broom or frame imager sees a ground point when the track passes abeam of it, at its closest point of
    approach. With ``exclude_end_caps``, a point whose closest approach lies before the first sample or after the last
    belongs to the neighbouring strip, and its distance here is ``inf``. Without it (the rule of release 1.5.0 and
    earlier) such a point takes its distance to the end sample, so every strip also claims a half-disc beyond each
    end and most observations appear in two consecutive strips. Repeated samples, from strips clipped at the edges of
    the window, are merged first.

    Returns:
        The distance in km per point, and the geocentric radius of the satellite at the nearest sample or arc.
    """
    samples = _unit_vectors(np.asarray(track_latitude, float), np.asarray(track_longitude, float))
    radius = np.asarray(track_radius_km, float)
    if exclude_end_caps and len(samples) > 1:
        keep = np.r_[True, np.linalg.norm(np.diff(samples, axis=0), axis=1) > 1e-15]
        samples, radius = samples[keep], radius[keep]
    points = _unit_vectors(np.asarray(point_latitude, float), np.asarray(point_longitude, float))
    to_samples = np.arccos(np.clip(points @ samples.T, -1.0, 1.0))
    best_angle, best_index = to_samples.min(axis=1), to_samples.argmin(axis=1)
    best_radius = radius[best_index]
    arc_tolerance = _ARC_END_TOLERANCE_RAD if exclude_end_caps else 0.0
    for index in range(len(samples) - 1):
        start, end = samples[index], samples[index + 1]
        normal = np.cross(start, end)
        length = np.linalg.norm(normal)
        if length < 1e-15:
            continue
        normal = normal / length
        sin_cross_track = points @ normal
        foot = points - np.outer(sin_cross_track, normal)
        on_arc = ((np.cross(start, foot) @ normal) >= -arc_tolerance) & (
            (np.cross(foot, end) @ normal) >= -arc_tolerance
        )
        angle = np.abs(np.arcsin(np.clip(sin_cross_track, -1.0, 1.0)))
        better = on_arc & (angle < best_angle)
        best_angle = np.where(better, angle, best_angle)
        best_radius = np.where(better, 0.5 * (radius[index] + radius[index + 1]), best_radius)
    if exclude_end_caps and len(samples) > 1:
        outside = _outside_end(samples[0], samples[1], points, True) | _outside_end(
            samples[-2], samples[-1], points, False
        )
        best_angle = np.where(outside, np.inf, best_angle)
    return best_angle * earth_radius_km, best_radius


def circular_orbit_period_min(altitude_km: float) -> float:
    """Return the period of a circular orbit at an altitude, in minutes: ``2 pi sqrt((R + h)^3 / mu)``.

    The releases use the period at the fleet's median altitude as the longest useful wait for a downlink.
    """
    return 2 * np.pi * np.sqrt((EARTH_RADIUS_KM + altitude_km) ** 3 / EARTH_MU_KM3_S2) / 60.0
