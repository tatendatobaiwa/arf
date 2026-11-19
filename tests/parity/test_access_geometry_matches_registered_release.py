"""arf.access.geometry against the geometry helpers of Cells 4 and 5 of the registered releases.

These helpers need no orbit propagation, so this file runs without Orekit. Results must be equal to the last bit.
"""

import numpy as np
import pytest
from registered import run_cells_4_5_geometry

from arf.access import downlink_reward
from arf.access import geometry as geo
from arf.access.scan import RegionIndex, ScanSettings

RELEASE_NAMES = ("1.5.3", "1.5.6")


@pytest.fixture(scope="module", params=RELEASE_NAMES)
def registered(request):
    return run_cells_4_5_geometry(request.param)


def points(seed, n, polar=False):
    rng = np.random.default_rng(seed)
    latitude = rng.uniform(-89.9, 89.9, n) if polar else rng.uniform(-60, 60, n)
    return latitude, rng.uniform(-180, 180, n)


def test_distances_and_radii(registered):
    lat1, lon1 = points(1, 500, polar=True)
    lat2, lon2 = points(2, 500, polar=True)
    assert np.array_equal(geo.haversine_km(lat1, lon1, lat2, lon2), registered["_hav_km"](lat1, lon1, lat2, lon2))
    assert np.array_equal(geo.wrap_longitude_difference(lon1, 170.0), registered["_wrap_dlon"](lon1, 170.0))
    assert registered["_KM_PER_DEG_LAT_MIN"] == geo.KM_PER_DEGREE_LATITUDE_MIN
    assert registered["_KM_PER_DEG_LON_EQ"] == geo.KM_PER_DEGREE_LONGITUDE_EQUATOR
    for altitude in (150.0, 475.0, 550.0, 786.0, 2000.0):
        for angle in (0.5, 1.5, 9.22, 45.0, 80.0):
            ours = geo.footprint_ground_radius_km(altitude, angle)
            assert ours == registered["_footprint_ground_radius_km"](altitude, angle)
        for elevation in (0.0, 5.0, 15.0, 60.0):
            assert geo.visibility_radius_km(altitude, elevation) == registered["_visibility_radius_km"](
                altitude, elevation
            )
    assert 2500 < geo.visibility_radius_km(550.0, 0.0) < 2600  # the horizon at 550 km
    for wait in (None, 0, 30, 95, 200):
        assert downlink_reward(wait, 95.0) == registered["downlink_reward"](wait)


def test_off_nadir_angle(registered):
    distance = np.r_[0.0, np.geomspace(0.1, 9000.0, 400)]
    for radius in (6378.137 + 475.0, 6378.137 + 786.0, np.linspace(6850.0, 6860.0, 401)):
        ours = geo.off_nadir_angle_deg(distance, 6378.137, radius)
        assert np.array_equal(ours, registered["_incidence_angle_deg_vec"](distance, 6378.137, radius))
    assert np.isinf(ours[-1]) and ours[0] == 0.0  # beyond the horizon; at nadir


def test_solar_elevation(registered):
    latitude, longitude = points(3, 300, polar=True)
    for time in ("2026-08-03T08:15:00", "2024-02-29T23:59:59", "2026-12-21T12:00:00", "2100-06-21T00:00:00"):
        t = np.datetime64(time)
        assert np.array_equal(
            geo.solar_elevation_deg(t, latitude, longitude), registered["solar_elevation_deg"](t, latitude, longitude)
        )
    times = np.datetime64("2026-08-03T00:00:00") + np.arange(300).astype("timedelta64[m]") * 7
    assert np.array_equal(
        geo.solar_elevation_deg(times, latitude, longitude),
        registered["solar_elevation_deg"](times, latitude, longitude),
    )
    noon = geo.solar_elevation_deg(np.datetime64("2026-12-21T10:25:00"), -23.44, 23.5)  # Tropic of Capricorn, solstice
    assert 89.0 < float(noon[0]) <= 90.0


def test_boxes(registered):
    cases = [
        (*points(4, 200), 50.0),
        (np.array([-22.0, -19.0]), np.array([22.0, 25.0]), 1100.0),
        (np.array([10.0, 12.0]), np.array([179.0, -179.5]), 300.0),  # across the antimeridian
        (np.array([84.0, 86.0]), np.array([10.0, 20.0]), 500.0),  # reaches the pole: every longitude
        (np.array([60.0, 61.0]), np.array([0.0, 1.0]), 12000.0),  # margin wider than half the globe
    ]
    test_lat, test_lon = points(5, 2000, polar=True)
    for latitude, longitude, margin in cases:
        ours, theirs = geo.GeoBox(latitude, longitude, margin), registered["_GeoBox"](latitude, longitude, margin)
        assert (ours.lat_lo, ours.lat_hi, ours.lon_centre, ours.lon_half, bool(ours.all_longitudes)) == (
            theirs.lat_lo,
            theirs.lat_hi,
            theirs.lon_c,
            theirs.lon_half,
            bool(theirs.all_lon),
        )
        assert np.array_equal(ours.contains(test_lat, test_lon), theirs.contains(test_lat, test_lon))


@pytest.mark.parametrize("exclude_end_caps", [True, False])
def test_distance_to_a_ground_track(registered, exclude_end_caps):
    rng = np.random.default_rng(6)
    for trial in range(30):
        # A short track: 31 samples about 14 km apart, heading south-south-west, as a strip of one minute.
        start_lat, start_lon = rng.uniform(-40, 40), rng.uniform(-180, 180)
        steps = np.arange(31)
        track_lat = start_lat - 0.125 * steps
        track_lon = start_lon - 0.03 * steps
        radius = 6378.137 + 500.0 + rng.uniform(-5, 5, 31)
        if trial % 5 == 0:  # a strip clipped at the edge of the window repeats its first sample
            track_lat[:6], track_lon[:6], radius[:6] = track_lat[5], track_lon[5], radius[5]
        if trial % 7 == 0:
            track_lat, track_lon, radius = track_lat[:1], track_lon[:1], radius[:1]  # a single sample
        point_lat = start_lat - 2.0 + rng.uniform(-3, 3, 400)
        point_lon = start_lon - 0.5 + rng.uniform(-3, 3, 400)
        ours = geo.min_track_distance_km(track_lat, track_lon, radius, point_lat, point_lon, 6378.137, exclude_end_caps)
        theirs = registered["_min_track_distance_km"](
            track_lat, track_lon, radius, point_lat, point_lon, 6378.137, exclude_caps=exclude_end_caps
        )
        assert np.array_equal(ours[0], theirs[0]) and np.array_equal(ours[1], theirs[1])
        if exclude_end_caps and len(track_lat) > 1:
            assert np.isinf(ours[0]).any() and np.isfinite(ours[0]).any()  # some points belong to a neighbouring strip


def test_search_radii_and_cells_near_a_track():
    wide = ScanSettings(swath_half_angle_deg=9.22, max_altitude_km=790.0, min_altitude_km=470.0, max_swath_km=290.0)
    legacy = ScanSettings(swath_half_angle_deg=9.22, max_altitude_km=560.0, min_altitude_km=540.0)
    assert 152 < wide.footprint_radius_km < 165 and 85 < legacy.footprint_radius_km < 100
    assert wide.coarse_margin_km > wide.strip_margin_km > wide.footprint_radius_km
    assert 430 < wide.max_ground_speed_km_min < 470  # about 7.5 km/s over the ground
    latitude, longitude = np.meshgrid(np.linspace(-22.5, -19.0, 40), np.linspace(22.0, 25.0, 40))
    cells = [f"cell{index:04d}" for index in range(latitude.size)]
    region = RegionIndex(cells[::-1], latitude.ravel()[::-1], longitude.ravel()[::-1])
    assert np.all(np.diff(region.latitude) >= 0)
    near = region.cells_near(np.array([-20.0, -20.1]), np.array([23.0, 23.0]), 30.0)
    distance = geo.haversine_km(-20.05, 23.0, region.latitude, region.longitude)
    assert near.size > 0 and set(np.flatnonzero(distance <= 24.0)) <= set(near) <= set(np.flatnonzero(distance <= 36.0))
    assert region.cells_near(np.array([40.0]), np.array([23.0]), 30.0).size == 0
