"""Orbit propagation with Orekit: the one module of the package that needs a Java machine.

Each satellite is propagated with SGP4 (Orekit's ``TLEPropagator``) from its own element set to the instants the
access computation asks for, in the ITRF (IERS 2010 conventions) over the WGS 84 ellipsoid. Reproduces the propagator
construction and the derived altitude and period of Cell 2, and the position and elevation calls of Cells 4 and 5, of
releases 1.5.3 to 1.5.6.

Needs the ``orekit-jpype`` and ``jdk4py`` packages and Orekit's data files (leap seconds, Earth orientation). Call
:func:`start_orekit` once before anything else here.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Mapping, Sequence

import numpy as np

from arf.fleet.elements import EARTH_RADIUS_KM, TleRecord, omm_tle_fields
from arf.regions import GroundStation

_started = False


def start_orekit(data_path: str | os.PathLike | None = None) -> None:
    """Start the Java machine and load Orekit's data. Later calls in the same process do nothing.

    Args:
        data_path: A folder or zip of Orekit data. By default the data of the ``orekitdata`` package is used, as in
            the registered releases.
    """
    global _started
    if _started:
        return
    import jdk4py
    import jpype
    import orekit_jpype

    if not jpype.isJVMStarted():
        library = os.path.join(jdk4py.JAVA_HOME, "lib", "server", "libjvm.so") if platform.system() == "Linux" else None
        if library and os.path.exists(library):
            orekit_jpype.initVM(jvmpath=library)
        else:
            orekit_jpype.initVM()
    if data_path is None:
        from orekit_jpype.pyhelpers import setup_orekit_curdir

        setup_orekit_curdir(from_pip_library=True)
    else:
        from java.io import File
        from org.orekit.data import DataContext, DirectoryCrawler, ZipJarCrawler

        path = File(str(data_path))
        crawler = DirectoryCrawler(path) if os.path.isdir(data_path) else ZipJarCrawler(path)
        DataContext.getDefault().getDataProvidersManager().addProvider(crawler)
    _started = True


def tle_from_omm(omm: Mapping):
    """Return an Orekit TLE built from the numbers of an OMM record (see :func:`arf.fleet.elements.omm_tle_fields`)."""
    from jpype import JChar
    from org.orekit.propagation.analytical.tle import TLE
    from org.orekit.time import AbsoluteDate, TimeScalesFactory

    utc = TimeScalesFactory.getUTC()
    fields = omm_tle_fields(omm)
    return TLE(
        fields["catalogue_number"],
        JChar(fields["classification"]),
        fields["launch_year"],
        fields["launch_number"],
        fields["launch_piece"],
        fields["ephemeris_type"],
        fields["element_number"],
        AbsoluteDate(fields["epoch_utc"], utc),
        fields["mean_motion_rad_s"],
        fields["mean_motion_first_derivative_rad_s2"],
        fields["mean_motion_second_derivative_rad_s3"],
        fields["eccentricity"],
        fields["inclination_rad"],
        fields["argument_of_perigee_rad"],
        fields["raan_rad"],
        fields["mean_anomaly_rad"],
        fields["revolution_number"],
        fields["bstar"],
        utc,
    )


def tle_from_lines(record: TleRecord):
    """Return an Orekit TLE from a ``(name, line1, line2)`` record. Orekit decodes Alpha-5 catalogue numbers itself."""
    from org.orekit.propagation.analytical.tle import TLE

    return TLE(record[1], record[2])


def check_omm_with_orekit(omm: Mapping) -> None:
    """Build the Orekit TLE of an OMM record and raise ``ValueError`` if Orekit refuses it.

    Pass it as ``check_elements`` to :func:`arf.fleet.select_fleet`: the registered releases build each TLE while
    selecting the fleet and count a record that Orekit refuses as malformed.
    """
    import jpype

    try:
        tle_from_omm(omm)
    except jpype.JException as error:
        raise ValueError(f"Orekit refused the element set: {error}") from error


def usable_tles(records: Sequence[TleRecord]) -> tuple[dict[str, object], list[str]]:
    """Build the Orekit TLE and propagator of each dense-constellation record, leaving out those Orekit refuses.

    Cell 2 of the registered releases skips such a record with a warning and goes on; a record passes the operational
    filter but is refused when, for instance, its two lines carry different catalogue numbers.

    Returns:
        The TLEs by satellite name, in record order, and the names of the records left out.

    Raises:
        ValueError: If two records share a name, since the satellites are told apart by name.
    """
    from org.orekit.propagation.analytical.tle import TLEPropagator

    tles: dict[str, object] = {}
    skipped: list[str] = []
    for record in records:
        try:
            tle = tle_from_lines(record)
            TLEPropagator.selectExtrapolator(tle)
        except Exception:  # the registered releases skip on any error here
            skipped.append(record[0])
            continue
        if record[0] in tles:
            raise ValueError(f"two satellites of the fleet are named {record[0]!r}")
        tles[record[0]] = tle
    return tles, skipped


class OrekitEphemeris:
    """The positions of a fleet, propagated with SGP4 from the start of a window.

    Implements :class:`arf.access.scan.Ephemeris`.
    """

    def __init__(
        self, tles: Mapping[str, object], start_utc_iso: str, stations: Mapping[str, GroundStation] | None = None
    ):
        """Build one propagator per satellite and one topocentric frame per ground station.

        Args:
            tles: Orekit TLEs by satellite identifier (:func:`tle_from_omm`, :func:`tle_from_lines`).
            start_utc_iso: The start of the window, UTC, ISO 8601.
            stations: The ground stations, by name.
        """
        from org.hipparchus.util import FastMath
        from org.orekit.bodies import GeodeticPoint, OneAxisEllipsoid
        from org.orekit.frames import FramesFactory, TopocentricFrame
        from org.orekit.propagation.analytical.tle import TLEPropagator
        from org.orekit.time import AbsoluteDate, TimeScalesFactory
        from org.orekit.utils import Constants, IERSConventions

        # Hipparchus' own conversions between degrees and radians are used, as in the registered releases: they are
        # not always equal, in the last bit, to numpy's.
        self._to_degrees, to_radians = FastMath.toDegrees, FastMath.toRadians
        self._utc = TimeScalesFactory.getUTC()
        self._itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
        self._earth = OneAxisEllipsoid(
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, self._itrf
        )
        self._start = AbsoluteDate(start_utc_iso, self._utc)
        self._propagators = {name: TLEPropagator.selectExtrapolator(tle) for name, tle in tles.items()}
        self._frames = {
            name: TopocentricFrame(
                self._earth,
                GeodeticPoint(
                    to_radians(float(station.latitude_deg)),
                    to_radians(float(station.longitude_deg)),
                    float(station.altitude_m),
                ),
                name,
            )
            for name, station in (stations or {}).items()
        }

    @property
    def start_datetime64(self) -> np.datetime64:
        """The start of the window as ``numpy.datetime64``, to the microsecond."""
        return np.datetime64(self._start.toString(self._utc).replace("Z", "")[:26])

    def subsatellite(self, satellite_id: str, t_min: float) -> tuple[float, float, float]:
        """Return geodetic latitude and longitude in degrees and geocentric radius in km, at minutes from the start."""
        date = self._start.shiftedBy(float(t_min * 60.0))
        position = self._propagators[satellite_id].getPVCoordinates(date, self._itrf).getPosition()
        point = self._earth.transform(position, self._itrf, date)
        return (
            float(self._to_degrees(point.getLatitude())),
            float(self._to_degrees(point.getLongitude())),
            position.getNorm() / 1000.0,
        )

    def elevations_deg(self, satellite_id: str, t_s: float, stations: Sequence[str]) -> list[float]:
        """Return the elevation of the satellite above each named ground station, at seconds from the start."""
        date = self._start.shiftedBy(float(t_s))
        position = self._propagators[satellite_id].getPVCoordinates(date, self._itrf).getPosition()
        return [
            float(self._to_degrees(self._frames[name].getElevation(position, self._itrf, date))) for name in stations
        ]

    def altitude_at_start_km(self, satellite_id: str) -> float:
        """Return the satellite's geocentric radius at the start of the window, less the equatorial radius, in km."""
        position = self._propagators[satellite_id].getPVCoordinates(self._start, self._itrf).getPosition()
        radius = np.sqrt(position.getX() ** 2 + position.getY() ** 2 + position.getZ() ** 2) / 1000.0
        return radius - EARTH_RADIUS_KM

    def altitudes_at_start_km(self) -> list[float]:
        """Return :meth:`altitude_at_start_km` for every satellite, in fleet order."""
        return [self.altitude_at_start_km(name) for name in self._propagators]
