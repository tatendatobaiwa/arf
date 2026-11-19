"""Reading orbital element sets: two-line elements (TLE) and OMM records.

Everything here is arithmetic on the text and numbers of an element set. No orbit is propagated, so no astrodynamics
library is needed. The functions reproduce the parsers of Cell 2 of release 1.5.6.

An element set is given either as a ``(name, line1, line2)`` tuple, the two-line format, or as an OMM record, the
mapping of the CCSDS Orbit Mean-Elements Message that CelesTrak and Space-Track deliver as JSON (``NORAD_CAT_ID``,
``EPOCH``, ``MEAN_MOTION`` in revolutions per day, ``ECCENTRICITY``, angles in degrees).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone

#: Earth's gravitational parameter, km3 s-2 (WGS 84), as used by the registered releases.
EARTH_MU_KM3_S2 = 398600.4418
#: Earth's equatorial radius, km (WGS 84).
EARTH_RADIUS_KM = 6378.137

#: The letters of an Alpha-5 catalogue number: A to Z without I and O, standing for 10 to 33.
ALPHA5_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"

TleRecord = tuple[str, str, str]


def alpha5_to_int(field: str) -> int:
    """Return the catalogue number in the five-character field of a TLE.

    Numbers up to 99999 are written as digits. Numbers from 100000 to 339999 are written in Alpha-5: a leading
    letter stands for the first two digits, A = 10 to Z = 33 with I and O omitted, so ``Z9999`` is 339999
    (Space-Track documentation, "Alpha-5").

    Raises:
        ValueError: If the field is neither digits nor Alpha-5.
    """
    text = field.strip()
    if len(text) == 5 and text[0] in ALPHA5_LETTERS and text[1:].isdigit():
        return (10 + ALPHA5_LETTERS.index(text[0])) * 10000 + int(text[1:])
    return int(text)


def catalogue_number(line1: str) -> int | None:
    """Return the catalogue number of a TLE from its first line (columns 3 to 7), or ``None`` if it cannot be read."""
    try:
        return alpha5_to_int(line1[2:7])
    except (ValueError, IndexError, TypeError):
        return None


def tle_epoch(line1: str) -> datetime:
    """Return the epoch of a TLE (first line, columns 19 to 32: two-digit year and day of year with fraction), in UTC.

    Two-digit years below 57 are in the 2000s, the others in the 1900s, the TLE convention.
    """
    two_digit_year = int(line1[18:20])
    year = 2000 + two_digit_year if two_digit_year < 57 else 1900 + two_digit_year
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=float(line1[20:32]) - 1.0)


def _perigee_apogee_km(mean_motion_rev_per_day: float, eccentricity: float) -> tuple[float, float]:
    """Return the perigee and apogee altitudes of an orbit from its mean motion and eccentricity.

    The semi-major axis is ``a = (mu / n^2)^(1/3)`` with ``n`` in radians per second. These are altitudes of the
    mean elements over a sphere of the equatorial radius; a propagated orbit dips a few kilometres lower.
    """
    mean_motion_rad_s = mean_motion_rev_per_day * 2.0 * math.pi / 86400.0
    semi_major_axis_km = (EARTH_MU_KM3_S2 / mean_motion_rad_s**2) ** (1.0 / 3.0)
    return (
        semi_major_axis_km * (1.0 - eccentricity) - EARTH_RADIUS_KM,
        semi_major_axis_km * (1.0 + eccentricity) - EARTH_RADIUS_KM,
    )


def tle_orbit_km(line2: str) -> tuple[float, float]:
    """Return the (perigee, apogee) altitudes in km from the second line of a TLE.

    Reads the eccentricity (columns 27 to 33, decimal point implied) and the mean motion (columns 53 to 63,
    revolutions per day).
    """
    eccentricity = float("0." + line2[26:33].strip())
    return _perigee_apogee_km(float(line2[52:63]), eccentricity)


def omm_orbit_km(omm: Mapping) -> tuple[float, float]:
    """Return the (perigee, apogee) altitudes in km from an OMM record."""
    return _perigee_apogee_km(float(omm["MEAN_MOTION"]), float(omm["ECCENTRICITY"]))


def omm_epoch(omm: Mapping) -> datetime:
    """Return the epoch of an OMM record, in UTC."""
    return datetime.fromisoformat(str(omm["EPOCH"]).replace("Z", "")).replace(tzinfo=timezone.utc)


def newest_tle_epoch_iso(records: Iterable[TleRecord]) -> str:
    """Return the newest epoch among TLE records, as ``YYYY-MM-DDTHH:MM:SS.ffffff`` in UTC.

    Raises:
        RuntimeError: If no record has a readable epoch.
    """
    epochs = []
    for _, line1, _ in records:
        try:
            epochs.append(tle_epoch(line1))
        except (ValueError, IndexError):
            continue
    if not epochs:
        raise RuntimeError("no element set with a readable epoch")
    return max(epochs).strftime("%Y-%m-%dT%H:%M:%S.%f")


def _history_epoch(text: str) -> datetime:
    """Return the epoch of a Space-Track history record, written with or without a fraction of a second."""
    try:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        return datetime.strptime(text, "%Y-%m-%dT%H:%M:%S")


def latest_per_satellite(history_records: Iterable[Mapping]) -> list[TleRecord]:
    """Reduce Space-Track ``gp_history`` records to the newest element set of each satellite.

    Returns ``(OBJECT_NAME, TLE_LINE1, TLE_LINE2)`` tuples in the order of ``NORAD_CAT_ID`` as the records hold it
    (Space-Track delivers it as text, so the order is that of the text).
    """
    newest_first = sorted(history_records, key=lambda record: _history_epoch(record["EPOCH"]), reverse=True)
    by_satellite = sorted(newest_first, key=lambda record: record["NORAD_CAT_ID"])  # stable: newest stays first
    seen, latest = set(), []
    for record in by_satellite:
        if record["NORAD_CAT_ID"] not in seen:
            seen.add(record["NORAD_CAT_ID"])
            latest.append((record["OBJECT_NAME"], record["TLE_LINE1"], record["TLE_LINE2"]))
    return latest


def omm_tle_fields(omm: Mapping) -> dict:
    """Return the values a TLE is built from, taken from the numbers of an OMM record.

    The registered releases build each propagator from these numbers and not from TLE text, because a catalogue
    number above 99999 cannot be written in a plain TLE. Mean motion is converted from revolutions per day to
    radians per second. The OMM fields ``MEAN_MOTION_DOT`` and ``MEAN_MOTION_DDOT`` hold, as in a TLE, the first
    derivative divided by two and the second divided by six; the derivatives themselves are returned, in radians
    per second squared and cubed. Angles are in radians.

    Raises:
        KeyError: If a required field is missing.
        ValueError: If a field is not a number.
        TypeError: If a field has the wrong type.
    """
    two_pi, day = 2.0 * math.pi, 86400.0
    object_id = str(omm.get("OBJECT_ID", "") or "")  # for example "1998-067A"
    try:
        launch_year, rest = int(object_id[:4]), object_id[5:]
        launch_number = int("".join(character for character in rest if character.isdigit()) or 0)
        launch_piece = "".join(character for character in rest if character.isalpha()) or "A"
    except ValueError:
        launch_year, launch_number, launch_piece = 2000, 0, "A"
    return {
        "catalogue_number": int(omm["NORAD_CAT_ID"]),
        "classification": str(omm.get("CLASSIFICATION_TYPE", "U"))[0],
        "launch_year": launch_year,
        "launch_number": launch_number,
        "launch_piece": launch_piece,
        "ephemeris_type": int(omm.get("EPHEMERIS_TYPE", 0)),
        "element_number": int(omm.get("ELEMENT_SET_NO", 999)),
        "epoch_utc": str(omm["EPOCH"]).replace("Z", ""),
        "mean_motion_rad_s": float(omm["MEAN_MOTION"]) * two_pi / day,
        "mean_motion_first_derivative_rad_s2": 2.0 * float(omm["MEAN_MOTION_DOT"]) * two_pi / day**2,
        "mean_motion_second_derivative_rad_s3": 6.0 * float(omm["MEAN_MOTION_DDOT"]) * two_pi / day**3,
        "eccentricity": float(omm["ECCENTRICITY"]),
        "inclination_rad": math.radians(float(omm["INCLINATION"])),
        "argument_of_perigee_rad": math.radians(float(omm["ARG_OF_PERICENTER"])),
        "raan_rad": math.radians(float(omm["RA_OF_ASC_NODE"])),
        "mean_anomaly_rad": math.radians(float(omm["MEAN_ANOMALY"])),
        "revolution_number": int(omm.get("REV_AT_EPOCH", 0)),
        "bstar": float(omm["BSTAR"]),
    }
