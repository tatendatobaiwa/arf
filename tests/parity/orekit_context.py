"""The Orekit context of the tests: a Java machine and Orekit's data.

Orekit needs data files to convert between time scales and to orient the Earth. If the environment variable
``ARF_OREKIT_DATA`` names a folder or zip of Orekit data (the leap-second table, the Earth orientation parameters and
``itrf-versions.conf``, as in https://gitlab.orekit.org/orekit/orekit-data), the tests use it, as the registered
releases do; continuous integration fetches those files. Otherwise the tests give Orekit only the leap-second table
below, written in the format of the USNO file ``tai-utc.dat``, and Orekit takes the Earth orientation parameters as
zero, which moves a ground track by metres.

Either way the tests do not reproduce the numbers of a real run: they run the registered release and release 2 in the
same context and require the same result from both. The comparison with real runs is
``tools/check_access_on_runs.py``.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# TAI - UTC in whole seconds from each date (IERS Bulletin C; unchanged since 1 January 2017).
LEAP_SECONDS = (
    (1972, "JAN", 2441317.5, 10),
    (1972, "JUL", 2441499.5, 11),
    (1973, "JAN", 2441683.5, 12),
    (1974, "JAN", 2442048.5, 13),
    (1975, "JAN", 2442413.5, 14),
    (1976, "JAN", 2442778.5, 15),
    (1977, "JAN", 2443144.5, 16),
    (1978, "JAN", 2443509.5, 17),
    (1979, "JAN", 2443874.5, 18),
    (1980, "JAN", 2444239.5, 19),
    (1981, "JUL", 2444786.5, 20),
    (1982, "JUL", 2445151.5, 21),
    (1983, "JUL", 2445516.5, 22),
    (1985, "JUL", 2446247.5, 23),
    (1988, "JAN", 2447161.5, 24),
    (1990, "JAN", 2447892.5, 25),
    (1991, "JAN", 2448257.5, 26),
    (1992, "JUL", 2448804.5, 27),
    (1993, "JUL", 2449169.5, 28),
    (1994, "JUL", 2449534.5, 29),
    (1996, "JAN", 2450083.5, 30),
    (1997, "JUL", 2450630.5, 31),
    (1999, "JAN", 2451179.5, 32),
    (2006, "JAN", 2453736.5, 33),
    (2009, "JAN", 2454832.5, 34),
    (2012, "JUL", 2456109.5, 35),
    (2015, "JUL", 2457204.5, 36),
    (2017, "JAN", 2457754.5, 37),
)

_started = False


def orekit_data_in_use() -> str:
    """Return which Orekit data the tests use: the folder named by ``ARF_OREKIT_DATA``, or the leap seconds only."""
    named = os.environ.get("ARF_OREKIT_DATA")
    return named if named else "leap-second table only"


def start_minimal_orekit() -> None:
    """Start Orekit once per process; skip the calling test if Orekit is not installed.

    Uses the data named by ``ARF_OREKIT_DATA`` when it is set, and the leap-second table otherwise.

    Raises:
        FileNotFoundError: If ``ARF_OREKIT_DATA`` names something that does not exist.
    """
    global _started
    pytest.importorskip("orekit_jpype")
    pytest.importorskip("jdk4py")
    if _started:
        return
    from arf.access.orekit import start_orekit

    named = os.environ.get("ARF_OREKIT_DATA")
    if named:
        if not Path(named).exists():
            raise FileNotFoundError(f"ARF_OREKIT_DATA names {named}, which does not exist")
        start_orekit(named)
        _started = True
        return
    folder = Path(tempfile.mkdtemp(prefix="orekit_minimal_"))
    lines = [
        f" {year} {month}  1 =JD {julian_day:.1f}  TAI-UTC=  {seconds:.1f}       S + (MJD - 41317.) X 0.0      S\n"
        for year, month, julian_day, seconds in LEAP_SECONDS
    ]
    (folder / "tai-utc.dat").write_text("".join(lines))
    start_orekit(folder)
    _started = True
