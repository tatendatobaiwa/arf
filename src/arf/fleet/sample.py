"""The dense-constellation fleet: satellites operational at the start of the window, and a seeded sample of N.

Space-Track's history returns every object with an element set in the days before the window, including satellites
that are deorbiting or no longer tracked. Taking the first N by catalogue number would pick the oldest satellites,
many of them decaying. Instead, the satellites whose newest element set describes an operational orbit are kept, put
in catalogue-number order, and shuffled with a seeded generator; the first N of the shuffle are the sample.

Reproduces ``select_operational_sample`` and ``select_pinned_satellites`` of Cell 2 of release 1.5.6. Releases before
1.5.6 could not read Alpha-5 catalogue numbers and counted those element sets as malformed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from arf.fleet.elements import TleRecord, alpha5_to_int, catalogue_number, tle_epoch, tle_orbit_km

logger = logging.getLogger(__name__)

#: What "operational" means. A perigee of at least 300 km leaves out deorbiting objects; an apogee of at most 700 km
#: leaves out anomalous orbits outside the constellation's shells; an element set at most 7 days older than the start
#: of the window leaves out stale and untracked objects. The altitudes are those of the mean elements; the propagated
#: altitude dips about 7 km below the mean-element perigee, so the bound admits true minima down to about 293 km.
OPERATIONAL_CRITERIA = {"min_perigee_km": 300.0, "max_apogee_km": 700.0, "max_elset_age_days": 7.0}


def operational_sample(
    records: Sequence[TleRecord],
    onset_iso: str,
    seed: int,
    criteria: Mapping[str, float] = OPERATIONAL_CRITERIA,
) -> tuple[list[TleRecord], dict]:
    """Keep the satellites operational at a time and return them in a seeded random order.

    Args:
        records: Element sets as ``(name, line1, line2)``, one per satellite.
        onset_iso: The start of the window, UTC, ISO 8601.
        seed: The seed of the shuffle (``numpy.random.default_rng(seed).permutation``).
        criteria: The limits, as :data:`OPERATIONAL_CRITERIA`.

    Returns:
        The operational records in shuffled order, of which the first N are the sample of size N, and the counts:
        ``fetched``, ``malformed``, ``stale_or_future``, ``low_perigee``, ``high_apogee``, ``operational``, with the
        ``criteria`` and the ``seed``. A record is counted under the first limit it fails, in that order.
    """
    onset = datetime.fromisoformat(onset_iso.replace("Z", "")).replace(tzinfo=timezone.utc)
    counts = {
        "fetched": len(records),
        "malformed": 0,
        "stale_or_future": 0,
        "low_perigee": 0,
        "high_apogee": 0,
        "operational": 0,
        "criteria": dict(criteria),
        "seed": seed,
    }
    kept = []
    for record in records:
        _, line1, line2 = record
        try:
            age_days = (onset - tle_epoch(line1)).total_seconds() / 86400.0
            perigee_km, apogee_km = tle_orbit_km(line2)
            number = alpha5_to_int(line1[2:7])
        except (ValueError, IndexError, ZeroDivisionError):
            counts["malformed"] += 1
            continue
        if not (0.0 <= age_days <= criteria["max_elset_age_days"]):
            counts["stale_or_future"] += 1
        elif perigee_km < criteria["min_perigee_km"]:
            counts["low_perigee"] += 1
        elif apogee_km > criteria["max_apogee_km"]:
            counts["high_apogee"] += 1
        else:
            kept.append((number, record))
    kept.sort(key=lambda pair: pair[0])  # stable: equal catalogue numbers keep the order of the records
    counts["operational"] = len(kept)
    order = np.random.default_rng(seed).permutation(len(kept))
    return [kept[position][1] for position in order], counts


def select_pinned(records: Sequence[TleRecord], pinned_numbers: Sequence[int]) -> list[TleRecord]:
    """Return the records of the pinned catalogue numbers, in the order of the pin.

    A pinned number with no record is left out. When two records share a catalogue number, the later one is used.
    """
    by_number = {}
    for record in records:
        number = catalogue_number(record[1])
        if number is not None:
            by_number[number] = tuple(record)
    return [by_number[number] for number in pinned_numbers if number in by_number]


def pinned_selection(records: Sequence[TleRecord], n: int, pin_path: str | Path) -> list[TleRecord]:
    """Return the sample of N, writing its catalogue numbers to a pin file or reading them from it.

    If the pin file exists, the satellites it lists are returned in its order, and those no longer among the records
    are left out with a warning. Otherwise the first ``n`` records are the sample, and their catalogue numbers are
    written to the pin file so that a later run uses the same satellites.
    """
    pin_path = Path(pin_path)
    if pin_path.exists():
        pinned = json.loads(pin_path.read_text())["norad_ids"]
        selected = select_pinned(records, pinned)
        if len(selected) < len(pinned):
            logger.warning("Only %d of %d pinned satellites are among the element sets.", len(selected), len(pinned))
        return selected
    chosen = list(records[:n])
    numbers = [number for record in chosen if (number := catalogue_number(record[1])) is not None]
    try:
        pin_path.write_text(json.dumps({"active_n": n, "norad_ids": numbers}, indent=2))
    except OSError as error:
        logger.warning("Could not write the pin file %s: %s", pin_path, error)
    return chosen
