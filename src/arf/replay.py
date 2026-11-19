"""Rebuilding the inputs of a finished run of the registered releases from what the run recorded.

A run records the element-set files it read, with their SHA-256, in its manifest, and its settings under ``inputs``
→ ``config``. The functions here find those files again and rebuild the run's fleet, so that release 2 can compute
from the same inputs and be compared with the run's outputs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from arf.access.scan import Imager
from arf.fleet import (
    ACQUISITION_EXCLUSIONS,
    CELESTRAK_GROUPS,
    elements_as_of,
    merge_groups,
    mission_table,
    operational_sample,
    select_fleet,
    select_pinned,
)
from arf.fleet.elements import omm_tle_fields
from arf.runs import read_manifest, recorded_config


class RecordedInputs:
    """The input files a run recorded in its manifest, found again on disk.

    Attributes:
        changed: Names of the files whose SHA-256 is no longer the one the run recorded.
    """

    def __init__(self, manifest: dict, cache_dir: str | Path | None = None):
        """Keep the file records of the manifest and where else to look for the files."""
        self.files = (manifest.get("inputs") or {}).get("files") or {}
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.changed: list[str] = []

    def _locate(self, record: dict) -> Path | None:
        recorded = Path(str(record.get("path") or ""))
        if recorded.is_file():
            return recorded
        if self.cache_dir is not None and recorded.name:
            return next(iter(sorted(self.cache_dir.rglob(recorded.name))), None)
        return None

    def path(self, key: str | None = None, name_starts: str | None = None) -> Path:
        """Return where the input with a manifest key, or whose file name starts with some text, is found now.

        A file whose SHA-256 is no longer the one the run recorded is still returned, and its name is added to
        :attr:`changed`.

        Raises:
            FileNotFoundError: If the run did not record such a file, or it cannot be found.
        """
        for record_key, record in self.files.items():
            name = Path(str(record.get("path") or "")).name
            if (key is not None and record_key == key) or (name_starts is not None and name.startswith(name_starts)):
                path = self._locate(record)
                if path is None:
                    raise FileNotFoundError(f"{name or record_key} is not at its recorded path nor under the cache")
                if record.get("sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                    self.changed.append(name)
                return path
        raise FileNotFoundError(f"the run recorded no input file {key or name_starts!r}")

    def read(self, key: str | None = None, name_starts: str | None = None) -> dict:
        """Return the JSON content of an input; see :meth:`path`."""
        return json.loads(self.path(key, name_starts).read_bytes())


@dataclass
class RecordedFleet:
    """The fleet of a finished run, rebuilt.

    Attributes:
        kind: ``"dense"`` for the dense-constellation sample, ``"real"`` for a real Earth-observation fleet.
        imagers: The satellites, in fleet order.
        elements: Per satellite identifier, its element set: a ``(name, line1, line2)`` record for the dense
            constellation, an OMM record for a real fleet.
        start_utc_iso: The start of the window.
        exclusions: The acquisition exclusions that apply (none for the dense constellation).
        counts: The counts of the selection, as the run records them under ``satellite_selection``.
        changed_inputs: Names of input files that are no longer those the run read.
        notes: Anything about the rebuilding worth reporting, such as a pin file that differs from the seeded sample.
    """

    kind: str
    imagers: list[Imager]
    elements: dict
    start_utc_iso: str
    exclusions: list
    counts: dict
    changed_inputs: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def recorded_fleet(
    run_folder: str | Path,
    cache_dir: str | Path | None = None,
    check_elements: Callable[[Mapping], object] = omm_tle_fields,
) -> RecordedFleet:
    """Rebuild the fleet of a finished run from the files it recorded. Reads only.

    The dense-constellation fleet is that of the run's pin file, as Cell 2 reads it: the pinned catalogue numbers,
    in pin order, looked up among the satellites operational at the start of the window. If the run recorded no pin,
    it is the first N of the seeded shuffle, as Cell 2 would have pinned it. A real fleet is selected again from the
    Space-Track history file and CelesTrak's lists the run recorded.

    Args:
        run_folder: The run's output folder.
        cache_dir: A folder searched for a recorded file that is no longer at its recorded path.
        check_elements: Applied to each OMM record of a real fleet; a ``ValueError`` marks it malformed. Pass
            :func:`arf.access.orekit.check_omm_with_orekit` to count, as the registered releases do, a record that
            Orekit refuses.

    Raises:
        FileNotFoundError: If a recorded file cannot be found.
        ValueError: If two satellites of the fleet share an identifier.
    """
    run_folder = Path(run_folder)
    summary = json.loads((run_folder / "arf_cell2_summary.json").read_text())
    manifest = read_manifest(run_folder)
    config = recorded_config(manifest)
    inputs = RecordedInputs(manifest, cache_dir)
    start = summary["reference_epoch"]
    notes: list[str] = []
    if summary.get("fleet", "starlink") == "starlink":
        records = [tuple(record) for record in inputs.read(key="TLE_RAW_CACHE_PATH")["records"]]
        shuffled, counts = operational_sample(records, start, int(config.get("random_seed", 42)))
        n = int(manifest["inputs"]["active_n"])
        try:
            pinned = inputs.read(key="SELECTION_CACHE_PATH")["norad_ids"]
        except FileNotFoundError:
            sample = shuffled[:n]
            notes.append("no pin file recorded: the first N of the seeded sample are used")
        else:
            sample = select_pinned(shuffled, pinned)
            if len(sample) < len(pinned):
                notes.append(f"{len(pinned) - len(sample)} pinned satellites are not among the operational ones")
            if sample != shuffled[:n]:
                notes.append("the pin file differs from the first N of the seeded sample")
        swath, sun = config.get("proxy_swath_km"), config.get("proxy_min_sun_elev_deg")
        imagers = [
            Imager(record[0], None if swath is None else float(swath), None if sun is None else float(sun))
            for record in sample
        ]
        fleet = RecordedFleet("dense", imagers, {record[0]: record for record in sample}, start, [], counts)
    else:
        groups = CELESTRAK_GROUPS[summary["fleet"]]
        live = merge_groups({group: inputs.read(name_starts=f"arf_gp_{group}.json")["records"] for group in groups})
        min_age_days = 0.0
        if summary.get("analysis_epoch"):
            history = inputs.read(name_starts="arf_gp_history_")["records"]
            records = elements_as_of(history, live, summary["analysis_epoch"])
        else:
            records = live
            min_age_days = -2.0 if config.get("reference_epoch_utc") else 0.0
        missions = mission_table(
            config.get("planet_swath_case", "conservative"), float(config.get("assumed_min_sun_elev_deg", 5.0))
        )
        satellites, counts = select_fleet(
            records,
            start,
            missions,
            min_age_days=min_age_days,
            exclude_names=config.get("exclude_satellites") or (),
            check_elements=check_elements,
        )
        imagers = [Imager(s.id, s.swath_km, s.min_sun_elev_deg) for s in satellites]
        elements = {s.id: s.elements for s in satellites}
        fleet = RecordedFleet("real", imagers, elements, start, list(ACQUISITION_EXCLUSIONS), counts)
    if len(fleet.elements) != len(fleet.imagers):
        raise ValueError("two satellites of the fleet share an identifier")
    fleet.changed_inputs = sorted(set(inputs.changed))
    fleet.notes = notes
    return fleet
