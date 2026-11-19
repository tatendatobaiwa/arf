"""Check the fleet of release 2 against the finished runs of the registered releases.

A run records which element-set files it read (``inputs`` → ``files`` in ``arf_reproducibility_manifest.json``, each
with its SHA-256), and what it selected from them: the counts under ``satellite_selection`` in
``arf_cell2_summary.json``, and the satellites in its pin file (dense constellation) or fleet file (real fleets).
For every run folder given, release 2 reads the same element-set files and selects again:

* dense constellation: the operational filter and the seeded shuffle; the counts must equal the run's, and the first
  N catalogue numbers must equal the pin file;
* real fleets: the element sets as of the start of the window, the mission table, the selection; the counts, the
  mission table and every satellite (name, catalogue number, mission, swath, field of view, daylight limit) must
  equal the run's.

Read-only, and it prints counts only: element sets are not redistributed (Space-Track user agreement). The files are
looked for at the path the run recorded and, failing that, by name under ``--cache``. A file whose SHA-256 is no
longer the one the run recorded is still used, and the line says so: CelesTrak's current list is cached for six hours
and then replaced, so for an old run it may name the satellites differently than it did.

Usage:
    python tools/check_fleet_on_runs.py RUN_FOLDER [RUN_FOLDER ...]
    python tools/check_fleet_on_runs.py --index ARF_batch/paper2_v156_a18_core_index.csv --cache ARF_batch/shared_cache

Exit status 0 if every run agrees, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from arf.fleet import (
    CELESTRAK_GROUPS,
    elements_as_of,
    fleet_file_entries,
    merge_groups,
    mission_table,
    operational_sample,
    select_fleet,
)
from arf.fleet.elements import catalogue_number
from arf.runs import find_run_folder, read_manifest, recorded_config


class _Inputs:
    """The element-set files a run recorded, found again on disk."""

    def __init__(self, manifest: dict, cache_dir: Path | None):
        """Keep the file records of the manifest and where else to look for the files."""
        self.files = (manifest.get("inputs") or {}).get("files") or {}
        self.cache_dir = cache_dir
        self.changed: list[str] = []

    def _locate(self, record: dict) -> Path | None:
        recorded = Path(str(record.get("path") or ""))
        if recorded.is_file():
            return recorded
        if self.cache_dir is not None and recorded.name:
            return next(iter(sorted(self.cache_dir.rglob(recorded.name))), None)
        return None

    def read(self, key: str | None = None, name_starts: str | None = None) -> dict:
        """Return the JSON content of the input with a manifest key, or whose file name starts with some text.

        Raises:
            FileNotFoundError: If the run did not record such a file or it cannot be found.
        """
        for record_key, record in self.files.items():
            name = Path(str(record.get("path") or "")).name
            if (key is not None and record_key == key) or (name_starts is not None and name.startswith(name_starts)):
                path = self._locate(record)
                if path is None:
                    raise FileNotFoundError(f"{name or record_key} is not at its recorded path nor under --cache")
                content = path.read_bytes()
                if record.get("sha256") and hashlib.sha256(content).hexdigest() != record["sha256"]:
                    self.changed.append(name)
                return json.loads(content)
        raise FileNotFoundError(f"the run recorded no input file {key or name_starts!r}")


def _count_differences(ours: dict, recorded: dict) -> list[str]:
    """Return the keys on which the selection counts of release 2 and of a run differ."""
    recorded = {key: value for key, value in recorded.items() if key != "method"}
    ours = json.loads(json.dumps(ours))  # as the run's counts were read: tuples as lists
    return [
        f"{key} (release 2 {str(ours.get(key))[:80]}, run {str(recorded.get(key))[:80]})"
        for key in sorted(set(ours) | set(recorded))
        if ours.get(key) != recorded.get(key)
    ]


def _check_dense(summary: dict, manifest: dict, inputs: _Inputs) -> tuple[list[str], str]:
    """Select the dense-constellation sample again; return the differences and a description."""
    config = recorded_config(manifest)
    seed, n = int(config.get("random_seed", 42)), int(manifest["inputs"]["active_n"])
    records = [tuple(record) for record in inputs.read(key="TLE_RAW_CACHE_PATH")["records"]]
    shuffled, counts = operational_sample(records, summary["reference_epoch"], seed)
    problems = _count_differences(counts, summary["satellite_selection"])
    sample = [catalogue_number(record[1]) for record in shuffled[:n]]
    pinned = inputs.read(key="SELECTION_CACHE_PATH")["norad_ids"]
    if sample != pinned:
        same = sum(ours == theirs for ours, theirs in zip(sample, pinned, strict=False))
        problems.append(f"sample of {len(sample)} differs from the pin of {len(pinned)} ({same} equal in place)")
    if len(sample) != int(summary["n_satellites"]):
        problems.append(f"{len(sample)} satellites sampled, the run used {summary['n_satellites']}")
    described = f"N={n}, seed {seed}: {counts['operational']} of {counts['fetched']} operational, sample equals the pin"
    return problems, described


def _check_real(summary: dict, manifest: dict, inputs: _Inputs) -> tuple[list[str], str]:
    """Select the real fleet again; return the differences and a description."""
    config = recorded_config(manifest)
    fleet = summary["fleet"]
    live = merge_groups(
        {group: inputs.read(name_starts=f"arf_gp_{group}.json")["records"] for group in CELESTRAK_GROUPS[fleet]}
    )
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
        summary["reference_epoch"],
        missions,
        min_age_days=min_age_days,
        exclude_names=config.get("exclude_satellites") or (),
    )
    problems = _count_differences(counts, summary["satellite_selection"])
    if missions != summary.get("mission_table"):
        problems.append("the mission table differs from the one the run recorded")
    recorded_fleet = inputs.read(key="SELECTION_CACHE_PATH")["satellites"]
    ours = fleet_file_entries(satellites)
    if ours != recorded_fleet:
        theirs_by_number = {entry["norad_id"]: entry for entry in recorded_fleet}
        differing = sum(theirs_by_number.get(entry["norad_id"]) != entry for entry in ours)
        problems.append(
            f"{differing} of {len(ours)} satellites differ from the fleet file, which lists {len(recorded_fleet)}"
        )
    described = f"{fleet}: {len(satellites)} satellites {counts['per_mission']} of {counts['fetched']} element sets"
    return problems, described


def check(folder: Path, cache_dir: Path | None) -> tuple[bool, str]:
    """Compare release 2 with the fleet of one run; return whether they agree and a line to print."""
    run_folder = find_run_folder(folder)
    summary_file = run_folder / "arf_cell2_summary.json"
    if not summary_file.exists():
        return False, "the run has no arf_cell2_summary.json"
    summary = json.loads(summary_file.read_text())
    manifest = read_manifest(run_folder)
    inputs = _Inputs(manifest, cache_dir)
    try:
        if summary.get("fleet", "starlink") == "starlink":
            problems, described = _check_dense(summary, manifest, inputs)
        else:
            problems, described = _check_real(summary, manifest, inputs)
    except FileNotFoundError as error:
        return False, f"cannot be checked: {error}"
    if inputs.changed:
        described += f" [changed since the run: {', '.join(sorted(set(inputs.changed)))}]"
    if problems:
        return False, f"{described}: " + " | ".join(problems)
    return True, described


def main(argv: list[str] | None = None) -> int:
    """Check every run folder named on the command line or listed as done in a batch index."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_folders", nargs="*", type=Path)
    parser.add_argument("--cache", type=Path, help="folder searched for an element-set file not at its recorded path")
    index_help = "batch index CSV; its done runs are checked"
    parser.add_argument("--index", type=Path, action="append", default=[], help=index_help)
    args = parser.parse_args(argv)
    folders = list(args.run_folders)
    for index in args.index:
        table = pd.read_csv(index)
        folders += [Path(p) for p in table.loc[table["status"] == "done", "out_dir"]]
    if not folders:
        parser.error("give at least one run folder or --index")
    failed = 0
    for number, folder in enumerate(folders, 1):
        agrees, detail = check(folder, args.cache)
        failed += not agrees
        verdict = "agrees" if agrees else "DOES NOT AGREE"
        print(f"[{number}/{len(folders)}] {folder.name}: {verdict}: {detail}", flush=True)
    print(f"{len(folders) - failed} of {len(folders)} runs agree with the registered release")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
