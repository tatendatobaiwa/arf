"""Check arf.selection against the finished runs of the registered releases.

For every run folder given, the floor, the FCFS image count, the earliest-completion time and image count (budget
30) and the reachable cells are recomputed with release 2 from the run's own ``arf_refined_events.csv`` and compared,
onset by onset, with the run's ``arf_onset_robustness.csv``, which the registered release wrote. Read-only.

Usage:
    python tools/check_selection_on_runs.py RUN_FOLDER [RUN_FOLDER ...]
    python tools/check_selection_on_runs.py --index ARF_batch/paper2_v156_a18_core_index.csv

Exit status 0 if every onset of every run agrees, 1 otherwise.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from arf.selection import ImageSet, evaluate_onset, onset_grid

#: Release 2 quantity -> column of arf_onset_robustness.csv.
COLUMNS = {
    "floor_min": "best_possible_clear_t95r_min",
    "ecc_min": "earliest_completion_clear_t95r_min",
    "ecc_images": "earliest_completion_clear_images95r",
    "fcfs_images": "fcfs_unbudgeted_clear_images95r",
    "reachable": "reachable_clear_tiles",
}


def differences(run_folder: Path) -> tuple[int, list[tuple[int, str, object, object]]]:
    """Return the number of onsets compared and every value that differs from the run's own file."""
    images = ImageSet.from_run_folder(run_folder)
    registered = pd.read_csv(run_folder / "arf_onset_robustness.csv").set_index("onset_shift_min")
    found = []
    compared = 0
    for onset in onset_grid():
        result = evaluate_onset(images, onset)
        if result is None:
            continue
        compared += 1
        ours = {
            "floor_min": result.floor_min,
            "ecc_min": result.completion[30].time_min,
            "ecc_images": result.completion[30].images,
            "fcfs_images": result.fcfs_images,
            "reachable": result.reachable,
        }
        for name, column in COLUMNS.items():
            theirs = registered.loc[onset, column]
            theirs = None if pd.isna(theirs) else float(theirs)
            mine = None if ours[name] is None else float(ours[name])
            agree = (mine is None and theirs is None) or (
                mine is not None and theirs is not None and math.isclose(mine, theirs)
            )
            if not agree:
                found.append((onset, name, mine, theirs))
    return compared, found


def main(argv: list[str] | None = None) -> int:
    """Check every run folder named on the command line or listed as done in a batch index."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_folders", nargs="*", type=Path)
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
        compared, found = differences(folder)
        if found:
            failed += 1
            print(f"[{number}/{len(folders)}] {folder.name}: {len(found)} DIFFERENCES in {compared} onsets")
            print(f"    first: {found[:3]}")
        else:
            print(f"[{number}/{len(folders)}] {folder.name}: {compared} onsets agree", flush=True)
    print(f"{len(folders) - failed} of {len(folders)} runs agree with the registered release")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
