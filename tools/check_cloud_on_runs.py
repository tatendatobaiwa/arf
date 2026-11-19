"""Check the cloud masks of release 2 against the finished runs of the registered releases.

A registered run does not save its cloud masks, but ``arf_cloud_robustness.csv`` records, for each of its cloud
realisations, the seed and the share of observations the mask made cloudy (``cloudy_rows_pct``). For every run folder
given, release 2 rebuilds the event table from the run's own files and the mask of every seed, with the cloud model,
correlation length, correlation time and probability recorded in the run's manifest, and compares the shares.
Read-only. Needs ``h3``.

The check is on the cloudy share of each realisation, not on each observation; the observation-by-observation
comparison with the registered code is in ``tests/parity/test_cloud_matches_registered_release.py``, on synthetic
events.

Usage:
    python tools/check_cloud_on_runs.py RUN_FOLDER [RUN_FOLDER ...]
    python tools/check_cloud_on_runs.py --index ARF_batch/paper2_v156_a18_core_index.csv

Exit status 0 if every realisation of every run agrees, 1 otherwise.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

from arf.cloud import LatentCloudField, correlated_cloud_mask, independent_cloud_mask
from arf.events import EventTable
from arf.grid import cell_centres
from arf.runs import find_run_folder, run_settings


def check(folder: Path) -> tuple[bool, str]:
    """Compare release 2 with the cloud realisations of one run; return whether they agree and a line to print."""
    run_folder = find_run_folder(folder)
    realisations_file = run_folder / "arf_cloud_robustness.csv"
    if not realisations_file.exists():
        return False, "the run has no arf_cloud_robustness.csv"
    realisations = pd.read_csv(realisations_file)
    settings = run_settings(run_folder)
    if settings["cloud_probability_source"] != "constant":
        return False, f"cloud probability from {settings['cloud_probability_source']!r} is not ported yet"
    events = EventTable.from_run_folder(run_folder)
    probability = float(settings["cloud_probability"])
    field = None
    if settings["cloud_model"] == "correlated":
        latitude, longitude = cell_centres(events.region_cells)
        field = LatentCloudField(latitude, longitude, float(settings["cloud_corr_length_km"]))
    differing = []
    for row in realisations.itertuples():
        seed = int(row.seed)
        if field is None:
            mask = independent_cloud_mask(events, probability, seed)
        else:
            mask = correlated_cloud_mask(events, probability, seed, field, float(settings["cloud_corr_time_h"]))
        ours = 100.0 * int(mask.sum()) / max(1, mask.size)
        if not math.isclose(ours, float(row.cloudy_rows_pct), rel_tol=0.0, abs_tol=1e-9):
            differing.append((seed, ours, float(row.cloudy_rows_pct)))
    described = f"{len(realisations)} realisations, {events.n_observations} observations, {settings['cloud_model']}"
    if differing:
        return False, f"{len(differing)} of {described} differ; first (seed, release 2, run): {differing[0]}"
    return True, described


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
        agrees, detail = check(folder)
        failed += not agrees
        verdict = "agrees" if agrees else "DOES NOT AGREE"
        print(f"[{number}/{len(folders)}] {folder.name}: {verdict}: {detail}", flush=True)
    print(f"{len(folders) - failed} of {len(folders)} runs agree with the registered release")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
