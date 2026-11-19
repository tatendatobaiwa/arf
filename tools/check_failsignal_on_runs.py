"""Check the Fail-Signal selector and its comparison plans of release 2 against finished runs.

``arf_cloud_robustness.csv`` of a registered run records, for each of its cloud realisations and for each objective,
the number of images and the coverage of six plans (the Fail-Signal selector, two open-loop plans, the greedy with
the cloud known, the earlier selector and an open-loop plan of its size), the fail-signals and the compensatory
images. For every run folder given, release 2 recomputes all of them from the run's own files, with the settings in
the run's manifest, and compares every figure. Read-only. Needs ``h3``.

Large runs take long: every realisation re-plans after each arriving image. ``--seeds 3`` checks the first three
realisations of each run only.

Usage:
    python tools/check_failsignal_on_runs.py RUN_FOLDER [RUN_FOLDER ...] [--seeds N]
    python tools/check_failsignal_on_runs.py --index ARF_batch/paper2_v156_a18_core_index.csv [--seeds N]

Exit status 0 if every figure of every realisation checked agrees, 1 otherwise.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from arf.cloud import LatentCloudField, correlated_cloud_mask, independent_cloud_mask
from arf.comparison import cloud_trial
from arf.events import EventTable
from arf.grid import cell_centres
from arf.runs import find_run_folder, orbit_period_min, run_settings

#: Plan name in arf_cloud_robustness.csv -> plan name in release 2.
PLAN_NAMES = {
    "adaptive": "adaptive",
    "openloop_expected": "open_loop_expected",
    "openloop_clear": "open_loop_clear",
    "oracle": "oracle",
    "failsignal_v140": "adaptive_clear_planning",
    "openloop_expected_v140n": "open_loop_expected_at_that_size",
}
NOT_PORTED = {"cloud_probability_source": "constant", "cloud_knowledge": "realistic", "cloud_planning": "expected"}


def trial_figures(trial, n_cells: int) -> dict[str, float]:
    """Lay out a release 2 trial under the column names of ``arf_cloud_robustness.csv``."""
    figures = {"cloudy_rows_pct": trial.cloudy_share_percent, "reachable_tiles": trial.reachable_cells}
    for objective, result in trial.objectives.items():
        figures[f"{objective}_replans"] = result.replans
        figures[f"{objective}_compensatory"] = result.compensatory
        for registered_name, name in PLAN_NAMES.items():
            figures[f"{objective}_{registered_name}_images"] = result.plans[name].images
            figures[f"{objective}_{registered_name}_cov_pct"] = 100.0 * result.plans[name].covered_cells / n_cells
    return figures


def check(folder: Path, n_seeds: int | None) -> tuple[bool, str]:
    """Compare release 2 with the cloud realisations of one run; return whether they agree and a line to print."""
    run_folder = find_run_folder(folder)
    realisations_file = run_folder / "arf_cloud_robustness.csv"
    if not realisations_file.exists():
        return False, "the run has no arf_cloud_robustness.csv"
    realisations = pd.read_csv(realisations_file)
    if n_seeds is not None:
        realisations = realisations.head(n_seeds)
    settings = run_settings(run_folder)
    for name, ported in NOT_PORTED.items():
        if settings[name] != ported:
            return False, f"{name} = {settings[name]!r} is not ported yet"
    events = EventTable.from_run_folder(run_folder)
    n_cells = len(events.region_cells)
    probability = np.full(n_cells, float(settings["cloud_probability"]))
    field = None
    if settings["cloud_model"] == "correlated":
        latitude, longitude = cell_centres(events.region_cells)
        field = LatentCloudField(latitude, longitude, float(settings["cloud_corr_length_km"]))
    objectives = (("time_aware", float(settings["latency_decay_rate"])), ("coverage_first", 0.0))
    differing, compared = [], 0
    for row in realisations.itertuples():
        seed = int(row.seed)
        if field is None:
            is_cloudy = independent_cloud_mask(events, float(settings["cloud_probability"]), seed)
        else:
            time_h = float(settings["cloud_corr_time_h"])
            is_cloudy = correlated_cloud_mask(events, float(settings["cloud_probability"]), seed, field, time_h)
        trial = cloud_trial(
            events,
            is_cloudy,
            probability,
            int(settings["max_selection_steps"]),
            orbit_period_min(run_folder),
            objectives,
            settings["time_aware_objective"],
            settings["latency_decay_scale"],
        )
        for column, ours in trial_figures(trial, n_cells).items():
            if column not in realisations.columns:
                differing.append((seed, column, ours, "column missing"))
                continue
            compared += 1
            theirs = float(getattr(row, column))
            if not math.isclose(float(ours), theirs, rel_tol=0.0, abs_tol=1e-9):
                differing.append((seed, column, float(ours), theirs))
    described = f"{len(realisations)} realisations, {compared} figures, {len(events)} images"
    if differing:
        return False, f"{len(differing)} of {described} differ; first (seed, figure, release 2, run): {differing[:2]}"
    return True, described


def main(argv: list[str] | None = None) -> int:
    """Check every run folder named on the command line or listed as done in a batch index."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_folders", nargs="*", type=Path)
    index_help = "batch index CSV; its done runs are checked"
    parser.add_argument("--index", type=Path, action="append", default=[], help=index_help)
    parser.add_argument("--seeds", type=int, default=None, help="check only the first N realisations of each run")
    args = parser.parse_args(argv)
    folders = list(args.run_folders)
    for index in args.index:
        table = pd.read_csv(index)
        folders += [Path(p) for p in table.loc[table["status"] == "done", "out_dir"]]
    if not folders:
        parser.error("give at least one run folder or --index")
    failed = 0
    for number, folder in enumerate(folders, 1):
        started = time.monotonic()
        agrees, detail = check(folder, args.seeds)
        failed += not agrees
        seconds = time.monotonic() - started
        verdict = "agrees" if agrees else "DOES NOT AGREE"
        print(f"[{number}/{len(folders)}] {folder.name}: {verdict}: {detail} ({seconds:.0f} s)", flush=True)
    print(f"{len(folders) - failed} of {len(folders)} runs agree with the registered release")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
