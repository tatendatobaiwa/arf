"""Check the onset comparison of release 2 against the finished runs of the registered releases.

Every run writes ``arf_onset_robustness.csv``, one row per onset with the floor and five selectors under clear sky and
cloud, and ``arf_onset_robustness_summary.json``, which states the protocol's H1a and H1b from it. For every run
folder given, both are computed again from the run's refined events (read exactly as written), with the run's cloud
model and the settings in its manifest, twice in the same session:

1. by release 2, and
2. by the registered release itself: Cell 6 up to its first selection, and Cell 7C's helpers and section 2,
   executed unchanged from the release file in this repository (``tests/parity/registered.py``), in a temporary folder.

Two verdicts follow.

* **Port.** Release 2 against the registered cells: table and summary must be identical, to the last bit. This shows
  that release 2 computes what the registered release computes. ``--no-registered`` skips it.
* **Run.** Release 2 against the run's own files. *Identical*; or *identical up to rounding*, when every value that is
  not a floating-point number (times, counts, images, the H1a and H1b counts and statements) is identical and the
  floating-point values differ by at most ``--rounding`` relative (default 1e-12); or *different*. A mean over thousands
  of utilities depends, in its last bits, on the order of summation, which NumPy may change between versions and
  processors; the run's NumPy version and this session's are printed.

Read-only; prints counts only. Needs ``h3``. Runs whose cloud probability came from the EarthEnv climatology are
reported as not checkable: that source is not ported.

Usage:
    python tools/check_onsets_on_runs.py RUN_FOLDER [RUN_FOLDER ...]
    python tools/check_onsets_on_runs.py --index ARF_batch/paper2_v156_a18_core_index.csv

Exit status 0 if every run agrees on both verdicts, 1 otherwise.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from arf.checking import (
    DEFAULT_ROUNDING,
    check_all,
    compare_summaries,
    compare_tables,
    folders_from,
    numpy_of,
    run_cloud,
    verdict_of,
)
from arf.events import EventTable
from arf.grid import region_cells_from_run_folder
from arf.report import OnsetSettings, onset_summary, onset_table
from arf.runs import find_run_folder, orbit_period_min, read_csv_exact, read_manifest, run_settings

NEEDED = (
    "arf_cell2_summary.json",
    "arf_onset_robustness.csv",
    "arf_onset_robustness_summary.json",
    "arf_refined_events.csv",
    "arf_mosaic_tiles.csv",
)
PARITY_HARNESS = Path(__file__).resolve().parents[1] / "tests" / "parity"


def _harness():
    if str(PARITY_HARNESS) not in sys.path:
        sys.path.insert(0, str(PARITY_HARNESS))
    import registered

    return registered


def compute(run_folder: Path, settings: dict):
    """Release 2's onset table and summary for a run."""
    events = EventTable.from_run_folder(run_folder)
    cell2 = json.loads((run_folder / "arf_cell2_summary.json").read_text())
    options = OnsetSettings(
        orbit_period_min=orbit_period_min(run_folder),
        cloud_probability=np.full(len(events.region_cells), float(settings["cloud_probability"])),
        budget=int(settings["max_selection_steps"]),
        decay_rate=settings["latency_decay_rate"],
        decay_scale=settings["latency_decay_scale"],
        window_hours=int(settings["window_hours"]),
        shift_hours=int(settings["onset_shift_hours"]),
        shift_step_min=int(settings["onset_shift_step_min"]),
        coverage_targets_pct=tuple(settings["coverage_targets_pct"]),
        reachable_level_pct=float(settings["h1_reachable_coverage_pct"]),
        random_draws=int(settings["random_baseline_draws"]),
        seed=int(settings["random_seed"]),
        cloud_knowledge=settings["cloud_knowledge"],
        cloud_planning=settings["cloud_planning"],
        time_objective=settings["time_aware_objective"],
    )
    table = onset_table(events, run_cloud(events, settings), options)
    summary = onset_summary(
        table,
        median_altitude_km=cell2["median_alt_km"],
        budget=int(settings["max_selection_steps"]),
        decay_rate=settings["latency_decay_rate"],
        decay_scale=settings["latency_decay_scale"],
        window_hours=int(settings["window_hours"]),
        shift_hours=int(settings["onset_shift_hours"]),
        shift_step_min=int(settings["onset_shift_step_min"]),
        coverage_targets_pct=tuple(settings["coverage_targets_pct"]),
        reachable_level_pct=settings["h1_reachable_coverage_pct"],
        min_onset_pct=settings["h1_min_onset_pct"],
        floor_margin_min=settings["h1_floor_margin_min"],
        min_evaluable_onsets=settings["h1_min_evaluable_onsets"],
    )
    return table, summary


def compute_registered(run_folder: Path, settings: dict, release: str):
    """The registered cells' onset table and summary for a run, from the same exact refined events."""
    refined = read_csv_exact(run_folder / "arf_refined_events.csv")
    cell2 = json.loads((run_folder / "arf_cell2_summary.json").read_text())
    with tempfile.TemporaryDirectory(prefix="arf_registered_onsets_") as work:
        namespace = _harness().run_cell7c_onsets(
            release,
            refined,
            list(region_cells_from_run_folder(run_folder)),
            work,
            budget=int(settings["max_selection_steps"]),
            orbit_period_min=cell2["t_max_wait_min"],
            median_altitude_km=cell2["median_alt_km"],
            decay_rate=settings["latency_decay_rate"],
            decay_scale=settings["latency_decay_scale"],
            cloud_probability=float(settings["cloud_probability"]),
            seed=int(settings["random_seed"]),
            cloud_model=settings["cloud_model"],
            length_km=float(settings["cloud_corr_length_km"]),
            time_h=float(settings["cloud_corr_time_h"]),
            window_hours=int(settings["window_hours"]),
            shift_hours=int(settings["onset_shift_hours"]),
            shift_step_min=int(settings["onset_shift_step_min"]),
            coverage_targets_pct=tuple(settings["coverage_targets_pct"]),
            reachable_level_pct=settings["h1_reachable_coverage_pct"],
            min_onset_pct=settings["h1_min_onset_pct"],
            floor_margin_min=settings["h1_floor_margin_min"],
            min_evaluable_onsets=settings["h1_min_evaluable_onsets"],
            random_draws=int(settings["random_baseline_draws"]),
            time_objective=settings["time_aware_objective"],
            cloud_knowledge=settings["cloud_knowledge"],
            cloud_planning=settings["cloud_planning"],
            swath_half_angle_deg=float(settings["swath_half_angle_deg"]),
        )
    return namespace["onset_df"], namespace["onset_robustness_summary"]


def _as_written(table: pd.DataFrame) -> pd.DataFrame:
    return read_csv_exact(io.StringIO(table.to_csv(index=False)))


def check(folder: Path, rounding: float, registered: bool) -> tuple[str, str]:
    """Check one run; return its verdict (``"agrees"``, ``"PORT DIFFERS"``, ...) and a line to print."""
    run_folder = find_run_folder(folder)
    missing = [name for name in NEEDED if not (run_folder / name).exists()]
    if missing:
        return "CANNOT BE CHECKED", f"the run lacks {', '.join(missing)}"
    settings = run_settings(run_folder)
    if not settings["onset_robustness_enabled"]:
        return "agrees", "the run did not compute the onset comparison; nothing to check"
    if settings["cloud_probability_source"] != "constant":
        return "CANNOT BE CHECKED", f"cloud probability from {settings['cloud_probability_source']!r} is not ported yet"
    started = time.time()
    manifest = read_manifest(run_folder)
    table, summary = compute(run_folder, settings)
    ours_written = json.loads(json.dumps(summary))
    parts = []
    if registered:
        release = str(manifest.get("framework_version") or "")
        release = release if release in _harness().RELEASES else "1.5.6"
        their_table, their_summary = compute_registered(run_folder, settings, release)
        port = compare_tables(table, their_table, 0.0, "onsets")
        port.add(compare_summaries(ours_written, json.loads(json.dumps(their_summary)), 0.0))
        parts.append(port.describe_port(release))
    else:
        port = None
    run = compare_tables(
        _as_written(table), read_csv_exact(run_folder / "arf_onset_robustness.csv"), rounding, "onsets"
    )
    run.add(
        compare_summaries(
            ours_written, json.loads((run_folder / "arf_onset_robustness_summary.json").read_text()), rounding
        )
    )
    parts.append(run.describe_run())
    h1a = summary["h1_revised"]["series"]["earliest_completion_clear"]
    h1b = summary["h1b_image_economy"]["series"]["clear"]
    parts.insert(
        0,
        f"{len(table)} onsets x {len(table.columns)} columns; H1a (ECC, clear) {h1a['n_within_margin']} of "
        f"{h1a['n_onsets_evaluated']} within {summary['h1_revised']['floor_margin_min']:.1f} min, H1b (clear) "
        f"{h1b['n_success']} of {h1b['n_onsets_evaluated']}",
    )
    parts.append(f"numpy {numpy_of(manifest)} at the run, {np.__version__} here; {time.time() - started:.0f} s")
    return verdict_of(port, run), " | ".join(parts)


def main(argv: list[str] | None = None) -> int:
    """Check every run folder named on the command line or listed as done in a batch index."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_folders", nargs="*", type=Path)
    index_help = "batch index CSV; its done runs are checked"
    parser.add_argument("--index", type=Path, action="append", default=[], help=index_help)
    rounding_help = "largest relative difference of a floating-point value still counted as rounding"
    parser.add_argument("--rounding", type=float, default=DEFAULT_ROUNDING, help=rounding_help)
    parser.add_argument(
        "--no-registered", action="store_true", help="skip the exact comparison with the registered cells"
    )
    args = parser.parse_args(argv)
    folders = folders_from(args.run_folders, args.index)
    if not folders:
        parser.error("give at least one run folder or --index")
    return check_all(folders, lambda folder: check(folder, args.rounding, not args.no_registered))


if __name__ == "__main__":
    sys.exit(main())
