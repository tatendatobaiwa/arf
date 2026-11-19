"""Check the cloud robustness tables and H2 of release 2 against the finished runs of the registered releases.

Cell 7C of every run writes ``arf_cloud_robustness.csv`` (one row per cloud realisation: the plans of both objectives,
the fail-signals, and the floor, ECC and FCFS at the level of the reachable cells), ``arf_cloud_sensitivity.csv`` (H2
at other correlation lengths and times) and ``arf_cloud_robustness_summary.json``, which states H2 (sections 4 and 5).
For every run folder given, all three are computed again from the run's refined events (read exactly as written),
with the run's cloud model and the settings in its manifest, twice in the same session:

1. by release 2 (``arf.report.cloud_robustness``), and
2. by the registered release itself: Cell 6 up to its first selection, Cell 7C's helpers and its sections 4 and 5,
   executed unchanged from the release file in this repository (``tests/parity/registered.py``), in a temporary folder.

Two verdicts follow, as for the onset tool (``arf.checking``): the **port**, release 2 against the registered cells,
identical to the last bit (``--no-registered`` skips it); and the **run**, release 2 against the run's own files,
identical, or identical up to rounding of floating-point values (``--rounding``, default 1e-12 relative).

Runs take long: every realisation re-plans after each arriving image, 20 realisations for two objectives and 90 for
the sweep with the releases' settings, and the port doubles it. ``--no-sensitivity`` leaves the sweep out of both
sides (and its file and summary entry out of the comparison), about a third of the time.

Read-only; prints counts only. Needs ``h3``. Runs whose cloud probability came from the EarthEnv climatology, or that
knew the cloud in advance, are reported as not checkable: neither is ported.

Usage:
    python tools/check_cloud_robustness_on_runs.py RUN_FOLDER [RUN_FOLDER ...]
    python tools/check_cloud_robustness_on_runs.py --index ARF_batch/paper2_v156_a18_core_index.csv

Exit status 0 if every run agrees on both verdicts, 1 otherwise.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
from dataclasses import replace
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
    verdict_of,
)
from arf.events import EventTable
from arf.grid import region_cells_from_run_folder
from arf.report.cloud_robustness import (
    CloudModel,
    CloudRobustnessSettings,
    cloud_robustness_summary,
    cloud_robustness_table,
    sensitivity_rows,
    summary_json,
)
from arf.report.onsets_summary import orbital_period_min
from arf.runs import find_run_folder, read_csv_exact, read_manifest, run_settings

NEEDED = (
    "arf_cell2_summary.json",
    "arf_cloud_robustness.csv",
    "arf_cloud_robustness_summary.json",
    "arf_refined_events.csv",
    "arf_mosaic_tiles.csv",
)
PARITY_HARNESS = Path(__file__).resolve().parents[1] / "tests" / "parity"


def _harness():
    if str(PARITY_HARNESS) not in sys.path:
        sys.path.insert(0, str(PARITY_HARNESS))
    import registered

    return registered


def settings_of(run_folder: Path, settings: dict, with_sensitivity: bool) -> CloudRobustnessSettings:
    """Release 2's settings for a run, from its manifest and Cell 2's orbital period."""
    cell2 = json.loads((run_folder / "arf_cell2_summary.json").read_text())
    options = CloudRobustnessSettings(
        orbit_period_min=float(cell2["t_max_wait_min"]),
        cloud_probability=float(settings["cloud_probability"]),
        budget=int(settings["max_selection_steps"]),
        decay_rate=float(settings["latency_decay_rate"]),
        decay_scale=settings["latency_decay_scale"],
        time_objective=settings["time_aware_objective"],
        seed=int(settings["random_seed"]),
        realisations=int(settings["cloud_seeds"]),
        cloud_model=settings["cloud_model"],
        length_km=float(settings["cloud_corr_length_km"]),
        time_h=float(settings["cloud_corr_time_h"]),
        sweep_lengths_km=tuple(float(x) for x in settings["cloud_sweep_lengths_km"]),
        sweep_times_h=tuple(float(x) for x in settings["cloud_sweep_times_h"]),
        sweep_realisations=int(settings["cloud_sweep_seeds"]),
        reachable_level_pct=float(settings["h1_reachable_coverage_pct"]),
        cloud_knowledge=settings["cloud_knowledge"],
        cloud_planning=settings["cloud_planning"],
    )
    return options if with_sensitivity else replace(options, sweep_lengths_km=(), sweep_times_h=())


def floor_margin(run_folder: Path, settings: dict) -> float:
    """The H1 margin of the run: as set, or one orbital period at the fleet's median altitude."""
    if settings["h1_floor_margin_min"] is not None:
        return float(settings["h1_floor_margin_min"])
    cell2 = json.loads((run_folder / "arf_cell2_summary.json").read_text())
    return orbital_period_min(cell2["median_alt_km"])


def compute(run_folder: Path, options: CloudRobustnessSettings, margin: float):
    """Release 2's realisation table, sensitivity rows and summary for a run."""
    events = EventTable.from_run_folder(run_folder)
    model = CloudModel(events, options)
    table = cloud_robustness_table(events, options, model)
    if table is None:
        return None, None, None
    sensitivity = sensitivity_rows(events, options, model)
    return table, sensitivity, cloud_robustness_summary(table, options, margin, model.record(), sensitivity)


def compute_registered(run_folder: Path, settings: dict, options: CloudRobustnessSettings, release: str):
    """The registered cells' realisation table, sensitivity rows and summary for a run, from the same inputs."""
    refined = read_csv_exact(run_folder / "arf_refined_events.csv")
    cell2 = json.loads((run_folder / "arf_cell2_summary.json").read_text())
    with tempfile.TemporaryDirectory(prefix="arf_registered_cloud_") as work:
        namespace = _harness().run_cell7c_cloud(
            release,
            refined,
            list(region_cells_from_run_folder(run_folder)),
            work,
            cloud_seeds=options.realisations,
            sweep_lengths_km=options.sweep_lengths_km,
            sweep_times_h=options.sweep_times_h,
            sweep_seeds=options.sweep_realisations,
            budget=options.budget,
            orbit_period_min=cell2["t_max_wait_min"],
            median_altitude_km=cell2["median_alt_km"],
            decay_rate=options.decay_rate,
            decay_scale=options.decay_scale,
            cloud_probability=options.cloud_probability,
            seed=options.seed,
            cloud_model=options.cloud_model,
            length_km=options.length_km,
            time_h=options.time_h,
            reachable_level_pct=options.reachable_level_pct,
            floor_margin_min=settings["h1_floor_margin_min"],
            time_objective=options.time_objective,
            cloud_knowledge=options.cloud_knowledge,
            cloud_planning=options.cloud_planning,
            swath_half_angle_deg=float(settings["swath_half_angle_deg"]),
        )
    table = namespace.get("cloud_seed_df")
    return table, namespace["cloud_sensitivity_rows"] or None, namespace["cloud_robustness_summary"] or None


def _as_written(table: pd.DataFrame) -> pd.DataFrame:
    return read_csv_exact(io.StringIO(table.to_csv(index=False)))


def _written_summary(summary: dict | None) -> dict | None:
    return None if summary is None else json.loads(summary_json(summary))


def _compare(ours, theirs, rounding: float, written: bool):
    """Compare release 2's three results with another's (registered cells, or the run's files)."""
    table, sensitivity, summary = ours
    their_table, their_sensitivity, their_summary = theirs
    if written:
        table = _as_written(table)
        sensitivity_table = None if sensitivity is None else _as_written(pd.DataFrame(sensitivity))
    else:
        sensitivity_table = None if sensitivity is None else pd.DataFrame(sensitivity)
    found = compare_tables(table, their_table, rounding, "realisations")
    if (sensitivity_table is None) != (their_sensitivity is None):
        text = "sensitivity computed on one side only"
        found.exact.append(text)
        found.beyond_rounding.append(text)
    elif sensitivity_table is not None:
        their_sensitivity = their_sensitivity if written else pd.DataFrame(their_sensitivity)
        found.add(compare_tables(sensitivity_table, their_sensitivity, rounding, "settings"))
    found.add(compare_summaries(_written_summary(summary), _written_summary(their_summary), rounding))
    return found


def check(folder: Path, rounding: float, registered: bool, with_sensitivity: bool) -> tuple[str, str]:
    """Check one run; return its verdict (``"agrees"``, ``"PORT DIFFERS"``, ...) and a line to print."""
    run_folder = find_run_folder(folder)
    missing = [name for name in NEEDED if not (run_folder / name).exists()]
    if missing:
        return "CANNOT BE CHECKED", f"the run lacks {', '.join(missing)}"
    settings = run_settings(run_folder)
    if settings["cloud_probability_source"] != "constant":
        return "CANNOT BE CHECKED", f"cloud probability from {settings['cloud_probability_source']!r} is not ported yet"
    if settings["cloud_knowledge"] != "realistic":
        return "CANNOT BE CHECKED", f"cloud_knowledge {settings['cloud_knowledge']!r} is not ported"
    started = time.time()
    manifest = read_manifest(run_folder)
    options = settings_of(run_folder, settings, with_sensitivity)
    ours = compute(run_folder, options, floor_margin(run_folder, settings))
    if ours[0] is None:
        return (
            "CANNOT BE CHECKED",
            "release 2 computes no realisations with the run's settings, yet the run has the files",
        )
    parts = []
    port = None
    if registered:
        release = str(manifest.get("framework_version") or "")
        release = release if release in _harness().RELEASES else "1.5.6"
        port = _compare(ours, compute_registered(run_folder, settings, options, release), 0.0, written=False)
        parts.append(port.describe_port(release))
    sensitivity_file = run_folder / "arf_cloud_sensitivity.csv"
    their_summary = json.loads((run_folder / "arf_cloud_robustness_summary.json").read_text())
    if with_sensitivity:
        their_sensitivity = read_csv_exact(sensitivity_file) if sensitivity_file.exists() else None
    else:
        their_sensitivity = None
        their_summary.pop("sensitivity", None)
    theirs = (read_csv_exact(run_folder / "arf_cloud_robustness.csv"), their_sensitivity, their_summary)
    run = _compare(ours, theirs, rounding, written=True)
    parts.append(run.describe_run())
    table, sensitivity, summary = ours
    h2 = summary["h2"]
    primary, other = h2["time_aware"], h2["coverage_first"]
    difference = primary["difference_vs_openloop_expected"]
    shown = "n/a" if difference["mean"] is None else f"{difference['mean']:.2f}"
    bound = "n/a" if difference["lcb"] is None else f"{difference['lcb']:.2f}"
    parts.insert(
        0,
        f"{len(table)} realisations x {len(table.columns)} columns"
        + (f", sweep of {len(sensitivity)} settings" if sensitivity else ", no sweep")
        + f"; H2 (time-aware) {primary['status']}: mean {shown} points, lower bound {bound}; "
        f"coverage-first {other['status']}",
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
    parser.add_argument("--no-sensitivity", action="store_true", help="leave the sweep of the cloud model out")
    args = parser.parse_args(argv)
    folders = folders_from(args.run_folders, args.index)
    if not folders:
        parser.error("give at least one run folder or --index")
    return check_all(
        folders, lambda folder: check(folder, args.rounding, not args.no_registered, not args.no_sensitivity)
    )


if __name__ == "__main__":
    sys.exit(main())
