"""Check the decay sweep, policy comparison, frontier and optimality gap of release 2 against the finished runs.

Cell 7C of every run writes, in section 1, ``arf_decay_sweep.csv``, ``arf_policy_comparison.csv`` and
``arf_cost_latency_frontier.csv``, and in section 3 ``arf_ilp_gap.csv``, whose summary goes into the scenario ledger
that the run's manifest records (``scenario_results``, entry ``ilp_gap``). For every run folder given, all are
computed again from the run's refined events (read exactly as written), with the run's cloud and the settings in its
manifest, twice in the same session:

1. by release 2 (``arf.report.policies``, ``arf.report.optimality``), and
2. by the registered release itself: Cell 6 up to its first selection, Cell 7C's helpers and its sections 1 and 3,
   executed unchanged from the release file in this repository (``tests/parity/registered.py``), in a temporary folder.

Two verdicts follow, as for the other tools (``arf.checking``): the **port**, release 2 against the registered cells,
identical to the last bit (``--no-registered`` skips it); and the **run**, release 2 against the run's own files,
identical, or identical up to rounding of floating-point values (``--rounding``, default 1e-12 relative).

The ILP solver's time (``ilp_seconds``) is wall-clock time and is never compared; the line reports the run's own total.
An instance the solver ends at its time limit (``ilp_time_limit_s``, 120 s by default) can end differently on another
machine: such a difference is reported, not hidden. ``--no-ilp`` leaves section 3 out; it is most of the time.

Read-only; prints counts only. Needs ``h3``. Runs whose cloud probability came from the EarthEnv climatology are
reported as not checkable: it is not ported.

Usage:
    python tools/check_policies_on_runs.py RUN_FOLDER [RUN_FOLDER ...]
    python tools/check_policies_on_runs.py --index ARF_batch/paper2_v156_a18_core_index.csv

Exit status 0 if every run agrees on both verdicts, 1 otherwise.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from arf.checking import (
    DEFAULT_ROUNDING,
    Differences,
    check_all,
    compare_summaries,
    compare_tables,
    folders_from,
    numpy_of,
    onset_settings,
    run_cloud,
    verdict_of,
)
from arf.events import EventTable
from arf.grid import region_cells_from_run_folder
from arf.report.optimality import IlpSettings, ilp_gap_summary, ilp_gap_table
from arf.report.policies import cost_latency_frontier, decay_sweep, policy_comparison
from arf.runs import find_run_folder, read_csv_exact, read_manifest, run_settings

NEEDED = (
    "arf_cell2_summary.json",
    "arf_decay_sweep.csv",
    "arf_policy_comparison.csv",
    "arf_cost_latency_frontier.csv",
    "arf_refined_events.csv",
    "arf_mosaic_tiles.csv",
)
TABLES = {
    "decay_sweep": "arf_decay_sweep.csv",
    "policies": "arf_policy_comparison.csv",
    "frontier": "arf_cost_latency_frontier.csv",
    "ilp": "arf_ilp_gap.csv",
}
#: Wall-clock time; no rerun reproduces it.
NOT_COMPARED = ("ilp_seconds",)
PARITY_HARNESS = Path(__file__).resolve().parents[1] / "tests" / "parity"


def _harness():
    if str(PARITY_HARNESS) not in sys.path:
        sys.path.insert(0, str(PARITY_HARNESS))
    import registered

    return registered


def ilp_settings(settings: dict) -> IlpSettings:
    """Release 2's settings of section 3 for a run."""
    return IlpSettings(
        enabled=bool(settings["ilp_enabled"]),
        instances=int(settings["ilp_instances"]),
        horizon_hours=int(settings["ilp_horizon_hours"]),
        budget=int(settings["ilp_budget_k"]),
        time_limit_s=float(settings["ilp_time_limit_s"]),
        max_pairs=int(settings["ilp_max_pairs"]),
    )


def compute(run_folder: Path, settings: dict, with_ilp: bool) -> dict:
    """Release 2's tables (and ILP summary) for a run."""
    events = EventTable.from_run_folder(run_folder)
    is_cloudy = run_cloud(events, settings)
    options = onset_settings(run_folder, events, settings)
    results: dict = {
        "decay_sweep": decay_sweep(events, is_cloudy, options, settings["decay_sweep"]),
        "policies": policy_comparison(events, is_cloudy, options),
        "frontier": cost_latency_frontier(events, is_cloudy, options),
    }
    if with_ilp:
        ilp = ilp_settings(settings)
        table = ilp_gap_table(events, ilp, options, int(settings["window_hours"]))
        results["ilp"] = table
        results["ilp_summary"] = None if table is None else ilp_gap_summary(table, ilp, options.time_objective)
    return results


def compute_registered(run_folder: Path, settings: dict, release: str, with_ilp: bool) -> dict:
    """The registered cells' tables (and ILP summary) for a run, from the same inputs."""
    harness = _harness()
    refined = read_csv_exact(run_folder / "arf_refined_events.csv")
    cell2 = json.loads((run_folder / "arf_cell2_summary.json").read_text())
    common = {
        "budget": int(settings["max_selection_steps"]),
        "orbit_period_min": cell2["t_max_wait_min"],
        "median_altitude_km": cell2["median_alt_km"],
        "decay_rate": settings["latency_decay_rate"],
        "decay_scale": settings["latency_decay_scale"],
        "cloud_probability": float(settings["cloud_probability"]),
        "seed": int(settings["random_seed"]),
        "cloud_model": settings["cloud_model"],
        "length_km": float(settings["cloud_corr_length_km"]),
        "time_h": float(settings["cloud_corr_time_h"]),
        "window_hours": int(settings["window_hours"]),
        "coverage_targets_pct": tuple(settings["coverage_targets_pct"]),
        "reachable_level_pct": float(settings["h1_reachable_coverage_pct"]),
        "random_draws": int(settings["random_baseline_draws"]),
        "time_objective": settings["time_aware_objective"],
        "cloud_knowledge": settings["cloud_knowledge"],
        "cloud_planning": settings["cloud_planning"],
        "swath_half_angle_deg": float(settings["swath_half_angle_deg"]),
    }
    cells = list(region_cells_from_run_folder(run_folder))
    with tempfile.TemporaryDirectory(prefix="arf_registered_policies_") as work:
        namespace = harness.run_cell7c_policies(
            release, refined, cells, work, decay_sweep=tuple(settings["decay_sweep"]), **common
        )
        results = {
            "decay_sweep": namespace["decay_sweep_df"],
            "policies": namespace["policy_df"],
            "frontier": namespace["frontier_df"],
        }
        if with_ilp:
            ilp = {f"ilp_{k}": settings[f"ilp_{k}"] for k in ("enabled", "instances", "horizon_hours")}
            ilp.update(
                ilp_budget_k=settings["ilp_budget_k"],
                ilp_time_limit_s=settings["ilp_time_limit_s"],
                ilp_max_pairs=settings["ilp_max_pairs"],
            )
            namespace = harness.run_cell7c_ilp(release, refined, cells, work, **ilp, **common)
            results["ilp"] = namespace.get("ilp_df")
            results["ilp_summary"] = namespace["ilp_summary"] or None
    return results


def _as_written(table: pd.DataFrame) -> pd.DataFrame:
    return read_csv_exact(io.StringIO(table.to_csv(index=False)))


def _without_time(table: pd.DataFrame) -> pd.DataFrame:
    return table.drop(columns=[c for c in NOT_COMPARED if c in table.columns])


def ledger_entry(manifest: dict, policies: pd.DataFrame) -> dict | None:
    """The run's own entry of the scenario ledger: the one whose policy comparison is the run's file."""
    coverage = pd.to_numeric(policies["coverage_pct"], errors="coerce").to_numpy(dtype=float)
    matches = []
    for entry in (manifest.get("scenario_results") or {}).values():
        rows = entry.get("policy_comparison") if isinstance(entry, dict) else None
        if not rows or len(rows) != len(coverage):
            continue
        theirs = np.array([np.nan if r.get("coverage_pct") is None else r["coverage_pct"] for r in rows], dtype=float)
        if np.allclose(theirs, coverage, rtol=1e-12, atol=0.0, equal_nan=True):
            matches.append(entry)
    return matches[0] if len(matches) == 1 else None


def _compare(ours: dict, theirs: dict, rounding: float, written: bool) -> Differences:
    """Compare release 2's results with another's (registered cells, or the run's files)."""
    found = Differences()
    for name in TABLES:
        if name not in ours:
            continue
        a, b = ours[name], theirs.get(name)
        if (a is None) != (b is None):
            text = f"{TABLES[name]} computed on one side only"
            found.exact.append(text)
            found.beyond_rounding.append(text)
            continue
        if a is None:
            continue
        a = _as_written(a) if written else a
        found.add(compare_tables(_without_time(a), _without_time(b), rounding, "rows"))
    if "ilp_summary" in ours and theirs.get("ilp_summary", "absent") != "absent":
        a = json.loads(json.dumps(ours["ilp_summary"]))
        found.add(compare_summaries(a, json.loads(json.dumps(theirs["ilp_summary"])), rounding, "ilp_gap"))
    return found


def _ratio(value) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.4f}"


def check(folder: Path, rounding: float, registered: bool, with_ilp: bool) -> tuple[str, str]:
    """Check one run; return its verdict (``"agrees"``, ``"PORT DIFFERS"``, ...) and a line to print."""
    run_folder = find_run_folder(folder)
    missing = [name for name in NEEDED if not (run_folder / name).exists()]
    if missing:
        return "CANNOT BE CHECKED", f"the run lacks {', '.join(missing)}"
    settings = run_settings(run_folder)
    if settings["cloud_probability_source"] != "constant":
        return "CANNOT BE CHECKED", f"cloud probability from {settings['cloud_probability_source']!r} is not ported yet"
    with_ilp = with_ilp and bool(settings["ilp_enabled"])
    if with_ilp and not (run_folder / TABLES["ilp"]).exists():
        return "CANNOT BE CHECKED", f"the run's settings enable the ILP but it lacks {TABLES['ilp']}"
    started = time.time()
    manifest = read_manifest(run_folder)
    ours = compute(run_folder, settings, with_ilp)
    parts = []
    port = None
    if registered:
        release = str(manifest.get("framework_version") or "")
        release = release if release in _harness().RELEASES else "1.5.6"
        port = _compare(ours, compute_registered(run_folder, settings, release, with_ilp), 0.0, written=False)
        parts.append(port.describe_port(release))
    theirs: dict = {name: read_csv_exact(run_folder / TABLES[name]) for name in ("decay_sweep", "policies", "frontier")}
    notes = []
    if with_ilp:
        theirs["ilp"] = read_csv_exact(run_folder / TABLES["ilp"])
        entry = ledger_entry(manifest, theirs["policies"])
        if entry is not None and "ilp_gap" in entry:
            theirs["ilp_summary"] = entry["ilp_gap"]
        else:
            notes.append("ILP summary not compared: the manifest's ledger has no single entry for this run")
        seconds = pd.to_numeric(theirs["ilp"]["ilp_seconds"], errors="coerce").sum()
        notes.append(f"the run's ILP solves took {seconds:.0f} s")
    run = _compare(ours, theirs, rounding, written=True)
    parts.append(run.describe_run())
    head = (
        f"decay sweep of {len(ours['decay_sweep'])} rates, {len(ours['policies'])} policy rows, "
        f"frontier of {len(ours['frontier'])} rows"
    )
    if with_ilp and ours["ilp"] is not None:
        table, summary = ours["ilp"], ours["ilp_summary"]
        head += (
            f"; ILP {summary['optimal']} of {summary['instances']} proven optimal, ratio to the optimum (minimum): "
            f"coverage-first {_ratio(summary['coverage_first_ratio_min'])}, "
            f"time-aware {_ratio(summary['time_aware_ratio_min'])}, "
            f"time-aware on its own objective {_ratio(summary['time_aware_tw_ratio_min'])}; "
            f"ECC cover at the minimum in {summary['ecc_matches_exact']} of {summary['ecc_instances_exact']}"
        )
        if table["ilp_status"].astype(str).str.startswith("time limit").any():
            notes.append("some solves ended at the time limit: they may differ between machines")
    elif not with_ilp:
        head += "; ILP not checked"
    parts.insert(0, head)
    parts += notes
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
    parser.add_argument("--no-ilp", action="store_true", help="leave section 3 (the optimality gap) out")
    args = parser.parse_args(argv)
    folders = folders_from(args.run_folders, args.index)
    if not folders:
        parser.error("give at least one run folder or --index")
    return check_all(folders, lambda folder: check(folder, args.rounding, not args.no_registered, not args.no_ilp))


if __name__ == "__main__":
    sys.exit(main())
