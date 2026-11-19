"""Reading the output folder of a run of the registered releases."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

#: Settings of the registered releases, used when a manifest does not state one.
RELEASE_DEFAULTS = {
    "random_seed": 42,
    "max_selection_steps": 30,
    "latency_decay_rate": 3.0,
    "latency_decay_scale": "fleet",
    "time_aware_objective": "facility",
    "cloud_knowledge": "realistic",
    "cloud_planning": "expected",
    "cloud_model": "correlated",
    "cloud_corr_length_km": 50.0,
    "cloud_corr_time_h": 3.0,
    "cloud_probability": 0.35,
    "cloud_probability_source": "constant",
    "window_hours": 48,
    "onset_robustness_enabled": True,
    "onset_shift_hours": 24,
    "onset_shift_step_min": 20,
    "coverage_targets_pct": [95.0, 99.0],
    "h1_reachable_coverage_pct": 95.0,
    "h1_min_onset_pct": 90.0,
    "h1_floor_margin_min": None,
    "h1_min_evaluable_onsets": None,
    "random_baseline_draws": 10,
    "swath_half_angle_deg": 9.22,
    "cloud_seeds": 20,
    "cloud_sweep_lengths_km": [0.0, 10.0, 25.0, 50.0, 100.0, 200.0],
    "cloud_sweep_times_h": [1.0, 3.0, 6.0, 24.0],
    "cloud_sweep_seeds": 10,
    "decay_sweep": [0.5, 1.0, 2.0, 3.0, 5.0],
    "ilp_enabled": True,
    "ilp_instances": 5,
    "ilp_horizon_hours": 6,
    "ilp_budget_k": 10,
    "ilp_time_limit_s": 120.0,
    "ilp_max_pairs": 100000,
}


def find_run_folder(folder: str | Path) -> Path:
    """Return the folder that holds a run's files: the one given, or the single subfolder that does."""
    folder = Path(folder)
    if (folder / "arf_refined_events.csv").exists():
        return folder
    found = [path.parent for path in folder.rglob("arf_refined_events.csv")]
    if len(found) != 1:
        raise FileNotFoundError(f"{folder}: expected one run, found {len(found)}")
    return found[0]


def read_manifest(run_folder: str | Path) -> dict:
    """Return a run's reproducibility manifest."""
    return json.loads((Path(run_folder) / "arf_reproducibility_manifest.json").read_text())


def recorded_config(manifest: dict) -> dict:
    """Return every setting a manifest records.

    The releases write the settings under ``inputs`` → ``config`` (``record_inputs`` in Cell 9).
    """
    return (manifest.get("inputs") or {}).get("config") or {}


def run_settings(run_folder: str | Path) -> dict:
    """Return the settings recorded in a run's manifest, completed with the release defaults."""
    config = recorded_config(read_manifest(run_folder))
    return {name: config.get(name, default) for name, default in RELEASE_DEFAULTS.items()}


def orbit_period_min(run_folder: str | Path) -> float:
    """Return one orbital period of the run's fleet, in minutes, as Cell 2 recorded it."""
    summary = json.loads((Path(run_folder) / "arf_cell2_summary.json").read_text())
    return float(summary["t_max_wait_min"])


def read_csv_exact(source, **options) -> pd.DataFrame:
    """Read a CSV with every number exactly as written.

    Before pandas 3.1 the default parser returns about a fifth of values one unit in the last place away from the one
    written (measured on 200,000 values with pandas 3.0.5), and ``float_precision="round_trip"`` is needed; from 3.1
    the default is exact and that argument is deprecated.
    """
    major, minor = (int(part) for part in pd.__version__.split(".")[:2])
    if (major, minor) >= (3, 1):
        return pd.read_csv(source, **options)
    return pd.read_csv(source, float_precision="round_trip", **options)
