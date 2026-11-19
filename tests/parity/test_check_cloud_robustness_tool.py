"""The cloud-robustness checking tool on run folders whose files were written by the registered release itself."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

h3 = pytest.importorskip("h3")
from registered import run_cell7c_cloud  # noqa: E402
from test_cloud_matches_registered_release import REGION  # noqa: E402
from test_onsets_matches_registered_release import synthetic_window  # noqa: E402

from arf.runs import read_csv_exact  # noqa: E402

TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_cloud_robustness_on_runs.py"
ORBIT_MIN, ALTITUDE_KM = 94.6, 515.0
SHORT = {"cloud_seeds": 3, "cloud_sweep_lengths_km": [0.0, 50.0], "cloud_sweep_times_h": [1.0], "cloud_sweep_seeds": 2}


def write_run(run, seed=21, cloud_model="correlated", **config):
    """A run folder: refined events, region cells, Cell 2 summary, manifest, and Cell 7C's three cloud files."""
    run.mkdir(parents=True)
    events = synthetic_window(seed)
    events.to_csv(run / "arf_refined_events.csv", index=False)
    pd.DataFrame({"tile": REGION}).to_csv(run / "arf_mosaic_tiles.csv", index=False)
    (run / "arf_cell2_summary.json").write_text(json.dumps({"median_alt_km": ALTITUDE_KM, "t_max_wait_min": ORBIT_MIN}))
    config = {**SHORT, "cloud_model": cloud_model, "random_seed": 42, **config}
    run_cell7c_cloud(
        "1.5.6",
        events,
        REGION,
        run,
        cloud_seeds=config["cloud_seeds"],
        sweep_lengths_km=tuple(config["cloud_sweep_lengths_km"]),
        sweep_times_h=tuple(config["cloud_sweep_times_h"]),
        sweep_seeds=config["cloud_sweep_seeds"],
        budget=config.get("max_selection_steps", 30),
        orbit_period_min=ORBIT_MIN,
        median_altitude_km=ALTITUDE_KM,
        cloud_model=cloud_model,
        swath_half_angle_deg=9.22,
    )
    (run / "arf_reproducibility_manifest.json").write_text(json.dumps({"inputs": {"config": config}}))


def fingerprints(folder):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(folder).rglob("*")) if p.is_file()}


def tool():
    return runpy.run_path(str(TOOL), run_name="tool")


def test_tool_agrees_with_runs_written_by_the_registered_release(tmp_path, capsys):
    write_run(tmp_path / "correlated" / "r")
    write_run(tmp_path / "iid", seed=22, cloud_model="iid", max_selection_steps=8, cloud_sweep_lengths_km=[])
    before = fingerprints(tmp_path)
    assert tool()["main"]([str(tmp_path / "correlated"), str(tmp_path / "iid")]) == 0
    printed = capsys.readouterr().out
    assert printed.count("port vs registered 1.5.6: IDENTICAL | vs the run's files: identical") == 2
    assert "3 realisations x " in printed and "sweep of 3 settings" in printed and "H2 (time-aware) " in printed
    assert "2 of 2 runs agree" in printed and "numpy not recorded at the run" in printed
    assert fingerprints(tmp_path) == before  # read-only
    assert tool()["main"](["--no-registered", "--no-sensitivity", str(tmp_path / "correlated")]) == 0
    printed = capsys.readouterr().out
    assert "port vs" not in printed and ", no sweep;" in printed and "identical" in printed


def change_table(run, name, column, how):
    """Change one value of one of a run's tables."""
    table = read_csv_exact(run / name)
    row = int(table[column].first_valid_index())
    table.loc[row, column] = how(table.loc[row, column])
    table.to_csv(run / name, index=False)


def test_rounding_is_told_apart_from_a_difference(tmp_path, capsys):
    names = ("last_bit", "float", "count", "sensitivity", "summary")
    for name in names:
        write_run(tmp_path / name)
    table, sweep = "arf_cloud_robustness.csv", "arf_cloud_sensitivity.csv"
    coverage = "time_aware_adaptive_cov_pct"
    change_table(tmp_path / "last_bit", table, coverage, lambda value: np.nextafter(value, np.inf))
    change_table(tmp_path / "float", table, coverage, lambda value: value * (1 + 1e-9))
    change_table(tmp_path / "count", table, "time_aware_replans", lambda value: value + 1)
    change_table(tmp_path / "sensitivity", sweep, "adaptive_cov_pct", lambda value: value + 0.5)
    summary_file = tmp_path / "summary" / "arf_cloud_robustness_summary.json"
    summary = json.loads(summary_file.read_text())
    summary["h2"]["time_aware"]["status"] = "SUPPORTED" if summary["h2"]["time_aware"]["status"] != "SUPPORTED" else "?"
    summary_file.write_text(json.dumps(summary))
    assert tool()["main"](["--no-registered", *(str(tmp_path / name) for name in names)]) == 1
    printed = capsys.readouterr().out
    assert "last_bit: agrees: " in printed and "identical up to rounding (1 floating-point values differ" in printed
    assert "float: RUN NOT REPRODUCED: " in printed and f"DIFFERENT in 1: {coverage} (1 realisations)" in printed
    assert "count: RUN NOT REPRODUCED" in printed and "time_aware_replans (1 realisations)" in printed
    assert "sensitivity: RUN NOT REPRODUCED" in printed and "adaptive_cov_pct (1 settings)" in printed
    assert "summary: RUN NOT REPRODUCED" in printed and "h2.time_aware.status" in printed
    assert "1 of 5 runs agree; RUN NOT REPRODUCED: 4" in printed


def test_a_fault_in_release_2_shows_as_a_port_difference(tmp_path, capsys, monkeypatch):
    write_run(tmp_path / "run")
    import arf.report.cloud_robustness as cloud_robustness

    original = cloud_robustness.paired_summary

    def nudged(differences, alpha=0.05):
        summary = original(differences, alpha)
        if summary["mean"] is not None:
            summary["mean"] = float(np.nextafter(summary["mean"], np.inf))
        return summary

    monkeypatch.setattr(cloud_robustness, "paired_summary", nudged)
    assert tool()["main"]([str(tmp_path / "run")]) == 1
    printed = capsys.readouterr().out
    assert "run: PORT DIFFERS" in printed and "difference_vs_openloop_expected.mean" in printed
    assert "identical up to rounding" in printed  # one unit in the last place: within rounding of the run's files


def test_tool_reports_what_it_cannot_check_and_goes_on(tmp_path, capsys):
    write_run(tmp_path / "good")
    for name, setting in (
        ("earthenv", {"cloud_probability_source": "earthenv"}),
        ("oracle", {"cloud_knowledge": "oracle"}),
    ):
        write_run(tmp_path / name, seed=23)
        manifest_file = tmp_path / name / "arf_reproducibility_manifest.json"
        manifest = json.loads(manifest_file.read_text())
        manifest["inputs"]["config"].update(setting)
        manifest_file.write_text(json.dumps(manifest))
    write_run(tmp_path / "broken", seed=25)
    (tmp_path / "broken" / "arf_refined_events.csv").write_text("sat,t_min\nnot,a number\n")
    write_run(tmp_path / "incomplete", seed=26)
    (tmp_path / "incomplete" / "arf_cloud_robustness_summary.json").unlink()
    folders = [str(tmp_path / name) for name in ("broken", "earthenv", "oracle", "incomplete", "good")]
    assert tool()["main"](folders) == 1
    printed = capsys.readouterr().out
    assert "broken: ERROR: ValueError" in printed
    assert "'earthenv' is not ported yet" in printed and "cloud_knowledge 'oracle' is not ported" in printed
    assert "lacks arf_cloud_robustness_summary.json" in printed and "good: agrees" in printed
    assert "1 of 5 runs agree; CANNOT BE CHECKED: 3, ERROR: 1" in printed
