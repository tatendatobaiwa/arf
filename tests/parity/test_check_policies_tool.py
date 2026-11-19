"""The policies-and-optimality checking tool on run folders whose files were written by the registered release."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

h3 = pytest.importorskip("h3")
from registered import run_cell7c_ilp, run_cell7c_policies  # noqa: E402
from test_cloud_matches_registered_release import REGION  # noqa: E402
from test_onsets_matches_registered_release import synthetic_window  # noqa: E402

from arf.runs import read_csv_exact  # noqa: E402

TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_policies_on_runs.py"
ORBIT_MIN, ALTITUDE_KM = 94.6, 515.0


def write_run(run, seed=21, cloud_model="correlated", **config):
    """A run folder: refined events, region cells, Cell 2 summary, Cell 7C's tables and a manifest with the ledger."""
    run.mkdir(parents=True)
    events = synthetic_window(seed)
    events.to_csv(run / "arf_refined_events.csv", index=False)
    pd.DataFrame({"tile": REGION}).to_csv(run / "arf_mosaic_tiles.csv", index=False)
    (run / "arf_cell2_summary.json").write_text(json.dumps({"median_alt_km": ALTITUDE_KM, "t_max_wait_min": ORBIT_MIN}))
    config = {"cloud_model": cloud_model, "random_seed": 42, "ilp_instances": 3, **config}
    common = {
        "orbit_period_min": ORBIT_MIN,
        "median_altitude_km": ALTITUDE_KM,
        "cloud_model": cloud_model,
        "budget": config.get("max_selection_steps", 30),
        "swath_half_angle_deg": 9.22,
    }
    policies = run_cell7c_policies("1.5.6", events, REGION, run, **common)
    ilp = run_cell7c_ilp("1.5.6", events, REGION, run, ilp_instances=config["ilp_instances"], **common)
    ledger = {
        "some_other_scenario__N5": {"policy_comparison": [{"coverage_pct": 1.0}], "ilp_gap": {}},
        "this_run__Nfleet": {"policy_comparison": policies["policy_rows"], "ilp_gap": ilp["ilp_summary"]},
    }
    manifest = {"inputs": {"config": config}, "scenario_results": ledger}
    (run / "arf_reproducibility_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def fingerprints(folder):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(folder).rglob("*")) if p.is_file()}


def tool():
    return runpy.run_path(str(TOOL), run_name="tool")


def test_tool_agrees_with_runs_written_by_the_registered_release(tmp_path, capsys):
    write_run(tmp_path / "correlated" / "r")
    write_run(tmp_path / "iid", seed=22, cloud_model="iid", max_selection_steps=8)
    before = fingerprints(tmp_path)
    assert tool()["main"]([str(tmp_path / "correlated"), str(tmp_path / "iid")]) == 0
    printed = capsys.readouterr().out
    assert printed.count("port vs registered 1.5.6: IDENTICAL | vs the run's files: identical") == 2
    assert "decay sweep of 6 rates, 12 policy rows" in printed and "ILP 3 of 3 proven optimal" in printed
    assert "the run's ILP solves took" in printed and "ILP summary not compared" not in printed
    assert "2 of 2 runs agree" in printed
    assert fingerprints(tmp_path) == before  # read-only
    assert tool()["main"](["--no-registered", "--no-ilp", str(tmp_path / "iid")]) == 0
    printed = capsys.readouterr().out
    assert "port vs" not in printed and "ILP not checked" in printed


def change_table(run, name, column, how):
    table = read_csv_exact(run / name)
    row = int(table[column].first_valid_index())
    table.loc[row, column] = how(table.loc[row, column])
    table.to_csv(run / name, index=False)


def test_rounding_and_wall_clock_are_told_apart_from_a_difference(tmp_path, capsys):
    names = ("last_bit", "seconds", "float", "count", "summary")
    for name in names:
        write_run(tmp_path / name)
    quality = "mean_quality_cloud"
    change_table(tmp_path / "last_bit", "arf_decay_sweep.csv", quality, lambda v: np.nextafter(v, np.inf))
    change_table(tmp_path / "seconds", "arf_ilp_gap.csv", "ilp_seconds", lambda v: v + 100.0)  # wall clock: ignored
    change_table(tmp_path / "float", "arf_cost_latency_frontier.csv", "t_complete_min", lambda v: v + 1e-6)
    change_table(tmp_path / "count", "arf_ilp_gap.csv", "ecc_greedy_images", lambda v: v + 1)
    manifest_file = tmp_path / "summary" / "arf_reproducibility_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["scenario_results"]["this_run__Nfleet"]["ilp_gap"]["optimal"] -= 1
    manifest_file.write_text(json.dumps(manifest))
    assert tool()["main"](["--no-registered", *(str(tmp_path / name) for name in names)]) == 1
    printed = capsys.readouterr().out
    assert "last_bit: agrees: " in printed and "identical up to rounding (1 floating-point values differ" in printed
    assert "seconds: agrees: " in printed
    assert "float: RUN NOT REPRODUCED" in printed and "t_complete_min (1 rows)" in printed
    assert "count: RUN NOT REPRODUCED" in printed and "ecc_greedy_images (1 rows)" in printed
    assert "summary: RUN NOT REPRODUCED" in printed and "ilp_gap.optimal" in printed
    assert "2 of 5 runs agree; RUN NOT REPRODUCED: 3" in printed


def test_a_fault_in_release_2_shows_as_a_port_difference(tmp_path, capsys, monkeypatch):
    write_run(tmp_path / "run")
    import arf.report.optimality as optimality

    original = optimality.facility_value
    monkeypatch.setattr(optimality, "facility_value", lambda *a: float(np.nextafter(original(*a), np.inf)))
    assert tool()["main"]([str(tmp_path / "run")]) == 1
    printed = capsys.readouterr().out
    assert "run: PORT DIFFERS" in printed and "time_aware_tw_objective" in printed


def test_tool_reports_what_it_cannot_check_and_goes_on(tmp_path, capsys):
    write_run(tmp_path / "good")
    write_run(tmp_path / "earthenv", seed=23)
    manifest_file = tmp_path / "earthenv" / "arf_reproducibility_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["inputs"]["config"]["cloud_probability_source"] = "earthenv"
    manifest_file.write_text(json.dumps(manifest))
    write_run(tmp_path / "no_ledger", seed=24)
    manifest_file = tmp_path / "no_ledger" / "arf_reproducibility_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest.pop("scenario_results")
    manifest_file.write_text(json.dumps(manifest))
    write_run(tmp_path / "no_ilp_file", seed=25)
    (tmp_path / "no_ilp_file" / "arf_ilp_gap.csv").unlink()
    write_run(tmp_path / "incomplete", seed=26)
    (tmp_path / "incomplete" / "arf_decay_sweep.csv").unlink()
    folders = [str(tmp_path / n) for n in ("earthenv", "no_ledger", "no_ilp_file", "incomplete", "good")]
    assert tool()["main"](["--no-registered", *folders]) == 1
    printed = capsys.readouterr().out
    assert "'earthenv' is not ported yet" in printed
    assert "no_ledger: agrees" in printed and "ILP summary not compared" in printed
    assert "enable the ILP but it lacks arf_ilp_gap.csv" in printed and "lacks arf_decay_sweep.csv" in printed
    assert "2 of 5 runs agree; CANNOT BE CHECKED: 3" in printed
