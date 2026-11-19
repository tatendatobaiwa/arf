"""The onset-checking tool on run folders whose files were written by the registered release itself."""

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

h3 = pytest.importorskip("h3")
from registered import run_cell7c_onsets  # noqa: E402
from test_cloud_matches_registered_release import REGION  # noqa: E402
from test_onsets_matches_registered_release import synthetic_window  # noqa: E402

from arf.runs import read_csv_exact  # noqa: E402

TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_onsets_on_runs.py"
ORBIT_MIN, ALTITUDE_KM = 94.6, 515.0


def write_run(run, seed=21, cloud_model="correlated", **options):
    """A run folder: refined events, region cells, Cell 2 summary, manifest, and Cell 7C's two files."""
    run.mkdir(parents=True)
    events = synthetic_window(seed)
    events.to_csv(run / "arf_refined_events.csv", index=False)
    pd.DataFrame({"tile": REGION}).to_csv(run / "arf_mosaic_tiles.csv", index=False)
    (run / "arf_cell2_summary.json").write_text(json.dumps({"median_alt_km": ALTITUDE_KM, "t_max_wait_min": ORBIT_MIN}))
    run_cell7c_onsets(
        "1.5.6",
        events,
        REGION,
        run,
        orbit_period_min=ORBIT_MIN,
        median_altitude_km=ALTITUDE_KM,
        cloud_model=cloud_model,
        shift_step_min=120,
        **options,
    )
    config = {"cloud_model": cloud_model, "onset_shift_step_min": 120, "random_seed": 42}
    config.update({"max_selection_steps": options["budget"]} if "budget" in options else {})
    (run / "arf_reproducibility_manifest.json").write_text(json.dumps({"inputs": {"config": config}}))


def fingerprints(folder):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(folder).rglob("*")) if p.is_file()}


def test_tool_agrees_with_runs_written_by_the_registered_release(tmp_path, capsys):
    write_run(tmp_path / "correlated" / "r")
    write_run(tmp_path / "iid", seed=22, cloud_model="iid", budget=8)
    before = fingerprints(tmp_path)
    tool = runpy.run_path(str(TOOL), run_name="tool")
    assert tool["main"]([str(tmp_path / "correlated"), str(tmp_path / "iid")]) == 0
    printed = capsys.readouterr().out
    assert printed.count("port vs registered 1.5.6: IDENTICAL | vs the run's files: identical") == 2
    assert "2 of 2 runs agree" in printed and "13 onsets x" in printed and "H1a (ECC, clear)" in printed
    assert "numpy not recorded at the run" in printed
    assert fingerprints(tmp_path) == before  # read-only
    assert tool["main"](["--no-registered", str(tmp_path / "iid")]) == 0
    assert "port vs" not in capsys.readouterr().out


def change(run, column, how):
    """Change one value of a run's onset table."""
    table = read_csv_exact(run / "arf_onset_robustness.csv")
    row = int(table[column].first_valid_index())
    table.loc[row, column] = how(table.loc[row, column])
    table.to_csv(run / "arf_onset_robustness.csv", index=False)


def test_rounding_is_told_apart_from_a_difference(tmp_path, capsys):
    for name in ("last_bit", "float", "count", "summary"):
        write_run(tmp_path / name)
    quality = "earliest_completion_clear_quality95r"
    change(tmp_path / "last_bit", quality, lambda value: np.nextafter(value, np.inf))  # one unit in the last place
    change(tmp_path / "float", quality, lambda value: value * (1 + 1e-9))
    change(tmp_path / "count", "earliest_completion_clear_images95r", lambda value: value + 1)
    summary_file = tmp_path / "summary" / "arf_onset_robustness_summary.json"
    summary = json.loads(summary_file.read_text())
    summary["h1b_image_economy"]["series"]["clear"]["n_success"] += 1
    summary_file.write_text(json.dumps(summary))
    tool = runpy.run_path(str(TOOL), run_name="tool")
    folders = [str(tmp_path / name) for name in ("last_bit", "float", "count", "summary")]
    assert tool["main"](["--no-registered", *folders]) == 1
    printed = capsys.readouterr().out
    assert "last_bit: agrees: " in printed and "identical up to rounding (1 floating-point values differ" in printed
    assert "float: RUN NOT REPRODUCED: " in printed and f"DIFFERENT in 1: {quality} (1 onsets)" in printed
    assert "count: RUN NOT REPRODUCED" in printed and "earliest_completion_clear_images95r (1 onsets)" in printed
    assert "summary: RUN NOT REPRODUCED" in printed and "h1b_image_economy.series.clear.n_success" in printed
    assert "1 of 4 runs agree; RUN NOT REPRODUCED: 3" in printed


def test_a_fault_in_release_2_shows_as_a_port_difference(tmp_path, capsys, monkeypatch):
    write_run(tmp_path / "run")
    import arf.report.onsets as onsets

    original = onsets.mosaic_quality

    def nudged(*arguments):
        value = original(*arguments)
        return None if value is None else np.nextafter(value, np.inf)

    monkeypatch.setattr(onsets, "mosaic_quality", nudged)
    tool = runpy.run_path(str(TOOL), run_name="tool")
    assert tool["main"]([str(tmp_path / "run")]) == 1
    printed = capsys.readouterr().out
    assert "run: PORT DIFFERS" in printed and "earliest_completion_clear_quality95r" in printed
    assert "identical up to rounding" in printed  # one unit in the last place: within rounding of the run's files


def test_tool_reports_what_it_cannot_check_and_goes_on(tmp_path, capsys):
    write_run(tmp_path / "good")
    write_run(tmp_path / "earthenv", seed=23)
    manifest = json.loads((tmp_path / "earthenv" / "arf_reproducibility_manifest.json").read_text())
    manifest["inputs"]["config"]["cloud_probability_source"] = "earthenv"
    (tmp_path / "earthenv" / "arf_reproducibility_manifest.json").write_text(json.dumps(manifest))
    write_run(tmp_path / "disabled", seed=24)
    manifest["inputs"]["config"] = {"onset_robustness_enabled": False}
    (tmp_path / "disabled" / "arf_reproducibility_manifest.json").write_text(json.dumps(manifest))
    write_run(tmp_path / "broken", seed=25)
    (tmp_path / "broken" / "arf_refined_events.csv").write_text("sat,t_min\nnot,a number\n")
    write_run(tmp_path / "incomplete", seed=26)
    (tmp_path / "incomplete" / "arf_onset_robustness_summary.json").unlink()
    folders = [str(tmp_path / name) for name in ("broken", "earthenv", "disabled", "incomplete", "good")]
    tool = runpy.run_path(str(TOOL), run_name="tool")
    assert tool["main"](folders) == 1
    printed = capsys.readouterr().out
    assert "broken: ERROR: ValueError" in printed
    assert "'earthenv' is not ported yet" in printed and "did not compute the onset comparison" in printed
    assert "lacks arf_onset_robustness_summary.json" in printed and "good: agrees" in printed
    assert "2 of 5 runs agree; CANNOT BE CHECKED: 2, ERROR: 1" in printed
