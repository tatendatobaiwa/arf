"""The cloud-checking tool on a run folder whose cloudy shares were computed by the registered release itself."""

import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

h3 = pytest.importorskip("h3")
from registered import run_cell6_events_and_cloud  # noqa: E402
from test_cloud_matches_registered_release import REGION, synthetic_events  # noqa: E402

TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_cloud_on_runs.py"
SEEDS = range(42, 62)  # the 20 realisations of a run: the run's seed and the 19 after it


@pytest.mark.parametrize("model", ["correlated", "iid"])
def test_tool_agrees_with_shares_from_the_registered_release(tmp_path, capsys, model):
    run = tmp_path / "ARF_Outputs_x" / "r"  # the tool must also find a run stored one folder down
    run.mkdir(parents=True)
    events = synthetic_events(seed=3)
    events.to_csv(run / "arf_refined_events.csv", index=False)
    pd.DataFrame({"tile": REGION}).to_csv(run / "arf_mosaic_tiles.csv", index=False)
    config = {"random_seed": 42, "cloud_model": model, "cloud_corr_length_km": 50.0, "cloud_corr_time_h": 3.0}
    (run / "arf_reproducibility_manifest.json").write_text(json.dumps({"inputs": {"config": config}}))
    registered = run_cell6_events_and_cloud("1.5.6", events, REGION, cloud_model=model)
    rows = []
    for seed in SEEDS:
        mask = registered["build_cloud_mask"](registered["event_data"], 0.35, seed)
        rows.append({"seed": seed, "cloudy_rows_pct": 100.0 * mask.n_cloudy / max(1, len(mask))})  # as _cloud_trial
    shares = pd.DataFrame(rows)
    shares.to_csv(run / "arf_cloud_robustness.csv", index=False)

    tool = runpy.run_path(str(TOOL), run_name="tool")
    assert tool["main"]([str(tmp_path / "ARF_Outputs_x")]) == 0
    assert "20 realisations" in capsys.readouterr().out
    # One realisation recorded with one more cloudy observation must be reported.
    shares.loc[7, "cloudy_rows_pct"] += 100.0 / len(mask)
    shares.to_csv(run / "arf_cloud_robustness.csv", index=False)
    assert tool["main"]([str(run)]) == 1
    assert "1 of 20 realisations" in capsys.readouterr().out


def test_tool_reports_a_run_without_cloud_realisations(tmp_path, capsys):
    events = synthetic_events(seed=3)
    events.to_csv(tmp_path / "arf_refined_events.csv", index=False)
    tool = runpy.run_path(str(TOOL), run_name="tool")
    assert tool["main"]([str(tmp_path)]) == 1
    assert "no arf_cloud_robustness.csv" in capsys.readouterr().out


@pytest.mark.parametrize("release", ["1.5.3", "1.5.6"])
def test_settings_are_read_where_the_releases_write_them(release, tmp_path):
    """The manifest of Cell 9 holds the settings under inputs -> config; a top-level "config" is not a setting."""
    from registered import RELEASES

    from arf.runs import run_settings

    source = RELEASES[release].read_text()
    assert '"inputs": record_inputs(),' in source
    record_inputs = source[source.index("def record_inputs()") : source.index("def validate_all_outputs")]
    assert '"config": config.to_dict()' in record_inputs
    manifest = {"inputs": {"config": {"cloud_model": "iid", "random_seed": 7}}, "config": {"random_seed": 99}}
    (tmp_path / "arf_reproducibility_manifest.json").write_text(json.dumps(manifest))
    settings = run_settings(tmp_path)
    assert (settings["cloud_model"], settings["random_seed"], settings["cloud_probability"]) == ("iid", 7, 0.35)
