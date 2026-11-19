"""The Fail-Signal checking tool on a run folder whose figures were computed by the registered release itself."""

import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

h3 = pytest.importorskip("h3")
from registered import run_cell6_selection_and_trial  # noqa: E402
from test_cloud_matches_registered_release import REGION, synthetic_events  # noqa: E402

TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_failsignal_on_runs.py"


def test_tool_agrees_with_figures_from_the_registered_release(tmp_path, capsys):
    run = tmp_path / "ARF_Outputs_x" / "r"
    run.mkdir(parents=True)
    events = synthetic_events(seed=5)
    events.to_csv(run / "arf_refined_events.csv", index=False)
    pd.DataFrame({"tile": REGION}).to_csv(run / "arf_mosaic_tiles.csv", index=False)
    config = {"random_seed": 42, "max_selection_steps": 12, "latency_decay_rate": 3.0}
    (run / "arf_reproducibility_manifest.json").write_text(json.dumps({"inputs": {"config": config}}))
    (run / "arf_cell2_summary.json").write_text(json.dumps({"t_max_wait_min": 95.123456789}))
    registered = run_cell6_selection_and_trial("1.5.3", events, REGION, budget=12, orbit_period_min=95.123456789)
    rows = []
    for seed in (42, 43, 44, 45):
        mask = registered["build_cloud_mask"](registered["event_data"], 0.35, seed)
        rows.append({"seed": seed, **registered["_cloud_trial"](mask, with_h1=False)})
    figures = pd.DataFrame(rows)
    figures.to_csv(run / "arf_cloud_robustness.csv", index=False)

    tool = runpy.run_path(str(TOOL), run_name="tool")
    assert tool["main"]([str(tmp_path / "ARF_Outputs_x")]) == 0
    assert "4 realisations, 120 figures" in capsys.readouterr().out  # 30 figures per realisation
    assert tool["main"]([str(run), "--seeds", "2"]) == 0
    assert "2 realisations, 60 figures" in capsys.readouterr().out
    # One image more recorded for one plan of one realisation must be reported, with its seed and name.
    figures.loc[2, "time_aware_openloop_expected_images"] += 1
    figures.to_csv(run / "arf_cloud_robustness.csv", index=False)
    assert tool["main"]([str(run)]) == 1
    out = capsys.readouterr().out
    assert "DOES NOT AGREE" in out and "time_aware_openloop_expected_images" in out and "(44," in out


def test_tool_refuses_settings_that_are_not_ported(tmp_path, capsys):
    events = synthetic_events(seed=5)
    events.to_csv(tmp_path / "arf_refined_events.csv", index=False)
    pd.DataFrame({"seed": [42]}).to_csv(tmp_path / "arf_cloud_robustness.csv", index=False)
    (tmp_path / "arf_reproducibility_manifest.json").write_text(
        json.dumps({"inputs": {"config": {"cloud_knowledge": "oracle"}}})
    )
    tool = runpy.run_path(str(TOOL), run_name="tool")
    assert tool["main"]([str(tmp_path)]) == 1
    assert "not ported yet" in capsys.readouterr().out
