"""The run-checking tool agrees on a folder written by release 2 itself and reports a changed value."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from arf.selection import ImageSet, evaluate_onset, onset_grid

TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_selection_on_runs.py"


def write_run(folder):
    rng = np.random.default_rng(7)
    rows = []
    for image in range(120):
        acquired = int(rng.integers(0, 2880))
        start = int(rng.integers(0, 80))
        rows += [(f"S{image % 9}", acquired, f"cell{c}", 0.0, 20.0, 1.0, 30.0) for c in range(start, start + 20)]
    columns = ["sat", "t_min", "tile", "A_nadir_deg", "theta_max_deg", "D_downlink", "T_wait_min"]
    pd.DataFrame(rows, columns=columns).to_csv(folder / "arf_refined_events.csv", index=False)
    images = ImageSet.from_run_folder(folder)
    onsets = []
    for onset in onset_grid():
        result = evaluate_onset(images, onset)
        if result is not None:
            onsets.append(
                {
                    "onset_shift_min": onset,
                    "best_possible_clear_t95r_min": result.floor_min,
                    "earliest_completion_clear_t95r_min": result.completion[30].time_min,
                    "earliest_completion_clear_images95r": result.completion[30].images,
                    "fcfs_unbudgeted_clear_images95r": result.fcfs_images,
                    "reachable_clear_tiles": result.reachable,
                }
            )
    table = pd.DataFrame(onsets)
    table.to_csv(folder / "arf_onset_robustness.csv", index=False)
    return table


def test_tool_agrees_then_reports_a_changed_value(tmp_path, capsys):
    table = write_run(tmp_path)
    tool = runpy.run_path(str(TOOL), run_name="tool")
    assert tool["main"]([str(tmp_path)]) == 0
    assert "onsets agree" in capsys.readouterr().out
    row = table["fcfs_unbudgeted_clear_images95r"].first_valid_index()
    table.loc[row, "fcfs_unbudgeted_clear_images95r"] += 1
    table.to_csv(tmp_path / "arf_onset_robustness.csv", index=False)
    assert tool["main"]([str(tmp_path)]) == 1
    assert "DIFFERENCES" in capsys.readouterr().out
