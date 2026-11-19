"""The grid-checking tool on run folders whose grid files were produced by the registered release itself."""

import json
import runpy
from pathlib import Path

import pandas as pd
import pytest

h3 = pytest.importorskip("h3")
pytest.importorskip("shapefile")
pytest.importorskip("shapely")
from registered import run_cell1_grid  # noqa: E402
from test_grid_matches_registered_release import write_ckgr_okavango  # noqa: E402

TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_grid_on_runs.py"


def write_run(folder, registered, key, resolution=7):
    """Write the grid files of a run from the namespace of the registered Cell 1, as Cell 1 and Cell 7 write them."""
    folder.mkdir(parents=True)
    (folder / "arf_refined_events.csv").write_text("sat,t_min,tile\n")
    validation = {
        "roi_active": key,
        "roi_tiles": len(registered["target_tiles"]),
        "sample_tile": min(registered["target_tiles"]),
        "wdpa_boundaries": registered["roi_boundary_report"],
        "h3_resolution": resolution,
    }
    (folder / "arf_cell1_validation.json").write_text(json.dumps(validation, indent=2, default=str))
    pd.DataFrame({"tile": sorted(registered["target_tiles"])}).to_csv(folder / "arf_mosaic_tiles.csv", index=False)


def test_tool_agrees_with_grids_from_the_registered_release(tmp_path, capsys):
    boundary_dir = tmp_path / "ARF_boundaries"
    write_ckgr_okavango(boundary_dir)
    wdpa_run, corridor_run = tmp_path / "ARF_Outputs_a" / "r", tmp_path / "ARF_Outputs_b"
    write_run(wdpa_run, run_cell1_grid("1.5.6", "ckgr_okavango", 7, boundary_dir), "ckgr_okavango")
    write_run(corridor_run, run_cell1_grid("1.5.3", "nepal_trishuli_corridor", 8), "nepal_trishuli_corridor", 8)
    tool = runpy.run_path(str(TOOL), run_name="tool")
    arguments = ["--boundaries", str(boundary_dir), str(tmp_path / "ARF_Outputs_a"), str(corridor_run)]
    assert tool["main"](arguments) == 0
    printed = capsys.readouterr().out
    assert "2 site reports" in printed and "2 of 2 runs agree" in printed

    # A run whose cell list lacks one cell, and whose recorded area differs, must be reported.
    cells = pd.read_csv(wdpa_run / "arf_mosaic_tiles.csv")
    cells.iloc[1:].to_csv(wdpa_run / "arf_mosaic_tiles.csv", index=False)
    validation = json.loads((wdpa_run / "arf_cell1_validation.json").read_text())
    validation["wdpa_boundaries"][1]["wdpa_gis_area_km2"] *= 1.0001
    (wdpa_run / "arf_cell1_validation.json").write_text(json.dumps(validation))
    assert tool["main"](arguments) == 1
    printed = capsys.readouterr().out
    assert "1 cells only in release 2 and 0 only in the run" in printed
    assert "site 555577555: wdpa_gis_area_km2" in printed and "1 of 2 runs agree" in printed


def test_tool_reports_missing_boundaries_and_missing_files(tmp_path, capsys):
    boundary_dir = tmp_path / "ARF_boundaries"
    write_ckgr_okavango(boundary_dir)
    run = tmp_path / "run"
    write_run(run, run_cell1_grid("1.5.6", "ckgr_okavango", 7, boundary_dir), "ckgr_okavango")
    tool = runpy.run_path(str(TOOL), run_name="tool")
    assert tool["main"](["--boundaries", str(tmp_path / "elsewhere"), str(run)]) == 1
    assert "not found anywhere under" in capsys.readouterr().out
    assert tool["main"]([str(run)]) == 1  # no --boundaries for a WDPA region
    assert "folder that holds the downloads" in capsys.readouterr().out
    (run / "arf_cell1_validation.json").unlink()
    assert tool["main"](["--boundaries", str(boundary_dir), str(run)]) == 1
    assert "no arf_cell1_validation.json" in capsys.readouterr().out
