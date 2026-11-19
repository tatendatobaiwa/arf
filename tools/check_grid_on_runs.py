"""Check the grid of release 2 against the finished runs of the registered releases.

Every run records, in ``arf_cell1_validation.json``, the region it mapped, the H3 resolution, the number of cells and,
for a region read from WDPA boundaries, a report per site (number of cells, number of file parts, areas). Its cells
are listed in ``arf_mosaic_tiles.csv``. For every run folder given, release 2 discretises the same region at the same
resolution from the boundary files in ``--boundaries`` and compares: the number of cells, the cells themselves, and
the report of every site. Each distinct region and resolution is discretised once.

Read-only, and it prints counts only: no cell and no boundary is written anywhere (WDPA licence). The boundary files
must be those of the WDPA release the runs used; with a later release, a revised boundary shows here as a
difference. Needs ``h3`` and, for WDPA regions, ``pyshp``; a corridor region needs ``shapely``.

Usage:
    python tools/check_grid_on_runs.py --boundaries ARF_boundaries RUN_FOLDER [RUN_FOLDER ...]
    python tools/check_grid_on_runs.py --boundaries ARF_boundaries --index ARF_batch/paper2_v156_a18_core_index.csv

Exit status 0 if every run agrees, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from arf.errors import ConfigurationError, SpatialDiscretisationError
from arf.grid import RegionGrid, discretise_region, region_cells_from_run_folder
from arf.regions import REGIONS
from arf.runs import find_run_folder


def _report_differences(ours: dict, recorded: dict) -> list[str]:
    """Return the keys on which two site reports differ; numbers are compared to a relative 1e-12."""
    differing = []
    for key in sorted(set(ours) | set(recorded)):
        mine, theirs = ours.get(key), recorded.get(key)
        both_numbers = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (mine, theirs))
        same = math.isclose(mine, theirs, rel_tol=1e-12, abs_tol=0.0) if both_numbers else mine == theirs
        if not same:
            differing.append(f"{key} (release 2 {mine!r}, run {theirs!r})")
    return differing


def _cell_differences(grid: RegionGrid, run_folder: Path, recorded_count: int) -> list[str]:
    """Return how the cells of release 2 differ from the count and the cell list a run recorded."""
    problems = []
    if grid.cells.size != recorded_count:
        problems.append(f"{grid.cells.size} cells, the run has {recorded_count}")
    if not (run_folder / "arf_mosaic_tiles.csv").exists():
        return [*problems, "the run has no arf_mosaic_tiles.csv"]
    run_cells = region_cells_from_run_folder(run_folder)
    if not np.array_equal(grid.cells, run_cells):
        only_ours = np.setdiff1d(grid.cells, run_cells).size
        only_run = np.setdiff1d(run_cells, grid.cells).size
        problems.append(f"{only_ours} cells only in release 2 and {only_run} only in the run")
    return problems


def check(folder: Path, boundary_dir: Path | None, grids: dict[tuple[str, int], RegionGrid | str]) -> tuple[bool, str]:
    """Compare release 2 with the grid of one run; return whether they agree and a line to print.

    ``grids`` holds the regions already discretised, by key and resolution, or the reason one could not be.
    """
    run_folder = find_run_folder(folder)
    validation_file = run_folder / "arf_cell1_validation.json"
    if not validation_file.exists():
        return False, "the run has no arf_cell1_validation.json"
    recorded = json.loads(validation_file.read_text())
    key, resolution = recorded["roi_active"], int(recorded.get("h3_resolution", 7))
    if key not in REGIONS:
        return False, f"region {key!r} is not one of release 2's regions"
    if (key, resolution) not in grids:
        try:
            grids[key, resolution] = discretise_region(REGIONS[key], resolution, boundary_dir)
        except (ConfigurationError, SpatialDiscretisationError) as error:
            grids[key, resolution] = f"release 2 could not discretise {key!r}: {error}"
    grid = grids[key, resolution]
    if isinstance(grid, str):
        return False, grid
    problems = _cell_differences(grid, run_folder, int(recorded["roi_tiles"]))
    recorded_reports = recorded.get("wdpa_boundaries") or []
    our_reports = list(grid.site_reports or [])
    if len(our_reports) != len(recorded_reports):
        problems.append(f"{len(our_reports)} site reports, the run has {len(recorded_reports)}")
    for ours, theirs in zip(our_reports, recorded_reports, strict=False):
        differing = _report_differences(ours, theirs)
        if differing:
            problems.append(f"site {ours['site_id']}: " + "; ".join(differing))
    described = f"{key}, resolution {resolution}, {grid.cells.size} cells, {len(our_reports)} site reports"
    if problems:
        return False, f"{described}: " + " | ".join(problems)
    return True, described


def main(argv: list[str] | None = None) -> int:
    """Check every run folder named on the command line or listed as done in a batch index."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_folders", nargs="*", type=Path)
    parser.add_argument("--boundaries", type=Path, help="folder that holds the WDPA downloads")
    index_help = "batch index CSV; its done runs are checked"
    parser.add_argument("--index", type=Path, action="append", default=[], help=index_help)
    args = parser.parse_args(argv)
    folders = list(args.run_folders)
    for index in args.index:
        table = pd.read_csv(index)
        folders += [Path(p) for p in table.loc[table["status"] == "done", "out_dir"]]
    if not folders:
        parser.error("give at least one run folder or --index")
    grids: dict[tuple[str, int], RegionGrid | str] = {}
    failed = 0
    for number, folder in enumerate(folders, 1):
        agrees, detail = check(folder, args.boundaries, grids)
        failed += not agrees
        verdict = "agrees" if agrees else "DOES NOT AGREE"
        print(f"[{number}/{len(folders)}] {folder.name}: {verdict}: {detail}", flush=True)
    print(f"{len(folders) - failed} of {len(folders)} runs agree with the registered release")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
