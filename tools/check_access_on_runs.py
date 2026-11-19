"""Check the access computation of release 2 against the finished runs of the registered releases.

For every run folder given, the run's fleet is rebuilt from the element-set files the run recorded, and the access
computation (the fleet's altitudes and orbital period, the access scan, the downlink contacts and the refined events)
is done again with Orekit, twice in the same session from the same inputs:

1. by release 2, and
2. by the registered release itself: its Cells 2 (fleet construction onwards), 4 and 5, executed unchanged from the
   release file in this repository (``tests/parity/registered.py``), on temporary copies of the run's inputs.

Two verdicts follow.

* **Port.** Release 2 against the registered cells: every number must be identical, to the last bit. This is what
  shows that release 2 computes what the registered release computes. ``--no-registered`` skips it, halving the time.
* **Run.** Release 2 against the files the run wrote months ago (``arf_access_events.csv``,
  ``arf_downlink_events.csv``, ``arf_refined_events.csv`` row by row; the summaries of Cells 2, 4 and 5). A table is
  *identical* when every value written is the same; *same rows* when it has the same rows (satellite, minute, cell or
  station) and its numbers differ by no more than ``--tolerance``, the largest difference stated; and *different*
  otherwise, which includes any row only one side has, whatever the tolerance. Identical tables need the Orekit data
  the run had: Earth orientation parameters are revised after the fact, and a revised value moves a ground track by
  centimetres, about a millionth of a degree in the angles. The default tolerance, 1e-4 (in degrees, about 10 m on the
  ground; the tables' other numbers are minutes, counts and a reward between 0 and 1), lies far above that and far
  below any physical difference. A run whose tables differ while the port is identical is not reproduced by either
  program today, which points at the inputs or the Orekit data, not at release 2.

Read-only: nothing is written to the run folders or the cache; the registered cells work on copies in a temporary
folder and cannot reach the network. Prints counts and differences only, never an element set or a cell. Needs ``h3``,
``orekit-jpype``, ``jdk4py`` and Orekit's data (``pip install git+https://gitlab.orekit.org/orekit/orekit-data.git``),
and must be run from a clone of the repository. Measured with both programs on 17 November 2026 (one processor; Colab
may be slower): 96 satellites over the CKGR, 23 s; a dense constellation of 250 satellites, 42 s; the time grows
about in proportion to the number of satellites, so ``--max-satellites`` skips larger fleets.

Usage:
    python tools/check_access_on_runs.py RUN_FOLDER [RUN_FOLDER ...]
    python tools/check_access_on_runs.py --index ARF_batch/paper2_v156_a18_core_index.csv --max-satellites 250

Exit status 0 if every run checked agrees on both verdicts, 1 otherwise.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from arf.access import RegionIndex, compute_access
from arf.fleet import CELESTRAK_GROUPS
from arf.grid import region_cells_from_run_folder
from arf.regions import REGIONS
from arf.replay import RecordedFleet, RecordedInputs, recorded_fleet
from arf.runs import find_run_folder, read_manifest, recorded_config

PARITY_HARNESS = Path(__file__).resolve().parents[1] / "tests" / "parity"

TABLES = {
    "access": ("arf_access_events.csv", ["sat", "t_min"]),
    "downlink": ("arf_downlink_events.csv", ["sat", "gs", "t_min"]),
    "refined": ("arf_refined_events.csv", ["sat", "t_min", "tile"]),
}
NEEDED = ("arf_cell2_summary.json", "arf_cell4_summary.json", "arf_cell5_summary.json", "arf_mosaic_tiles.csv")


@dataclass
class Verdict:
    """The outcome of one run: whether the port and the run agree (``None``: not checked) and what was found."""

    port: bool | None = None
    run: bool | None = None
    parts: list[str] = field(default_factory=list)
    skipped: str | None = None


def as_written(table: pd.DataFrame) -> pd.DataFrame:
    """Return a table as it reads back after being written to CSV, the form in which a run keeps its tables."""
    return pd.read_csv(io.StringIO(table.to_csv(index=False)))


def compare_tables(ours: pd.DataFrame, theirs: pd.DataFrame, keys: list[str], tolerance: float) -> tuple[str, str]:
    """Compare two tables; return ``"identical"``, ``"same rows"`` or ``"different"``, and a description."""
    if list(ours.columns) == list(theirs.columns) and len(ours) == len(theirs) and ours.equals(theirs):
        return "identical", f"{len(ours)} rows identical"
    if list(ours.columns) != list(theirs.columns):
        return "different", f"columns differ: {list(ours.columns)} and {list(theirs.columns)}"
    merged = ours.merge(theirs, on=keys, how="outer", suffixes=("_ours", "_run"), indicator=True)
    only_ours, only_run = int((merged["_merge"] == "left_only").sum()), int((merged["_merge"] == "right_only").sum())
    both = merged[merged["_merge"] == "both"]
    duplicated = len(both) != len(ours) - only_ours or len(both) != len(theirs) - only_run
    largest = {}
    for column in ours.columns:
        if column in keys:
            continue
        mine, run = both[f"{column}_ours"].to_numpy(dtype=float), both[f"{column}_run"].to_numpy(dtype=float)
        gap = np.where(np.isnan(mine) & np.isnan(run), 0.0, np.abs(mine - run))
        gap = np.where(np.isnan(gap), np.inf, gap)  # a value on one side only
        if gap.size and gap.max() > 0:
            largest[column] = float(gap.max())
    described = f"{len(both)} common rows, {only_ours} only in release 2, {only_run} only in the run"
    if duplicated:
        described += ", repeated keys"
    if largest:
        described += "; largest differences " + ", ".join(f"{name} {value:.3g}" for name, value in largest.items())
    if only_ours or only_run or duplicated or any(value > tolerance for value in largest.values()):
        return "different", described
    return "same rows", described


def _numbers_differ(ours, recorded, tolerance: float) -> bool:
    """Return whether two recorded values differ: numbers by more than a relative tolerance, others at all."""
    if isinstance(ours, (list, tuple)):
        return (
            not isinstance(recorded, (list, tuple))
            or len(ours) != len(recorded)
            or any(_numbers_differ(a, b, tolerance) for a, b in zip(ours, recorded, strict=False))
        )
    if isinstance(ours, float) or isinstance(recorded, float):
        try:
            return not math.isclose(float(ours), float(recorded), rel_tol=tolerance, abs_tol=tolerance)
        except (TypeError, ValueError):
            return True
    return ours != recorded


def _summary_differences(result, run_folder: Path, n_satellites: int, tolerance: float) -> list[str]:
    """Compare release 2's numbers with the summaries of Cells 2, 4 and 5 of a run."""
    cell2 = json.loads((run_folder / "arf_cell2_summary.json").read_text())
    cell4 = json.loads((run_folder / "arf_cell4_summary.json").read_text())
    cell5 = json.loads((run_folder / "arf_cell5_summary.json").read_text())
    settings, counts, events = result.settings, result.refinement.counts, result.refinement.events
    has_rows = len(events) > 0
    pairs = {
        "n_satellites": (n_satellites, cell2["n_satellites"]),
        "median_alt_km": (result.median_altitude_km, cell2["median_alt_km"]),
        "t_max_wait_min": (float(result.orbit_period_min), cell2["t_max_wait_min"]),
        "footprint_search_radius_km": (float(settings.footprint_radius_km), cell4["footprint_search_radius_km"]),
        "coarse_box_margin_km": (float(settings.coarse_margin_km), cell4["coarse_box_margin_km"]),
        "strip_margin_km": (float(settings.strip_margin_km), cell4["strip_margin_km"]),
        "altitude_range_km": (
            [float(settings.min_altitude_km), float(settings.max_altitude_km)],
            cell4["altitude_range_km"],
        ),
        "n_access_strips": (len(result.scan.events), cell4["n_access_strips"]),
        "n_access_events_input": (len(result.scan.events), cell5.get("n_access_events_input")),
        "n_refined_events": (len(events), cell5["n_refined_events"]),
        "n_unique_tiles": (int(events["tile"].nunique()) if has_rows else 0, cell5.get("n_unique_tiles")),
        "n_unique_sats": (int(events["sat"].nunique()) if has_rows else 0, cell5.get("n_unique_sats")),
        "a_nadir_mean": (float(events["A_nadir_deg"].mean()) if has_rows else 0.0, cell5.get("a_nadir_mean")),
        "d_downlink_mean": (float(events["D_downlink"].mean()) if has_rows else 0.0, cell5.get("d_downlink_mean")),
        "theta_max_deg_range": (
            [float(events["theta_max_deg"].min()), float(events["theta_max_deg"].max())] if has_rows else None,
            cell5.get("theta_max_deg_range"),
        ),
        "n_downlink_windows": (len(result.contacts), cell5["n_downlink_windows"]),
        "rows_end_cap_excluded": (counts["rows_end_cap_excluded"], cell5.get("rows_end_cap_excluded", 0)),
        "rows_below_sun_limit": (counts["rows_dark"], cell5.get("rows_below_sun_limit", 0)),
        "rows_acquisition_excluded": (counts["rows_acquisition_excluded"], cell5.get("rows_acquisition_excluded", 0)),
        "passes_acquisition_excluded": (
            counts["passes_acquisition_excluded"],
            cell5.get("passes_acquisition_excluded", 0),
        ),
        "acquisition_exclusions_hit": (
            [[hit["passes"], hit["rows"]] for _, hit in sorted(counts["acquisition_exclusions_hit"].items())],
            [[hit["passes"], hit["rows"]] for hit in cell5.get("acquisition_exclusions_hit", [])],
        ),
    }
    return [
        f"{name} (release 2 {ours!r}, run {recorded!r})"
        for name, (ours, recorded) in pairs.items()
        if _numbers_differ(ours, recorded, tolerance)
    ]


def _harness():
    """Return the parity harness, which executes the registered releases (``tests/parity/registered.py``)."""
    if str(PARITY_HARNESS) not in sys.path:
        sys.path.insert(0, str(PARITY_HARNESS))
    import registered

    return registered


def _release_of(manifest: dict) -> tuple[str, str | None]:
    """Return the registered release to execute for a run, and a note if it is not the run's own."""
    RELEASES = _harness().RELEASES  # noqa: N806 (the harness's name)

    version = str(manifest.get("framework_version") or "")
    if version in RELEASES:
        return version, None
    return "1.5.6", f"run made with release {version or 'unknown'}, not in this repository; 1.5.6 executed"


def _run_registered(run_folder: Path, fleet: RecordedFleet, cells: np.ndarray, stations: dict, release: str):
    """Execute the registered Cells 2, 4 and 5 on temporary copies of the run's inputs; return their namespace."""
    run_cells_2_4_5 = _harness().run_cells_2_4_5

    manifest = read_manifest(run_folder)
    config = recorded_config(manifest)
    summary = json.loads((run_folder / "arf_cell2_summary.json").read_text())
    inputs = RecordedInputs(manifest)
    options = {
        "window_hours": int(config.get("window_hours", 48)),
        "swath_half_angle_deg": float(config.get("swath_half_angle_deg", 9.22)),
        "min_elevation_deg": float(config.get("downlink_min_elevation_deg", 15.0)),
        "legacy_end_caps": bool(config.get("legacy_strip_end_caps", False)),
    }
    notes, pin_name = [], None
    with tempfile.TemporaryDirectory(prefix="arf_registered_") as work:
        work = Path(work)
        if fleet.kind == "dense":
            records = [tuple(record) for record in inputs.read(key="TLE_RAW_CACHE_PATH")["records"]]
            try:
                pin = inputs.path(key="SELECTION_CACHE_PATH")
            except FileNotFoundError:
                pass
            else:
                pin_name = pin.name
                shutil.copyfile(pin, work / pin_name)
            sun, swath = config.get("proxy_min_sun_elev_deg"), config.get("proxy_swath_km")
            options.update(
                fleet="starlink",
                raw_records=records,
                active_n=int(manifest["inputs"]["active_n"]),
                seed=int(config.get("random_seed", 42)),
                proxy_swath_km=None if swath is None else float(swath),
                proxy_min_sun_elev_deg=None if sun is None else float(sun),
            )
        else:
            for group in CELESTRAK_GROUPS[summary["fleet"]]:
                source = inputs.path(name_starts=f"arf_gp_{group}.json")
                content = json.loads(source.read_bytes())
                content["fetched_at"] = time.time()  # the copy is fresh, so the registered code reads it
                (work / source.name).write_text(json.dumps(content))
            history = inputs.path(name_starts="arf_gp_history_")
            shutil.copyfile(history, work / history.name)
            options.update(
                fleet=summary["fleet"],
                fleet_tag=manifest["inputs"]["fleet_tag"],
                planet_swath_case=config.get("planet_swath_case", "conservative"),
                assumed_min_sun_elev_deg=float(config.get("assumed_min_sun_elev_deg", 5.0)),
                exclude_satellites=tuple(str(name).strip().upper() for name in config.get("exclude_satellites") or ()),
            )
        namespace = run_cells_2_4_5(release, cells, stations, summary["analysis_epoch"], work, **options)
        if pin_name and Path(namespace["SELECTION_CACHE_PATH"]).name != pin_name:
            notes.append("the registered release looked for its pin under another name and drew its own sample")
    return namespace, notes


def _release2(fleet: RecordedFleet, cells: np.ndarray, stations, config: dict):
    """Compute the run's access with release 2; return the result and the names of satellites Orekit refused."""
    from arf.access.orekit import OrekitEphemeris, tle_from_omm, usable_tles

    if fleet.kind == "dense":
        tles, skipped = usable_tles([fleet.elements[imager.id] for imager in fleet.imagers])
        imagers = [imager for imager in fleet.imagers if imager.id in tles]
    else:
        tles, skipped = {imager.id: tle_from_omm(fleet.elements[imager.id]) for imager in fleet.imagers}, []
        imagers = fleet.imagers
    result = compute_access(
        imagers,
        OrekitEphemeris(tles, fleet.start_utc_iso, stations),
        RegionIndex.from_cells(cells),
        stations,
        window_hours=int(config.get("window_hours", 48)),
        swath_half_angle_deg=float(config.get("swath_half_angle_deg", 9.22)),
        min_elevation_deg=float(config.get("downlink_min_elevation_deg", 15.0)),
        exclusions=fleet.exclusions,
        legacy_end_caps=bool(config.get("legacy_strip_end_caps", False)),
    )
    return result, imagers, skipped


def check(folder: Path, cache_dir: Path | None, max_satellites: int | None, tolerance: float, registered: bool):
    """Check one run; return its :class:`Verdict`."""
    from arf.access.orekit import check_omm_with_orekit

    verdict = Verdict()
    run_folder = find_run_folder(folder)
    missing = [name for name in (*NEEDED, *(file for file, _ in TABLES.values())) if not (run_folder / name).exists()]
    if missing:
        verdict.run = False
        verdict.parts.append(f"the run lacks {', '.join(missing)}")
        return verdict
    manifest = read_manifest(run_folder)
    config = recorded_config(manifest)
    region_key = manifest["inputs"]["active_roi"]
    if region_key not in REGIONS:
        verdict.run = False
        verdict.parts.append(f"region {region_key!r} is not one of release 2's regions")
        return verdict
    try:
        fleet = recorded_fleet(run_folder, cache_dir, check_elements=check_omm_with_orekit)
    except FileNotFoundError as error:
        verdict.run = False
        verdict.parts.append(f"cannot be checked: {error}")
        return verdict
    if max_satellites is not None and len(fleet.imagers) > max_satellites:
        verdict.skipped = f"{len(fleet.imagers)} satellites, more than --max-satellites {max_satellites}"
        return verdict
    cells = region_cells_from_run_folder(run_folder)
    stations = REGIONS[region_key].ground_stations
    started = time.time()
    result, imagers, refused = _release2(fleet, cells, stations, config)
    verdict.parts.append(f"{len(imagers)} satellites")
    if refused:
        verdict.parts.append(f"{len(refused)} element sets refused by Orekit and left out, as in Cell 2")

    if registered and not json.loads((run_folder / "arf_cell2_summary.json").read_text()).get("analysis_epoch"):
        verdict.parts.append("port not checked: the registered cells are executed for runs of a past window only")
    elif registered:
        release, note = _release_of(manifest)
        station_table = {
            name: {"lat": s.latitude_deg, "lng": s.longitude_deg, "alt_m": s.altitude_m} for name, s in stations.items()
        }
        namespace, notes = _run_registered(run_folder, fleet, cells, station_table, release)
        differences = _harness().compare_with_registered(result, namespace)
        theirs = [satellite["id"] for satellite in namespace["constellation"]]
        if theirs != [imager.id for imager in imagers]:
            differences.insert(0, f"fleets differ: {len(imagers)} satellites in release 2, {len(theirs)} registered")
        verdict.port = not differences
        label = f"port vs registered {release}: " + (
            "IDENTICAL" if verdict.port else "DIFFERS: " + " | ".join(differences[:4])
        )
        verdict.parts.append(label + "".join(f" [{text}]" for text in [note, *notes] if text))

    run_parts, run_agrees = [], True
    ours = {"access": result.scan.events, "downlink": result.contacts, "refined": result.refinement.events}
    for name, (file_name, keys) in TABLES.items():
        table_verdict, described = compare_tables(
            as_written(ours[name]), pd.read_csv(run_folder / file_name), keys, tolerance
        )
        run_agrees &= table_verdict != "different"
        run_parts.append(f"{name} {table_verdict.upper() if table_verdict == 'different' else table_verdict}")
        if table_verdict != "identical":
            run_parts[-1] += f" ({described})"
    differing = _summary_differences(result, run_folder, len(imagers), tolerance)
    run_agrees &= not differing
    run_parts.append("summaries equal" if not differing else "summaries DIFFER: " + "; ".join(differing))
    verdict.run = run_agrees
    verdict.parts.append("vs the run's files: " + ", ".join(run_parts))
    verdict.parts.append(f"{time.time() - started:.0f} s")
    verdict.parts.extend(f"[{text}]" for text in fleet.notes)
    if fleet.changed_inputs:
        verdict.parts.append(f"[changed since the run: {', '.join(fleet.changed_inputs)}]")
    return verdict


def main(argv: list[str] | None = None) -> int:
    """Check every run folder named on the command line or listed as done in a batch index."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_folders", nargs="*", type=Path)
    parser.add_argument("--cache", type=Path, help="folder searched for an element-set file not at its recorded path")
    parser.add_argument("--max-satellites", type=int, help="skip runs whose fleet is larger")
    tolerance_help = "largest difference of a number for tables with the same rows to count as agreeing"
    parser.add_argument("--tolerance", type=float, default=1e-4, help=tolerance_help)
    no_registered_help = "skip the exact comparison with the registered cells"
    parser.add_argument("--no-registered", action="store_true", help=no_registered_help)
    parser.add_argument("--orekit-data", type=Path, help="Orekit data folder or zip; default: the orekitdata package")
    index_help = "batch index CSV; its done runs are checked"
    parser.add_argument("--index", type=Path, action="append", default=[], help=index_help)
    args = parser.parse_args(argv)
    folders = list(args.run_folders)
    for index in args.index:
        table = pd.read_csv(index)
        folders += [Path(p) for p in table.loc[table["status"] == "done", "out_dir"]]
    if not folders:
        parser.error("give at least one run folder or --index")
    from arf.access.orekit import start_orekit

    start_orekit(args.orekit_data)
    tally = {"port differs": 0, "run not reproduced": 0, "errors": 0, "skipped": 0, "agree": 0}
    for number, folder in enumerate(folders, 1):
        try:
            verdict = check(folder, args.cache, args.max_satellites, args.tolerance, not args.no_registered)
        except Exception as error:  # one run's failure must not stop the others
            tally["errors"] += 1
            where = traceback.extract_tb(error.__traceback__)[-1]
            print(
                f"[{number}/{len(folders)}] {folder.name}: ERROR: {type(error).__name__}: {error} "
                f"(at {Path(where.filename).name}:{where.lineno})",
                flush=True,
            )
            continue
        if verdict.skipped:
            tally["skipped"] += 1
            print(f"[{number}/{len(folders)}] {folder.name}: skipped: {verdict.skipped}", flush=True)
            continue
        tally["port differs"] += verdict.port is False
        tally["run not reproduced"] += verdict.run is False
        agrees = verdict.port is not False and verdict.run is not False
        tally["agree"] += agrees
        label = "agrees" if agrees else ("PORT DIFFERS" if verdict.port is False else "RUN NOT REPRODUCED")
        print(f"[{number}/{len(folders)}] {folder.name}: {label}: " + " | ".join(verdict.parts), flush=True)
    checked = len(folders) - tally["skipped"]
    print(
        f"{tally['agree']} of {checked} runs checked agree; port differs in {tally['port differs']}, run not "
        f"reproduced in {tally['run not reproduced']}, errors in {tally['errors']}; {tally['skipped']} skipped"
    )
    return 0 if tally["agree"] == checked else 1


if __name__ == "__main__":
    sys.exit(main())
