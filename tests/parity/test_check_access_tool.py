"""The access-checking tool on run folders written by the registered release itself, with Orekit."""

import hashlib
import json
import re
import runpy
import time
from pathlib import Path

import pandas as pd
import pytest
from orekit_context import start_minimal_orekit
from registered import NetworkRefusedError, run_cell2_fleet_functions, run_cells_2_4_5
from test_access_matches_registered_release import (
    REGION_CELLS,
    invented_dense_constellation,
    invented_real_fleet,
    valid_tle,
    write_real_fleet_cache,
)
from test_check_fleet_tool import file_record

from arf.regions import REGIONS

TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_access_on_runs.py"
REFERENCE = "2026-08-03T00:00:00.000000"
TAG = "earth-resources_planet-conservative_from20260803"
STATIONS = {
    name: {"lat": s.latitude_deg, "lng": s.longitude_deg, "alt_m": s.altitude_m}
    for name, s in REGIONS["ckgr_okavango"].ground_stations.items()
}


@pytest.fixture(scope="module", autouse=True)
def orekit():
    start_minimal_orekit()


def finish_run_folder(run, config, files, active_n=None, fleet_tag=None):
    """Add what Cells 7 and 9 write and the tool reads: the region's cells and the manifest."""
    pd.DataFrame({"tile": REGION_CELLS}).to_csv(run / "arf_mosaic_tiles.csv", index=False)
    inputs = {"active_roi": "ckgr_okavango", "active_n": active_n, "fleet_tag": fleet_tag, "config": config}
    inputs["files"] = files
    (run / "arf_reproducibility_manifest.json").write_text(json.dumps({"framework_version": "1.5.6", "inputs": inputs}))


def write_real_run(run, cache):
    run.mkdir(parents=True)
    cache.mkdir()
    write_real_fleet_cache(cache, invented_real_fleet("2026-08-02T10:00:00.000000"), TAG, REFERENCE)
    run_cells_2_4_5("1.5.6", REGION_CELLS, STATIONS, REFERENCE, cache, fleet_tag=TAG, out_dir=run)
    paths = [
        cache / "arf_gp_resource.json",
        cache / "arf_gp_planet.json",
        cache / f"arf_gp_history_{TAG}_20260803T000000.json",
    ]
    files = {f"GP_INPUT_{i}": file_record(path) for i, path in enumerate(paths)}
    files["SELECTION_CACHE_PATH"] = file_record(cache / f"arf_fleet_{TAG}.json")
    config = {"window_hours": 48, "planet_swath_case": "conservative", "assumed_min_sun_elev_deg": 5.0}
    finish_run_folder(run, config, files, fleet_tag=TAG)


def write_dense_run(run, cache, n, records=None):
    run.mkdir(parents=True)
    cache.mkdir()
    records = records or invented_dense_constellation()
    raw = cache / "arf_gp_history_starlink_20260803T000000_7d.json"
    raw.write_text(json.dumps({"records": records}))
    registered = run_cells_2_4_5(
        "1.5.6",
        REGION_CELLS,
        STATIONS,
        REFERENCE,
        cache,
        fleet="starlink",
        raw_records=[tuple(record) for record in records],
        active_n=n,
        proxy_swath_km=32.5,
        proxy_min_sun_elev_deg=10.0,
        out_dir=run,
    )
    files = {
        "TLE_RAW_CACHE_PATH": file_record(raw),
        "SELECTION_CACHE_PATH": file_record(registered["SELECTION_CACHE_PATH"]),
    }
    config = {"window_hours": 48, "random_seed": 42, "proxy_swath_km": 32.5, "proxy_min_sun_elev_deg": 10.0}
    finish_run_folder(run, config, files, active_n=n, fleet_tag=f"N{n}_from20260803_sun10_sw32.5km")
    return registered


def fingerprints(*folders):
    """SHA-256 and modification time of every file under some folders: the tool must change none of them."""
    found = {}
    for folder in folders:
        for path in sorted(Path(folder).rglob("*")):
            if path.is_file():
                found[str(path)] = (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
    return found


def test_tool_agrees_with_runs_written_by_the_registered_release(tmp_path, capsys):
    real, dense, large = tmp_path / "real" / "r", tmp_path / "dense", tmp_path / "large"
    write_real_run(real, tmp_path / "cache_real")
    write_dense_run(dense, tmp_path / "cache_dense", 40)
    write_dense_run(large, tmp_path / "cache_large", 70)
    before = fingerprints(tmp_path)
    tool = runpy.run_path(str(TOOL), run_name="tool")
    assert tool["main"](["--max-satellites", "50", str(tmp_path / "real"), str(dense), str(large)]) == 0
    printed = capsys.readouterr().out
    assert printed.count("port vs registered 1.5.6: IDENTICAL") == 2
    assert printed.count("access identical, downlink identical, refined identical, summaries equal") == 2
    assert "skipped: 70 satellites" in printed and "2 of 2 runs checked agree" in printed and "1 skipped" in printed
    assert fingerprints(tmp_path) == before  # read-only: no file of the runs or the caches was changed or added
    assert "STARLINK" not in printed and "SENTINEL" not in printed  # no satellite name is printed ...
    assert not re.search(r"\b8[0-9a-f]{14}\b", printed)  # ... and no H3 cell identifier


def test_port_and_run_verdicts_are_separate(tmp_path, capsys, monkeypatch):
    real, dense = tmp_path / "real", tmp_path / "dense"
    write_real_run(real, tmp_path / "cache_real")
    write_dense_run(dense, tmp_path / "cache_dense", 30)
    tool = runpy.run_path(str(TOOL), run_name="tool")

    # The run's files changed, the port is still identical: the run is not reproduced, release 2 is not at fault.
    refined = pd.read_csv(dense / "arf_refined_events.csv")
    refined.iloc[1:].to_csv(dense / "arf_refined_events.csv", index=False)
    summary = json.loads((dense / "arf_cell2_summary.json").read_text())
    summary["t_max_wait_min"] += 0.01
    (dense / "arf_cell2_summary.json").write_text(json.dumps(summary))
    refined = pd.read_csv(real / "arf_refined_events.csv")
    refined["A_nadir_deg"] += 1e-9  # same rows, within the default tolerance
    refined.to_csv(real / "arf_refined_events.csv", index=False)
    assert tool["main"]([str(real), str(dense)]) == 1
    printed = capsys.readouterr().out
    assert printed.count("IDENTICAL") == 2
    assert "real: agrees" in printed and "refined same rows" in printed and "A_nadir_deg 1e-09" in printed
    assert "dense: RUN NOT REPRODUCED" in printed and "1 only in release 2, 0 only in the run" in printed
    assert "summaries DIFFER: t_max_wait_min" in printed and "run not reproduced in 1" in printed
    assert tool["main"](["--tolerance", "1e-12", str(real)]) == 1  # a tighter tolerance rejects the ninth decimal
    capsys.readouterr()

    # A fault in release 2 shows as a port difference, even one too small for the comparison with the run's files.
    import arf.access.refine as refine

    original = refine.solar_elevation_deg
    monkeypatch.setattr(refine, "solar_elevation_deg", lambda *a: original(*a) + 1e-12)
    assert tool["main"]([str(real)]) == 1
    printed = capsys.readouterr().out
    assert "real: PORT DIFFERS" in printed and "refined events" in printed and "port differs in 1" in printed


def test_element_sets_refused_by_orekit_are_left_out_as_in_cell_2(tmp_path, capsys):
    records = invented_dense_constellation(40)
    good = valid_tle(43999, 214.5, 10.0, 20.0)  # below the invented range, so no other satellite has its name
    refused = (good[0], good[1], good[2][:2] + "43998" + good[2][7:])  # the two lines name different objects
    registered = write_dense_run(tmp_path / "dense", tmp_path / "cache", 40, records=[*records, refused])
    assert "STARLINK-43999" not in [satellite["id"] for satellite in registered["constellation"]]
    tool = runpy.run_path(str(TOOL), run_name="tool")
    assert tool["main"]([str(tmp_path / "dense")]) == 0
    printed = capsys.readouterr().out
    assert "1 element sets refused by Orekit and left out" in printed and "IDENTICAL" in printed


def test_one_failing_run_does_not_stop_the_others(tmp_path, capsys):
    real, broken = tmp_path / "real", tmp_path / "broken"
    write_real_run(real, tmp_path / "cache_real")
    write_real_run(broken, tmp_path / "cache_broken")
    (tmp_path / "cache_broken" / f"arf_gp_history_{TAG}_20260803T000000.json").write_text("{not json")
    tool = runpy.run_path(str(TOOL), run_name="tool")
    assert tool["main"]([str(broken), str(real)]) == 1
    printed = capsys.readouterr().out
    assert "broken: ERROR: JSONDecodeError" in printed and "real: agrees" in printed and "errors in 1" in printed
    (real / "arf_cell4_summary.json").unlink()
    assert tool["main"]([str(real)]) == 1
    assert "the run lacks arf_cell4_summary.json" in capsys.readouterr().out


def test_registered_code_cannot_download(tmp_path):
    """A CelesTrak list older than six hours makes the registered code download; here it must refuse instead."""
    registered = run_cell2_fleet_functions("1.5.6")
    stale = {"fetched_at": time.time() - 7 * 3600, "url": "x", "records": []}
    (tmp_path / "arf_gp_resource.json").write_text(json.dumps(stale))
    with pytest.raises(NetworkRefusedError):
        registered["load_gp_json"](["resource"], str(tmp_path))
