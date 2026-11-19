"""The fleet-checking tool on run folders whose fleet records were produced by the registered release itself."""

import hashlib
import json
import runpy
import time
from pathlib import Path

from registered import orekit_stand_ins, run_cell2_fleet_functions
from test_fleet_matches_registered_release import ONSET, invented_omm_records, invented_tles, omm

from arf.fleet import merge_groups

TOOL = Path(__file__).resolve().parents[2] / "tools" / "check_fleet_on_runs.py"


def file_record(path):
    """A file as Cell 9 records it in the manifest."""
    content = Path(path).read_bytes()
    return {
        "path": str(path),
        "status": "FOUND",
        "size_bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def write_dense_run(run, cache, n, seed=42):
    """A dense-constellation run: the element sets, the pin and the counts as release 1.5.6 writes them."""
    registered = run_cell2_fleet_functions("1.5.6")
    run.mkdir(parents=True)
    cache.mkdir(exist_ok=True)
    (run / "arf_refined_events.csv").write_text("sat,t_min,tile\n")
    records = invented_tles(21)
    raw = cache / "arf_gp_history_starlink_20260803T000000_7d.json"
    raw.write_text(json.dumps({"fetched_at": 0, "records": records}))
    pin = cache / f"arf_selection_starlink_20260803T000000_N{n}_opsample_a5_seed{seed}.json"
    candidates, counts = registered["select_operational_sample"](records, ONSET, seed)
    selected = registered["select_pinned_satellites"](candidates, n, str(pin))
    summary = {
        "n_satellites": len(selected),
        "reference_epoch": ONSET,
        "analysis_epoch": ONSET,
        "fleet": "starlink",
        "satellite_selection": dict(method="operational at reference epoch, seeded random sample", **counts),
        "mission_table": None,
    }
    (run / "arf_cell2_summary.json").write_text(json.dumps(summary, indent=2))
    files = {
        "TLE_RAW_CACHE_PATH": file_record(raw),
        "SELECTION_CACHE_PATH": file_record(pin),
        "GP_INPUT_0": file_record(raw),
    }
    manifest = {"inputs": {"active_n": n, "fleet": "starlink", "config": {"random_seed": seed}, "files": files}}
    (run / "arf_reproducibility_manifest.json").write_text(json.dumps(manifest))
    return raw, pin


def write_real_run(run, cache, release, case="conservative", exclude=()):
    """A real-fleet run of a past window: group lists, history, fleet file and counts as the releases write them."""
    tag = f"earth-resources_planet-{case}_from20260803"
    registered = run_cell2_fleet_functions(release, case, fleet_tag=tag)
    run.mkdir(parents=True)
    cache.mkdir(exist_ok=True)
    (run / "arf_refined_events.csv").write_text("sat,t_min,tile\n")
    everything = invented_omm_records()
    groups = {"resource": [], "planet": []}
    for record in everything:
        groups[record.get("_GROUP") or "planet"].append({k: v for k, v in record.items() if k != "_GROUP"})
    groups["planet"].append(dict(groups["resource"][0]))  # a satellite listed in both groups is taken once
    paths = []
    for group, records in groups.items():
        path = cache / f"arf_gp_{group}.json"
        path.write_text(json.dumps({"fetched_at": time.time(), "url": "x", "records": records}))
        paths.append(path)
    live = registered["load_gp_json"](["resource", "planet"], str(cache))  # reads the fresh cache: no network
    assert live == merge_groups(groups)
    # Space-Track's own names differ from CelesTrak's; a history record always has a readable epoch.
    history = [
        dict(record, OBJECT_NAME=str(record["OBJECT_NAME"]).replace("-", " "))
        for record in everything
        if record["EPOCH"] != "yesterday"
    ]
    history += [
        omm(40697, "SENTINEL 2A", epoch="2026-07-30T00:00:00"),
        omm(40697, "SENTINEL 2A", epoch="2026-08-04T00:00:00"),
    ]
    history_path = cache / f"arf_gp_history_{tag}_20260803T000000.json"
    history_path.write_text(json.dumps({"fetched_at": 0, "epoch": ONSET, "records": history}))
    records = registered["load_eo_gp_history"](live, ONSET, str(cache))
    with orekit_stand_ins():
        constellation, counts = registered["build_eo_constellation"](
            records, ONSET, min_age_days=0.0, exclude_names=exclude
        )
    fleet_path = cache / f"arf_fleet_{tag}.json"
    satellites = [
        {
            "id": s["id"],
            "norad_id": s["norad_id"],
            "mission": s["mission"],
            "swath_km": round(s["swath_km"], 3),
            "fov_deg": s.get("fov_deg"),
            "min_sun_elev_deg": s["min_sun_elev_deg"],
        }
        for s in constellation
    ]
    fleet_path.write_text(
        json.dumps({"fleet": "earth-resources+planet", "reference_epoch": ONSET, "satellites": satellites}, indent=2)
    )
    summary = {
        "n_satellites": len(constellation),
        "reference_epoch": ONSET,
        "analysis_epoch": ONSET,
        "fleet": "earth-resources+planet",
        "satellite_selection": dict(method="all operational satellites with a sourced MISSION_TABLE entry", **counts),
        "mission_table": registered["MISSION_TABLE"],
    }
    (run / "arf_cell2_summary.json").write_text(json.dumps(summary, indent=2))
    files = {"TLE_RAW_CACHE_PATH": file_record(paths[0]), "SELECTION_CACHE_PATH": file_record(fleet_path)}
    files.update({f"GP_INPUT_{i}": file_record(path) for i, path in enumerate([*paths, history_path])})
    config = {"planet_swath_case": case, "assumed_min_sun_elev_deg": 5.0, "exclude_satellites": list(exclude)}
    manifest = {"inputs": {"fleet": "earth-resources+planet", "config": config, "files": files}}
    (run / "arf_reproducibility_manifest.json").write_text(json.dumps(manifest))
    return paths, history_path, fleet_path


def test_tool_agrees_with_fleets_from_the_registered_releases(tmp_path, capsys):
    cache = tmp_path / "shared_cache"
    write_dense_run(tmp_path / "dense_5" / "r", cache, 5)
    write_dense_run(tmp_path / "dense_100", cache, 100, seed=43)
    write_real_run(tmp_path / "real_153", cache, "1.5.3")
    write_real_run(tmp_path / "real_156_optimistic", tmp_path / "cache_2", "1.5.6", "optimistic")
    write_real_run(tmp_path / "real_without", tmp_path / "cache_3", "1.5.3", exclude=("SENTINEL-2A",))
    folders = [
        str(tmp_path / name) for name in ("dense_5", "dense_100", "real_153", "real_156_optimistic", "real_without")
    ]
    tool = runpy.run_path(str(TOOL), run_name="tool")
    assert tool["main"](folders) == 0
    printed = capsys.readouterr().out
    assert "5 of 5 runs agree" in printed and "N=100, seed 43" in printed and "changed since the run" not in printed
    assert "STARLINK" not in printed and "FLOCK" not in printed  # counts only, never an element set or a name


def test_tool_reports_differences_and_changed_or_missing_files(tmp_path, capsys):
    cache = tmp_path / "shared_cache"
    _, pin = write_dense_run(tmp_path / "dense", cache, 50)
    _, history_path, fleet_path = write_real_run(tmp_path / "real", tmp_path / "cache_2", "1.5.6")
    tool = runpy.run_path(str(TOOL), run_name="tool")
    dense, real = str(tmp_path / "dense"), str(tmp_path / "real")

    pinned = json.loads(pin.read_text())
    pinned["norad_ids"][3], pinned["norad_ids"][4] = pinned["norad_ids"][4], pinned["norad_ids"][3]
    pin.write_text(json.dumps(pinned))
    fleet = json.loads(fleet_path.read_text())
    fleet["satellites"][0]["swath_km"] += 0.001
    fleet_path.write_text(json.dumps(fleet))
    assert tool["main"]([dense, real]) == 1
    printed = capsys.readouterr().out
    assert "differs from the pin of 50 (48 equal in place)" in printed
    assert "1 of 11 satellites differ from the fleet file" in printed
    assert "changed since the run: arf_fleet_" in printed and "0 of 2 runs agree" in printed

    # A wrong count in the run's summary, and an element-set file that was moved: found again under --cache.
    summary = json.loads((tmp_path / "dense" / "arf_cell2_summary.json").read_text())
    summary["satellite_selection"]["low_perigee"] += 1
    (tmp_path / "dense" / "arf_cell2_summary.json").write_text(json.dumps(summary))
    moved = tmp_path / "elsewhere" / "deeper"
    moved.mkdir(parents=True)
    history_path.rename(moved / history_path.name)
    assert tool["main"]([dense, real]) == 1
    printed = capsys.readouterr().out
    assert "low_perigee (release 2" in printed and "is not at its recorded path nor under --cache" in printed
    assert tool["main"](["--cache", str(tmp_path / "elsewhere"), real]) == 1  # found; the fleet file still differs
    assert "satellites differ from the fleet file" in capsys.readouterr().out

    (tmp_path / "dense" / "arf_cell2_summary.json").unlink()
    assert tool["main"]([dense]) == 1
    assert "no arf_cell2_summary.json" in capsys.readouterr().out
