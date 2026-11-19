"""Repository checks that need no network, no Orekit and no licensed data.

Run from the repository root:  python -m pytest -q tests
  1. The release of record is the file the protocol was frozen against (SHA-256).
  2. The protocol carries amendments A1 to A19, in order.
  3. Every Python file in the repository compiles.
  4. The manuscript tables regenerate from results/ and equal the reference copies.
  5. The result figures and the A12/A13 figure regenerate (each script asserts its
     inputs against the stored summaries before drawing).
  6. No file holds a Space-Track element set or a WDPA/WDPCA boundary file.
  7. Release 1.5.6 is the file amendments A18 and A19 pin, and decodes Alpha-5 catalogue numbers
     (100000 and above), which release 1.5.5 rejected.
  8. The stored dense-constellation results (A18, A19) are complete and consistent: 90 runs of release
     1.5.6, 90 replays with the identity check passed, and the restricted fit recomputed from the run table.
"""
import hashlib, json, os, py_compile, re, subprocess, sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "manuscript_tools"
REF = ROOT / "results" / "reference_tables"
RELEASE = ROOT / "pipeline" / "v1.5.3" / "algorithmic_reconstruction_framework.py"
PROTOCOL = ROOT / "protocol" / "arf_confirmatory_protocol.json"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run(args, cwd):
    env = dict(os.environ, ARF_ROOT=str(ROOT), MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r


def test_release_of_record_matches_protocol():
    p = json.loads(PROTOCOL.read_text())
    assert sha256(RELEASE) == p["pipeline_sha256"]
    assert sha256(RELEASE).startswith("14426c89")


def test_protocol_amendments_in_order():
    ids = [a["id"] for a in json.loads(PROTOCOL.read_text())["amendments"]]
    assert ids == [f"A{i}" for i in range(1, 20)]


def test_every_python_file_compiles(tmp_path):
    files = [f for f in ROOT.rglob("*.py") if ".git" not in f.parts]
    assert len(files) > 50
    for f in files:
        py_compile.compile(str(f), cfile=str(tmp_path / "x.pyc"), doraise=True)


def test_tables_regenerate(tmp_path):
    (tmp_path / "tables").mkdir()
    run([TOOLS / "make_tables.py"], cwd=tmp_path)
    for name in ("tab_confirm.tex", "tab_benchmark.tex", "tab_a4_exact_economy.tex",
                 "tab_a9_budget.tex", "tab_a9_failsignal.tex", "tab_a8_pooled.tex",
                 "tab_a11_volume.tex"):
        assert (tmp_path / "tables" / name).read_text() == (REF / name).read_text(), name


def test_a12_a13_figure_and_table(tmp_path):
    summ = ROOT / "results" / "summary" / "a12_a13_summary.json"
    run([TOOLS / "figscripts" / "fig_a12_a13.py", summ, tmp_path / "fig.png", tmp_path / "tab_a13.tex"],
        cwd=tmp_path)
    assert (tmp_path / "tab_a13.tex").read_text() == (REF / "tab_a13.tex").read_text()
    assert (tmp_path / "fig.png").stat().st_size > 10_000


def test_result_figures(tmp_path):
    run([TOOLS / "figscripts" / "fig_results.py", str(tmp_path) + "/"], cwd=tmp_path)
    assert len(list(tmp_path.glob("*.png"))) >= 5


TLE_LINE = re.compile(r"^1 \d{5}[A-Z ] \d{5}[A-Z ]{0,3}\s+\d{5}\.\d+", re.M)
FORBIDDEN_EXT = {".tle", ".3le", ".shp", ".shx", ".dbf", ".gpkg", ".geojson", ".kml", ".kmz"}


def test_no_licensed_data():
    for f in ROOT.rglob("*"):
        if ".git" in f.parts or not f.is_file():
            continue
        assert f.suffix.lower() not in FORBIDDEN_EXT, f
        assert "wdpa" not in f.name.lower() and "wdpca" not in f.name.lower(), f
        if f.suffix.lower() in {".txt", ".csv", ".json", ".py", ".log", ".md"}:
            assert not TLE_LINE.search(f.read_text(errors="ignore")), f


# ---------------------------------------------------------------- release 1.5.6, amendments A18 and A19
V156 = ROOT / "pipeline" / "v1.5.6" / "algorithmic_reconstruction_framework_v156.py"
V155 = ROOT / "pipeline" / "v1.5.5" / "algorithmic_reconstruction_framework_v155.py"
DENSE = ROOT / "results" / "dense_constellation"
A19 = ROOT / "analyses" / "A19_dense_posthoc"


def _cell2_functions(path):
    """The catalogue-number and sample functions of a release, taken from the file itself."""
    import ast
    import numpy as np
    src = Path(path).read_text(); tree = ast.parse(src)
    keep = {"_alpha5_to_int", "_parse_norad_id", "_elset_epoch_utc", "_elset_orbit_km", "select_operational_sample"}
    parts = [ast.get_source_segment(src, n) for n in tree.body
             if (isinstance(n, ast.FunctionDef) and n.name in keep) or
             (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("OPERATIONAL_CRITERIA", "_ALPHA5_LETTERS")
                                                for t in n.targets))]
    ns = {"np": np}; exec("\n".join(parts), ns)
    return ns, src


def _element_lines(cat):
    """Two lines in TLE layout for a made-up satellite (not an element set of any real object)."""
    l1 = "1 " + cat + "U 26001A   26214.50000000  .00000000  00000-0  00000-0 0  0000"
    l2 = "2 " + cat + "  53.0000 100.0000 0001000  90.0000 270.0000 15.06000000    10"
    return l1[:69], l2[:69]


def test_release_156_is_pinned_and_decodes_alpha5():
    proto = json.loads(PROTOCOL.read_text())
    by_id = {a["id"]: a for a in proto["amendments"]}
    assert sha256(V156) == by_id["A18"]["script"]["algorithmic_reconstruction_framework_v156.py"]
    assert sha256(V156) == by_id["A19"]["script"]["algorithmic_reconstruction_framework_v156.py"]
    n6, s6 = _cell2_functions(V156); n5, s5 = _cell2_functions(V155)
    dec = n6["_alpha5_to_int"]
    assert (dec("A0000"), dec("Z9999"), dec("H9999"), dec("J0000"), dec("N9999"), dec("P0000")) == \
        (100000, 339999, 179999, 180000, 229999, 230000)
    assert dec("25544") == 25544 and n6["_parse_norad_id"]("1 I0001U") is None and n6["_parse_norad_id"]("1 O0001U") is None
    recs = [("X",) + _element_lines(c) for c in ("44000", "44001", "99999", "A0000", "B1234", "Z9999")]
    epoch = "2026-08-03T00:00:00"
    _, st5 = n5["select_operational_sample"](recs, epoch, 42)
    _, st6 = n6["select_operational_sample"](recs, epoch, 42)
    assert st5["malformed"] == 3 and st6["malformed"] == 0
    num = recs[:3]
    assert n5["select_operational_sample"](num, epoch, 42) == n6["select_operational_sample"](num, epoch, 42)
    assert "_opsample_a5_seed" in s6 and "_opsample_a5_seed" not in s5


def test_dense_constellation_results_are_consistent():
    import numpy as np
    import pandas as pd
    runs = pd.read_csv(DENSE / "a19" / "a19_runs.csv")
    idx = pd.concat([pd.read_csv(DENSE / f"paper2_v156_{b}_index.csv") for b in ("a18_core", "a18_sens", "a19_n8000")])
    assert len(runs) == 90 == len(idx) and set(idx.status) == {"done"} and set(idx.release) == {"1.5.6"}
    assert set(idx.pipeline_sha256) == {sha256(V156)}
    reps = sorted((DENSE / "a19" / "replay").glob("*_summary.json"))
    assert len(reps) == 90
    S = [json.loads(f.read_text()) for f in reps]
    assert all(s["identity_check"] == "passed" and s["images_with_repeated_cells"] == 0 for s in S)
    assert max(s["caps"]["nocap"]["more_images_than_fcfs_pct"] for s in S) == 0.0
    sys.path.insert(0, str(A19))
    try:
        import runpy
        M = runpy.run_path(str(A19 / "arf_paper2_summary_a19.py"), run_name="a19")
    finally:
        sys.path.pop(0)
    core = runs[(runs.swath_km == 32.5) & (runs.sun_deg == 10.0) & (runs.N >= 500)]
    (tinf, a, b), _ = M["fit_power"](core.N, core.floor_h)
    stored = json.loads((DENSE / "a19" / "a19_fit.json").read_text())
    assert stored["N_8000_included"] and len(core) == stored["all_windows"]["runs"] == 28
    assert abs(tinf - stored["all_windows"]["estimate"]["T_inf_h"]) < 1e-3 and abs(b - stored["all_windows"]["estimate"]["b"]) < 1e-3
    gains = pd.read_csv(DENSE / "a19" / "a19_doubling_gains.csv")
    last = gains[(gains.to_N == 8000) & gains.matched].gain_h
    assert len(last) == 2 and last.max() < 0.5
