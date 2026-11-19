"""Test of arf_colab_launch_paper2_a19.py on a mock MyDrive in the state the A18 work left it (release 1.5.3; the A18
protocol with A17 kept; release 1.5.6, batch runner 1.5.4, A18 runner and summary), with the REAL A19 zip.
Install: checksums; A19 protocol installed, A18 kept byte for byte; A19 files copied; every A18 file unchanged;
rerun changes nothing; refused: a protocol ending at A17, a changed release 1.5.3, a different file in ARF_code.
End to end with the installed files: synthetic finished runs (N = 500 to 4000, one needing more than 30 images) ->
replay --all (identity check against each run's own onset file) -> summary: the replay table is filled from the
replay's own output, the fit runs, and the run whose ECC needed more than 30 images shows as not completing at k = 30.
Usage: test_colab_paper2_a19.py CELL ZIP RELEASE_153 PROTOCOL_A18 PROTOCOL_A17 A18_DIR REPLAY_TEST"""
import sys, os, json, shutil, tempfile, runpy, io, contextlib, hashlib
import numpy as np, pandas as pd

CELL, ZIP, REL, P18, P17, A18, RT = sys.argv[1:8]
RES = []


def check(c, m):
    RES.append(bool(c)); print(("PASS " if c else "FAIL ") + m, flush=True)


sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
C = runpy.run_path(CELL, run_name="cell")
root = tempfile.mkdtemp(prefix="p2c19_"); code = os.path.join(root, "ARF_code"); os.makedirs(code)
shutil.copy(REL, os.path.join(code, "algorithmic_reconstruction_framework.py"))
shutil.copy(P18, os.path.join(code, "arf_confirmatory_protocol.json"))
shutil.copy(P17, os.path.join(code, "arf_confirmatory_protocol.a17.json"))
a18_files = ("algorithmic_reconstruction_framework_v156.py", "arf_batch_runner_v154.py", "arf_paper2_runner_a18.py",
             "arf_paper2_summary_a18.py")
for f in a18_files:
    shutil.copy(os.path.join(A18, f), code)
before = {f: sha(os.path.join(code, f)) for f in a18_files + ("arf_confirmatory_protocol.a17.json",)}
shutil.copy(ZIP, os.path.join(root, C["ZIP_NAME"]))
work = os.path.join(root, "work")
with contextlib.redirect_stdout(io.StringIO()):
    C["install"](root=root, work=work)
p = json.load(open(os.path.join(code, "arf_confirmatory_protocol.json")))
check([a["id"] for a in p["amendments"]][-2:] == ["A18", "A19"], "A19 protocol installed, A18 inside it")
check(sha(os.path.join(code, "arf_confirmatory_protocol.a18.json")) == sha(P18), "A18 protocol kept byte for byte")
check(all(os.path.exists(os.path.join(code, f)) for f in C["FILES"]), "A19 replay, runner and summary in ARF_code")
check(all(sha(os.path.join(code, f)) == h for f, h in before.items()), "every A18 file and the A17 protocol unchanged")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    C["install"](root=root, work=work)
check("already installed" in buf.getvalue(), "rerun: nothing changed")

# ---- end to end with the installed files
src = open(RT).read().split("root = tempfile.mkdtemp")[0]
sys.argv = ["x", os.path.join(code, "arf_ecc_cap_replay_a19.py"), sys.argv[0]]
ns = {}
exec(src.replace('R = runpy.run_path(A14, run_name="a14")', "R = None"), ns)        # make_run and POOL only
REP = runpy.run_path(os.path.join(code, "arf_ecc_cap_replay_a19.py"), run_name="rep")
batch = os.path.join(root, "ARF_batch"); os.makedirs(batch)
rows = []
for n, n_img, width in ((500, 300, 40), (1000, 600, 40), (2000, 1200, 40), (4000, 2500, 6)):
    name = f"ARF_Outputs_ckgr_okavango_N{n}_from20260803_sun10_sw32.5km"
    rd = ns["make_run"](os.path.join(root, "runs"), name, n, max(n // 10, 5), n_img, width, False)
    t_img, wait, cells, n_tiles = REP["load"](rd)
    on = []
    for d in range(0, 1441, 20):
        r = REP["onset"](d, t_img, wait, cells, n_tiles, caps=(30,))
        if r is not None:
            on.append({"onset_shift_min": d, "best_possible_clear_t95r_min": r["floor_min"], "best_possible_clear_t95_min": r["floor_min"],
                       "earliest_completion_clear_t95r_min": r["ecc_k30_min"], "earliest_completion_clear_images95r": r["ecc_k30_images"],
                       "fcfs_unbudgeted_clear_images95r": r["fcfs_images"], "reachable_clear_tiles": r["reachable"]})
    pd.DataFrame(on).to_csv(os.path.join(rd, "arf_onset_robustness.csv"), index=False)
    pd.DataFrame({"tile": ns["POOL"]}).to_csv(os.path.join(rd, "arf_mosaic_tiles.csv"), index=False)
    cfg = {"active_n": n, "analysis_epoch": "2026-08-03T00:00:00", "proxy_swath_km": 32.5, "proxy_min_sun_elev_deg": 10.0}
    rows.append({"run_id": name, "status": "done", "release": "1.5.6", "out_dir": rd, "minutes": 1.0, "config": json.dumps(cfg)})
pd.DataFrame(rows).to_csv(os.path.join(batch, "paper2_v156_a18_core_index.csv"), index=False)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    REP["main"](["--all", "--root", root])
check("4/4 runs done" in buf.getvalue() and buf.getvalue().count("identity ok") == 4, "replay: 4 runs, identity check passed in each")
SUM = runpy.run_path(os.path.join(code, "arf_paper2_summary_a19.py"), run_name="sum")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    df, fit, A = SUM["main"](["--code", code, "--batch-dir", batch])
check(A is not None and len(A) == 4 and A.ecc_nocap_images.notna().all() and "replay read for 4 runs" in buf.getvalue(),
      "summary: replay table filled from the replay's own output")
narrow = A[A.N == 4000].iloc[0]
check(narrow.ecc_nocap_images > 30 and narrow.completes_k30_pct < 100.0 and narrow.completes_k120_pct >= narrow.completes_k60_pct >= narrow.completes_k30_pct,
      f"the narrow run needs {narrow.ecc_nocap_images:.0f} images without a cap and completes at k = 30 in {narrow.completes_k30_pct:.0f}% of onsets")
check(np.isfinite(fit["all_windows"]["estimate"]["T_inf_h"]) and os.path.exists(os.path.join(batch, "paper2_a19", "tab_a19_replay.tex")),
      "fit and tables written")


def refused(fn, text):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
        return False
    except AssertionError as e:
        return text in str(e)


f = os.path.join(code, "arf_paper2_runner_a19.py"); open(f, "a").write("# x\n")
check(refused(lambda: C["install"](root=root, work=work), "differs from the package"), "a different file in ARF_code is refused")
shutil.copy(os.path.join(work, "arf_paper2_runner_a19.py"), f)
shutil.copy(P17, os.path.join(code, "arf_confirmatory_protocol.json")); os.remove(os.path.join(code, "arf_confirmatory_protocol.a18.json"))
check(refused(lambda: C["install"](root=root, work=work), "it must end at A18"), "a protocol ending at A17 is refused")
shutil.copy(P18, os.path.join(code, "arf_confirmatory_protocol.json"))
open(os.path.join(code, "algorithmic_reconstruction_framework.py"), "a").write("# x\n")
check(refused(lambda: C["install"](root=root, work=work), "release 1.5.3 changed"), "a changed release 1.5.3 is refused")
print(f"{sum(RES)}/{len(RES)} passed"); sys.exit(0 if all(RES) else 1)
