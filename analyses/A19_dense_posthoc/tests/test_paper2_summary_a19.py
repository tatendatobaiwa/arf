"""Test of arf_paper2_summary_a19.py on a mock MyDrive. Run folders are synthetic, their per-onset floors set to
(1) the REAL per-run floors of the 88 finished runs (a18_runs.csv, as uploaded on 15 November 2026): the summary
    must read back the same floors and return the restricted fit computed independently on that file
    (T_inf 2.37 h, b 1.05; matched windows 1.99 h; without N = 4000 2.36 h) and the matched gains 1.35 and 1.13 h;
(2) a known curve T = 2.5 + 4000/N for N >= 500 with N = 8000 added: the fit must recover 2.5 and b = 1, N*_1h = 4000,
    N = 8000 must be read, and the direct statement must follow the last doubling.
Also: replay summaries are read into the table; without them and without N = 8000 the summary still runs and says so;
a protocol without A19 is refused; both LaTeX tables compile without overfull boxes; the figure is written.
Usage: test_paper2_summary_a19.py SUMMARY_A19 SUMMARY_A18 A18_RUNS_CSV"""
import sys, os, json, shutil, tempfile, subprocess, runpy, io, contextlib
import numpy as np, pandas as pd

S19, S18, RUNS = sys.argv[1:4]
RES = []


def check(c, m):
    RES.append(bool(c)); print(("PASS " if c else "FAIL ") + m, flush=True)


M = runpy.run_path(S19, run_name="s19")
REAL = pd.read_csv(RUNS)


def drive(table, with_replay, proto_ids=("A18", "A19")):
    root = tempfile.mkdtemp(prefix="p2s19_")
    code, batch = os.path.join(root, "ARF_code"), os.path.join(root, "ARF_batch")
    os.makedirs(code); os.makedirs(os.path.join(batch, "paper2_a19", "replay"))
    shutil.copy(S18, os.path.join(code, "arf_paper2_summary_a18.py"))
    json.dump({"amendments": [{"id": i} for i in proto_ids]}, open(os.path.join(code, "arf_confirmatory_protocol.json"), "w"))
    idx = {}
    for r in table.itertuples():
        name = f"ARF_Outputs_N{r.N}_from{r.window.replace('-', '')}_sun{r.sun_deg:g}_sw{r.swath_km:g}km"
        d = os.path.join(root, "runs", name); os.makedirs(d)
        o = pd.DataFrame({"onset_shift_min": range(0, 1441, 20)})
        o["best_possible_clear_t95r_min"] = r.floor_h * 60.0
        o["best_possible_clear_t95_min"] = r.floor_h * 60.0
        o["earliest_completion_clear_t95r_min"] = r.floor_h * 60.0
        o["earliest_completion_clear_images95r"] = 28.0
        o["fcfs_unbudgeted_clear_images95r"] = 50.0
        o["reachable_clear_tiles"] = int(round(r.reach_pct / 100.0 * 1000))
        o.to_csv(os.path.join(d, "arf_onset_robustness.csv"), index=False)
        json.dump({"t_max_wait_min": 94.0}, open(os.path.join(d, "arf_cell2_summary.json"), "w"))
        pd.DataFrame({"tile": [f"t{i}" for i in range(1000)]}).to_csv(os.path.join(d, "arf_mosaic_tiles.csv"), index=False)
        cfg = {"active_n": int(r.N), "analysis_epoch": r.window + "T00:00:00", "proxy_swath_km": r.swath_km,
               "proxy_min_sun_elev_deg": r.sun_deg}
        idx.setdefault(r.batch, []).append({"run_id": name, "status": "done", "release": "1.5.6", "out_dir": d,
                                            "minutes": 1.0, "config": json.dumps(cfg)})
        if with_replay:
            cap = lambda pct: {"completes_pct": pct, "within_margin_of_floor_pct": pct, "delay_over_floor_min_median": 0.0,
                               "images_mean": 40.0, "images_max": 55, "fewer_images_than_fcfs_pct": 100.0,
                               "more_images_than_fcfs_pct": 0.0}
            json.dump({"fcfs_images_mean": 50.0, "images_with_repeated_cells": 0,
                       "caps": {"k30": cap(10.0), "k60": cap(90.0), "k120": cap(100.0), "nocap": cap(100.0)}},
                      open(os.path.join(batch, "paper2_a19", "replay", name + "_summary.json"), "w"))
    for b, rows in idx.items():
        pd.DataFrame(rows).to_csv(os.path.join(batch, b + "_index.csv"), index=False)
    return root, code, batch


def run(root, code, batch):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df, fit, A = M["main"](["--code", code, "--batch-dir", batch])
    return df, fit, A, buf.getvalue()


# (1) the real floors
root, code, batch = drive(REAL, with_replay=False)
df, fit, A, txt = run(root, code, batch)
check(len(df) == 88 and np.allclose(df.sort_values(["swath_km", "sun_deg", "N", "window"]).floor_h.to_numpy(),
                                    REAL.sort_values(["swath_km", "sun_deg", "N", "window"]).floor_h.to_numpy()),
      "88 runs read; floors equal the uploaded a18_runs.csv")
e = fit["all_windows"]
check(e["runs"] == 26 and abs(e["estimate"]["T_inf_h"] - 2.37) < 0.01 and abs(e["estimate"]["b"] - 1.05) < 0.01,
      f"real data, N >= 500: T_inf {e['estimate']['T_inf_h']:.2f} h, b {e['estimate']['b']:.2f} (independent computation: 2.37, 1.05)")
check(abs(e["ci95"]["T_inf_h"][0] - 1.51) < 0.15 and abs(e["ci95"]["T_inf_h"][1] - 2.96) < 0.15,
      f"bootstrap interval of T_inf {e['ci95']['T_inf_h'][0]:.2f} to {e['ci95']['T_inf_h'][1]:.2f} h (independent: 1.51 to 2.96)")
check(abs(fit["matched_windows"]["estimate"]["T_inf_h"] - 1.99) < 0.01 and
      abs(fit["all_windows_without_largest_N"]["estimate"]["T_inf_h"] - 2.36) < 0.01, "matched windows 1.99 h; without N = 4000 2.36 h")
g = [x for x in fit["doubling_gains_matched_windows_h"] if x["from_N"] == 2000][0]
check(abs(g["min"] - 1.13) < 0.01 and abs(g["max"] - 1.35) < 0.01 and "no saturation shown directly" in fit["direct_statement"],
      "matched gains 2000 to 4000: 1.13 and 1.35 h; direct statement: no saturation shown")
check(not fit["N_8000_included"] and "N = 8000 NOT yet run" in txt and "replay NOT yet run" in txt and A is None,
      "runs without N = 8000 and without the replay, and says so")
check(fit["reach_pct_min_in_fit"] > 99.0, "every run in the fit has the region reachable")

# (2) a known curve with N = 8000
core = REAL[(REAL.swath_km == 32.5) & (REAL.sun_deg == 10.0)].copy()
rng = np.random.default_rng(3)
core.loc[core.N >= 500, "floor_h"] = 2.5 + 4000.0 / core.loc[core.N >= 500, "N"] + rng.normal(0, 0.05, int((core.N >= 500).sum()))
extra = core[(core.N == 4000)].copy(); extra["N"] = 8000; extra["floor_h"] = 2.5 + 4000.0 / 8000; extra["batch"] = "paper2_v156_a19_n8000"
T2 = pd.concat([core, extra, REAL[REAL.swath_km != 32.5]])
root, code, batch = drive(T2, with_replay=True)
df, fit, A, txt = run(root, code, batch)
s = fit["all_windows"]["estimate"]
check(fit["N_8000_included"] and abs(s["T_inf_h"] - 2.5) < 0.1 and abs(s["b"] - 1.0) < 0.05 and abs(s["N_star_1h"] - 4000) < 300,
      f"known curve: T_inf {s['T_inf_h']:.2f} (2.5), b {s['b']:.2f} (1), N*_1h {s['N_star_1h']:.0f} (4000); N = 8000 read")
lo, hi = fit["all_windows"]["ci95"]["T_inf_h"]
check(lo <= 2.5 <= hi, "bootstrap interval contains the true limit")
check("below 0.5 h per doubling" not in fit["direct_statement"] and "below 1 h per doubling" in fit["direct_statement"],
      "direct statement follows the last doubling (0.5 h gained from 4000 to 8000: below 1 h, not below 0.5 h)")
check(A is not None and len(A) == T2.groupby(["swath_km", "sun_deg", "N"]).ngroups and float(A.completes_k60_pct.iloc[0]) == 90.0,
      "replay summaries read into the table by camera and N")
out = os.path.join(batch, "paper2_a19")
check(all(os.path.exists(os.path.join(out, f)) for f in ("a19_runs.csv", "a19_fit.json", "a19_doubling_gains.csv",
      "a19_replay_by_case_and_N.csv", "tab_a19_fit.tex", "tab_a19_replay.tex", "fig_a19_floor_vs_N.png")), "all output files written")
tex = ("\\documentclass[10pt]{article}\\usepackage[a4paper,margin=2cm]{geometry}\\begin{document}\\input{tab_a19_fit}"
       "\\input{tab_a19_replay}\\end{document}")
open(os.path.join(out, "t.tex"), "w").write(tex)
r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "t.tex"], cwd=out, capture_output=True, text=True)
log = open(os.path.join(out, "t.log")).read() if os.path.exists(os.path.join(out, "t.log")) else ""
check(os.path.exists(os.path.join(out, "t.pdf")) and "Overfull" not in log and "booktabs" not in open(os.path.join(out, "tab_a19_fit.tex")).read(),
      "both tables compile without overfull boxes; enclosed grids, no booktabs")
root, code, batch = drive(REAL.head(5), with_replay=False, proto_ids=("A18",))
try:
    run(root, code, batch); ok = False
except AssertionError as ex:
    ok = "A19" in str(ex)
check(ok, "a protocol without A19 is refused")
print(f"{sum(RES)}/{len(RES)} passed"); sys.exit(0 if all(RES) else 1)
