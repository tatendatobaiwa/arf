# =============================================================================
# ARF PAPER 2 SUMMARY, amendment A19 (post hoc; read-only; release 1.5.6)
# =============================================================================
# A17's registered saturation fit (one curve through N = 5 to 4000) FAILED and is
# reported as failed (ARF_batch/paper2_a18/a18_summary.json: limit 0 h, N* of 10^12):
# below N = 500 the reachable cells grow with N, so the floor to 95% of the
# reachable cells rises to N = 100 before it falls. A19 registers, AFTER seeing
# those results (post hoc, labelled as such everywhere):
#
# (b) RESTRICTED FIT. Core case (32.5 km, sun 10 deg), runs with N >= 500 only (the
#     fleet sizes at which the median reachable share is 100%):
#         T(N) = T_inf + a N^-b, least squares, 0 <= T_inf <= min T, 0.01 <= b <= 5,
#     with a window-block bootstrap (B = 2000, seed 42). Reported: T_inf, b, and
#     N*_m = (a/m)^(1/b), the fleet size at which the fitted floor is within m of
#     its limit, m = 0.5 h and 1 h (A17's definition). Three variants, all reported:
#     all windows; the two windows that hold the largest fleets (3 August and
#     7 September; point estimates and one fit per window, no interval from two
#     windows); all windows without the largest N (stability).
#     DIRECT EVIDENCE, reported beside the fit and given precedence over it: the
#     hours gained by each doubling of N, per window, on matched windows.
# (d) N = 8000 in the two matched windows (batch paper2_v156_a19_n8000), read here
#     when finished; the summary runs with or without it and says which.
# (a) ECC CAP REPLAY (arf_ecc_cap_replay_a19.py): per camera and N, means over
#     windows of FCFS images, ECC images without a cap, the share of onsets at which
#     ECC completes with at most 30, 60 and 120 images, and its delay over the floor.
#
# Writes ARF_batch/paper2_a19/: a19_runs.csv, a19_fit.json, a19_doubling_gains.csv,
# a19_replay_by_case_and_N.csv, tab_a19_fit.tex and tab_a19_replay.tex (fully
# enclosed grids), fig_a19_floor_vs_N.png.
# Use (Colab): through arf_colab_launch_paper2_a19.py with STAGE = "summary".
# =============================================================================
import os
import json
import glob
import runpy
import argparse

import numpy as np
import pandas as pd

B, SEED = 2000, 42
MARGINS_H = (0.5, 1.0)
N_MIN = 500
PRIMARY = (32.5, 10.0)
MATCHED = ("2026-08-03", "2026-09-07")
BATCHES = ("paper2_v156_a18_core", "paper2_v156_a18_sens", "paper2_v156_a19_n8000")


def fit_power(N, T):
    """Least squares of T = T_inf + a N^-b with 0 <= T_inf <= min T; several starts, the best kept."""
    from scipy.optimize import least_squares
    N, T = np.asarray(N, float), np.asarray(T, float)
    res = lambda p: p[0] + p[1] * N ** (-p[2]) - T
    best = None
    for b0 in (0.3, 0.7, 1.0, 1.5):
        t0 = max(0.5 * T.min(), 1e-6)
        r = least_squares(res, [t0, max(T.max() - t0, 1e-6) * N.min() ** b0, b0],
                          bounds=([0.0, 0.0, 0.01], [max(T.min(), 1e-6), np.inf, 5.0]))
        if best is None or r.cost < best.cost:
            best = r
    return tuple(float(x) for x in best.x), float(np.sqrt(np.mean(best.fun ** 2)))


def n_star(a, b, m):
    return float((a / m) ** (1.0 / b)) if a > 0 else float("nan")


def describe(p):
    tinf, a, b = p
    return {"T_inf_h": tinf, "a": a, "b": b, **{f"N_star_{m:g}h": n_star(a, b, m) for m in MARGINS_H}}


def fit_block(sub, rng, boot=True):
    """sub: rows (window, N, floor_h). Point estimate, and window-block bootstrap intervals if asked."""
    p, rmse = fit_power(sub.N, sub.floor_h)
    out = {"runs": int(len(sub)), "windows": int(sub.window.nunique()), "N_values": sorted(int(n) for n in sub.N.unique()),
           "estimate": describe(p), "rmse_h": rmse}
    if boot and sub.window.nunique() >= 4:
        W = sorted(sub.window.unique()); groups = {w: sub[sub.window == w] for w in W}; reps = []
        for _ in range(B):
            s = pd.concat([groups[w] for w in rng.choice(W, len(W))])
            if s.N.nunique() >= 3:
                reps.append(describe(fit_power(s.N, s.floor_h)[0]))
        R = pd.DataFrame(reps)
        out["ci95"] = {c: [float(np.nanpercentile(R[c], 2.5)), float(np.nanpercentile(R[c], 97.5))] for c in R.columns}
        out["bootstrap_fits"] = int(len(R))
        out["T_inf_at_zero_pct"] = 100.0 * float((R["T_inf_h"] < 1e-3).mean())
    return out


def collect(code, batch_dir):
    S18 = runpy.run_path(os.path.join(code, "arf_paper2_summary_a18.py"), run_name="a18_summary")
    rows = []
    for b in BATCHES:
        p = os.path.join(batch_dir, b + "_index.csv")
        if not os.path.exists(p):
            continue
        idx = pd.read_csv(p)
        for r in idx[(idx.status == "done") & (idx.release == "1.5.6")].itertuples():
            c = json.loads(r.config)
            rows.append({"batch": b, "window": c["analysis_epoch"][:10], "N": int(c["active_n"]),
                         "swath_km": float(c["proxy_swath_km"]), "sun_deg": float(c["proxy_min_sun_elev_deg"]),
                         "out_dir": r.out_dir, "minutes": float(r.minutes), **S18["run_metrics"](r.out_dir)})
    return pd.DataFrame(rows).sort_values(["swath_km", "sun_deg", "N", "window"]).reset_index(drop=True)


def doubling_gains(core):
    P = core.pivot(index="window", columns="N", values="floor_h")
    Ns = [n for n in sorted(P.columns) if n >= N_MIN]
    rows = []
    for lo, hi in zip(Ns[:-1], Ns[1:]):
        if hi != 2 * lo:
            continue
        g = (P[lo] - P[hi]).dropna()
        for w, v in g.items():
            rows.append({"from_N": int(lo), "to_N": int(hi), "window": w, "gain_h": float(v), "matched": w in MATCHED})
    return pd.DataFrame(rows)


def replay_table(df, batch_dir):
    rep = os.path.join(batch_dir, "paper2_a19", "replay")
    rows = []
    for r in df.itertuples():
        p = os.path.join(rep, os.path.basename(r.out_dir.rstrip("/")) + "_summary.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p)); c = s["caps"]
        rows.append({"swath_km": r.swath_km, "sun_deg": r.sun_deg, "N": r.N, "window": r.window,
                     "fcfs_images": s["fcfs_images_mean"], "ecc_nocap_images": c["nocap"]["images_mean"],
                     "ecc_nocap_images_max": c["nocap"]["images_max"],
                     "nocap_fewer_than_fcfs_pct": c["nocap"]["fewer_images_than_fcfs_pct"],
                     "nocap_more_than_fcfs_pct": c["nocap"]["more_images_than_fcfs_pct"],
                     **{f"completes_{k}_pct": c[k]["completes_pct"] for k in ("k30", "k60", "k120")},
                     **{f"delay_{k}_min": c[k]["delay_over_floor_min_median"] for k in ("k30", "k60", "k120")},
                     "images_with_repeated_cells": s["images_with_repeated_cells"]})
    if not rows:
        return None, None
    R = pd.DataFrame(rows)
    num = [c for c in R.columns if c not in ("swath_km", "sun_deg", "N", "window")]
    A = R.groupby(["swath_km", "sun_deg", "N"]).agg(windows=("window", "size"), **{c: (c, "mean") for c in num}).reset_index()
    return R, A


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="/content/drive/MyDrive/ARF_code")
    ap.add_argument("--batch-dir", default="/content/drive/MyDrive/ARF_batch")
    a = ap.parse_args(argv)
    proto = json.load(open(os.path.join(a.code, "arf_confirmatory_protocol.json")))
    assert any(x["id"] == "A19" for x in proto.get("amendments", [])), "protocol without amendment A19: install it first"
    df = collect(a.code, a.batch_dir)
    assert len(df), "no finished release-1.5.6 run found"
    out = os.path.join(a.batch_dir, "paper2_a19"); os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "a19_runs.csv"), index=False)
    core = df[(df.swath_km == PRIMARY[0]) & (df.sun_deg == PRIMARY[1])][["window", "N", "floor_h", "reach_pct"]].dropna()
    use = core[core.N >= N_MIN]
    rng = np.random.default_rng(SEED)
    has8000 = bool((use.N == 8000).any())
    fit = {"label": "PAPER 2, amendment A19: POST HOC restricted fit; the registered A17 fit failed and is reported as failed",
           "model": "T(N) = T_inf + a N^-b, least squares, core case, N >= 500",
           "N_8000_included": has8000,
           "reach_pct_min_in_fit": float(use.reach_pct.min()),
           "all_windows": fit_block(use, rng),
           "matched_windows": fit_block(use[use.window.isin(MATCHED)], rng, boot=False),
           "matched_per_window": {w: fit_block(use[use.window == w], rng, boot=False) for w in MATCHED
                                  if use[use.window == w].N.nunique() >= 4},
           "all_windows_without_largest_N": fit_block(use[use.N < use.N.max()], rng)}
    G = doubling_gains(core)
    G.to_csv(os.path.join(out, "a19_doubling_gains.csv"), index=False)
    gm = G[G.matched].groupby(["from_N", "to_N"]).gain_h.agg(["mean", "min", "max", "size"]).reset_index()
    ga = G.groupby(["from_N", "to_N"]).gain_h.agg(["mean", "min", "max", "size"]).reset_index()
    fit["doubling_gains_matched_windows_h"] = gm.to_dict("records")
    fit["doubling_gains_all_windows_h"] = ga.to_dict("records")
    last = gm.iloc[-1] if len(gm) else None
    fit["direct_statement"] = None if last is None else (
        f"From N = {int(last.from_N)} to {int(last.to_N)} the floor fell by {last['mean']:.2f} h "
        f"({last['min']:.2f} to {last['max']:.2f} h in the {int(last['size'])} matched windows): "
        + ("below 0.5 h per doubling." if last["max"] < 0.5 else "below 1 h per doubling." if last["max"] < 1.0
           else "still 1 h or more per doubling in at least one window; no saturation shown directly."))
    json.dump(fit, open(os.path.join(out, "a19_fit.json"), "w"), indent=1, default=float)

    R, A = replay_table(df, a.batch_dir)
    fmt = lambda x, d=1: "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"
    e = fit["all_windows"]; ci = e.get("ci95", {})
    ci_s = lambda k, d=2: "--" if k not in ci else f"[{ci[k][0]:.{d}f}, {ci[k][1]:.{d}f}]"
    L = ["\\begin{table}[!t]", "\\centering", "\\footnotesize", "\\renewcommand{\\arraystretch}{1.2}",
         "\\caption{Post hoc restricted fit of the floor $T(N)=T_\\infty+aN^{-b}$ (amendment A19; core case, "
         "$N\\geq500$). The fit registered in A17 failed and is reported as failed. Intervals: window-block bootstrap, 95\\%.}",
         "\\label{tab:a19fit}", "\\begin{tabular}{|l|r|r|r|r|r|r|}", "\\hline",
         "Variant & Runs & $T_\\infty$ (h) & $b$ & $N^*_{1\\,\\mathrm{h}}$ & $N^*_{0.5\\,\\mathrm{h}}$ & RMSE (h) \\\\", "\\hline"]
    for name, key in (("All windows", "all_windows"), ("Matched windows", "matched_windows"),
                      ("Without largest $N$", "all_windows_without_largest_N")):
        v = fit[key]; s = v["estimate"]
        L += [f"{name} & {v['runs']} & {fmt(s['T_inf_h'], 2)} & {fmt(s['b'], 2)} & {fmt(s['N_star_1h'], 0)} & "
              f"{fmt(s['N_star_0.5h'], 0)} & {fmt(v['rmse_h'], 2)} \\\\", "\\hline"]
    L += [f"All windows, 95\\% & & {ci_s('T_inf_h')} & {ci_s('b')} & {ci_s('N_star_1h', 0)} & {ci_s('N_star_0.5h', 0)} & \\\\",
          "\\hline", "\\end{tabular}", "\\end{table}"]
    open(os.path.join(out, "tab_a19_fit.tex"), "w").write("\n".join(L) + "\n")
    if A is not None:
        R.to_csv(os.path.join(out, "a19_replay_runs.csv"), index=False)
        A.to_csv(os.path.join(out, "a19_replay_by_case_and_N.csv"), index=False)
        L = ["\\begin{table}[!t]", "\\centering", "\\footnotesize", "\\renewcommand{\\arraystretch}{1.2}",
             "\\caption{ECC with the image cap lifted (amendment A19, post hoc replay of the finished runs): means over "
             "windows. Images to 95\\% of the reachable cells for FCFS and for ECC without a cap, and the share of onsets "
             "(\\%) at which ECC completes with at most 30, 60 and 120 images.}", "\\label{tab:a19replay}",
             "\\begin{tabular}{|c|c|r|r|r|r|r|r|r|}", "\\hline",
             "Swath (km) & Sun ($^\\circ$) & $N$ & Windows & FCFS & ECC & $k\\leq30$ & $k\\leq60$ & $k\\leq120$ \\\\", "\\hline"]
        for r in A.itertuples():
            L += [f"{r.swath_km:g} & {r.sun_deg:g} & {r.N} & {r.windows} & {fmt(r.fcfs_images)} & {fmt(r.ecc_nocap_images)} & "
                  f"{fmt(r.completes_k30_pct, 0)} & {fmt(r.completes_k60_pct, 0)} & {fmt(r.completes_k120_pct, 0)} \\\\", "\\hline"]
        L += ["\\end{tabular}", "\\end{table}"]
        open(os.path.join(out, "tab_a19_replay.tex"), "w").write("\n".join(L) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for w, g in core.groupby("window"):
        ax.plot(g.N, g.floor_h, "-", color="0.75", lw=0.8, zorder=1)
    m = core.groupby("N").floor_h.mean()
    ax.plot(m.index, m.values, "o", color="C0", label="mean over windows", zorder=3)
    s = e["estimate"]; n = np.geomspace(N_MIN, max(core.N), 200)
    ax.plot(n, s["T_inf_h"] + s["a"] * n ** -s["b"], "k--", lw=1.2, label=f"post hoc fit, N ≥ {N_MIN}", zorder=2)
    ax.axhline(s["T_inf_h"], color="k", lw=0.6, ls=":"); ax.axvline(N_MIN, color="0.5", lw=0.6, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("Fleet size N"); ax.set_ylabel("Floor to 95% of reachable cells (h)")
    ax.legend(fontsize=8, frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_a19_floor_vs_N.png"), dpi=300); plt.close(fig)

    print(f"A19 summary: {len(df)} runs; N = 8000 {'included' if has8000 else 'NOT yet run'}; "
          f"ECC cap replay {'read for ' + str(len(R)) + ' runs' if R is not None else 'NOT yet run'}")
    print(json.dumps({k: fit[k] for k in ("all_windows", "matched_windows", "all_windows_without_largest_N",
                                           "doubling_gains_matched_windows_h", "direct_statement")}, indent=1, default=float))
    if A is not None:
        print(A[["swath_km", "sun_deg", "N", "windows", "fcfs_images", "ecc_nocap_images", "completes_k30_pct",
                 "completes_k60_pct", "completes_k120_pct", "delay_k60_min", "images_with_repeated_cells"]].round(1).to_string(index=False))
    print(f"Files: {out}/")
    return df, fit, A


if __name__ == "__main__":
    main()
