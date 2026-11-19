# =============================================================================
# ARF PAPER 2 SUMMARY (protocol amendment A17; read-only)
# =============================================================================
# Reads the finished runs of the A17 batches (core, sensitivity) and reports, per
# assumed camera (swath, sun limit) and fleet size N, the means over windows of:
#   floor_h      median over onsets of the physical floor to 95% of reachable cells
#   reach_pct    median over onsets of the reachable share of the region
#   whole_pct    share of onsets at which 95% of the WHOLE region is covered
#   ecc_images, fcfs_images, saving_images   images to the 95% level, and FCFS - ECC
#   h1a_pct      share of onsets at which ECC is within the H1a margin of the floor
#   h1b_pct      among onsets where FCFS needs 2+ images, share where ECC also uses fewer
#   h2_diff      Fail-Signal coverage gain over the open-loop plan (points), seeds as in the protocol
# with percentile 95% intervals from a window-block bootstrap (B = 2000, seed 42).
# Saturation (core case only, registered in A17): the floor against N is fitted by
#   (1) T(N) = T_inf + a N^-b (least squares); N*_m = (a/m)^(1/b) and the smallest
#       grid N with T(N) - T_inf <= m, for m = 0.5 h and 1 h;
#   (2) one-breakpoint segmented regression of T on log10 N (least squares over the
#       breakpoint, the model of Muggeo 2003);
#   (3) the Kneedle knee (Satopaa et al. 2011): the grid point of largest
#       difference between the normalised chord and the normalised curve;
# each with window-block bootstrap intervals. Descriptive: no hypothesis decision.
# Writes ARF_batch/paper2_a17/ (CSV, JSON, LaTeX table with a fully enclosed grid, figure).
#
# Use (Colab):  !python /content/drive/MyDrive/ARF_code/arf_paper2_summary_a17.py
# =============================================================================
import os
import json
import argparse

import numpy as np
import pandas as pd

LEVEL = 95.0
B, SEED = 2000, 42
MARGINS_H = (0.5, 1.0)
BATCHES = ("paper2_v155_a17_core", "paper2_v155_a17_sens")
PRIMARY = (32.5, 10.0)


def run_metrics(d):
    o = pd.read_csv(os.path.join(d, "arf_onset_robustness.csv"))
    c2 = json.load(open(os.path.join(d, "arf_cell2_summary.json")))
    margin = float(c2["t_max_wait_min"])
    n_region = int(len(pd.read_csv(os.path.join(d, "arf_mosaic_tiles.csv"), usecols=["tile"])))
    fl = o["best_possible_clear_t95r_min"]
    ok = o[fl.notna()]
    ecc_t, ecc_n = ok["earliest_completion_clear_t95r_min"], ok["earliest_completion_clear_images95r"]
    fc_n = ok["fcfs_unbudgeted_clear_images95r"]
    at_floor = (ecc_t - ok["best_possible_clear_t95r_min"] <= margin + 1e-9) & ecc_t.notna()
    elig = fc_n >= 2
    h2 = None
    p = os.path.join(d, "arf_cloud_robustness_summary.json")
    if os.path.exists(p):
        try:
            h2 = float(json.load(open(p))["h2"]["time_aware"]["difference_vs_openloop_expected"]["mean"])
        except (KeyError, TypeError, ValueError):
            h2 = None
    return {"onsets": int(len(o)), "evaluable": int(len(ok)), "margin_min": margin,
            "floor_h": float(fl.median()) / 60 if fl.notna().any() else np.nan,
            "reach_pct": 100.0 * float(o["reachable_clear_tiles"].median()) / n_region,
            "whole_pct": 100.0 * float(o["best_possible_clear_t95_min"].notna().mean()),
            "ecc_images": float(ecc_n.mean()), "fcfs_images": float(fc_n.mean()),
            "saving_images": float((fc_n - ecc_n).mean()),
            "h1a_pct": 100.0 * float(at_floor.mean()) if len(ok) else np.nan,
            "h1b_pct": 100.0 * float(((ecc_n < fc_n) & at_floor)[elig].mean()) if elig.any() else np.nan,
            "h2_diff": h2}


def boot_ci(vals, rng):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, np.nan, np.nan
    if v.size == 1:
        return float(v[0]), np.nan, np.nan
    m = rng.choice(v, size=(B, v.size), replace=True).mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fit_power(N, T):
    from scipy.optimize import curve_fit
    f = lambda n, tinf, a, b: tinf + a * np.power(n, -b)
    p0 = (float(np.min(T)), float(np.max(T) - np.min(T)) * float(np.min(N)) ** 0.5, 0.5)
    (tinf, a, b), _ = curve_fit(f, N, T, p0=p0, bounds=([0, 0, 0.01], [np.inf, np.inf, 5]), maxfev=20000)
    return tinf, a, b


def segmented(x, y):
    """One-breakpoint continuous piecewise-linear fit of y on x; returns the breakpoint."""
    best = None
    for c in np.linspace(np.sort(np.unique(x))[1], np.sort(np.unique(x))[-2], 200):
        X = np.column_stack([np.ones_like(x), x, np.clip(x - c, 0, None)])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = float(((X @ coef - y) ** 2).sum())
        if best is None or r < best[0]:
            best = (r, c)
    return best[1]


def kneedle(x, y):
    """Knee of a decreasing convex curve: largest gap between the normalised chord and the curve."""
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else np.zeros_like(y)
    chord = 1 - xn                                   # from (0, 1) to (1, 0)
    return int(np.argmax(chord - yn))


def saturation(table, rng):
    """table: rows (window, N, floor_h) of the core case."""
    grid = np.array(sorted(table.N.unique()), float)
    wins = sorted(table.window.unique())

    def estimate(sub):
        N, T = sub.N.to_numpy(float), sub.floor_h.to_numpy(float)
        out = {}
        tinf, a, b = fit_power(N, T)
        out.update({"T_inf_h": tinf, "a": a, "b": b})
        for m in MARGINS_H:
            out[f"N_star_power_{m:g}h"] = float((a / m) ** (1 / b)) if a > 0 else 1.0
            ok = grid[(a * grid ** -b) <= m]
            out[f"N_star_grid_{m:g}h"] = float(ok.min()) if ok.size else np.nan
        means = sub.groupby("N").floor_h.mean().reindex(grid)
        out["breakpoint_N"] = float(10 ** segmented(np.log10(N), T))
        out["knee_N"] = float(grid[kneedle(np.log10(grid), means.to_numpy())]) if means.notna().all() else np.nan
        return out

    est = estimate(table)
    boots = []
    for _ in range(B // 10):                            # 200 window-block resamples (curve fits are slow)
        pick = rng.choice(wins, size=len(wins), replace=True)
        sub = pd.concat([table[table.window == w] for w in pick])
        try:
            boots.append(estimate(sub))
        except Exception:                               # a resample the fit cannot follow: recorded, not hidden
            continue
    ci = {k: [float(np.nanpercentile([b_[k] for b_ in boots], 2.5)), float(np.nanpercentile([b_[k] for b_ in boots], 97.5))]
          for k in est} if boots else {}
    return {"estimate": est, "ci95": ci, "bootstrap_fits": len(boots), "grid": grid.tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", default="/content/drive/MyDrive/ARF_batch")
    ap.add_argument("--code", default="/content/drive/MyDrive/ARF_code")
    a, _ = ap.parse_known_args(argv)
    proto = json.load(open(os.path.join(a.code, "arf_confirmatory_protocol.json")))
    assert any(x["id"] == "A17" for x in proto.get("amendments", [])), "protocol without amendment A17"
    rows = []
    for b in BATCHES:
        p = os.path.join(a.batch_dir, f"{b}_index.csv")
        if not os.path.exists(p):
            continue
        idx = pd.read_csv(p, dtype=str).fillna("")
        for _, r in idx[(idx.status == "done") & (idx.release == "1.5.5")].iterrows():
            cfg = json.loads(r["config"])
            rows.append({"batch": b, "window": cfg["analysis_epoch"][:10], "N": int(cfg["active_n"]),
                         "swath_km": float(cfg["proxy_swath_km"]), "sun_deg": float(cfg["proxy_min_sun_elev_deg"]),
                         "out_dir": r["out_dir"], **run_metrics(r["out_dir"])})
    assert rows, "no finished A17 run found"
    df = pd.DataFrame(rows).drop_duplicates(["window", "N", "swath_km", "sun_deg"]).sort_values(
        ["swath_km", "sun_deg", "N", "window"])
    out = os.path.join(a.batch_dir, "paper2_a17")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "a17_runs.csv"), index=False)
    rng = np.random.default_rng(SEED)
    cols = ("floor_h", "reach_pct", "whole_pct", "ecc_images", "fcfs_images", "saving_images", "h1a_pct", "h1b_pct", "h2_diff")
    agg = []
    for (sw, sun, n), g in df.groupby(["swath_km", "sun_deg", "N"]):
        r = {"swath_km": sw, "sun_deg": sun, "N": n, "windows": int(len(g))}
        for c in cols:
            m, lo, hi = boot_ci(g[c].tolist(), rng)
            r.update({c: m, f"{c}_lo": lo, f"{c}_hi": hi})
        agg.append(r)
    A = pd.DataFrame(agg)
    A.to_csv(os.path.join(out, "a17_by_case_and_N.csv"), index=False)
    core = df[(df.swath_km == PRIMARY[0]) & (df.sun_deg == PRIMARY[1])][["window", "N", "floor_h"]].dropna()
    summ = {"label": "PAPER 2, amendment A17: descriptive", "runs": int(len(df)),
            "cases": sorted({f"{s:g} km, sun {u:g} deg" for s, u in zip(df.swath_km, df.sun_deg)})}
    if core.N.nunique() >= 4 and core.window.nunique() >= 2:
        summ["saturation_core"] = saturation(core, rng)
    json.dump(summ, open(os.path.join(out, "a17_summary.json"), "w"), indent=1, default=float)

    fmt = lambda x, d=1: "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"
    L = ["\\begin{table}[!t]", "\\centering", "\\footnotesize", "\\renewcommand{\\arraystretch}{1.2}",
         "\\caption{Starlink orbits as a hypothetical dense Earth-observation constellation (amendment A17): means over "
         "windows of the per-run medians. Floor: hours from onset to 95\\% of the reachable cells. Reach: reachable "
         "share of the region (\\%). ECC, FCFS: images to that level.}", "\\label{tab:a17}",
         "\\begin{tabular}{|c|c|r|r|r|r|r|r|}", "\\hline",
         "Swath (km) & Sun ($^\\circ$) & $N$ & Windows & Floor (h) & Reach (\\%) & ECC & FCFS \\\\", "\\hline"]
    for r in A.itertuples():
        L += [f"{r.swath_km:g} & {r.sun_deg:g} & {r.N} & {r.windows} & {fmt(r.floor_h)} & {fmt(r.reach_pct)} & "
              f"{fmt(r.ecc_images)} & {fmt(r.fcfs_images)} \\\\", "\\hline"]
    L += ["\\end{tabular}", "\\end{table}"]
    open(os.path.join(out, "tab_a17.tex"), "w").write("\n".join(L) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for (sw, sun), g in A.groupby(["swath_km", "sun_deg"]):
        ax.errorbar(g.N, g.floor_h, yerr=[g.floor_h - g.floor_h_lo, g.floor_h_hi - g.floor_h], marker="o",
                    capsize=3, label=f"{sw:g} km, sun {sun:g}°")
    if "saturation_core" in summ:
        e = summ["saturation_core"]["estimate"]
        n = np.geomspace(min(A.N), max(A.N), 200)
        ax.plot(n, e["T_inf_h"] + e["a"] * n ** -e["b"], "k--", lw=1, label="fit, 32.5 km")
    ax.set_xscale("log"); ax.set_xlabel("Fleet size N"); ax.set_ylabel("Floor to 95% of reachable cells (h)")
    ax.legend(fontsize=8, frameon=False); fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_a17_floor_vs_N.png"), dpi=300)
    print(A[["swath_km", "sun_deg", "N", "windows", "floor_h", "reach_pct", "ecc_images", "fcfs_images",
             "h1a_pct", "h2_diff"]].round(2).to_string(index=False))
    print(json.dumps(summ.get("saturation_core", {}), indent=1, default=float)[:2500])
    print(f"Files: {out}/")
    return df, A, summ


if __name__ == "__main__":
    main()
