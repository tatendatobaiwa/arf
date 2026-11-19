# =============================================================================
# ARF ADDED DRY-SEASON WINDOWS: SUMMARY AND DECISIONS (protocol amendment A15;
# read-only)
# =============================================================================
# Reads the runs of batch dryext_v153_a15 and decides H1a, H1b and H2 on these
# eight windows alone by the protocol's rules: per window, the protocol's own
# window outcomes (arf_confirmatory_analysis.window_outcomes); over windows, the
# one-sided Student-t lower bound at alpha/3 against the protocol's threshold
# (H1a, H1b) or 0 (H2). The helpers are those of the A5 summary
# (arf_wetseason_summary_a5.py, checked by SHA-256), unchanged. Beside the
# decisions it gives, descriptively, the same statistics over all sixteen
# dry-season windows (the eight confirmatory windows plus these eight).
# Writes ARF_batch/dryext_a15/ (CSV, JSON, LaTeX table with a fully enclosed grid).
#
# Use (Colab):  !python /content/drive/MyDrive/ARF_code/arf_dryext_summary_a15.py
# =============================================================================
import os
import json
import runpy
import hashlib
import argparse

import numpy as np
import pandas as pd

SHA_A5_SUMMARY = "321abaca621ee81e87ea7d644248a9c50a42914e37ecee8fbdaa116cf6192ce3"   # the registered A5 summary (arf_wetseason_summary_a5.py)


def decide(pool, th):
    d = {}
    for col, ref in (("h1a_pct", th), ("h1b_pct", th), ("h2_diff", 0.0)):
        p = pool[col]
        b = p["lcb_alpha_over_3"]
        ok = p["windows"] >= 5 and b is not None and (b >= ref if col != "h2_diff" else b > ref)
        d[col.split("_")[0].upper()] = {"windows": p["windows"], "mean": p["mean"], "lcb_alpha_over_3": b,
                                        "supported": bool(ok) if p["windows"] >= 5 else None}
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description="A15 summary and decisions.")
    ap.add_argument("--code", default="/content/drive/MyDrive/ARF_code")
    ap.add_argument("--batch-dir", default="/content/drive/MyDrive/ARF_batch")
    ap.add_argument("--sha-a5", default=SHA_A5_SUMMARY)
    a, _ = ap.parse_known_args(argv)
    p5 = os.path.join(a.code, "arf_wetseason_summary_a5.py")
    if a.sha_a5:
        assert hashlib.sha256(open(p5, "rb").read()).hexdigest() == a.sha_a5, "A5 summary differs from its release"
    S = runpy.run_path(p5, run_name="a5_summary")
    proto = json.load(open(os.path.join(a.code, "arf_confirmatory_protocol.json")))
    assert any(x["id"] == "A15" for x in proto.get("amendments", [])), "protocol without amendment A15"
    an = runpy.run_path(os.path.join(a.code, "arf_confirmatory_analysis.py"), run_name="arf_confirmatory_analysis")
    idx = pd.read_csv(os.path.join(a.batch_dir, "dryext_v153_a15_index.csv"), dtype=str).fillna("")
    rows = [S["run_row"](r["out_dir"], an, int(proto["min_evaluable_onsets"]))
            for _, r in idx[(idx.status == "done") & (idx.release == proto["release"])].iterrows()]
    df = pd.DataFrame(rows).sort_values("window")
    out = os.path.join(a.batch_dir, "dryext_a15")
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, "a15_runs.csv"), index=False)
    alpha, th = float(proto["alpha"]), float(proto["threshold_pct"])
    pool = S["pooled"](df, alpha, th)
    summ = {"label": "REGISTERED REPLICATION (amendment A15): decided on the eight added windows alone",
            "runs": int(len(df)), "pooled": pool, "decisions": decide(pool, th)}
    p = os.path.join(a.batch_dir, "arf_confirmatory_results_windows.csv")
    if os.path.exists(p):
        c = pd.read_csv(p)
        c = c[(c.roi == "ckgr_okavango") & (c.fleet == "earth-resources+planet") &
              (c.planet_swath_case.fillna("conservative") == "conservative") & (c.exclude_satellites.fillna("") == "")]
        both = pd.concat([c[["h1a_pct", "h1b_pct", "h2_diff"]], df[["h1a_pct", "h1b_pct", "h2_diff"]]])
        summ["sixteen_dry_windows_descriptive"] = S["pooled"](both, alpha, th)
    json.dump(summ, open(os.path.join(out, "a15_summary.json"), "w"), indent=1, default=str)

    f = S["f"]
    L = ["\\begin{table}[!t]", "\\centering", "\\footnotesize", "\\renewcommand{\\arraystretch}{1.2}",
         "\\caption{Added dry-season windows (amendment A15, registered replication): primary design, eight 48-h "
         "windows of July and late September to October 2026, release 1.5.3, $p = 0.35$. H1a, H1b: share of onsets "
         "(\\%). H2: adaptive minus open-loop coverage under cloud, same images (points). Bound: one-sided Student-$t$ "
         "lower bound at $\\alpha/3$.}", "\\label{tab:a15}", "\\begin{tabular}{|l|r|r|r|}", "\\hline",
         "Window & H1a & H1b & H2 \\\\", "\\hline"]
    for _, r in df.iterrows():
        L += [f"{r['window']} & {f(r['h1a_pct'])} & {f(r['h1b_pct'])} & {f(r['h2_diff'], 2)} \\\\", "\\hline"]
    L += [f"Mean & {f(pool['h1a_pct']['mean'])} & {f(pool['h1b_pct']['mean'])} & {f(pool['h2_diff']['mean'], 2)} \\\\",
          "\\hline",
          f"Bound & {f(pool['h1a_pct']['lcb_alpha_over_3'])} & {f(pool['h1b_pct']['lcb_alpha_over_3'])} & "
          f"{f(pool['h2_diff']['lcb_alpha_over_3'], 3)} \\\\", "\\hline",
          "Supported & " + " & ".join({True: "yes", False: "no", None: "n.e."}[summ["decisions"][h]["supported"]]
                                      for h in ("H1A", "H1B", "H2")) + " \\\\", "\\hline",
          "\\end{tabular}", "\\end{table}"]
    open(os.path.join(out, "a15_table.tex"), "w").write("\n".join(L) + "\n")
    print(f"A15 (registered replication): {len(df)} runs")
    cols = [c for c in ("window", "h1a_pct", "h1b_pct", "h2_diff", "archive_matched", "observed_cloud_mean_pct")
            if c in df.columns]
    print(df[cols].round(3).to_string(index=False))
    print(json.dumps(summ["decisions"], indent=1, default=str))
    if "sixteen_dry_windows_descriptive" in summ:
        print("Sixteen dry windows (descriptive):", json.dumps(summ["sixteen_dry_windows_descriptive"], default=str)[:1500])
    print(f"Files: {out}/a15_runs.csv, a15_summary.json, a15_table.tex")
    return df, summ


if __name__ == "__main__":
    main()
