"""Product latency scenarios (post hoc amendment A16, part a).

Why: every receiving-station wait in the runs was 0, so the floor counts only
the wait for acquisition. Real products reach users later. This replay adds a
per-mission product latency L_m to every delivery time and recomputes the
floor, the reachable count, ECC (k = 30) and FCFS with the registered A13 port
of Cell 7C (arf_planet_thinning.py, checked by SHA-256).

Latencies (minutes) and their sources:
  Sentinel-2  Real-Time: no later than 100 min after sensing; Near Real-Time:
              100 min to 3 h; Nominal: 3 to 24 h (ESA, Sentinel-2 MSI Technical
              Guide, glossary of timeliness).
  Landsat 8/9 Level-1 Real-Time: available in 4 to 6 h, less than 12 h (USGS,
              Landsat Collection 2 Level-1 pages and archive structure page).
  PlanetScope no published figure was found: scenario values only.
Scenarios (S2, Landsat, PlanetScope, in hours):
  fast      1.67, 4, 2      sourced lower ends; PlanetScope assumed
  typical   3, 6, 6         upper ends of NRT and of Landsat RT; PlanetScope assumed
  slow      24, 12, 24      upper ends of Nominal and of Landsat RT; PlanetScope assumed
  ps2, ps6, ps12, ps24      typical S2 and Landsat, PlanetScope 2, 6, 12 or 24 h
Identity check: with all latencies 0 every onset must equal the run's
arf_onset_robustness.csv, or the script stops.

Usage: arf_product_latency.py RUN_DIR OUT_PREFIX [--k 30] [--margin-min M]
"""
import argparse
import hashlib
import json
import os
import re
import runpy
import sys

import numpy as np
import pandas as pd

SHA_A13 = "bfb122f7dd57e3adfa69e033f6882f3cf5e2f019654bc3ad7d1ebf45f8434d9a"
SCEN = {"zero": (0, 0, 0), "fast": (100 / 60, 4, 2), "typical": (3, 6, 6), "slow": (24, 12, 24),
        "ps2": (3, 6, 2), "ps6": (3, 6, 6), "ps12": (3, 6, 12), "ps24": (3, 6, 24)}
MISSION = [re.compile(r"^SENTINEL-2", re.I), re.compile(r"^LANDSAT", re.I), re.compile(r"^FLOCK", re.I)]


def mission_of(sat):
    for i, p in enumerate(MISSION):
        if p.match(sat):
            return i
    raise ValueError(sat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("out_prefix")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--margin-min", type=float, default=None)
    ap.add_argument("--a13", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "arf_planet_thinning.py"))
    a = ap.parse_args()
    assert hashlib.sha256(open(a.a13, "rb").read()).hexdigest() == SHA_A13, "arf_planet_thinning.py is not the A13 release"
    T = runpy.run_path(a.a13, run_name="a13")
    margin = a.margin_min
    c2 = os.path.join(a.run_dir, "arf_cell2_summary.json")
    if margin is None and os.path.exists(c2):
        margin = float(json.load(open(c2))["t_max_wait_min"])
    if margin is None:
        sys.exit("H1a margin unknown: pass --margin-min")
    keys, cells, wait, n_tiles = T["load"](a.run_dir)
    mis = np.array([mission_of(s) for s, _ in keys])
    full = list(range(len(keys)))
    shifts = list(range(0, T["SHIFT_H"] * 60 + 1, T["STEP"]))
    ref = pd.read_csv(os.path.join(a.run_dir, "arf_onset_robustness.csv")).set_index("onset_shift_min")
    rows = []
    for name, lat_h in SCEN.items():
        w = wait + np.array(lat_h)[mis] * 60.0
        for d in shifts:
            m = T["onset_metrics"](full, keys, cells, w, n_tiles, d, a.k) or {}
            rows.append({"scenario": name, "onset_min": d, **m})
            if name == "zero":
                for mine, theirs in {"floor_min": "best_possible_clear_t95r_min",
                                     "ecc_min": "earliest_completion_clear_t95r_min",
                                     "ecc_images": "earliest_completion_clear_images95r",
                                     "fcfs_images": "fcfs_unbudgeted_clear_images95r"}.items():
                    x, y = m.get(mine), ref.loc[d, theirs]
                    if (x is None) != pd.isna(y) or (x is not None and not np.isclose(float(x), float(y))):
                        sys.exit(f"IDENTITY CHECK FAILED at onset {d}: {mine} {x} != {y}")
        if name == "zero":
            print(f"identity check: {len(shifts)} onsets equal the run's arf_onset_robustness.csv at zero latency")
    out = pd.DataFrame(rows)
    out.to_csv(a.out_prefix + "_onsets.csv", index=False)
    summ = {"run_dir": os.path.basename(os.path.normpath(a.run_dir)), "margin_min": margin, "k": a.k,
            "identity_check": "passed", "latency_h": {k: list(v) for k, v in SCEN.items()}, "scenarios": {}}
    for name in SCEN:
        s = T["window_stats"](out[out.scenario == name], margin)
        g = out[out.scenario == name]
        s["floor_shift_h_median"] = float((g["floor_min"] - out[out.scenario == "zero"]["floor_min"].to_numpy()).median()) / 60
        summ["scenarios"][name] = s
    print(json.dumps(summ, indent=1))
    json.dump(summ, open(a.out_prefix + "_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
