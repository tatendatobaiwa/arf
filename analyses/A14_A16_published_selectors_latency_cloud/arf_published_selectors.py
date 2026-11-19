"""Published image selectors on the predicted passes of a run (post hoc amendment A14).

Why: ECC has so far been compared with FCFS, two greedy objectives and exact
optima. Reviewers will ask how it compares with published image selectors. Three
are re-implemented here from their published descriptions and given exactly the
same candidates as ECC (the run's events of the 24 h after each onset, clear sky,
H3 resolution 7, the same 95% level of the reachable cells):

  WANG26  Wang et al. (2026), Remote Sensing 18(11):1855, sections 3.3 to 3.5.
          Candidates ranked by temporal relevance, then spatial resolution, then
          cloud (lowest); ties to the first image. An image is accepted when it
          intersects the remaining target cells, whose covered cells are then
          removed; the loop ends when the target or the candidates are exhausted.
          "Invalid coverage" (an image all of whose covered cells are covered by
          other selected images) is then removed.
          Adaptation: "most recent" becomes "earliest delivered after the onset";
          resolution is the sensor's ground sample distance; cloud is 0 (clear
          sky). The target is the run's reachable cells and the loop also stops
          once the level is reached (the variant at the level is reported; the
          variant run to full coverage is reported beside it).
  KS17    Kempeneers and Soille (2017), Big Earth Data 1(1-2):145-158, Algorithm 1.
          Per tile, the minimum set of at most M_max = 5 images maximising the
          cloud-free covered pixels of the tile, by Sequential Floating Forward
          Search. Adaptation: a tile is the H3 resolution-3 parent of the cells
          (about 12,400 km^2, close to a 100 km MGRS tile); the criterion counts
          the tile's reachable cells covered; the selected images are the union
          over tiles. The SFFS body is the standard one (Pudil et al. 1994): the
          article's inclusion and exclusion steps were not available to us, so
          this is an assumption, stated in the amendment. Ties go to the earlier
          image, which can only help the method's completion time.
  PAN24   Pan et al. (2024), Geo-spatial Information Science 27(5):1524-1541.
          Shortest time span of acquisition giving full coverage (found by uniform
          cost search), then the images of that span with better quality and
          larger area. Adaptation: the span is searched over delivery times after
          the onset (the time of interest) with a two-pointer scan, which returns
          the same minimum span as uniform cost search over contiguous spans; ties
          go to the span starting earliest; within the span, images are taken by
          largest new coverage (quality is equal in clear sky) until the level is
          reached. HFA-Net quality grading has no counterpart for predicted passes
          and is not used.

For every onset the script also recomputes the floor, ECC and FCFS with the rules
of Cell 7C (the A13 port) and stops unless they equal the run's
arf_onset_robustness.csv. Reported per onset and method: time to the level (min
after onset), images, coverage reached (% of reachable cells).

Usage: arf_published_selectors.py RUN_DIR OUT_PREFIX [--k 30] [--margin-min M]
"""
import argparse
import json
import math
import os
import re
import sys

import h3
import numpy as np
import pandas as pd

LEVEL = 95.0
HORIZON, SHIFT_H, STEP = 1440, 24, 20
KS_MMAX, KS_TILE_RES = 5, 3
# Ground sample distance (m) used only for WANG26's second ranking key; ties in
# delivery time are rare. PlanetScope 3 m (Planet, PSScene ortho product),
# Sentinel-2 10 m (ESA), Landsat 8/9 30 m (USGS).
GSD = [(re.compile(r"^FLOCK", re.I), 3.0), (re.compile(r"^SENTINEL-2", re.I), 10.0),
       (re.compile(r"^LANDSAT", re.I), 30.0)]


def gsd(sat):
    for pat, v in GSD:
        if pat.match(sat):
            return v
    raise ValueError(f"no ground sample distance for {sat}")


def load(run_dir):
    ev = pd.read_csv(os.path.join(run_dir, "arf_refined_events.csv"))
    util = (np.clip(1.0 - ev["A_nadir_deg"].to_numpy(float) / ev["theta_max_deg"].to_numpy(float), 0.0, None)
            * ev["D_downlink"].to_numpy(float))
    ev = ev.assign(util=util)
    tiles = pd.Index(pd.unique(ev["tile"]))
    ev = ev.assign(tid=tiles.get_indexer(ev["tile"]))
    keys, cells, wait = [], [], []
    for (sat, t), g in ev.groupby(["sat", "t_min"], sort=True):
        keys.append((sat, int(t)))
        cells.append(g.loc[g["util"] > 0.0, "tid"].to_numpy(np.int64))
        w = g["T_wait_min"].to_numpy(float)[0]
        wait.append(np.inf if np.isnan(w) else float(w))
    parent = pd.Index([h3.cell_to_parent(c, KS_TILE_RES) for c in tiles])
    return keys, cells, np.array(wait), len(tiles), parent


# --------------------------------------------------------------------------- reference: floor, ECC, FCFS (Cell 7C, A13 port)
def reference(sub, order, times, cells, keys, wait, n_tiles, d, need, reach, k):
    seen = np.zeros(n_tiles, bool); n_seen = 0; floor = None; fcfs = None; n_kept = 0; pos_floor = None
    for pos, o in enumerate(order):
        c = cells[sub[o]]
        new = c[~seen[c]]
        if new.size:
            seen[new] = True; n_seen += new.size; n_kept += 1
            if floor is None and 100.0 * n_seen / reach >= LEVEL - 1e-9:
                floor = float(times[pos])
            if fcfs is None and n_seen >= need:
                fcfs = n_kept; pos_floor = pos
        if floor is not None and fcfs is not None:
            break

    def probe(pos):
        end = int(np.searchsorted(times, times[pos], side="right"))
        cand = [sub[o] for o in order[:end]]
        covered = np.zeros(n_tiles, bool); avail = np.ones(len(cand), bool); chosen = []; n_cov = 0
        for _ in range(k):
            gain = np.array([(~covered[cells[e]]).sum() if avail[i] else -1 for i, e in enumerate(cand)])
            j = int(np.argmax(gain))
            if gain[j] <= 0:
                break
            avail[j] = False; chosen.append(cand[j])
            c = cells[cand[j]]
            n_cov += int((~covered[c]).sum()); covered[c] = True
            if n_cov >= need:
                break
        return n_cov >= need, chosen
    ecc_t = ecc_n = None
    if pos_floor is not None:
        hi = len(order) - 1
        ok, best = probe(hi)
        if ok:
            a, b = pos_floor, hi
            while a < b:
                mid = (a + b) // 2
                okm, ch = probe(mid)
                if okm:
                    b, best = mid, ch
                else:
                    a = mid + 1
            ecc_t = float(max(keys[e][1] - d + wait[e] for e in best)); ecc_n = len(best)
    return {"floor_min": floor, "ecc_min": ecc_t, "ecc_images": ecc_n, "fcfs_images": fcfs}


def level_time(selected, dl_of, cells, n_tiles, need, reach):
    """Time at which the selected images, taken in delivery order, first cover `need` cells; final coverage %."""
    covered = np.zeros(n_tiles, bool); n = 0; t_lvl = None
    for e in sorted(selected, key=lambda e: (dl_of[e], e)):
        c = cells[e]
        n += int((~covered[c]).sum()); covered[c] = True
        if t_lvl is None and n >= need:
            t_lvl = float(dl_of[e])
    return t_lvl, 100.0 * n / reach


# --------------------------------------------------------------------------- WANG26
def wang26(cand, dl_of, keys, cells, n_tiles, target, need, stop_at_level):
    rank = sorted(cand, key=lambda e: (dl_of[e], gsd(keys[e][0]), 0.0, e))   # time, resolution, cloud (0), first
    Z = target.copy(); n_target = int(Z.sum()); got = 0; R = []
    for e in rank:
        c = cells[e]
        hit = c[Z[c]]
        if hit.size:
            R.append(e); Z[hit] = False; got += hit.size
            if stop_at_level and got >= need:
                break
        if not Z.any():
            break
    # section 3.5: remove invalid coverage, checking the images in selection order
    cnt = np.zeros(n_tiles, np.int32)
    for e in R:
        np.add.at(cnt, np.unique(cells[e]), 1)
    keep = []
    for e in R:
        c = np.unique(cells[e])
        if c.size and np.all(cnt[c] >= 2):
            cnt[c] -= 1                                    # every cell is covered by another selected image
        else:
            keep.append(e)
    return keep


# --------------------------------------------------------------------------- KS17
def sffs(cand, cells_in_tile, mmax):
    """Standard SFFS (Pudil et al. 1994) maximising J(X) = |union of covered tile cells|; minimal X reaching max J."""
    def J(X):
        if not X:
            return 0
        return int(np.unique(np.concatenate([cells_in_tile[e] for e in X])).size)
    best = {0: ([], 0)}
    X = []
    full = J(cand)
    while len(X) < mmax:
        # inclusion: most significant feature (ties: earliest in `cand`, which is delivery order)
        gains = [(J(X + [e]), -i, e) for i, e in enumerate(cand) if e not in X]
        if not gains:
            break
        g, _, e = max(gains)
        X = X + [e]
        if len(X) not in best or g > best[len(X)][1]:
            best[len(X)] = (list(X), g)
        # conditional exclusion
        while len(X) > 2:
            vals = [(J([x for x in X if x != r]), -X.index(r), r) for r in X[:-1]]
            v, _, r = max(vals)
            if v > best[len(X) - 1][1]:
                X = [x for x in X if x != r]
                best[len(X)] = (list(X), v)
            else:
                break
        if best[len(X)][1] >= full:
            break
    top = max(v for _, v in best.values())
    m = min(n for n, (_, v) in best.items() if v == top)            # the minimum set reaching the best criterion
    return best[m][0]


def ks17(cand, cells, parent_of_cell, target):
    by_tile = {}
    for e in cand:
        c = cells[e]
        c = c[target[c]]
        if not c.size:
            continue
        for p in np.unique(parent_of_cell[c]):
            by_tile.setdefault(p, {})[e] = c[parent_of_cell[c] == p]
    sel = set()
    for p, d in by_tile.items():
        sel.update(sffs(list(d.keys()), d, KS_MMAX))
    return sorted(sel)


# --------------------------------------------------------------------------- PAN24
def pan24(cand, dl_of, cells, n_tiles, need):
    order = sorted(cand, key=lambda e: (dl_of[e], e))
    t = np.array([dl_of[e] for e in order])
    cnt = np.zeros(n_tiles, np.int32); n = 0; best = None; lo = 0
    for hi, e in enumerate(order):                          # two-pointer over delivery times: shortest span
        c = np.unique(cells[e]); n += int((cnt[c] == 0).sum()); cnt[c] += 1
        while n >= need:
            span = t[hi] - t[lo]
            if best is None or span < best[0] - 1e-9:
                best = (span, lo, hi)
            c0 = np.unique(cells[order[lo]]); cnt[c0] -= 1; n -= int((cnt[c0] == 0).sum()); lo += 1
    if best is None:
        return []
    _, a, b = best
    # extend to all images delivered at the span's end time (same minute)
    while b + 1 < len(order) and t[b + 1] == t[b]:
        b += 1
    pool = order[a:b + 1]
    covered = np.zeros(n_tiles, bool); got = 0; X = []; avail = list(pool)
    while got < need and avail:
        gains = [(int((~covered[cells[e]]).sum()), -dl_of[e], -e) for e in avail]
        j = int(np.argmax([g[0] * 1e9 + g[1] * 1e3 + g[2] * 1e-6 for g in gains]))
        if gains[j][0] <= 0:
            break
        e = avail.pop(j); X.append(e); c = cells[e]; got += int((~covered[c]).sum()); covered[c] = True
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("out_prefix")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--margin-min", type=float, default=None)
    ap.add_argument("--onsets", type=int, default=None, help="test only: first N onsets")
    a = ap.parse_args()
    margin = a.margin_min
    c2 = os.path.join(a.run_dir, "arf_cell2_summary.json")
    if margin is None and os.path.exists(c2):
        margin = float(json.load(open(c2))["t_max_wait_min"])
    if margin is None:
        sys.exit("H1a margin unknown: no arf_cell2_summary.json; pass --margin-min")
    keys, cells, wait, n_tiles, parents = load(a.run_dir)
    pidx = pd.Index(pd.unique(parents)); parent_of_cell = pidx.get_indexer(parents)
    ref = pd.read_csv(os.path.join(a.run_dir, "arf_onset_robustness.csv")).set_index("onset_shift_min")
    shifts = list(range(0, SHIFT_H * 60 + 1, STEP))[: a.onsets]
    rows, bad = [], []
    for d in shifts:
        sub = [e for e in range(len(keys)) if d <= keys[e][1] < d + HORIZON]
        if not sub:
            continue
        reach_mask = np.zeros(n_tiles, bool)
        for e in sub:
            reach_mask[cells[e]] = True
        reach = int(reach_mask.sum())
        if reach == 0:
            continue
        need = int(math.ceil(LEVEL / 100.0 * reach - 1e-9))
        dl = np.array([keys[e][1] - d + wait[e] for e in sub])
        order = np.argsort(dl, kind="stable"); order = order[np.isfinite(dl[order])]
        times = dl[order]
        r = reference(sub, order, times, cells, keys, wait, n_tiles, d, need, reach, a.k)
        for mine, theirs in {"floor_min": "best_possible_clear_t95r_min", "ecc_min": "earliest_completion_clear_t95r_min",
                             "ecc_images": "earliest_completion_clear_images95r",
                             "fcfs_images": "fcfs_unbudgeted_clear_images95r"}.items():
            x, y = r[mine], ref.loc[d, theirs]
            if (x is None) != pd.isna(y) or (x is not None and not np.isclose(float(x), float(y))):
                bad.append((d, mine, x, y))
        if reach != int(ref.loc[d, "reachable_clear_tiles"]):
            bad.append((d, "reachable", reach, ref.loc[d, "reachable_clear_tiles"]))
        cand = [sub[o] for o in order]                       # deliverable images in delivery order
        dl_of = {sub[o]: float(dl[o]) for o in order}
        row = {"onset_min": d, "reachable": reach, "need": need, **r}
        for name, X in (("wang26", wang26(cand, dl_of, keys, cells, n_tiles, reach_mask, need, True)),
                        ("wang26_full", wang26(cand, dl_of, keys, cells, n_tiles, reach_mask, need, False)),
                        ("ks17", ks17(cand, cells, parent_of_cell, reach_mask)),
                        ("pan24", pan24(cand, dl_of, cells, n_tiles, need))):
            t_lvl, cov = level_time(X, dl_of, cells, n_tiles, need, reach)
            row.update({f"{name}_min": t_lvl, f"{name}_images": len(X), f"{name}_cov_pct": cov,
                        f"{name}_last_min": max((dl_of[e] for e in X), default=None)})
        rows.append(row)
    if bad:
        print("IDENTITY CHECK FAILED (first 10):", bad[:10]); sys.exit(1)
    print(f"identity check: {len(rows)} onsets: floor, ECC and FCFS equal the run's arf_onset_robustness.csv")
    out = pd.DataFrame(rows)
    out.to_csv(a.out_prefix + "_onsets.csv", index=False)
    ok = out.dropna(subset=["floor_min"])
    summ = {"run_dir": os.path.basename(os.path.normpath(a.run_dir)), "onsets": int(len(ok)), "margin_min": margin,
            "k": a.k, "identity_check": "passed", "methods": {}}
    for m in ("ecc", "wang26", "wang26_full", "ks17", "pan24"):
        t = ok[f"{m}_min"]; n = ok[f"{m}_images"] if m != "ecc" else ok["ecc_images"]
        at_floor = ((t - ok["floor_min"]) <= margin + 1e-9) & t.notna()
        s = {"reached_level_pct": 100.0 * float(t.notna().mean()),
             "within_margin_of_floor_pct": 100.0 * float(at_floor.mean()),
             "time_h_median": None if t.dropna().empty else float(t.median()) / 60,
             "images_mean": float(n.mean()), "images_median": float(n.median())}
        if m != "ecc":
            s["coverage_pct_median"] = float(ok[f"{m}_cov_pct"].median())
            s["ecc_fewer_images_pct"] = 100.0 * float((ok["ecc_images"] < n).mean())
            s["ecc_not_later_pct"] = 100.0 * float(((ok["ecc_min"] <= t + 1e-9) | t.isna()).mean())
        summ["methods"][m] = s
    summ["fcfs_images_mean"] = float(ok["fcfs_images"].mean())
    print(json.dumps(summ, indent=1))
    json.dump(summ, open(a.out_prefix + "_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
