"""ECC with the image cap lifted, replayed on finished Paper 2 runs (post hoc amendment A19; read-only).

Why: the pipeline runs the earliest-completion cover (ECC) with at most k = 30 images (max_selection_steps,
the Paper 1 setting), while its FCFS comparator is not capped. In the dense Starlink proxy the cap binds:
at an 8 km swath the level needs 78 to 230 images, so ECC never completes from N = 250, and at 32.5 km ECC
uses 27 to 29 images at every N from 250. This script recomputes, for every onset of a run, from the run's
own arf_refined_events.csv (clear sky, 95% of the reachable cells, the 24 h after each onset, as Cell 7C):

  floor, FCFS images          as the run reports them
  ECC at k = 30               as the run reports it (IDENTITY CHECK: the script stops unless floor, ECC time,
                              ECC images, FCFS images and reachable cells equal arf_onset_robustness.csv)
  ECC at k = 60, 120          the same rule with a larger cap
  ECC without a cap           greedy cover of the images delivered by the floor. NOTE: without a cap the
                              images delivered by the floor always suffice, so this ECC completes AT the
                              floor by construction; what is measured is how many images it needs.

The rules are those of arf_published_selectors.py (A14: its reference(), the A13 port of Cell 7C, equal to the
pipeline at every onset of 24 real windows). Here the greedy step is evaluated lazily (a heap of upper bounds
on the gains, which only fall as cells are covered), which returns the same choice at every step, ties
included (largest gain, then the earlier delivered image), and is fast enough for thousands of images.
tests/test_ecc_cap_replay_a19.py checks it against the A14 function itself.

Usage:  arf_ecc_cap_replay_a19.py RUN_DIR OUT_PREFIX            one run
        arf_ecc_cap_replay_a19.py --all [--root MyDrive]        every finished run of the A18 and A19 batches
                                                                (resumable; writes ARF_batch/paper2_a19/replay)
"""
import argparse
import heapq
import json
import math
import os
import sys

import numpy as np
import pandas as pd

LEVEL = 95.0
HORIZON, SHIFT_H, STEP = 1440, 24, 20
CAPS = (30, 60, 120, None)                     # None = no cap
INDEXES = ("paper2_v156_a18_core_index.csv", "paper2_v156_a18_sens_index.csv", "paper2_v156_a19_n8000_index.csv")
COLS = ["sat", "t_min", "tile", "A_nadir_deg", "theta_max_deg", "D_downlink", "T_wait_min"]


def cap_name(k):
    return "nocap" if k is None else f"k{k}"


def load(run_dir):
    """Images of a run: time (min), wait to delivery (min), cells with positive utility. Order as in A14:
    images sorted by satellite then time; the cells of an image in file order."""
    ev = pd.read_csv(os.path.join(run_dir, "arf_refined_events.csv"), usecols=COLS)
    util = (np.clip(1.0 - ev["A_nadir_deg"].to_numpy(float) / ev["theta_max_deg"].to_numpy(float), 0.0, None)
            * ev["D_downlink"].to_numpy(float))
    tid, tiles = pd.factorize(ev["tile"])                       # first appearance, as pd.unique
    sat_code = pd.factorize(ev["sat"], sort=True)[0]
    t = ev["t_min"].to_numpy(float)
    order = np.lexsort((np.arange(len(ev)), t, sat_code))       # satellite, time, then file order
    sc, tt = sat_code[order], t[order]
    start = np.flatnonzero(np.r_[True, (sc[1:] != sc[:-1]) | (tt[1:] != tt[:-1])])
    end = np.r_[start[1:], len(order)]
    w = ev["T_wait_min"].to_numpy(float)[order][start]
    wait = np.where(np.isnan(w), np.inf, w)
    pos_util = util[order] > 0.0
    tid_o = tid[order].astype(np.int64)
    cells = [tid_o[a:b][pos_util[a:b]] for a, b in zip(start, end)]
    return tt[start].astype(np.int64), wait, cells, len(tiles)


def repeated_cells(cells):
    """Images that list a cell more than once. The counting rule of Cell 7C counts such a cell twice, so the
    statement 'without a cap ECC completes at the floor' is exact only when this is 0 (reported per run)."""
    return int(sum(np.unique(c).size != c.size for c in cells))


def greedy(cand, cells, n_tiles, need, k):
    """Greedy cover of `need` cells from the images `cand` (in delivery order), at most k images (None: no cap).
    Each step takes the largest gain, the earlier image on a tie, exactly as np.argmax over the gains."""
    covered = np.zeros(n_tiles, bool)
    heap = [(-int(cells[e].size), i, 0) for i, e in enumerate(cand)]      # (-upper bound, position, stamp)
    heapq.heapify(heap)
    chosen, n_cov, step = [], 0, 0
    while heap and (k is None or step < k):
        neg, i, stamp = heapq.heappop(heap)
        if stamp != step:                                       # stale bound: refresh and look again
            g = int((~covered[cells[cand[i]]]).sum())
            heapq.heappush(heap, (-g, i, step))
            continue
        if -neg <= 0:
            break
        c = cells[cand[i]]
        n_cov += int((~covered[c]).sum()); covered[c] = True
        chosen.append(cand[i]); step += 1
        if n_cov >= need:
            break
    return n_cov >= need, chosen


def onset(d, t_img, wait, cells, n_tiles, caps=CAPS):
    sub = np.flatnonzero((t_img >= d) & (t_img < d + HORIZON))
    if sub.size == 0:
        return None
    reach_mask = np.zeros(n_tiles, bool)
    for e in sub:
        reach_mask[cells[e]] = True
    reach = int(reach_mask.sum())
    if reach == 0:
        return None
    need = int(math.ceil(LEVEL / 100.0 * reach - 1e-9))
    dl = t_img[sub] - d + wait[sub]
    order = np.argsort(dl, kind="stable"); order = order[np.isfinite(dl[order])]
    times = dl[order]
    seen = np.zeros(n_tiles, bool); n_seen = 0; n_kept = 0; floor = fcfs = pos_floor = None
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
    row = {"onset_min": d, "reachable": reach, "need": need, "images_delivered": int(order.size),
           "floor_min": floor, "fcfs_images": fcfs}
    dl_of = dict(zip(sub[order].tolist(), times.tolist()))

    def probe(pos, k):
        end = int(np.searchsorted(times, times[pos], side="right"))
        return greedy(sub[order[:end]].tolist(), cells, n_tiles, need, k)

    for k in caps:
        t_k = n_k = None
        if pos_floor is not None:
            if k is None:
                ok, best = probe(pos_floor, None)
            else:
                hi = len(order) - 1
                ok, best = probe(hi, k)
                if ok:
                    a, b = pos_floor, hi
                    while a < b:
                        mid = (a + b) // 2
                        okm, ch = probe(mid, k)
                        if okm:
                            b, best = mid, ch
                        else:
                            a = mid + 1
            if ok:
                t_k, n_k = float(max(dl_of[e] for e in best)), len(best)
        row[f"ecc_{cap_name(k)}_min"], row[f"ecc_{cap_name(k)}_images"] = t_k, n_k
    return row


def replay(run_dir, out_prefix, onsets=None, caps=CAPS):
    t_img, wait, cells, n_tiles = load(run_dir)
    ref = pd.read_csv(os.path.join(run_dir, "arf_onset_robustness.csv")).set_index("onset_shift_min")
    margin = float(json.load(open(os.path.join(run_dir, "arf_cell2_summary.json")))["t_max_wait_min"])
    rows, bad = [], []
    for d in list(range(0, SHIFT_H * 60 + 1, STEP))[:onsets]:
        r = onset(d, t_img, wait, cells, n_tiles, caps)
        if r is None:
            continue
        if 30 in caps:
            for mine, theirs in {"floor_min": "best_possible_clear_t95r_min",
                                 "ecc_k30_min": "earliest_completion_clear_t95r_min",
                                 "ecc_k30_images": "earliest_completion_clear_images95r",
                                 "fcfs_images": "fcfs_unbudgeted_clear_images95r",
                                 "reachable": "reachable_clear_tiles"}.items():
                x, y = r[mine], ref.loc[d, theirs]
                if (x is None) != pd.isna(y) or (x is not None and not np.isclose(float(x), float(y))):
                    bad.append((d, mine, x, None if pd.isna(y) else float(y)))
        rows.append(r)
    if bad:
        raise SystemExit(f"IDENTITY CHECK FAILED for {run_dir} (first 10): {bad[:10]}")
    out = pd.DataFrame(rows)
    ok = out.dropna(subset=["floor_min"])
    summ = {"run_dir": os.path.basename(os.path.normpath(run_dir)), "onsets": int(len(out)), "evaluable": int(len(ok)),
            "margin_min": margin, "identity_check": "passed" if 30 in caps else "not run",
            "images": int(len(cells)), "images_with_repeated_cells": repeated_cells(cells),
            "floor_h_median": float(ok["floor_min"].median()) / 60 if len(ok) else None,
            "fcfs_images_mean": float(ok["fcfs_images"].mean()) if len(ok) else None, "caps": {}}
    for k in caps:
        t, n = ok[f"ecc_{cap_name(k)}_min"], ok[f"ecc_{cap_name(k)}_images"]
        done = t.notna()
        summ["caps"][cap_name(k)] = {
            "completes_pct": 100.0 * float(done.mean()) if len(ok) else None,
            "within_margin_of_floor_pct": 100.0 * float((done & (t - ok["floor_min"] <= margin + 1e-9)).mean()) if len(ok) else None,
            "delay_over_floor_min_median": float((t - ok["floor_min"])[done].median()) if done.any() else None,
            "images_mean": float(n[done].mean()) if done.any() else None,
            "images_max": int(n[done].max()) if done.any() else None,
            "fewer_images_than_fcfs_pct": 100.0 * float((done & (n < ok["fcfs_images"])).mean()) if len(ok) else None,
            "more_images_than_fcfs_pct": 100.0 * float((done & (n > ok["fcfs_images"])).mean()) if len(ok) else None}
    out.to_csv(out_prefix + "_onsets.csv", index=False)
    json.dump(summ, open(out_prefix + "_summary.json", "w"), indent=1)
    return summ


def run_all(root):
    batch = os.path.join(root, "ARF_batch"); out = os.path.join(batch, "paper2_a19", "replay"); os.makedirs(out, exist_ok=True)
    runs = []
    for f in INDEXES:
        p = os.path.join(batch, f)
        if os.path.exists(p):
            d = pd.read_csv(p)
            runs += [(f.replace("_index.csv", ""), r.out_dir) for r in d[(d.status == "done") & (d.release == "1.5.6")].itertuples()]
    assert runs, "no finished release-1.5.6 run found in the A18 or A19 indexes"
    n_ok = 0
    for i, (b, rd) in enumerate(runs, 1):
        name = os.path.basename(rd.rstrip("/"))
        pre = os.path.join(out, name)
        if os.path.exists(pre + "_summary.json"):
            print(f"[{i}/{len(runs)}] {name}: done earlier, skipped", flush=True); n_ok += 1; continue
        s = replay(rd, pre)
        c = s["caps"]
        f1 = lambda v, u="": "n/a" if v is None else f"{v:.1f}{u}"
        print(f"[{i}/{len(runs)}] {name}: identity ok; FCFS {f1(s['fcfs_images_mean'])} images; ECC no cap "
              f"{f1(c['nocap']['images_mean'])}; completes at k=30 {f1(c['k30']['completes_pct'], '%')}, "
              f"k=60 {f1(c['k60']['completes_pct'], '%')}, k=120 {f1(c['k120']['completes_pct'], '%')}; "
              f"images with repeated cells: {s['images_with_repeated_cells']}", flush=True)
        n_ok += 1
    print(f"ECC cap replay (A19): {n_ok}/{len(runs)} runs done. Files: {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?"); ap.add_argument("out_prefix", nargs="?")
    ap.add_argument("--all", action="store_true"); ap.add_argument("--root", default="/content/drive/MyDrive")
    ap.add_argument("--onsets", type=int, default=None, help="test only: first N onsets")
    a = ap.parse_args(argv)
    if a.all:
        return run_all(a.root)
    assert a.run_dir and a.out_prefix, "give RUN_DIR and OUT_PREFIX, or --all"
    print(json.dumps(replay(a.run_dir, a.out_prefix, a.onsets), indent=1))


if __name__ == "__main__":
    main()
