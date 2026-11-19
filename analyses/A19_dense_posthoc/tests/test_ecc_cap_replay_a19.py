"""Test of arf_ecc_cap_replay_a19.py against the A14 reference function itself (arf_published_selectors.py:
reference(), the A13 port of Cell 7C, equal to the pipeline at every onset of 24 real windows).
Synthetic run folders built to provoke every tie: equal gains, equal delivery times, repeated cells inside an
image, undelivered images (no wait), images of zero utility, sparse and dense fleets, needs above 30 images.
Checks: at every onset and k = 30, 60, 120 and without a cap, floor, FCFS images, ECC time and ECC images equal
A14's; the script's own identity check passes on a folder whose arf_onset_robustness.csv was written from A14,
and stops when one value of that file is changed; without a cap ECC completes at the floor; a larger cap is
never later; --all is resumable and reads only finished 1.5.6 runs.
Usage: test_ecc_cap_replay_a19.py SCRIPT A14_SCRIPT"""
import sys, os, json, tempfile, runpy, math, io, contextlib
import numpy as np, pandas as pd, h3

S, A14 = sys.argv[1:3]
RES = []


def check(c, m):
    RES.append(bool(c)); print(("PASS " if c else "FAIL ") + m, flush=True)


M = runpy.run_path(S, run_name="replay")
R = runpy.run_path(A14, run_name="a14")
base = h3.latlng_to_cell(-21.5, 23.5, 7)
POOL = sorted(h3.grid_disk(base, 14))                      # 631 valid resolution-7 cells


def make_run(root, name, seed, n_sat, n_img, width, dense):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_img):
        sat = f"STARLINK-{rng.integers(1000, 1000 + n_sat)}"
        t = int(rng.integers(0, 2880 // 5)) * 5                               # equal times happen
        start = int(rng.integers(0, len(POOL) - width)) if not dense else int(rng.integers(0, 6)) * (width // 2)
        idx = list(range(start, min(start + width, len(POOL))))
        if dense and rng.random() < 0.3:
            idx += idx[:3]                                                     # repeated cells inside one image (run "ties" only)
        wait = float(rng.choice([15.0, 30.0, 30.0, 94.0])) if rng.random() > 0.08 else np.nan
        dlk = 1.0 if rng.random() > 0.05 else 0.0                              # zero utility: never counted
        for j in idx:
            rows.append((sat, t, POOL[j], float(rng.choice([0.0, 5.0, 10.0])), 20.0, dlk, wait))
    d = os.path.join(root, name); os.makedirs(d)
    pd.DataFrame(rows, columns=["sat", "t_min", "tile", "A_nadir_deg", "theta_max_deg", "D_downlink", "T_wait_min"]
                 ).sample(frac=1.0, random_state=seed).to_csv(os.path.join(d, "arf_refined_events.csv"), index=False)
    json.dump({"t_max_wait_min": 94.0}, open(os.path.join(d, "arf_cell2_summary.json"), "w"))
    return d


def a14_rows(run_dir, k):
    keys, cells, wait, n_tiles, _ = R["load"](run_dir)
    out = {}
    for d in range(0, 1441, 20):
        sub = [e for e in range(len(keys)) if d <= keys[e][1] < d + 1440]
        if not sub:
            continue
        mask = np.zeros(n_tiles, bool)
        for e in sub:
            mask[cells[e]] = True
        reach = int(mask.sum())
        if reach == 0:
            continue
        need = int(math.ceil(95.0 / 100.0 * reach - 1e-9))
        dl = np.array([keys[e][1] - d + wait[e] for e in sub])
        order = np.argsort(dl, kind="stable"); order = order[np.isfinite(dl[order])]
        out[d] = dict(R["reference"](sub, order, dl[order], cells, keys, wait, n_tiles, d, need, reach, k), reachable=reach)
    return out


root = tempfile.mkdtemp(prefix="a19r_")
CASES = [("sparse", 1, 4, 30, 40, False), ("medium", 2, 40, 400, 25, False), ("ties", 3, 60, 500, 60, True),
         ("narrow", 4, 200, 2500, 6, False), ("narrow2", 5, 300, 4000, 4, False)]
same = {30: True, 60: True, 120: True, None: True}; n_on = 0; n_clean = 0; nocap_at_floor = True; monotone = True; over30 = 0
for name, seed, n_sat, n_img, width, dense in CASES:
    rd = make_run(root, name, seed, n_sat, n_img, width, dense)
    t_img, wait, cells, n_tiles = M["load"](rd)
    mine = {d: M["onset"](d, t_img, wait, cells, n_tiles) for d in range(0, 1441, 20)}
    for k in (30, 60, 120, None):
        ref = a14_rows(rd, 10 ** 9 if k is None else k)
        tag = M["cap_name"](k)
        for d, r in ref.items():
            m = mine[d]
            n_on += k == 30
            eq = lambda x, y: (x is None and y is None) or (x is not None and y is not None and np.isclose(float(x), float(y)))
            if not (eq(m["floor_min"], r["floor_min"]) and eq(m["fcfs_images"], r["fcfs_images"]) and m["reachable"] == r["reachable"]
                    and eq(m[f"ecc_{tag}_min"], r["ecc_min"]) and eq(m[f"ecc_{tag}_images"], r["ecc_images"])):
                same[k] = False
        if set(ref) != {d for d, m in mine.items() if m is not None}:
            same[k] = False
    for d, m in mine.items():
        if m is None or m["floor_min"] is None:
            continue
        if M["repeated_cells"](cells) == 0:                    # exact only without repeated cells
            n_clean += 1
            nocap_at_floor &= bool(np.isclose(m["ecc_nocap_min"], m["floor_min"]))
        over30 += m["ecc_nocap_images"] > 30
        ts = [m[f"ecc_{M['cap_name'](k)}_min"] for k in (30, 60, 120, None)]
        ts = [np.inf if t is None else t for t in ts]
        monotone &= all(ts[i] >= ts[i + 1] - 1e-9 for i in range(3))
    # the run's own onset file, written from A14 at k = 30
    ref = a14_rows(rd, 30)
    rows = [{"onset_shift_min": d, "best_possible_clear_t95r_min": r["floor_min"], "earliest_completion_clear_t95r_min": r["ecc_min"],
             "earliest_completion_clear_images95r": r["ecc_images"], "fcfs_unbudgeted_clear_images95r": r["fcfs_images"],
             "reachable_clear_tiles": r["reachable"]} for d, r in ref.items()]
    pd.DataFrame(rows).to_csv(os.path.join(rd, "arf_onset_robustness.csv"), index=False)
for k in (30, 60, 120, None):
    check(same[k], f"{'no cap' if k is None else 'k = ' + str(k)}: floor, FCFS, ECC time and ECC images equal A14 at every onset")
check(n_on >= 300, f"{n_on} onsets compared in {len(CASES)} synthetic runs")
check(over30 > 50, f"{over30} onsets need more than 30 images without a cap (the cap binds in the test data)")
check(nocap_at_floor and n_clean > 50, f"without a cap ECC completes at the floor ({n_clean} onsets of runs without repeated cells)")
check(monotone, "a larger cap is never later")

# the script on folders, through --all
batch = os.path.join(root, "ARF_batch"); os.makedirs(batch)
idx = [{"run_id": n, "status": "done", "release": "1.5.6", "out_dir": os.path.join(root, n)} for n, *_ in CASES[:3]]
idx.append({"run_id": "old", "status": "done", "release": "1.5.5", "out_dir": os.path.join(root, "narrow")})
idx.append({"run_id": "failed", "status": "error", "release": "1.5.6", "out_dir": os.path.join(root, "narrow2")})
pd.DataFrame(idx).to_csv(os.path.join(batch, "paper2_v156_a18_core_index.csv"), index=False)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    out = M["main"](["--all", "--root", root])
got = sorted(f for f in os.listdir(out) if f.endswith("_summary.json"))
check(got == ["medium_summary.json", "sparse_summary.json", "ties_summary.json"] and "3/3 runs done" in buf.getvalue(),
      "--all: the three finished 1.5.6 runs replayed; the 1.5.5 run and the failed run left out")
s = json.load(open(os.path.join(out, "medium_summary.json")))
check(s["identity_check"] == "passed" and s["caps"]["nocap"]["completes_pct"] == 100.0, "identity check passed; no-cap ECC completes at every evaluable onset")
check(s["images_with_repeated_cells"] == 0 and json.load(open(os.path.join(out, "ties_summary.json")))["images_with_repeated_cells"] > 0,
      "repeated cells inside an image are counted and reported per run")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    M["main"](["--all", "--root", root])
check(buf.getvalue().count("done earlier, skipped") == 3, "--all is resumable")
rd = os.path.join(root, "narrow")
o = pd.read_csv(os.path.join(rd, "arf_onset_robustness.csv"))
j = o["earliest_completion_clear_images95r"].first_valid_index() if o["earliest_completion_clear_images95r"].notna().any() else None
col = "earliest_completion_clear_images95r" if j is not None else "fcfs_unbudgeted_clear_images95r"
j = o[col].first_valid_index()
o.loc[j, col] += 1; o.to_csv(os.path.join(rd, "arf_onset_robustness.csv"), index=False)
try:
    with contextlib.redirect_stdout(io.StringIO()):
        M["replay"](rd, os.path.join(root, "x"))
    stopped = False
except SystemExit as e:
    stopped = "IDENTITY CHECK FAILED" in str(e)
check(stopped, "one changed value in the run's onset file stops the replay")
print(f"{sum(RES)}/{len(RES)} passed"); sys.exit(0 if all(RES) else 1)
