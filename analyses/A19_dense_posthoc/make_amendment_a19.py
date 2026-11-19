"""Append amendment A19 (Paper 2, post hoc: ECC cap replay, restricted saturation fit, N = 8000) to the A18
protocol. Refuses any other input.
Usage: make_amendment_a19.py IN_A18_PROTOCOL OUT_PROTOCOL SCRIPT_DIR SHA_A18"""
import sys, os, json, hashlib, datetime

SRC, DST, D, SHA_A18 = sys.argv[1:5]
raw = open(SRC, "rb").read()
assert hashlib.sha256(raw).hexdigest() == SHA_A18, "input is not the A18 protocol"
p = json.loads(raw)
assert [a["id"] for a in p["amendments"]] == [f"A{i}" for i in range(1, 19)]
sha = lambda f: hashlib.sha256(open(os.path.join(D, f), "rb").read()).hexdigest()
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
p["amendments"].append({
    "id": "A19", "date_utc": now,
    "status_when_amended": ("POST HOC. Registered AFTER the 88 runs of A17/A18 (release 1.5.6; 58 core, 30 sensitivity, all "
                            "done, 10 to 15 November 2026) and after their summary had been read. Nothing of A19 had been "
                            "run when it was registered. Every A19 result is to be labelled post hoc."),
    "what_was_seen": ("(1) The saturation fit registered in A17 failed: fitted to N = 5 to 4000 it returned a limit of "
                      "0 h and N* of the order of 10^12. Cause: the floor is the time to 95% of the REACHABLE cells, and "
                      "below N = 500 the reachable share itself grows with N (9, 29, 81, 98% at N = 5, 20, 100, 250; "
                      "100% from N = 500), so the mean floor rises from 14.4 h (N = 5) to 19.4 h (N = 100) before it "
                      "falls: 15.3, 9.8, 6.0, 4.1 and 3.2 h at N = 250 to 4000. The failed fit is reported as failed. "
                      "(2) In the two windows that hold N = 4000 (3 August, 7 September) the floor fell by 1.35 and "
                      "1.13 h from N = 2000 to 4000: not yet level. (3) ECC runs with at most 30 images "
                      "(max_selection_steps, the Paper 1 setting) and its FCFS comparator without a cap. At 8 km FCFS "
                      "needs 78 to 230 images from N = 250, so ECC never completes there (8 of 8 runs); at 32.5 km ECC "
                      "uses 27 to 29 images at N = 250 and reaches the floor within the margin at 88.7% of onsets."),
    "a_ecc_cap_replay": ("Read-only replay of every finished 1.5.6 run from its own arf_refined_events.csv with the rules "
                         "of Cell 7C: ECC at k = 30 (which must equal the run's arf_onset_robustness.csv at every onset: "
                         "floor, ECC time, ECC images, FCFS images, reachable cells; the replay stops otherwise), at k = 60 "
                         "and 120, and without a cap. Without a cap ECC completes at the floor by construction when no "
                         "image lists a cell twice (reported per run); what that case measures is the number of images. "
                         "Reported per camera and N: images of FCFS and of ECC without a cap, share of onsets completed "
                         "at each cap, delay over the floor."),
    "b_restricted_fit": ("Core case (32.5 km, sun 10 deg), runs with N >= 500 only. T(N) = T_inf + a N^-b by least "
                         "squares (0 <= T_inf <= min T, 0.01 <= b <= 5), window-block bootstrap (B = 2000, seed 42). "
                         "N*_m = (a/m)^(1/b), m = 0.5 h and 1 h (A17's definition). Variants, all reported: all windows; "
                         "the two matched windows (point estimates, no interval); all windows without the largest N. "
                         "The hours gained per doubling of N on matched windows are reported beside the fit and take "
                         "precedence over it: saturation is claimed directly only if the last doubling gains less than "
                         "the margin in every matched window."),
    "d_n8000": ("New runs: N = 8000, 32.5 km, sun 10 deg, windows of 3 August and 7 September 2026 (batch "
                "paper2_v156_a19_n8000), release 1.5.6 unchanged (SHA-256 "
                + sha("algorithmic_reconstruction_framework_v156.py") + "), every other setting as A17/A18. If a run "
                "cannot finish (memory or time), that is reported and the fit is given without N = 8000."),
    "not_adopted": "N = 4000 in the six other windows was considered and not adopted.",
    "scope": "Descriptive; no hypothesis decision. Paper 1 and release 1.5.3 are not touched.",
    "script": {f: sha(f) for f in ("arf_ecc_cap_replay_a19.py", "arf_paper2_summary_a19.py", "arf_paper2_runner_a19.py",
                                   "algorithmic_reconstruction_framework_v156.py", "arf_batch_runner_v154.py",
                                   "arf_paper2_summary_a18.py")}})
p["amended_utc"] = now
open(DST, "w").write(json.dumps(p, indent=1))
print(DST, now, hashlib.sha256(open(DST, "rb").read()).hexdigest())
