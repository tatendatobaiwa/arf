# =============================================================================
# ARF PAPER 2 RUNNER (protocol amendment A17): Starlink orbits as a hypothetical
# dense Earth-observation constellation, release 1.5.5
# =============================================================================
# Every proxy satellite carries an ASSUMED nadir camera of ground swath
# proxy_swath_km, imaging only with the Sun at least proxy_min_sun_elev_deg above
# the imaged point. Region CKGR + Okavango; the eight protocol windows (Mondays
# 3 August to 21 September 2026); element sets archived by Space-Track
# (gp_history) for each window; cloud-sensitivity sweep off; every other setting
# as in the protocol. Fleet sizes are nested: a fleet of N is the first N
# satellites of one seeded random order per window (release 1.5.4 behaviour).
#
# STAGES (each a batch of its own, resumable; a finished run is skipped)
#   smoke        N = 5 and 20 in the window of 3 August (minutes): checks the live
#                Space-Track path and the folder names before anything long
#   core         32.5 km, sun 10 deg: N = 5, 20, 100, 250, 500, 1000 in all eight
#                windows, then N = 2000 in all eight, then N = 4000 in the windows
#                of 3 August and 7 September (largest runs last)
#   sensitivity  8 km and 185 km at sun 10 deg, and 32.5 km at sun 5 deg:
#                N = 100, 250, 500, 1000, 2000 in the windows of 3 August and
#                7 September
# Real fleets are not touched: release 1.5.3 stays the release of record; 1.5.5
# runs from ARF_code/algorithmic_reconstruction_framework_v155.py. Every run's
# folder must carry the tags _sun<deg>_sw<km>km, which no earlier run has; the
# batch stops if a run writes anywhere else.
#
# Use (Colab): through arf_colab_launch_paper2_a17.py.
# =============================================================================
import os
import sys
import json
import hashlib
import runpy
import argparse

ROOT = "/content/drive/MyDrive"
CODE = f"{ROOT}/ARF_code"
PIPELINE = f"{CODE}/algorithmic_reconstruction_framework_v155.py"
BATCH_RUNNER = f"{CODE}/arf_batch_runner_v154.py"
SHA_155 = "5cbfdd734fd4d68cd1b72fe701f7438d34456dee53db279b2ffab66d569d2788"   # release 1.5.5
SHA_RUNNER_154 = "bb69e27fa5776630915cbe9c7d3e0081ef0203a5365c5471aa86ab293c4ae6e0"
WINDOWS = ("2026-08-03T00:00:00", "2026-08-10T00:00:00", "2026-08-17T00:00:00", "2026-08-24T00:00:00",
           "2026-08-31T00:00:00", "2026-09-07T00:00:00", "2026-09-14T00:00:00", "2026-09-21T00:00:00")
SENS_WINDOWS = ("2026-08-03T00:00:00", "2026-09-07T00:00:00")
BASE = {"active_roi": "ckgr_okavango", "fleet": "starlink",
        "cloud_sweep_lengths_km": [], "cloud_sweep_times_h": []}
PRIMARY = {"proxy_swath_km": 32.5, "proxy_min_sun_elev_deg": 10.0}
CORE_N = (5, 20, 100, 250, 500, 1000)
CORE_LARGE = 2000
CORE_XL = 4000
SENS_N = (100, 250, 500, 1000, 2000)
SENS_CASES = ({"proxy_swath_km": 8.0, "proxy_min_sun_elev_deg": 10.0},
              {"proxy_swath_km": 185.0, "proxy_min_sun_elev_deg": 10.0},
              {"proxy_swath_km": 32.5, "proxy_min_sun_elev_deg": 5.0})
BATCHES = {"smoke": "paper2_v155_a17_smoke", "core": "paper2_v155_a17_core", "sensitivity": "paper2_v155_a17_sens"}


def cfg(n, w, case):
    return dict(BASE, **case, active_n=int(n), analysis_epoch=w)


def build_runs(stage):
    if stage == "smoke":
        return [cfg(n, WINDOWS[0], PRIMARY) for n in (5, 20)]
    if stage == "core":
        runs = [cfg(n, w, PRIMARY) for w in WINDOWS for n in CORE_N]
        runs += [cfg(CORE_LARGE, w, PRIMARY) for w in WINDOWS]
        runs += [cfg(CORE_XL, w, PRIMARY) for w in SENS_WINDOWS]
        return runs
    if stage == "sensitivity":
        return [cfg(n, w, c) for c in SENS_CASES for w in SENS_WINDOWS for n in SENS_N]
    raise ValueError(f"unknown stage {stage!r}")


def tag_of(c):
    return f"_sun{c['proxy_min_sun_elev_deg']:g}_sw{c['proxy_swath_km']:g}km"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None, pipeline=PIPELINE, batch_runner=BATCH_RUNNER, batch_dir=f"{ROOT}/ARF_batch", sha_155=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(BATCHES))
    a = ap.parse_args(argv)
    assert sha(pipeline) == (sha_155 or SHA_155), f"{pipeline} is not release 1.5.5"
    assert sha(batch_runner) == SHA_RUNNER_154, f"{batch_runner} is not the 1.5.4 batch runner"
    proto = json.load(open(os.path.join(os.path.dirname(pipeline), "arf_confirmatory_protocol.json")))
    assert any(x["id"] == "A17" for x in proto.get("amendments", [])), "protocol without amendment A17: install it first"
    R = runpy.run_path(batch_runner, run_name="arf_batch_runner_v154")
    runs = build_runs(a.stage)
    print(f"Paper 2 (A17), stage {a.stage}: {len(runs)} runs")

    index = {}
    for c in runs:                                         # one run at a time, so each folder is checked at once
        idx = R["run_batch"]([c], BATCHES[a.stage], pipeline_path=pipeline, batch_dir=batch_dir,
                             shared_cache_dir=os.path.join(batch_dir, "shared_cache"))
        rec = idx[R["run_id"](c)]
        out = rec.get("out_dir", "")
        if rec.get("status") == "done":
            assert tag_of(c) in os.path.basename(out.rstrip("/")) and f"N{c['active_n']}_" in os.path.basename(out), \
                f"STOPPED: run {rec['run_id']} wrote to {out}, which lacks its tags {tag_of(c)}"
        index.update(idx)
    done = sum(r.get("status") == "done" for r in index.values())
    print(f"Paper 2 stage {a.stage}: {done}/{len(runs)} runs done")
    return index


if __name__ == "__main__":
    main()
