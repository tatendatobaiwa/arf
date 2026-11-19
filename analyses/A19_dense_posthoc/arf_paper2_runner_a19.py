# =============================================================================
# ARF PAPER 2 RUNNER, amendment A19 part (d): N = 8000, release 1.5.6
# =============================================================================
# The registered saturation fit of A17 failed (the floor is not one curve over
# N = 5 to 4000: below N = 500 the reachable cells themselves grow with N). Where
# the whole region is reachable the floor still falls at N = 4000 (3.2 h; 1.24 h
# gained from N = 2000 in the two matched windows). A19 (d) adds the only direct
# test of where it levels off: N = 8000, 32.5 km swath, sun 10 deg, in the two
# windows that have N = 4000 (3 August and 7 September 2026). Everything else as
# A17/A18: region CKGR + Okavango, element sets of Space-Track gp_history (already
# cached), nested seeded fleets, cloud sweep off, release 1.5.6 unchanged.
#
# STAGE  n8000   2 runs, one batch (paper2_v156_a19_n8000), resumable. Each run is
#                expected to take 4 to 5 h in a fresh runtime (N = 2000: 49 to 67
#                min; N = 4000: 113 and 123 min). Memory at N = 8000 is NOT tested:
#                if the session crashes, stop and report; do not loop it.
# Use (Colab): through arf_colab_launch_paper2_a19.py.
# =============================================================================
import os
import sys
import json
import hashlib
import runpy
import argparse

ROOT = "/content/drive/MyDrive"
CODE = f"{ROOT}/ARF_code"
PIPELINE = f"{CODE}/algorithmic_reconstruction_framework_v156.py"
BATCH_RUNNER = f"{CODE}/arf_batch_runner_v154.py"
SHA_156 = "240698a9c52c6f4a17bf16ae489fdea5cba721e0ec0795a9973c1a368b2532b3"   # release 1.5.6
SHA_RUNNER_154 = "bb69e27fa5776630915cbe9c7d3e0081ef0203a5365c5471aa86ab293c4ae6e0"
WINDOWS = ("2026-08-03T00:00:00", "2026-09-07T00:00:00")      # the windows that have N = 4000
BASE = {"active_roi": "ckgr_okavango", "fleet": "starlink",
        "cloud_sweep_lengths_km": [], "cloud_sweep_times_h": []}
PRIMARY = {"proxy_swath_km": 32.5, "proxy_min_sun_elev_deg": 10.0}
N_A19 = 8000
BATCHES = {"n8000": "paper2_v156_a19_n8000"}


def cfg(n, w, case):
    return dict(BASE, **case, active_n=int(n), analysis_epoch=w)


def build_runs(stage):
    if stage == "n8000":
        return [cfg(N_A19, w, PRIMARY) for w in WINDOWS]
    raise ValueError(f"unknown stage {stage!r}")


def tag_of(c):
    return f"_sun{c['proxy_min_sun_elev_deg']:g}_sw{c['proxy_swath_km']:g}km"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv=None, pipeline=PIPELINE, batch_runner=BATCH_RUNNER, batch_dir=f"{ROOT}/ARF_batch", sha_156=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=sorted(BATCHES))
    a = ap.parse_args(argv)
    assert sha(pipeline) == (sha_156 or SHA_156), f"{pipeline} is not release 1.5.6"
    assert sha(batch_runner) == SHA_RUNNER_154, f"{batch_runner} is not the 1.5.4 batch runner"
    proto = json.load(open(os.path.join(os.path.dirname(pipeline), "arf_confirmatory_protocol.json")))
    assert any(x["id"] == "A19" for x in proto.get("amendments", [])), "protocol without amendment A19: install it first"
    R = runpy.run_path(batch_runner, run_name="arf_batch_runner_v154")
    runs = build_runs(a.stage)
    print(f"Paper 2 (A19), stage {a.stage}: {len(runs)} runs")

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
