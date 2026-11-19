# =============================================================================
# ARF ADDED DRY-SEASON WINDOWS (protocol amendment A15, REGISTERED REPLICATION):
# release 1.5.3 unchanged, primary design, eight further weekly 48-h windows of
# the 2026 dry season (Mondays 6 to 27 July, 28 September to 19 October),
# run retrospectively with the protocol's settings (p = 0.35)
# =============================================================================
# 8 runs, about 4 min each. Registered after the confirmatory results were read
# and before any of these windows was run: H1a, H1b and H2 are decided on these
# eight windows alone by the protocol's rules (Bonferroni alpha/3).
#
# Same isolation as the A4 and A5 runners (it reuses arf_sensitivity_runner_a4.py,
# checked by SHA-256): each run executes with google.colab hidden, in its own
# folder MyDrive/ARF_dryext_A15/<variant>/<window>; the confirmatory folders,
# the A4 sensitivity folders and the A5 folders are recorded before and checked after every
# run, and the batch stops at the first change.
# Before anything runs it checks: release 1.5.3; the element sets of all eight
# windows in ARF_batch/shared_cache (run arf_colab_prefetch_dryext_a15.py first).
#
# Use (Colab):  !python /content/drive/MyDrive/ARF_code/arf_dryext_runner_a15.py
# Resumable: completed runs are skipped.
# =============================================================================
import os
import sys
import glob
import runpy
import hashlib

os.environ["MPLBACKEND"] = "Agg"

ROOT = "/content/drive/MyDrive"
CODE = f"{ROOT}/ARF_code"
BATCH = f"{ROOT}/ARF_batch"
WET_ROOT = f"{ROOT}/ARF_dryext_A15"
BATCH_NAME = "dryext_v153_a15"
SHA_153 = "14426c89c916ffcaac032cb94ec186ef7a7f8011748044c99f2386585d917950"
SHA_A4_RUNNER = "02195261d337d5da93cd72950f1a9811c0da55a14861bd2f144b503b25df2d42"   # the A4 runner of the A4 package (arf_sensitivity_runner_a4.py)
WINDOWS = ("2026-07-06T00:00:00", "2026-07-13T00:00:00", "2026-07-20T00:00:00", "2026-07-27T00:00:00",
           "2026-09-28T00:00:00", "2026-10-05T00:00:00", "2026-10-12T00:00:00", "2026-10-19T00:00:00")
VARIANTS = {"p035": {}}
CLIMATOLOGY_MONTHS = ()
PRIMARY_DESIGN = {"active_roi": "ckgr_okavango", "fleet": "earth-resources+planet", "planet_swath_case": "conservative"}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_runs(a4, windows=WINDOWS, order=("p035",)):
    assert a4["PRIMARY_DESIGN"] == PRIMARY_DESIGN, "A4 runner's primary design differs"
    return [(v, w, {**PRIMARY_DESIGN, "analysis_epoch": w, **VARIANTS[v]}) for v in order for w in windows]


def preflight(a4, root, code, batch, windows=WINDOWS, climatology_dir=None):
    pf = runpy.run_path(os.path.join(code, "arf_colab_prefetch_dryext_a15.py"), run_name="arf_colab_prefetch_dryext_a15")
    missing = [os.path.basename(pf["history_path"](os.path.join(batch, "shared_cache"),
                                                   {**a4["PRIMARY_DESIGN"], "analysis_epoch": w}))
               for w in windows]
    missing = [m for m in missing if not os.path.exists(os.path.join(batch, "shared_cache", m))]
    assert not missing, f"element sets not cached for {len(missing)} windows (run arf_colab_prefetch_dryext_a15.py): {missing[:3]}"
    cdir = climatology_dir or os.path.join(root, "ARF_climatology")
    lack = [m for m in CLIMATOLOGY_MONTHS if not os.path.exists(os.path.join(cdir, f"MODCF_monthlymean_{m:02d}.tif"))]
    assert not lack, (f"EarthEnv files missing for month(s) {lack} in {cdir}: download MODCF_monthlymean_01.tif and "
                      "_02.tif from https://www.earthenv.org/cloud")


def main(root=ROOT, code=CODE, batch=BATCH, wet_root=WET_ROOT, pipeline=None, sha_a4=None):
    pipeline = pipeline or os.path.join(code, "algorithmic_reconstruction_framework.py")
    assert sha(pipeline) == SHA_153, f"{pipeline} is not release 1.5.3"
    a4p = os.path.join(code, "arf_sensitivity_runner_a4.py")
    want = sha_a4 if sha_a4 is not None else SHA_A4_RUNNER
    assert want is None or sha(a4p) == want, "ARF_code/arf_sensitivity_runner_a4.py differs from the A4 release"
    a4 = runpy.run_path(a4p, run_name="arf_sensitivity_runner_a4")
    R = runpy.run_path(os.path.join(code, "arf_batch_runner.py"), run_name="arf_batch_runner")
    preflight(a4, root, code, batch)
    runs = build_runs(a4)
    protected = a4["confirmatory_folders"](batch) + sorted(
        d for pat in ("ARF_sensitivity_A4", "ARF_wetseason_A5")
        for d in glob.glob(os.path.join(root, pat, "*", "*")) if os.path.isdir(d))
    assert protected, "no confirmatory run folder found"
    print(f"A15 added dry-season windows: {len(runs)} runs of release 1.5.3; {len(protected)} earlier folders protected")
    before = a4["snapshot"](protected)
    for i, (v, w, cfg) in enumerate(runs, 1):
        work = os.path.join(wet_root, v, w[:10])
        print(f"--- [{i}/{len(runs)}] {v} {w[:10]} -> {work}", flush=True)
        with a4["outside_colab_folder"](work):
            R["run_batch"]([cfg], BATCH_NAME, pipeline_path=pipeline, batch_dir=batch,
                           shared_cache_dir=os.path.join(batch, "shared_cache"))
        bad = a4["changed"](before, protected)
        assert not bad, f"STOPPED: an earlier run folder changed during {v} {w}: {bad[:5]}"
    print(f"Done. Index: {os.path.join(batch, BATCH_NAME + '_index.csv')}; earlier folders unchanged.")


if __name__ == "__main__":
    main()
