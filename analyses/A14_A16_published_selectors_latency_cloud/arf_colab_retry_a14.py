# =============================================================================
# ARF A14 RETRY (Colab): reruns ONLY the published-selector replay (A14)
# =============================================================================
# The first A14-A16 cell installed everything and ran A16a, but A14 stopped on
# every window. Paste this into a NEW cell and run it. It
#   1. installs h3 4.5.0 if the runtime lacks it (A14 needs h3; A16a does not);
#   2. runs ARF_code/arf_published_selectors.py READ-ONLY on every primary-design
#      run it finds (8 dry, 8 wet A5 p = 0.35, and the 8 A15 windows once run);
#   3. writes to a NEW folder MyDrive/ARF_A14_results_<time> and a zip beside it.
# If a window still fails, the last lines of its log are printed here.
# =============================================================================
import os, sys, json, hashlib, zipfile, subprocess, datetime, importlib.util

ROOT = "/content/drive/MyDrive"
SHA_A14 = "86e67b9a8f6237668e7ad6ec8b3f6877910395cfee452c848ddcfa7dc2b96388"
DRY = ["2026-08-03", "2026-08-10", "2026-08-17", "2026-08-24", "2026-08-31", "2026-09-07", "2026-09-14", "2026-09-21"]
WET = ["2026-01-05", "2026-01-12", "2026-01-19", "2026-01-26", "2026-02-02", "2026-02-09", "2026-02-16", "2026-02-23"]
ADDED = ["2026-07-06", "2026-07-13", "2026-07-20", "2026-07-27", "2026-09-28", "2026-10-05", "2026-10-12", "2026-10-19"]
NEED = ("arf_refined_events.csv", "arf_onset_robustness.csv", "arf_mosaic_tiles.csv", "arf_cell2_summary.json")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run_folder(path):
    if not os.path.isdir(path):
        return None
    hits = [r for r, _, fs in os.walk(path) if all(n in fs for n in NEED)]
    assert len(hits) == 1, f"{path}: expected one run folder, found {len(hits)}"
    return hits[0]


def ensure_h3():
    ok = subprocess.run([sys.executable, "-I", "-c", "import h3; assert h3.__version__.startswith('4.')"],
                        capture_output=True).returncode == 0
    if not ok:
        print("h3 missing in this runtime: installing h3==4.5.0")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "h3==4.5.0"], check=True)
    v = subprocess.run([sys.executable, "-I", "-c", "import h3; print(h3.__version__)"], capture_output=True, text=True)
    assert v.returncode == 0, "h3 still not importable: " + v.stderr[-300:]
    print(f"h3 {v.stdout.strip()} available")


def main(root=ROOT, install=True):
    if root.startswith("/content/drive") and not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    if install:
        ensure_h3()
    script = os.path.join(root, "ARF_code", "arf_published_selectors.py")
    assert sha(script) == SHA_A14, "ARF_code/arf_published_selectors.py is not the registered A14 script"
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(root, f"ARF_A14_results_{stamp}")
    os.makedirs(out)
    runs = [("dry", d, os.path.join(root, "ARF_Outputs_ckgr_okavango_earth-resources_planet-conservative_from"
                                     + d.replace("-", ""))) for d in DRY]
    runs += [("wet", d, os.path.join(root, "ARF_wetseason_A5", "p035", d)) for d in WET]
    runs += [("dry_added", d, os.path.join(root, "ARF_dryext_A15", "p035", d)) for d in ADDED]
    log = []
    for season, d, path in runs:
        rd = run_folder(path)
        if rd is None:
            if season == "dry_added":
                continue
            raise AssertionError(f"{path}: run folder missing")
        tag = f"{season}_{d}"
        r = subprocess.run([sys.executable, "-I", script, rd, os.path.join(out, tag + "_sel")], capture_output=True, text=True)
        open(os.path.join(out, f"{tag}_A14.log"), "w").write(r.stdout + r.stderr)
        status = "ok" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
        print(f"{tag} A14: {status}", flush=True)
        if r.returncode:
            print("   " + "\n   ".join((r.stdout + r.stderr).strip().splitlines()[-6:]))
        log.append({"season": season, "window": d, "run_dir": rd, "A14": status})
    json.dump(log, open(os.path.join(out, "run_log.json"), "w"), indent=1)
    zpath = os.path.join(root, f"ARF_A14_results_{stamp}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(out)):
            z.write(os.path.join(out, f), f)
    n_ok = sum(x["A14"] == "ok" for x in log)
    print(f"\nDone: {n_ok} of {len(log)} windows. Upload {zpath} to the chat.")
    return zpath


if __name__ == "__main__" or "google.colab" in sys.modules:
    main()
