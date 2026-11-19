# =============================================================================
# ARF PAPER 2 LAUNCH CELL (Colab): amendment A17, release 1.5.5
# =============================================================================
# Before running: the A14-A16 package must be installed (its cell prints "installed:
# protocol with A14 to A16"). Upload ARF_paper2_A17.zip to MyDrive (do not unzip it
# there). Colab secrets SPACETRACK_USERNAME and SPACETRACK_PASSWORD as before.
# Fresh runtime, ONE notebook, tab kept open. Set STAGE and run:
#   "smoke"        N = 5 and 20, window of 3 August (a few minutes). Run this FIRST
#                  and send me its output.
#   "core"         58 runs (32.5 km, sun 10 deg; N = 5 to 2000 in eight windows,
#                  4000 in two). Resumable: if Colab stops, run the cell again.
#   "sensitivity"  30 runs (8 and 185 km; 32.5 km at sun 5 deg)
#   "summary"      tables, saturation fits and figure from the finished runs
# Installs (never overwriting a different file): the A17 protocol (A16 kept as
# arf_confirmatory_protocol.a16.json), algorithmic_reconstruction_framework_v155.py,
# arf_batch_runner_v154.py, the A17 runner and summary. Release 1.5.3 and all
# earlier run folders are left untouched.
# =============================================================================
STAGE = "smoke"
ROOT = "/content/drive/MyDrive"
ZIP_NAME = "ARF_paper2_A17.zip"
SHA_153 = "14426c89c916ffcaac032cb94ec186ef7a7f8011748044c99f2386585d917950"
FILES = ("algorithmic_reconstruction_framework_v155.py", "arf_batch_runner_v154.py",
         "arf_paper2_runner_a17.py", "arf_paper2_summary_a17.py")

import os, sys, json, shutil, hashlib, zipfile, runpy


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def install(root=ROOT, work="/content/arf_paper2_a17"):
    code = os.path.join(root, "ARF_code")
    shutil.rmtree(work, ignore_errors=True)
    with zipfile.ZipFile(os.path.join(root, ZIP_NAME)) as z:
        for n in z.namelist():
            assert not n.startswith("/") and ".." not in n.split("/"), f"unsafe path in zip: {n}"
        z.extractall(work)
    for line in open(os.path.join(work, "SHA256SUMS.txt")).read().splitlines():
        h, n = line.split(maxsplit=1)
        assert sha(os.path.join(work, n.strip())) == h, f"{n}: checksum mismatch (damaged upload?)"
    assert sha(os.path.join(code, "algorithmic_reconstruction_framework.py")) == SHA_153, "ARF_code: release 1.5.3 changed"
    cur_p = os.path.join(code, "arf_confirmatory_protocol.json")
    cur, new = json.load(open(cur_p)), json.load(open(os.path.join(work, "arf_confirmatory_protocol.json")))
    if cur == new:
        print("the A17 protocol is already installed")
    else:
        ids = [a["id"] for a in cur["amendments"]]
        assert ids == [f"A{i}" for i in range(1, 17)], \
            f"ARF_code protocol has {ids[-1] if ids else 'none'}; install the A14-A16 package first (it must end at A16)"
        strip = lambda d: {k: v for k, v in d.items() if k not in ("amendments", "amended_utc")}
        assert strip(cur) == strip(new) and new["amendments"][:16] == cur["amendments"] and \
            new["amendments"][-1]["id"] == "A17", "the zip's protocol is not the installed A16 protocol with A17 appended"
        keep = os.path.join(code, "arf_confirmatory_protocol.a16.json")
        assert not os.path.exists(keep), f"{keep} exists already; not overwritten"
        shutil.copy2(cur_p, keep)
        shutil.copy2(os.path.join(work, "arf_confirmatory_protocol.json"), cur_p)
        print("installed: protocol with A17 (A16 protocol kept as arf_confirmatory_protocol.a16.json)")
    for f in FILES:
        dst = os.path.join(code, f)
        if os.path.exists(dst):
            assert sha(dst) == sha(os.path.join(work, f)), f"ARF_code/{f} differs from the package; not overwritten"
        else:
            shutil.copy2(os.path.join(work, f), dst)
    print("in ARF_code: " + ", ".join(FILES))
    return code


def secrets_present():
    try:
        from google.colab import userdata
        return all(userdata.get(k) for k in ("SPACETRACK_USERNAME", "SPACETRACK_PASSWORD"))
    except Exception:
        return False


if __name__ == "__main__" or "google.colab" in sys.modules:
    if not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    code = install()
    if STAGE == "summary":
        runpy.run_path(os.path.join(code, "arf_paper2_summary_a17.py"), run_name="__main__")
    else:
        assert STAGE in ("smoke", "core", "sensitivity"), f"unknown STAGE {STAGE!r}"
        assert secrets_present(), "Colab secrets SPACETRACK_USERNAME and SPACETRACK_PASSWORD are missing or not shared with this notebook"
        R = runpy.run_path(os.path.join(code, "arf_paper2_runner_a17.py"), run_name="arf_paper2_runner_a17")
        R["main"]([STAGE])
