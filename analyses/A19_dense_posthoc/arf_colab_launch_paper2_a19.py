# =============================================================================
# ARF PAPER 2 LAUNCH CELL (Colab): amendment A19 (post hoc), release 1.5.6
# =============================================================================
# Before running: the A18 protocol must be installed (the A18 cell did it). Upload
# ARF_paper2_A19.zip to MyDrive (do not unzip it there). Set STAGE and run:
#   "replay"   ECC with the image cap lifted, recomputed from the 88 finished runs
#              (read-only; minutes; no Space-Track; resumable). Run this FIRST.
#   "summary"  restricted fit, gains per doubling and the replay table, from what
#              is finished so far (minutes). Run after "replay", and again after
#              "n8000".
#   "n8000"    N = 8000 in the windows of 3 August and 7 September (2 runs, expected
#              4 to 5 h each; fresh runtime; Space-Track secrets as before). Memory
#              at N = 8000 is not tested: if the session crashes, stop and tell me;
#              do not loop it. Resumable: a finished run is skipped.
# Installs (never overwriting a different file): the A19 protocol (A18 kept as
# arf_confirmatory_protocol.a18.json), the A19 replay, runner and summary. Release
# 1.5.6, the batch runner and the A18 summary are already there and must match.
# Nothing of the earlier runs is changed.
# =============================================================================
STAGE = "replay"
ROOT = "/content/drive/MyDrive"
ZIP_NAME = "ARF_paper2_A19.zip"
SHA_153 = "14426c89c916ffcaac032cb94ec186ef7a7f8011748044c99f2386585d917950"
FILES = ("algorithmic_reconstruction_framework_v156.py", "arf_batch_runner_v154.py", "arf_paper2_summary_a18.py",
         "arf_ecc_cap_replay_a19.py", "arf_paper2_runner_a19.py", "arf_paper2_summary_a19.py")

import os, sys, json, shutil, hashlib, zipfile, runpy


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def install(root=ROOT, work="/content/arf_paper2_a19"):
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
        print("the A19 protocol is already installed")
    else:
        ids = [a["id"] for a in cur["amendments"]]
        assert ids == [f"A{i}" for i in range(1, 19)], \
            f"ARF_code protocol has {ids[-1] if ids else 'none'}; it must end at A18 (install the A18 package first)"
        strip = lambda d: {k: v for k, v in d.items() if k not in ("amendments", "amended_utc")}
        assert strip(cur) == strip(new) and new["amendments"][:18] == cur["amendments"] and \
            new["amendments"][-1]["id"] == "A19", "the zip's protocol is not the installed A18 protocol with A19 appended"
        keep = os.path.join(code, "arf_confirmatory_protocol.a18.json")
        assert not os.path.exists(keep), f"{keep} exists already; not overwritten"
        shutil.copy2(cur_p, keep)
        shutil.copy2(os.path.join(work, "arf_confirmatory_protocol.json"), cur_p)
        print("installed: protocol with A19 (A18 protocol kept as arf_confirmatory_protocol.a18.json)")
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
    assert STAGE in ("replay", "summary", "n8000"), f"unknown STAGE {STAGE!r}"
    if STAGE == "replay":
        runpy.run_path(os.path.join(code, "arf_ecc_cap_replay_a19.py"), run_name="arf_ecc_cap_replay_a19")["main"](["--all", "--root", ROOT])
    elif STAGE == "summary":
        runpy.run_path(os.path.join(code, "arf_paper2_summary_a19.py"), run_name="arf_paper2_summary_a19")["main"]([])
    else:
        assert secrets_present(), "Colab secrets SPACETRACK_USERNAME and SPACETRACK_PASSWORD are missing or not shared with this notebook"
        R = runpy.run_path(os.path.join(code, "arf_paper2_runner_a19.py"), run_name="arf_paper2_runner_a19")
        R["main"]([STAGE])
