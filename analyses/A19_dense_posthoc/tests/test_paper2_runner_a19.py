"""Test of arf_paper2_runner_a19.py with the REAL 1.5.4 batch runner and a stub pipeline (it writes a manifest in a
folder named as release 1.5.6 names it). Checks: the stage is exactly N = 8000 in the windows of 3 August and
7 September, 32.5 km, sun 10 deg; run identifiers differ from every A18 run; both runs done in their own folders with
N, date, sun and swath in the name; resumable; refused: a pipeline other than the pinned one, a changed batch runner,
a protocol without A19; a run that writes to a folder without its tags stops the batch; a run that fails (as on a
memory error) is recorded as an error and not as done.
Usage: test_paper2_runner_a19.py RUNNER_A19 BATCH_RUNNER_154 RUNNER_A18"""
import sys, os, json, tempfile, runpy, io, contextlib, hashlib, shutil

RUNNER, BR, R18 = sys.argv[1:4]
RES = []


def check(c, m):
    RES.append(bool(c)); print(("PASS " if c else "FAIL ") + m, flush=True)


STUB = '''__version__ = "1.5.6"
import os, json
c = ARF_CONFIG_OVERRIDES
if os.environ.get("STUB_FAIL") == "1":
    raise MemoryError("stub: out of memory")
tag = "" if os.environ.get("STUB_NOTAG") == "1" else f"_sun{c['proxy_min_sun_elev_deg']:g}_sw{c['proxy_swath_km']:g}km"
OUT_DIR = os.path.join(os.environ["STUB_ROOT"], f"ARF_Outputs_{c['active_roi']}_N{c['active_n']}_from{c['analysis_epoch'][:10].replace('-', '')}" + tag)
os.makedirs(OUT_DIR, exist_ok=True)
json.dump({"framework_version": __version__, "config_overrides": c}, open(os.path.join(OUT_DIR, "arf_reproducibility_manifest.json"), "w"))
'''
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
R = runpy.run_path(RUNNER, run_name="r19"); A18 = runpy.run_path(R18, run_name="r18")
runs = R["build_runs"]("n8000")
check(len(runs) == 2 and all(c["active_n"] == 8000 and c["proxy_swath_km"] == 32.5 and c["proxy_min_sun_elev_deg"] == 10.0
                             and c["fleet"] == "starlink" and c["active_roi"] == "ckgr_okavango" for c in runs)
      and [c["analysis_epoch"][:10] for c in runs] == ["2026-08-03", "2026-09-07"], "stage n8000: N = 8000, 32.5 km, sun 10 deg, 3 August and 7 September")
core4000 = [c for c in A18["build_runs"]("core") if c["active_n"] == 4000]
same = lambda a, b: {k: v for k, v in a.items() if k != "active_n"} == {k: v for k, v in b.items() if k != "active_n"}
check(len(core4000) == 2 and all(same(a, b) for a, b in zip(runs, core4000)), "identical to the A18 N = 4000 runs except for N")
B = runpy.run_path(BR, run_name="br")
old = {B["run_id"](c) for s in ("smoke", "core", "sensitivity") for c in A18["build_runs"](s)}
check(not ({B["run_id"](c) for c in runs} & old), "run identifiers differ from every A18 run")


def setup(proto_ids=("A18", "A19")):
    root = tempfile.mkdtemp(prefix="p2r19_"); code = os.path.join(root, "ARF_code"); os.makedirs(code)
    pipe = os.path.join(code, "algorithmic_reconstruction_framework_v156.py"); open(pipe, "w").write(STUB)
    br = os.path.join(code, "arf_batch_runner_v154.py"); shutil.copy(BR, br)
    json.dump({"amendments": [{"id": i} for i in proto_ids]}, open(os.path.join(code, "arf_confirmatory_protocol.json"), "w"))
    os.environ["STUB_ROOT"] = root
    return root, pipe, br


def go(root, pipe, br, sha_156=None):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        idx = R["main"](["n8000"], pipeline=pipe, batch_runner=br, batch_dir=os.path.join(root, "ARF_batch"),
                        sha_156=sha_156 or sha(pipe))
    return idx, buf.getvalue()


root, pipe, br = setup()
idx, txt = go(root, pipe, br)
dirs = sorted(os.path.basename(r["out_dir"]) for r in idx.values())
check("2/2 runs done" in txt and dirs == ["ARF_Outputs_ckgr_okavango_N8000_from20260803_sun10_sw32.5km",
                                           "ARF_Outputs_ckgr_okavango_N8000_from20260907_sun10_sw32.5km"], f"both runs done: {dirs}")
check(os.path.exists(os.path.join(root, "ARF_batch", "paper2_v156_a19_n8000_index.csv")), "index paper2_v156_a19_n8000_index.csv written")
idx, txt = go(root, pipe, br)
check(txt.count("done earlier, skipped") == 2, "resumable: second call skips both")


def refused(fn, text):
    try:
        fn(); return False
    except AssertionError as e:
        return text in str(e)


check(refused(lambda: go(root, pipe, br, sha_156="0" * 64), "is not release 1.5.6"), "a pipeline other than the pinned one is refused")
open(br, "a").write("# x\n")
check(refused(lambda: go(root, pipe, br), "is not the 1.5.4 batch runner"), "a changed batch runner is refused")
root, pipe, br = setup(proto_ids=("A18",))
check(refused(lambda: go(root, pipe, br), "without amendment A19"), "a protocol without A19 is refused")
root, pipe, br = setup(); os.environ["STUB_NOTAG"] = "1"
check(refused(lambda: go(root, pipe, br), "lacks its tags"), "a run writing to a folder without its tags stops the batch")
os.environ.pop("STUB_NOTAG"); root, pipe, br = setup(); os.environ["STUB_FAIL"] = "1"
idx, txt = go(root, pipe, br); os.environ.pop("STUB_FAIL")
check("0/2 runs done" in txt and all(r["status"] == "error" and "MemoryError" in r["error"] for r in idx.values()),
      "a run that fails on memory is recorded as an error, not as done")
idx, txt = go(root, pipe, br)
check("2/2 runs done" in txt, "and is run again on the next call")
print(f"{sum(RES)}/{len(RES)} passed"); sys.exit(0 if all(RES) else 1)
