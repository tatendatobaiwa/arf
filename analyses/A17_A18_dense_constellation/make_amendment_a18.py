"""Append amendment A18 (Paper 2: release 1.5.6 replaces 1.5.5, which rejected Alpha-5 catalogue numbers) to the
A17 protocol. Refuses any other input.
Usage: make_amendment_a18.py IN_A17_PROTOCOL OUT_PROTOCOL SCRIPT_DIR SHA_A17"""
import sys, os, json, hashlib, datetime

SRC, DST, D, SHA_A17 = sys.argv[1:5]
raw = open(SRC, "rb").read()
assert hashlib.sha256(raw).hexdigest() == SHA_A17, "input is not the A17 protocol"
p = json.loads(raw)
assert [a["id"] for a in p["amendments"]] == [f"A{i}" for i in range(1, 18)]
sha = lambda f: hashlib.sha256(open(os.path.join(D, f), "rb").read()).hexdigest()
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
p["amendments"].append({
    "id": "A18", "date_utc": now,
    "status_when_amended": ("CORRECTION OF A17'S RELEASE, REGISTERED BEFORE ANY OF ITS CORE OR SENSITIVITY RUNS. Only "
                            "the A17 smoke stage had run (release 1.5.5; N = 5 and 20, window of 3 August 2026, 10 "
                            "November 2026). Its log reported 104 of 10,881 Starlink element sets as malformed. A "
                            "read-only check of the cached Space-Track files, with the parsing steps of 1.5.5, found "
                            "every rejected record to carry an Alpha-5 catalogue number, and the number rising with "
                            "the window: 104, 152, 176, 229, 309, 365, 418 and 424 of 10,881 to 11,135 records "
                            "(3 August to 21 September 2026). The two smoke runs are superseded and not reported; their "
                            "folders are kept. No other A17 run exists."),
    "reason": ("Catalogue numbers of 100000 and above are written in the 5-character TLE field as Alpha-5: a leading "
               "capital letter, A = 10 to Z = 33 with I and O omitted, so that 339999 => Z9999 (Space-Track "
               "documentation, section Alpha-5, read 10 November 2026). Release 1.5.5 read the field as an integer, so "
               "every Starlink satellite catalogued since number 100000 was reached was rejected, "
               "and the samples were drawn from the older satellites only, a bias that grows across the eight windows."),
    "change": ("Release 1.5.6 decodes Alpha-5 in the operational filter and in the pin file (Cell 2) and names its pin "
               "files _opsample_a5 so that no 1.5.5 pin is reused. Nothing else changes; Orekit 13.1.9 already decodes "
               "Alpha-5 in TLE(line1, line2). Real fleets are unaffected: their catalogue numbers come from the OMM "
               "field NORAD_CAT_ID, not from the TLE line, in release 1.5.3 as in 1.5.6."),
    "checks": ("Sandbox, no network: the decoding agrees with Space-Track's examples and, for all 24 letters, with "
               "Orekit's own decoding of checksummed lines; on synthetic Starlink-like element sets numbered from 99000 "
               "(9,085 of 10,084 Alpha-5), 1.5.5 rejects all 9,085 and 1.5.6 none, the two select identically on 5-digit "
               "records, and Cells 0 to 9 of 1.5.6 run without error at N = 20 and 100 with 16 of 20 and 88 of 100 "
               "Alpha-5 satellites in the fleet."),
    "design": ("A17's design, analysis and scope unchanged in every other respect: release 1.5.6 (SHA-256 "
               + sha("algorithmic_reconstruction_framework_v156.py") + ") in place of 1.5.5; batches "
               "paper2_v156_a18_smoke, paper2_v156_a18_core and paper2_v156_a18_sens; summary written to "
               "ARF_batch/paper2_a18. Where a 1.5.5 smoke folder already holds a complete run, 1.5.6 writes to the same "
               "name with the suffix _v156 (Cell 0 rule since 1.5.2)."),
    "script": {f: sha(f) for f in ("arf_paper2_runner_a18.py", "arf_paper2_summary_a18.py",
                                   "algorithmic_reconstruction_framework_v156.py", "arf_batch_runner_v154.py")}})
p["amended_utc"] = now
open(DST, "w").write(json.dumps(p, indent=1))
print(DST, now, hashlib.sha256(open(DST, "rb").read()).hexdigest())
