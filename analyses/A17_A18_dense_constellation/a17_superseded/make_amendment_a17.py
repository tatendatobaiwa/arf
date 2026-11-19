"""Append amendment A17 (Paper 2: Starlink orbits as a hypothetical dense Earth-observation constellation,
release 1.5.5) to the A16 protocol. Refuses any other input.
Usage: make_amendment_a17.py IN_A16_PROTOCOL OUT_PROTOCOL SCRIPT_DIR SHA_A16"""
import sys, os, json, hashlib, datetime

SRC, DST, D, SHA_A16 = sys.argv[1:5]
raw = open(SRC, "rb").read()
assert hashlib.sha256(raw).hexdigest() == SHA_A16, "input is not the A16 protocol"
p = json.loads(raw)
assert [a["id"] for a in p["amendments"]] == [f"A{i}" for i in range(1, 17)]
sha = lambda f: hashlib.sha256(open(os.path.join(D, f), "rb").read()).hexdigest()
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
p["amendments"].append({
    "id": "A17", "date_utc": now,
    "status_when_amended": ("EXPLORATORY, REGISTERED BEFORE ANY OF ITS RUNS. Written after the confirmatory results and "
                            "A4 to A16 were registered, and after the 26 runs of the dense arm of A3 (release 1.5.4), "
                            "which are superseded and not reported: they gave every Starlink satellite an off-nadir "
                            "half-angle of 9.22 deg that has no source (a ground swath of about 155 to 180 km). The "
                            "scripts were tested on synthetic Starlink-like orbits only (N = 5, 20 and 100 with real "
                            "Orekit, window of 3 August 2026); no result has been seen."),
    "question": ("How do the earliest-completion floor, the reachable share of the region and the image economy of ECC "
                 "change with the size N of a dense Earth-observation constellation, and beyond which N does the floor "
                 "stop falling? (Objectives O1 to O4 of the proposal at the densities it names, N = 5, 20 and 100, and "
                 "beyond.)"),
    "design": {
        "release": "1.5.5 (SHA-256 " + sha("algorithmic_reconstruction_framework_v155.py") + ")",
        "geometry": ("Starlink satellites operational at each window start (Space-Track gp_history), one seeded random "
                     "order per window, nested fleets (the first N)"),
        "assumed_camera": ("nadir, ground swath proxy_swath_km; imaging only with the Sun at least proxy_min_sun_elev_deg "
                           "above the imaged point; primary 32.5 km (PlanetScope SuperDove frame width, ESA Earth Online, "
                           "accessed 4 Nov 2026) and 10 deg (Planet's constraint for the Doves, cited in Paper 1)"),
        "region_and_windows": "CKGR + Okavango; the eight protocol windows (3 August to 21 September 2026)",
        "settings": "every protocol setting, cloud-sensitivity sweep off",
        "core": "32.5 km, 10 deg; N = 5, 20, 100, 250, 500, 1000, 2000 in all eight windows; N = 4000 in the windows of 3 August and 7 September",
        "sensitivity": ("8 km (Planet Pelican Gen 1) and 185 km (Landsat 9) at 10 deg, and 32.5 km at 5 deg; "
                        "N = 100, 250, 500, 1000, 2000 in the windows of 3 August and 7 September"),
        "not_included": ("comparator constellations (Walker-delta, the sun-synchronous Starlink shell alone, PlanetScope at "
                         "matched N) and a 2025 window set: both await the supervisors' decision and, if approved, are "
                         "added by a later amendment before they are run"),
        "fleet_size_note": "N = 250 replaces the 200 of the Paper 2 plan: 200 is not a fleet size the release accepts"},
    "analysis": ("Descriptive; no hypothesis decision. Per run: the medians over onsets of the floor to 95% of the "
                 "reachable cells and of the reachable share; the share of onsets covering 95% of the whole region; ECC "
                 "and FCFS images; H1a- and H1b-type shares; the Fail-Signal gain. Means over windows with window-block "
                 "bootstrap 95% intervals (B = 2000, seed 42). Saturation of the core floor against N, three ways, each "
                 "with window-block bootstrap intervals: T(N) = T_inf + a N^-b with N* = (a/m)^(1/b) and the smallest "
                 "grid N within m of T_inf, for m = 0.5 h and 1 h; a one-breakpoint segmented regression on log10 N "
                 "(Muggeo 2003); the Kneedle knee (Satopaa et al. 2011)."),
    "scope": ("New runs only, in folders tagged _sun<deg>_sw<km>km that no earlier run uses; real-fleet results and "
              "decisions unchanged. Reported in Paper 2 as an exploratory study on an assumed instrument."),
    "script": {f: sha(f) for f in ("arf_paper2_runner_a17.py", "arf_paper2_summary_a17.py",
                                   "algorithmic_reconstruction_framework_v155.py", "arf_batch_runner_v154.py")}})
p["amended_utc"] = now
open(DST, "w").write(json.dumps(p, indent=1))
print(DST, now, hashlib.sha256(open(DST, "rb").read()).hexdigest())
