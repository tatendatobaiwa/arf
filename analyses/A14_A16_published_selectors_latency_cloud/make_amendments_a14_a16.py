"""Append amendments A14 (published selectors), A15 (added dry-season windows, registered replication) and
A16 (product latency; observed Landsat cloud) to the A13 protocol. Refuses any other input protocol.
Usage: make_amendments_a14_a16.py IN_A13_PROTOCOL OUT_PROTOCOL SCRIPT_DIR"""
import sys, os, json, hashlib, datetime

SRC, DST, D = sys.argv[1], sys.argv[2], sys.argv[3]
raw = open(SRC, "rb").read()
assert hashlib.sha256(raw).hexdigest() == "e49f073a68265357ce2af00bac6583b67672f4f1011d7c297c458c470d275bda", \
    "input is not the A13 protocol"
p = json.loads(raw)
assert [a["id"] for a in p["amendments"]] == [f"A{i}" for i in range(1, 14)]
sha = lambda f: hashlib.sha256(open(os.path.join(D, f), "rb").read()).hexdigest()
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
W16 = ("the 8 primary dry-season windows (3 Aug to 21 Sep 2026) and the 8 wet-season windows of A5 "
       "(5 Jan to 23 Feb 2026, p = 0.35 runs)")
A15_WINDOWS = ["2026-07-06T00:00:00", "2026-07-13T00:00:00", "2026-07-20T00:00:00", "2026-07-27T00:00:00",
               "2026-09-28T00:00:00", "2026-10-05T00:00:00", "2026-10-12T00:00:00", "2026-10-19T00:00:00"]
p["amendments"] += [
    {"id": "A14", "date_utc": now,
     "status_when_amended": ("POST HOC. Written after the confirmatory results and A4 to A13 were read. While the "
                             "script was tested, it was run on the window of 3 August 2026 (sandbox copy of that "
                             "run, 73 onsets): mean images ECC 6.88, WANG26 7.63, PAN24 7.93, KS17 10.32, FCFS 11.37; "
                             "every method reached the level within the H1a margin of the floor at every onset. No "
                             "other window has been replayed."),
     "question": ("How does ECC compare with published image selectors given exactly the same candidates: the "
                  "run's predicted events of the 24 h after each onset, clear sky, H3 resolution 7, 95% of the "
                  "reachable cells?"),
     "method": ("Three selectors re-implemented from their published descriptions, adapted to predicted passes "
                "measured from the onset: WANG26 (Wang et al. 2026, Remote Sensing 18(11):1855, sections 3.3 to "
                "3.5: rank by time, resolution, cloud; accept an image that intersects the remaining target; "
                "remove invalid coverage; earliest delivery after the onset stands for 'most recent'), reported "
                "stopped at the level and run to full coverage; KS17 (Kempeneers and Soille 2017, Big Earth Data "
                "1(1-2):145-158: per tile, minimum set of at most 5 images maximising covered cells by Sequential "
                "Floating Forward Search; tile = H3 resolution-3 parent; the SFFS body is the standard one of "
                "Pudil et al. 1994, an assumption, because the article's steps were not available to us); PAN24 "
                "(Pan et al. 2024, Geo-spatial Information Science 27(5):1524-1541: shortest span of delivery "
                "times after the onset whose images reach the level, ties to the earliest span, then the images "
                "of that span by largest new coverage; HFA-Net quality grading has no counterpart for predicted "
                "passes and is not used). Per onset: time to the level, images, coverage reached. The floor, ECC "
                "and FCFS are recomputed with the A13 port of Cell 7C and must equal the run's "
                "arf_onset_robustness.csv, or the script stops."),
     "scope": f"Read-only replay of {W16}, and of the A15 windows once run. Descriptive; no hypothesis, no decision changes.",
     "script": {"arf_published_selectors.py": sha("arf_published_selectors.py")}},
    {"id": "A15", "date_utc": now,
     "status_when_amended": ("REGISTERED REPLICATION, POST HOC IN MOTIVATION. Written after the confirmatory results "
                             "and A4 to A13 were read (H1b was decided by one of eight windows), and before any run of "
                             "the windows below existed."),
     "question": "Do H1a, H1b and H2 hold in eight further windows of the 2026 dry season?",
     "windows": A15_WINDOWS,
     "reason": ("The dry season over the region runs from May to October, with dryland fire activity peaking in "
                "September (Heinl et al. 2007) and the Okavango flood at its maximum extent in July to September "
                "(Inman and Lyons 2020). The added windows extend the confirmatory windows (3 August to 21 September) "
                "to July and to late September and October. The same weeks of 2025 are not used: the pipeline builds "
                "the fleet from the current CelesTrak lists, so satellites retired since 2025 would drop out."),
     "method": ("Release 1.5.3 unchanged, primary design (CKGR + Okavango, earth-resources+planet, conservative "
                "swath, p = 0.35), run retrospectively with Space-Track gp_history element sets; runner "
                "arf_dryext_runner_a15.py (the A5 runner with these windows and the p = 0.35 setting only)."),
     "decision_rules": ("H1a, H1b and H2 are decided on these eight windows alone, by the protocol's decision_rules: "
                        "one-sided Student-t lower bound of the mean over windows at alpha/3 at least threshold_pct "
                        "(H1a, H1b) or above 0 (H2); a window counts with at least min_evaluable_onsets evaluable "
                        "onsets; a hypothesis is decided only with at least 5 windows. The original confirmatory "
                        "decisions are not changed. Over all sixteen dry-season windows the statistics are reported "
                        "descriptively, with no decision."),
     "scope": "Adds eight runs. Decisions reported as a registered replication beside the confirmatory decisions.",
     "script": {f: sha(f) for f in ("arf_dryext_runner_a15.py", "arf_colab_prefetch_dryext_a15.py",
                                    "arf_dryext_summary_a15.py")}},
    {"id": "A16", "date_utc": now,
     "status_when_amended": ("POST HOC. Written after the confirmatory results and A4 to A13 were read. Part a was "
                             "tested on the window of 3 August 2026 (sandbox copy of that run): ECC stayed within the "
                             "H1a margin of the floor at every onset in every scenario; the median floor rose from "
                             "12.1 h to 13.8 h (fast), 15.3 h (typical) and 36.1 h (slow). Part b was tested only on "
                             "synthetic quality bands; no real Landsat scene has been read."),
     "question": ("(a) How do the floor, ECC and FCFS change when real product latencies are added to delivery? "
                  "(b) How cloudy were the Landsat 8 and 9 acquisitions of the primary windows?"),
     "method": ("(a) Per-mission latency added to every delivery time; scenarios (Sentinel-2, Landsat, PlanetScope, "
                "hours): fast 1.67, 4, 2; typical 3, 6, 6; slow 24, 12, 24; and PlanetScope 2, 6, 12 or 24 with "
                "Sentinel-2 3 and Landsat 6. Sources: ESA Sentinel-2 MSI Technical Guide timeliness (Real-Time no "
                "later than 100 min; Near Real-Time 100 min to 3 h; Nominal 3 to 24 h); USGS Landsat Collection 2 "
                "Level-1 (Real-Time 4 to 6 h, under 12 h). No published PlanetScope latency was found: its values "
                "are scenarios. Recomputed with the registered A13 port; identity with arf_onset_robustness.csv at "
                "zero latency, or the script stops. (b) Landsat 8/9 Collection 2 Level-2 QA_PIXEL (USGS LSDS-1619 "
                "v6.0, Table 6-2) read from Microsoft Planetary Computer (collection landsat-c2-l2, anonymous "
                "signing), sampled at the 7 points per tile of A8; a point is cloudy when bit 1, 3 or 4 is set "
                "(sensitivity: or bit 2); the tile and overpass rules of A8."),
     "scope": (f"(a) Read-only replay of {W16} and of the A15 windows once run. (b) The same windows. Descriptive; "
               "no hypothesis, no decision changes. PlanetScope cloud (UDM2) is not measured: it needs a Planet "
               "licence."),
     "script": {"arf_product_latency.py": sha("arf_product_latency.py"),
                "arf_real_cloud_landsat_a16.py": sha("arf_real_cloud_landsat_a16.py")}},
]
p["amended_utc"] = now
open(DST, "w").write(json.dumps(p, indent=1))
print(DST, now, hashlib.sha256(open(DST, "rb").read()).hexdigest())
