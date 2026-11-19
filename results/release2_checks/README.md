# Release 2 checked against real runs

Output of the tools in `tools/`, run where the run folders are (they are not in the repository).

The branches and pull-request numbers named below belong to the repository in which release 2 was developed
until 19 November 2026; see `docs/PROVENANCE.md`. The `arf` version of each row identifies the code.

| File | Date | Release 2 commit | Runs | Compared | Result |
|---|---|---|---|---|---|
| `selection_on_a18_core_runs_20261116.txt` | 16 November 2026 | branch `release/2.0-dev`, `arf` 2.0.0.dev1 | 58 core runs of the dense-constellation study (release 1.5.6; N = 5 to 4000) | 4,211 | All agree: floor, first-come-first-served images, earliest-completion time and images at a budget of 30, reachable cells. |
| `selection_on_release153_real_fleet_runs_20261116.txt` | 16 November 2026 | the same | 16 real-fleet runs of release 1.5.3 over CKGR and the Okavango, dry-season protocol windows of 3 August to 21 September 2026: the 8 primary runs (Paper 1) and the 8 runs without Sentinel-2A | 1,168 | All agree, same five quantities. |
| `cloud_on_real_runs_20261116.txt` | 16 November 2026 | branch `release/2.0-cloud`, `arf` 2.0.0.dev2 | The same 16 real-fleet runs (release 1.5.3) and 58 core runs (release 1.5.6) | 1,480 cloud realisations (20 per run) over 14,505,064 observations | All agree: the share of cloudy observations of every realisation equals the one in the run's `arf_cloud_robustness.csv` (tool: `tools/check_cloud_on_runs.py`). |
| `failsignal_on_real_runs_20261116.txt` | 16 November 2026 | branch `release/2.0-failsignal`, `arf` 2.0.0.dev3 | The same 16 real-fleet runs, all 20 cloud realisations each; the same 58 core runs, the first 3 realisations each | 14,820 figures of 494 realisations | All agree: for both objectives, the images and coverage of the Fail-Signal selector, the two open-loop plans, the greedy with the cloud known, the earlier selector and its open-loop plan, with the fail-signals and compensatory images (tool: `tools/check_failsignal_on_runs.py`). |
| `grid_on_a18_core_runs_20261117.txt` | 17 November 2026 | branch `main`, `arf` 2.0.0.dev4 | The 58 core runs of the dense-constellation study (release 1.5.6), all over CKGR and the Okavango Delta at H3 resolution 7 | 58 runs; one region, discretised once from the WDPA downloads (sites 7510 and 555577555) | All agree: 12,231 cells, the same cells as each run's `arf_mosaic_tiles.csv`, and both site reports equal to those in each run's `arf_cell1_validation.json` (tool: `tools/check_grid_on_runs.py`). The file holds counts only: no cell and no boundary. |
| `grid_on_real_fleet_runs_20261117.txt` | 17 November 2026 | the same | 56 real-fleet run folders found on the author's drive: 40 over CKGR and the Okavango Delta (the 8 primary runs of Paper 1 and the 8 without Sentinel-2A among them), 8 over Lower Zambezi and 8 over Karoo Nature Reserve, protocol windows of 3 August to 21 September 2026 | 56 runs; three regions, each discretised once from the WDPA downloads | All agree: 12,231 cells (CKGR and Okavango), 672 cells (Lower Zambezi, WDPA 7962) and 32 cells (Karoo Nature Reserve, WDPA 555563860), with the cell lists and site reports of every run. Counts only. The release that produced each folder was not read by the tool. |
| `fleet_on_a18_core_runs_20261117.txt` | 17 November 2026 | branch `main`, `arf` 2.0.0.dev5 | The 58 core runs of the dense-constellation study (release 1.5.6): eight windows, N = 5 to 4000 | 58 samples drawn from 8 sets of Space-Track element sets (10,881 on 3 August to 11,133 on 21 September 2026), 39,000 sampled satellites in all | All agree: the counts of the operational filter equal those in each run's `arf_cell2_summary.json`, and the first N catalogue numbers of the seeded shuffle equal the run's pin file, in order (tool: `tools/check_fleet_on_runs.py`). No input file had changed since the run. Counts only: no element set. |
| `fleet_on_real_fleet_runs_20261117.txt` | 17 November 2026 | the same | The 56 real-fleet run folders of the grid check: 16 with the open-data fleet alone (8 of them without Sentinel-2A), 16 with PlanetScope at the conservative swath (8 without Sentinel-2A; the primary runs of Paper 1 are the other 8), 8 at the optimistic swath, 8 over Lower Zambezi and 8 over Karoo Nature Reserve | 56 fleets selected again from the Space-Track history file each run recorded, with names from CelesTrak's current lists | All agree: counts, mission table and every satellite (name, catalogue number, mission, swath, field of view, daylight limit) equal the run's summary and fleet file: 96 satellites (3 Sentinel-2, 2 Landsat, 91 PlanetScope), 95 without Sentinel-2A, 5 and 4 for the open-data fleet. CelesTrak's lists had been replaced since every run (their SHA-256 differ from the manifests'); the history files had not. In every run all element sets passed the filter (96 of 96), so these runs test the mission matching, the swaths and the exclusion by name, and do not exercise the rejection of stale, low or malformed element sets, which only the parity tests cover. |
| `access_on_real_fleet_runs_20261117.txt` | 17 November 2026 | branch `release/2.0-access` (PR #10), `arf` 2.0.0.dev7 | The 56 real-fleet runs of the grid and fleet checks; their manifests name release 1.5.3, so the registered cells executed were those of 1.5.3 | 56 runs: access computed with Orekit by release 2 and by the registered Cells 2, 4 and 5 executed unchanged, in the same Colab session from each run's recorded inputs, with the Orekit data of 17 November 2026 | **Port: identical in all 56** (altitudes, period, search radii, access events with every strip sample, downlink contacts, refined events, drop counts). Run: 51 agree with the run's own files (identical for the windows of 3 to 24 August; same rows for later windows, largest differences 2.5e-6 degrees in the off-nadir angle and 3.0e-6 degrees in elevation); 5 not reproduced, see below. |
| `access_on_a18_core_runs_upto250_20261117.txt` | 17 November 2026 | the same | The 58 core runs of the dense-constellation study (release 1.5.6), of which the 32 with at most 250 satellites were checked (N = 5, 20, 100, 250 in all eight windows); 26 larger runs skipped | 32 runs, as above, with release 1.5.6's cells | **Port: identical in all 32. Run: identical in all 32**, every table and summary. The runs with 500 to 4000 satellites are not yet checked. |
One run has 50 onsets, not 73: with five satellites, the other onsets have no candidate image.

Runs over one region share one set of boundary files, so the grid checks are three independent tests of the discretisation (one per region), repeated over 114 runs; they also show that every run of a region recorded the same grid.

Not yet checked: the 30 sensitivity runs and the two N = 8000 runs of the dense-constellation study; the wet-season, further dry-season and secondary-region runs of release 1.5.3. For the Fail-Signal figures, realisations 4 to 20 of the dense runs are also unchecked. For the fleet: the sensitivity and N = 8000 dense runs, and the wet-season and further dry-season real-fleet runs. For the grid: the polygon and corridor regions (Botswana bounding box, Trishuli corridor) have no real run to compare with; they are covered by the parity tests only.

## Correction of 17 November 2026: where the settings were read

Until `arf` 2.0.0.dev4, `arf.runs.run_settings` looked for a run's settings under a top-level `config` key of the
manifest. The releases write them under `inputs` → `config`, so the tools found none and used the release defaults for
every run (seed 42, correlated cloud model, 50 km, 3 h, probability 0.35, budget 30, decay 3.0). The cloud and
Fail-Signal checks above were therefore made with the defaults, not with each run's recorded settings. Their agreement
stands: a wrong seed or cloud model would not reproduce 1,480 cloudy shares and 14,820 figures exactly, so those runs
did use the defaults. The fault would have mattered for a run with other settings, which would have been reported as
not agreeing. Fixed in 2.0.0.dev5, with a test that reads the location from the release source.
| `access_five_runs_with_run_date_orekit_data_20261117.txt` | 17 November 2026 | the same, with `--no-registered` | The 5 runs not reproduced above, and one control (planet-conservative, window of 31 August, same rows on 17 November) | Release 2 with the Orekit data of the commit of https://gitlab.orekit.org/orekit/orekit-data last before each run finished (all six finished on 26 October 2026; commit `9de5188d8410` of 23 October) | **All six identical**: every table and summary equals the run's own files. |
| `onsets_on_real_fleet_runs_first_tool_20261118.txt`, `onsets_on_a18_core_runs_first_tool_20261118.txt` | 18 November 2026 | branch `release/2.0-report-onsets` (PR #11), `arf` 2.0.0.dev8, first version of `tools/check_onsets_on_runs.py` | The 56 real-fleet runs (release 1.5.3) and the 58 core runs of the dense-constellation study (release 1.5.6) | 114 onset tables (73 onsets by 81 columns) and their summaries, release 2 against the run's files, exactly | Superseded by the rows below. 5 of 56 and 2 of 58 identical; in all others only mean utilities (`*_quality95r`) and the medians derived from them differed, by at most 3.3e-16 in the values listed. This version could not say whose the differences were, and listed at most four per run. |
| `onsets_on_real_fleet_runs_20261118.txt` | 18 November 2026 | the same, `arf` 2.0.0.dev8 with the tool's second version | The 56 real-fleet runs | 56 runs: Cell 7C section 2 computed by release 2 and by the registered Cell 6 helpers and Cell 7C of release 1.5.3 executed unchanged, in the same Colab session, from each run's refined events and cells | **Port: identical in all 56**, every value of table and summary to the last bit. Run: 5 identical to the run's files; 51 identical up to rounding (every integer and every H1 statement identical; floating-point values within 1e-12 relative, see below). |
| `onsets_on_a18_core_runs_upto250_20261118.txt` | 18 November 2026 | the same | The 32 core runs with at most 250 satellites (release 1.5.6) | 32 runs, as above, with release 1.5.6's cells | **Port: identical in all 32.** Run: 2 identical, 30 identical up to rounding. |
| `onsets_on_a18_core_runs_above250_no_registered_20261118.txt` | 18 November 2026 | the same, with `--no-registered` | The 26 core runs with 500 to 4000 satellites | 26 runs, release 2 against the run's files only (the registered cells were not executed, to save time) | Run: 26 identical up to rounding. The port is not tested on these runs; it is tested on the 88 above and in `tests/parity/`. |
| `cloud_robustness_on_real_fleet_runs_20261119.txt` | 19 November 2026 | `arf` 2.0.0.dev9 (milestone 6b merged), `tools/check_cloud_robustness_on_runs.py` | The 56 real-fleet runs (release 1.5.3) | 56 runs: Cell 7C sections 4 and 5 (20 cloud realisations, H2, the sweep of 9 cloud-model settings) computed by release 2 and by the registered Cell 6 and Cell 7C of 1.5.3 executed unchanged, in the same session, from each run's refined events | **Port: identical in all 56. Run: identical in all 56**: `arf_cloud_robustness.csv`, `arf_cloud_sensitivity.csv` and the summary, every value, with no rounding difference. |
| `cloud_robustness_on_a18_core_runs_upto250_20261119.txt` | 19 November 2026 | the same | The 32 core runs with at most 250 satellites (release 1.5.6) | 32 runs, as above with release 1.5.6's cells; these runs computed no sweep | **Port: identical in all 32. Run: identical in all 32.** |
| `cloud_robustness_on_a18_core_runs_above250_no_registered_20261119.txt` | 19 November 2026 | the same, with `--no-registered --no-sensitivity` | The 26 core runs with 500 to 4000 satellites | 26 runs, release 2 against the run's files only | Run: identical in all 26. The port is not tested on these runs. |

## The five real-fleet runs not reproduced on 17 November 2026

The five runs over CKGR and the Okavango of the window starting 21 September 2026 (open-data fleet with and without
Sentinel-2A, PlanetScope conservative with and without Sentinel-2A, PlanetScope optimistic) each differ from their own
files in one value: `tiles_hit` of one access strip, by one cell. For the same runs the port is identical, so release 2
computes exactly what release 1.5.3 computes today; the difference is between today's computation and the run's.

* `tiles_hit` counts the cells within the scan's search radius around a strip: 1.05 times the half-swath plus a
  sampling margin (Cell 4). It is a buffer for finding candidates, not coverage. A cell at the edge of that buffer lies
  outside every swath, so it cannot enter the refined events, and in all five runs the refined events have the same
  rows, with angles that differ by at most 2.2e-6 degrees.
* Cause, confirmed: revised Earth orientation parameters. The runs loaded Orekit's data of their day; the check
  loaded that of 17 November. For the windows of 3 to 24 August every table is identical; from 31 August on, the
  positions differ by 2e-8 to 2e-7 degrees (up to about 2 cm), growing with the date of the window, the pattern
  expected when values that were predictions or rapid estimates at run time have since been replaced. A shift of 2 cm
  moves a cell across the edge of the search radius only if its centre lies within 2 cm of that edge; the flip lies in
  a strip that all five fleets share, one of the nine strips of the open-data satellites in this window, and a rough
  estimate (edge length times 2 cm, one cell per 5.16 km2) put the chance of it at a few per cent.
* Confirmation (`access_five_runs_with_run_date_orekit_data_20261117.txt`): run again with the Orekit data of the
  commit of https://gitlab.orekit.org/orekit/orekit-data last before each run finished (`9de5188d8410`, 23 October
  2026; the runs finished on 26 October), all five, and a control run of 31 August whose tables had differed within
  the tolerance, reproduce their own files exactly. With the data the runs had, release 2 reproduces every table and
  summary they wrote.

These runs' refined events, which are all that the selection and every result of Paper 1 use, are unaffected.

## Correction of 17 November 2026: how run files were read

Until `arf` 2.0.0.dev7, `EventTable.from_run_folder` read `arf_refined_events.csv` with pandas' default float parser,
which, before pandas 3.1, returns a value one unit in the last place away from the one written for about a fifth of all
values (40,172 of 200,000 with pandas 3.0.5; the `round_trip` parser, and from pandas 3.1 the default, return every
one exactly). The selection, cloud and Fail-Signal checks
above therefore worked on refined events perturbed in the sixteenth significant digit, while the runs had worked on the
exact values in memory. Their agreement stands, and is the stronger for it: they compare counts, selections and
coverage, and none moved. From 2.0.0.dev8 the refined events are read exactly whatever the pandas version (`arf.runs.read_csv_exact`), which the comparison of the onset
tables needs, since those hold mean utilities compared to the last bit.

## The onset tables: rounding, and what it does not touch

On 18 November 2026 every onset table and summary computed by release 2 was identical, to the last bit, to the one
computed by the registered cells in the same session (88 runs). Against the runs' own files, 7 of 114 were identical
and 107 differed in floating-point values only, by at most 1e-12 relative. What is known about those differences:

* Every integer of table and summary was identical in all 114 runs: images, counts of cells and onsets, success
  flags, and the H1a and H1b counts and statements. The tool requires this; one differing integer would have made the
  run "not reproduced".
* Every differing value the tools named (1,020 entries of the table, each one column of one run over one or more onsets, and 382 of the summary, over both tools and all 114 runs) is a mean utility of a mosaic at the
  coverage level (`*_quality95r`, the mean of the decayed utility over the cells covered) or a summary value computed
  from those (`quality_at_level`: medians, their differences and relative differences). The second tool names at most
  six per run and the first at most four, so in many runs further differing values went unnamed; they are
  floating-point values within the tolerance, but that they are all mean utilities is likely, not shown.
* In the table the named values differ by at most 3.3e-16 (one to three units in the last place for values between
  0.25 and 1). The relative differences of up to 5.9e-13 printed by the second tool are in derived summary values that
  subtract two nearly equal medians (for example a median difference of -1.9e-4 from medians near 0.57), where the
  same last-bit difference is a larger share of a small result.
* Cause, not proven: the order in which NumPy sums the utilities. A mean of thousands of values is exact only up to
  that order, and NumPy chooses its summation loops by version and by the processor's instruction set. The real-fleet
  runs used NumPy 2.1.3 and the check 2.4.6; the dense runs used 2.4.6 for both and still differ, so the version is
  not the whole cause, and the processor of the Colab machine, which the runs did not record, is the likely remainder.
  The identical port excludes release 2 as a cause.
* Tolerance: floating-point values may differ by 1e-12 relative. The largest difference seen, 5.9e-13, was within a
  factor of two of it, in a relative difference near zero; a derived value nearer zero could exceed the tolerance by
  rounding alone and be reported as not reproduced. None was.

The integers Paper 1 reports from these files (images, H1a and H1b counts) are identical; no time to a coverage level
was among the named differences; the mean utilities agree to about fifteen significant digits.

## The cloud robustness tables: no rounding

Unlike the onset tables, the cloud robustness tables of all 114 runs were reproduced exactly, NumPy 2.1.3 against
2.4.6 included. Their floating-point values are coverage shares (a count of cells divided by another) and means and
standard deviations over 20 or 10 realisations, short sums in which a different summation order has few chances to
show; the onset tables average utilities over thousands of cells. That fits the explanation given above for the onset
tables, but does not prove it.
