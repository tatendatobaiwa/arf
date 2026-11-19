ARF AMENDMENTS A14, A15, A16 (7 November 2026). Release 1.5.3 unchanged.
A14  published image selectors (Wang et al. 2026; Kempeneers and Soille 2017; Pan et al. 2024) on the
     same predicted passes as ECC. Post hoc, descriptive.
A15  eight added dry-season windows (6 to 27 July, 28 September to 19 October 2026), primary design,
     p = 0.35. Registered replication: H1a, H1b and H2 decided on these windows alone.
A16  (a) product latency scenarios added to delivery (ESA and USGS timeliness; PlanetScope assumed);
     (b) observed cloud in Landsat 8/9 QA_PIXEL from Microsoft Planetary Computer. Post hoc, descriptive.

Colab, in this order (send me the printed output of steps 2, 4 and 6, and the zips of steps 5 and 6)
1. Upload ARF_posthoc_A14A16.zip to MyDrive (do not unzip it there).
2. New cell:  !pip install -q h3==4.5.0      then paste arf_colab_run_a14_a16.py below it and run.
   Installs the protocol (A13 kept as arf_confirmatory_protocol.a13.json) and the scripts, then runs
   A14 and A16a on the 16 existing windows (about 1 min per window). Writes ARF_A14A16_results_<time>.zip.
3. With the Space-Track secrets set and this tab open, paste ARF_code/arf_colab_prefetch_dryext_a15.py
   into a cell and run it (8 windows, one query each; nothing is printed of the secrets).
4. !python /content/drive/MyDrive/ARF_code/arf_dryext_runner_a15.py      (8 runs, about 30 to 40 min)
   !python /content/drive/MyDrive/ARF_code/arf_dryext_summary_a15.py     (decisions for H1a, H1b, H2)
5. Run the cell of step 2 again: it now adds the 8 A15 windows (a new results zip).
6. !pip install -q rasterio h3==4.5.0 pystac-client planetary-computer
   !python /content/drive/MyDrive/ARF_code/arf_real_cloud_landsat_a16.py
   Then zip MyDrive/ARF_batch/real_cloud_landsat_a16 and upload it.

Safety: every script is read-only on earlier runs; A15 runs write only to MyDrive/ARF_dryext_A15/p035/<window>,
and the batch stops if any confirmatory, A4 or A5 folder changes. Do not rerun earlier install or launch cells.

Tested here (sandbox, 7 November 2026)
test_a14.py                    4/4   selectors against brute force on 300 random instances
A14 on the real 3 August run   identity check 73/73 onsets; all methods reported
A16a on the real 3 August run  identity check at zero latency 73/73; floors monotone; ECC never before the floor
test_landsat_a16.py           11/11  QA bits, A8 tile rule, real GeoTIFF sampling, main() on the real 12,231 cells
                                     with a fake catalogue. NOT tested: a live Planetary Computer request
                                     (the sandbox cannot reach it); asset key "qa_pixel" is tried first, then "QA_PIXEL".
test_dryext_runner_a15.py     11/11  real batch runner and A4 runner; Cell 0 folder code verbatim
test_dryext_summary_a15.py     9/9   real release-1.5.3 run outputs; statistics equal scipy; table compiles
test_colab_a14_a16.py         see its log: install, 16 windows replayed, refusals, rerun
Stated limit (A15): the fleet is built from the current CelesTrak list, so a satellite retired since July 2026
drops out (Cell 2 prints the fleet).
