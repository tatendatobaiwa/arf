PAPER 2, AMENDMENT A17 (9 November 2026): Starlink orbits as a hypothetical dense Earth-observation
constellation, release 1.5.5, assumed nadir camera. Exploratory, registered before any of its runs.

Colab, in this order (after the A14-A16 package has been installed)
1. Upload ARF_paper2_A17.zip to MyDrive (do not unzip it there).
2. Paste arf_colab_launch_paper2_a17.py into a new cell with STAGE = "smoke"; run. Send me the output.
3. STAGE = "core": 58 runs. Run the cell again whenever Colab stops; finished runs are skipped.
4. STAGE = "sensitivity": 30 runs.
5. STAGE = "summary"; then zip MyDrive/ARF_batch/paper2_a17 and send it.

Run time (sandbox, real Orekit, synthetic Starlink-like orbits, window of 3 August 2026, 32.5 km, sun 10 deg):
  N = 5: 20 s; N = 20: 18 s; N = 100: 105 s; N = 1000: see tests/timing_n1000.json
  (the superseded 160-km runs took 22 min at N = 100: the narrower swath gives far fewer events)

Tests (sandbox)
  test_paper2_runner_a17.py   11/11  real 1.5.4 batch runner; folder tags; resumable; refusals
  test_paper2_summary_a17.py  10/10  real 1.5.5 run outputs with a known floor curve: fit recovers it
  test_colab_paper2_a17.py    see its log: install on a mock Drive, refusals, rerun
  Real 1.5.5 Cells 0-9 at N = 5, 20, 100, 1000 (synthetic orbits): no error
  NOT tested: the live Space-Track gp_history query (the sandbox cannot reach it); the smoke stage tests it.
