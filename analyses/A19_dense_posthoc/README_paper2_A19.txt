PAPER 2, AMENDMENT A19 (15 November 2026): POST HOC. Registered after the 88 runs of A17/A18 were read.

Why
  1. The saturation fit registered in A17 failed (limit 0 h, N* of 10^12): the floor is the time to 95% of the
     REACHABLE cells, and below N = 500 the reachable share grows with N, so the floor is not one curve.
  2. Where the whole region is reachable the floor still falls at N = 4000 (1.35 and 1.13 h gained from N = 2000).
  3. ECC runs with at most 30 images, FCFS without a cap: at 8 km ECC never completes from N = 250 (8 of 8 runs).

What A19 registers (all labelled post hoc)
  (a) arf_ecc_cap_replay_a19.py   read-only replay of every finished 1.5.6 run: ECC at k = 30 (must equal the run at
                                  every onset, or the replay stops), 60, 120 and without a cap
  (b) arf_paper2_summary_a19.py   fit T = T_inf + a N^-b on the core case, N >= 500 only, with window-block bootstrap;
                                  hours gained per doubling on matched windows, which take precedence over the fit
  (d) arf_paper2_runner_a19.py    N = 8000, 32.5 km, sun 10 deg, windows of 3 August and 7 September (2 runs)
  Not adopted: N = 4000 in the six other windows.

Colab, in this order (arf_colab_launch_paper2_a19.py; upload ARF_paper2_A19.zip to MyDrive, do not unzip)
  1. STAGE = "replay"    minutes; send me the output
  2. STAGE = "summary"   minutes; send me the output
  3. STAGE = "n8000"     fresh runtime; 2 runs, expected 4 to 5 h each; if the session crashes, stop and tell me
  4. STAGE = "summary"   again; then zip MyDrive/ARF_batch/paper2_a19 and send it

Tests (sandbox, no network)
  tests/test_ecc_cap_replay_a19.py   13/13  against the A14 function itself at 365 onsets, k = 30, 60, 120, no cap; ties;
                                            identity check passes and stops on one changed value; --all resumable
  tests/test_paper2_summary_a19.py   14/14  on the REAL floors of the 88 runs (T_inf 2.37 h [1.51, 2.95], b 1.05;
                                            matched windows 1.99 h; without N = 4000 2.36 h) and on a known curve
  tests/test_paper2_runner_a19.py    12/12  real 1.5.4 batch runner; exactly the two N = 8000 runs; resumable; refusals;
                                            a memory error is recorded as an error
  tests/test_colab_paper2_a19.py     see tests/colab.log: install over the A18 state; replay then summary end to end
  NOT tested: N = 8000 itself (time and memory in Colab); the replay on the real run folders (the sandbox does not
  hold them): its identity check against every run's own file is the test, and it stops if one value differs.
