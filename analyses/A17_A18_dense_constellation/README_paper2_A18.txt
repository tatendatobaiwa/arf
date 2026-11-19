PAPER 2, AMENDMENT A18 (10 November 2026): release 1.5.6 replaces 1.5.5 for the A17 design.

Why: the A17 smoke stage (release 1.5.5) excluded 104 of 10,881 Starlink element sets as "malformed".
Every one carried an Alpha-5 catalogue number (100000 and above: a leading letter, A = 10 to Z = 33,
I and O omitted; Space-Track documentation, section Alpha-5). 1.5.5 read the field as an integer, so the
newest satellites were left out of every sample, 104 to 424 of them per window (3 August to 21 September).
1.5.6 decodes Alpha-5 (Cell 2 only) and names its pin files _opsample_a5. Everything else is A17's design.
The two 1.5.5 smoke runs are superseded and not reported; their folders are kept, so the 1.5.6 smoke runs
write to the same names with the suffix _v156 (Cell 0 rule since 1.5.2). Real fleets are not affected.

Colab, in this order
1. Upload ARF_paper2_A18.zip to MyDrive (do not unzip it there).
2. Fresh runtime. Paste arf_colab_launch_paper2_a18.py into a new cell with STAGE = "smoke"; run.
   Expect: "installed: protocol with A18 (A17 protocol kept as arf_confirmatory_protocol.a17.json)",
   then "... 0 malformed" in both Cell 2 lines. Send me the output.
3. STAGE = "core": 58 runs. Run the cell again whenever Colab stops; finished runs are skipped.
4. STAGE = "sensitivity": 30 runs.
5. STAGE = "summary"; then zip MyDrive/ARF_batch/paper2_a18 and send it.

Tests (sandbox, no network)
  tests/test_alpha5_v156.py      8/8   decoding vs Space-Track examples and Orekit 13.1.9 (all 24 letters);
                                       synthetic set from 99000: 1.5.5 rejects all 9,085 Alpha-5, 1.5.6 none;
                                       identical selection on 5-digit records; own pin names
  Real 1.5.6 Cells 0-9, real Orekit, synthetic orbits, 32.5 km, sun 10 deg, 3 August:
                                 N = 20: 18 s, 16 of 20 Alpha-5;  N = 100: 93 s, 88 of 100 Alpha-5
                                 (tests/timing_n20.json, tests/timing_n100.json); a 1.5.5 pin is not reused
  test_paper2_runner_a18.py     11/11  real 1.5.4 batch runner; folder tags; resumable; refusals
  test_paper2_summary_a18.py    10/10  real 1.5.6 run outputs with a known floor curve: fit recovers it
  test_colab_paper2_a18.py       see tests/colab.log: install over the A17 state, refusals, rerun
  NOT tested: the live Space-Track query (the sandbox cannot reach it); the smoke stage tests it.
