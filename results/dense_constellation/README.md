# Dense-constellation study (amendments A17 to A19)

Starlink orbits used as a hypothetical dense Earth-observation constellation: real orbits (Space-Track `gp_history`), an **assumed** camera. Region: Central Kalahari Game Reserve with the Okavango Delta (12,231 H3 cells at resolution 7). All reported runs use release 1.5.6. Nothing here is a claim about the Starlink system.

| File | Contents |
|---|---|
| `paper2_v156_a18_core_index.csv`, `…_sens_index.csv`, `…_smoke_index.csv`, `paper2_v156_a19_n8000_index.csv` | One row per run: configuration, release, pipeline SHA-256, status, minutes. 58 core, 30 sensitivity and 2 runs at N = 8000. |
| `a18/` | Summary of the 88 runs as registered in A17/A18, **including the registered saturation fit, which failed** (`a18_summary.json`: a limit of 0 h and N* of the order of 10^12). |
| `a19/` | Post hoc (A19): the per-run table of all 90 runs, the restricted fit (`a19_fit.json`), the hours gained per doubling of N (`a19_doubling_gains.csv`), the ECC cap replay by camera and N, and `replay/` with one summary and one per-onset file per run. |

## What was registered, and what happened

1. **A17** registered the design and one saturation fit through N = 5 to 4000. **A18** replaced release 1.5.5 by 1.5.6 before any core run, because 1.5.5 rejected satellites with Alpha-5 catalogue numbers.
2. The registered fit **failed**. The floor is the time to 95% of the *reachable* cells, and below N = 500 the reachable share itself grows with N, so the floor rises to N = 100 before it falls. It is reported as failed.
3. **A19** was registered after those results had been read and is **post hoc**: a fit restricted to N >= 500, the gains per doubling on matched windows (which take precedence over the fit), N = 8000 in the two windows that hold N = 4000, and a replay of ECC without its cap of 30 images.

## Main figures (32.5 km swath, sun at least 10 degrees; see the files for intervals)

* Mean floor over windows: 9.8, 6.0, 4.1 h at N = 500, 1000, 2000 (eight windows); 3.2 and 2.9 h at N = 4000 and 8000 (two windows).
* Gain per doubling in the two matched windows: 3.08, 1.37, 1.24 and 0.30 h (500 to 8000). The last is below 0.5 h in both windows.
* Restricted fit, all windows, N = 8000 included: limit 2.49 h (95% interval 1.61 to 3.01), exponent 1.09.
* ECC without a cap completes at the floor at every onset of every run and uses about 30 to 33 images from N = 250 to 8000, against about 53 for first-come-first-served. It never uses more images than first-come-first-served.

## Limits

Post hoc; the last doubling rests on two windows; one region and one assumed camera per case; first-come-first-served is the only comparator. Element sets are not redistributed, and the run folders with cell-level files are held back (see `DATA.md`).
