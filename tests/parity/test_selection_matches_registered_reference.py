"""arf.selection against the registered reference implementation.

The reference is ``reference()`` in ``analyses/A14_.../arf_published_selectors.py``, the port of Cell 7C that
amendment A14 pinned by hash and that equalled the pipeline at every onset of 24 real windows. Here both are run
on synthetic runs built to provoke ties (equal gains, equal delivery times), repeated cells inside an image,
undelivered images, images of zero utility, and needs above the budget of 30 images.
"""

import math
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from arf.selection import ImageSet, evaluate_onset, onset_grid

h3 = pytest.importorskip("h3")

ROOT = Path(__file__).resolve().parents[2]
A14 = ROOT / "analyses" / "A14_A16_published_selectors_latency_cloud" / "arf_published_selectors.py"
POOL = sorted(h3.grid_disk(h3.latlng_to_cell(-21.5, 23.5, 7), 14))  # 631 valid resolution-7 cells
CASES = [  # name, seed, satellites, images, cells per image, clustered
    ("sparse", 1, 4, 30, 40, False),
    ("medium", 2, 40, 400, 25, False),
    ("ties", 3, 60, 500, 60, True),
    ("narrow", 4, 200, 2500, 6, False),
]
BUDGETS = (30, 60, None)


def synthetic_events(seed, n_satellites, n_images, width, clustered):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_images):
        satellite = f"SAT-{rng.integers(1000, 1000 + n_satellites)}"
        acquired = int(rng.integers(0, 2880 // 5)) * 5
        start = int(rng.integers(0, 6)) * (width // 2) if clustered else int(rng.integers(0, len(POOL) - width))
        cell_positions = list(range(start, min(start + width, len(POOL))))
        if clustered and rng.random() < 0.3:
            cell_positions += cell_positions[:3]
        wait = float(rng.choice([15.0, 30.0, 30.0, 94.0])) if rng.random() > 0.08 else np.nan
        downlinked = 1.0 if rng.random() > 0.05 else 0.0
        for position in cell_positions:
            off_nadir = float(rng.choice([0.0, 5.0, 10.0]))
            rows.append((satellite, acquired, POOL[position], off_nadir, 20.0, downlinked, wait))
    columns = ["sat", "t_min", "tile", "A_nadir_deg", "theta_max_deg", "D_downlink", "T_wait_min"]
    return pd.DataFrame(rows, columns=columns).sample(frac=1.0, random_state=seed)


def reference_results(registered, run_folder, budget):
    keys, cells, wait, n_cells, _ = registered["load"](str(run_folder))
    results = {}
    for onset in onset_grid():
        candidates = [e for e in range(len(keys)) if onset <= keys[e][1] < onset + 1440]
        if not candidates:
            continue
        reachable_mask = np.zeros(n_cells, bool)
        for e in candidates:
            reachable_mask[cells[e]] = True
        reachable = int(reachable_mask.sum())
        if reachable == 0:
            continue
        required = math.ceil(95.0 / 100.0 * reachable - 1e-9)
        delivery = np.array([keys[e][1] - onset + wait[e] for e in candidates])
        order = np.argsort(delivery, kind="stable")
        order = order[np.isfinite(delivery[order])]
        results[onset] = registered["reference"](
            candidates,
            order,
            delivery[order],
            cells,
            keys,
            wait,
            n_cells,
            onset,
            required,
            reachable,
            10**9 if budget is None else budget,
        ) | {"reachable": reachable}
    return results


def same(a, b):
    return (a is None and b is None) or (a is not None and b is not None and math.isclose(float(a), float(b)))


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_selection_equals_registered_reference(case, tmp_path):
    name, seed, n_satellites, n_images, width, clustered = case
    events = synthetic_events(seed, n_satellites, n_images, width, clustered)
    events.to_csv(tmp_path / "arf_refined_events.csv", index=False)
    registered = runpy.run_path(str(A14), run_name="registered_a14")
    images = ImageSet.from_run_folder(tmp_path)
    ours = {onset: evaluate_onset(images, onset, budgets=BUDGETS) for onset in onset_grid()}
    compared = 0
    for budget in BUDGETS:
        reference = reference_results(registered, tmp_path, budget)
        assert set(reference) == {onset for onset, result in ours.items() if result is not None}
        for onset, expected in reference.items():
            got = ours[onset]
            assert got.reachable == expected["reachable"]
            assert same(got.floor_min, expected["floor_min"]), (name, onset)
            assert same(got.fcfs_images, expected["fcfs_images"]), (name, onset)
            assert same(got.completion[budget].time_min, expected["ecc_min"]), (name, onset, budget)
            assert same(got.completion[budget].images, expected["ecc_images"]), (name, onset, budget)
            compared += 1
    assert compared >= 3 * 40
    if name == "narrow":  # the test data must make the budget of 30 bind
        over = [r for r in ours.values() if r and r.completion[None].images and r.completion[None].images > 30]
        assert len(over) > 20
