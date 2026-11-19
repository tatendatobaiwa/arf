"""arf.report (the onset table and its summary) against the registered releases themselves.

Cell 6 of releases 1.5.3 and 1.5.6 up to its first selection, and Cell 7C's helpers and section 2 (onset robustness),
are executed unchanged (``registered.py``) on synthetic refined events with real cell geometry, spread over a 48-hour
window. They write ``arf_onset_robustness.csv`` and ``arf_onset_robustness_summary.json``; release 2 must give the
same table, value for value and in the same column order, the same CSV text and the same JSON text, for the default
settings of both releases and for settings that take every other branch: cloud known in advance, the clear-sky
planning rule, the legacy objective on the fixed time scale, the independent cloud model with a small budget, and a
shorter window with other levels. The registered releases call the independent cloud model "iid".
"""

import json

import numpy as np
import pandas as pd
import pytest

from arf.cloud import LatentCloudField, correlated_cloud_mask, independent_cloud_mask
from arf.events import EventTable
from arf.grid import cell_centres
from arf.report import OnsetSettings, onset_summary, onset_table

h3 = pytest.importorskip("h3")
from registered import run_cell7c_onsets  # noqa: E402
from test_cloud_matches_registered_release import COLUMNS, REGION  # noqa: E402

ORBIT_MIN = 94.6
ALTITUDE_KM = 515.0


def synthetic_window(seed, n_images=220):
    """Refined events over 48 hours: shared acquisition minutes, undelivered images, images of zero utility."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_images):
        satellite = f"SAT-{rng.integers(100, 150)}"
        acquired = int(rng.integers(0, 96)) * 30
        start = int(rng.integers(0, len(REGION) - 250))
        wait = float(rng.choice([0.0, 15.0, 30.0, 94.0])) if rng.random() > 0.1 else np.nan
        downlinked = float(rng.choice([1.0, 0.6])) if rng.random() > 0.1 else 0.0
        swath_edge = float(rng.choice([15.0, 20.0, 25.0]))
        # Images of 120 to 240 cells: 30 of them can reach 95% of the reachable cells, so every branch is taken.
        for position in rng.choice(np.arange(start, start + 250), size=int(rng.integers(120, 240)), replace=False):
            rows.append(
                (satellite, acquired, REGION[position], float(rng.uniform(0, 30)), swath_edge, downlinked, wait)
            )
    return pd.DataFrame(rows, columns=COLUMNS).sample(frac=1.0, random_state=seed).reset_index(drop=True)


CASES = {
    "1.5.3-defaults": ("1.5.3", {}),
    "1.5.6-defaults": ("1.5.6", {}),
    "oracle-cloud": ("1.5.6", {"cloud_knowledge": "oracle", "shift_step_min": 60}),
    "clear-sky-planning": ("1.5.6", {"cloud_planning": "clear", "shift_step_min": 60}),
    "legacy-fixed-scale": (
        "1.5.6",
        {"time_objective": "legacy", "decay_scale": "fixed", "decay_rate": 1.0, "shift_step_min": 60},
    ),
    "independent-cloud-small-budget": (
        "1.5.6",
        {"cloud_model": "iid", "budget": 5, "random_draws": 3, "seed": 7, "shift_step_min": 60},
    ),
    "short-window-other-levels": (
        "1.5.6",
        {
            "window_hours": 36,
            "shift_hours": 12,
            "shift_step_min": 30,
            "coverage_targets_pct": (90.0, 99.0),
            "reachable_level_pct": 90.0,
            "min_onset_pct": 80.0,
            "floor_margin_min": 45.0,
            "cloud_probability": 0.6,
        },
    ),
}
TABLE_SETTINGS = (
    "budget",
    "decay_rate",
    "decay_scale",
    "window_hours",
    "shift_hours",
    "shift_step_min",
    "coverage_targets_pct",
    "reachable_level_pct",
    "random_draws",
    "seed",
    "cloud_knowledge",
    "cloud_planning",
    "time_objective",
)
SUMMARY_SETTINGS = (
    "budget",
    "decay_rate",
    "decay_scale",
    "window_hours",
    "shift_hours",
    "shift_step_min",
    "coverage_targets_pct",
    "reachable_level_pct",
    "min_onset_pct",
    "floor_margin_min",
    "min_evaluable_onsets",
)


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def case(request, tmp_path_factory):
    release, options = CASES[request.param]
    events = synthetic_window(seed=21)
    out_dir = tmp_path_factory.mktemp(request.param)
    registered = run_cell7c_onsets(
        release, events, REGION, out_dir, orbit_period_min=ORBIT_MIN, median_altitude_km=ALTITUDE_KM, **options
    )
    table = EventTable.from_refined_events(events, REGION)
    probability = float(options.get("cloud_probability", 0.35))
    seed = int(options.get("seed", 42))
    if options.get("cloud_model", "correlated") == "correlated":
        latitude, longitude = cell_centres(REGION)
        is_cloudy = correlated_cloud_mask(table, probability, seed, LatentCloudField(latitude, longitude, 50.0), 3.0)
    else:
        is_cloudy = independent_cloud_mask(table, probability, seed)
    assert np.array_equal(is_cloudy, registered["cloud_mask"].flags)
    settings = OnsetSettings(
        orbit_period_min=ORBIT_MIN,
        cloud_probability=np.full(len(REGION), probability),
        **{name: options[name] for name in TABLE_SETTINGS if name in options},
    )
    ours = onset_table(table, is_cloudy, settings)
    summary = onset_summary(
        ours,
        median_altitude_km=ALTITUDE_KM,
        **{name: options[name] for name in SUMMARY_SETTINGS if name in options},
    )
    return registered, ours, summary, out_dir, options


def test_onset_table_equals_registered(case):
    registered, ours, _, out_dir, options = case
    pd.testing.assert_frame_equal(ours, registered["onset_df"], check_exact=True)
    assert ours.to_csv(index=False) == (out_dir / "arf_onset_robustness.csv").read_text()
    assert len(ours) >= 25 and ours["n_events"].min() > 0  # every onset has images
    for sky in ("clear", "cloud"):
        images = [column for column in ours.columns if column.startswith(f"earliest_completion_{sky}_images")]
        if options.get("budget", 30) >= 30:  # the ECC reaches the level at some onsets
            assert ours[images[0]].notna().any()
        else:  # five images never reach it: the other branch
            assert ours[images[0]].isna().all()


def test_onset_summary_equals_registered(case):
    registered, _, summary, out_dir, _ = case
    assert json.dumps(summary, indent=2) == (out_dir / "arf_onset_robustness_summary.json").read_text()
    assert summary == registered["onset_robustness_summary"]


def test_branches_are_reached():
    """The synthetic window gives onsets at which FCFS needs one image, and onsets where the ECC saves images."""
    events = synthetic_window(seed=21)
    table = EventTable.from_refined_events(events, REGION)
    settings = OnsetSettings(orbit_period_min=ORBIT_MIN, cloud_probability=np.full(len(REGION), 0.35))
    is_clear = np.zeros(table.n_observations, bool)
    summary = onset_summary(onset_table(table, is_clear, settings), median_altitude_km=ALTITUDE_KM)
    clear = summary["h1b_image_economy"]["series"]["clear"]
    assert clear["n_onsets_evaluated"] > 0 and clear["n_success"] > 0
    assert summary["h1_revised"]["series"]["earliest_completion_clear"]["n_within_margin"] > 0
