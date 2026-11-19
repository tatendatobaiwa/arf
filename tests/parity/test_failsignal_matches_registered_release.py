"""arf.delivery, arf.greedy, arf.failsignal and arf.comparison against the registered releases themselves.

Cell 6 of releases 1.5.3 and 1.5.6 is executed unchanged, with Cell 7C's cloud trial (``registered.py``), on synthetic
refined events with real cell geometry. Compared with release 2: the time weights (bit for bit); the images each
selector chooses, in order, in clear sky and under known cloud, with and without gap-filling, for the coverage-first,
facility and legacy objectives; the images the Fail-Signal selector takes, its fail-signals and its compensatory
images, for both planning rules; and every figure of the cloud trial for several cloud realisations.
"""

import numpy as np
import pytest

from arf.cloud import LatentCloudField, correlated_cloud_mask
from arf.comparison import cloud_trial
from arf.delivery import delivery_times, floor_time, time_weights
from arf.events import EventTable
from arf.failsignal import adaptive_selection, open_loop_plan
from arf.greedy import greedy_selection
from arf.grid import cell_centres

h3 = pytest.importorskip("h3")
from registered import run_cell6_selection_and_trial  # noqa: E402
from test_cloud_matches_registered_release import REGION, synthetic_events  # noqa: E402

ORBIT_MIN = 94.0
#: Registered plan name -> release 2 plan name.
PLAN_NAMES = {
    "adaptive": "adaptive",
    "openloop_expected": "open_loop_expected",
    "openloop_clear": "open_loop_clear",
    "oracle": "oracle",
    "failsignal_v140": "adaptive_clear_planning",
    "openloop_expected_v140n": "open_loop_expected_at_that_size",
}
CASES = [("1.5.3", "facility", 30), ("1.5.6", "facility", 30), ("1.5.6", "legacy", 30), ("1.5.6", "facility", 8)]


@pytest.fixture(scope="module", params=CASES, ids=[f"{r}-{o}-k{k}" for r, o, k in CASES])
def case(request):
    release, time_objective, budget = request.param
    events = synthetic_events(seed=11)
    registered = run_cell6_selection_and_trial(
        release, events, REGION, budget=budget, orbit_period_min=ORBIT_MIN, time_objective=time_objective
    )
    table = EventTable.from_refined_events(events, REGION)
    latitude, longitude = cell_centres(REGION)
    field = LatentCloudField(latitude, longitude, 50.0)
    return registered, table, field, time_objective, budget


def keys(table, images):
    return [(table.satellites[i], int(table.acquired_min[i])) for i in images]


def test_delivery_floor_and_time_weights(case):
    registered, table, _, _, _ = case
    store = registered["event_data"]
    assert np.array_equal(delivery_times(table), registered["event_delivery_times"](store))
    assert floor_time(table, 95.0) == registered["floor_time"](store, 95.0)
    for rate in (1.0, 3.0):
        assert np.array_equal(time_weights(table, rate, ORBIT_MIN), registered["time_aware_weights"](store, rate))
    assert not np.isfinite(delivery_times(table)).all()  # the data hold images that are never delivered


@pytest.mark.parametrize("time_aware", [False, True])
@pytest.mark.parametrize("known_cloud", [False, True])
@pytest.mark.parametrize("budget", [5, 30, 400])
def test_greedy_selection_equals_registered(case, time_aware, known_cloud, budget):
    registered, table, field, time_objective, _ = case
    store = registered["event_data"]
    mask = registered["build_cloud_mask"](store, 0.35, 44) if known_cloud else None
    is_cloudy = correlated_cloud_mask(table, 0.35, 44, field, 3.0) if known_cloud else None
    weights = time_weights(table, 3.0, ORBIT_MIN) if time_aware else None
    for gap_fill in (True, False):
        expected, expected_utility, _, _ = registered["run_submodular_greedy_selection"](
            store, budget, cloud_mask=mask, gap_fill=gap_fill, time_aware=time_aware, latency_decay_rate=3.0
        )
        got = greedy_selection(table, budget, is_cloudy, weights, time_objective, gap_fill)
        assert keys(table, got.images) == expected
        assert {REGION[i]: float(got.best_utility[i]) for i in np.flatnonzero(got.best_utility > 0)} == expected_utility


@pytest.mark.parametrize("time_aware", [False, True])
@pytest.mark.parametrize("planning", ["expected", "clear"])
@pytest.mark.parametrize("seed", [42, 47, 55])
def test_adaptive_selection_equals_registered(case, time_aware, planning, seed):
    registered, table, field, time_objective, budget = case
    store = registered["event_data"]
    mask = registered["build_cloud_mask"](store, 0.35, seed)
    is_cloudy = correlated_cloud_mask(table, 0.35, seed, field, 3.0)
    probability = np.full(len(REGION), 0.35)
    weights = time_weights(table, 3.0, ORBIT_MIN) if time_aware else None
    for gap_fill in (True, False):
        expected, expected_utility, _, _ = registered["simulate_adaptive_selection"](
            store, budget, mask, gap_fill=gap_fill, time_aware=time_aware, latency_decay_rate=3.0, planning=planning
        )
        stats = dict(registered["last_adaptive_run"])
        got = adaptive_selection(table, budget, is_cloudy, probability, weights, time_objective, planning, gap_fill)
        assert keys(table, got.images) == expected
        assert (got.replans, got.compensatory) == (stats["replans"], stats["compensatory"])
        assert {REGION[i]: float(got.best_utility[i]) for i in np.flatnonzero(got.best_utility > 0)} == expected_utility


def test_open_loop_plan_equals_registered(case):
    registered, table, _, _, _ = case
    probability = np.random.default_rng(2).uniform(0.05, 0.9, len(REGION))
    for size, weights in ((12, None), (40, time_weights(table, 3.0, ORBIT_MIN))):
        registered_weights = (
            None if weights is None else registered["time_aware_weights"](registered["event_data"], 3.0)
        )
        expected = registered["plan_expected_coverage"](registered["event_data"], size, probability, registered_weights)
        assert keys(table, open_loop_plan(table, size, probability, weights)) == expected


@pytest.mark.parametrize("seed", [42, 43, 49, 58])
def test_cloud_trial_equals_registered(case, seed):
    registered, table, field, time_objective, budget = case
    row = registered["_cloud_trial"](
        registered["build_cloud_mask"](registered["event_data"], 0.35, seed), with_h1=False
    )
    is_cloudy = correlated_cloud_mask(table, 0.35, seed, field, 3.0)
    trial = cloud_trial(table, is_cloudy, np.full(len(REGION), 0.35), budget, ORBIT_MIN, time_objective=time_objective)
    assert trial.cloudy_share_percent == row["cloudy_rows_pct"]
    assert trial.reachable_cells == row["reachable_tiles"]
    compared = 0
    for objective, result in trial.objectives.items():
        assert result.replans == row[f"{objective}_replans"]
        assert result.compensatory == row[f"{objective}_compensatory"]
        for registered_name, name in PLAN_NAMES.items():
            assert result.plans[name].images == row[f"{objective}_{registered_name}_images"]
            assert (
                100.0 * result.plans[name].covered_cells / len(REGION) == row[f"{objective}_{registered_name}_cov_pct"]
            )
            compared += 1
    assert compared == 12
