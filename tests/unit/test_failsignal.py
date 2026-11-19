"""Unit tests of delivery, greedy selection and the Fail-Signal selector on cases small enough to check by hand."""

import numpy as np
import pandas as pd

from arf.comparison import cloud_trial, covered_cells
from arf.delivery import decay_time_scale, delivery_times, floor_time, time_weights
from arf.events import EventTable
from arf.failsignal import adaptive_selection, open_loop_plan
from arf.greedy import greedy_selection

CELLS = np.array(["c0", "c1", "c2", "c3", "c4", "c5"])
COLUMNS = ["sat", "t_min", "tile", "A_nadir_deg", "theta_max_deg", "D_downlink", "T_wait_min"]


def table(images):
    """Build an EventTable from (satellite, minute, wait, cells), every observation at nadir.

    As in the releases, an image that is never delivered (wait ``nan``) has a downlink reward of zero, and so
    a utility of zero.
    """
    rows = [
        (sat, minute, cell, 0.0, 20.0, 0.0 if np.isnan(wait) else 1.0, wait)
        for sat, minute, wait, cells in images
        for cell in cells
    ]
    return EventTable.from_refined_events(pd.DataFrame(rows, columns=COLUMNS), CELLS)


# Four images in satellite order A, B, C, D. Delivery: A at 10, B at 20, C at 30, D never.
EVENTS = table(
    [
        ("A", 0, 10.0, ["c0", "c1", "c2"]),
        ("B", 10, 10.0, ["c2", "c3"]),
        ("C", 20, 10.0, ["c3", "c4", "c5"]),
        ("D", 30, np.nan, ["c0", "c1", "c2", "c3", "c4", "c5"]),
    ]
)


def cloud(*cloudy_rows):
    is_cloudy = np.zeros(EVENTS.n_observations, dtype=bool)
    is_cloudy[list(cloudy_rows)] = True
    return is_cloudy


def test_delivery_times_and_floor():
    assert delivery_times(EVENTS).tolist() == [10.0, 20.0, 30.0, np.inf]
    assert floor_time(EVENTS, 95.0) == 30.0  # all six cells are first seen once C is delivered
    assert floor_time(EVENTS, 50.0) == 10.0  # A alone gives three of six
    assert floor_time(table([("D", 0, np.nan, ["c0"])])) is None


def test_time_weights_use_the_longer_of_orbit_and_floor():
    assert decay_time_scale(EVENTS, orbit_period_min=5.0) == 30.0  # the floor is longer than the orbit
    assert decay_time_scale(EVENTS, orbit_period_min=95.0) == 95.0
    assert decay_time_scale(EVENTS, orbit_period_min=5.0, scale="fixed") == 5.0
    weights = time_weights(EVENTS, decay_rate=3.0, orbit_period_min=5.0)
    assert np.allclose(weights[:3], np.exp(-3.0 * np.array([10.0, 20.0, 30.0]) / 30.0))
    assert weights[3] == 0.0  # never delivered


def test_greedy_clear_sky_takes_the_largest_gain_and_stops_when_nothing_adds():
    selection = greedy_selection(EVENTS, budget=10)
    # A and C each add three cells, A first on the tie; then B and the undelivered D add nothing.
    assert selection.images == (0, 2) and selection.covered_cells == 6
    assert greedy_selection(EVENTS, budget=1).images == (0,)


def test_known_cloud_removes_observations_and_gap_fill_compensates():
    # Rows: A = 0,1,2; B = 3,4; C = 5,6,7; D = 8..13. Cloud over A's view of c2.
    is_cloudy = cloud(2)
    one = greedy_selection(EVENTS, budget=1, is_cloudy=is_cloudy, gap_fill=False)
    assert one.images == (2,) and one.covered_cells == 3  # C clears three cells; A clears two
    filled = greedy_selection(EVENTS, budget=1, is_cloudy=is_cloudy, gap_fill=True)
    assert filled.images == (2, 0, 1) and filled.compensatory == 2 and filled.covered_cells == 6


def test_open_loop_plan_ignores_undelivered_images():
    probability = np.full(6, 0.35)
    assert open_loop_plan(EVENTS, 2, probability) == (0, 2)  # A and C: three new cells each, A first on the tie
    assert open_loop_plan(EVENTS, 0, probability) == ()


def test_adaptive_selection_in_clear_sky_follows_the_plan_without_fail_signals():
    probability = np.full(6, 0.35)
    selection = adaptive_selection(EVENTS, 2, cloud(), probability)
    assert selection.images == (0, 2) and selection.replans == 0 and selection.compensatory == 0
    assert selection.covered_cells == 6


def test_fail_signal_triggers_a_replan_and_compensation():
    probability = np.full(6, 0.35)
    # A arrives with c2 cloudy: a fail-signal. With one image left, the planner now prefers C (three missing
    # cells) to B (two: c2, c3); c2 is then still missing and B arrives before C, so it cannot compensate.
    selection = adaptive_selection(EVENTS, 2, cloud(2), probability)
    assert selection.images == (0, 2) and selection.replans == 1 and selection.compensatory == 0
    assert selection.covered_cells == 5
    # With a budget of one, only A is planned; after it, every arriving image that clears a missing cell is taken.
    compensated = adaptive_selection(EVENTS, 1, cloud(2), probability)
    assert compensated.images == (0, 1, 2) and compensated.compensatory == 2 and compensated.covered_cells == 6
    assert adaptive_selection(EVENTS, 1, cloud(2), probability, gap_fill=False).images == (0,)


def test_no_decision_uses_the_cloud_of_an_image_that_has_not_arrived():
    probability = np.full(6, 0.35)
    # Cloud over C only. The first decision (A) and the plan made after A arrives must not depend on it.
    clear = adaptive_selection(EVENTS, 2, cloud(), probability, gap_fill=False)
    c_cloudy = adaptive_selection(EVENTS, 2, cloud(5, 6, 7), probability, gap_fill=False)
    assert clear.images == c_cloudy.images == (0, 2)
    assert c_cloudy.covered_cells == 3 and c_cloudy.replans == 1


def test_cloud_trial_reports_every_plan_at_the_adaptive_size():
    is_cloudy = cloud(2)  # A's view of c2
    trial = cloud_trial(EVENTS, is_cloudy, np.full(6, 0.35), budget=1, orbit_period_min=95.0)
    assert trial.cloudy_share_percent == 100.0 / 14 and trial.reachable_cells == 6
    result = trial.objectives["coverage_first"]
    # The adaptive selector plans A, sees the fail-signal, and compensates with B and C: three images.
    assert result.plans["adaptive"] == result.plans["adaptive_clear_planning"]
    assert (result.plans["adaptive"].images, result.plans["adaptive"].covered_cells) == (3, 6)
    assert (result.replans, result.compensatory) == (1, 2)
    # At that size: the open-loop planner takes A, C, then B; the clear-sky plan stops at A and C and misses c2;
    # the greedy that knows the cloud takes C, A, then B.
    assert (result.plans["open_loop_expected"].images, result.plans["open_loop_expected"].covered_cells) == (3, 6)
    assert (result.plans["open_loop_clear"].images, result.plans["open_loop_clear"].covered_cells) == (2, 5)
    assert (result.plans["oracle"].images, result.plans["oracle"].covered_cells) == (3, 6)
    assert covered_cells(EVENTS, is_cloudy, (0, 2)) == 5
