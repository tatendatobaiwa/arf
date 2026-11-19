"""Unit tests of arf.selection on cases small enough to check by hand."""

import numpy as np
import pandas as pd
import pytest

from arf.selection import ImageSet, evaluate_onset, greedy_cover, onset_grid


def image_set(spec, n_cells):
    """Build an ImageSet from (acquired_min, wait_min, cells) triples."""
    return ImageSet(
        acquired_min=np.array([s[0] for s in spec], dtype=np.int64),
        wait_min=np.array([s[1] for s in spec], dtype=float),
        cells=tuple(np.array(s[2], dtype=np.int64) for s in spec),
        n_cells=n_cells,
    )


def test_greedy_takes_largest_gain_then_earlier_candidate():
    cells = [np.array([0, 1]), np.array([2, 3]), np.array([0, 1, 2])]
    result = greedy_cover([0, 1, 2], cells, n_cells=4, required=4)
    # Image 2 covers three cells and goes first; images 0 and 1 then add nothing and one cell.
    assert result.chosen == (2, 1) and result.complete and result.covered == 4
    tie = greedy_cover([1, 0], cells, n_cells=4, required=2)
    assert tie.chosen == (1,)  # equal gains: the candidate listed first


def test_greedy_respects_budget_and_stops_when_nothing_is_gained():
    cells = [np.array([0]), np.array([1]), np.array([2]), np.array([0, 1])]
    assert not greedy_cover([0, 1, 2], cells, 3, required=3, budget=2).complete
    assert greedy_cover([0, 1, 2], cells, 3, required=3, budget=3).complete
    stuck = greedy_cover([0, 3], cells, 3, required=3)
    assert not stuck.complete and stuck.covered == 2 and len(stuck.chosen) == 1


def test_floor_fcfs_and_ecc_on_a_hand_case():
    # Ten cells. Delivery times after an onset at 0: A 10+5, B 20+5, C 30+5, D 40+60.
    images = image_set(
        [
            (10, 5, [0, 1, 2]),  # A
            (20, 5, [2, 3, 4]),  # B
            (30, 5, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),  # C covers everything
            (40, 60, [9]),  # D
        ],
        n_cells=10,
    )
    result = evaluate_onset(images, 0, budgets=(1, 30, None))
    assert result.reachable == 10 and result.required == 10  # ceil(0.95 * 10)
    assert result.floor_min == 35.0  # only C completes the region
    assert result.fcfs_images == 3  # A, B and C each add cells
    assert result.completion[30].time_min == 35.0 and result.completion[30].images == 1
    assert result.completion[1].images == 1 and result.completion[None].images == 1


def test_budget_too_small_gives_no_completion_and_window_limits_candidates():
    images = image_set([(0, 0, [0]), (5, 0, [1]), (10, 0, [2]), (2000, 0, [3])], n_cells=4)
    result = evaluate_onset(images, 0, budgets=(2, 3))
    assert result.reachable == 3  # the image at minute 2000 is outside the 24 h horizon
    assert result.completion[2].time_min is None and result.completion[3].time_min == 10.0
    assert evaluate_onset(images, 1500, budgets=(1,)).reachable == 1
    assert evaluate_onset(images, 3000) is None


def test_undelivered_images_count_as_reachable_but_cannot_be_chosen():
    images = image_set([(0, 0, [0, 1]), (5, np.inf, [2, 3])], n_cells=4)
    result = evaluate_onset(images, 0)
    assert result.reachable == 4 and result.delivered_images == 1
    assert result.floor_min is None and result.fcfs_images is None and result.completion[30].time_min is None


def test_image_set_from_refined_events_orders_and_filters():
    events = pd.DataFrame(
        {
            "sat": ["B", "A", "A", "A", "B"],
            "t_min": [5, 9, 9, 3, 5],
            "tile": ["x", "y", "z", "x", "y"],
            "A_nadir_deg": [0.0, 10.0, 30.0, 5.0, 0.0],  # 30 deg is past the 20 deg limit: zero utility
            "theta_max_deg": [20.0] * 5,
            "D_downlink": [1.0, 1.0, 1.0, 1.0, 0.0],  # the last row is never downlinked: zero utility
            "T_wait_min": [np.nan, 12.0, 12.0, 7.0, np.nan],
        }
    )
    images = ImageSet.from_refined_events(events)
    assert len(images) == 3 and images.n_cells == 3
    assert images.acquired_min.tolist() == [3, 9, 5]  # satellite A (3, 9), then satellite B
    assert images.wait_min.tolist() == [7.0, 12.0, np.inf]
    assert [c.tolist() for c in images.cells] == [[0], [1], [0]]  # cells indexed by first appearance: x, y, z
    assert images.images_with_repeated_cells == 0
    with pytest.raises(ValueError, match="lack the columns"):
        ImageSet.from_refined_events(events.drop(columns="tile"))


def test_onset_grid_is_the_protocol_grid():
    grid = onset_grid()
    assert len(grid) == 73 and grid[0] == 0 and grid[-1] == 1440 and grid[1] == 20


def test_run_files_are_read_exactly():
    """Whatever the pandas version, every number of a run's CSV is read back as the value that was written."""
    import io

    from arf.runs import read_csv_exact

    values = np.random.default_rng(0).uniform(0, 30, 20_000)
    text = pd.DataFrame({"a": values}).to_csv(index=False)
    assert np.array_equal(read_csv_exact(io.StringIO(text))["a"].to_numpy(), values)
