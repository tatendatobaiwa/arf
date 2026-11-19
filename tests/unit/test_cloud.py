"""Unit tests of the cloud model: properties that hold whatever the registered release does."""

import numpy as np
import pandas as pd
import pytest

from arf.cloud import LatentCloudField, correlated_cloud_mask, independent_cloud_mask
from arf.events import EventTable

# A 40 x 40 lattice of points 5 km apart near 21.5 S, 23.5 E, standing in for cell centres.
STEP_DEG = 5.0 / 111.195
LATITUDE, LONGITUDE = (
    a.ravel() for a in np.meshgrid(-21.5 + STEP_DEG * np.arange(40), 23.5 + STEP_DEG * np.arange(40))
)
CELLS = np.array([f"cell{i:05d}" for i in range(LATITUDE.size)])


def one_image_per_time(times):
    """Every cell observed once at each of the given minutes."""
    rows = [(f"S{t}", t, cell, 0.0, 20.0, 1.0, 30.0) for t in times for cell in CELLS]
    columns = ["sat", "t_min", "tile", "A_nadir_deg", "theta_max_deg", "D_downlink", "T_wait_min"]
    return EventTable.from_refined_events(pd.DataFrame(rows, columns=columns), CELLS)


def test_field_has_unit_variance_and_the_stated_correlation():
    field = LatentCloudField(LATITUDE, LONGITUDE, length_km=50.0)
    rng = np.random.default_rng(0)
    everywhere = np.arange(LATITUDE.size)
    samples = np.array([field.at_cells(field.draw(rng), everywhere) for _ in range(400)])
    assert abs(field.draw(rng).std() - 1.0) < 0.15
    # Correlation between points 50 km apart (10 lattice steps along a row): exp(-1) = 0.37.
    grid = samples.reshape(400, 40, 40)
    pairs = [(grid[:, 5, 5], grid[:, 5, 15]), (grid[:, 20, 10], grid[:, 20, 20]), (grid[:, 30, 25], grid[:, 30, 35])]
    correlation = np.mean([np.corrcoef(a, b)[0, 1] for a, b in pairs])
    assert abs(correlation - np.exp(-1.0)) < 0.12


def test_marginal_probability_is_the_one_asked_for():
    events = one_image_per_time([0])
    field = LatentCloudField(LATITUDE, LONGITUDE, length_km=50.0)
    share = np.mean([correlated_cloud_mask(events, 0.35, seed, field, time_h=3.0).mean() for seed in range(300)])
    assert abs(share - 0.35) < 0.03
    assert abs(np.mean([independent_cloud_mask(events, 0.35, seed).mean() for seed in range(20)]) - 0.35) < 0.01


def test_same_minute_sees_the_same_cloud_and_time_decorrelates():
    field = LatentCloudField(LATITUDE, LONGITUDE, length_km=50.0)
    n = CELLS.size
    rows = [(s, 0, cell, 0.0, 20.0, 1.0, 30.0) for s in ("A", "B") for cell in CELLS]
    columns = ["sat", "t_min", "tile", "A_nadir_deg", "theta_max_deg", "D_downlink", "T_wait_min"]
    together = EventTable.from_refined_events(pd.DataFrame(rows, columns=columns), CELLS)
    mask = correlated_cloud_mask(together, 0.35, 3, field, time_h=3.0)
    assert np.array_equal(mask[:n], mask[n:])

    events = one_image_per_time([0, 10, 1440])
    agree_soon, agree_late = [], []
    for seed in range(60):
        mask = correlated_cloud_mask(events, 0.35, seed, field, time_h=3.0)
        agree_soon.append((mask[:n] == mask[n : 2 * n]).mean())
        agree_late.append((mask[:n] == mask[2 * n :]).mean())
    assert np.mean(agree_soon) > 0.9  # ten minutes apart: almost the same cloud
    assert abs(np.mean(agree_late) - (0.35**2 + 0.65**2)) < 0.05  # a day apart: independent


def test_masks_are_reproducible_and_seed_dependent():
    events = one_image_per_time([0, 60])
    field = LatentCloudField(LATITUDE, LONGITUDE, length_km=50.0)
    first = correlated_cloud_mask(events, 0.35, 1, field, time_h=3.0)
    assert np.array_equal(first, correlated_cloud_mask(events, 0.35, 1, field, time_h=3.0))
    assert not np.array_equal(first, correlated_cloud_mask(events, 0.35, 2, field, time_h=3.0))


def test_event_table_rejects_unsorted_cells_and_missing_columns():
    events = pd.DataFrame(
        {
            "sat": ["A"],
            "t_min": [0],
            "tile": ["b"],
            "A_nadir_deg": [0.0],
            "theta_max_deg": [20.0],
            "D_downlink": [1.0],
            "T_wait_min": [5.0],
        }
    )
    with pytest.raises(ValueError, match="sorted"):
        EventTable.from_refined_events(events, np.array(["b", "a"]))
    with pytest.raises(ValueError, match="lack the columns"):
        EventTable.from_refined_events(events.drop(columns="sat"), np.array(["a", "b"]))
    table = EventTable.from_refined_events(events, np.array(["a", "b"]))
    assert len(table) == 1 and table.n_observations == 1 and table.cell_index.tolist() == [1]
