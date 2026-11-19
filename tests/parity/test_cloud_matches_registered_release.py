"""arf.events and arf.cloud against the registered releases themselves.

Cell 6 of releases 1.5.3 and 1.5.6 is executed unchanged (``registered.py``) on synthetic refined events, and its
EventStore and cloud masks are compared, element by element, with those of release 2: the same seed must give the same
cloudy observations, for the correlated model at several correlation lengths and times, for per-cell probabilities,
and for the independent model.
"""

import numpy as np
import pandas as pd
import pytest

from arf.cloud import LatentCloudField, correlated_cloud_mask, independent_cloud_mask
from arf.events import EventTable
from arf.grid import cell_centres, sorted_region_cells

h3 = pytest.importorskip("h3")
from registered import run_cell6_events_and_cloud  # noqa: E402

REGION = sorted_region_cells(h3.grid_disk(h3.latlng_to_cell(-21.5, 23.5, 7), 20))  # 1,261 cells, about 100 km across
COLUMNS = ["sat", "t_min", "tile", "A_nadir_deg", "theta_max_deg", "D_downlink", "T_wait_min"]


def synthetic_events(seed, n_images=160):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_images):
        satellite = f"SAT-{rng.integers(100, 140)}"
        acquired = int(rng.integers(0, 60)) * 30  # few distinct minutes: several images share a time
        start = int(rng.integers(0, len(REGION) - 80))
        wait = float(rng.choice([15.0, 30.0, 94.0])) if rng.random() > 0.1 else np.nan
        downlinked = 1.0 if rng.random() > 0.1 else 0.0
        swath_edge = float(rng.choice([15.0, 20.0, 25.0]))
        for position in rng.choice(np.arange(start, start + 80), size=50, replace=False):
            rows.append(
                (satellite, acquired, REGION[position], float(rng.uniform(0, 30)), swath_edge, downlinked, wait)
            )
    return pd.DataFrame(rows, columns=COLUMNS).sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture(scope="module", params=["1.5.3", "1.5.6"])
def case(request):
    events = synthetic_events(seed=11)
    registered = run_cell6_events_and_cloud(request.param, events, REGION)
    table = EventTable.from_refined_events(events, REGION)
    return registered, table


def test_event_table_equals_registered_event_store(case):
    registered, table = case
    store = registered["event_data"]
    assert [(s, int(t)) for s, t in zip(table.satellites, table.acquired_min, strict=True)] == store.keys_
    assert np.array_equal(table.offsets, store.offsets)
    assert np.array_equal(table.cell_index, store.tile_idx)
    assert np.array_equal(table.utility, store.util)  # bit for bit
    waits = [None if np.isnan(w) else float(w) for w in table.wait_min]
    assert waits == [store.wait_lookup[key] for key in store.keys_]
    assert (table.utility == 0).any() and (table.utility > 0).any()


@pytest.mark.parametrize(
    ("length_km", "time_h", "seed"),
    [(50.0, 3.0, 42), (50.0, 3.0, 43), (10.0, 3.0, 42), (200.0, 3.0, 42), (50.0, 0.0, 42), (50.0, 12.0, 61)],
)
def test_correlated_mask_equals_registered(case, length_km, time_h, seed):
    registered, table = case
    expected = registered["build_cloud_mask"](
        registered["event_data"], 0.35, seed, model="correlated", length_km=length_km, time_h=time_h
    ).flags
    latitude, longitude = cell_centres(REGION)
    field = LatentCloudField(latitude, longitude, length_km)
    got = correlated_cloud_mask(table, 0.35, seed, field, time_h)
    assert np.array_equal(got, expected)
    assert 0.0 < got.mean() < 1.0  # neither empty nor full; a correlated realisation can be far from 0.35


def test_default_mask_of_the_release_is_reproduced(case):
    registered, table = case
    latitude, longitude = cell_centres(REGION)
    got = correlated_cloud_mask(table, 0.35, 42, LatentCloudField(latitude, longitude, 50.0), 3.0)
    assert np.array_equal(got, registered["cloud_mask"].flags)  # the mask the cell builds when it runs


def test_per_cell_probability_equals_registered(case):
    registered, table = case
    probability = np.random.default_rng(5).uniform(0.05, 0.9, len(REGION))
    expected = registered["build_cloud_mask"](
        registered["event_data"], 0.35, 7, model="correlated", length_km=50.0, time_h=3.0, p_tile=probability
    ).flags
    latitude, longitude = cell_centres(REGION)
    got = correlated_cloud_mask(table, probability, 7, LatentCloudField(latitude, longitude, 50.0), 3.0)
    assert np.array_equal(got, expected)
    expected_iid = registered["build_cloud_mask"](registered["event_data"], 0.35, 7, model="iid", p_tile=probability)
    assert np.array_equal(independent_cloud_mask(table, probability, 7), expected_iid.flags)


@pytest.mark.parametrize("seed", [42, 43])
def test_independent_mask_equals_registered(case, seed):
    registered, table = case
    expected = registered["build_cloud_mask"](registered["event_data"], 0.35, seed, model="iid").flags
    assert np.array_equal(independent_cloud_mask(table, 0.35, seed), expected)
