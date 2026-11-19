"""arf.report.policies and arf.report.optimality against the registered releases themselves (Cell 7C, sections 1, 3).

Cell 6 of releases 1.5.3 and 1.5.6 up to its first selection, Cell 7C's helpers, and its section 1 (decay sweep,
policy comparison, cost-latency frontier) or section 3 (optimality gap against the exact ILP) are executed unchanged
(``registered.py``) on the synthetic 48-hour window of the onset tests. Release 2 must give the same tables, value for
value, and the same CSV text; the ILP summary must equal the one the releases put into their scenario ledger. The
solver's wall-clock time (``ilp_seconds``) is the one column no rerun reproduces, so it is left out of the comparison.
"""

import io

import numpy as np
import pandas as pd
import pytest

from arf.cloud import LatentCloudField, correlated_cloud_mask, independent_cloud_mask
from arf.events import EventTable
from arf.grid import cell_centres
from arf.report.onsets import OnsetSettings
from arf.report.optimality import IlpSettings, ilp_gap_summary, ilp_gap_table
from arf.report.policies import cost_latency_frontier, decay_sweep, policy_comparison

h3 = pytest.importorskip("h3")
from registered import run_cell7c_ilp, run_cell7c_policies  # noqa: E402
from test_cloud_matches_registered_release import REGION  # noqa: E402
from test_onsets_matches_registered_release import ALTITUDE_KM, ORBIT_MIN, TABLE_SETTINGS, synthetic_window  # noqa: E402

POLICY_CASES = {
    "1.5.3-defaults": ("1.5.3", {}),
    "1.5.6-defaults": ("1.5.6", {}),
    "independent-cloud-small-budget": ("1.5.6", {"cloud_model": "iid", "budget": 5, "random_draws": 3, "seed": 7}),
    "cloud-known-in-advance": ("1.5.6", {"cloud_knowledge": "oracle"}),
    "legacy-fixed-scale-other-levels": (
        "1.5.6",
        {
            "time_objective": "legacy",
            "decay_scale": "fixed",
            "decay_rate": 1.0,
            "coverage_targets_pct": (90.0, 99.0),
            "reachable_level_pct": 90.0,
            "decay_sweep": (0.0, 2.0, 2.0, 0.25),
            "cloud_probability": 0.6,
        },
    ),
}
ILP_CASES = {
    "1.5.3-defaults": ("1.5.3", {}),
    "1.5.6-shrunk-windows": ("1.5.6", {"ilp_max_pairs": 3000, "ilp_instances": 4}),
    "legacy-objective-large-budget": ("1.5.6", {"time_objective": "legacy", "ilp_budget_k": 25, "ilp_instances": 3}),
    "nothing-fits": ("1.5.6", {"ilp_max_pairs": 50, "ilp_instances": 2}),
    "disabled": ("1.5.6", {"ilp_enabled": False}),
}
ILP_SETTINGS = {
    "ilp_enabled": "enabled",
    "ilp_instances": "instances",
    "ilp_horizon_hours": "horizon_hours",
    "ilp_budget_k": "budget",
    "ilp_time_limit_s": "time_limit_s",
    "ilp_max_pairs": "max_pairs",
}


def inputs(options):
    """The synthetic window, its cloud as Cell 6 draws it, and release 2's settings."""
    events = synthetic_window(seed=21)
    table = EventTable.from_refined_events(events, REGION)
    probability = float(options.get("cloud_probability", 0.35))
    seed = int(options.get("seed", 42))
    if options.get("cloud_model", "correlated") == "correlated":
        latitude, longitude = cell_centres(REGION)
        is_cloudy = correlated_cloud_mask(table, probability, seed, LatentCloudField(latitude, longitude, 50.0), 3.0)
    else:
        is_cloudy = independent_cloud_mask(table, probability, seed)
    settings = OnsetSettings(
        orbit_period_min=ORBIT_MIN,
        cloud_probability=np.full(len(REGION), probability),
        **{name: options[name] for name in TABLE_SETTINGS if name in options},
    )
    return events, table, is_cloudy, settings


@pytest.fixture(scope="module", params=list(POLICY_CASES), ids=list(POLICY_CASES))
def policies(request, tmp_path_factory):
    release, options = POLICY_CASES[request.param]
    events, table, is_cloudy, settings = inputs(options)
    out_dir = tmp_path_factory.mktemp(request.param)
    registered = run_cell7c_policies(
        release, events, REGION, out_dir, orbit_period_min=ORBIT_MIN, median_altitude_km=ALTITUDE_KM, **options
    )
    assert np.array_equal(is_cloudy, registered["cloud_mask"].flags)
    ours = {
        "decay_sweep_df": decay_sweep(
            table, is_cloudy, settings, options.get("decay_sweep", (0.5, 1.0, 2.0, 3.0, 5.0))
        ),
        "policy_df": policy_comparison(table, is_cloudy, settings),
        "frontier_df": cost_latency_frontier(table, is_cloudy, settings),
    }
    return registered, ours, out_dir, request.param


FILES = {
    "decay_sweep_df": "arf_decay_sweep.csv",
    "policy_df": "arf_policy_comparison.csv",
    "frontier_df": "arf_cost_latency_frontier.csv",
}


@pytest.mark.parametrize("name", list(FILES))
def test_section_1_tables_equal_registered(policies, name):
    registered, ours, out_dir, _ = policies
    pd.testing.assert_frame_equal(ours[name], registered[name], check_exact=True)
    assert ours[name].to_csv(index=False) == (out_dir / FILES[name]).read_text()


def test_section_1_cases_take_their_branches(policies):
    _, ours, _, case = policies
    sweep, frontier = ours["decay_sweep_df"], ours["frontier_df"]
    assert sweep["objective"].iloc[0] == "coverage-first" and (sweep["objective"].iloc[1:] == "time-aware").all()
    reached = frontier["t_complete_min"].notna()
    assert reached.any() and not reached.all()  # small budgets miss a level that larger ones reach
    if case == "legacy-fixed-scale-other-levels":
        assert list(sweep["decay_rate"]) == [0.0, 0.25, 2.0]  # zero first, duplicates once
        assert "t90_min" in ours["policy_df"].columns


@pytest.fixture(scope="module", params=list(ILP_CASES), ids=list(ILP_CASES))
def ilp(request, tmp_path_factory):
    release, options = ILP_CASES[request.param]
    onset_options = {k: v for k, v in options.items() if k not in ILP_SETTINGS}
    events, table, _, settings = inputs(onset_options)
    out_dir = tmp_path_factory.mktemp(request.param)
    registered = run_cell7c_ilp(
        release, events, REGION, out_dir, orbit_period_min=ORBIT_MIN, median_altitude_km=ALTITUDE_KM, **options
    )
    ilp_settings = IlpSettings(**{ours: options[theirs] for theirs, ours in ILP_SETTINGS.items() if theirs in options})
    table_ = ilp_gap_table(table, ilp_settings, settings, window_hours=48)
    summary = None if table_ is None else ilp_gap_summary(table_, ilp_settings, settings.time_objective)
    return registered, table_, summary, out_dir, request.param


def test_ilp_table_equals_registered(ilp):
    registered, table, _, out_dir, case = ilp
    if case == "disabled":
        assert table is None and "ilp_df" not in registered and not (out_dir / "arf_ilp_gap.csv").exists()
        return
    theirs = registered["ilp_df"]
    pd.testing.assert_frame_equal(
        table.drop(columns="ilp_seconds"), theirs.drop(columns="ilp_seconds"), check_exact=True
    )
    written = pd.read_csv(out_dir / "arf_ilp_gap.csv").drop(columns="ilp_seconds")
    ours_written = pd.read_csv(io.StringIO(table.to_csv(index=False))).drop(columns="ilp_seconds")
    assert ours_written.to_csv(index=False) == written.to_csv(index=False)


def test_ilp_summary_equals_registered(ilp):
    registered, _, summary, _, _ = ilp
    assert (summary or {}) == registered["ilp_summary"]


def test_ilp_cases_take_their_branches(ilp):
    _, table, summary, _, case = ilp
    if case == "disabled":
        return
    statuses = set(table["ilp_status"])
    if case == "nothing-fits":
        assert all(s.startswith("skipped") for s in statuses) and summary["solved"] == 0
        assert summary["coverage_first_ratio_median"] is None
    else:
        assert statuses == {"optimal"} and summary["ecc_instances_exact"] > 0
        assert (table["coverage_first_ratio"] <= 1.0 + 1e-9).all()  # the greedy never beats a proven optimum
    if case == "1.5.6-shrunk-windows":
        assert min(summary["horizons_used_min"]) < 360  # windows were halved to fit
    if case == "legacy-objective-large-budget":
        assert summary["guarantee_note"].startswith("legacy objective")
