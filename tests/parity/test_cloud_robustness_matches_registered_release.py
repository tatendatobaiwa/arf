"""arf.report.cloud_robustness (H2 and its sensitivity) against the registered releases themselves.

Cell 6 of releases 1.5.3 and 1.5.6 up to its first selection, Cell 7C's helpers, and its sections 4 and 5 (cloud
robustness over realisations, and the sweep of the cloud model) are executed unchanged (``registered.py``) on the
synthetic 48-hour window of the onset tests. They write ``arf_cloud_robustness.csv``, ``arf_cloud_sensitivity.csv``
and ``arf_cloud_robustness_summary.json``; release 2 must give the same tables, value for value, and the same CSV and
JSON text. The configurations take every branch: both releases' defaults, the independent model as the run's model,
the clear-sky planning rule, the legacy objective on the fixed time scale, no cloud at all (H2 not evaluable), a
small budget with another level, a set margin and no sweep, and too few realisations (nothing written). Fewer
realisations than the releases' 20 and 10 keep the tests short; the code does not depend on the number.
"""

import json

import pandas as pd
import pytest

from arf.events import EventTable
from arf.report.cloud_robustness import (
    CloudModel,
    CloudRobustnessSettings,
    cloud_robustness_summary,
    cloud_robustness_table,
    sensitivity_rows,
    summary_json,
)
from arf.report.onsets_summary import orbital_period_min

h3 = pytest.importorskip("h3")
from registered import run_cell7c_cloud  # noqa: E402
from test_cloud_matches_registered_release import REGION  # noqa: E402
from test_onsets_matches_registered_release import ALTITUDE_KM, ORBIT_MIN, synthetic_window  # noqa: E402

SHORT = {"cloud_seeds": 4, "sweep_lengths_km": (0.0, 25.0, 50.0), "sweep_times_h": (1.0, 3.0), "sweep_seeds": 2}
CASES = {
    "1.5.3-defaults": ("1.5.3", {}),
    "1.5.6-defaults": ("1.5.6", {}),
    "independent-model": ("1.5.6", {"cloud_model": "iid", "seed": 7}),
    "clear-sky-planning": ("1.5.6", {"cloud_planning": "clear"}),
    "legacy-fixed-scale": ("1.5.6", {"time_objective": "legacy", "decay_scale": "fixed", "decay_rate": 1.0}),
    "no-cloud": ("1.5.6", {"cloud_probability": 0.0, "sweep_lengths_km": (0.0, 50.0), "sweep_times_h": ()}),
    "small-budget-other-level-no-sweep": (
        "1.5.6",
        {
            "budget": 5,
            "reachable_level_pct": 90.0,
            "floor_margin_min": 45.0,
            "cloud_probability": 0.6,
            "sweep_lengths_km": (),
            "sweep_times_h": (),
        },
    ),
}


def settings_for(options):
    names = {
        "cloud_seeds": "realisations",
        "sweep_seeds": "sweep_realisations",
        "sweep_lengths_km": "sweep_lengths_km",
        "sweep_times_h": "sweep_times_h",
        "budget": "budget",
        "decay_rate": "decay_rate",
        "decay_scale": "decay_scale",
        "time_objective": "time_objective",
        "seed": "seed",
        "cloud_model": "cloud_model",
        "cloud_probability": "cloud_probability",
        "reachable_level_pct": "reachable_level_pct",
        "cloud_planning": "cloud_planning",
    }
    return CloudRobustnessSettings(
        orbit_period_min=ORBIT_MIN, **{ours: options[theirs] for theirs, ours in names.items() if theirs in options}
    )


def release2(events, options):
    settings = settings_for(options)
    table = EventTable.from_refined_events(events, REGION)
    model = CloudModel(table, settings)
    robustness = cloud_robustness_table(table, settings, model)
    if robustness is None:
        return settings, None, None, None
    sensitivity = sensitivity_rows(table, settings, model)
    margin = options.get("floor_margin_min") or orbital_period_min(ALTITUDE_KM)
    summary = cloud_robustness_summary(robustness, settings, margin, model.record(), sensitivity)
    return settings, robustness, sensitivity, summary


@pytest.fixture(scope="module", params=list(CASES), ids=list(CASES))
def case(request, tmp_path_factory):
    release, options = CASES[request.param]
    options = {**SHORT, **options}
    events = synthetic_window(seed=21)
    out_dir = tmp_path_factory.mktemp(request.param)
    registered = run_cell7c_cloud(
        release, events, REGION, out_dir, orbit_period_min=ORBIT_MIN, median_altitude_km=ALTITUDE_KM, **options
    )
    return registered, release2(events, options), out_dir, request.param


def test_realisation_table_equals_registered(case):
    registered, (_, table, _, _), out_dir, _ = case
    pd.testing.assert_frame_equal(table, registered["cloud_seed_df"], check_exact=True)
    assert table.to_csv(index=False) == (out_dir / "arf_cloud_robustness.csv").read_text()


def test_sensitivity_equals_registered(case):
    registered, (_, _, sensitivity, _), out_dir, _ = case
    written = out_dir / "arf_cloud_sensitivity.csv"
    if sensitivity is None:
        assert registered["cloud_sensitivity_rows"] == [] and not written.exists()
        return
    assert sensitivity == registered["cloud_sensitivity_rows"]
    assert pd.DataFrame(sensitivity).to_csv(index=False) == written.read_text()


def test_summary_equals_registered(case):
    registered, (_, _, _, summary), out_dir, _ = case
    assert summary_json(summary) == (out_dir / "arf_cloud_robustness_summary.json").read_text()
    assert json.loads(summary_json(summary)) == json.loads(summary_json(registered["cloud_robustness_summary"]))


def test_each_case_takes_its_branch(case):
    """Check that the cases differ: H2 reaches each status, the sweep runs or not, H1 under cloud has both outcomes."""
    _, (settings, table, sensitivity, summary), _, name = case
    status = summary["h2"]["time_aware"]["status"]
    if name == "no-cloud":
        assert status == "NOT EVALUABLE" and summary["h2"]["time_aware"]["replans_total"] == 0
        assert table["cloudy_rows_pct"].eq(0.0).all()
    else:
        assert status in ("SUPPORTED", "NOT SUPPORTED") and table["time_aware_replans"].sum() > 0
    assert ("sensitivity" in summary) == (name not in ("small-budget-other-level-no-sweep",))
    if sensitivity:
        assert sensitivity[0]["model"] == "iid"
        assert any(row["primary"] for row in sensitivity) == (settings.cloud_model == "correlated")
    tag = settings.level_tag
    if name == "small-budget-other-level-no-sweep":  # five images never reach the level
        assert table[f"ecc_t{tag}r_min"].isna().all()
    else:
        assert table[f"ecc_t{tag}r_min"].notna().any()


def test_too_few_realisations_write_nothing(tmp_path):
    events = synthetic_window(seed=21)
    options = {**SHORT, "cloud_seeds": 1}
    registered = run_cell7c_cloud(
        "1.5.6", events, REGION, tmp_path, orbit_period_min=ORBIT_MIN, median_altitude_km=ALTITUDE_KM, **options
    )
    assert registered["cloud_robustness_summary"] == {} and not list(tmp_path.iterdir())
    assert release2(events, options)[1] is None


def test_cloud_known_in_advance_is_refused():
    with pytest.raises(ValueError, match="not ported"):
        CloudRobustnessSettings(orbit_period_min=ORBIT_MIN, cloud_knowledge="oracle")
