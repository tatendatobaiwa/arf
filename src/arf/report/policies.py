"""The objectives and selectors over the full window, and what each image budget buys (Cell 7C, section 1).

Three tables, each under clear sky and under the run's cloud:

* **Decay sweep** (``arf_decay_sweep.csv``): Cell 6's selector at the budget, without gap-fill, at each decay rate of
  the time-aware objective (rate zero is coverage-first): the completion time (H1 latency), the coverage, and the mean
  quality of the covered cells.
* **Policy comparison** (``arf_policy_comparison.csv``): the floor (every image) and the five selectors of
  :mod:`arf.report.onsets` at the budget: completion time, coverage, quality, and the time to each coverage level.
* **Cost-latency frontier** (``arf_cost_latency_frontier.csv``): for budgets of 1 to 200 images and the run's own,
  the earliest time the earliest-completion cover (ECC) reaches the H1 level of the reachable cells, and each coverage
  level of the region, against the floor.

Under cloud the selectors see the cloud as images arrive (the Fail-Signal selector), unless it is known in advance.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from arf.events import EventTable
from arf.report.onsets import (
    POLICIES,
    OnsetSettings,
    _metrics,
    coverage_times,
    earliest_completion,
    policy_outcome,
    reachable_cells,
    select_greedy,
)

#: Image budgets of the frontier; those above the number of images are left out, and the run's budget is added.
FRONTIER_BUDGETS = (1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200)


def _skies(is_cloudy: np.ndarray):
    return (("clear", None), ("cloud", is_cloudy))


def decay_sweep(
    events: EventTable, is_cloudy: np.ndarray, settings: OnsetSettings, rates: Sequence[float]
) -> pd.DataFrame:
    """Return ``arf_decay_sweep.csv``: Cell 6's selector at rate zero and at every other rate, clear and cloudy."""
    rows = []
    for rate in [0.0, *sorted({float(r) for r in rates} - {0.0})]:
        row: dict = {"decay_rate": rate, "objective": "coverage-first" if rate == 0 else "time-aware"}
        for sky, mask in _skies(is_cloudy):
            outcome = _metrics(events, mask, select_greedy(events, mask, rate, settings), ())
            row[f"h1_latency_{sky}_min"] = outcome["lat"]
            row[f"coverage_{sky}_pct"] = outcome["cov"]
            row[f"mean_quality_{sky}"] = outcome["q"]
        rows.append(row)
    return pd.DataFrame(rows)


def policy_comparison(events: EventTable, is_cloudy: np.ndarray, settings: OnsetSettings) -> pd.DataFrame:
    """Return ``arf_policy_comparison.csv``: the floor and every selector at the budget, clear and cloudy."""
    targets = settings.coverage_targets_pct
    rows = []
    for sky, mask in _skies(is_cloudy):
        floor = coverage_times(events, mask, targets)
        rows.append(
            {
                "sky": sky,
                "policy": "best_possible",
                "h1_latency_min": None,
                "coverage_pct": None,
                "mean_quality": None,
                **{f"t{target:g}_min": floor[target] for target in targets},
            }
        )
        for name in POLICIES:
            outcome = policy_outcome(events, mask, name, settings)
            rows.append(
                {
                    "sky": sky,
                    "policy": name,
                    "h1_latency_min": outcome["lat"],
                    "coverage_pct": outcome["cov"],
                    "mean_quality": outcome["q"],
                    **{f"t{target:g}_min": outcome["t"][target] for target in targets},
                }
            )
    return pd.DataFrame(rows)


def frontier_budgets(events: EventTable, budget: int) -> list[int]:
    """The budgets of the frontier for a window: those up to its number of images, and the run's own."""
    return sorted({k for k in FRONTIER_BUDGETS if k <= len(events)} | {int(budget)})


def cost_latency_frontier(events: EventTable, is_cloudy: np.ndarray, settings: OnsetSettings) -> pd.DataFrame:
    """Return ``arf_cost_latency_frontier.csv``: the ECC's completion time at each budget, for each target."""
    level = float(settings.reachable_level_pct)
    n_cells = len(events.region_cells)
    budgets = frontier_budgets(events, settings.budget)
    rows = []
    for sky, mask in _skies(is_cloudy):
        reach = reachable_cells(events, mask)
        targets = [("reachable", level, reach)] + [("roi", t, n_cells) for t in settings.coverage_targets_pct]
        for kind, target, denominator in targets:
            floor = coverage_times(events, mask, (target,), denominator=denominator)[target] if denominator else None
            for k in budgets:
                if denominator:
                    completed, _, used = earliest_completion(events, mask, k, target, denominator)
                else:
                    completed, used = None, 0
                rows.append(
                    {
                        "sky": sky,
                        "target": kind,
                        "target_pct": target,
                        "denominator_tiles": denominator,
                        "k": k,
                        "t_complete_min": completed,
                        "images_used": used if completed is not None else None,
                        "floor_min": floor,
                    }
                )
    return pd.DataFrame(rows)
