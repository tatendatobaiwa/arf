"""How far the greedy selectors are from the exact optimum (Cell 7C, section 3).

On ``instances`` short windows of clear sky spread over the run (``horizon_hours`` each, halved down to 30 minutes
while a window has more than ``max_pairs`` usable observations), the budgeted maximum-utility selection is solved
exactly as a mixed-integer program (SciPy's HiGHS), and the greedy selectors' objectives are divided by the proven
optimum. The time-aware greedy is also measured against the exact optimum of its own objective, the time-weighted
coverage, and the cover step of the earliest-completion cover (ECC) against the exact minimum number of images for
the H1 level of the window's reachable cells. One row per window: ``arf_ilp_gap.csv``; the summary goes into the
run's scenario ledger (``ilp_gap``).

Only ratios against a proven optimum count: a time-limited solve's best solution can be arbitrarily poor. The
solver's time (``ilp_seconds``) is wall-clock time, so no other run reproduces it, and an instance near the time
limit can end differently on another machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

from arf.delivery import time_weights
from arf.events import EventTable
from arf.greedy import greedy_selection
from arf.report.onsets import (
    OnsetSettings,
    completion_time,
    earliest_completion,
    effective_utility,
    reachable_cells,
    select_fcfs,
    subset_by_time,
)

#: The shortest window an instance is shrunk to.
MIN_HORIZON_MIN = 30


@dataclass(frozen=True)
class IlpSettings:
    """The settings of section 3, with the names of the registered releases' configuration.

    Attributes:
        enabled: ``ilp_enabled``.
        instances: ``ilp_instances``.
        horizon_hours: ``ilp_horizon_hours``.
        budget: ``ilp_budget_k``.
        time_limit_s: ``ilp_time_limit_s``, per solve.
        max_pairs: ``ilp_max_pairs``.
    """

    enabled: bool = True
    instances: int = 5
    horizon_hours: int = 6
    budget: int = 10
    time_limit_s: float = 120.0
    max_pairs: int = 100_000


def _pairs(events: EventTable, utility: np.ndarray):
    keep = utility > 0.0
    pair_image = np.repeat(np.arange(len(events)), events.observations_per_image)[keep]
    return keep, pair_image, events.cell_index[keep]


def solve_max_utility(
    events: EventTable,
    is_cloudy: np.ndarray | None,
    budget: int,
    time_limit_s: float = 60.0,
    max_pairs: int = 300_000,
    weight: np.ndarray | None = None,
) -> dict:
    """Solve the budgeted maximum-utility selection exactly (HiGHS).

    Maximise the sum over pairs (image, cell) of ``u z`` subject to: each cell takes one utility, only from a chosen
    image, and at most ``budget`` images are chosen; ``x`` binary, ``0 <= z <= 1``. With ``weight`` (one per image),
    ``u`` is replaced by ``weight u``: the time-weighted coverage, the objective of the time-aware selector. Returns
    ``status``, ``objective``, ``bound``, ``selected`` (image indices), ``seconds`` and ``n_pairs``.
    """
    utility = effective_utility(events, is_cloudy)
    scale = 1.0
    if weight is not None:
        w = np.asarray(weight, dtype=np.float64)
        counts = events.observations_per_image
        has = (
            np.add.reduceat((utility > 0.0).astype(np.int64), events.offsets[:-1]) > 0
            if len(events)
            else np.zeros(0, bool)
        )
        # Weights normalised to a maximum of 1 over the images that can contribute, so the solver's absolute
        # tolerances do not swallow a small objective; the objective is scaled back on return.
        scale = float(w[has].max()) if has.any() and w[has].max() > 0 else 1.0
        utility = utility * np.repeat(w / scale, counts)
    keep, pair_image, pair_cell = _pairs(events, utility)
    pair_utility = utility[keep]
    n_images, n_pairs = len(events), int(keep.sum())
    if n_pairs == 0:
        return {"status": "empty", "objective": 0.0, "bound": 0.0, "selected": [], "seconds": 0.0, "n_pairs": 0}
    if n_pairs > max_pairs:
        return {
            "status": f"skipped (>{max_pairs} pairs)",
            "objective": None,
            "bound": None,
            "selected": [],
            "seconds": 0.0,
            "n_pairs": n_pairs,
        }
    _, cell_row = np.unique(pair_cell, return_inverse=True)
    n_cells = int(cell_row.max()) + 1
    c = np.concatenate([np.zeros(n_images), -pair_utility])
    width = n_images + n_pairs
    budget_row = coo_matrix((np.ones(n_images), (np.zeros(n_images), np.arange(n_images))), shape=(1, width))
    per_cell = coo_matrix((np.ones(n_pairs), (cell_row, n_images + np.arange(n_pairs))), shape=(n_cells, width))
    link = coo_matrix(
        (
            np.r_[np.ones(n_pairs), -np.ones(n_pairs)],
            (np.r_[np.arange(n_pairs), np.arange(n_pairs)], np.r_[n_images + np.arange(n_pairs), pair_image]),
        ),
        shape=(n_pairs, width),
    )
    a = vstack([budget_row, per_cell, link]).tocsr()
    upper = np.r_[budget, np.ones(n_cells), np.zeros(n_pairs)]
    started = time.time()
    result = milp(
        c,
        constraints=LinearConstraint(a, -np.inf, upper),
        bounds=Bounds(0, 1),
        integrality=np.r_[np.ones(n_images), np.zeros(n_pairs)],
        options={"time_limit": float(time_limit_s), "mip_rel_gap": 1e-9},
    )
    seconds = time.time() - started
    if result.x is None:
        return {
            "status": f"no solution ({result.message})",
            "objective": None,
            "bound": None,
            "selected": [],
            "seconds": seconds,
            "n_pairs": n_pairs,
        }
    bound = getattr(result, "mip_dual_bound", None)
    return {
        "status": "optimal" if result.status == 0 else "time limit (best found)",
        "objective": float(-result.fun) * scale,
        "bound": None if bound is None else float(-bound) * scale,
        "selected": [int(image) for image in np.flatnonzero(result.x[:n_images] > 0.5)],
        "seconds": seconds,
        "n_pairs": n_pairs,
    }


def solve_min_images(
    events: EventTable,
    is_cloudy: np.ndarray | None,
    need: int,
    time_limit_s: float = 60.0,
    max_pairs: int = 300_000,
) -> dict:
    """Solve the minimum number of images covering at least ``need`` cells exactly (HiGHS).

    Minimise the number of images subject to: a cell counts only if a chosen image covers it, and at least ``need``
    cells count; all binary. The check of the ECC's greedy cover step. Returns ``status``, ``images``, ``seconds`` and
    ``n_pairs``.
    """
    utility = effective_utility(events, is_cloudy)
    keep, pair_image, pair_cell = _pairs(events, utility)
    n_images, n_pairs = len(events), int(keep.sum())
    if n_pairs == 0:
        return {"status": "empty", "images": None, "seconds": 0.0, "n_pairs": 0}
    if n_pairs > max_pairs:
        return {"status": f"skipped (>{max_pairs} pairs)", "images": None, "seconds": 0.0, "n_pairs": n_pairs}
    _, cell_row = np.unique(pair_cell, return_inverse=True)
    n_cells = int(cell_row.max()) + 1
    if n_cells < need:
        return {"status": "infeasible", "images": None, "seconds": 0.0, "n_pairs": n_pairs}
    c = np.r_[np.ones(n_images), np.zeros(n_cells)]
    width = n_images + n_cells
    cover = coo_matrix(
        (
            np.r_[-np.ones(n_pairs), np.ones(n_cells)],
            (np.r_[cell_row, np.arange(n_cells)], np.r_[pair_image, n_images + np.arange(n_cells)]),
        ),
        shape=(n_cells, width),
    )
    total = coo_matrix((np.ones(n_cells), (np.zeros(n_cells), n_images + np.arange(n_cells))), shape=(1, width))
    started = time.time()
    result = milp(
        c,
        constraints=[LinearConstraint(cover.tocsr(), -np.inf, 0.0), LinearConstraint(total.tocsr(), need, np.inf)],
        bounds=Bounds(0, 1),
        integrality=np.ones(width),
        options={"time_limit": float(time_limit_s)},
    )
    seconds = time.time() - started
    if result.x is None or result.status != 0:
        return {"status": "time limit or no solution", "images": None, "seconds": seconds, "n_pairs": n_pairs}
    # int(): under NumPy 1.x, round() of a NumPy float is a float.
    images = int(round(result.fun))  # noqa: RUF046
    return {"status": "optimal", "images": images, "seconds": seconds, "n_pairs": n_pairs}


def facility_value(events: EventTable, images, weight: np.ndarray) -> float:
    """The time-weighted coverage of a selection under clear sky: the sum over cells of the best ``weight u``."""
    best = np.zeros(len(events.region_cells))
    for image in images:
        rows = events.rows(image)
        np.maximum.at(best, events.cell_index[rows], weight[image] * events.utility[rows])
    return float(best.sum())


def _windows(window_hours: int, settings: IlpSettings) -> list[int]:
    span = max(0, window_hours - settings.horizon_hours) * 60
    return sorted({int(x) for x in np.linspace(0, span, settings.instances)})


def ilp_gap_row(events: EventTable, start_min: int, settings: IlpSettings, onsets: OnsetSettings) -> dict | None:
    """One window of ``arf_ilp_gap.csv``; ``None`` if it has no images."""
    horizon = settings.horizon_hours * 60
    window, _ = subset_by_time(events, None, start_min, start_min + horizon)
    while int((window.utility > 0.0).sum()) > settings.max_pairs and horizon > MIN_HORIZON_MIN:
        horizon = max(MIN_HORIZON_MIN, horizon // 2)
        window, _ = subset_by_time(events, None, start_min, start_min + horizon)
    if len(window) == 0:
        return None
    exact = solve_max_utility(window, None, settings.budget, settings.time_limit_s, settings.max_pairs)
    row: dict = {
        "shift_min": start_min,
        "horizon_min": horizon,
        "n_events": len(window),
        "n_pairs": exact["n_pairs"],
        "ilp_status": exact["status"],
        "ilp_seconds": round(exact["seconds"], 2),
        "ilp_objective": exact["objective"],
        "ilp_bound": exact["bound"],
        "ilp_latency_min": completion_time(window, exact["selected"], None) if exact["selected"] else None,
    }
    time_aware_images: list[int] = []
    for name, rate in (("coverage_first", 0.0), ("time_aware", onsets.decay_rate)):
        weights = (
            time_weights(window, rate, onsets.orbit_period_min, onsets.decay_scale, onsets.reachable_level_pct)
            if rate > 0
            else None
        )
        selection = greedy_selection(window, settings.budget, None, weights, onsets.time_objective, False)
        best = selection.best_utility
        # A sum in cell order, one value at a time, as the registered releases add them.
        row[f"{name}_objective"] = float(sum(float(v) for v in best[best > 0.0]))
        if rate > 0:
            time_aware_images = list(selection.images)
    fcfs = select_fcfs(window, None, settings.budget)
    fcfs_best = np.zeros(len(window.region_cells))
    for image in fcfs:
        rows = window.rows(image)
        np.maximum.at(fcfs_best, window.cell_index[rows], window.utility[rows])
    row["fcfs_objective"] = float(fcfs_best.sum())
    # The time-aware selector against the exact optimum of its own objective, the time-weighted coverage.
    weights = time_weights(
        window, onsets.decay_rate, onsets.orbit_period_min, onsets.decay_scale, onsets.reachable_level_pct
    )
    weighted = solve_max_utility(window, None, settings.budget, settings.time_limit_s, settings.max_pairs, weights)
    row.update(
        {
            "tw_ilp_status": weighted["status"],
            "tw_ilp_objective": weighted["objective"],
            "time_aware_tw_objective": facility_value(window, time_aware_images, weights),
        }
    )
    row["time_aware_tw_ratio"] = (
        row["time_aware_tw_objective"] / weighted["objective"]
        if weighted["status"] == "optimal" and weighted["objective"]
        else None
    )
    optimal = exact["status"] == "optimal" and exact["objective"]
    for name in ("coverage_first", "time_aware", "fcfs"):
        row[f"{name}_ratio"] = row[f"{name}_objective"] / exact["objective"] if optimal else None
    # The ECC's greedy cover step against the exact minimum number of images, at the H1 level of the reachable cells.
    level = float(onsets.reachable_level_pct)
    reach = reachable_cells(window, None)
    need = int(np.ceil(level / 100.0 * reach - 1e-9))
    if reach:
        completed, _, greedy_images = earliest_completion(window, None, len(window), level, reach)
        minimum = solve_min_images(window, None, need, settings.time_limit_s, settings.max_pairs)
    else:
        completed, greedy_images, minimum = None, 0, {"status": "empty", "images": None}
    row.update(
        {
            "ecc_target_pct": level,
            "ecc_greedy_images": greedy_images if completed is not None else None,
            "ecc_exact_images": minimum["images"],
            "ecc_exact_status": minimum["status"],
        }
    )
    return row


def ilp_gap_table(
    events: EventTable, settings: IlpSettings, onsets: OnsetSettings, window_hours: int
) -> pd.DataFrame | None:
    """Return ``arf_ilp_gap.csv``; ``None`` if the analysis is disabled."""
    if not settings.enabled:
        return None
    rows = [row for start in _windows(window_hours, settings) if (row := ilp_gap_row(events, start, settings, onsets))]
    return pd.DataFrame(rows)


def ilp_gap_summary(table: pd.DataFrame, settings: IlpSettings, time_objective: str) -> dict:
    """The summary of the windows, as the scenario ledger records it (``ilp_gap``)."""
    solved = table[table["ilp_status"] == "optimal"] if len(table) else table
    summary: dict = {
        "instances": len(table),
        "solved": len(solved),
        "optimal": int((table["ilp_status"] == "optimal").sum()) if len(table) else 0,
        "time_limited": int(table["ilp_status"].astype(str).str.startswith("time limit").sum()) if len(table) else 0,
        "horizon_hours": settings.horizon_hours,
        "budget_k": settings.budget,
        "horizons_used_min": sorted(int(h) for h in table["horizon_min"].unique()) if len(table) else [],
    }
    for name in ("coverage_first", "time_aware", "fcfs"):
        for statistic in ("median", "min"):
            summary[f"{name}_ratio_{statistic}"] = (
                float(getattr(solved[f"{name}_ratio"], statistic)()) if len(solved) else None
            )
    summary["time_aware_objective"] = time_objective
    summary["tw_solved"] = int((table["tw_ilp_status"] == "optimal").sum()) if len(table) else 0
    ratios = table["time_aware_tw_ratio"].dropna() if len(table) else None
    for statistic in ("median", "min"):
        summary[f"time_aware_tw_ratio_{statistic}"] = (
            float(getattr(ratios, statistic)()) if len(table) and table["time_aware_tw_ratio"].notna().any() else None
        )
    summary["guarantee_note"] = (
        "facility objective: the k-step greedy attains >= 1 - 1/e of the optimum of the time-"
        "weighted coverage (Nemhauser, Wolsey & Fisher 1978); the ratio above is the measured value"
        if time_objective == "facility"
        else "legacy objective: no guarantee; the ratio is measured against the time-weighted coverage "
        "optimum for comparison only"
    )
    if len(table) and "ecc_exact_status" in table:
        exact = table[(table.get("ecc_exact_status") == "optimal") & table["ecc_greedy_images"].notna()]
    else:
        exact = pd.DataFrame(columns=["ecc_greedy_images", "ecc_exact_images"])
    summary["ecc_instances_exact"] = len(exact)
    summary["ecc_matches_exact"] = int((exact["ecc_greedy_images"] == exact["ecc_exact_images"]).sum())
    summary["ecc_max_extra_images"] = (
        int((exact["ecc_greedy_images"] - exact["ecc_exact_images"]).max()) if len(exact) else None
    )
    return summary
