"""The selectors compared at every onset of a window (Cell 7C, section 2, of the registered releases).

One onset is one sample of orbital geometry: the first pass after it may come early or late by chance. The onset is
therefore moved from 0 to ``shift_hours`` after the start of the window in steps of ``shift_step_min``, and for each
onset only the images acquired in the following ``window_hours - shift_hours`` hours are used, with times counted
from the onset. Each onset gives one row of ``arf_onset_robustness.csv``:

* the **floor**: when all delivered images together first reach full coverage, or a given level (no selector can do
  better);
* five selectors at the budget of ``budget`` images, without the Fail-Signal gap-fill (which adds images beyond the
  budget that the other selectors have no counterpart for): coverage-first, time-aware, first-come-first-served
  (FCFS), random (the median of seeded draws) and the earliest-completion cover (ECC);
* for the level ``reachable_level_pct`` of the cells reachable in the onset's horizon (H1a and H1b of the protocol):
  each selector's time to the level, and the images it has taken delivery of and the mean utility of its mosaic then;
  FCFS also without a budget.

Everything is computed under clear sky and under the run's cloud, the same per-observation cloud for every onset.
A cell counts as covered when an observation of it has positive utility and is not cloudy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from arf.delivery import decay_time_scale, time_weights
from arf.events import EventTable
from arf.failsignal import adaptive_selection
from arf.greedy import greedy_selection

POLICIES = ("coverage_first", "time_aware", "fcfs", "random", "earliest_completion")
#: Guards the comparison of a share with a level against rounding, as in the registered releases.
_TOLERANCE = 1e-9


@dataclass(frozen=True)
class OnsetSettings:
    """The settings of the onset comparison, with the names of the registered releases' configuration.

    Attributes:
        orbit_period_min: The orbital period at the fleet's median altitude (``T_MAX_WAIT_MINUTES`` after Cell 2);
            the time scale of the time-aware weights.
        cloud_probability: The cloud probability of each region cell, for the adaptive selector's plan.
        budget: ``max_selection_steps``.
        decay_rate: ``latency_decay_rate``, the rate of the time-aware selector.
        decay_scale: ``latency_decay_scale``.
        window_hours: ``window_hours``.
        shift_hours: ``onset_shift_hours``.
        shift_step_min: ``onset_shift_step_min``.
        coverage_targets_pct: ``coverage_targets_pct``, levels as a share of all region cells.
        reachable_level_pct: ``h1_reachable_coverage_pct``, the level as a share of the reachable cells.
        random_draws: ``random_baseline_draws``.
        seed: ``random_seed``; onset ``d`` draws its random selections with seed ``seed + d``.
        cloud_knowledge: ``cloud_knowledge``: ``"realistic"`` (cloud revealed as images arrive) or ``"oracle"``.
        cloud_planning: ``cloud_planning`` of the adaptive selector.
        time_objective: ``time_aware_objective``.
    """

    orbit_period_min: float
    cloud_probability: np.ndarray
    budget: int = 30
    decay_rate: float = 3.0
    decay_scale: str = "fleet"
    window_hours: int = 48
    shift_hours: int = 24
    shift_step_min: int = 20
    coverage_targets_pct: tuple[float, ...] = (95.0, 99.0)
    reachable_level_pct: float = 95.0
    random_draws: int = 10
    seed: int = 42
    cloud_knowledge: str = "realistic"
    cloud_planning: str = "expected"
    time_objective: str = "facility"


# --- the observations ------------------------------------------------------------------------------------------------


def effective_utility(events: EventTable, is_cloudy: np.ndarray | None) -> np.ndarray:
    """Return each observation's utility, zero where it is cloudy."""
    return np.where(is_cloudy, 0.0, events.utility) if is_cloudy is not None else events.utility


def subset_by_time(
    events: EventTable, is_cloudy: np.ndarray | None, start_min: int, end_min: int
) -> tuple[EventTable, np.ndarray | None]:
    """Return the images acquired in ``[start_min, end_min)``, with times counted from ``start_min``.

    Utilities, waits and cloud are those of the full table.
    """
    keep = [image for image in range(len(events)) if start_min <= int(events.acquired_min[image]) < end_min]
    if not keep:
        empty = EventTable((), np.zeros(0, np.int64), np.zeros(0), np.zeros(1, np.int64),
                           np.zeros(0, np.int32), np.zeros(0), events.region_cells)  # fmt: skip
        return empty, (np.zeros(0, bool) if is_cloudy is not None else None)
    rows = np.concatenate([np.arange(events.offsets[i], events.offsets[i + 1]) for i in keep])
    counts = np.diff(events.offsets)[keep]
    subset = EventTable(
        satellites=tuple(events.satellites[i] for i in keep),
        acquired_min=events.acquired_min[keep] - start_min,
        wait_min=events.wait_min[keep],
        offsets=np.r_[0, np.cumsum(counts)],
        cell_index=events.cell_index[rows],
        utility=events.utility[rows],
        region_cells=events.region_cells,
    )
    return subset, (is_cloudy[rows] if is_cloudy is not None else None)


def _delivery_key(events: EventTable, image: int) -> float:
    """The sort key of an image by delivery: its delivery time, ``inf`` when it is never delivered."""
    wait = events.wait_min[image]
    return math.inf if np.isnan(wait) else int(events.acquired_min[image]) + float(wait)


def _delivery_time(events: EventTable, image: int) -> float:
    """The delivery time of an image, ``nan`` when it is never delivered."""
    return int(events.acquired_min[image]) + float(events.wait_min[image])


def in_delivery_order(events: EventTable, images: Sequence[int]) -> list[int]:
    """Return images sorted by delivery time; ties keep the given order, and undelivered images come last."""
    return sorted(images, key=lambda image: _delivery_key(events, image))


def _delivery_times(events: EventTable) -> np.ndarray:
    """Return each image's delivery time for sorting, ``nan`` for an image never delivered (it sorts last)."""
    return events.acquired_min.astype(np.float64) + events.wait_min


def reachable_cells(events: EventTable, is_cloudy: np.ndarray | None) -> int:
    """Return the number of cells with at least one usable observation: the most any selector can cover."""
    usable = effective_utility(events, is_cloudy) > 0.0
    return int(np.unique(events.cell_index[usable]).size)


# --- the floor and coverage over time --------------------------------------------------------------------------------


def best_possible_completion(events: EventTable, is_cloudy: np.ndarray | None) -> tuple[float | None, float]:
    """Return when all delivered images together first cover every reachable cell, and that coverage in percent.

    The coverage is a share of all region cells. ``(None, coverage)`` if the delivered images never cover every
    reachable cell, ``(None, 0.0)`` if no cell is reachable.
    """
    if len(events) == 0:
        return None, 0.0
    utility = effective_utility(events, is_cloudy)
    reachable = np.unique(events.cell_index[utility > 0.0]).size
    if reachable == 0:
        return None, 0.0
    total = len(events.region_cells)
    deliveries = [
        (int(events.acquired_min[image]) + float(events.wait_min[image]), image)
        for image in range(len(events))
        if not np.isnan(events.wait_min[image])
    ]
    seen = np.zeros(total, dtype=bool)
    seen_count = 0
    for delivered, image in sorted(deliveries):
        rows = events.rows(image)
        cells = events.cell_index[rows][utility[rows] > 0.0]
        new_cells = cells[~seen[cells]]
        if new_cells.size:
            seen[new_cells] = True
            seen_count += new_cells.size
            if seen_count == reachable:
                return float(delivered), 100.0 * reachable / total
    return None, 100.0 * seen_count / total


def coverage_times(
    events: EventTable,
    is_cloudy: np.ndarray | None,
    targets_pct: Sequence[float],
    images: Sequence[int] | None = None,
    denominator: int | None = None,
) -> dict[float, float | None]:
    """Return, per level, the delivery time at which the delivered images first reach it.

    Args:
        events: The observations.
        is_cloudy: One boolean per observation, or ``None`` for clear sky.
        targets_pct: Levels, in percent of ``denominator`` cells.
        images: The images considered, in selection order; all images (the floor) by default.
        denominator: The number of cells the levels refer to; all region cells by default.

    Returns:
        The time per level, ``None`` for a level never reached.
    """
    utility = effective_utility(events, is_cloudy)
    total = len(events.region_cells) if denominator is None else denominator
    if total <= 0:
        return {target: None for target in targets_pct}
    order = in_delivery_order(events, range(len(events)) if images is None else images)
    seen = np.zeros(len(events.region_cells), dtype=bool)
    seen_count, reached, pending = 0, {}, sorted(targets_pct)
    for image in order:
        if not pending:
            break
        delivered = _delivery_time(events, image)
        if np.isnan(delivered):
            continue
        rows = events.rows(image)
        cells = events.cell_index[rows][utility[rows] > 0.0]
        new_cells = cells[~seen[cells]]
        if new_cells.size:
            seen[new_cells] = True
            seen_count += new_cells.size
            share = 100.0 * seen_count / total
            while pending and share >= pending[0] - _TOLERANCE:
                reached[pending.pop(0)] = float(delivered)
    reached.update({target: None for target in pending})
    return reached


def completion_time(events: EventTable, images: Sequence[int], is_cloudy: np.ndarray | None) -> float | None:
    """Return the delivery time of the last image, in delivery order, that adds a usable cell (Cell 6's H1 latency)."""
    if not images:
        return None
    utility = effective_utility(events, is_cloudy)
    seen = np.zeros(len(events.region_cells), dtype=bool)
    completed = None
    for image in in_delivery_order(events, images):
        rows = events.rows(image)
        cells = events.cell_index[rows][utility[rows] > 0.0]
        new_cells = cells[~seen[cells]]
        if new_cells.size:
            seen[new_cells] = True
            completed = _delivery_time(events, image)
    return completed


def level_stats(
    events: EventTable, is_cloudy: np.ndarray | None, images: Sequence[int], need: int
) -> tuple[float | None, int | None, float | None]:
    """Return the selection's delivered mosaic when it first covers ``need`` cells.

    Returns:
        The delivery time of the image at which that happens, the number of images delivered up to and including
        it, and the mean utility of the cells covered then; ``(None, None, None)`` if it never happens.
    """
    if need <= 0 or not images:
        return None, None, None
    utility = effective_utility(events, is_cloudy)
    best = np.zeros(len(events.region_cells))
    covered = 0
    for count, image in enumerate(in_delivery_order(events, images), 1):
        delivered = _delivery_time(events, image)
        if np.isnan(delivered):
            break  # undelivered images sort last
        rows = events.rows(image)
        cells, values = events.cell_index[rows], utility[rows]
        covered += int(np.unique(cells[(values > 0.0) & (best[cells] == 0.0)]).size)
        np.maximum.at(best, cells, values)
        if covered >= need:
            return float(delivered), count, float(best[best > 0.0].mean())
    return None, None, None


def mosaic_quality(events: EventTable, is_cloudy: np.ndarray | None, images: Sequence[int]) -> float | None:
    """Return the mean, over the cells covered, of the best usable utility among the images; ``None`` if none."""
    utility = effective_utility(events, is_cloudy)
    best = np.zeros(len(events.region_cells))
    for image in images:
        rows = events.rows(image)
        np.maximum.at(best, events.cell_index[rows], utility[rows])
    covered = best > 0.0
    return float(best[covered].mean()) if covered.any() else None


# --- the earliest-completion cover -----------------------------------------------------------------------------------


def _count_cover(
    events: EventTable, utility: np.ndarray, candidates: np.ndarray, budget: int, need: int
) -> tuple[int, list[int]]:
    """Greedy maximum coverage, each usable cell counting once, with at most ``budget`` of the candidates.

    Candidates are in delivery order, and ties go to the earlier one. Stops once ``need`` cells are covered.

    Returns:
        The cells covered and the images chosen, in greedy order.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    if candidates.size == 0 or budget <= 0:
        return 0, []
    counts = np.diff(events.offsets)[candidates]
    rows = np.concatenate([np.arange(events.offsets[i], events.offsets[i + 1]) for i in candidates])
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    cells = events.cell_index[rows]
    usable = utility[rows] > 0.0
    covered = np.zeros(len(events.region_cells), dtype=bool)
    available = np.ones(candidates.size, dtype=bool)
    chosen, covered_count = [], 0
    for _ in range(budget):
        gain = np.add.reduceat((usable & ~covered[cells]).astype(np.int64), starts)
        gain[~available] = -1
        best = int(np.argmax(gain))
        if gain[best] <= 0:
            break
        available[best] = False
        chosen.append(int(candidates[best]))
        span = slice(int(starts[best]), int(starts[best] + counts[best]))
        new_cells = cells[span][usable[span]]
        covered_count += int((~covered[new_cells]).sum())
        covered[new_cells] = True
        if covered_count >= need:
            break
    return covered_count, chosen


def earliest_completion(
    events: EventTable,
    is_cloudy: np.ndarray | None,
    budget: int,
    target_pct: float,
    denominator: int | None = None,
) -> tuple[float | None, list[int], int]:
    """Return the earliest-completion cover (ECC) for a level.

    The earliest delivery time ``t`` at which at most ``budget`` images delivered by ``t`` cover ``target_pct`` of
    the cells (all region cells, or ``denominator`` cells), found by bisection over delivery times with a greedy
    cover at each probe; the floor bounds the search from below. Only images delivered by ``t`` are used, so under
    cloud the selection is causal: each image's cloud is known once it arrives.

    Returns:
        The completion time (the latest delivery among the chosen images) or ``None``, the chosen images, and their
        number.
    """
    utility = effective_utility(events, is_cloudy)
    total = len(events.region_cells) if denominator is None else denominator
    if total <= 0 or len(events) == 0:
        return None, [], 0
    need = int(np.ceil(target_pct / 100.0 * total - _TOLERANCE))
    delivered = _delivery_times(events)
    order = np.argsort(delivered, kind="stable")
    order = order[np.isfinite(delivered[order])]
    times = delivered[order]
    seen = np.zeros(len(events.region_cells), dtype=bool)
    seen_count, lowest = 0, None
    for position, image in enumerate(order):  # the floor: all delivered images
        rows = events.rows(image)
        cells = events.cell_index[rows][utility[rows] > 0.0]
        new_cells = cells[~seen[cells]]
        seen[new_cells] = True
        seen_count += new_cells.size
        if seen_count >= need:
            lowest = position
            break
    if lowest is None:
        return None, [], 0

    def probe(position: int) -> tuple[bool, list[int]]:
        end = int(np.searchsorted(times, times[position], side="right"))  # every image delivered by then
        covered, chosen = _count_cover(events, utility, order[:end], budget, need)
        return covered >= need, chosen

    highest = len(order) - 1
    succeeded, best = probe(highest)
    if not succeeded:
        return None, [], 0
    low, high = lowest, highest
    while low < high:
        middle = (low + high) // 2
        succeeded, chosen = probe(middle)
        if succeeded:
            high, best = middle, chosen
        else:
            low = middle + 1
    return float(max(delivered[image] for image in best)), best, len(best)


def _ecc_best_effort(events: EventTable, is_cloudy: np.ndarray | None, budget: int) -> list[int]:
    """The ECC when no level is reached: the greedy cover of every delivered image, at most ``budget`` images."""
    utility = effective_utility(events, is_cloudy)
    delivered = _delivery_times(events)
    order = np.argsort(delivered, kind="stable")
    order = order[np.isfinite(delivered[order])]
    return _count_cover(events, utility, order, budget, need=10**12)[1]


# --- the selectors ---------------------------------------------------------------------------------------------------


def select_fcfs(events: EventTable, is_cloudy: np.ndarray | None, budget: int) -> list[int]:
    """First-come-first-served: images in delivery order, each kept if it adds a usable cell, up to ``budget``."""
    utility = effective_utility(events, is_cloudy)
    best = np.zeros(len(events.region_cells))
    selected: list[int] = []
    for image in in_delivery_order(events, range(len(events))):
        if len(selected) >= budget:
            break
        rows = events.rows(image)
        cells, values = events.cell_index[rows], utility[rows]
        if np.any((values > 0.0) & (best[cells] == 0.0)):
            selected.append(image)
            np.maximum.at(best, cells, values)
    return selected


def select_random(
    events: EventTable,
    is_cloudy: np.ndarray | None,
    budget: int,
    rng: np.random.Generator,
    cloud_knowledge: str = "realistic",
) -> list[int]:
    """Random baseline: ``budget`` images drawn without replacement from those that can cover a cell.

    Under realistic cloud knowledge the draw uses clear-sky utilities, since which images turn out cloudy is not
    known in advance.
    """
    utility = effective_utility(events, is_cloudy)
    draw_utility = events.utility if is_cloudy is not None and cloud_knowledge != "oracle" else utility
    useful = [image for image in range(len(events)) if np.any(draw_utility[events.rows(image)] > 0.0)]
    picks = rng.choice(len(useful), size=min(budget, len(useful)), replace=False) if useful else []
    return [useful[int(position)] for position in picks]


def select_greedy(
    events: EventTable, is_cloudy: np.ndarray | None, decay_rate: float, settings: OnsetSettings
) -> list[int]:
    """Cell 6's selector at the budget, without gap-fill: coverage-first at rate 0, time-aware otherwise.

    Clear sky, or cloud known in advance: the greedy selection. Cloud revealed as images arrive: the adaptive
    (Fail-Signal) selection.
    """
    weights = (
        time_weights(events, decay_rate, settings.orbit_period_min, settings.decay_scale, settings.reachable_level_pct)
        if decay_rate > 0
        else None
    )
    if is_cloudy is None or settings.cloud_knowledge == "oracle":
        selection = greedy_selection(events, settings.budget, is_cloudy, weights, settings.time_objective, False)
    else:
        selection = adaptive_selection(
            events,
            settings.budget,
            is_cloudy,
            settings.cloud_probability,
            weights,
            settings.time_objective,
            settings.cloud_planning,
            False,
        )
    return list(selection.images)


def policy_selections(
    events: EventTable, is_cloudy: np.ndarray | None, policy: str, settings: OnsetSettings, seed: int | None = None
) -> list[list[int]]:
    """One selection per deterministic selector, or ``random_draws`` seeded selections for the random one."""
    if policy in ("coverage_first", "time_aware"):
        rate = 0.0 if policy == "coverage_first" else settings.decay_rate
        return [select_greedy(events, is_cloudy, rate, settings)]
    if policy == "fcfs":
        return [select_fcfs(events, is_cloudy, settings.budget)]
    rng = np.random.default_rng(settings.seed if seed is None else seed)
    return [
        select_random(events, is_cloudy, settings.budget, rng, settings.cloud_knowledge)
        for _ in range(settings.random_draws)
    ]


def _median_or_none(values: Sequence[float | None]) -> float | None:
    """Median with ``None`` counted as never (infinity); ``None`` if the median is never."""
    array = np.array([np.inf if value is None else value for value in values], dtype=float)
    middle = float(np.median(array)) if array.size else np.inf
    return None if np.isinf(middle) else middle


def _metrics(events: EventTable, is_cloudy: np.ndarray | None, images: Sequence[int], targets: Sequence[float]):
    """Completion time, coverage in percent of all cells, mean quality, and the time to each level, of a selection."""
    utility = effective_utility(events, is_cloudy)
    best = np.zeros(len(events.region_cells))
    for image in images:
        rows = events.rows(image)
        np.maximum.at(best, events.cell_index[rows], utility[rows])
    covered = best > 0.0
    return {
        "lat": completion_time(events, images, is_cloudy),
        "cov": 100.0 * int(covered.sum()) / len(events.region_cells),
        "q": float(best[covered].mean()) if covered.any() else 0.0,
        "t": coverage_times(events, is_cloudy, targets, images),
    }


def policy_outcome(
    events: EventTable, is_cloudy: np.ndarray | None, policy: str, settings: OnsetSettings, seed: int | None = None
) -> dict:
    """Completion time, coverage, quality and time to each level of one selector; the random one: the medians."""
    targets = settings.coverage_targets_pct
    if len(events) == 0:
        return {"lat": None, "cov": 0.0, "q": 0.0, "t": {target: None for target in targets}}
    if policy == "earliest_completion":
        # One ECC per level; coverage, quality and completion are those of the highest level reached, or of the
        # best cover within the budget if none is reached.
        times, best_selection = {}, None
        for target in sorted(targets):
            completed, chosen, _ = earliest_completion(events, is_cloudy, settings.budget, target)
            times[target] = completed
            if chosen:
                best_selection = chosen
        chosen = best_selection if best_selection is not None else _ecc_best_effort(events, is_cloudy, settings.budget)
        metrics = _metrics(events, is_cloudy, chosen, targets)
        metrics["t"] = times
        return metrics
    outcomes = [
        _metrics(events, is_cloudy, images, targets)
        for images in policy_selections(events, is_cloudy, policy, settings, seed)
    ]
    if len(outcomes) == 1:
        return outcomes[0]
    return {
        "lat": _median_or_none([outcome["lat"] for outcome in outcomes]),
        "cov": float(np.median([outcome["cov"] for outcome in outcomes])),
        "q": float(np.median([outcome["q"] for outcome in outcomes])),
        "t": {target: _median_or_none([outcome["t"][target] for outcome in outcomes]) for target in targets},
    }


# --- the table -------------------------------------------------------------------------------------------------------


def _sky_columns(row: dict, events: EventTable, is_cloudy: np.ndarray | None, sky: str, onset: int, settings):
    """Add the columns of one sky (``"clear"`` or ``"cloud"``) to an onset's row, in the registered order."""
    level = settings.reachable_level_pct
    tag = f"{level:g}r"
    row[f"best_possible_{sky}_min"], row[f"best_possible_{sky}_cov_pct"] = best_possible_completion(events, is_cloudy)
    for target, value in coverage_times(events, is_cloudy, settings.coverage_targets_pct).items():
        row[f"best_possible_{sky}_t{target:g}_min"] = value
    for name in POLICIES:
        outcome = policy_outcome(events, is_cloudy, name, settings, seed=settings.seed + onset)
        row[f"{name}_{sky}_min"], row[f"{name}_{sky}_cov_pct"] = outcome["lat"], outcome["cov"]
        for target, value in outcome["t"].items():
            row[f"{name}_{sky}_t{target:g}_min"] = value
    reachable = reachable_cells(events, is_cloudy)
    row[f"reachable_{sky}_tiles"] = reachable
    row[f"best_possible_{sky}_t{tag}_min"] = coverage_times(events, is_cloudy, (level,), denominator=reachable)[level]
    need = int(np.ceil(level / 100.0 * reachable - _TOLERANCE)) if reachable else 0
    for name in ("coverage_first", "time_aware", "fcfs"):
        images = policy_selections(events, is_cloudy, name, settings)[0] if len(events) else []
        row[f"{name}_{sky}_t{tag}_min"] = (
            coverage_times(events, is_cloudy, (level,), images, denominator=reachable)[level] if reachable else None
        )
        if name != "fcfs":
            _, count, quality = level_stats(events, is_cloudy, images, need) if reachable else (None, None, None)
            row[f"{name}_{sky}_images{tag}"], row[f"{name}_{sky}_quality{tag}"] = count, quality
    # First-come-first-served without a budget: the images it takes to reach the level, at the floor.
    unbudgeted = (
        level_stats(events, is_cloudy, select_fcfs(events, is_cloudy, len(events)), need)
        if reachable
        else (None, None, None)
    )
    row[f"fcfs_unbudgeted_{sky}_t{tag}_min"] = unbudgeted[0]
    row[f"fcfs_unbudgeted_{sky}_images{tag}"], row[f"fcfs_unbudgeted_{sky}_quality{tag}"] = unbudgeted[1:]
    completed, chosen, count = (
        earliest_completion(events, is_cloudy, settings.budget, level, denominator=reachable)
        if reachable
        else (None, [], 0)
    )
    row[f"earliest_completion_{sky}_t{tag}_min"] = completed
    row[f"earliest_completion_{sky}_images{tag}"] = count if completed is not None else None
    row[f"earliest_completion_{sky}_quality{tag}"] = (
        mosaic_quality(events, is_cloudy, chosen) if completed is not None else None
    )


def onset_rows(events: EventTable, is_cloudy: np.ndarray, settings: OnsetSettings) -> list[dict]:
    """Return one row per onset, with the columns of ``arf_onset_robustness.csv`` in their order.

    Args:
        events: The observations of the whole window.
        is_cloudy: The run's cloud, one boolean per observation, used for every onset.
        settings: The settings of the comparison.
    """
    horizon_min = (settings.window_hours - settings.shift_hours) * 60
    rows = []
    for onset in range(0, settings.shift_hours * 60 + 1, settings.shift_step_min):
        subset, subset_cloud = subset_by_time(events, is_cloudy, onset, onset + horizon_min)
        row = {
            "onset_shift_min": onset,
            "n_events": len(subset),
            "decay_tau_min": (
                decay_time_scale(subset, settings.orbit_period_min, settings.decay_scale, settings.reachable_level_pct)
                if len(subset)
                else None
            ),
        }
        for sky, cloud in (("clear", None), ("cloud", subset_cloud)):
            _sky_columns(row, subset, cloud, sky, onset, settings)
        rows.append(row)
    return rows


def onset_table(events: EventTable, is_cloudy: np.ndarray, settings: OnsetSettings):
    """Return ``arf_onset_robustness.csv`` as a table (see :func:`onset_rows`)."""
    import pandas as pd

    return pd.DataFrame(onset_rows(events, is_cloudy, settings))
