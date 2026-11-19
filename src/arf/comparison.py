"""One cloud realisation: the Fail-Signal selector against plans made without observing cloud.

For each objective, four plans with the same number of images are compared, as
in the robustness analysis of the registered releases:

* ``adaptive``: the Fail-Signal selector (``arf.failsignal.adaptive_selection``);
  its image count, compensation included, sets the size of the other plans;
* ``open_loop_expected``: the same planner run once in advance, with no
  observation of cloud. The difference from ``adaptive`` is the value of acting
  on the fail-signals, and is the quantity hypothesis H2 tests;
* ``open_loop_clear``: a plan on clear-sky geometry, with no cloud information at all;
* ``oracle``: the greedy with the realised cloud known in advance, for reference.
  It is a greedy, so it is not a strict upper bound.

Two more are kept for the record: the selector of release 1.4.0, which plans on
clear-sky geometry and re-plans only after a fail-signal (``adaptive_clear_planning``),
and an open-loop plan of that selector's size (``open_loop_expected_at_that_size``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from arf.delivery import DEFAULT_LEVEL_PERCENT, time_weights
from arf.events import EventTable
from arf.failsignal import adaptive_selection, open_loop_plan
from arf.greedy import greedy_selection

#: Objectives of the releases: name and decay rate of the time weight (zero: no weighting).
DEFAULT_OBJECTIVES = (("time_aware", 3.0), ("coverage_first", 0.0))


@dataclass(frozen=True)
class PlanOutcome:
    """What one plan achieves under one cloud realisation.

    Attributes:
        images: Number of images in the plan.
        covered_cells: Region cells with a usable, cloud-free observation among them.
    """

    images: int
    covered_cells: int


@dataclass(frozen=True)
class ObjectiveTrial:
    """The plans of one objective under one cloud realisation.

    Attributes:
        plans: Outcome of each plan, by the names in the module docstring.
        replans: Fail-signals seen by the adaptive selector.
        compensatory: Images the adaptive selector added beyond its budget.
    """

    plans: dict[str, PlanOutcome]
    replans: int
    compensatory: int


@dataclass(frozen=True)
class CloudTrial:
    """One cloud realisation.

    Attributes:
        cloudy_share_percent: Share of observations the realisation makes cloudy.
        reachable_cells: Region cells with at least one usable, cloud-free observation among all images.
        objectives: The trial of each objective, by name.
    """

    cloudy_share_percent: float
    reachable_cells: int
    objectives: dict[str, ObjectiveTrial]


def covered_cells(events: EventTable, is_cloudy: np.ndarray | None, images: Sequence[int]) -> int:
    """Count the region cells with a usable observation, not cloudy, among the given images."""
    utility = events.utility if is_cloudy is None else np.where(is_cloudy, 0.0, events.utility)
    seen = np.zeros(len(events.region_cells), dtype=bool)
    for image in images:
        rows = events.rows(image)
        seen[events.cell_index[rows][utility[rows] > 0.0]] = True
    return int(seen.sum())


def reachable_cells(events: EventTable, is_cloudy: np.ndarray | None) -> int:
    """Count the region cells with a usable observation, not cloudy, among all images."""
    utility = events.utility if is_cloudy is None else np.where(is_cloudy, 0.0, events.utility)
    return int(np.unique(events.cell_index[utility > 0.0]).size)


def cloud_trial(
    events: EventTable,
    is_cloudy: np.ndarray,
    cloud_probability: np.ndarray,
    budget: int,
    orbit_period_min: float,
    objectives: Sequence[tuple[str, float]] = DEFAULT_OBJECTIVES,
    time_objective: str = "facility",
    decay_scale: str = "fleet",
    *,
    planning: str = "expected",
    level_percent: float = DEFAULT_LEVEL_PERCENT,
) -> CloudTrial:
    """Run every plan of every objective under one cloud realisation.

    Args:
        events: The observations.
        is_cloudy: The realisation, one boolean per observation.
        cloud_probability: Cloud probability of each region cell, for the planners.
        budget: Budget of the adaptive selectors (``max_selection_steps`` of the releases).
        orbit_period_min: One orbital period of the fleet, the shortest time scale of the time weight.
        objectives: Name and decay rate of each objective.
        time_objective: ``"facility"`` or ``"legacy"`` (``arf.greedy.greedy_selection``).
        decay_scale: ``"fleet"`` or ``"fixed"`` (``arf.delivery.decay_time_scale``).
        planning: ``cloud_planning`` of the adaptive selector (``arf.failsignal.adaptive_selection``); the
            selector of release 1.4.0 always plans on clear-sky geometry.
        level_percent: ``h1_reachable_coverage_pct``, the level that sets the fleet time scale of the weights.

    Returns:
        The outcomes, by objective and plan.
    """
    trials = {}
    for name, decay_rate in objectives:
        weights = (
            time_weights(events, decay_rate, orbit_period_min, decay_scale, level_percent) if decay_rate > 0 else None
        )
        adaptive = adaptive_selection(events, budget, is_cloudy, cloud_probability, weights, time_objective, planning)
        size = len(adaptive.images)
        earlier_selector = adaptive_selection(
            events, budget, is_cloudy, cloud_probability, weights, time_objective, planning="clear"
        )
        plans = {
            "adaptive": adaptive.images,
            "open_loop_expected": open_loop_plan(events, size, cloud_probability, weights),
            "open_loop_clear": greedy_selection(events, size, None, weights, time_objective, gap_fill=False).images,
            "oracle": greedy_selection(events, size, is_cloudy, weights, time_objective, gap_fill=False).images,
            "adaptive_clear_planning": earlier_selector.images,
            "open_loop_expected_at_that_size": open_loop_plan(
                events, len(earlier_selector.images), cloud_probability, weights
            ),
        }
        trials[name] = ObjectiveTrial(
            plans={
                plan: PlanOutcome(images=len(images), covered_cells=covered_cells(events, is_cloudy, images))
                for plan, images in plans.items()
            },
            replans=adaptive.replans,
            compensatory=adaptive.compensatory,
        )
    return CloudTrial(
        cloudy_share_percent=100.0 * int(is_cloudy.sum()) / max(1, is_cloudy.size),
        reachable_cells=reachable_cells(events, is_cloudy),
        objectives=trials,
    )
