"""The Fail-Signal process: selection that learns the cloud only as images arrive.

A plan is made before any image is acquired. Images are then taken in the order
they are delivered, and the cloud over each is revealed on delivery. When a
planned observation comes back cloudy, that is a fail-signal: the failure is
announced by the arriving image itself, and the remaining budget is planned
again from what has actually been obtained. No decision uses the cloud over an
image before that image has arrived.

The name follows the fail-signal processes of Mpoeleng, Ezhilchelvan and Speirs
(2003), in which a faulty component signals its own failure. Here the component
is an image and the failure is cloud; network or node failures are not modelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arf.delivery import delivery_times
from arf.events import EventTable
from arf.greedy import GAIN_TOLERANCE, Selection, marginal_gains


def expected_coverage_plan(
    events: EventTable,
    still_missing: np.ndarray,
    is_candidate: np.ndarray,
    budget: int,
    cloud_probability: np.ndarray,
    weights: np.ndarray | None = None,
) -> list[int]:
    """Plan up to ``budget`` images to maximise the expected number of missing cells seen cloud-free.

    The planner assumes each cell is cloudy with its probability in every image
    independently. The gain of an image is the sum, over its usable observations
    of missing cells, of the probability that the plan has not yet covered the
    cell times the probability that this view is clear. This is the greedy on
    ``sum over cells of 1 - product of p``, a monotone submodular function.
    With ``weights`` the score is the gain times the image's weight, a heuristic
    with no approximation bound.

    Args:
        events: The observations.
        still_missing: One boolean per region cell: no cloud-free view yet.
        is_candidate: One boolean per image: may be planned.
        budget: Largest number of images to plan.
        cloud_probability: Cloud probability of each region cell (climatology, not the realised cloud).
        weights: Optional weight per image (time-aware objective).

    Returns:
        The planned images, in the order the greedy chose them.
    """
    cell_index = events.cell_index
    usable = events.utility > 0.0
    probability = np.asarray(cloud_probability, dtype=np.float64)
    not_yet_covered = np.where(still_missing, 1.0, 0.0)
    clear_probability = 1.0 - probability
    is_candidate = is_candidate.copy()
    plan: list[int] = []
    for _ in range(int(budget)):
        per_row = np.where(usable, not_yet_covered[cell_index] * clear_probability[cell_index], 0.0)
        gain = np.add.reduceat(per_row, events.offsets[:-1])
        score = gain * weights if weights is not None else gain
        score = np.where(is_candidate & (gain > GAIN_TOLERANCE), score, -np.inf)
        image = int(np.argmax(score))
        if not np.isfinite(score[image]):
            break
        is_candidate[image] = False
        plan.append(image)
        rows = events.rows(image)
        cells = np.unique(cell_index[rows][usable[rows]])
        not_yet_covered[cells] *= probability[cells]
    return plan


def open_loop_plan(
    events: EventTable,
    budget: int,
    cloud_probability: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[int, ...]:
    """Choose ``budget`` images in advance with the planner of the adaptive selector and no observation of cloud.

    The adaptive selector differs from this plan only in revising it from what
    arriving images show, so comparing the two isolates the value of acting on
    the fail-signals.
    """
    if len(events) == 0 or budget <= 0:
        return ()
    everything_missing = np.ones(len(events.region_cells), dtype=bool)
    is_delivered = np.isfinite(delivery_times(events))
    return tuple(
        expected_coverage_plan(events, everything_missing, is_delivered, int(budget), cloud_probability, weights)
    )


def clear_sky_plan(
    events: EventTable,
    best_so_far: np.ndarray,
    is_candidate: np.ndarray,
    budget: int,
    weights: np.ndarray | None,
    weighted_utility: np.ndarray | None,
) -> list[int]:
    """Plan up to ``budget`` images greedily on clear-sky utilities, starting from what is already covered.

    With ``weighted_utility`` the plan maximises the facility objective, and
    ``best_so_far`` is the best weighted utility per cell; otherwise it uses the
    plain utilities, scored by gain times weight when ``weights`` are given.
    """
    cell_index = events.cell_index
    value = events.utility if weighted_utility is None else weighted_utility
    tolerance = GAIN_TOLERANCE if weighted_utility is None else 0.0
    best = best_so_far.copy()
    is_candidate = is_candidate.copy()
    plan: list[int] = []
    for _ in range(budget):
        gains = marginal_gains(value, best, events)
        if weighted_utility is None:
            score = gains * weights if weights is not None else gains
        else:
            score = np.where(gains > GAIN_TOLERANCE * weights, gains, -np.inf)
        score = np.where(is_candidate, score, -np.inf)
        image = int(np.argmax(score))
        if not np.isfinite(score[image]) or gains[image] <= tolerance:
            break
        is_candidate[image] = False
        plan.append(image)
        rows = events.rows(image)
        np.maximum.at(best, cell_index[rows], value[rows])
    return plan


def _next_to_arrive(plan: list[int], delivered_at: np.ndarray) -> int:
    """Remove and return the planned image that is delivered first; ties go to the one planned first."""
    position = min(range(len(plan)), key=lambda j: (delivered_at[plan[j]], j))
    return plan.pop(position)


@dataclass
class _Progress:
    """What the adaptive selector has taken and obtained so far."""

    best: np.ndarray
    is_taken: np.ndarray
    taken: list[int] = field(default_factory=list)
    remaining_budget: int = 0
    last_arrival: float = -np.inf
    replans: int = 0
    compensatory: int = 0

    def take(self, image: int, delivered_at: np.ndarray) -> None:
        """Record that a planned image has arrived and been taken."""
        self.last_arrival = delivered_at[image]
        self.is_taken[image] = True
        self.taken.append(image)
        self.remaining_budget -= 1

    def not_arrived(self, is_delivered: np.ndarray, delivered_at: np.ndarray) -> np.ndarray:
        """Return the images that can still be planned: delivered, not taken, arriving no earlier than the last."""
        return (~self.is_taken) & is_delivered & (delivered_at >= self.last_arrival)


def adaptive_selection(
    events: EventTable,
    budget: int,
    is_cloudy: np.ndarray,
    cloud_probability: np.ndarray,
    weights: np.ndarray | None = None,
    objective: str = "facility",
    planning: str = "expected",
    gap_fill: bool = True,
) -> Selection:
    """Select images under cloud that is revealed only when each image arrives.

    ``planning="expected"`` (the releases' default): plan the budget for the
    largest expected coverage of the cells still missing
    (``expected_coverage_plan``). Take the planned image that arrives first,
    see its cloud, and plan the remaining budget again over the images not yet
    arrived, from the coverage actually obtained. A planned new cell that comes
    back cloudy is counted as a fail-signal.

    ``planning="clear"`` (release 1.4.0): plan greedily on clear-sky utilities
    and plan again only after a fail-signal, that is, when a cell a planned
    observation should have improved stays below its clear-sky utility.

    Compensation (``gap_fill``): after the budget, every arriving image that
    clearly covers a still-uncovered cell is taken, decided on arrival.

    Args:
        events: The observations.
        budget: Number of images in the plan.
        is_cloudy: The realised cloud, one boolean per observation. Read for an
            image only once that image has arrived.
        cloud_probability: Cloud probability of each region cell, for the planner.
        weights: Optional weight per image (time-aware objective), from clear-sky geometry.
        objective: ``"facility"`` or ``"legacy"``; used only with ``weights`` and ``planning="clear"``.
        planning: ``"expected"`` or ``"clear"``.
        gap_fill: Whether to compensate after the budget.

    Returns:
        The images taken, the best cloud-free utility per cell, the number of
        fail-signals and the number of compensatory images.
    """
    n_cells = len(events.region_cells)
    progress = _Progress(
        best=np.zeros(n_cells, dtype=np.float64),
        is_taken=np.zeros(len(events), dtype=bool),
        remaining_budget=budget,
    )
    if len(events) == 0:
        return Selection(images=(), best_utility=progress.best)
    realised = np.where(is_cloudy, 0.0, events.utility)
    delivered_at = delivery_times(events)
    if planning == "expected":
        probability = np.asarray(cloud_probability, dtype=np.float64)
        _follow_expected_plan(events, is_cloudy, realised, delivered_at, probability, weights, progress)
    else:
        facility = weights is not None and objective == "facility"
        _follow_clear_sky_plan(events, is_cloudy, realised, delivered_at, weights, facility, progress)
    if gap_fill and progress.taken:
        _compensate(events, realised, delivered_at, progress)
    return Selection(
        images=tuple(progress.taken),
        best_utility=progress.best,
        replans=progress.replans,
        compensatory=progress.compensatory,
    )


def _follow_expected_plan(
    events: EventTable,
    is_cloudy: np.ndarray,
    realised: np.ndarray,
    delivered_at: np.ndarray,
    probability: np.ndarray,
    weights: np.ndarray | None,
    progress: _Progress,
) -> None:
    """Take planned images as they arrive, planning again after every arrival from what was obtained."""
    cell_index = events.cell_index
    usable = events.utility > 0.0
    is_delivered = np.isfinite(delivered_at)
    has_clear_view = np.zeros(len(events.region_cells), dtype=bool)
    plan = expected_coverage_plan(
        events, ~has_clear_view, is_delivered, progress.remaining_budget, probability, weights
    )
    while progress.remaining_budget > 0 and plan:
        image = _next_to_arrive(plan, delivered_at)
        rows = events.rows(image)
        cells = cell_index[rows]
        # A fail-signal: a usable view of a cell still missing has come back cloudy.
        progress.replans += int(np.any(is_cloudy[rows] & usable[rows] & ~has_clear_view[cells]))
        np.maximum.at(progress.best, cells, realised[rows])
        has_clear_view[cells[realised[rows] > 0.0]] = True
        progress.take(image, delivered_at)
        if progress.remaining_budget > 0:
            candidates = progress.not_arrived(is_delivered, delivered_at)
            plan = expected_coverage_plan(
                events, ~has_clear_view, candidates, progress.remaining_budget, probability, weights
            )


def _follow_clear_sky_plan(
    events: EventTable,
    is_cloudy: np.ndarray,
    realised: np.ndarray,
    delivered_at: np.ndarray,
    weights: np.ndarray | None,
    facility: bool,
    progress: _Progress,
) -> None:
    """Take the images of a clear-sky plan as they arrive, planning again only after a fail-signal."""
    cell_index, utility = events.cell_index, events.utility
    is_delivered = np.isfinite(delivered_at)
    if facility:
        weight_per_row = np.repeat(weights, events.observations_per_image)
        planned_value, realised_value = utility * weight_per_row, realised * weight_per_row
        best_value = np.zeros(len(events.region_cells), dtype=np.float64)
    else:
        # Without the facility objective the plan is made on the plain utilities already obtained.
        planned_value, realised_value, best_value = None, None, progress.best
    plan = clear_sky_plan(events, best_value, is_delivered, progress.remaining_budget, weights, planned_value)
    while progress.remaining_budget > 0 and plan:
        image = _next_to_arrive(plan, delivered_at)
        rows = events.rows(image)
        cells = cell_index[rows]
        # A fail-signal: a cloudy view that should have raised the best value of its cell.
        if facility:
            failed = bool(np.any(is_cloudy[rows] & (planned_value[rows] > best_value[cells] + GAIN_TOLERANCE)))
            np.maximum.at(best_value, cells, realised_value[rows])
        else:
            failed = bool(np.any(is_cloudy[rows] & (utility[rows] > progress.best[cells] + GAIN_TOLERANCE)))
        np.maximum.at(progress.best, cells, realised[rows])
        progress.take(image, delivered_at)
        if failed and progress.remaining_budget > 0:
            progress.replans += 1
            candidates = progress.not_arrived(is_delivered, delivered_at)
            plan = clear_sky_plan(events, best_value, candidates, progress.remaining_budget, weights, planned_value)


def _compensate(events: EventTable, realised: np.ndarray, delivered_at: np.ndarray, progress: _Progress) -> None:
    """After the budget, take every arriving image that clearly covers a cell still uncovered."""
    cell_index = events.cell_index
    reachable = np.zeros(len(events.region_cells), dtype=bool)
    reachable[cell_index[events.utility > 0.0]] = True
    remaining = reachable & ~(progress.best > 0.0)
    if not remaining.any():
        return
    for image in np.argsort(delivered_at, kind="stable"):
        arrives_in_time = np.isfinite(delivered_at[image]) and delivered_at[image] >= progress.last_arrival
        if progress.is_taken[image] or not arrives_in_time:
            continue
        rows = events.rows(image)
        cells, values = cell_index[rows], realised[rows]
        clears = (values > 0.0) & remaining[cells]
        if clears.any():
            np.maximum.at(progress.best, cells, values)
            remaining[cells[clears]] = False
            progress.is_taken[image] = True
            progress.taken.append(int(image))
            progress.compensatory += 1
            if not remaining.any():
                break
