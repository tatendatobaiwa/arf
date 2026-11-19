"""Greedy selection of a fixed number of images, with the cloud known in advance or absent.

This is the cardinality-constrained greedy for a monotone submodular objective
(Nemhauser, Wolsey and Fisher, 1978). It is used for clear-sky plans and as the
clairvoyant reference under cloud. The selector that learns the cloud only as
images arrive is in ``arf.failsignal``.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from arf.events import EventTable

#: A gain no larger than this, in utility units, is rounding noise.
GAIN_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Selection:
    """Images chosen by a selector, and what they achieve.

    Attributes:
        images: Indices of the chosen images, in the order they were chosen.
        best_utility: For each region cell, the highest utility among the
            chosen observations that are not cloudy; zero where there is none.
        replans: Times the plan was revised after a failure (adaptive selector only).
        compensatory: Images added beyond the budget to fill gaps left by cloud.
    """

    images: tuple[int, ...]
    best_utility: np.ndarray
    replans: int = 0
    compensatory: int = 0

    @property
    def covered_cells(self) -> int:
        """Number of region cells with a usable, cloud-free observation."""
        return int((self.best_utility > 0.0).sum())


def marginal_gains(per_row_value: np.ndarray, best_per_cell: np.ndarray, events: EventTable) -> np.ndarray:
    """Return, per image, the total by which its observations would raise the best value of their cells."""
    improvement = np.maximum(per_row_value - best_per_cell[events.cell_index], 0.0)
    return np.add.reduceat(improvement, events.offsets[:-1])


def greedy_selection(
    events: EventTable,
    budget: int,
    is_cloudy: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    objective: str = "facility",
    gap_fill: bool = True,
) -> Selection:
    """Choose up to ``budget`` images, each step taking the largest marginal gain.

    Without ``weights`` the objective is coverage-first: the sum over cells of
    the best utility. With ``weights`` (one per image, from
    ``arf.delivery.time_weights``) it is time-aware:

    * ``"facility"``: the sum over cells of the best weighted utility, a
      facility-location function. It is monotone submodular, so the greedy
      attains at least ``1 - 1/e`` of the best value for the budget.
    * ``"legacy"``: the unweighted gain times the image's weight. These scores
      are not the marginal gains of one set function, and no bound applies.

    Ties go to the image that comes first in satellite-then-time order. The
    selection stops early when no remaining image adds to the objective.

    Args:
        events: The observations.
        budget: Largest number of images in the main selection.
        is_cloudy: One boolean per observation; cloudy observations count for
            nothing. Passing it means the cloud is known in advance.
        weights: One weight per image for the time-aware objective.
        objective: ``"facility"`` or ``"legacy"``; used only with ``weights``.
        gap_fill: With ``is_cloudy``, add images after the budget, always the
            one that clears the most still-uncovered cells, until none helps.

    Returns:
        The chosen images and the best utility per cell.
    """
    n_cells = len(events.region_cells)
    best = np.zeros(n_cells, dtype=np.float64)
    if len(events) == 0:
        return Selection(images=(), best_utility=best)
    cell_index = events.cell_index
    utility = np.where(is_cloudy, 0.0, events.utility) if is_cloudy is not None else events.utility
    is_selected = np.zeros(len(events), dtype=bool)
    facility = weights is not None and objective == "facility"
    if facility:
        weighted_utility = utility * np.repeat(weights, events.observations_per_image)
        best_weighted = np.zeros(n_cells, dtype=np.float64)
    chosen: list[int] = []

    for _ in range(budget):
        if facility:
            gains = marginal_gains(weighted_utility, best_weighted, events)
            # A gain counts only above the tolerance in the image's own utility units,
            # so that a new cell always counts however small the weight.
            score = np.where(gains > GAIN_TOLERANCE * weights, gains, -np.inf)
            tolerance = 0.0
        else:
            gains = marginal_gains(utility, best, events)
            score = gains * weights if weights is not None else gains
            tolerance = GAIN_TOLERANCE
        score = np.where(is_selected, -np.inf, score)
        image = int(np.argmax(score))
        if not np.isfinite(score[image]) or gains[image] <= tolerance:
            break
        is_selected[image] = True
        chosen.append(image)
        rows = events.rows(image)
        np.maximum.at(best, cell_index[rows], utility[rows])
        if facility:
            np.maximum.at(best_weighted, cell_index[rows], weighted_utility[rows])

    compensatory = 0
    if gap_fill and is_cloudy is not None and chosen:
        compensatory = _fill_gaps(events, utility, is_cloudy, is_selected, best, chosen)
    return Selection(images=tuple(chosen), best_utility=best, compensatory=compensatory)


def _fill_gaps(
    events: EventTable,
    utility: np.ndarray,
    is_cloudy: np.ndarray,
    is_selected: np.ndarray,
    best: np.ndarray,
    chosen: list[int],
) -> int:
    """Add images that clear cells the main selection left uncovered; return how many were added.

    Lazy greedy set cover (Minoux, 1978): a heap of the number of uncovered
    cells each image would clear, recounted only for the image at the top.
    Updates ``is_selected``, ``best`` and ``chosen`` in place.
    """
    cell_index = events.cell_index
    observed = np.zeros(len(events.region_cells), dtype=bool)
    observed[cell_index] = True
    remaining = observed & ~(best > 0.0)
    if not remaining.any():
        return 0
    clears = (events.utility > 0.0) & ~is_cloudy
    counts = np.add.reduceat((clears & remaining[cell_index]).astype(np.int64), events.offsets[:-1])
    heap = [(-int(count), image) for image, count in enumerate(counts) if count > 0 and not is_selected[image]]
    heapq.heapify(heap)
    added = 0
    while remaining.any() and heap:
        _, image = heapq.heappop(heap)
        if is_selected[image]:
            continue
        rows = events.rows(image)
        count = int((clears[rows] & remaining[cell_index[rows]]).sum())
        if count == 0:
            continue
        if heap and count < -heap[0][0]:
            heapq.heappush(heap, (-count, image))
            continue
        is_selected[image] = True
        chosen.append(image)
        added += 1
        cells, values = cell_index[rows], utility[rows]
        improves = values > best[cells]
        best[cells[improves]] = values[improves]
        remaining[cells[improves]] = False
    return added
