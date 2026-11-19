"""Greedy cover: repeatedly take the image that adds the most uncovered cells."""

from __future__ import annotations

import heapq
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GreedyCover:
    """Outcome of a greedy cover.

    Attributes:
        complete: Whether the chosen images cover the required number of cells.
        chosen: The chosen images, in the order they were taken.
        covered: Number of cells covered by the chosen images.
    """

    complete: bool
    chosen: tuple[int, ...]
    covered: int


def greedy_cover(
    candidates: Sequence[int],
    cells: Sequence[np.ndarray],
    n_cells: int,
    required: int,
    budget: int | None = None,
) -> GreedyCover:
    """Cover ``required`` cells with as few of the candidate images as the greedy rule finds.

    Each step takes the candidate that covers the most cells not yet covered.
    Ties go to the candidate listed first, so the order of ``candidates`` matters;
    the callers list them by delivery time. The cover stops when the required
    number is reached, when the budget is spent, or when no candidate adds a cell.

    The gain of a candidate can only fall as cells are covered. The gains are
    therefore kept as upper bounds in a heap and recomputed only for the
    candidate at the top (lazy evaluation). The result is the same, at every
    step and for every tie, as recomputing all gains each time, at a fraction of
    the cost when there are thousands of candidates.

    Args:
        candidates: Indices into ``cells`` of the images that may be chosen.
        cells: For each image, the indices of the cells it covers.
        n_cells: Number of distinct cells in the run.
        required: Number of cells to cover.
        budget: Largest number of images that may be chosen; ``None`` for no limit.

    Returns:
        The chosen images and whether they suffice.
    """
    is_covered = np.zeros(n_cells, dtype=bool)
    # Heap entries: (minus the gain bound, position in `candidates`, step at which it was computed).
    heap = [(-int(cells[image].size), position, 0) for position, image in enumerate(candidates)]
    heapq.heapify(heap)
    chosen: list[int] = []
    covered = 0
    step = 0
    while heap and (budget is None or step < budget):
        negative_gain, position, computed_at = heapq.heappop(heap)
        image = candidates[position]
        if computed_at != step:
            gain = int((~is_covered[cells[image]]).sum())
            heapq.heappush(heap, (-gain, position, step))
            continue
        if negative_gain >= 0:
            break
        image_cells = cells[image]
        covered += int((~is_covered[image_cells]).sum())
        is_covered[image_cells] = True
        chosen.append(image)
        step += 1
        if covered >= required:
            break
    return GreedyCover(complete=covered >= required, chosen=tuple(chosen), covered=covered)
