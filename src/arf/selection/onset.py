"""What can be delivered after an onset, and what each selector needs to deliver it.

An onset is the moment an event (a fire, a flood) starts. The images acquired in
the following horizon are the candidates; the coverage level is a share of the
cells those candidates can reach at all. Three quantities are computed:

* the **floor**: the earliest time at which all images delivered so far reach
  the level. No selector can do better;
* **first-come-first-served (FCFS)**: take every delivered image that adds a
  cell, in delivery order, until the level is reached;
* the **earliest-completion cover (ECC)**: the earliest time at which a greedy
  cover of at most ``budget`` delivered images reaches the level, and that cover.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from arf.selection.cover import greedy_cover
from arf.selection.images import ImageSet

#: Coverage level of the protocol: 95% of the reachable cells.
DEFAULT_LEVEL_PERCENT = 95.0
#: Candidates are the images acquired in the 24 h after the onset.
DEFAULT_HORIZON_MIN = 1440
#: Largest number of images in the registered releases (``max_selection_steps``).
DEFAULT_BUDGET = 30
#: Guards the comparison of a share with the level against rounding.
_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Completion:
    """The earliest-completion cover for one budget.

    Attributes:
        time_min: Minutes after the onset at which the last chosen image is
            delivered, or ``None`` if no cover within the budget reaches the level.
        images: Number of images chosen, or ``None`` likewise.
    """

    time_min: float | None
    images: int | None


@dataclass(frozen=True)
class OnsetResult:
    """Floor, baseline and earliest-completion cover for one onset.

    Attributes:
        onset_min: The onset, in minutes after the start of the run.
        reachable: Cells covered by at least one candidate image.
        required: Cells needed to reach the level.
        delivered_images: Candidate images that are delivered at all.
        floor_min: Minutes after the onset at which the level is first
            reachable, or ``None`` if the delivered images never reach it.
        fcfs_images: Images first-come-first-served needs, or ``None`` likewise.
        completion: The earliest-completion cover for each budget asked for;
            the key ``None`` stands for no limit on the number of images.
    """

    onset_min: int
    reachable: int
    required: int
    delivered_images: int
    floor_min: float | None
    fcfs_images: int | None
    completion: dict[int | None, Completion] = field(default_factory=dict)


def onset_grid(span_h: int = 24, step_min: int = 20) -> list[int]:
    """Return the onsets of the protocol: every ``step_min`` minutes over ``span_h`` hours, both ends included."""
    return list(range(0, span_h * 60 + 1, step_min))


def evaluate_onset(
    images: ImageSet,
    onset_min: int,
    budgets: Iterable[int | None] = (DEFAULT_BUDGET,),
    level_percent: float = DEFAULT_LEVEL_PERCENT,
    horizon_min: int = DEFAULT_HORIZON_MIN,
) -> OnsetResult | None:
    """Compute the floor, FCFS and the earliest-completion cover for one onset, in clear sky.

    Args:
        images: The candidate images of the run.
        onset_min: The onset, in minutes after the start of the run.
        budgets: Budgets for the earliest-completion cover; ``None`` means no limit.
        level_percent: Coverage level, in percent of the reachable cells.
        horizon_min: Images acquired within this many minutes of the onset are candidates.

    Returns:
        The result, or ``None`` when no candidate image covers any cell.
    """
    in_horizon = (images.acquired_min >= onset_min) & (images.acquired_min < onset_min + horizon_min)
    candidates = np.flatnonzero(in_horizon)
    if candidates.size == 0:
        return None
    is_reachable = np.zeros(images.n_cells, dtype=bool)
    for image in candidates:
        is_reachable[images.cells[image]] = True
    reachable = int(is_reachable.sum())
    if reachable == 0:
        return None
    required = math.ceil(level_percent / 100.0 * reachable - _TOLERANCE)

    # Candidates in delivery order; the stable sort keeps satellite-then-time order on ties.
    delivery = images.acquired_min[candidates] - onset_min + images.wait_min[candidates]
    order = np.argsort(delivery, kind="stable")
    order = order[np.isfinite(delivery[order])]
    by_delivery = candidates[order]
    delivery = delivery[order]

    floor_min, fcfs_images, floor_position = _floor_and_fcfs(
        images, by_delivery, delivery, reachable, required, level_percent
    )
    completion = {
        budget: _earliest_completion(images, by_delivery, delivery, required, floor_position, budget)
        for budget in budgets
    }
    return OnsetResult(
        onset_min=onset_min,
        reachable=reachable,
        required=required,
        delivered_images=int(by_delivery.size),
        floor_min=floor_min,
        fcfs_images=fcfs_images,
        completion=completion,
    )


def _floor_and_fcfs(
    images: ImageSet,
    by_delivery: np.ndarray,
    delivery: np.ndarray,
    reachable: int,
    required: int,
    level_percent: float,
) -> tuple[float | None, int | None, int | None]:
    """Walk the images in delivery order until the level is reached.

    Returns the floor, the number of images FCFS keeps, and the position in
    delivery order at which the level is reached; all ``None`` if it never is.
    """
    seen = np.zeros(images.n_cells, dtype=bool)
    seen_count = 0
    kept = 0
    floor_min: float | None = None
    fcfs_images: int | None = None
    floor_position: int | None = None
    for position, image in enumerate(by_delivery):
        image_cells = images.cells[image]
        new_cells = image_cells[~seen[image_cells]]
        if new_cells.size == 0:
            continue
        seen[new_cells] = True
        seen_count += new_cells.size
        kept += 1
        if floor_min is None and 100.0 * seen_count / reachable >= level_percent - _TOLERANCE:
            floor_min = float(delivery[position])
        if fcfs_images is None and seen_count >= required:
            fcfs_images = kept
            floor_position = position
        if floor_min is not None and fcfs_images is not None:
            break
    return floor_min, fcfs_images, floor_position


def _earliest_completion(
    images: ImageSet,
    by_delivery: np.ndarray,
    delivery: np.ndarray,
    required: int,
    floor_position: int | None,
    budget: int | None,
) -> Completion:
    """Find the earliest delivery time at which a greedy cover within the budget reaches the level.

    A cover from the images delivered by a later time is searched by bisection
    between the floor and the last delivery, as in the registered releases. The
    bisection assumes that a cover found at one time is also found at any later
    time; the greedy rule does not guarantee this, and the releases share the
    assumption, so the result is theirs exactly.
    """
    if floor_position is None:
        return Completion(time_min=None, images=None)

    def cover_at(position: int) -> tuple[bool, tuple[int, ...]]:
        # All images delivered no later than the image at `position`, ties included.
        stop = int(np.searchsorted(delivery, delivery[position], side="right"))
        result = greedy_cover(by_delivery[:stop].tolist(), images.cells, images.n_cells, required, budget)
        return result.complete, result.chosen

    if budget is None:
        complete, best = cover_at(floor_position)
    else:
        last = len(by_delivery) - 1
        complete, best = cover_at(last)
        if complete:
            low, high = floor_position, last
            while low < high:
                middle = (low + high) // 2
                found, chosen = cover_at(middle)
                if found:
                    high, best = middle, chosen
                else:
                    low = middle + 1
    if not complete:
        return Completion(time_min=None, images=None)
    delivered_at = dict(zip(by_delivery.tolist(), delivery.tolist(), strict=True))
    return Completion(time_min=float(max(delivered_at[image] for image in best)), images=len(best))
