"""When images reach the ground, and how the time-aware objective weights them."""

from __future__ import annotations

import numpy as np

from arf.events import EventTable

#: Level of the protocol: 95% of the cells the images can reach.
DEFAULT_LEVEL_PERCENT = 95.0
#: Weights are floored here: ``exp`` underflows to zero beyond about 745 time scales,
#: which would make a cell seen only by a late image impossible to select.
SMALLEST_WEIGHT = 1e-300


def delivery_times(events: EventTable) -> np.ndarray:
    """Return, per image, the minutes from the start of the run to delivery; ``inf`` if never delivered."""
    wait = np.where(np.isnan(events.wait_min), np.inf, events.wait_min)
    return events.acquired_min.astype(np.float64) + wait


def floor_time(events: EventTable, level_percent: float = DEFAULT_LEVEL_PERCENT) -> float | None:
    """Return the delivery time at which all images together first reach the level, in clear sky.

    The level is a share of the cells that at least one image observes with
    positive utility. Images are taken in delivery order; ties keep
    satellite-then-time order.

    Returns:
        The time in minutes from the start of the run, or ``None`` when no cell
        is reachable or the delivered images never reach the level.
    """
    if len(events) == 0:
        return None
    usable = events.utility > 0.0
    reachable = int(np.unique(events.cell_index[usable]).size)
    if reachable == 0:
        return None
    delivered_at = delivery_times(events)
    seen = np.zeros(len(events.region_cells), dtype=bool)
    seen_count = 0
    for image in np.argsort(delivered_at, kind="stable"):
        if not np.isfinite(delivered_at[image]):
            break  # undelivered images sort last
        rows = events.rows(image)
        cells = events.cell_index[rows][usable[rows]]
        new_cells = cells[~seen[cells]]
        if new_cells.size:
            seen[new_cells] = True
            seen_count += new_cells.size
            if 100.0 * seen_count / reachable >= level_percent - 1e-9:
                return float(delivered_at[image])
    return None


def decay_time_scale(
    events: EventTable,
    orbit_period_min: float,
    scale: str = "fleet",
    level_percent: float = DEFAULT_LEVEL_PERCENT,
) -> float:
    """Return the time scale ``tau``, in minutes, of the time-aware weight.

    ``"fleet"``: the longer of one orbital period and the floor of these
    images, so that an image delivered at the floor is weighted ``exp(-rate)``
    whatever the fleet. A sparse fleet needs a day to cover a region; on a
    one-orbit scale every image after the first orbit would weigh almost nothing.
    ``"fixed"``: one orbital period.
    """
    if scale == "fixed":
        return float(orbit_period_min)
    floor = floor_time(events, level_percent)
    return float(orbit_period_min) if floor is None else max(float(orbit_period_min), floor)


def time_weights(
    events: EventTable,
    decay_rate: float,
    orbit_period_min: float,
    scale: str = "fleet",
    level_percent: float = DEFAULT_LEVEL_PERCENT,
) -> np.ndarray:
    """Return the weight ``exp(-decay_rate * delivery time / tau)`` of each image.

    An image that is never delivered has weight zero; its utility is zero too.
    """
    tau = decay_time_scale(events, orbit_period_min, scale, level_percent)
    delivered_at = delivery_times(events)
    is_delivered = np.isfinite(delivered_at)
    weight = np.where(is_delivered, np.exp(-decay_rate * np.where(is_delivered, delivered_at, 0.0) / tau), 0.0)
    return np.where(is_delivered, np.maximum(weight, SMALLEST_WEIGHT), 0.0)
