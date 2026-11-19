"""Which observations are cloudy in one realisation of the cloud model."""

from __future__ import annotations

import random

import numpy as np

from arf.cloud.field import LatentCloudField
from arf.events import EventTable


def correlated_cloud_mask(
    events: EventTable,
    cloud_probability: np.ndarray | float,
    seed: int,
    field: LatentCloudField,
    time_h: float,
) -> np.ndarray:
    """Draw the cloud mask of the correlated model for every observation.

    One latent field is held for the region and updated at each distinct
    acquisition minute, in time order, as a first-order autoregression:
    ``Z(t2) = rho Z(t1) + sqrt(1 - rho^2) E`` with ``rho = exp(-(t2 - t1) / time_h)``
    and ``E`` an independent field. An observation is cloudy when the field at
    its cell exceeds the cell's threshold. Images acquired in the same minute
    see the same cloud. ``time_h = 0`` makes the acquisition times independent.

    Args:
        events: The observations.
        cloud_probability: Marginal cloud probability, one value or one per region cell.
        seed: Seed of the realisation (``numpy.random.default_rng``).
        field: The latent field laid out for the region's cells and correlation length.
        time_h: The e-folding time of the temporal correlation, in hours.

    Returns:
        One boolean per observation, ``True`` where it is cloudy, aligned with ``events.cell_index``.
    """
    is_cloudy = np.zeros(events.n_observations, dtype=bool)
    if events.n_observations == 0:
        return is_cloudy
    probability = np.broadcast_to(np.asarray(cloud_probability, dtype=np.float64), events.region_cells.shape)
    threshold = field.thresholds(np.array(probability))

    time_of_row = np.repeat(events.acquired_min.astype(np.float64), events.observations_per_image)
    times, time_index = np.unique(time_of_row, return_inverse=True)
    rows_by_time = np.argsort(time_index, kind="stable")
    bounds = np.searchsorted(time_index[rows_by_time], np.arange(len(times) + 1))

    rng = np.random.default_rng(seed)
    current = None
    previous_time = None
    for step, time in enumerate(times):
        innovation = field.draw(rng)
        if current is None:
            current = innovation
        else:
            rho = 0.0 if time_h <= 0 else float(np.exp(-(time - previous_time) / (60.0 * time_h)))
            current = rho * current + np.sqrt(max(0.0, 1.0 - rho * rho)) * innovation
        previous_time = time
        rows = rows_by_time[bounds[step] : bounds[step + 1]]
        cells = events.cell_index[rows]
        is_cloudy[rows] = field.at_cells(current, cells) > threshold[cells]
    return is_cloudy


def independent_cloud_mask(events: EventTable, cloud_probability: np.ndarray | float, seed: int) -> np.ndarray:
    """Draw a cloud mask in which every observation is cloudy independently of all others.

    This is the model of the early releases, kept for the sensitivity analysis.
    Images are visited in satellite-then-time order and, within an image, cells
    in order of identifier; each observation takes the next number from
    ``random.Random(seed)``. The order is part of the definition: it fixes which
    random number goes to which observation.

    Args:
        events: The observations.
        cloud_probability: Marginal cloud probability, one value or one per region cell.
        seed: Seed of the realisation.

    Returns:
        One boolean per observation, ``True`` where it is cloudy.
    """
    is_cloudy = np.zeros(events.n_observations, dtype=bool)
    per_cell = None if np.isscalar(cloud_probability) else np.asarray(cloud_probability, dtype=np.float64)
    rng = random.Random(seed)
    for image in range(len(events)):
        image_rows = events.rows(image)
        rows = np.arange(image_rows.start, image_rows.stop)
        rows = rows[np.argsort(events.cell_index[image_rows], kind="stable")]
        if per_cell is None:
            is_cloudy[rows] = [rng.random() < cloud_probability for _ in range(rows.size)]
        else:
            probability = per_cell[events.cell_index[rows]]
            is_cloudy[rows] = [rng.random() < probability[i] for i in range(rows.size)]
    return is_cloudy
