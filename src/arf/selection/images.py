"""The candidate images of a run: when each is acquired, when it is delivered, what it covers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Columns of ``arf_refined_events.csv`` that the selection needs.
REFINED_EVENT_COLUMNS = (
    "sat",
    "t_min",
    "tile",
    "A_nadir_deg",
    "theta_max_deg",
    "D_downlink",
    "T_wait_min",
)


@dataclass(frozen=True)
class ImageSet:
    """Candidate images over a region divided into cells.

    An image is one continuous pass of one satellite over the region. Images are
    ordered by satellite name and then by acquisition time; that order is part of
    the method, because ties between images are broken by it.

    Attributes:
        acquired_min: Acquisition time of each image, in whole minutes after the
            start of the run.
        wait_min: Minutes from acquisition to delivery on the ground. ``inf``
            marks an image that is never delivered.
        cells: For each image, the indices of the cells it covers with positive
            utility, in file order. A cell listed twice is counted twice, as in
            the registered releases; ``images_with_repeated_cells`` reports it.
        n_cells: Number of distinct cells in the run.
    """

    acquired_min: np.ndarray
    wait_min: np.ndarray
    cells: tuple[np.ndarray, ...]
    n_cells: int

    def __post_init__(self) -> None:
        """Check that the three per-image sequences have one entry per image."""
        if not len(self.acquired_min) == len(self.wait_min) == len(self.cells):
            raise ValueError("acquired_min, wait_min and cells must have one entry per image")

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.cells)

    @property
    def images_with_repeated_cells(self) -> int:
        """Number of images that list a cell more than once."""
        return int(sum(np.unique(c).size != c.size for c in self.cells))

    @classmethod
    def from_refined_events(cls, events: pd.DataFrame) -> ImageSet:
        """Build the image set from the rows of ``arf_refined_events.csv``.

        Each row is one cell seen by one image. The utility of a row is
        ``max(0, 1 - A_nadir_deg / theta_max_deg) * D_downlink``; a cell counts
        for an image only when its utility is positive.

        Args:
            events: Table with the columns in ``REFINED_EVENT_COLUMNS``.

        Returns:
            The images, ordered by satellite name and then acquisition time.
        """
        missing = [c for c in REFINED_EVENT_COLUMNS if c not in events.columns]
        if missing:
            raise ValueError(f"refined events lack the columns {missing}")
        off_nadir = events["A_nadir_deg"].to_numpy(float)
        limit = events["theta_max_deg"].to_numpy(float)
        utility = np.clip(1.0 - off_nadir / limit, 0.0, None) * events["D_downlink"].to_numpy(float)
        cell_index, cell_names = pd.factorize(events["tile"])
        satellite = pd.factorize(events["sat"], sort=True)[0]
        acquired = events["t_min"].to_numpy(float)

        # Satellite, then time, then file order: the order the registered releases use.
        order = np.lexsort((np.arange(len(events)), acquired, satellite))
        satellite, acquired = satellite[order], acquired[order]
        new_image = np.r_[True, (satellite[1:] != satellite[:-1]) | (acquired[1:] != acquired[:-1])]
        start = np.flatnonzero(new_image)
        stop = np.r_[start[1:], len(order)]

        wait = events["T_wait_min"].to_numpy(float)[order][start]
        counts = utility[order] > 0.0
        cell_index = cell_index[order].astype(np.int64)
        cells = tuple(cell_index[a:b][counts[a:b]] for a, b in zip(start, stop, strict=True))
        return cls(
            acquired_min=acquired[start].astype(np.int64),
            wait_min=np.where(np.isnan(wait), np.inf, wait),
            cells=cells,
            n_cells=len(cell_names),
        )

    @classmethod
    def from_run_folder(cls, run_folder: str | Path) -> ImageSet:
        """Read ``arf_refined_events.csv`` from the output folder of a run."""
        path = Path(run_folder) / "arf_refined_events.csv"
        return cls.from_refined_events(pd.read_csv(path, usecols=list(REFINED_EVENT_COLUMNS)))
