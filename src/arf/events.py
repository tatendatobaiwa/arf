"""Every observation of a cell by an image, as the access and refinement stages write them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from arf.grid import region_cells_from_run_folder

#: Columns of ``arf_refined_events.csv`` that the event table needs.
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
class EventTable:
    """Observations grouped by image (compressed sparse rows).

    An image is one pass of one satellite, identified by satellite name and
    acquisition minute. Images are ordered by satellite name and then time; the
    observations of image ``i`` are rows ``offsets[i]:offsets[i + 1]`` of
    ``cell_index`` and ``utility``, in the order of the refined events file.
    Unlike ``arf.selection.ImageSet``, observations of zero utility are kept:
    the cloud model draws for every observation.

    Attributes:
        satellites: Satellite name of each image.
        acquired_min: Acquisition time of each image, in whole minutes after the start of the run.
        wait_min: Minutes from acquisition to delivery; ``nan`` when the image is never delivered.
        offsets: Row boundaries of the images; one entry more than there are images.
        cell_index: For each observation, the index of the cell in ``region_cells``.
        utility: For each observation, ``max(0, 1 - off-nadir angle / swath-edge angle) * downlink reward``.
        region_cells: Identifiers of all cells of the region, sorted.
    """

    satellites: tuple[str, ...]
    acquired_min: np.ndarray
    wait_min: np.ndarray
    offsets: np.ndarray
    cell_index: np.ndarray
    utility: np.ndarray
    region_cells: np.ndarray

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.satellites)

    @property
    def n_observations(self) -> int:
        """Number of observations (rows)."""
        return int(self.cell_index.size)

    @property
    def observations_per_image(self) -> np.ndarray:
        """Number of observations of each image."""
        return np.diff(self.offsets)

    def rows(self, image: int) -> slice:
        """Return the rows that hold the observations of one image."""
        return slice(int(self.offsets[image]), int(self.offsets[image + 1]))

    @classmethod
    def from_refined_events(cls, events: pd.DataFrame, region_cells: np.ndarray) -> EventTable:
        """Build the table from the rows of ``arf_refined_events.csv``.

        Args:
            events: Table with the columns in ``REFINED_EVENT_COLUMNS``.
            region_cells: Identifiers of all cells of the region, sorted.

        Returns:
            The observations, grouped by image in satellite-then-time order.
        """
        missing = [c for c in REFINED_EVENT_COLUMNS if c not in events.columns]
        if missing:
            raise ValueError(f"refined events lack the columns {missing}")
        region_cells = np.asarray(region_cells)
        if not np.array_equal(region_cells, np.array(sorted(region_cells))):
            raise ValueError("region_cells must be sorted")
        position = {cell: index for index, cell in enumerate(region_cells)}
        n_rows = len(events)
        cell_index = np.fromiter((position[cell] for cell in events["tile"]), dtype=np.int32, count=n_rows)
        satellite_names = sorted(events["sat"].unique())
        satellite_code = pd.Categorical(events["sat"], categories=satellite_names).codes.astype(np.int64)
        acquired = events["t_min"].to_numpy(np.int64)

        off_nadir = events["A_nadir_deg"].to_numpy(np.float64)
        swath_edge = events["theta_max_deg"].to_numpy(np.float64)
        utility = (1.0 - off_nadir / swath_edge).clip(min=0.0) * events["D_downlink"].to_numpy(np.float64)

        order = np.lexsort((np.arange(n_rows), acquired, satellite_code))
        satellite_code, acquired = satellite_code[order], acquired[order]
        new_image = np.r_[True, (satellite_code[1:] != satellite_code[:-1]) | (acquired[1:] != acquired[:-1])]
        start = np.flatnonzero(new_image) if n_rows else np.zeros(0, dtype=np.int64)
        return cls(
            satellites=tuple(satellite_names[code] for code in satellite_code[start]),
            acquired_min=acquired[start],
            wait_min=events["T_wait_min"].to_numpy(np.float64)[order][start],
            offsets=np.r_[start, n_rows].astype(np.int64),
            cell_index=cell_index[order],
            utility=utility[order],
            region_cells=region_cells,
        )

    @classmethod
    def from_run_folder(cls, run_folder: str | Path) -> EventTable:
        """Read the refined events and the region cells from the output folder of a run, every number exactly."""
        from arf.runs import read_csv_exact

        events = read_csv_exact(Path(run_folder) / "arf_refined_events.csv", usecols=list(REFINED_EVENT_COLUMNS))
        return cls.from_refined_events(events, region_cells_from_run_folder(run_folder))
