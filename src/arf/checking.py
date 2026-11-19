"""Comparing what release 2 computes with what a registered run wrote, for the tools in ``tools/``.

Each tool gives two verdicts per run:

* **Port**: release 2 against the registered cells executed unchanged in the same session, on the same inputs. They
  must be identical to the last bit.
* **Run**: release 2 against the run's own files. *Identical*; *identical up to rounding*, when every value that is
  not a floating-point number is identical and the floating-point values differ by at most a relative tolerance; or
  *different*. A mean over thousands of values depends in its last bits on the order of summation, which NumPy may
  change between versions and processors.
"""

from __future__ import annotations

import math
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from arf.cloud import LatentCloudField, correlated_cloud_mask, independent_cloud_mask
from arf.events import EventTable
from arf.grid import cell_centres
from arf.report.onsets import OnsetSettings
from arf.runs import orbit_period_min

#: Largest relative difference of a floating-point value still counted as rounding, by default.
DEFAULT_ROUNDING = 1e-12


@dataclass
class Differences:
    """How two results differ: in any value, and beyond rounding (in anything but a float within tolerance)."""

    exact: list[str] = field(default_factory=list)
    beyond_rounding: list[str] = field(default_factory=list)
    largest_relative: float = 0.0

    def add(self, other: Differences) -> None:
        """Add another comparison's differences to these."""
        self.exact += other.exact
        self.beyond_rounding += other.beyond_rounding
        self.largest_relative = max(self.largest_relative, other.largest_relative)

    def describe_port(self, release: str) -> str:
        """The port verdict, as the tools print it."""
        if not self.exact:
            return f"port vs registered {release}: IDENTICAL"
        return f"port vs registered {release}: DIFFERS in {len(self.exact)}: {', '.join(self.exact[:6])}"

    def describe_run(self) -> str:
        """The run verdict, as the tools print it."""
        if not self.exact:
            return "vs the run's files: identical"
        if not self.beyond_rounding:
            return (
                f"vs the run's files: identical up to rounding ({len(self.exact)} floating-point values differ, "
                f"largest relative difference {self.largest_relative:.1e}: {', '.join(self.exact[:6])}"
                + (", ..." if len(self.exact) > 6 else "")
                + ")"
            )
        return f"vs the run's files: DIFFERENT in {len(self.beyond_rounding)}: {', '.join(self.beyond_rounding[:6])}"


def _relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), np.finfo(float).tiny)


def compare_tables(ours: pd.DataFrame, theirs: pd.DataFrame, rounding: float, unit: str = "rows") -> Differences:
    """Compare two tables column by column; every column that differs is listed, with how many ``unit`` differ."""
    found = Differences()
    if list(ours.columns) != list(theirs.columns) or len(ours) != len(theirs):
        text = f"shape or columns differ ({ours.shape} and {theirs.shape})"
        found.exact.append(text)
        found.beyond_rounding.append(text)
        return found
    for column in ours.columns:
        a, b = ours[column], theirs[column]
        same = (a == b) | (a.isna() & b.isna())
        if same.all():
            continue
        found.exact.append(f"{column} ({int((~same).sum())} {unit})")
        floats = pd.api.types.is_float_dtype(a) and pd.api.types.is_float_dtype(b)
        both = (~same) & a.notna() & b.notna()
        relative = [_relative(float(x), float(y)) for x, y in zip(a[both], b[both], strict=True)] if floats else []
        found.largest_relative = max([found.largest_relative, *relative])
        if not floats or (~same & ~both).any() or max(relative, default=0.0) > rounding:
            found.beyond_rounding.append(f"{column} ({int((~same).sum())} {unit})")
    return found


def compare_summaries(ours, theirs, rounding: float, path: str = "") -> Differences:
    """Compare two summaries key by key, and lists item by item; every path that differs is listed."""
    found = Differences()
    if isinstance(ours, dict) and isinstance(theirs, dict):
        for key in dict.fromkeys([*theirs, *ours]):
            found.add(compare_summaries(ours.get(key), theirs.get(key), rounding, f"{path}.{key}" if path else key))
        return found
    if isinstance(ours, list) and isinstance(theirs, list) and len(ours) == len(theirs):
        for i, (a, b) in enumerate(zip(ours, theirs, strict=True)):
            found.add(compare_summaries(a, b, rounding, f"{path}[{i}]"))
        return found
    if ours == theirs or (_is_nan(ours) and _is_nan(theirs)):
        return found
    found.exact.append(path)
    if isinstance(ours, float) and isinstance(theirs, float) and math.isfinite(ours) and math.isfinite(theirs):
        relative = _relative(ours, theirs)
        found.largest_relative = relative
        if relative <= rounding:
            return found
    found.beyond_rounding.append(f"{path} (release 2 {str(ours)[:40]}, run {str(theirs)[:40]})")
    return found


def _is_nan(value) -> bool:
    return isinstance(value, float) and math.isnan(value)


def run_cloud(events: EventTable, settings: dict) -> np.ndarray:
    """Return the cloud of a run's Cell 6: the mask its model draws with its seed."""
    probability = float(settings["cloud_probability"])
    seed = int(settings["random_seed"])
    if settings["cloud_model"] == "correlated":
        latitude, longitude = cell_centres(events.region_cells)
        field_ = LatentCloudField(latitude, longitude, float(settings["cloud_corr_length_km"]))
        return correlated_cloud_mask(events, probability, seed, field_, float(settings["cloud_corr_time_h"]))
    if settings["cloud_model"] == "iid":
        return independent_cloud_mask(events, probability, seed)
    raise ValueError(f"cloud model {settings['cloud_model']!r} is not one of the registered releases'")


def onset_settings(run_folder: Path, events: EventTable, settings: dict) -> OnsetSettings:
    """The settings of Cell 7C's selectors for a run, from its manifest and Cell 2's orbital period."""
    return OnsetSettings(
        orbit_period_min=orbit_period_min(run_folder),
        cloud_probability=np.full(len(events.region_cells), float(settings["cloud_probability"])),
        budget=int(settings["max_selection_steps"]),
        decay_rate=settings["latency_decay_rate"],
        decay_scale=settings["latency_decay_scale"],
        window_hours=int(settings["window_hours"]),
        shift_hours=int(settings["onset_shift_hours"]),
        shift_step_min=int(settings["onset_shift_step_min"]),
        coverage_targets_pct=tuple(settings["coverage_targets_pct"]),
        reachable_level_pct=float(settings["h1_reachable_coverage_pct"]),
        random_draws=int(settings["random_baseline_draws"]),
        seed=int(settings["random_seed"]),
        cloud_knowledge=settings["cloud_knowledge"],
        cloud_planning=settings["cloud_planning"],
        time_objective=settings["time_aware_objective"],
    )


def numpy_of(manifest: dict) -> str:
    """The NumPy version a run's manifest records."""
    return str(((manifest.get("environment") or {}).get("packages") or {}).get("numpy", "not recorded"))


def verdict_of(port: Differences | None, run: Differences) -> str:
    """``"PORT DIFFERS"``, ``"RUN NOT REPRODUCED"`` or ``"agrees"``."""
    if port is not None and port.exact:
        return "PORT DIFFERS"
    if run.beyond_rounding:
        return "RUN NOT REPRODUCED"
    return "agrees"


def check_all(folders: list[Path], check: Callable[[Path], tuple[str, str]]) -> int:
    """Check every folder, printing one line each and a tally; return the exit status (0 if all agree)."""
    tally: dict[str, int] = {}
    for number, folder in enumerate(folders, 1):
        try:
            verdict, detail = check(folder)
        except Exception as error:  # one run's failure must not stop the others
            where = traceback.extract_tb(error.__traceback__)[-1]
            verdict = "ERROR"
            detail = f"{type(error).__name__}: {error} (at {Path(where.filename).name}:{where.lineno})"
        tally[verdict] = tally.get(verdict, 0) + 1
        print(f"[{number}/{len(folders)}] {folder.name}: {verdict}: {detail}", flush=True)
    others = ", ".join(f"{name}: {count}" for name, count in sorted(tally.items()) if name != "agrees")
    print(f"{tally.get('agrees', 0)} of {len(folders)} runs agree" + (f"; {others}" if others else ""))
    return 0 if tally.get("agrees", 0) == len(folders) else 1


def folders_from(run_folders: list[Path], indexes: list[Path]) -> list[Path]:
    """The run folders named, and those listed as done in batch index CSVs."""
    folders = list(run_folders)
    for index in indexes:
        table = pd.read_csv(index)
        folders += [Path(p) for p in table.loc[table["status"] == "done", "out_dir"]]
    return folders
