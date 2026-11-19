"""Run parts of the registered releases, unchanged, so that release 2 can be compared with them.

The registered pipeline is a notebook exported as one file: its cells share global variables and run on import.
A part of it is therefore executed here as the notebook would execute it: the source between two markers is taken
from the release file as it is, and run in a namespace that holds the globals the earlier cells would have set.
"""

from __future__ import annotations

import functools
import heapq
import io
import logging
import os
import random
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RELEASES = {
    "1.5.3": ROOT / "pipeline" / "v1.5.3" / "algorithmic_reconstruction_framework.py",
    "1.5.6": ROOT / "pipeline" / "v1.5.6" / "algorithmic_reconstruction_framework_v156.py",
}


def source_between(release: str, start_marker: str, stop_marker: str, after: str | None = None) -> str:
    """Return the source of a release from the line starting with one marker up to the line starting with the other.

    With ``after``, the start marker is looked for only below the first line that starts with ``after``.
    """
    lines = RELEASES[release].read_text().splitlines(keepends=True)
    first = 0 if after is None else next(i for i, line in enumerate(lines) if line.startswith(after))
    start = next(i for i, line in enumerate(lines) if i >= first and line.startswith(start_marker))
    stop = next(i for i, line in enumerate(lines) if i > start and line.startswith(stop_marker))
    return "".join(lines[start:stop])


def run_cell6_events_and_cloud(
    release: str,
    refined_events: pd.DataFrame,
    region_cells: np.ndarray,
    cloud_probability: float = 0.35,
    seed: int = 42,
    cloud_model: str = "correlated",
    length_km: float = 50.0,
    time_h: float = 3.0,
) -> dict:
    """Execute the event-store and cloud-model part of Cell 6 of a registered release.

    Returns the namespace after execution: ``event_data`` (the EventStore), ``cloud_mask`` and ``build_cloud_mask``
    among others.
    """
    import h3

    source = source_between(release, "class EventStore:", "# DELIVERY ORDER AND THE TIME SCALE")
    namespace = {
        "np": np,
        "pd": pd,
        "h3": h3,
        "os": os,
        "random": random,
        "heapq": heapq,
        "logger": logging.getLogger("registered"),
        "_tile_ids": np.array(sorted(region_cells)),
        "target_tiles": list(region_cells),
        "refined_events_df": refined_events,
        "SWATH_HALF_ANGLE_DEG": 20.0,
        "CLOUD_PROBABILITY": cloud_probability,
        "CLOUD_PROBABILITY_SOURCE": "constant",
        "RANDOM_SEED": seed,
        "CLOUD_MODEL": cloud_model,
        "CLOUD_CORR_LENGTH_KM": length_km,
        "CLOUD_CORR_TIME_H": time_h,
    }
    exec(compile(source, str(RELEASES[release]), "exec"), namespace)
    return namespace


#: Functions of Cell 7C that the cloud trial needs, taken from the release by name.
CELL_7C_TRIAL_FUNCTIONS = (
    "_effective_util",
    "reachable_tile_count",
    "realized_coverage",
    "plan_expected_coverage",
    "_cloud_trial",
)


def functions_by_name(release: str, names: tuple[str, ...]) -> str:
    """Return the source of the named top-level functions of a release, unchanged."""
    import ast

    source = RELEASES[release].read_text()
    tree = ast.parse(source)
    found = {node.name: ast.get_source_segment(source, node) for node in tree.body if isinstance(node, ast.FunctionDef)}
    return "\n\n".join(found[name] for name in names)


def run_cell6_selection_and_trial(
    release: str,
    refined_events: pd.DataFrame,
    region_cells: np.ndarray,
    budget: int = 30,
    orbit_period_min: float = 94.0,
    decay_rate: float = 3.0,
    cloud_probability: float = 0.35,
    seed: int = 42,
    time_objective: str = "facility",
) -> dict:
    """Execute Cell 6 of a registered release up to its selectors, and define Cell 7C's cloud trial.

    Returns the namespace: ``event_data``, ``build_cloud_mask``, ``run_submodular_greedy_selection``,
    ``simulate_adaptive_selection``, ``time_aware_weights`` and ``_cloud_trial`` among others.
    """
    import h3

    source = source_between(release, "class EventStore:", "def compute_selection_latency")
    namespace = {
        "np": np,
        "pd": pd,
        "h3": h3,
        "os": os,
        "random": random,
        "heapq": heapq,
        "logger": logging.getLogger("registered"),
        "_tile_ids": np.array(sorted(region_cells)),
        "target_tiles": list(region_cells),
        "refined_events_df": refined_events,
        "SWATH_HALF_ANGLE_DEG": 20.0,
        "CLOUD_PROBABILITY": cloud_probability,
        "CLOUD_PROBABILITY_SOURCE": "constant",
        "RANDOM_SEED": seed,
        "CLOUD_MODEL": "correlated",
        "CLOUD_CORR_LENGTH_KM": 50.0,
        "CLOUD_CORR_TIME_H": 3.0,
        "MAX_SELECTION_STEPS": budget,
        "T_MAX_WAIT_MINUTES": orbit_period_min,
        "LATENCY_DECAY_RATE": decay_rate,
        "LATENCY_DECAY_SCALE": "fleet",
        "H1_REACHABLE_COVERAGE_PCT": 95.0,
        "TIME_AWARE_OBJECTIVE": time_objective,
        "CLOUD_KNOWLEDGE": "realistic",
        "CLOUD_PLANNING": "expected",
    }
    exec(compile(source, str(RELEASES[release]), "exec"), namespace)
    exec(functions_by_name(release, CELL_7C_TRIAL_FUNCTIONS), namespace)
    namespace["_OBJECTIVES"] = (("time_aware", float(decay_rate)), ("coverage_first", 0.0))
    namespace["_ptile"] = np.full(len(region_cells), float(cloud_probability))
    return namespace


def run_cell1_grid(
    release: str,
    active_roi: str | None = None,
    resolution: int = 7,
    boundary_dir: str | Path | None = None,
) -> dict:
    """Execute the region definitions and the discretisation of Cell 1 of a registered release.

    The definitions (``ROI_DEFINITIONS``, the WDPA readers, ``resolve_roi_coords``) are always executed. If
    ``active_roi`` is given, the statements of Cell 1 that discretise the active region are executed after them, as the
    notebook does. Returns the namespace: ``ROI_DEFINITIONS``, ``load_wdpa_site``, and, with ``active_roi``,
    ``target_tiles``, ``roi_boundary_report``, ``ROI_OUTLINE_GEOMS`` and ``ROI_OUTLINE_SOURCE``.
    """
    import typing

    import h3

    from arf.errors import ConfigurationError, SpatialDiscretisationError

    namespace = {
        "os": os,
        "h3": h3,
        "logger": logging.getLogger("registered"),
        "List": typing.List,  # noqa: UP006 (the registered source uses the typing names)
        "Tuple": typing.Tuple,  # noqa: UP006
        "Set": typing.Set,  # noqa: UP006
        "Dict": typing.Dict,  # noqa: UP006
        "ConfigurationError": ConfigurationError,
        "SpatialDiscretisationError": SpatialDiscretisationError,
        "ARF_BOUNDARY_DIR": str(boundary_dir) if boundary_dir is not None else None,
    }
    definitions = source_between(release, "def _buffer_waypoints_to_polygon", "# Resolve active ROI from Cell 0")
    exec(compile(definitions, str(RELEASES[release]), "exec"), namespace)
    if active_roi is not None:
        namespace["ACTIVE_ROI"] = active_roi
        namespace["H3_RESOLUTION"] = resolution
        discretisation = source_between(release, "# Resolve active ROI from Cell 0", "# 6. GROUND STATION NETWORK")
        exec(compile(discretisation, str(RELEASES[release]), "exec"), namespace)
    return namespace


#: Functions and constants of Cell 2 that choose the fleet, taken from the release by name.
CELL_2_FLEET_NAMES = (
    "TLE_CACHE_MAX_AGE_HOURS",
    "_parse_gp_history_epoch",
    "_dedupe_gp_history_to_latest_per_satellite",
    "omm_to_tle",
    "load_gp_json",
    "load_eo_gp_history",
    "_omm_orbit_km",
    "swath_from_fov_km",
    "_omm_epoch",
    "_match_mission",
    "build_eo_constellation",
    "_ALPHA5_LETTERS",
    "_alpha5_to_int",
    "_parse_norad_id",
    "OPERATIONAL_CRITERIA",
    "_elset_epoch_utc",
    "_elset_orbit_km",
    "select_operational_sample",
    "select_pinned_satellites",
    "_newest_elset_epoch_iso",
)


def definitions_by_name(release: str, names: tuple[str, ...]) -> str:
    """Return the source of the named top-level functions and assignments of a release, unchanged.

    Names the release does not define are left out (release 1.5.3 has no Alpha-5 decoder).
    """
    found = _top_level_definitions(release)
    return "\n\n".join(found[name] for name in names if name in found)


@functools.cache
def _top_level_definitions(release: str) -> dict[str, str]:
    """Return the source of every top-level function, class and single-name assignment of a release, first one kept."""
    import ast

    source = RELEASES[release].read_text()
    lines = source.splitlines(keepends=True)
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        else:
            continue
        found.setdefault(name, "".join(lines[node.lineno - 1 : node.end_lineno]))
    return found


class RecordedTle:
    """Stands in for Orekit's TLE: it keeps the arguments it was built with."""

    def __init__(self, *arguments):
        """Keep the arguments."""
        self.arguments = arguments


class _RecordedPropagator:
    @staticmethod
    def selectExtrapolator(tle):  # noqa: N802 (the name Orekit uses)
        return ("propagator", tle)


class _StubJavaError(Exception):
    """Stands in for jpype.JException; never raised by the stand-ins."""


def orekit_stand_ins():
    """Return a context manager that puts stand-ins for Orekit and JPype in ``sys.modules``.

    The fleet functions of Cell 2 import Orekit's TLE classes when called. Orekit needs a Java machine and data files
    that continuous integration does not have, and which satellites are chosen does not depend on it, so the imports
    are served by stand-ins that record what they are given: ``TLE`` keeps its arguments, ``AbsoluteDate`` its text.
    The construction of real Orekit propagators is checked with ``arf.access``.
    """
    import contextlib
    import sys
    import types

    @contextlib.contextmanager
    def stand_ins():
        modules = {}
        for name in ("org", "org.orekit", "org.orekit.propagation", "org.orekit.propagation.analytical"):
            modules[name] = types.ModuleType(name)
        tle_module = types.ModuleType("org.orekit.propagation.analytical.tle")
        tle_module.TLE, tle_module.TLEPropagator = RecordedTle, _RecordedPropagator
        time_module = types.ModuleType("org.orekit.time")
        time_module.AbsoluteDate = lambda text, scale: ("date", text, scale)
        time_module.TimeScalesFactory = types.SimpleNamespace(getUTC=lambda: "UTC")
        jpype_module = types.ModuleType("jpype")
        jpype_module.JChar, jpype_module.JException = (lambda character: character), _StubJavaError
        modules[tle_module.__name__], modules[time_module.__name__], modules["jpype"] = (
            tle_module,
            time_module,
            jpype_module,
        )
        saved = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)
        try:
            yield
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    return stand_ins()


class NetworkRefusedError(RuntimeError):
    """Raised when the registered code tries to download something: everything it reads must be given to it."""


def _refuse_network(*arguments, **keywords):
    raise NetworkRefusedError(
        "the registered release tried to download element sets; a cached file it expected is missing or older than "
        "its six-hour limit"
    )


#: Stands in for ``urllib`` in the namespace of the registered fleet functions, so that they can never download.
_NO_NETWORK = types.SimpleNamespace(
    request=types.SimpleNamespace(Request=_refuse_network, urlopen=_refuse_network),
    parse=types.SimpleNamespace(urlencode=_refuse_network),
)


def run_cell2_fleet_functions(
    release: str,
    planet_swath_case: str = "conservative",
    assumed_min_sun_elev_deg: float = 5.0,
    fleet_tag: str = "earth-resources_planet-conservative",
) -> dict:
    """Define the mission tables of Cell 0 and the fleet functions of Cell 2 of a registered release.

    Returns the namespace: ``MISSION_TABLE``, ``CELESTRAK_GROUPS``, ``ACQUISITION_EXCLUSIONS``,
    ``select_operational_sample``, ``select_pinned_satellites``, ``build_eo_constellation``, ``load_eo_gp_history``
    and the parsers. Call the functions that build TLEs inside :func:`orekit_stand_ins`. The functions cannot reach
    the network: a download raises ``RuntimeError``.
    """
    import json
    import re
    import time

    from arf.errors import ConfigurationError

    namespace = {
        "np": np,
        "os": os,
        "json": json,
        "time": time,
        "re": re,
        "urllib": _NO_NETWORK,
        "logger": logging.getLogger("registered"),
        "ConfigurationError": ConfigurationError,
        "PLANET_SWATH_CASE": planet_swath_case,
        "ASSUMED_MIN_SUN_ELEV_DEG": assumed_min_sun_elev_deg,
        "FLEET_TAG": fleet_tag,
        "TLEPropagator": _RecordedPropagator,
    }
    tables = source_between(release, "CELESTRAK_GROUPS = {", "H3_RESOLUTION = ")
    exec(compile(tables, str(RELEASES[release]), "exec"), namespace)
    functions = definitions_by_name(release, CELL_2_FLEET_NAMES)
    exec(compile(functions, str(RELEASES[release]), "exec"), namespace)
    return namespace


def run_cells_2_4_5(
    release: str,
    region_cells,
    stations: dict,
    reference_iso: str,
    cache_dir: str | Path,
    fleet: str = "earth-resources+planet",
    fleet_tag: str = "earth-resources_planet-conservative_from20260803",
    raw_records=None,
    active_n: int = 0,
    seed: int = 42,
    proxy_swath_km: float | None = None,
    proxy_min_sun_elev_deg: float | None = None,
    planet_swath_case: str = "conservative",
    assumed_min_sun_elev_deg: float = 5.0,
    exclude_satellites: tuple = (),
    window_hours: int = 48,
    swath_half_angle_deg: float = 9.22,
    min_elevation_deg: float = 15.0,
    legacy_end_caps: bool = False,
    out_dir: str | Path | None = None,
) -> dict:
    """Execute, with Orekit, the fleet construction of Cell 2 and the whole of Cells 4 and 5 of a registered release.

    The statements are run as the notebook runs them, from the choice of the fleet to the refined events. For a real
    fleet the CelesTrak group lists and the Space-Track history must already be in ``cache_dir`` (no network is used);
    for the dense constellation ``raw_records`` are the element sets. Orekit must have been started. With ``out_dir``,
    the statements that write the summaries and tables of the three cells are executed too, so the folder holds
    ``arf_cell2_summary.json``, ``arf_access_events.csv``, ``arf_cell4_summary.json``, ``arf_downlink_events.csv``,
    ``arf_refined_events.csv`` and ``arf_cell5_summary.json`` as a run writes them.

    Returns the namespace: ``constellation``, ``pos_list``, ``CONSTELLATION_ALTITUDE_KM``, ``T_MAX_WAIT_MINUTES``,
    ``sensor_config``, ``access_events_df``, ``strip_samples``, ``downlink_events_df``, ``refined_events_df``,
    ``refine_daylight_stats`` among others.
    """
    import dataclasses
    import json
    import time
    import typing

    import h3
    from org.hipparchus.util import FastMath
    from org.orekit.bodies import GeodeticPoint, OneAxisEllipsoid
    from org.orekit.frames import FramesFactory, TopocentricFrame
    from org.orekit.propagation.analytical.tle import TLE, TLEPropagator
    from org.orekit.utils import Constants, IERSConventions

    namespace = run_cell2_fleet_functions(release, planet_swath_case, assumed_min_sun_elev_deg, fleet_tag)
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, itrf)
    namespace.update(
        {
            "pd": pd,
            "h3": h3,
            "time": time,
            "json": json,
            "dataclass": dataclasses.dataclass,
            "Optional": typing.Optional,
            "Any": typing.Any,
            "List": typing.List,  # noqa: UP006 (the registered source uses the typing names)
            "Dict": typing.Dict,  # noqa: UP006
            "display": lambda table: None,
            "FastMath": FastMath,
            "GeodeticPoint": GeodeticPoint,
            "TopocentricFrame": TopocentricFrame,
            "Constants": Constants,
            "TLE": TLE,
            "TLEPropagator": TLEPropagator,
            "itrf": itrf,
            "earth": earth,
            "FLEET": fleet,
            "ANALYSIS_EPOCH": reference_iso,
            "ARF_CACHE_DIR": str(cache_dir),
            "TLE_RAW_CACHE_PATH": str(Path(cache_dir) / "raw.json"),
            "TLE_SOURCE": "spacetrack_history",
            "ACTIVE_ROI": "test_region",
            "ACTIVE_N": active_n,
            "RANDOM_SEED": seed,
            "raw_records": raw_records,
            "PROXY_MIN_SUN_ELEV_DEG": proxy_min_sun_elev_deg,
            "PROXY_SWATH_KM": proxy_swath_km,
            "EXCLUDE_SATELLITES": tuple(exclude_satellites),
            "target_tiles": set(region_cells),
            "SWATH_HALF_ANGLE_DEG": swath_half_angle_deg,
            "WINDOW_HOURS": window_hours,
            "GROUND_STATIONS": stations,
            "DOWNLINK_MIN_ELEVATION_DEG": min_elevation_deg,
            "LEGACY_STRIP_END_CAPS": legacy_end_caps,
            "OUT_DIR": str(out_dir) if out_dir is not None else os.getcwd(),
            "_out_dir": str(out_dir) if out_dir is not None else os.getcwd(),
        }
    )
    file_name = str(RELEASES[release])
    cell2 = source_between(
        release, 'if FLEET == "starlink":', "# SUMMARY & PERSISTENCE", after="def _newest_elset_epoch_iso"
    )
    exec(compile(cell2, file_name, "exec"), namespace)
    cell4 = source_between(release, "def _footprint_ground_radius_km", "import os, json")
    exec(compile(cell4, file_name, "exec"), namespace)
    cell5 = source_between(release, "class RefinementError", "_refined_csv = os.path.join")
    exec(compile(cell5, file_name, "exec"), namespace)
    if out_dir is not None:
        exports = (
            ("# SUMMARY & PERSISTENCE", "# CELL 3: SPATIAL", "def _newest_elset_epoch_iso"),
            ("import os, json", "# CELL 5: PHYSICAL", "def detect_access_events"),
            ("_refined_csv = os.path.join", "# CELL 6: SUBMODULAR", "def refine_access_events"),
        )
        for start, stop, after in exports:
            exec(compile(source_between(release, start, stop, after=after), file_name, "exec"), namespace)
    return namespace


#: Geometry helpers of Cells 4 and 5 that need no orbit propagation, taken from the release by name.
CELLS_4_5_GEOMETRY_NAMES = (
    "_footprint_ground_radius_km",
    "_KM_PER_DEG_LAT_MIN",
    "_KM_PER_DEG_LON_EQ",
    "_wrap_dlon",
    "_hav_km",
    "_GeoBox",
    "_visibility_radius_km",
    "downlink_reward",
    "_incidence_angle_deg_vec",
    "solar_elevation_deg",
    "_min_track_distance_km",
)


def run_cells_4_5_geometry(release: str) -> dict:
    """Define the geometry helpers of Cells 4 and 5 of a registered release; no Orekit is needed."""
    import types
    import typing

    namespace = {
        "np": np,
        "Re_km": 6378.137,
        "Optional": typing.Optional,
        "refine_config": types.SimpleNamespace(t_max_wait_minutes=95.0),
    }
    source = definitions_by_name(release, CELLS_4_5_GEOMETRY_NAMES)
    exec(compile(source, str(RELEASES[release]), "exec"), namespace)
    return namespace


#: The derived settings of the access scan, compared between release 2 and the registered ``sensor_config``.
SCAN_SETTING_NAMES = (
    "max_altitude_km",
    "min_altitude_km",
    "max_ground_speed_km_min",
    "footprint_radius_km",
    "coarse_margin_km",
    "strip_margin_km",
    "max_swath_km",
)


def _frame_difference(name: str, ours: pd.DataFrame, theirs: pd.DataFrame) -> str | None:
    """Describe how two tables differ, or return ``None`` if they are identical, dtypes and categories included."""
    try:
        pd.testing.assert_frame_equal(ours, theirs, check_exact=True)
    except AssertionError as error:
        first_line = str(error).strip().splitlines()[0] if str(error).strip() else "differ"
        return f"{name}: {len(ours)} rows in release 2, {len(theirs)} in the registered release; {first_line}"
    return None


def compare_with_registered(result, registered: dict) -> list[str]:
    """Return every way in which release 2's access result differs from the namespace of the registered cells.

    Compared exactly, to the last bit: the altitude of each satellite at the start of the window, their median, the
    orbital period, the derived scan settings, the access events and every strip sample, the downlink contacts, the
    refined events (as a table and as written to CSV) and the counts of what the refinement dropped. An empty list
    means identical.
    """
    differences = []
    pairs = {
        "altitudes": (result.altitudes_km, registered["pos_list"]),
        "median altitude": (result.median_altitude_km, registered["CONSTELLATION_ALTITUDE_KM"]),
        "orbital period": (result.orbit_period_min, registered["T_MAX_WAIT_MINUTES"]),
    }
    pairs.update(
        {
            f"scan setting {name}": (getattr(result.settings, name), getattr(registered["sensor_config"], name))
            for name in SCAN_SETTING_NAMES
        }
    )
    for name, (ours, theirs) in pairs.items():
        if ours != theirs:
            differences.append(f"{name}: release 2 {ours!r}, registered {theirs!r}")
    for name, ours, theirs in (
        ("access events", result.scan.events, registered["access_events_df"]),
        ("downlink contacts", result.contacts, registered["downlink_events_df"]),
        ("refined events", result.refinement.events, registered["refined_events_df"]),
    ):
        described = _frame_difference(name, ours, theirs)
        if described:
            differences.append(described)
    ours_samples, their_samples = result.scan.strip_samples, registered["strip_samples"]
    if ours_samples.keys() != their_samples.keys():
        differences.append(
            f"strip samples: {len(ours_samples.keys() - their_samples.keys())} strips only in release 2, "
            f"{len(their_samples.keys() - ours_samples.keys())} only in the registered release"
        )
    else:
        differing = sum(
            not all(np.array_equal(a, b) for a, b in zip(ours_samples[key], their_samples[key], strict=True))
            for key in ours_samples
        )
        if differing:
            differences.append(f"strip samples: {differing} of {len(ours_samples)} strips differ")
    if result.refinement.counts != registered["refine_daylight_stats"]:
        differences.append(
            f"refinement counts: release 2 {result.refinement.counts}, registered {registered['refine_daylight_stats']}"
        )
    written = io.StringIO()
    result.refinement.events.to_csv(written, index=False)
    if written.getvalue() != registered["refined_events_df"].to_csv(index=False):
        differences.append("refined events as written to CSV differ")
    return differences


def _cell7c_namespace(
    release: str,
    refined_events: pd.DataFrame,
    region_cells,
    out_dir,
    *,
    budget: int = 30,
    orbit_period_min: float = 94.0,
    median_altitude_km: float = 500.0,
    decay_rate: float = 3.0,
    decay_scale: str = "fleet",
    cloud_probability: float = 0.35,
    seed: int = 42,
    cloud_model: str = "correlated",
    length_km: float = 50.0,
    time_h: float = 3.0,
    window_hours: int = 48,
    shift_hours: int = 24,
    shift_step_min: int = 20,
    coverage_targets_pct: tuple = (95.0, 99.0),
    reachable_level_pct: float = 95.0,
    min_onset_pct: float = 90.0,
    floor_margin_min: float | None = None,
    min_evaluable_onsets: int | None = None,
    random_draws: int = 10,
    time_objective: str = "facility",
    cloud_knowledge: str = "realistic",
    cloud_planning: str = "expected",
    swath_half_angle_deg: float = 20.0,
    extra: dict | None = None,
) -> dict:
    """Execute Cell 6 up to its first selection and Cell 7C's helpers, unchanged; return the namespace.

    The settings are those of Cell 0, with the names of its configuration; ``extra`` adds further globals of Cell 0.
    """
    import json
    import math
    import sys

    import h3

    namespace = {
        "np": np,
        "pd": pd,
        "h3": h3,
        "os": os,
        "sys": sys,
        "math": math,
        "json": json,
        "random": random,
        "heapq": heapq,
        "logger": logging.getLogger("registered"),
        "_tile_ids": np.array(sorted(region_cells)),
        "target_tiles": list(region_cells),
        "refined_events_df": refined_events,
        "SWATH_HALF_ANGLE_DEG": swath_half_angle_deg,  # used only for refined events without theta_max_deg
        "CLOUD_PROBABILITY": cloud_probability,
        "CLOUD_PROBABILITY_SOURCE": "constant",
        "RANDOM_SEED": seed,
        "CLOUD_MODEL": cloud_model,
        "CLOUD_CORR_LENGTH_KM": length_km,
        "CLOUD_CORR_TIME_H": time_h,
        "MAX_SELECTION_STEPS": budget,
        "T_MAX_WAIT_MINUTES": orbit_period_min,
        "LATENCY_DECAY_RATE": decay_rate,
        "LATENCY_DECAY_SCALE": decay_scale,
        "TIME_AWARE_OBJECTIVE": time_objective,
        "CLOUD_KNOWLEDGE": cloud_knowledge,
        "CLOUD_PLANNING": cloud_planning,
        "Re_km": 6378.137,
        "mu_earth": 398600.4418,
        "CONSTELLATION_ALTITUDE_KM": median_altitude_km,
        "WINDOW_HOURS": window_hours,
        "ONSET_ROBUSTNESS_ENABLED": True,
        "ONSET_SHIFT_HOURS": shift_hours,
        "ONSET_SHIFT_STEP_MIN": shift_step_min,
        "COVERAGE_TARGETS_PCT": tuple(coverage_targets_pct),
        "H1_REACHABLE_COVERAGE_PCT": reachable_level_pct,
        "H1_MIN_ONSET_PCT": min_onset_pct,
        "H1_FLOOR_MARGIN_MIN_CONFIG": floor_margin_min,
        "H1_MIN_EVALUABLE_ONSETS_CONFIG": min_evaluable_onsets,
        "RANDOM_BASELINE_DRAWS": random_draws,
        "OUT_DIR": str(out_dir),
        "_out_dir": str(out_dir),
        **(extra or {}),
    }
    cell6 = source_between(release, "class EventStore:", 'logger.info(f"=== Running clear-sky selection')
    exec(compile(cell6, str(RELEASES[release]), "exec"), namespace)
    exec(functions_by_name(release, ("compute_completion_latency",)), namespace)
    helpers = source_between(release, "_H1_TARGET_MIN = 10.0", "_sweep_rates = [0.0]")
    exec(compile(helpers, str(RELEASES[release]), "exec"), namespace)
    return namespace


def run_cell7c_onsets(release: str, refined_events: pd.DataFrame, region_cells, out_dir, **settings) -> dict:
    """Execute Cell 6 up to its first selection, and the onset comparison of Cell 7C, of a registered release.

    Cell 6 builds the event store, the cloud mask and the cloud probability per cell from the refined events; Cell 7C
    then defines its helpers and runs section 2 (onset robustness), which writes ``arf_onset_robustness.csv`` and
    ``arf_onset_robustness_summary.json`` to ``out_dir``. All unchanged. The settings are those of
    ``_cell7c_namespace``. Returns the namespace: ``onset_df`` and ``onset_robustness_summary`` among others.
    """
    namespace = _cell7c_namespace(release, refined_events, region_cells, out_dir, **settings)
    onsets = source_between(release, "onset_robustness_summary = {}", "    # Figure 5: empirical CDF")
    exec(compile(onsets, str(RELEASES[release]), "exec"), namespace)
    return namespace


def run_cell7c_cloud(
    release: str,
    refined_events: pd.DataFrame,
    region_cells,
    out_dir,
    *,
    cloud_seeds: int = 20,
    sweep_lengths_km: tuple = (0.0, 10.0, 25.0, 50.0, 100.0, 200.0),
    sweep_times_h: tuple = (1.0, 3.0, 6.0, 24.0),
    sweep_seeds: int = 10,
    **settings,
) -> dict:
    """Execute Cell 6 up to its first selection, and the cloud robustness of Cell 7C, of a registered release.

    After Cell 7C's helpers, sections 4 and 5 run unchanged up to Figure 8: the realisations, H2 and the sensitivity
    sweep, written to ``arf_cloud_robustness.csv``, ``arf_cloud_sensitivity.csv`` and
    ``arf_cloud_robustness_summary.json`` in ``out_dir``. Sections 1 to 3 do not run: section 4 reads none of what
    they define. Returns the namespace: ``cloud_seed_df``, ``cloud_robustness_summary`` and
    ``cloud_sensitivity_rows`` among others.
    """
    extra = {
        "CLOUD_SEEDS": cloud_seeds,
        "CLOUD_SWEEP_LENGTHS_KM": tuple(sweep_lengths_km),
        "CLOUD_SWEEP_TIMES_H": tuple(sweep_times_h),
        "CLOUD_SWEEP_SEEDS": sweep_seeds,
    }
    namespace = _cell7c_namespace(release, refined_events, region_cells, out_dir, extra=extra, **settings)
    cloud = source_between(release, "cloud_seed_rows, cloud_robustness_summary = [], {}", "    # Figure 8 (IEEE")
    exec(compile(cloud, str(RELEASES[release]), "exec"), namespace)
    return namespace
