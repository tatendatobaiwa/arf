"""The Fail-Signal selector over many cloud realisations, and hypothesis H2 (Cell 7C, sections 4 and 5).

One cloud realisation is one random draw, so a comparison of plans under cloud needs many. For ``realisations``
realisations of the run's cloud model (seeds ``seed``, ``seed + 1``, ...), over the full window, each objective
(time-aware, the primary, and coverage-first) is run with the plans of :func:`arf.comparison.cloud_trial`, all of the
size of the adaptive selector's: the Fail-Signal selector, the same planner run in advance with no cloud observation
(open-loop, expected coverage), an open-loop plan on clear-sky geometry, and the greedy with the cloud known; and,
for the record, the selector of release 1.4.0 with an open-loop plan of its own size. Each realisation also records
the floor, the earliest-completion cover (ECC) and first-come-first-served (FCFS) under that cloud, at the level
``reachable_level_pct`` of the cells it leaves reachable. One row per realisation: ``arf_cloud_robustness.csv``.

**H2**: with the same number of images, the Fail-Signal selector covers more of the region than the open-loop plan
(expected coverage): the mean paired difference over the realisations has a one-sided 95% lower confidence bound
above zero (Student t). It is not evaluable when the selector never re-planned or compensated, when even the greedy
that knows the cloud never beats the open-loop plan, or when the cloud is known in advance.

**Sensitivity** (section 5): the H2 difference of the time-aware objective at other correlation lengths (the
independent model at length zero) and times, one varied at a time, over ``sweep_realisations`` realisations each:
``arf_cloud_sensitivity.csv``. The summary of both is ``arf_cloud_robustness_summary.json``.

Coverage is the share of all region cells with a usable, cloud-free observation among the plan's images.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from arf.cloud import LatentCloudField, correlated_cloud_mask, independent_cloud_mask
from arf.comparison import cloud_trial
from arf.events import EventTable
from arf.grid import cell_centres
from arf.report.onsets import coverage_times, earliest_completion, level_stats, select_fcfs

#: The four plans H2 compares, under the names of ``arf_cloud_robustness.csv``.
PLANS = ("adaptive", "openloop_expected", "openloop_clear", "oracle")
#: Plan names of ``arf_cloud_robustness.csv``, in its column order, and the same plans in ``arf.comparison``.
TABLE_PLANS = {
    "adaptive": "adaptive",
    "openloop_expected": "open_loop_expected",
    "openloop_clear": "open_loop_clear",
    "oracle": "oracle",
    "failsignal_v140": "adaptive_clear_planning",
    "openloop_expected_v140n": "open_loop_expected_at_that_size",
}
#: Guards comparisons of coverage against rounding, as in the registered releases.
_TOLERANCE = 1e-9

# The texts the registered releases write into the summary, unchanged.
_DESIGN = (
    "Per cloud realisation, four plans with the same number of images: the Fail-Signal adaptive "
    "selector; the same expected-coverage planner run once in advance with no cloud observation "
    "(open-loop); an open-loop plan on clear-sky geometry; and the greedy given the cloud in advance. Also "
    "the v1.4.0 Fail-Signal (clear-sky planning). Coverage = tiles with a usable cloud-free "
    "observation, % of ROI tiles."
)
_STATEMENT = (
    "H2 (v1.5.0): with the same number of images, the Fail-Signal adaptive selector covers more of "
    "the region under cloud than the same planner run in advance without seeing any image (open-"
    "loop, maximising expected coverage from the tiles' cloud probabilities): the mean paired "
    "difference over cloud realisations has a one-sided 95% lower confidence bound above zero."
)
_CLOUD_SIZES = (
    "Wood, R. & Field, P. R. (2011). The distribution of cloud horizontal sizes. "
    "J. Climate 24, 4800-4816, doi:10.1175/2011JCLI4056.1 (scale range); "
    "the length and time scales are assumptions, swept in Cell 7C."
)


@dataclass(frozen=True)
class CloudRobustnessSettings:
    """The settings of sections 4 and 5 of Cell 7C, with the names of the registered releases' configuration.

    Attributes:
        orbit_period_min: The orbital period at the fleet's median altitude (``T_MAX_WAIT_MINUTES`` after Cell 2).
        cloud_probability: ``cloud_probability``, the same for every region cell (the constant source).
        budget: ``max_selection_steps``.
        decay_rate: ``latency_decay_rate``, the rate of the time-aware objective.
        decay_scale: ``latency_decay_scale``.
        time_objective: ``time_aware_objective``.
        seed: ``random_seed``; realisation ``i`` has seed ``seed + i``.
        realisations: ``cloud_seeds``.
        cloud_model: ``cloud_model``: ``"correlated"`` or ``"iid"`` (independent).
        length_km: ``cloud_corr_length_km``.
        time_h: ``cloud_corr_time_h``.
        sweep_lengths_km: ``cloud_sweep_lengths_km``; zero is the independent model.
        sweep_times_h: ``cloud_sweep_times_h``.
        sweep_realisations: ``cloud_sweep_seeds``.
        reachable_level_pct: ``h1_reachable_coverage_pct``.
        cloud_knowledge: ``cloud_knowledge``. Only ``"realistic"`` is ported.
        cloud_planning: ``cloud_planning`` of the adaptive selector.
    """

    orbit_period_min: float
    cloud_probability: float = 0.35
    budget: int = 30
    decay_rate: float = 3.0
    decay_scale: str = "fleet"
    time_objective: str = "facility"
    seed: int = 42
    realisations: int = 20
    cloud_model: str = "correlated"
    length_km: float = 50.0
    time_h: float = 3.0
    sweep_lengths_km: tuple[float, ...] = (0.0, 10.0, 25.0, 50.0, 100.0, 200.0)
    sweep_times_h: tuple[float, ...] = (1.0, 3.0, 6.0, 24.0)
    sweep_realisations: int = 10
    reachable_level_pct: float = 95.0
    cloud_knowledge: str = "realistic"
    cloud_planning: str = "expected"

    def __post_init__(self) -> None:
        """Refuse the settings that are not ported."""
        if self.cloud_knowledge != "realistic":
            raise ValueError(
                f"cloud_knowledge {self.cloud_knowledge!r} is not ported: with the cloud known in advance the "
                "registered releases record the fail-signals of an earlier selection"
            )
        if self.cloud_model not in ("correlated", "iid"):
            raise ValueError(f"cloud model {self.cloud_model!r} is not one of the registered releases'")

    @property
    def objectives(self) -> tuple[tuple[str, float], ...]:
        """Name and decay rate of each objective, the primary first."""
        return (("time_aware", float(self.decay_rate)), ("coverage_first", 0.0))

    @property
    def level_tag(self) -> str:
        """The level as it appears in column names, ``95`` for 95.0."""
        return f"{float(self.reachable_level_pct):g}"


class CloudModel:
    """Draws the realisations of the cloud model for one set of observations; keeps one latent field per length."""

    def __init__(self, events: EventTable, settings: CloudRobustnessSettings) -> None:
        """Hold the observations and the settings; fields are laid out when first needed."""
        self.events = events
        self.settings = settings
        self._fields: dict[float, LatentCloudField] = {}

    def field(self, length_km: float) -> LatentCloudField:
        """The latent field of the region at a correlation length."""
        length_km = float(length_km)
        if length_km not in self._fields:
            latitude, longitude = cell_centres(self.events.region_cells)
            self._fields[length_km] = LatentCloudField(latitude, longitude, length_km)
        return self._fields[length_km]

    def draw(self, seed: int, model: str | None = None, length_km: float | None = None, time_h: float | None = None):
        """One realisation: the cloudy observations, as Cell 6's ``build_cloud_mask`` draws them."""
        model = model or self.settings.cloud_model
        probability = float(self.settings.cloud_probability)
        if model == "iid":
            return independent_cloud_mask(self.events, probability, seed)
        length_km = float(self.settings.length_km if length_km is None else length_km)
        time_h = float(self.settings.time_h if time_h is None else time_h)
        return correlated_cloud_mask(self.events, probability, seed, self.field(length_km), time_h)

    def record(self) -> dict:
        """The cloud model as the summary records it (``cloud_model_info`` of Cell 6)."""
        probability = float(self.settings.cloud_probability)
        record: dict = {"model": self.settings.cloud_model, "probability": {"source": "constant", "value": probability}}
        if self.settings.cloud_model == "correlated":
            field = self.field(self.settings.length_km)
            record.update(
                length_km=float(self.settings.length_km),
                time_h=float(self.settings.time_h),
                grid_spacing_km=float(field.spacing_km),
                grid_cells=int(field.n_columns * field.n_rows),
                reference=_CLOUD_SIZES,
            )
        return record


# --- one realisation ---------------------------------------------------------------------------------------------


def realisation_row(
    events: EventTable,
    is_cloudy: np.ndarray,
    settings: CloudRobustnessSettings,
    objectives: tuple[tuple[str, float], ...] | None = None,
    with_h1: bool = True,
) -> dict:
    """One realisation as a row of ``arf_cloud_robustness.csv``, without its seed.

    For each objective, the images and coverage of every plan (as a share of all region cells, and of the cells
    reachable under this cloud), the fail-signals and the compensatory images; with ``with_h1``, the floor, the ECC
    and FCFS (no budget) at the level of the reachable cells: time, and images.
    """
    objectives = settings.objectives if objectives is None else objectives
    n_cells = len(events.region_cells)
    trial = cloud_trial(
        events,
        is_cloudy,
        np.full(n_cells, float(settings.cloud_probability)),
        int(settings.budget),
        settings.orbit_period_min,
        objectives,
        settings.time_objective,
        settings.decay_scale,
        planning=settings.cloud_planning,
        level_percent=settings.reachable_level_pct,
    )
    reach = trial.reachable_cells
    row: dict = {"cloudy_rows_pct": trial.cloudy_share_percent, "reachable_tiles": reach}
    for name, _ in objectives:
        result = trial.objectives[name]
        for tag, plan in TABLE_PLANS.items():
            covered = result.plans[plan].covered_cells
            row[f"{name}_{tag}_images"] = result.plans[plan].images
            row[f"{name}_{tag}_cov_pct"] = 100.0 * covered / n_cells
            row[f"{name}_{tag}_reach_pct"] = (100.0 * covered / reach) if reach else None
        row[f"{name}_replans"] = int(result.replans)
        row[f"{name}_compensatory"] = int(result.compensatory)
    if with_h1:
        level, tag = float(settings.reachable_level_pct), settings.level_tag
        if reach:
            need = int(np.ceil(level / 100.0 * reach - _TOLERANCE))
            floor = coverage_times(events, is_cloudy, (level,), denominator=reach)[level]
            ecc_time, _, ecc_images = earliest_completion(events, is_cloudy, int(settings.budget), level, reach)
            fcfs = select_fcfs(events, is_cloudy, len(events))
            fcfs_time, fcfs_images, _ = level_stats(events, is_cloudy, fcfs, need)
        else:
            floor, ecc_time, ecc_images, fcfs_time, fcfs_images = None, None, 0, None, None
        row.update(
            {
                f"floor_t{tag}r_min": floor,
                f"ecc_t{tag}r_min": ecc_time,
                f"ecc_images{tag}r": ecc_images if ecc_time is not None else None,
                f"fcfs_t{tag}r_min": fcfs_time,
                f"fcfs_images{tag}r": fcfs_images,
            }
        )
    return row


def cloud_robustness_table(
    events: EventTable, settings: CloudRobustnessSettings, model: CloudModel | None = None
) -> pd.DataFrame | None:
    """Return ``arf_cloud_robustness.csv``: one row per realisation; ``None`` if the releases would not compute it."""
    if settings.realisations < 2 or len(events) == 0:
        return None
    model = model or CloudModel(events, settings)
    rows = []
    for i in range(settings.realisations):
        seed = settings.seed + i
        rows.append({"seed": seed, **realisation_row(events, model.draw(seed), settings)})
    return pd.DataFrame(rows)


# --- the summary -------------------------------------------------------------------------------------------------


def paired_summary(differences, alpha: float = 0.05) -> dict:
    """Summarise paired differences: their mean and its one-sided lower confidence bound.

    The bound is at level ``1 - alpha``, Student t with ``n - 1`` degrees of freedom; also the standard deviation and
    the shares of positive and negative pairs. Missing and infinite differences are left out.
    """
    d = np.asarray([x for x in differences if x is not None and np.isfinite(x)], dtype=np.float64)
    n = int(d.size)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "lcb": None, "pct_positive": None, "pct_negative": None}
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    lcb = mean - float(student_t.ppf(1.0 - alpha, n - 1)) * sd / math.sqrt(n) if n > 1 else None
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "lcb": lcb,
        "alpha": alpha,
        "pct_positive": float(100.0 * (d > _TOLERANCE).mean()),
        "pct_negative": float(100.0 * (d < -_TOLERANCE).mean()),
    }


def _column(table: pd.DataFrame, name: str) -> np.ndarray:
    """A column as floats, missing values as NaN; all NaN if the table has no such column."""
    if name in table.columns:
        return pd.to_numeric(table[name], errors="coerce").to_numpy(dtype=float)
    return np.full(len(table), np.nan)


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(np.nanmean(values)) if np.isfinite(values).any() else None


def h2_summary(table: pd.DataFrame, objective: str, settings: CloudRobustnessSettings) -> dict:
    """H2 for one objective over the realisations of a table."""
    adaptive, expected, clear, oracle = (_column(table, f"{objective}_{plan}_cov_pct") for plan in PLANS)
    replans, compensatory = _column(table, f"{objective}_replans"), _column(table, f"{objective}_compensatory")
    adapted = int(np.nansum(replans) + np.nansum(compensatory))
    room = bool(np.any(oracle > expected + _TOLERANCE))
    versus_expected, versus_clear = paired_summary(adaptive - expected), paired_summary(adaptive - clear)
    captured = [
        (a - x) / (o - x)
        for a, x, o in zip(adaptive, expected, oracle, strict=True)
        if np.isfinite(o - x) and o - x > _TOLERANCE
    ]
    reasons = []
    if adapted == 0:
        reasons.append("the selector never re-planned or compensated in any realisation (nothing to adapt)")
    if not room:
        reasons.append(
            "even the greedy that knows the cloud never beats the open-loop plan with the same images "
            "(no room to adapt)"
        )
    evaluable = not reasons and versus_expected["n"] >= 2
    if not evaluable:
        status = "NOT EVALUABLE"
    elif versus_expected["lcb"] is not None and versus_expected["lcb"] > 0:
        status = "SUPPORTED"
    else:
        status = "NOT SUPPORTED"
    earlier = _column(table, f"{objective}_failsignal_v140_cov_pct")
    earlier_open_loop = _column(table, f"{objective}_openloop_expected_v140n_cov_pct")
    return {
        "status": status,
        "evaluable": evaluable,
        "not_evaluable_reasons": reasons,
        "n_realisations": len(adaptive),
        "planning": settings.cloud_planning,
        "coverage_pct_mean": {
            plan: _mean_or_none(v) for plan, v in zip(PLANS, (adaptive, expected, clear, oracle), strict=True)
        },
        "images_mean": float(np.nanmean(_column(table, f"{objective}_adaptive_images"))) if len(table) else None,
        "difference_vs_openloop_expected": versus_expected,
        "difference_vs_openloop_clear": versus_clear,
        "v140_failsignal": (
            {
                "coverage_pct_mean": float(np.nanmean(earlier)),
                "images_mean": float(np.nanmean(_column(table, f"{objective}_failsignal_v140_images"))),
                "difference_vs_openloop_expected_same_images": paired_summary(earlier - earlier_open_loop),
            }
            if np.isfinite(earlier).any()
            else None
        ),
        "share_of_oracle_gain_captured_median": float(np.median(captured)) if captured else None,
        "replans_total": int(np.nansum(replans)),
        "compensatory_total": int(np.nansum(compensatory)),
    }


def h1_under_cloud(table: pd.DataFrame, settings: CloudRobustnessSettings, floor_margin_min: float) -> dict:
    """Summarise H1a and H1b over the realisations (secondary).

    H1a: the ECC within the margin of the floor. H1b: where FCFS needs two or more images, the ECC within the margin
    with fewer images than FCFS.
    """
    tag = settings.level_tag
    floor, ecc_time = _column(table, f"floor_t{tag}r_min"), _column(table, f"ecc_t{tag}r_min")
    ecc_images, fcfs_images = _column(table, f"ecc_images{tag}r"), _column(table, f"fcfs_images{tag}r")
    evaluated = np.isfinite(floor)
    within = evaluated & np.isfinite(ecc_time) & (ecc_time - floor <= floor_margin_min + _TOLERANCE)
    eligible = evaluated & np.isfinite(fcfs_images) & (fcfs_images >= 2)
    economy = eligible & within & np.isfinite(ecc_images) & (ecc_images < fcfs_images)
    with_ecc = eligible & np.isfinite(ecc_images)
    return {
        "n_realisations": int(evaluated.sum()),
        "pct_ecc_within_margin": float(100.0 * within.sum() / evaluated.sum()) if evaluated.sum() else None,
        "n_realisations_fcfs_two_plus_images": int(eligible.sum()),
        "pct_ecc_fewer_images_within_margin": float(100.0 * economy.sum() / eligible.sum()) if eligible.sum() else None,
        "median_images_ecc": float(np.median(ecc_images[with_ecc])) if with_ecc.any() else None,
        "median_images_fcfs": float(np.median(fcfs_images[eligible])) if eligible.any() else None,
    }


# --- section 5: sensitivity to the cloud model -------------------------------------------------------------------


def sensitivity_settings(settings: CloudRobustnessSettings) -> list[tuple[float, float | None]]:
    """Return the (length, time) pairs of the sweep.

    Each length at the run's time, then each other time at the run's length; length zero is the independent model,
    with no time.
    """
    time_h = float(settings.time_h)
    pairs = [(float(length), None if float(length) == 0.0 else time_h) for length in settings.sweep_lengths_km]
    pairs += [
        (float(settings.length_km), float(time_h))
        for time_h in settings.sweep_times_h
        if float(time_h) != float(settings.time_h)
    ]
    return list(dict.fromkeys(pairs))


def sensitivity_rows(
    events: EventTable, settings: CloudRobustnessSettings, model: CloudModel | None = None
) -> list[dict] | None:
    """Return the rows of ``arf_cloud_sensitivity.csv``; ``None`` if the sweep is empty or too short."""
    pairs = sensitivity_settings(settings)
    if not pairs or settings.sweep_realisations < 2:
        return None
    model = model or CloudModel(events, settings)
    primary_objective = settings.objectives[:1]
    rows = []
    for length_km, time_h in pairs:
        trials = []
        for i in range(settings.sweep_realisations):
            if length_km == 0.0:
                is_cloudy = model.draw(settings.seed + i, model="iid")
            else:
                is_cloudy = model.draw(settings.seed + i, "correlated", length_km, time_h)
            trials.append(realisation_row(events, is_cloudy, settings, primary_objective, with_h1=False))
        table = pd.DataFrame(trials)
        adaptive, expected, clear, oracle = (_column(table, f"time_aware_{plan}_cov_pct") for plan in PLANS)
        difference = paired_summary(adaptive - expected)
        rows.append(
            {
                "model": "iid" if length_km == 0.0 else "correlated",
                "length_km": length_km,
                "time_h": time_h,
                "primary": (
                    settings.cloud_model == "correlated"
                    and length_km == float(settings.length_km)
                    and time_h == float(settings.time_h)
                ),
                "n_realisations": len(table),
                "images_mean": float(np.nanmean(_column(table, "time_aware_adaptive_images"))),
                "adaptive_cov_pct": float(np.nanmean(adaptive)),
                "openloop_expected_cov_pct": float(np.nanmean(expected)),
                "openloop_clear_cov_pct": float(np.nanmean(clear)),
                "oracle_cov_pct": float(np.nanmean(oracle)),
                "diff_mean": difference["mean"],
                "diff_lcb95": difference["lcb"],
                "diff_pct_positive": difference["pct_positive"],
                "replans_total": int(np.nansum(_column(table, "time_aware_replans"))),
                "compensatory_total": int(np.nansum(_column(table, "time_aware_compensatory"))),
            }
        )
    return rows


def cloud_robustness_summary(
    table: pd.DataFrame,
    settings: CloudRobustnessSettings,
    floor_margin_min: float,
    cloud_model: dict,
    sensitivity: list[dict] | None = None,
) -> dict:
    """Return ``arf_cloud_robustness_summary.json``.

    Args:
        table: ``arf_cloud_robustness.csv`` (:func:`cloud_robustness_table`).
        settings: The settings it was computed with.
        floor_margin_min: The H1 margin: one orbital period at the fleet's median altitude unless set
            (``h1_floor_margin_min``); see :func:`arf.report.onsets_summary.orbital_period_min`.
        cloud_model: The cloud model's record (:meth:`CloudModel.record`).
        sensitivity: The rows of :func:`sensitivity_rows`, if the sweep ran.
    """
    summary = {
        "design": _DESIGN,
        "statement": _STATEMENT,
        "cloud_planning": settings.cloud_planning,
        "primary_objective": "time_aware",
        "cloud_model": cloud_model,
        "seeds": [int(settings.seed), int(settings.seed + settings.realisations - 1)],
        "h2": {name: h2_summary(table, name, settings) for name, _ in settings.objectives},
        "h1_under_cloud": h1_under_cloud(table, settings, floor_margin_min),
    }
    if sensitivity:
        summary["sensitivity"] = sensitivity
    return summary


def summary_json(summary: dict) -> str:
    """The summary as the registered releases write it."""
    import json

    return json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
