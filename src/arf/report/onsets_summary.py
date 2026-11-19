"""What the onset table shows (``arf_onset_robustness_summary.json``; Cell 7C, section 2, of the registered releases).

For the level ``L`` (a share of the cells reachable in each onset's horizon), the protocol's H1 is tested on two
counts, each as the share of onsets at which it holds:

* **H1a, latency**: the selector reaches ``L`` no later than the margin after the floor.
* **H1b, image economy**: at onsets where first-come-first-served (FCFS, no budget) needs two or more images to reach
  ``L``, the earliest-completion cover (ECC) reaches it within the margin of the floor with fewer images than FCFS.

A share is evaluable only over at least ``min_evaluable_onsets`` onsets, and is reported with its exact one-sided
lower confidence bound (Clopper and Pearson, 1934, Biometrika 26(4), 404-413, doi:10.1093/biomet/26.4.404). The
onsets of one window share most of their passes, so they are not independent trials: within a run the share
describes the window, and the bound overstates the evidence.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from arf.fleet.elements import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from arf.report.onsets import POLICIES

#: The original H1 threshold (completion within 10 minutes), still summarised; superseded by H1a and H1b.
H1_TARGET_MIN = 10.0
#: The confidence level of the bound is ``1 - CONFIDENCE_ALPHA``.
CONFIDENCE_ALPHA = 0.05
_TOLERANCE = 1e-9
_CLOPPER_PEARSON = "Clopper, C. J. & Pearson, E. S. (1934). Biometrika 26(4), 404-413, doi:10.1093/biomet/26.4.404"
_DESCRIPTIVE = (
    "descriptive for this window: consecutive onsets share passes, so the Clopper-Pearson bound, which assumes "
    "independent trials, overstates the evidence; test across independent windows"
)


def clopper_pearson_lower(successes: int, trials: int, alpha: float = CONFIDENCE_ALPHA) -> float | None:
    """Return the exact one-sided lower confidence bound of a share, as a fraction; ``None`` without trials.

    It is the ``alpha`` quantile of Beta(``successes``, ``trials - successes + 1``); 0 without successes.
    """
    if not trials:
        return None
    if successes <= 0:
        return 0.0
    from scipy.stats import beta

    return float(beta.ppf(alpha, successes, trials - successes + 1))


def min_onsets_for(min_onset_pct: float, alpha: float = CONFIDENCE_ALPHA) -> int | None:
    """Return the fewest onsets at which success at all of them gives a lower bound of at least ``min_onset_pct``.

    That is the smallest ``n`` with ``alpha ** (1 / n) >= p``; ``None`` for 100%, which no finite sample reaches.
    """
    share = float(min_onset_pct) / 100.0
    if share >= 1.0:
        return None
    return math.ceil(math.log(alpha) / math.log(share) - 1e-12)


def orbital_period_min(median_altitude_km: float) -> float:
    """Return the period of a circular orbit at the fleet's median altitude, in minutes (Kepler's third law)."""
    return float(2.0 * np.pi * np.sqrt((EARTH_RADIUS_KM + median_altitude_km) ** 3 / EARTH_MU_KM3_S2) / 60.0)


def _series_stats(table: pd.DataFrame, column: str) -> dict:
    """Median, 90th percentile and largest time of a series, and how often it is reached and under 10 minutes."""
    values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
    reached = ~np.isnan(values)
    return {
        "median_min": float(np.median(values[reached])) if reached.any() else None,
        "p90_min": float(np.percentile(values[reached], 90)) if reached.any() else None,
        "max_min": float(np.max(values[reached])) if reached.any() else None,
        # Onsets where the series never completes count as not meeting the target.
        "pct_onsets_under_h1": float(100.0 * np.mean(reached & (np.nan_to_num(values, nan=np.inf) < H1_TARGET_MIN))),
        "n_onsets": len(values),
        "n_completed": int(reached.sum()),
        "pct_onsets_reached": float(100.0 * reached.mean()) if len(values) else 0.0,
    }


def _number(row: pd.Series, column: str) -> float | None:
    value = row.get(column)
    return None if value is None or pd.isna(value) else float(value)


def _h1a(table: pd.DataFrame, name: str, sky: str, level_tag: str, margin_min: float, min_evaluable: int) -> dict:
    """H1a for one selector and sky: the share of evaluable onsets at which it reaches the level within the margin."""
    within = evaluated = 0
    gaps = []
    for _, row in table.iterrows():
        if not row.get(f"reachable_{sky}_tiles"):
            continue
        floor = row.get(f"best_possible_{sky}_t{level_tag}_min")
        if floor is None or pd.isna(floor):
            continue
        evaluated += 1
        value = row.get(f"{name}_{sky}_t{level_tag}_min")
        if value is not None and not pd.isna(value):
            gaps.append(float(value) - float(floor))
            within += int(float(value) - float(floor) <= margin_min + _TOLERANCE)
    bound = clopper_pearson_lower(within, evaluated)
    return {
        "n_onsets_evaluated": evaluated,
        "n_within_margin": within,
        "pct_within_margin": (100.0 * within / evaluated) if evaluated else None,
        "pct_within_margin_lcb95": None if bound is None else 100.0 * bound,
        "evaluable": evaluated >= min_evaluable,
        "median_gap_min": float(np.median(gaps)) if gaps else None,
        "p90_gap_min": float(np.percentile(gaps, 90)) if gaps else None,
        "pct_reached": (100.0 * len(gaps) / evaluated) if evaluated else None,
    }


def _h1b(table: pd.DataFrame, sky: str, level_tag: str, margin_min: float, min_evaluable: int) -> dict:
    """H1b for one sky: the share of evaluable onsets at which the ECC reaches the level with fewer images than FCFS."""
    evaluated = successes = one_image = 0
    ecc_images, fcfs_images, ratios = [], [], []
    for _, row in table.iterrows():
        floor = _number(row, f"best_possible_{sky}_t{level_tag}_min")
        fcfs = _number(row, f"fcfs_unbudgeted_{sky}_images{level_tag}")
        if not row.get(f"reachable_{sky}_tiles") or floor is None or fcfs is None:
            continue
        if fcfs < 2:
            one_image += 1  # one image: no room for economy
            continue
        evaluated += 1
        ecc_time = _number(row, f"earliest_completion_{sky}_t{level_tag}_min")
        ecc = _number(row, f"earliest_completion_{sky}_images{level_tag}")
        fcfs_images.append(fcfs)
        if ecc_time is not None and ecc is not None:
            ecc_images.append(ecc)
            ratios.append(ecc / fcfs)
            successes += int(ecc_time - floor <= margin_min + _TOLERANCE and ecc < fcfs)
    bound = clopper_pearson_lower(successes, evaluated)
    return {
        "n_onsets_evaluated": evaluated,
        "n_onsets_one_image": one_image,
        "n_success": successes,
        "pct_success": (100.0 * successes / evaluated) if evaluated else None,
        "pct_success_lcb95": None if bound is None else 100.0 * bound,
        "evaluable": evaluated >= min_evaluable,
        "median_images_ecc": float(np.median(ecc_images)) if ecc_images else None,
        "median_images_fcfs": float(np.median(fcfs_images)) if fcfs_images else None,
        "median_image_ratio_ecc_to_fcfs": float(np.median(ratios)) if ratios else None,
        "max_images_ecc": float(np.max(ecc_images)) if ecc_images else None,
    }


def _quality(table: pd.DataFrame, sky: str, level_tag: str) -> dict:
    """Mean utility of each selector's mosaic when it first reaches the level, and its difference from FCFS."""
    out = {}
    for name in ("time_aware", "coverage_first", "fcfs_unbudgeted", "earliest_completion"):
        quality = pd.to_numeric(table[f"{name}_{sky}_quality{level_tag}"], errors="coerce").dropna()
        out[name] = {"median": float(quality.median()) if len(quality) else None, "n_onsets": len(quality)}
    for name in ("time_aware", "earliest_completion"):
        ours = pd.to_numeric(table[f"{name}_{sky}_quality{level_tag}"], errors="coerce")
        fcfs = pd.to_numeric(table[f"fcfs_unbudgeted_{sky}_quality{level_tag}"], errors="coerce")
        both = ours.notna() & fcfs.notna() & (fcfs > 0)
        out[f"{name}_vs_fcfs"] = {
            "n_onsets": int(both.sum()),
            "median_difference": float((ours[both] - fcfs[both]).median()) if both.any() else None,
            "median_relative_difference_pct": (
                float((100.0 * (ours[both] - fcfs[both]) / fcfs[both]).median()) if both.any() else None
            ),
            "pct_onsets_higher": float(100.0 * (ours[both] > fcfs[both] + 1e-12).mean()) if both.any() else None,
        }
    return out


def onset_summary(
    table: pd.DataFrame,
    *,
    median_altitude_km: float,
    budget: int = 30,
    decay_rate: float = 3.0,
    decay_scale: str = "fleet",
    window_hours: int = 48,
    shift_hours: int = 24,
    shift_step_min: int = 20,
    coverage_targets_pct: tuple[float, ...] = (95.0, 99.0),
    reachable_level_pct: float = 95.0,
    min_onset_pct: float = 90.0,
    floor_margin_min: float | None = None,
    min_evaluable_onsets: int | None = None,
) -> dict:
    """Return ``arf_onset_robustness_summary.json`` for an onset table (see :func:`arf.report.onsets.onset_table`).

    Args:
        table: The onset table.
        median_altitude_km: The fleet's median altitude at the start of the window; the H1 margin is one orbital
            period there unless ``floor_margin_min`` is given.
        budget: ``max_selection_steps``.
        decay_rate: ``latency_decay_rate``.
        decay_scale: ``latency_decay_scale``.
        window_hours: ``window_hours``.
        shift_hours: ``onset_shift_hours``.
        shift_step_min: ``onset_shift_step_min``.
        coverage_targets_pct: ``coverage_targets_pct``.
        reachable_level_pct: ``h1_reachable_coverage_pct``, the level ``L``.
        min_onset_pct: ``h1_min_onset_pct``, the share of onsets H1 requires.
        floor_margin_min: ``h1_floor_margin_min``; ``None`` derives it.
        min_evaluable_onsets: ``h1_min_evaluable_onsets``; ``None`` derives it from ``min_onset_pct``.
    """
    level, level_tag = float(reachable_level_pct), f"{float(reachable_level_pct):g}r"
    min_share = float(min_onset_pct)
    if floor_margin_min is None:
        margin = orbital_period_min(median_altitude_km)
        margin_source = f"derived: one orbital period at the fleet median altitude {median_altitude_km:.1f} km"
    else:
        margin, margin_source = float(floor_margin_min), "set in Cell 0"
    if min_evaluable_onsets is None:
        min_evaluable = min_onsets_for(min_share) or 1
        evaluability_source = (
            f"derived: smallest n with {CONFIDENCE_ALPHA:g}**(1/n) >= {min_share / 100:g} "
            f"(all-pass Clopper-Pearson one-sided {100 * (1 - CONFIDENCE_ALPHA):g}% lower bound)"
        )
    else:
        min_evaluable, evaluability_source = int(min_evaluable_onsets), "set in Cell 0"
    taus = pd.to_numeric(table["decay_tau_min"], errors="coerce").dropna()
    summary = {
        "h1_target_min": H1_TARGET_MIN,
        "latency_decay_rate": decay_rate,
        "latency_decay_scale": decay_scale,
        "decay_tau_min_over_onsets": (
            {"median": float(taus.median()), "min": float(taus.min()), "max": float(taus.max())} if len(taus) else None
        ),
        "onset_shift_hours": shift_hours,
        "onset_shift_step_min": shift_step_min,
        "horizon_hours": (window_hours - shift_hours) * 60 / 60,
        "series": {
            f"{name}_{sky}": _series_stats(table, f"{name}_{sky}_min")
            for sky in ("clear", "cloud")
            for name in ("best_possible", *POLICIES)
        },
        "coverage_targets_pct": list(coverage_targets_pct),
        "equal_coverage_series": {
            f"{name}_{sky}_t{target:g}": _series_stats(table, f"{name}_{sky}_t{target:g}_min")
            for target in coverage_targets_pct
            for sky in ("clear", "cloud")
            for name in ("best_possible", *POLICIES)
        },
    }
    evaluability = {
        "min_evaluable_onsets": min_evaluable,
        "source": evaluability_source,
        "confidence_alpha": CONFIDENCE_ALPHA,
        "reference": _CLOPPER_PEARSON,
        "caveat": "consecutive onsets share most of their passes, so they are not independent trials: the lower "
        "bound is optimistic and the minimum count necessary, not sufficient",
    }
    summary["h1_revised"] = {
        "framing": (
            "H1a: latency non-inferiority to the physical floor. Pre-specified 2026-10-09 for the time-aware "
            "selector; from v1.5.0 tested on the earliest-completion cover (ECC), the selector H1b also tests, with "
            "the time-aware selector reported (series time_aware_*)."
        ),
        "primary_series": "earliest_completion",
        "inference": _DESCRIPTIVE,
        "statement": (
            f"At >= {min_share:g}% of onsets the earliest-completion cover reaches {level:g}% of reachable tiles "
            f"within {margin:.1f} min of the physical floor, using at most {budget} images (clear sky)."
        ),
        "reachable_coverage_pct": level,
        "floor_margin_min": margin,
        "floor_margin_source": margin_source,
        "min_onset_pct": min_share,
        "budget_k": budget,
        **evaluability,
        "series": {
            f"{name}_{sky}": _h1a(table, name, sky, level_tag, margin, min_evaluable)
            for sky in ("clear", "cloud")
            for name in ("time_aware", "coverage_first", "fcfs", "earliest_completion")
        },
    }
    summary["h1b_image_economy"] = {
        "framing": "H1b: image economy (added 2026-10-12, after the 15 runs of that day; confirmatory on fresh data "
        "only)",
        "inference": _DESCRIPTIVE,
        "statement": (
            f"At >= {min_share:g}% of onsets where first-come-first-served needs two or more images to reach "
            f"{level:g}% of reachable tiles, the earliest-completion cover reaches that level within {margin:.1f} min "
            f"of the physical floor using fewer images than first-come-first-served, with at most {budget} images "
            f"(clear sky)."
        ),
        "reachable_coverage_pct": level,
        "floor_margin_min": margin,
        "min_onset_pct": min_share,
        "budget_k": budget,
        **evaluability,
        "series": {sky: _h1b(table, sky, level_tag, margin, min_evaluable) for sky in ("clear", "cloud")},
    }
    summary["quality_at_level"] = {
        "note": (
            f"Mean utility of the tiles covered when each policy's delivered mosaic first reaches {level:g}% of "
            f"reachable tiles; first-come-first-served without a budget. Reported, not tested."
        ),
        "series": {sky: _quality(table, sky, level_tag) for sky in ("clear", "cloud")},
    }
    return summary
