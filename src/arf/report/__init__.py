"""The tables and summaries a run writes, and what follows from them (H1a, H1b and H2).

The selectors at every onset (``onsets``), the plans over many cloud realisations (``cloud_robustness``), the
objectives and selectors over the full window (``policies``), and the gap to the exact optimum (``optimality``).
"""

from arf.report.cloud_robustness import (
    CloudModel,
    CloudRobustnessSettings,
    cloud_robustness_summary,
    cloud_robustness_table,
    sensitivity_rows,
)
from arf.report.onsets import OnsetSettings, onset_table
from arf.report.onsets_summary import onset_summary
from arf.report.optimality import IlpSettings, ilp_gap_summary, ilp_gap_table
from arf.report.policies import cost_latency_frontier, decay_sweep, policy_comparison

__all__ = [
    "CloudModel",
    "CloudRobustnessSettings",
    "IlpSettings",
    "OnsetSettings",
    "cloud_robustness_summary",
    "cloud_robustness_table",
    "cost_latency_frontier",
    "decay_sweep",
    "ilp_gap_summary",
    "ilp_gap_table",
    "onset_summary",
    "onset_table",
    "policy_comparison",
    "sensitivity_rows",
]
