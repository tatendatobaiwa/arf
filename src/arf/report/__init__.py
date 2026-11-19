"""The tables and summaries a run writes, and what follows from them (H1a, H1b and H2).

The selectors at every onset (``onsets``) and the plans over many cloud realisations (``cloud_robustness``).
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

__all__ = [
    "CloudModel",
    "CloudRobustnessSettings",
    "OnsetSettings",
    "cloud_robustness_summary",
    "cloud_robustness_table",
    "onset_summary",
    "onset_table",
    "sensitivity_rows",
]
