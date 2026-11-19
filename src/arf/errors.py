"""Errors raised by the package."""

from __future__ import annotations


class ConfigurationError(Exception):
    """A setting or an input file is missing, or is not the one the run was told to use."""


class SpatialDiscretisationError(Exception):
    """A region could not be turned into H3 cells, or its cells do not match its boundary."""
