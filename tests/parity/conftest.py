"""Report which Orekit data the parity tests ran with, in pytest's header."""

from orekit_context import orekit_data_in_use


def pytest_report_header(config):
    """Add a line naming the Orekit data to the header of the test report."""
    return f"Orekit data for the parity tests: {orekit_data_in_use()}"
