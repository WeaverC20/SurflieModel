"""
Surfzone Wave Statistics Module

Provides pluggable statistics for analyzing swell partitions at mesh points.
Statistics are computed vectorized across all points for efficiency.

Example usage:
    from data.surfzone.statistics import StatisticsRunner, StatisticsRegistry

    # Run all registered statistics
    runner = StatisticsRunner()
    result = runner.run_from_boundary(boundary, depths, region="socal")
    result.save("data/surfzone/output/socal/")

    # Or run specific statistics
    runner = StatisticsRunner(statistics=["set_period", "waves_per_set"])
"""

from .base import StatisticFunction, StatisticOutput
from .registry import StatisticsRegistry
from .runner import StatisticsRunner, StatisticsResult

# Import all statistic implementations to register them
from . import set_frequency
from . import waves_per_set
from . import peakiness
from . import height_amplification
from . import groupiness
from . import set_duration
from . import lull_duration
from . import breaking

__all__ = [
    "StatisticFunction",
    "StatisticOutput",
    "StatisticsRegistry",
    "StatisticsRunner",
    "StatisticsResult",
]
