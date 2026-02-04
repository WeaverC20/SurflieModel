"""
Groupiness Factor Statistic

Measures the degree of wave grouping based on spectral correlation.
"""

from typing import List

import numpy as np

from data.surfzone.runner.swan_input_provider import WavePartition
from .base import StatisticFunction, StatisticOutput
from .registry import StatisticsRegistry
from .waves_per_set import _spectral_correlation


@StatisticsRegistry.register
class GroupinessStatistic(StatisticFunction):
    """
    Groupiness factor (GF).

    Measures the degree of wave grouping based on envelope variance.
    Higher values indicate more defined wave groups (sets).

    GF is approximated from the spectral correlation coefficient:
    GF ≈ 0.5 + 0.7 * gamma

    Typical values:
    - 0.5: Wind chop, poorly defined groups
    - 0.8-1.0: Moderate grouping
    - 1.2-1.5: Clean ground swell, well-defined sets
    """

    @property
    def name(self) -> str:
        return "groupiness_factor"

    @property
    def units(self) -> str:
        return "ratio"

    @property
    def description(self) -> str:
        return "Degree of wave grouping (0.5=choppy, 1.5=clean swell)"

    def compute_vectorized(
        self,
        partitions: List[WavePartition],
        depths: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> StatisticOutput:
        n_points = len(depths)

        # Create combined validity mask
        valid_mask = np.zeros(n_points, dtype=bool)
        for p in partitions:
            valid_mask |= p.is_valid

        # Calculate spectral correlation
        gamma = _spectral_correlation(partitions, valid_mask)

        # Groupiness factor approximation
        # Higher correlation = more defined groups = higher GF
        GF = 0.5 + 0.7 * gamma

        return StatisticOutput(
            name=self.name,
            values=GF,
            units=self.units,
            description=self.description
        )
