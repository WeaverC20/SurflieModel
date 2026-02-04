"""
Set Duration Statistic

Calculates the duration of a typical wave set in seconds.
"""

from typing import List

import numpy as np

from data.surfzone.runner.swan_input_provider import WavePartition
from .base import StatisticFunction, StatisticOutput
from .registry import StatisticsRegistry
from .waves_per_set import _spectral_correlation


@StatisticsRegistry.register
class SetDurationStatistic(StatisticFunction):
    """
    Duration of a typical wave set.

    Set duration = waves_per_set × mean_period

    This gives the approximate time from the first large wave
    in a set to the last.
    """

    @property
    def name(self) -> str:
        return "set_duration"

    @property
    def units(self) -> str:
        return "s"

    @property
    def description(self) -> str:
        return "Duration of a typical wave set"

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

        # Calculate spectral correlation and waves per set
        gamma = _spectral_correlation(partitions, valid_mask)
        waves_per_set = (1 + gamma) / (1 - gamma)
        waves_per_set = np.clip(waves_per_set, 1, 20)

        # Calculate energy-weighted mean period
        total_E = np.zeros(n_points)
        weighted_T = np.zeros(n_points)

        for p in partitions:
            energy = np.where(p.is_valid, p.hs**2, 0)
            total_E += energy
            weighted_T += energy * p.tp

        total_E = np.where(total_E > 0, total_E, 1)
        T_mean = weighted_T / total_E

        # Set duration = waves per set × mean period
        set_duration = waves_per_set * T_mean

        return StatisticOutput(
            name=self.name,
            values=set_duration,
            units=self.units,
            description=self.description
        )
