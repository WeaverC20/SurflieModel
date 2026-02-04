"""
Set Frequency Statistic

Calculates the beat period between dominant swell pairs, which determines
how often sets arrive at a surf spot.
"""

from typing import List

import numpy as np

from data.surfzone.runner.swan_input_provider import WavePartition
from .base import StatisticFunction, StatisticOutput
from .registry import StatisticsRegistry


@StatisticsRegistry.register
class SetFrequencyStatistic(StatisticFunction):
    """
    Time between wave sets (beat period).

    Sets form from constructive interference between wave trains of
    different frequencies. The beat period is: T_set = 1 / |f1 - f2|

    For multiple swells, we use the two most energetic swells at each point.
    """

    @property
    def name(self) -> str:
        return "set_period"

    @property
    def units(self) -> str:
        return "s"

    @property
    def description(self) -> str:
        return "Time between wave sets (beat period from dominant swell pair)"

    def compute_vectorized(
        self,
        partitions: List[WavePartition],
        depths: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> StatisticOutput:
        n_points = len(depths)

        # Get swells only (partition_id > 0, i.e., not wind sea)
        swells = [p for p in partitions if p.partition_id > 0]

        if len(swells) < 2:
            # Not enough swells for beat frequency
            return StatisticOutput(
                name=self.name,
                values=np.full(n_points, np.inf),
                units=self.units,
                description=self.description
            )

        # Stack energies: shape (n_swells, n_points)
        # Use Hs^2 as proxy for energy
        energies = np.array([p.hs**2 for p in swells])

        # Handle invalid values (replace with 0 energy)
        for i, p in enumerate(swells):
            energies[i] = np.where(p.is_valid, energies[i], 0)

        # Get indices of top 2 by energy at each point
        sorted_idx = np.argsort(energies, axis=0)
        idx1 = sorted_idx[-1, :]  # Most energetic
        idx2 = sorted_idx[-2, :]  # Second most

        # Get Tp for each using advanced indexing
        tp_stack = np.array([p.tp for p in swells])
        tp1 = np.take_along_axis(tp_stack, idx1[np.newaxis, :], axis=0)[0]
        tp2 = np.take_along_axis(tp_stack, idx2[np.newaxis, :], axis=0)[0]

        # Calculate beat period
        f1, f2 = 1 / tp1, 1 / tp2
        delta_f = np.abs(f1 - f2)

        # Avoid division by near-zero (similar periods = very long set period)
        delta_f = np.where(delta_f < 0.001, np.nan, delta_f)
        set_period = 1 / delta_f

        # Cap at reasonable maximum (10 minutes = 600s)
        set_period = np.clip(set_period, 0, 600)

        return StatisticOutput(
            name=self.name,
            values=set_period,
            units=self.units,
            description=self.description
        )
