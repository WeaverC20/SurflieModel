"""
Lull Duration Statistic

Calculates the time between sets (the "lull" period).
"""

from typing import List

import numpy as np

from data.surfzone.runner.swan_input_provider import WavePartition
from .base import StatisticFunction, StatisticOutput
from .registry import StatisticsRegistry
from .waves_per_set import _spectral_correlation


@StatisticsRegistry.register
class LullDurationStatistic(StatisticFunction):
    """
    Duration of the lull between wave sets.

    Lull duration = set_period - set_duration

    This is the approximate waiting time between the end of one set
    and the start of the next.
    """

    @property
    def name(self) -> str:
        return "lull_duration"

    @property
    def units(self) -> str:
        return "s"

    @property
    def description(self) -> str:
        return "Time between wave sets (lull period)"

    def compute_vectorized(
        self,
        partitions: List[WavePartition],
        depths: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> StatisticOutput:
        n_points = len(depths)

        # Get swells only for set period calculation
        swells = [p for p in partitions if p.partition_id > 0]

        # Calculate set period (beat period)
        if len(swells) >= 2:
            energies = np.array([
                np.where(p.is_valid, p.hs**2, 0) for p in swells
            ])
            sorted_idx = np.argsort(energies, axis=0)
            idx1 = sorted_idx[-1, :]
            idx2 = sorted_idx[-2, :]

            tp_stack = np.array([p.tp for p in swells])
            tp1 = np.take_along_axis(tp_stack, idx1[np.newaxis, :], axis=0)[0]
            tp2 = np.take_along_axis(tp_stack, idx2[np.newaxis, :], axis=0)[0]

            f1, f2 = 1 / tp1, 1 / tp2
            delta_f = np.abs(f1 - f2)
            delta_f = np.where(delta_f < 0.001, np.nan, delta_f)
            set_period = np.clip(1 / delta_f, 0, 600)
        else:
            set_period = np.full(n_points, np.inf)

        # Calculate set duration
        valid_mask = np.zeros(n_points, dtype=bool)
        for p in partitions:
            valid_mask |= p.is_valid

        gamma = _spectral_correlation(partitions, valid_mask)
        waves_per_set = np.clip((1 + gamma) / (1 - gamma), 1, 20)

        total_E = np.zeros(n_points)
        weighted_T = np.zeros(n_points)
        for p in partitions:
            energy = np.where(p.is_valid, p.hs**2, 0)
            total_E += energy
            weighted_T += energy * p.tp
        total_E = np.where(total_E > 0, total_E, 1)
        T_mean = weighted_T / total_E

        set_duration = waves_per_set * T_mean

        # Lull = set period - set duration
        lull_duration = np.maximum(0, set_period - set_duration)

        return StatisticOutput(
            name=self.name,
            values=lull_duration,
            units=self.units,
            description=self.description
        )
