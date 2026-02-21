"""
Lull Duration Statistic

Calculates the mean time between wave sets using Rice envelope crossing theory.

The lull duration is the complement of set duration within one set cycle:

    tau_lull = T_set - tau_set
             = (exp(2*alpha^2) - 1) / (2*alpha * sqrt(2*pi) * sigma_f)

Self-consistent with set_frequency.py and set_duration.py by construction:
    tau_set + tau_lull = T_set (exact).

References
----------
Rice, S.O. (1945). Mathematical analysis of random noise. Bell System Tech. J. 24.
"""

from typing import List

import numpy as np

from data.surfzone.runner.swan_input_provider import WavePartition
from .base import StatisticFunction, StatisticOutput
from .registry import StatisticsRegistry
from .spectral_utils import reconstruct_spectral_moments

ALPHA_DEFAULT = 1.0


@StatisticsRegistry.register
class LullDurationStatistic(StatisticFunction):
    """
    Duration of the lull between wave sets via Rice envelope theory.

    tau_lull = (exp(2*alpha^2) - 1) / (2*alpha * sqrt(2*pi) * sigma_f)

    This is the mean time the wave envelope stays below alpha*Hs between sets.
    """

    @property
    def name(self) -> str:
        return "lull_duration"

    @property
    def units(self) -> str:
        return "s"

    @property
    def description(self) -> str:
        return "Time between wave sets (Rice envelope theory)"

    def compute_vectorized(
        self,
        partitions: List[WavePartition],
        depths: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> StatisticOutput:
        n_points = len(depths)
        moments = reconstruct_spectral_moments(partitions, n_points)

        # tau_lull = (exp(2*alpha^2) - 1) / (2*alpha * sqrt(2*pi) * sigma_f)
        denominator = 2.0 * ALPHA_DEFAULT * np.sqrt(2.0 * np.pi) * moments.sigma_f
        lull_duration = np.where(
            denominator > 0,
            (np.exp(2.0 * ALPHA_DEFAULT ** 2) - 1.0) / denominator,
            np.nan,
        )

        return StatisticOutput(
            name=self.name,
            values=lull_duration,
            units=self.units,
            description=self.description,
        )
