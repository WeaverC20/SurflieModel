"""
Set Duration Statistic

Calculates the mean duration of a wave set using Rice envelope crossing theory.

The set duration is the mean time the envelope spends above the threshold
per crossing. From Rice (1945), the fraction of time above threshold alpha*Hs
is P = exp(-2*alpha^2) (Rayleigh). Combined with the set period T_set:

    tau_set = T_set * exp(-2*alpha^2) = 1 / (2*alpha * sqrt(2*pi) * sigma_f)

The exp(-2*alpha^2) cancels between numerator and denominator, giving a
result that depends only on alpha and sigma_f.

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
class SetDurationStatistic(StatisticFunction):
    """
    Duration of a typical wave set via direct Rice formula.

    tau_set = 1 / (2*alpha * sqrt(2*pi) * sigma_f)

    This is the mean time the wave envelope exceeds alpha*Hs per set cycle.
    Self-consistent with set_frequency.py: tau_set + tau_lull = T_set exactly.
    """

    @property
    def name(self) -> str:
        return "set_duration"

    @property
    def units(self) -> str:
        return "s"

    @property
    def description(self) -> str:
        return "Duration of a typical wave set (Rice envelope theory)"

    def compute_vectorized(
        self,
        partitions: List[WavePartition],
        depths: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        **kwargs,
    ) -> StatisticOutput:
        n_points = len(depths)
        moments = reconstruct_spectral_moments(partitions, n_points)

        # tau_set = 1 / (2*alpha * sqrt(2*pi) * sigma_f)
        denominator = 2.0 * ALPHA_DEFAULT * np.sqrt(2.0 * np.pi) * moments.sigma_f
        set_duration = np.where(denominator > 0, 1.0 / denominator, np.nan)

        return StatisticOutput(
            name=self.name,
            values=set_duration,
            units=self.units,
            description=self.description,
        )
