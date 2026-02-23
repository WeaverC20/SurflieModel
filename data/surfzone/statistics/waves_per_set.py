"""
Waves Per Set Statistic

Estimates the mean number of waves in a set using Longuet-Higgins (1984) group
length theory with phasor-based spectral autocorrelation.

The spectral correlation gamma is computed as the magnitude of the complex
autocorrelation of the reconstructed spectrum at lag T_01 (mean period).
This properly accounts for within-partition bandwidth via the Gaussian damping
factor, preventing the gamma -> 1 artifact that occurred with the old method.

Longuet-Higgins' formula j = pi / (2 * arccos(gamma)) gives more physically
realistic group lengths than Kimura's (1+gamma)/(1-gamma), which overestimates
for high correlation.

References
----------
Longuet-Higgins, M.S. (1984). Statistical properties of wave groups in a random sea.
    Phil. Trans. R. Soc. London A312.
Kimura, A. (1980). Statistical properties of random wave groups. Proc. ICCE.
"""

from typing import List

import numpy as np

from data.surfzone.runner.swan_input_provider import WavePartition
from .base import StatisticFunction, StatisticOutput
from .registry import StatisticsRegistry
from .spectral_utils import reconstruct_spectral_moments, phasor_correlation


@StatisticsRegistry.register
class WavesPerSetStatistic(StatisticFunction):
    """
    Mean number of waves in a set using Longuet-Higgins group length theory.

    j = pi / (2 * arccos(gamma))

    where gamma is the phasor-based spectral correlation at one-period lag.

    Typical values:
    - Single peaked swell (gamma ~ 0.9): ~4-5 waves per set
    - Two similar swells (gamma ~ 0.8): ~3-4 waves per set
    - Mixed sea state (gamma ~ 0.5): ~2 waves per set
    """

    @property
    def name(self) -> str:
        return "waves_per_set"

    @property
    def units(self) -> str:
        return "count"

    @property
    def description(self) -> str:
        return "Mean number of waves in a set (Longuet-Higgins theory)"

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
        gamma = phasor_correlation(moments)

        # Longuet-Higgins (1984): j = pi / (2 * arccos(gamma))
        # arccos is defined for gamma in [0, 1]; gamma is clipped to [0, 0.999]
        waves_per_set = np.where(
            moments.valid,
            np.pi / (2.0 * np.arccos(gamma)),
            np.nan,
        )

        return StatisticOutput(
            name=self.name,
            values=waves_per_set,
            units=self.units,
            description=self.description,
            extra={"gamma": gamma},
        )
