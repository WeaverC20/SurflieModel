"""
Waves Per Set Statistic

Uses Kimura's (1980) theory to estimate the mean number of waves in a set
based on the spectral correlation coefficient.
"""

from typing import List

import numpy as np

from data.surfzone.runner.swan_input_provider import WavePartition
from .base import StatisticFunction, StatisticOutput
from .registry import StatisticsRegistry


def _spectral_correlation(
    partitions: List[WavePartition],
    valid_mask: np.ndarray
) -> np.ndarray:
    """
    Estimate spectral correlation coefficient from partition parameters.

    For narrow-band spectra: gamma ≈ exp(-2 * (pi * sigma_f * T_mean)^2)
    where sigma_f is the spectral width estimated from partition spread.

    Args:
        partitions: List of WavePartition objects
        valid_mask: Boolean array of shape (n_points,)

    Returns:
        Correlation coefficient array of shape (n_points,)
    """
    n_points = len(valid_mask)

    # Calculate energy-weighted mean period
    total_E = np.zeros(n_points)
    weighted_T = np.zeros(n_points)

    for p in partitions:
        energy = np.where(p.is_valid, p.hs**2, 0)
        total_E += energy
        weighted_T += energy * p.tp

    # Avoid division by zero
    total_E = np.where(total_E > 0, total_E, 1)
    T_mean = weighted_T / total_E

    # Estimate spectral width from partition frequency spread
    # Calculate variance of frequencies weighted by energy
    f_mean = 1 / T_mean
    freq_variance = np.zeros(n_points)
    total_weight = np.zeros(n_points)

    for p in partitions:
        weight = np.where(p.is_valid, p.hs**2, 0)
        freq = 1 / np.where(p.tp > 0, p.tp, 1)
        freq_variance += weight * (freq - f_mean)**2
        total_weight += weight

    # Standard deviation of frequencies
    total_weight = np.where(total_weight > 0, total_weight, 1)
    sigma_f = np.sqrt(freq_variance / total_weight)

    # Correlation coefficient (narrow-band approximation)
    gamma = np.exp(-2 * (np.pi * sigma_f * T_mean)**2)

    # Clip to valid range [0, 1)
    gamma = np.clip(gamma, 0, 0.99)

    return gamma


@StatisticsRegistry.register
class WavesPerSetStatistic(StatisticFunction):
    """
    Mean number of waves in a set using Kimura's theory.

    Based on Kimura (1980) "Statistical properties of random wave groups".
    The mean group length is: j = (1 + gamma) / (1 - gamma)
    where gamma is the spectral correlation coefficient.

    Typical values:
    - Single peaked swell (gamma ≈ 0.7): ~5-6 waves per set
    - Mixed swell (gamma ≈ 0.4): ~2-3 waves per set
    - Very mixed (gamma ≈ 0.2): ~1-2 waves per set
    """

    @property
    def name(self) -> str:
        return "waves_per_set"

    @property
    def units(self) -> str:
        return "count"

    @property
    def description(self) -> str:
        return "Mean number of waves in a set (Kimura theory)"

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

        # Kimura's mean run length formula: j = (1 + gamma) / (1 - gamma)
        waves_per_set = (1 + gamma) / (1 - gamma)

        # Clip to reasonable range (1 to 20 waves per set)
        waves_per_set = np.clip(waves_per_set, 1, 20)

        return StatisticOutput(
            name=self.name,
            values=waves_per_set,
            units=self.units,
            description=self.description,
            extra={"gamma": gamma}  # Include correlation for reference
        )
