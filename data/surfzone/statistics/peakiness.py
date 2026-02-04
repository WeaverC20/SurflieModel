"""
Wave Peakiness / Wavelength Statistic

Calculates wavelength and wave steepness for each partition.
Steepness (H/L) indicates wave shape - from flat/rolling to steep/peaky.
"""

from typing import List

import numpy as np

from data.surfzone.runner.swan_input_provider import WavePartition
from .base import StatisticFunction, StatisticOutput
from .registry import StatisticsRegistry


def wavelength_at_depth(L_deep: np.ndarray, depth: np.ndarray, iterations: int = 10) -> np.ndarray:
    """
    Calculate wavelength at a given depth using iterative dispersion relation.

    The deep water wavelength is: L_deep = g * T^2 / (2 * pi)
    At finite depth: L = L_deep * tanh(2 * pi * d / L)

    Args:
        L_deep: Deep water wavelength, shape (n_points,)
        depth: Water depth (positive = below sea level), shape (n_points,)
        iterations: Number of iterations for convergence

    Returns:
        Wavelength at depth, shape (n_points,)
    """
    # Ensure depth is positive (below sea level)
    # Points above sea level (negative depth) get deep water wavelength
    safe_depth = np.maximum(depth, 0.1)  # Minimum 0.1m to avoid issues

    L = L_deep.copy()
    for _ in range(iterations):
        # Handle very shallow water
        kd = 2 * np.pi * safe_depth / np.where(L > 0, L, 1)
        L = L_deep * np.tanh(kd)

    # For very shallow or land (depth <= 0), use NaN
    L = np.where(depth > 0, L, np.nan)

    return L


def categorize_steepness(steepness: np.ndarray) -> np.ndarray:
    """
    Categorize wave steepness into surfing-relevant categories.

    Categories:
    - 0: flat_rolling (< 0.02) - Very smooth, long-period waves
    - 1: moderate (0.02-0.04) - Ideal for most surfing
    - 2: steep_peaky (0.04-0.067) - Hollow, barrel potential
    - 3: very_steep (> 0.067) - Near breaking threshold

    Args:
        steepness: H/L ratio, shape (n_points,)

    Returns:
        Category codes (0-3), shape (n_points,)
    """
    cats = np.zeros_like(steepness, dtype=int)
    cats = np.where(steepness >= 0.02, 1, cats)  # moderate
    cats = np.where(steepness >= 0.04, 2, cats)  # steep_peaky
    cats = np.where(steepness >= 0.067, 3, cats)  # very_steep
    return cats


@StatisticsRegistry.register
class PeakinessStatistic(StatisticFunction):
    """
    Wavelength and steepness per partition.

    Calculates wavelength at local depth using the dispersion relation,
    then computes steepness (H/L) as an indicator of wave shape.

    Output columns:
    - wavelength_0, wavelength_1, wavelength_2, wavelength_3: Wavelength per partition (m)
    - steepness_0, steepness_1, steepness_2, steepness_3: Steepness ratio per partition
    """

    @property
    def name(self) -> str:
        return "peakiness"

    @property
    def units(self) -> str:
        return "m"

    @property
    def description(self) -> str:
        return "Wavelength and steepness at local depth for each partition"

    @property
    def output_columns(self) -> List[str]:
        return [
            "wavelength_0", "wavelength_1", "wavelength_2", "wavelength_3",
            "steepness_0", "steepness_1", "steepness_2", "steepness_3"
        ]

    def compute_vectorized(
        self,
        partitions: List[WavePartition],
        depths: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> StatisticOutput:
        g = 9.81
        n_points = len(depths)
        n_parts = len(partitions)

        wavelengths = np.zeros((n_parts, n_points))
        steepness = np.zeros((n_parts, n_points))

        for i, p in enumerate(partitions):
            # Deep water wavelength: L = g * T^2 / (2 * pi)
            L_deep = g * p.tp**2 / (2 * np.pi)

            # Adjust for depth
            L = wavelength_at_depth(L_deep, depths)

            # Handle invalid partitions
            L = np.where(p.is_valid, L, np.nan)

            wavelengths[i] = L
            steepness[i] = np.where(L > 0, p.hs / L, 0)

        # Stack into (n_points, 2*n_parts) array: wavelengths then steepness
        values = np.column_stack([wavelengths.T, steepness.T])

        # Determine actual columns used
        cols = self.output_columns[:n_parts] + self.output_columns[4:4+n_parts]

        return StatisticOutput(
            name=self.name,
            values=values,
            units=self.units,
            description=self.description,
            extra={'columns': cols}
        )
