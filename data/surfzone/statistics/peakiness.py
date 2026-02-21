"""
Wave Peakiness / Wavelength Statistic

Calculates wavelength, wave steepness, combined steepness, and Ursell number.

Per-partition: wavelength at local depth (dispersion relation) and H/L steepness.
Combined: overall steepness Hs/L_02 using the zero-crossing period T_02 from
spectral moments. Ursell number Ur = Hs * L^2 / d^3 classifies wave nonlinearity:
  Ur < 10: linear waves
  10-26: weakly nonlinear
  26-75: cnoidal regime
  > 75: bore-like

References
----------
Ursell, F. (1953). The long-wave paradox in the theory of gravity waves.
    Proc. Camb. Phil. Soc. 49.
"""

from typing import List

import numpy as np

from data.surfzone.runner.swan_input_provider import WavePartition
from .base import StatisticFunction, StatisticOutput
from .registry import StatisticsRegistry
from .spectral_utils import reconstruct_spectral_moments


def wavelength_at_depth(L_deep: np.ndarray, depth: np.ndarray, iterations: int = 10) -> np.ndarray:
    """
    Calculate wavelength at a given depth using iterative dispersion relation.

    The deep water wavelength is: L_deep = g * T^2 / (2 * pi)
    At finite depth: L = L_deep * tanh(2 * pi * d / L)
    """
    safe_depth = np.maximum(depth, 0.1)

    L = L_deep.copy()
    for _ in range(iterations):
        kd = 2 * np.pi * safe_depth / np.where(L > 0, L, 1)
        L = L_deep * np.tanh(kd)

    L = np.where(depth > 0, L, np.nan)
    return L


@StatisticsRegistry.register
class PeakinessStatistic(StatisticFunction):
    """
    Wavelength, steepness, combined steepness, and Ursell number.

    Output columns:
    - wavelength_0..3: Wavelength per partition at local depth (m)
    - steepness_0..3: H/L steepness per partition
    - combined_steepness: Hs_total / L_02 (deep water)
    - ursell_number: Hs * L_dominant^2 / depth^3 (wave nonlinearity)
    """

    @property
    def name(self) -> str:
        return "peakiness"

    @property
    def units(self) -> str:
        return "m"

    @property
    def description(self) -> str:
        return "Wavelength, steepness, combined steepness, and Ursell number"

    @property
    def output_columns(self) -> List[str]:
        return [
            "wavelength_0", "wavelength_1", "wavelength_2", "wavelength_3",
            "steepness_0", "steepness_1", "steepness_2", "steepness_3",
            "combined_steepness", "ursell_number",
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
            L_deep = g * p.tp ** 2 / (2 * np.pi)
            L = wavelength_at_depth(L_deep, depths)
            L = np.where(p.is_valid, L, np.nan)
            wavelengths[i] = L
            steepness[i] = np.where(L > 0, p.hs / L, 0)

        # --- Combined steepness: Hs_total / L_02 (deep water) ---
        moments = reconstruct_spectral_moments(partitions, n_points)
        Hs_total = 4.0 * np.sqrt(np.maximum(moments.m0, 0.0))
        L_02_deep = g * moments.T_02 ** 2 / (2 * np.pi)
        combined_steepness = np.where(
            moments.valid & (L_02_deep > 0),
            Hs_total / L_02_deep,
            np.nan,
        )

        # --- Ursell number: Hs * L_dominant^2 / depth^3 ---
        # Use wavelength at depth from the most energetic partition
        max_E_idx = np.zeros(n_points, dtype=int)
        max_E_val = np.zeros(n_points)
        for i, E_i in enumerate(moments.partition_E):
            better = E_i > max_E_val
            max_E_idx = np.where(better, i, max_E_idx)
            max_E_val = np.where(better, E_i, max_E_val)

        # Gather dominant wavelength at depth
        L_dominant = np.zeros(n_points)
        for i in range(n_parts):
            mask = max_E_idx == i
            L_dominant = np.where(mask, wavelengths[i], L_dominant)

        safe_depth = np.maximum(depths, 0.1)
        ursell = np.where(
            moments.valid & (L_dominant > 0) & (depths > 0),
            Hs_total * L_dominant ** 2 / safe_depth ** 3,
            np.nan,
        )

        # --- Assemble output ---
        # Columns: wavelength_0..3, steepness_0..3, combined_steepness, ursell
        n_cols = 2 * 4 + 2  # max 4 partitions * 2 + 2 new columns
        values = np.full((n_points, n_cols), np.nan)

        for i in range(min(n_parts, 4)):
            values[:, i] = wavelengths[i]
            values[:, 4 + i] = steepness[i]

        values[:, 8] = combined_steepness
        values[:, 9] = ursell

        # Build column list matching actual data
        cols = self.output_columns[:min(n_parts, 4)]
        cols += self.output_columns[4:4 + min(n_parts, 4)]
        cols += ["combined_steepness", "ursell_number"]

        return StatisticOutput(
            name=self.name,
            values=values,
            units=self.units,
            description=self.description,
            extra={"columns": cols},
        )
