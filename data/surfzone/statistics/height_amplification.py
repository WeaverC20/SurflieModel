"""
Height Amplification Statistic

Calculates the amplification factor for set waves due to constructive
interference between multiple swells.
"""

from typing import List

import numpy as np

from data.surfzone.runner.swan_input_provider import WavePartition
from .base import StatisticFunction, StatisticOutput
from .registry import StatisticsRegistry


@StatisticsRegistry.register
class HeightAmplificationStatistic(StatisticFunction):
    """
    Height amplification factor for set waves.

    When multiple swells are present, waves can constructively interfere
    during sets, producing waves larger than the combined Hs would suggest.

    For two swells with Hs1 and Hs2:
    - Combined Hs = sqrt(Hs1^2 + Hs2^2) (energy addition)
    - Max aligned height = Hs1 + Hs2 (linear superposition)
    - Amplification = (Hs1 + Hs2) / sqrt(Hs1^2 + Hs2^2)

    Maximum amplification of sqrt(2) ≈ 1.41 occurs when two swells have equal energy.
    """

    @property
    def name(self) -> str:
        return "height_amplification"

    @property
    def units(self) -> str:
        return "ratio"

    @property
    def description(self) -> str:
        return "Ratio of max set wave height to combined Hs"

    def compute_vectorized(
        self,
        partitions: List[WavePartition],
        depths: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> StatisticOutput:
        n_points = len(depths)

        # Get swells only (partition_id > 0)
        swells = [p for p in partitions if p.partition_id > 0]

        if len(swells) == 0:
            # No swells, no amplification
            return StatisticOutput(
                name=self.name,
                values=np.ones(n_points),
                units=self.units,
                description=self.description
            )

        # Stack Hs values: shape (n_swells, n_points)
        hs_stack = np.array([
            np.where(p.is_valid, p.hs, 0) for p in swells
        ])

        # Get top 2 swells by energy at each point
        energies = hs_stack**2
        sorted_idx = np.argsort(energies, axis=0)

        if len(swells) >= 2:
            idx1 = sorted_idx[-1, :]
            idx2 = sorted_idx[-2, :]
            hs1 = np.take_along_axis(hs_stack, idx1[np.newaxis, :], axis=0)[0]
            hs2 = np.take_along_axis(hs_stack, idx2[np.newaxis, :], axis=0)[0]
        else:
            # Only one swell
            hs1 = hs_stack[0]
            hs2 = np.zeros(n_points)

        # Calculate amplification factor
        # Linear superposition (max when aligned)
        hs_aligned = hs1 + hs2

        # Energy-based combined Hs
        hs_combined = np.sqrt(hs1**2 + hs2**2)

        # Amplification ratio
        amplification = np.where(
            hs_combined > 0,
            hs_aligned / hs_combined,
            1.0
        )

        # Clip to valid range (1.0 to sqrt(2))
        amplification = np.clip(amplification, 1.0, np.sqrt(2))

        return StatisticOutput(
            name=self.name,
            values=amplification,
            units=self.units,
            description=self.description
        )
