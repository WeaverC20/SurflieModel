"""
Groupiness Factor Statistic

Measures the degree of wave grouping using the Funke & Mansard (1980) definition:

    GF = sqrt(2) * sigma_A / mu_A

where sigma_A is the envelope amplitude standard deviation and mu_A is the mean.

For a single narrow-band process (Rayleigh envelope), the baseline is GF ~ 0.74.
Values above 0.74 require multi-modal beating between partitions with different
peak frequencies. The beating contribution is modulated by:
  - Bandwidth damping: partitions with large intrinsic bandwidth relative to
    their frequency separation produce less coherent beating
  - Directional coherence: partitions arriving from different directions
    interfere less effectively

References
----------
Funke, E.R. & Mansard, E.P.D. (1980). On the synthesis of realistic sea states.
    Proc. ICCE.
Longuet-Higgins, M.S. (1984). Statistical properties of wave groups in a random sea.
    Phil. Trans. R. Soc. London A312.
"""

from typing import List

import numpy as np

from data.surfzone.runner.swan_input_provider import WavePartition
from .base import StatisticFunction, StatisticOutput
from .registry import StatisticsRegistry
from .spectral_utils import (
    reconstruct_spectral_moments,
    directional_coherence,
)


@StatisticsRegistry.register
class GroupinessStatistic(StatisticFunction):
    """
    Groupiness factor (GF) via Funke & Mansard (1980).

    GF = sqrt(2) * sigma_A / mu_A

    Baseline for single narrow-band swell: GF ~ 0.74 (Rayleigh).
    Multi-modal beating increases GF above 0.74.
    """

    @property
    def name(self) -> str:
        return "groupiness_factor"

    @property
    def units(self) -> str:
        return "ratio"

    @property
    def description(self) -> str:
        return "Wave groupiness factor (0.74=single swell baseline, higher=more grouping)"

    def compute_vectorized(
        self,
        partitions: List[WavePartition],
        depths: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> StatisticOutput:
        n_points = len(depths)
        moments = reconstruct_spectral_moments(partitions, n_points)

        m0 = moments.m0
        safe_m0 = np.where(moments.valid, m0, 1.0)

        # Rayleigh baseline: variance of envelope amplitude for narrow-band process
        # sigma_A^2 = (4 - pi) / 2 * m0
        sigma_A_sq = (4.0 - np.pi) / 2.0 * safe_m0

        # Cross-partition beating contributions
        n_parts = len(moments.partition_E)
        for i in range(n_parts):
            for j in range(i + 1, n_parts):
                E_i = moments.partition_E[i]
                E_j = moments.partition_E[j]
                f_i = moments.partition_f[i]
                f_j = moments.partition_f[j]
                s_i = moments.partition_sigma[i]
                s_j = moments.partition_sigma[j]

                delta_f = np.abs(f_i - f_j)

                # Bandwidth damping: exp(-2*pi^2*(sigma_i^2 + sigma_j^2) / delta_f^2)
                # When delta_f -> 0, damping -> 0 (no beating at same frequency)
                safe_df_sq = np.where(delta_f > 1e-6, delta_f ** 2, 1.0)
                bw_damp = np.where(
                    delta_f > 1e-6,
                    np.exp(
                        -2.0 * np.pi ** 2 * (s_i ** 2 + s_j ** 2) / safe_df_sq
                    ),
                    0.0,
                )

                dir_coh = directional_coherence(
                    moments.partition_dir[i], moments.partition_dir[j]
                )

                sigma_A_sq += 2.0 * E_i * E_j * bw_damp * dir_coh

        # Funke & Mansard: GF = sqrt(2) * sigma_A / mu_A
        # mu_A = sqrt(pi * m0 / 2) (Rayleigh mean envelope amplitude)
        mu_A = np.sqrt(np.pi * safe_m0 / 2.0)
        GF = np.where(
            moments.valid & (mu_A > 0),
            np.sqrt(2.0) * np.sqrt(np.maximum(sigma_A_sq, 0.0)) / mu_A,
            np.nan,
        )

        return StatisticOutput(
            name=self.name,
            values=GF,
            units=self.units,
            description=self.description,
        )
