"""
Height Amplification Statistic

Calculates the expected ratio of maximum wave height to significant wave height
using Forristall's (1978) empirical Weibull distribution with bandwidth correction.

The effective number of independent waves N_eff accounts for spectral bandwidth:
narrow-band spectra have fewer independent waves (stronger grouping, higher
H_max/Hs), while broad spectra have more independent waves.

Also outputs the energy distribution ratio (Hs_total / Hs_dominant) as a
secondary column indicating how spread the energy is across partitions.

References
----------
Forristall, G.Z. (1978). On the statistical distribution of wave heights in a storm.
    J. Geophys. Res. 83(C5).
Cartwright, D.E. & Longuet-Higgins, M.S. (1956). Statistical distribution of the
    maxima of a random function. Proc. R. Soc. London A237.
"""

from typing import List

import numpy as np

from data.surfzone.runner.swan_input_provider import WavePartition
from .base import StatisticFunction, StatisticOutput
from .registry import StatisticsRegistry
from .spectral_utils import reconstruct_spectral_moments

# Observation window for H_max estimate (30 minutes)
T_RECORD = 1800.0


@StatisticsRegistry.register
class HeightAmplificationStatistic(StatisticFunction):
    """
    Expected H_max/Hs ratio using Forristall (1978) Weibull distribution.

    H_max/Hs = 0.681 * ln(N_eff)^(1/2.126)

    where N_eff = N / sqrt(1 + nu^2) corrects for spectral bandwidth.

    Output columns:
    - height_amplification: H_max/Hs ratio (Forristall)
    - energy_ratio: Hs_total / Hs_dominant (energy distribution)
    """

    @property
    def name(self) -> str:
        return "height_amplification"

    @property
    def units(self) -> str:
        return "ratio"

    @property
    def description(self) -> str:
        return "Expected H_max/Hs ratio (Forristall) and energy distribution"

    @property
    def output_columns(self) -> List[str]:
        return ["height_amplification", "energy_ratio"]

    def compute_vectorized(
        self,
        partitions: List[WavePartition],
        depths: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> StatisticOutput:
        n_points = len(depths)
        moments = reconstruct_spectral_moments(partitions, n_points)

        # Number of waves in observation window
        safe_T_01 = np.where(moments.valid & (moments.T_01 > 0), moments.T_01, 1.0)
        N = T_RECORD / safe_T_01

        # Effective independent waves (bandwidth correction)
        # nu = spectral narrowness; higher nu = broader spectrum = more independent waves
        safe_nu = np.where(moments.valid, moments.nu, 0.0)
        N_eff = N / np.sqrt(1.0 + safe_nu ** 2)
        N_eff = np.maximum(N_eff, 2.0)  # minimum 2 waves to avoid log issues

        # Forristall (1978): H_max/Hs = 0.681 * ln(N_eff)^(1/2.126)
        amplification = np.where(
            moments.valid,
            0.681 * np.log(N_eff) ** (1.0 / 2.126),
            np.nan,
        )

        # Energy distribution ratio: Hs_total / Hs_dominant
        # Hs_total = 4*sqrt(m0), Hs_dominant from most energetic partition
        max_E = np.zeros(n_points)
        for E_i in moments.partition_E:
            max_E = np.maximum(max_E, E_i)

        # Hs_dominant = 4*sqrt(E_dominant), Hs_total = 4*sqrt(m0)
        # ratio = sqrt(m0) / sqrt(E_dominant) = sqrt(m0 / E_dominant)
        safe_max_E = np.where(max_E > 0, max_E, 1.0)
        safe_m0 = np.where(moments.valid, moments.m0, 1.0)
        energy_ratio = np.where(
            moments.valid & (max_E > 0),
            np.sqrt(safe_m0 / safe_max_E),
            np.nan,
        )

        # Assemble multi-column output
        values = np.column_stack([amplification, energy_ratio])

        return StatisticOutput(
            name=self.name,
            values=values,
            units=self.units,
            description=self.description,
            extra={"columns": self.output_columns},
        )
