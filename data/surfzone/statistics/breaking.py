"""
Wave Breaking Detection and Breaker Type Classification

Loads pre-computed breaking characterization from the surfzone simulation
output when available. The simulation computes breaking during ray tracing
(Thornton & Guza per-ray dissipation + cross-shore transect correction) and
characterizes it post-correction (Rattanapitikon breaker index, Iribarren
number, breaker type classification).

When simulation breaking fields are not available (old simulation output),
falls back to independent computation using Rattanapitikon & Shibayama.

Output columns:
    - is_breaking: 1.0 if breaking, 0.0 if not
    - breaker_index: gamma_b (Rattanapitikon, slope+steepness dependent)
    - iribarren: surf similarity parameter xi
    - breaker_type: 0=Spilling, 1=Plunging, 2=Collapsing, 3=Surging
    - breaking_intensity: H/(gamma_b*h), >1 means breaking

References
----------
Rattanapitikon, W. & Shibayama, T. (2000). Verification and modification
    of breaker height formulas. Coastal Eng. Journal 42(4).
Battjes, J.A. (1974). Surf similarity. Proc. 14th Coastal Eng. Conf.
Thornton, E.B. & Guza, R.T. (1983). Transformation of wave height
    distribution. JGR 88(C10), 5925-5938.
"""

from typing import List

import numpy as np

from data.surfzone.runner.swan_input_provider import WavePartition
from .base import StatisticFunction, StatisticOutput
from .registry import StatisticsRegistry

G = 9.81
TWO_PI = 2.0 * np.pi


@StatisticsRegistry.register
class BreakingStatistic(StatisticFunction):
    """
    Wave breaking detection and breaker type classification.

    Prefers pre-computed breaking fields from the surfzone simulation
    (passed via context as 'sim_breaking'). Falls back to independent
    Rattanapitikon computation for old simulation output.

    Output columns:
    - is_breaking: 1.0 if breaking, 0.0 if not, NaN if invalid
    - breaker_index: gamma_b (Rattanapitikon, slope+steepness dependent)
    - iribarren: surf similarity parameter xi
    - breaker_type: 0=Spilling, 1=Plunging, 2=Collapsing, 3=Surging (NaN if not breaking)
    - breaking_intensity: H/(gamma_b*h), >1 means breaking
    """

    @property
    def name(self) -> str:
        return "breaking"

    @property
    def units(self) -> str:
        return "mixed"

    @property
    def description(self) -> str:
        return "Breaking detection, Iribarren number, and breaker type (Rattanapitikon)"

    @property
    def output_columns(self) -> List[str]:
        return [
            "is_breaking",
            "breaker_index",
            "iribarren",
            "breaker_type",
            "breaking_intensity",
        ]

    def compute_vectorized(
        self,
        partitions: List[WavePartition],
        depths: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        **kwargs,
    ) -> StatisticOutput:
        n_points = len(depths)

        # Check for pre-computed breaking fields from simulation
        sim_breaking = kwargs.get("sim_breaking")
        if sim_breaking is not None:
            return self._from_simulation(sim_breaking, n_points)

        # Fallback: independent computation for old simulation output
        return self._compute_legacy(partitions, depths, kwargs)

    def _from_simulation(self, sim_breaking: dict, n_points: int) -> StatisticOutput:
        """Use pre-computed breaking fields from the surfzone simulation."""
        values = np.column_stack([
            sim_breaking['is_breaking'],
            sim_breaking['breaker_index'],
            sim_breaking['iribarren'],
            sim_breaking['breaker_type'],
            sim_breaking['breaking_intensity'],
        ])

        return StatisticOutput(
            name=self.name,
            values=values,
            units=self.units,
            description=self.description,
            extra={"columns": self.output_columns},
        )

    def _compute_legacy(
        self,
        partitions: List[WavePartition],
        depths: np.ndarray,
        kwargs: dict,
    ) -> StatisticOutput:
        """
        Legacy fallback: independent breaking computation using Rattanapitikon.

        Used when simulation output doesn't include breaking fields (old format).
        """
        n_points = len(depths)

        slopes = kwargs.get("slopes", None)
        if slopes is None:
            slopes = np.full(n_points, np.nan)

        # Combine partitions: Hs_total, dominant Tp
        hs_sq_sum = np.zeros(n_points)
        any_valid = np.zeros(n_points, dtype=bool)
        max_energy = np.zeros(n_points)
        tp_dominant = np.zeros(n_points)

        for p in partitions:
            valid = p.is_valid
            hs_safe = np.where(valid, p.hs, 0.0)
            energy = hs_safe**2
            hs_sq_sum += energy
            any_valid |= valid
            better = energy > max_energy
            tp_dominant = np.where(better, p.tp, tp_dominant)
            max_energy = np.where(better, energy, max_energy)

        hs_total = np.sqrt(hs_sq_sum)

        # Deep water reference
        L0 = G * tp_dominant**2 / TWO_PI
        safe_L0_div = np.where(L0 > 0, L0, 1.0)
        H0_L0 = np.where(L0 > 0, hs_total / safe_L0_div, 0.0)

        safe_depth = np.maximum(depths, 0.05)

        # Breaker index (Rattanapitikon & Shibayama)
        safe_H0_L0 = np.where(H0_L0 > 0, H0_L0, 1e-6)
        safe_slopes = np.where(np.isfinite(slopes) & (slopes > 0), slopes, 0.005)
        gamma_b = np.clip(0.57 + 0.71 * safe_H0_L0**0.12 * safe_slopes**0.36, 0.5, 1.5)

        # Breaking criterion
        H_over_h = np.where(safe_depth > 0.05, hs_total / safe_depth, 0.0)
        is_breaking_bool = (H_over_h >= gamma_b) & (depths > 0.05) & any_valid
        is_breaking = np.where(any_valid, is_breaking_bool.astype(float), np.nan)

        # Breaking intensity
        breaking_intensity = np.where(
            any_valid & (safe_depth > 0.05),
            hs_total / (gamma_b * safe_depth),
            np.nan,
        )

        # Iribarren number
        steepness = np.where(L0 > 0, hs_total / safe_L0_div, 0.0)
        safe_steepness = np.where(steepness > 0, steepness, np.nan)
        iribarren = np.where(
            any_valid & (steepness > 0) & np.isfinite(slopes),
            slopes / np.sqrt(safe_steepness),
            np.nan,
        )

        # Breaker type classification
        breaker_type = np.full(n_points, np.nan)
        valid_breaking = is_breaking_bool & np.isfinite(iribarren)
        xi = np.where(np.isfinite(iribarren), iribarren, 0.0)
        breaker_type = np.where(valid_breaking & (xi < 0.5), 0.0, breaker_type)
        breaker_type = np.where(valid_breaking & (xi >= 0.5) & (xi < 3.3), 1.0, breaker_type)
        breaker_type = np.where(valid_breaking & (xi >= 3.3) & (xi < 5.0), 2.0, breaker_type)
        breaker_type = np.where(valid_breaking & (xi >= 5.0), 3.0, breaker_type)

        breaker_index_out = np.where(any_valid, gamma_b, np.nan)
        values = np.column_stack(
            [is_breaking, breaker_index_out, iribarren, breaker_type, breaking_intensity]
        )

        return StatisticOutput(
            name=self.name,
            values=values,
            units=self.units,
            description=self.description,
            extra={"columns": self.output_columns},
        )
