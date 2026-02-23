"""
Wave Breaking Detection and Breaker Type Classification

Determines whether waves are breaking at each mesh point and classifies
the breaker type using the Iribarren number (surf similarity parameter).

Physics (no-wind formulation):
    1. Combine partitions: Hs_total = sqrt(sum(Hs_i^2))
    2. Deep water reference: L0 = g*Tp^2/(2*pi)
    3. Local wave properties: L via Fenton-McKee, C = L/Tp
    4. Breaker index: Rattanapitikon & Shibayama (slope + steepness dependent)
       gamma_b = 0.57 + 0.71*(H0/L0)^0.12 * m^0.36
    5. Breaking criterion: H/h >= gamma_b
    6. Iribarren number: xi = m / sqrt(H/L0)
    7. Breaker type classification from xi

Known limitations:
    - H0 approximation: Uses propagated Hs at mesh point as H0 proxy.
      The actual deep water H0 would require inverse-shoaling or loading
      SWAN offshore boundary data.
    - Wind effects deferred: The full formulation includes wind-modified
      breaker index and effective steepness. Requires GFS wind data
      integration into the statistics pipeline.

References
----------
McCowan, J. (1894). On the highest wave of permanent type.
Rattanapitikon, W. & Shibayama, T. (2000). Verification and modification
    of breaker height formulas. Coastal Eng. Journal 42(4).
Battjes, J.A. (1974). Surf similarity. Proc. 14th Coastal Eng. Conf.
Douglass, S.L. (1990). Influence of wind on breaking waves.
    J. Waterway, Port, Coastal, Ocean Eng. 116(6).
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

        # Get slopes from context (computed by run_statistics.py from mesh)
        slopes = kwargs.get("slopes", None)
        if slopes is None:
            slopes = np.full(n_points, np.nan)

        # --- Step 1: Combine partitions ---
        # Hs_total = sqrt(sum(Hs_i^2)), Tp from highest-energy partition
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

            # Track dominant partition (highest Hs^2) for Tp selection
            better = energy > max_energy
            tp_dominant = np.where(better, p.tp, tp_dominant)
            max_energy = np.where(better, energy, max_energy)

        hs_total = np.sqrt(hs_sq_sum)

        # --- Step 2: Deep water reference ---
        L0 = G * tp_dominant**2 / TWO_PI
        # H0/L0 steepness (using local Hs as H0 proxy — see Known Limitations)
        safe_L0_div = np.where(L0 > 0, L0, 1.0)
        H0_L0 = np.where(L0 > 0, hs_total / safe_L0_div, 0.0)

        # --- Step 3: Local wave properties (Fenton-McKee) ---
        safe_depth = np.maximum(depths, 0.05)
        safe_L0 = np.where(L0 > 0, L0, 1.0)
        x_fm = (TWO_PI * safe_depth / safe_L0) ** 0.75
        L_local = safe_L0 * np.tanh(x_fm) ** (2.0 / 3.0)
        safe_tp = np.where(tp_dominant > 0, tp_dominant, 1.0)
        C_local = L_local / safe_tp

        # --- Step 4: Breaker index (Rattanapitikon & Shibayama) ---
        # gamma_b = 0.57 + 0.71 * (H0/L0)^0.12 * m^0.36
        safe_H0_L0 = np.where(H0_L0 > 0, H0_L0, 1e-6)
        safe_slopes = np.where(np.isfinite(slopes) & (slopes > 0), slopes, 0.005)
        gamma_b = 0.57 + 0.71 * safe_H0_L0**0.12 * safe_slopes**0.36
        # Clamp to reasonable range
        gamma_b = np.clip(gamma_b, 0.5, 1.5)

        # --- Step 5: Breaking criterion ---
        # H/h >= gamma_b
        H_over_h = np.where(safe_depth > 0.05, hs_total / safe_depth, 0.0)
        is_breaking_bool = (H_over_h >= gamma_b) & (depths > 0.05) & any_valid
        is_breaking = np.where(any_valid, is_breaking_bool.astype(float), np.nan)

        # Breaking intensity: H / (gamma_b * h), >1 means breaking
        breaking_intensity = np.where(
            any_valid & (safe_depth > 0.05),
            hs_total / (gamma_b * safe_depth),
            np.nan,
        )

        # --- Step 6: Iribarren number ---
        # xi = m / sqrt(H/L0)
        steepness = np.where(L0 > 0, hs_total / safe_L0_div, 0.0)
        safe_steepness = np.where(steepness > 0, steepness, np.nan)
        iribarren = np.where(
            any_valid & (steepness > 0) & np.isfinite(slopes),
            slopes / np.sqrt(safe_steepness),
            np.nan,
        )

        # --- Step 7: Breaker type classification ---
        # Only classify where actually breaking
        breaker_type = np.full(n_points, np.nan)
        valid_breaking = is_breaking_bool & np.isfinite(iribarren)
        xi = np.where(np.isfinite(iribarren), iribarren, 0.0)
        breaker_type = np.where(valid_breaking & (xi < 0.5), 0.0, breaker_type)
        breaker_type = np.where(
            valid_breaking & (xi >= 0.5) & (xi < 3.3), 1.0, breaker_type
        )
        breaker_type = np.where(
            valid_breaking & (xi >= 3.3) & (xi < 5.0), 2.0, breaker_type
        )
        breaker_type = np.where(valid_breaking & (xi >= 5.0), 3.0, breaker_type)

        # Assemble output: [is_breaking, breaker_index, iribarren, breaker_type, breaking_intensity]
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
