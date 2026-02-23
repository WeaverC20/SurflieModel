"""
Set Frequency Statistic

Computes wave set period, set wave height, lull wave height, and set fraction
using spectral reconstruction + Rice/Longuet-Higgins envelope threshold crossing theory.

Physical basis
--------------
Each SWAN partition (Hs_i, Tp_i, dir_i) is modelled as a narrow-band Gaussian
spectral peak centred at f_i = 1/Tp_i with energy E_i = (Hs_i/4)² and an assumed
intrinsic relative bandwidth β.  The combined spectrum's frequency spread σ_f is
reconstructed from these partitions; it captures both within-partition bandwidth
and the frequency separation between different swells.

The wave envelope A(t) of the combined sea surface follows a Rayleigh distribution
with parameter m₀ = Σ E_i.  The rate at which A(t) up-crosses a threshold level
corresponding to "waves exceeding α × Hs_total" is given by the Rice (1945) formula
applied to the envelope, which after substituting H = 2A and Hs = 4√m₀ simplifies to:

    ν⁺(α) = √(2π) × σ_f × 2α × exp(-2α²)

The mean set period at threshold α is T_set = 1 / ν⁺(α).

This correctly handles:
- Single swell (finite set period from intrinsic bandwidth alone)
- Multiple swells of any number (σ_f widens with frequency separation)
- Wind sea partitions (broader β broadens σ_f further, shortening set periods)
- Wave height distribution (α anchored to the Rayleigh exceedance P = exp(-2α²))

Tuning
------
β (intrinsic bandwidth):
  0.05  → very narrow groundswell
  0.07  → typical distant swell (default)
  0.12  → wind sea (default)
  Increasing β shortens set periods; decreasing β lengthens them.

α (set threshold, fraction of combined Hs):
  0.8  → ~28% of waves qualify (frequent sets)
  1.0  → ~13.5% of waves qualify (default, waves exceeding Hs)
  1.2  →  ~5.7% of waves qualify
  1.5  →  ~1.1% of waves qualify (only the largest)
  Increase α to lengthen set periods; decrease to shorten them.

References
----------
Rice, S.O. (1945). Mathematical analysis of random noise. Bell System Tech. J. 24.
Longuet-Higgins, M.S. (1984). Statistical properties of wave groups in a random sea.
    Phil. Trans. R. Soc. London A312.
"""

from typing import List

import numpy as np
from scipy.special import erfc

from data.surfzone.runner.swan_input_provider import WavePartition
from .base import StatisticFunction, StatisticOutput
from .registry import StatisticsRegistry
from .spectral_utils import reconstruct_spectral_moments

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default set-height threshold as a fraction of combined Hs
ALPHA_DEFAULT = 1.0

# Threshold sweep for the distribution output (ascending order required)
ALPHA_SWEEP = [0.8, 1.0, 1.2, 1.5]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _set_wave_height(m0: np.ndarray, alpha: float) -> np.ndarray:
    """
    Conditional expected wave height E[H | H > α × Hs_total].

    Derived from the Rayleigh wave height PDF p(H) = (H/(4m0)) exp(-H²/(8m0)).

    Full derivation:
        ∫_h^∞ H p(H) dH = 2√(2m0) × [(√π/2)·erfc(√t0) + √t0·e^(-t0)]
        P(H > h)         = e^(-t0)
        where t0 = h²/(8m0) = 2α²  (for h = α·Hs = α·4√m0)

    Dividing and simplifying (denominator absorbed into expression):
        E[H | H > α·Hs] = 2√(2m0) × [(√π/2)·erfc(α√2)·e^(2α²) + α√2]

    This is the FINAL expression — no additional division by P is needed.

    Intuition:
        α = 1.0  →  E ≈ 1.21 × Hs_total
        α = 1.2  →  E ≈ 1.36 × Hs_total
    """
    z = alpha * np.sqrt(2.0)          # = sqrt(t0), where t0 = 2*alpha²
    sqrt_m0 = np.sqrt(np.maximum(m0, 0.0))
    # Denominator exp(-t0) = exp(-z²) already absorbed: multiply erfc term by exp(z²)
    return 2.0 * np.sqrt(2.0) * sqrt_m0 * (
        (np.sqrt(np.pi) / 2.0) * erfc(z) * np.exp(z ** 2) + z
    )


def _lull_wave_height(
    m0: np.ndarray,
    set_height: np.ndarray,
    set_fraction: float,
) -> np.ndarray:
    """
    Conditional expected wave height E[H | H ≤ α × Hs_total] (lull waves).

    Derived from the law of total expectation:
        E[H] = P_set × E_set + P_lull × E_lull
        →  E_lull = (E[H] - P_set × E_set) / P_lull

    Unconditional Rayleigh mean for p(H) = (H/(4m0))·exp(-H²/(8m0)):
        E[H] = σ × √(π/2) = 2√m0 × √(π/2) = √(2π·m0)
    """
    e_total = np.sqrt(2.0 * np.pi * np.maximum(m0, 0.0))
    p_lull = 1.0 - set_fraction
    return np.where(
        p_lull > 0,
        (e_total - set_fraction * set_height) / p_lull,
        np.nan,
    )


# ---------------------------------------------------------------------------
# Statistic class
# ---------------------------------------------------------------------------

@StatisticsRegistry.register
class SetFrequencyStatistic(StatisticFunction):
    """
    Wave set statistics using Rice/Longuet-Higgins envelope crossing theory.

    Outputs (multi-column, one row per mesh point):

      set_period          — mean time between sets at default α=1.0 (s)
      set_height          — E[H | H > α·Hs], expected height of a set wave (m)
      lull_height         — E[H | H ≤ α·Hs], expected height of a lull wave (m)
      set_fraction        — fraction of all waves qualifying as set waves
      set_period_alpha_*  — set period at each threshold in ALPHA_SWEEP (s)

    See module docstring for full derivation, assumptions, and tuning guide.
    """

    @property
    def name(self) -> str:
        return "set_period"

    @property
    def units(self) -> str:
        return "s"

    @property
    def description(self) -> str:
        return (
            "Mean time between wave sets (Rice envelope crossing theory, α=1.0). "
            "Also provides set_height, lull_height, set_fraction, and set period "
            "distribution across multiple thresholds."
        )

    @property
    def output_columns(self) -> List[str]:
        sweep_cols = [f"set_period_alpha_{a}" for a in ALPHA_SWEEP]
        return ["set_period", "set_height", "lull_height", "set_fraction"] + sweep_cols

    def compute_vectorized(
        self,
        partitions: List[WavePartition],
        depths: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
        **kwargs,
    ) -> StatisticOutput:
        n_points = len(depths)

        # --- Step 1: reconstruct spectral moments ---
        moments = reconstruct_spectral_moments(partitions, n_points)
        m0, sigma_f = moments.m0, moments.sigma_f

        # --- Step 2: set period at default threshold ---
        # ν⁺(α) = √(2π) × σ_f × 2α × exp(-2α²)
        # T_set  = 1 / ν⁺(α)
        def _crossing_period(alpha: float) -> np.ndarray:
            rate = (
                np.sqrt(2.0 * np.pi)
                * sigma_f
                * 2.0 * alpha
                * np.exp(-2.0 * alpha ** 2)
            )
            return np.where(rate > 0, 1.0 / rate, np.nan)

        set_period = _crossing_period(ALPHA_DEFAULT)

        # --- Step 3: set/lull wave heights from Rayleigh distribution ---
        set_fraction = float(np.exp(-2.0 * ALPHA_DEFAULT ** 2))  # ~0.135 at α=1.0
        set_height = _set_wave_height(m0, ALPHA_DEFAULT)
        lull_height = _lull_wave_height(m0, set_height, set_fraction)

        # --- Step 4: threshold sweep (distribution) ---
        sweep_values = [_crossing_period(a) for a in ALPHA_SWEEP]

        # --- Assemble multi-column output ---
        cols = self.output_columns
        values = np.full((n_points, len(cols)), np.nan)

        invalid = np.isnan(m0)
        values[:, 0] = set_period
        values[:, 1] = set_height
        values[:, 2] = lull_height
        values[:, 3] = np.where(invalid, np.nan, set_fraction)
        for i, sv in enumerate(sweep_values):
            values[:, 4 + i] = sv

        return StatisticOutput(
            name=self.name,
            values=values,
            units=self.units,
            description=self.description,
            extra={"columns": cols},
        )
