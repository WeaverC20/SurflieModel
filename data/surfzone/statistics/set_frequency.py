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

from typing import List, Tuple

import numpy as np
from scipy.special import erfc

from data.surfzone.runner.swan_input_provider import WavePartition
from .base import StatisticFunction, StatisticOutput
from .registry import StatisticsRegistry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Relative bandwidth β for each partition type (std_freq / peak_freq)
BETA_SWELL = 0.07       # typical distant swell
BETA_WIND_SEA = 0.12    # wind sea / short-period chop

# Default set-height threshold as a fraction of combined Hs
ALPHA_DEFAULT = 1.0

# Threshold sweep for the distribution output (ascending order required)
ALPHA_SWEEP = [0.8, 1.0, 1.2, 1.5]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _reconstruct_spectral_moments(
    partitions: List[WavePartition],
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct combined spectral moments from SWAN partition summaries.

    Each partition is modelled as a Gaussian spectral peak at f_i = 1/Tp_i with
    energy E_i = (Hs_i/4)² and intrinsic relative bandwidth β_i.

    Returns
    -------
    m0 : np.ndarray, shape (n_points,)
        Zeroth moment (total energy).  Hs_total = 4 * sqrt(m0).
    sigma_f : np.ndarray, shape (n_points,)
        Combined frequency spread (std of energy-weighted frequency distribution).
        Captures both within-partition bandwidth and between-partition separation.
    """
    m0 = np.zeros(n_points)
    m1 = np.zeros(n_points)
    m2 = np.zeros(n_points)

    for p in partitions:
        beta = BETA_WIND_SEA if p.partition_id == 0 else BETA_SWELL
        # Energy from this partition (zero out invalid points)
        E = np.where(p.is_valid & (p.tp > 0), (p.hs / 4.0) ** 2, 0.0)
        # Peak frequency and intrinsic bandwidth
        f_i = np.where(p.tp > 0, 1.0 / p.tp, 0.0)
        sigma_fi = beta * f_i

        m0 += E
        m1 += E * f_i
        m2 += E * (f_i ** 2 + sigma_fi ** 2)

    # Combined frequency spread: σ_f = sqrt(m2/m0 - (m1/m0)²)
    # Protected against m0 == 0
    valid = m0 > 0
    safe_m0 = np.where(valid, m0, 1.0)
    sigma_f = np.where(
        valid,
        np.sqrt(np.maximum(0.0, m2 / safe_m0 - (m1 / safe_m0) ** 2)),
        np.nan,
    )
    m0 = np.where(valid, m0, np.nan)
    return m0, sigma_f


def _directional_coherence(dir_i: np.ndarray, dir_j: np.ndarray) -> np.ndarray:
    """
    Directional coherence factor between two swell partitions.

    Returns a value in [0, 1]:
      1.0 → same direction (full constructive interference)
      0.0 → opposite / perpendicular direction (no beating)

    Uses cos²(Δθ/2) for a smooth rolloff with angular separation.

    DORMANT — not called by the current implementation.
    Enable this function if results for mixed-direction swell combos
    (e.g. NW swell + S swell at a SoCal point) appear too short.
    To activate: multiply cross-partition energy contributions to m2 by this
    coherence factor inside _reconstruct_spectral_moments, specifically the
    between-partition frequency separation terms in the m2 expansion.

    Parameters
    ----------
    dir_i, dir_j : np.ndarray
        Nautical directions (degrees, 0=N clockwise) for two partitions.

    Returns
    -------
    np.ndarray
        Coherence factor, shape (n_points,).
    """
    delta_deg = np.abs(dir_i - dir_j) % 360.0
    delta_deg = np.where(delta_deg > 180.0, 360.0 - delta_deg, delta_deg)
    delta_rad = np.radians(delta_deg)
    return np.cos(delta_rad / 2.0) ** 2


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
    ) -> StatisticOutput:
        n_points = len(depths)

        # --- Step 1: reconstruct spectral moments ---
        m0, sigma_f = _reconstruct_spectral_moments(partitions, n_points)

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
