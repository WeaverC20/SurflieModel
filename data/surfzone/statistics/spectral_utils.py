"""
Shared spectral reconstruction utilities for wave statistics.

Reconstructs spectral moments (m0, m1, m2) from SWAN partition summaries
(Hs, Tp, direction per partition). Each partition is modelled as a narrow-band
Gaussian spectral peak. These moments feed Rice/Longuet-Higgins envelope
theory for set period, set duration, groupiness, and related statistics.

References
----------
Rice, S.O. (1945). Mathematical analysis of random noise. Bell System Tech. J. 24.
Longuet-Higgins, M.S. (1984). Statistical properties of wave groups in a random sea.
    Phil. Trans. R. Soc. London A312.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np

from data.surfzone.runner.swan_input_provider import WavePartition

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BETA_SWELL = 0.07       # relative bandwidth for distant swell
BETA_WIND_SEA = 0.12    # relative bandwidth for wind sea / short-period chop


# ---------------------------------------------------------------------------
# SpectralMoments dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpectralMoments:
    """Spectral moments and derived parameters reconstructed from SWAN partitions."""

    m0: np.ndarray          # zeroth moment (total energy), shape (n_points,)
    m1: np.ndarray          # first moment, shape (n_points,)
    m2: np.ndarray          # second moment, shape (n_points,)
    sigma_f: np.ndarray     # combined frequency spread, shape (n_points,)
    f_mean: np.ndarray      # energy-weighted mean frequency m1/m0, shape (n_points,)
    T_01: np.ndarray        # mean period m0/m1, shape (n_points,)
    T_02: np.ndarray        # zero-crossing period sqrt(m0/m2), shape (n_points,)
    nu: np.ndarray          # spectral narrowness sqrt(m0*m2/m1^2 - 1), shape (n_points,)
    partition_E: List[np.ndarray] = field(default_factory=list)
    partition_f: List[np.ndarray] = field(default_factory=list)
    partition_sigma: List[np.ndarray] = field(default_factory=list)
    partition_dir: List[np.ndarray] = field(default_factory=list)
    valid: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))


# ---------------------------------------------------------------------------
# Spectral reconstruction
# ---------------------------------------------------------------------------

def reconstruct_spectral_moments(
    partitions: List[WavePartition],
    n_points: int,
) -> SpectralMoments:
    """
    Reconstruct spectral moments from SWAN partition summaries.

    Each partition is modelled as a Gaussian spectral peak at f_i = 1/Tp_i with
    energy E_i = (Hs_i/4)^2 and intrinsic relative bandwidth beta_i.

    Returns a SpectralMoments dataclass with m0, m1, m2, sigma_f, f_mean,
    T_01, T_02, nu, and per-partition arrays. All fields are NaN where no
    valid partition data exists.
    """
    m0 = np.zeros(n_points)
    m1 = np.zeros(n_points)
    m2 = np.zeros(n_points)

    part_E_list = []
    part_f_list = []
    part_sigma_list = []
    part_dir_list = []

    for p in partitions:
        beta = BETA_WIND_SEA if p.partition_id == 0 else BETA_SWELL
        E = np.where(p.is_valid & (p.tp > 0), (p.hs / 4.0) ** 2, 0.0)
        f_i = np.where(p.tp > 0, 1.0 / p.tp, 0.0)
        sigma_fi = beta * f_i

        m0 += E
        m1 += E * f_i
        m2 += E * (f_i ** 2 + sigma_fi ** 2)

        part_E_list.append(E)
        part_f_list.append(f_i)
        part_sigma_list.append(sigma_fi)
        part_dir_list.append(np.where(p.is_valid, p.direction, 0.0))

    valid = m0 > 0
    safe_m0 = np.where(valid, m0, 1.0)
    safe_m1 = np.where(valid & (m1 > 0), m1, 1.0)

    sigma_f = np.where(
        valid,
        np.sqrt(np.maximum(0.0, m2 / safe_m0 - (m1 / safe_m0) ** 2)),
        np.nan,
    )
    f_mean = np.where(valid, m1 / safe_m0, np.nan)
    T_01 = np.where(valid & (m1 > 0), m0 / safe_m1, np.nan)
    T_02 = np.where(
        valid & (m2 > 0),
        np.sqrt(m0 / np.where(m2 > 0, m2, 1.0)),
        np.nan,
    )
    nu = np.where(
        valid & (m1 > 0),
        np.sqrt(np.maximum(0.0, m0 * m2 / (safe_m1 ** 2) - 1.0)),
        np.nan,
    )

    # NaN out aggregate moments where invalid
    m0_out = np.where(valid, m0, np.nan)
    m1_out = np.where(valid, m1, np.nan)
    m2_out = np.where(valid, m2, np.nan)

    return SpectralMoments(
        m0=m0_out, m1=m1_out, m2=m2_out,
        sigma_f=sigma_f, f_mean=f_mean,
        T_01=T_01, T_02=T_02, nu=nu,
        partition_E=part_E_list,
        partition_f=part_f_list,
        partition_sigma=part_sigma_list,
        partition_dir=part_dir_list,
        valid=valid,
    )


# ---------------------------------------------------------------------------
# Phasor correlation
# ---------------------------------------------------------------------------

def phasor_correlation(moments: SpectralMoments) -> np.ndarray:
    """
    Phasor-based spectral autocorrelation coefficient at lag T_01.

    gamma = |R(T_01)| / m0 where R is the complex autocorrelation:
        R = sum_i E_i * exp(j*2pi*f_i*T_01) * exp(-2*(pi*sigma_fi*T_01)^2)

    This properly handles multi-peaked spectra and preserves within-partition
    bandwidth damping (via sigma_fi), unlike the old _spectral_correlation
    which omitted intrinsic bandwidth and gave gamma -> 1 for single swell.

    Returns gamma in [0, 0.999], NaN where invalid.
    """
    # Use safe T_01 (0 where invalid) to avoid NaN propagation in trig
    safe_T_01 = np.where(moments.valid, moments.T_01, 0.0)
    R_real = np.zeros_like(safe_T_01)
    R_imag = np.zeros_like(safe_T_01)

    for E_i, f_i, sigma_fi in zip(
        moments.partition_E, moments.partition_f, moments.partition_sigma
    ):
        phase = 2.0 * np.pi * f_i * safe_T_01
        damping = np.exp(-2.0 * (np.pi * sigma_fi * safe_T_01) ** 2)
        R_real += E_i * np.cos(phase) * damping
        R_imag += E_i * np.sin(phase) * damping

    R_mag = np.sqrt(R_real ** 2 + R_imag ** 2)
    safe_m0 = np.where(moments.valid, moments.m0, 1.0)
    gamma = np.where(
        moments.valid,
        np.clip(R_mag / safe_m0, 0.0, 0.999),
        np.nan,
    )
    return gamma


# ---------------------------------------------------------------------------
# Directional coherence
# ---------------------------------------------------------------------------

def directional_coherence(dir_i: np.ndarray, dir_j: np.ndarray) -> np.ndarray:
    """
    Directional coherence factor between two swell partitions.

    Returns cos^2(delta_theta/2) in [0, 1]:
      1.0 -> same direction (full constructive interference)
      0.0 -> opposite direction (no beating)
    """
    delta_deg = np.abs(dir_i - dir_j) % 360.0
    delta_deg = np.where(delta_deg > 180.0, 360.0 - delta_deg, delta_deg)
    delta_rad = np.radians(delta_deg)
    return np.cos(delta_rad / 2.0) ** 2
