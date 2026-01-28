"""
Wave Physics for Surfzone Model

Numba-accelerated functions for wave propagation calculations.
All functions use SI units (meters, seconds, radians where applicable).

References:
- Fenton & McKee (1990): Wavelength approximation
- Snell's Law for wave refraction
- Douglass (1990): Wind effects on breaking
- Rattanapitikon & Shibayama (2000): Breaking index
- Weggel (1972): Slope-dependent breaking
"""

import numpy as np
from numba import njit, prange
from typing import Tuple

# Physical constants
G = 9.81  # Gravitational acceleration (m/s²)
TWO_PI = 2.0 * np.pi


# =============================================================================
# Deep Water Reference Properties (calculate once per wave)
# =============================================================================

@njit(cache=True)
def deep_water_wavelength(T: float) -> float:
    """
    Calculate deep water wavelength.

    L₀ = g·T² / (2π)

    Args:
        T: Wave period (s)

    Returns:
        Deep water wavelength L₀ (m)
    """
    return G * T * T / TWO_PI


@njit(cache=True)
def deep_water_celerity(L0: float, T: float) -> float:
    """
    Calculate deep water wave celerity (phase speed).

    C₀ = L₀ / T

    Args:
        L0: Deep water wavelength (m)
        T: Wave period (s)

    Returns:
        Deep water celerity C₀ (m/s)
    """
    return L0 / T


@njit(cache=True)
def deep_water_group_velocity(C0: float) -> float:
    """
    Calculate deep water group velocity.

    Cg₀ = C₀ / 2

    Args:
        C0: Deep water celerity (m/s)

    Returns:
        Deep water group velocity Cg₀ (m/s)
    """
    return C0 / 2.0


@njit(cache=True)
def deep_water_properties(T: float) -> Tuple[float, float, float]:
    """
    Calculate all deep water reference properties.

    Args:
        T: Wave period (s)

    Returns:
        Tuple of (L0, C0, Cg0) - wavelength, celerity, group velocity
    """
    L0 = deep_water_wavelength(T)
    C0 = deep_water_celerity(L0, T)
    Cg0 = deep_water_group_velocity(C0)
    return L0, C0, Cg0


# =============================================================================
# Local Wave Properties (at each position along ray)
# =============================================================================

@njit(cache=True)
def local_wavelength_fenton_mckee(L0: float, h: float) -> float:
    """
    Calculate local wavelength using Fenton & McKee (1990) approximation.

    L = L₀ · [tanh((2πh/L₀)^(3/4))]^(2/3)

    This approximation is accurate to within 0.5% for all depths.

    Args:
        L0: Deep water wavelength (m)
        h: Local water depth (m)

    Returns:
        Local wavelength L (m)
    """
    if h <= 0:
        return L0  # Return deep water value for invalid depth

    # Fenton & McKee approximation
    x = (TWO_PI * h / L0) ** 0.75
    return L0 * (np.tanh(x) ** (2.0 / 3.0))


@njit(cache=True)
def local_wavenumber(L: float) -> float:
    """
    Calculate local wavenumber.

    k = 2π / L

    Args:
        L: Local wavelength (m)

    Returns:
        Wavenumber k (rad/m)
    """
    return TWO_PI / L


@njit(cache=True)
def local_celerity(L: float, T: float) -> float:
    """
    Calculate local wave celerity.

    C = L / T

    Args:
        L: Local wavelength (m)
        T: Wave period (s)

    Returns:
        Local celerity C (m/s)
    """
    return L / T


@njit(cache=True)
def group_velocity_ratio(k: float, h: float) -> float:
    """
    Calculate group velocity ratio n.

    n = (1/2) · [1 + 2kh / sinh(2kh)]

    In deep water (kh → ∞): n → 0.5
    In shallow water (kh → 0): n → 1.0

    Args:
        k: Wavenumber (rad/m)
        h: Water depth (m)

    Returns:
        Group velocity ratio n (dimensionless)
    """
    kh = k * h

    # Handle numerical limits
    if kh > 10.0:
        return 0.5  # Deep water limit
    if kh < 0.01:
        return 1.0  # Shallow water limit

    sinh_2kh = np.sinh(2.0 * kh)
    return 0.5 * (1.0 + 2.0 * kh / sinh_2kh)


@njit(cache=True)
def local_group_velocity(n: float, C: float) -> float:
    """
    Calculate local group velocity.

    Cg = n · C

    Args:
        n: Group velocity ratio
        C: Local celerity (m/s)

    Returns:
        Local group velocity Cg (m/s)
    """
    return n * C


@njit(cache=True)
def local_wave_properties(L0: float, T: float, h: float) -> Tuple[float, float, float, float, float]:
    """
    Calculate all local wave properties at a given depth.

    Args:
        L0: Deep water wavelength (m)
        T: Wave period (s)
        h: Local water depth (m)

    Returns:
        Tuple of (L, k, C, n, Cg) - wavelength, wavenumber, celerity,
        group velocity ratio, group velocity
    """
    L = local_wavelength_fenton_mckee(L0, h)
    k = local_wavenumber(L)
    C = local_celerity(L, T)
    n = group_velocity_ratio(k, h)
    Cg = local_group_velocity(n, C)
    return L, k, C, n, Cg


# =============================================================================
# Shoaling and Refraction
# =============================================================================

@njit(cache=True)
def shoaling_coefficient(Cg0: float, Cg: float) -> float:
    """
    Calculate shoaling coefficient.

    Ks = √(Cg₀ / Cg)

    Args:
        Cg0: Deep water group velocity (m/s)
        Cg: Local group velocity (m/s)

    Returns:
        Shoaling coefficient Ks (dimensionless)
    """
    if Cg <= 0:
        return 1.0
    return np.sqrt(Cg0 / Cg)


@njit(cache=True)
def refraction_snell(C: float, C0: float, theta0: float) -> float:
    """
    Calculate local wave angle using Snell's Law.

    θ = arcsin((C/C₀) · sin(θ₀))

    Args:
        C: Local celerity (m/s)
        C0: Deep water celerity (m/s)
        theta0: Deep water wave angle (radians, relative to shore normal)

    Returns:
        Local wave angle θ (radians)
    """
    sin_theta = (C / C0) * np.sin(theta0)

    # Clamp to valid range for arcsin
    sin_theta = max(-1.0, min(1.0, sin_theta))

    return np.arcsin(sin_theta)


@njit(cache=True)
def refraction_coefficient(theta0: float, theta: float) -> float:
    """
    Calculate refraction coefficient.

    Kr = √(cos(θ₀) / cos(θ))

    Args:
        theta0: Deep water wave angle (radians)
        theta: Local wave angle (radians)

    Returns:
        Refraction coefficient Kr (dimensionless)
    """
    cos_theta0 = np.cos(theta0)
    cos_theta = np.cos(theta)

    if cos_theta <= 0:
        return 1.0  # Wave at or past critical angle

    return np.sqrt(cos_theta0 / cos_theta)


# =============================================================================
# Wind Effects
# =============================================================================

@njit(cache=True)
def relative_wind_angle(theta_wind: float, theta_wave: float) -> float:
    """
    Calculate relative wind-wave angle.

    φ = θᵥ - θ

    Args:
        theta_wind: Wind direction (radians, FROM convention)
        theta_wave: Wave direction (radians)

    Returns:
        Relative angle φ (radians)
    """
    return theta_wind - theta_wave


@njit(cache=True)
def wind_modification(
    Kw_prev: float,
    U_wind: float,
    phi: float,
    C: float,
    L: float,
    dx: float,
    alpha: float = 0.03,
) -> float:
    """
    Update wind modification factor.

    Kw(x) = Kw(x-Δx) · exp[α · (Uᵥ·cos(φ) - C)/C · Δx/L]

    Positive Kw modification when wind is in wave direction (onshore wind adds energy).
    Negative when offshore wind (removes energy).

    Args:
        Kw_prev: Previous wind modification factor
        U_wind: Wind speed (m/s)
        phi: Relative wind-wave angle (radians)
        C: Local wave celerity (m/s)
        L: Local wavelength (m)
        dx: Step distance (m)
        alpha: Wind energy coefficient (default 0.03)

    Returns:
        Updated wind modification factor Kw
    """
    if C <= 0 or L <= 0:
        return Kw_prev

    wind_component = U_wind * np.cos(phi)
    exponent = alpha * (wind_component - C) / C * dx / L

    # Limit extreme modifications
    exponent = max(-2.0, min(2.0, exponent))

    return Kw_prev * np.exp(exponent)


# =============================================================================
# Wave Height Evolution
# =============================================================================

@njit(cache=True)
def wave_height(H0: float, Ks: float, Kr: float, Kw: float = 1.0) -> float:
    """
    Calculate local wave height.

    H(x) = H₀ · Ks · Kr · Kw

    Args:
        H0: Deep water wave height (m)
        Ks: Shoaling coefficient
        Kr: Refraction coefficient
        Kw: Wind modification factor (default 1.0)

    Returns:
        Local wave height H (m)
    """
    return H0 * Ks * Kr * Kw


@njit(cache=True)
def wave_steepness(H: float, L: float) -> float:
    """
    Calculate wave steepness.

    Steepness = H / L

    Args:
        H: Wave height (m)
        L: Wavelength (m)

    Returns:
        Wave steepness (dimensionless)
    """
    if L <= 0:
        return 0.0
    return H / L


@njit(cache=True)
def effective_steepness(
    H: float,
    L0: float,
    U_wind: float,
    phi: float,
    C: float,
    gamma_w: float = 0.15,
) -> float:
    """
    Calculate effective steepness including wind effects.

    (H/L)eff = (H/L₀) · [1 + γw · Uᵥ·cos(φ)/C]

    Args:
        H: Wave height (m)
        L0: Deep water wavelength (m)
        U_wind: Wind speed (m/s)
        phi: Relative wind-wave angle (radians)
        C: Local celerity (m/s)
        gamma_w: Wind modification coefficient (default 0.15)

    Returns:
        Effective steepness (dimensionless)
    """
    if L0 <= 0 or C <= 0:
        return 0.0

    base_steepness = H / L0
    wind_factor = 1.0 + gamma_w * U_wind * np.cos(phi) / C

    return base_steepness * wind_factor


# =============================================================================
# Breaking Criteria
# =============================================================================

@njit(cache=True)
def breaker_index_mcowan() -> float:
    """
    McCowan (1894) constant breaker index.

    γb = 0.78

    Returns:
        Breaker index (dimensionless)
    """
    return 0.78


@njit(cache=True)
def breaker_index_rattanapitikon(H0_L0: float, m: float) -> float:
    """
    Rattanapitikon & Shibayama (2000) breaker index.

    γb = 0.57 + 0.71 · (H₀/L₀)^0.12 · m^0.36

    Args:
        H0_L0: Deep water wave steepness H₀/L₀
        m: Beach slope (rise/run)

    Returns:
        Breaker index (dimensionless)
    """
    if H0_L0 <= 0 or m <= 0:
        return 0.78  # Fallback to McCowan

    return 0.57 + 0.71 * (H0_L0 ** 0.12) * (m ** 0.36)


@njit(cache=True)
def breaker_index_weggel(H: float, T: float, m: float) -> float:
    """
    Weggel (1972) breaker index with slope dependence.

    γb = b - a·(H / gT²)

    where:
        a = 43.8 · (1 - e^(-19m))
        b = 1.56 / (1 + e^(-19.5m))

    Args:
        H: Wave height at breaking (m)
        T: Wave period (s)
        m: Beach slope

    Returns:
        Breaker index (dimensionless)
    """
    if m <= 0:
        return 0.78

    a = 43.8 * (1.0 - np.exp(-19.0 * m))
    b = 1.56 / (1.0 + np.exp(-19.5 * m))

    H_gT2 = H / (G * T * T)

    gamma_b = b - a * H_gT2

    # Ensure reasonable bounds
    return max(0.5, min(1.5, gamma_b))


@njit(cache=True)
def breaker_index_wind_modified(
    gamma_b0: float,
    U_wind: float,
    phi: float,
    C: float,
    Cw: float = 0.15,
) -> float:
    """
    Wind-modified breaker index (Douglass, 1990).

    γb = γb,0 · (1 - Cw · Uᵥ·cos(φ) / C)

    Onshore winds (cos(φ) > 0): Decreases γb → earlier breaking
    Offshore winds (cos(φ) < 0): Increases γb → later breaking

    Args:
        gamma_b0: No-wind breaker index
        U_wind: Wind speed (m/s)
        phi: Relative wind-wave angle (radians)
        C: Local celerity (m/s)
        Cw: Wind modification coefficient (default 0.15)

    Returns:
        Wind-modified breaker index
    """
    if C <= 0:
        return gamma_b0

    wind_factor = 1.0 - Cw * U_wind * np.cos(phi) / C

    # Ensure reasonable bounds
    return max(0.5, min(1.5, gamma_b0 * wind_factor))


@njit(cache=True)
def check_breaking(H: float, h: float, gamma_b: float) -> bool:
    """
    Check if wave is breaking.

    Breaking occurs when H ≥ γb · h

    Args:
        H: Local wave height (m)
        h: Local water depth (m)
        gamma_b: Breaker index

    Returns:
        True if wave is breaking
    """
    # Don't classify hitting shore/land as "breaking"
    # Let the ray tracer classify this as "reached shore" instead
    if h <= 0.05:
        return False
    return H >= gamma_b * h


# =============================================================================
# Iribarren Number and Breaker Classification
# =============================================================================

@njit(cache=True)
def iribarren_number(m: float, H: float, L0: float) -> float:
    """
    Calculate Iribarren number (surf similarity parameter).

    ξ = m / √(H/L₀)

    Args:
        m: Beach slope
        H: Wave height (m)
        L0: Deep water wavelength (m)

    Returns:
        Iribarren number ξ (dimensionless)
    """
    if H <= 0 or L0 <= 0:
        return 0.0

    steepness = H / L0
    if steepness <= 0:
        return 0.0

    return m / np.sqrt(steepness)


@njit(cache=True)
def classify_breaker_type(xi: float) -> int:
    """
    Classify breaker type from Iribarren number.

    Args:
        xi: Iribarren number

    Returns:
        Breaker type code:
        0 = Spilling (ξ < 0.5)
        1 = Plunging (0.5 ≤ ξ < 3.3)
        2 = Collapsing (3.3 ≤ ξ < 5.0)
        3 = Surging (ξ ≥ 5.0)
    """
    if xi < 0.5:
        return 0  # Spilling
    elif xi < 3.3:
        return 1  # Plunging
    elif xi < 5.0:
        return 2  # Collapsing
    else:
        return 3  # Surging


# Breaker type labels (for use outside Numba)
BREAKER_TYPE_LABELS = {
    0: "Spilling",
    1: "Plunging",
    2: "Collapsing",
    3: "Surging",
}


# =============================================================================
# Beach Slope Calculation
# =============================================================================

@njit(cache=True)
def calculate_slope(dh: float, dx: float) -> float:
    """
    Calculate beach slope.

    m = Δh / Δx

    Args:
        dh: Change in depth (m)
        dx: Horizontal distance (m)

    Returns:
        Beach slope m (dimensionless, positive shoreward)
    """
    if dx <= 0:
        return 0.0
    return abs(dh) / dx


# =============================================================================
# Direction Conversions
# =============================================================================

@njit(cache=True)
def nautical_to_math(nautical_deg: float) -> float:
    """
    Convert nautical direction (FROM, clockwise from N) to math angle (radians).

    Math convention: 0 = East, counter-clockwise positive

    Args:
        nautical_deg: Direction in degrees (FROM, 0=N, 90=E)

    Returns:
        Angle in radians (math convention)
    """
    # Nautical: 0=N, 90=E, 180=S, 270=W (clockwise from N, direction FROM)
    # Math: 0=E, π/2=N, π=W, 3π/2=S (counter-clockwise from E)
    # Wave travels TOWARD = 180° opposite to FROM direction
    travel_deg = (nautical_deg + 180.0) % 360.0
    math_deg = (90.0 - travel_deg) % 360.0
    return np.radians(math_deg)


@njit(cache=True)
def math_to_nautical(math_rad: float) -> float:
    """
    Convert math angle (radians) to nautical direction (degrees, FROM).

    Args:
        math_rad: Angle in radians (math convention)

    Returns:
        Direction in degrees (nautical FROM convention)
    """
    math_deg = np.degrees(math_rad)
    travel_deg = (90.0 - math_deg) % 360.0
    from_deg = (travel_deg + 180.0) % 360.0
    return from_deg


# =============================================================================
# Ray Direction Update (for refraction)
# =============================================================================

@njit(cache=True)
def update_ray_direction(
    dx: float,
    dy: float,
    C: float,
    dC_dx: float,
    dC_dy: float,
    ds: float,
) -> Tuple[float, float]:
    """
    Update ray direction due to refraction.

    The ray bends toward regions of slower wave speed (shallower water).

    Uses the standard ray equation:
        dθ/ds = -(1/C) · ∂C/∂n

    where ∂C/∂n is the celerity gradient perpendicular to the ray direction.
    The perpendicular direction (pointing left of ray) is (-dy, dx).

    So: ∂C/∂n = -dy · ∂C/∂x + dx · ∂C/∂y

    The NEGATIVE sign is critical: waves refract toward slower celerity
    (shallower water). If C decreases to the left (dC_dn < 0), the ray
    should turn left (dtheta > 0).

    Args:
        dx: Current x-direction component (normalized)
        dy: Current y-direction component (normalized)
        C: Local celerity (m/s)
        dC_dx: Celerity gradient in x (m/s per m)
        dC_dy: Celerity gradient in y (m/s per m)
        ds: Step distance (m)

    Returns:
        Updated (dx, dy) direction components (normalized)
    """
    if C <= 0:
        return dx, dy

    # Gradient perpendicular to ray direction
    # Perpendicular (left-hand normal) is (-dy, dx)
    dC_dn = -dy * dC_dx + dx * dC_dy

    # Angular change - NEGATIVE sign is critical for correct refraction!
    # Waves bend toward slower celerity (shallower water)
    dtheta = -(ds / C) * dC_dn

    # Current angle
    theta = np.arctan2(dy, dx)

    # Update angle
    theta_new = theta + dtheta

    # New direction components
    new_dx = np.cos(theta_new)
    new_dy = np.sin(theta_new)

    return new_dx, new_dy


# =============================================================================
# Vectorized Batch Operations
# =============================================================================

@njit(parallel=True, cache=True)
def batch_deep_water_properties(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate deep water properties for an array of periods.

    Args:
        T: Array of wave periods (s)

    Returns:
        Tuple of (L0, C0, Cg0) arrays
    """
    n = len(T)
    L0 = np.empty(n, dtype=np.float64)
    C0 = np.empty(n, dtype=np.float64)
    Cg0 = np.empty(n, dtype=np.float64)

    for i in prange(n):
        L0[i], C0[i], Cg0[i] = deep_water_properties(T[i])

    return L0, C0, Cg0


@njit(parallel=True, cache=True)
def batch_local_wavelength(L0: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Calculate local wavelength for arrays of L0 and depth.

    Args:
        L0: Array of deep water wavelengths (m)
        h: Array of water depths (m)

    Returns:
        Array of local wavelengths (m)
    """
    n = len(L0)
    L = np.empty(n, dtype=np.float64)

    for i in prange(n):
        L[i] = local_wavelength_fenton_mckee(L0[i], h[i])

    return L
