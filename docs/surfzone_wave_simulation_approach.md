# Surfzone Wave-by-Wave Simulation Approach

## Reference Document

This document describes the approach for simulating individual wave propagation through the surfzone to produce aerial maps of breaking probability, breaking frequency, and breaker type along the California coast.

---

## Table of Contents

1. [Goals and Outputs](#1-goals-and-outputs)
2. [Physical Concepts](#2-physical-concepts)
3. [Algorithm Overview](#3-algorithm-overview)
4. [Key Equations](#4-key-equations)
5. [Data Structures](#5-data-structures)
6. [Implementation Details](#6-implementation-details)
7. [Computational Cost](#7-computational-cost)
8. [Optimizations](#8-optimizations)
9. [Output Visualization](#9-output-visualization)

---

## 1. Goals and Outputs

### Primary Outputs

1. **Breaking Probability Map**: At each mesh cell, the probability that an arriving wave will break (0-1)

2. **Breaking Frequency Map**: Number of breaking waves per unit time at each cell (waves/hour)

3. **Breaker Type Map**: Classification of dominant breaker type at each cell:
   - Spilling (gentle, crumbly)
   - Plunging (hollow, barreling)
   - Surging (steep beaches, wave surges up)

4. **Breaking Height Distribution**: Statistical distribution of breaking wave heights at each cell

### Secondary Outputs

5. **Set Statistics** (derived from spectral parameters, no simulation needed):
   - Mean waves per set
   - Mean time between sets
   - Set height enhancement factor

6. **Confidence/Coverage Map**: Cells with good ray coverage vs. shadow zones

---

## 2. Physical Concepts

### 2.1 Wave Crest Width

Individual wave crests have finite alongshore extent, determined by the directional spread of the wave spectrum.

```
L_crest = λ / (2π × σ_θ)
```

Where:
- `L_crest` = crest width (meters)
- `λ` = wavelength (meters)
- `σ_θ` = directional spread (radians)

**Typical values:**

| Wave Type | Period | λ (deep) | σ_θ | L_crest |
|-----------|--------|----------|-----|---------|
| Clean groundswell | 15-18s | 350-500m | 5-10° | 400-1000m |
| Average swell | 10-14s | 150-300m | 10-20° | 100-300m |
| Wind swell | 6-9s | 50-120m | 25-40° | 30-80m |

**Physical meaning**: Long crests (swell) break uniformly alongshore. Short crests (wind waves) create sectioning, choppy conditions.

### 2.2 Wave Groupiness (Sets)

Waves arrive in groups (sets) due to the narrow-banded nature of ocean spectra. Kimura (1980) showed that successive wave heights form a Markov chain.

**Key parameters:**

- **Spectral peakedness**: Controls mean group length (waves per set)
- **Spectral width (ν)**: Controls period consistency
- **Correlation coefficient (γ_s)**: Correlation between successive wave heights

**Mean group length** (waves per set):

```
J̄ ≈ π / (2 × arccos(γ_s))
```

For typical swell (γ_s ≈ 0.7): J̄ ≈ 4-5 waves per set

**Set enhancement factor**: Waves in sets are typically 1.2-1.5× the mean Hs

### 2.3 Wave Height Distribution

Individual wave heights follow a **Rayleigh distribution** for a single partition:

```
P(H > h) = exp(-2 × (h/Hs)²)
```

For **multiple partitions** (crossed seas), the distribution has heavier tails. Use enhancement factor:

```
κ = 1 + 0.1 × (n_partitions - 1) × (1 - cos(Δθ))
```

Where Δθ is angular separation between partitions.

### 2.4 Partition Interference

When waves from multiple partitions arrive at the same location:

**Surface elevation superposition** (linear):
```
η_total(t) = η_partition1(t) + η_partition2(t) + ...
```

**Combined wave height** depends on phase relationship:
- In phase (constructive): H_combined ≈ H_A + H_B
- Out of phase (destructive): H_combined ≈ |H_A - H_B|
- Random phase (statistical): H_combined ≈ √(H_A² + H_B²)

For uncorrelated partitions, use RMS combination:
```
Hs_combined = √(Hs_1² + Hs_2² + ... + Hs_n²)
```

### 2.5 Breaking Criterion

Waves break when height exceeds a depth-dependent threshold.

**Basic criterion**:
```
H/d ≥ γ_b
```

Where γ_b ≈ 0.78 (McCowan) or varies with slope (Battjes).

**Wind-modified breaking index**:
```
γ_b = γ_b,0 × (1 - C_w × U_w × cos(φ) / C)
```

Where:
- `γ_b,0` = base breaking index (~0.78)
- `C_w` = wind coefficient (0.1-0.2)
- `U_w` = wind speed (m/s)
- `φ` = angle between wind and wave direction
- `C` = wave celerity at breaking depth

**Physical effect**:
- Onshore wind (cos(φ) > 0): Waves break earlier (deeper water)
- Offshore wind (cos(φ) < 0): Waves hold up, break later (shallower water)

### 2.6 Breaker Type Classification

Determined by **Iribarren number** (surf similarity parameter):

```
ξ = tan(β) / √(H/L_0)
```

Where:
- `tan(β)` = beach slope
- `H` = wave height
- `L_0` = deep water wavelength = 1.56 × T²

**Classification**:
| ξ Range | Breaker Type | Description |
|---------|--------------|-------------|
| ξ < 0.5 | Spilling | Gentle tumbling, foam, long surf zone |
| 0.5 ≤ ξ < 3.3 | Plunging | Curling, hollow, classic "barrel" |
| ξ ≥ 3.3 | Surging | Wave surges up steep beach, minimal breaking |

---

## 3. Algorithm Overview

### 3.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUTS                                                          │
│  - SWAN partition outputs (Hs, Tp, direction, spread per partition)│
│  - Surfzone mesh (bathymetry, coastlines)                        │
│  - Wind data (speed, direction)                                  │
│  - Simulation parameters (duration, resolution)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: PRE-COMPUTE RAY PATHS                                  │
│  For each partition:                                             │
│    - Calculate crest width from directional spread               │
│    - Generate boundary points spaced by crest width              │
│    - Trace ray from each boundary point to shore                 │
│    - Store path (x, y, depth, travel_time, cells_affected)       │
│                                                                  │
│  Output: Ray templates (paths are height-independent)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: GENERATE WAVE TIME SERIES                              │
│  For each partition:                                             │
│    - Calculate groupiness parameters from spectrum               │
│    - Generate N wave heights using Markov chain                  │
│    - N = simulation_time / period                                │
│                                                                  │
│  Output: Wave heights with temporal correlation (sets)           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: PROPAGATE WAVES AND ACCUMULATE                         │
│  For each partition:                                             │
│    For each time step (wave):                                    │
│      For each ray template:                                      │
│        - Assign wave height from time series                     │
│        - Record arrival at affected cells with:                  │
│          - Arrival time = start_time + travel_time               │
│          - Wave height (shoaled)                                 │
│          - Partition ID                                          │
│                                                                  │
│  Output: Per-cell list of wave arrivals                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: COMPUTE BREAKING STATISTICS                            │
│  For each cell:                                                  │
│    - Sort arrivals by time                                       │
│    - Identify overlapping waves from different partitions        │
│    - Combine overlapping waves (RMS or phase-aware)              │
│    - Apply wind-modified breaking criterion                      │
│    - Classify breaker type                                       │
│    - Accumulate statistics                                       │
│                                                                  │
│  Output: Breaking probability, frequency, type, height dist.    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUTS                                                         │
│  - Aerial breaking probability map                               │
│  - Breaker type map                                              │
│  - Breaking frequency (waves/hour)                               │
│  - Height distribution of breaking waves                         │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Design Decisions

1. **Pre-compute ray paths**: Ray trajectories depend on bathymetry and wave period, NOT wave height. Trace once per partition, reuse for all waves.

2. **Influence radius = crest width / 2**: Each ray affects cells within half-crest-width distance.

3. **One ray per crest width at boundary**: Avoids double-counting energy.

4. **Markov chain for wave heights**: Captures realistic groupiness/set structure.

5. **RMS combination for partition interference**: Assumes random phase between partitions.

---

## 4. Key Equations

### 4.1 Crest Width from Directional Spread

```python
def compute_crest_width(wavelength: float, directional_spread_deg: float) -> float:
    """
    Calculate wave crest width from wavelength and directional spread.

    Args:
        wavelength: Deep water wavelength (m)
        directional_spread_deg: Directional spread in degrees (standard deviation)

    Returns:
        Crest width in meters
    """
    sigma_theta = np.radians(directional_spread_deg)
    if sigma_theta < 0.01:  # Avoid division by zero for very narrow spread
        sigma_theta = 0.01
    return wavelength / (2 * np.pi * sigma_theta)
```

### 4.2 Wavelength from Period and Depth

```python
def compute_wavelength(period: float, depth: float) -> float:
    """
    Compute wavelength using dispersion relation.

    Args:
        period: Wave period (s)
        depth: Water depth (m)

    Returns:
        Wavelength (m)
    """
    g = 9.81
    L_deep = g * period**2 / (2 * np.pi)  # Deep water wavelength

    # Iterative solution for intermediate/shallow water
    L = L_deep
    for _ in range(20):
        L = L_deep * np.tanh(2 * np.pi * depth / L)

    return L
```

### 4.3 Wave Celerity

```python
def compute_celerity(period: float, depth: float) -> float:
    """
    Compute wave phase velocity.

    Args:
        period: Wave period (s)
        depth: Water depth (m)

    Returns:
        Wave celerity (m/s)
    """
    wavelength = compute_wavelength(period, depth)
    return wavelength / period
```

### 4.4 Shoaling Coefficient

```python
def compute_shoaling_coefficient(period: float, depth: float, depth_ref: float = 100.0) -> float:
    """
    Compute shoaling coefficient (Ks) relative to reference depth.

    Args:
        period: Wave period (s)
        depth: Local water depth (m)
        depth_ref: Reference depth for normalization (m)

    Returns:
        Shoaling coefficient Ks
    """
    g = 9.81

    def group_velocity(d):
        L = compute_wavelength(period, d)
        k = 2 * np.pi / L
        C = L / period
        n = 0.5 * (1 + 2 * k * d / np.sinh(2 * k * d))
        return n * C

    Cg_ref = group_velocity(depth_ref)
    Cg_local = group_velocity(depth)

    return np.sqrt(Cg_ref / Cg_local)
```

### 4.5 Wind-Modified Breaking Index

```python
def compute_breaking_index(
    wave_direction: float,
    wind_speed: float,
    wind_direction: float,
    wave_celerity: float,
    gamma_base: float = 0.78,
    C_wind: float = 0.15
) -> float:
    """
    Compute wind-modified breaker index.

    Args:
        wave_direction: Wave propagation direction (degrees, nautical convention)
        wind_speed: Wind speed (m/s)
        wind_direction: Wind direction (degrees, nautical - direction FROM)
        wave_celerity: Wave phase speed at breaking (m/s)
        gamma_base: Base breaking index (default 0.78)
        C_wind: Wind coefficient (0.1-0.2)

    Returns:
        Modified breaking index gamma_b
    """
    # Angle between wind and wave propagation
    # Wind is FROM direction, wave is TO direction
    phi = np.radians(wind_direction - wave_direction)

    # Wind modification factor
    # Positive when wind opposes waves (offshore wind) -> larger gamma -> breaks later
    # Negative when wind follows waves (onshore wind) -> smaller gamma -> breaks earlier
    wind_factor = 1 - C_wind * wind_speed * np.cos(phi) / wave_celerity

    # Clamp to reasonable range
    wind_factor = np.clip(wind_factor, 0.7, 1.3)

    return gamma_base * wind_factor
```

### 4.6 Breaking Probability (Statistical)

```python
def compute_breaking_probability(
    Hs: float,
    depth: float,
    gamma_b: float,
    kappa: float = 1.0
) -> float:
    """
    Compute probability that a wave breaks at this depth.

    Based on Rayleigh distribution of wave heights.

    Args:
        Hs: Significant wave height (m)
        depth: Water depth (m)
        gamma_b: Breaking index (wind-modified)
        kappa: Crossed-seas enhancement factor

    Returns:
        Breaking probability (0-1)
    """
    H_break = gamma_b * depth  # Minimum height to break here

    # Rayleigh exceedance probability with crossed-seas enhancement
    P_break = np.exp(-2 * (H_break / (kappa * Hs))**2)

    return P_break
```

### 4.7 Iribarren Number and Breaker Type

```python
def compute_iribarren(slope: float, wave_height: float, period: float) -> float:
    """
    Compute Iribarren number (surf similarity parameter).

    Args:
        slope: Beach slope (tan(beta), dimensionless)
        wave_height: Wave height (m)
        period: Wave period (s)

    Returns:
        Iribarren number
    """
    L_0 = 1.56 * period**2  # Deep water wavelength
    return slope / np.sqrt(wave_height / L_0)


def classify_breaker(iribarren: float) -> str:
    """
    Classify breaker type from Iribarren number.

    Args:
        iribarren: Iribarren number (surf similarity parameter)

    Returns:
        Breaker type: 'spilling', 'plunging', or 'surging'
    """
    if iribarren < 0.5:
        return 'spilling'
    elif iribarren < 3.3:
        return 'plunging'
    else:
        return 'surging'
```

### 4.8 Markov Chain Wave Height Generation

```python
def generate_wave_heights_markov(
    Hs: float,
    n_waves: int,
    correlation: float = 0.7
) -> np.ndarray:
    """
    Generate correlated wave heights using Markov chain.

    Captures realistic wave groupiness (sets).

    Args:
        Hs: Significant wave height (m)
        n_waves: Number of waves to generate
        correlation: Correlation coefficient between successive waves (0-1)
                    Higher = longer groups. Typical swell: 0.6-0.8

    Returns:
        Array of individual wave heights
    """
    # Rayleigh scale parameter
    sigma = Hs / (2 * np.sqrt(2 * np.log(2)))  # Mode of Rayleigh = Hs/1.416

    heights = np.zeros(n_waves)
    heights[0] = np.random.rayleigh(sigma)

    for i in range(1, n_waves):
        # Conditional distribution: given previous height, next height is
        # correlated Rayleigh
        # Using a simple autoregressive approximation:
        # H_i = correlation * H_{i-1} + (1-correlation) * Rayleigh_sample

        # Scale to maintain correct marginal distribution
        innovation_scale = sigma * np.sqrt(1 - correlation**2)
        innovation = np.random.rayleigh(innovation_scale)

        heights[i] = correlation * heights[i-1] + (1 - correlation) * np.random.rayleigh(sigma)

        # Ensure positive
        heights[i] = max(heights[i], 0.01)

    return heights
```

### 4.9 Combined Wave Height from Multiple Partitions

```python
def combine_wave_heights(
    heights: List[float],
    method: str = 'rms'
) -> float:
    """
    Combine wave heights from multiple partitions.

    Args:
        heights: List of wave heights from each partition
        method: Combination method:
                'rms' - Root mean square (random phase assumption)
                'linear' - Direct sum (perfect constructive interference)
                'max' - Maximum of all (conservative)

    Returns:
        Combined wave height
    """
    heights = np.array(heights)

    if method == 'rms':
        return np.sqrt(np.sum(heights**2))
    elif method == 'linear':
        return np.sum(heights)
    elif method == 'max':
        return np.max(heights)
    else:
        raise ValueError(f"Unknown method: {method}")
```

### 4.10 Groupiness Statistics from Spectrum

```python
def compute_groupiness_parameters(
    spectrum_peakedness: float,
    spectral_width: float
) -> dict:
    """
    Compute wave groupiness parameters from spectral shape.

    Args:
        spectrum_peakedness: JONSWAP gamma or similar (typically 1-7)
        spectral_width: Spectral width parameter nu (typically 0.3-0.6)

    Returns:
        Dictionary with groupiness parameters
    """
    # Correlation coefficient (Kimura 1980)
    # Higher peakedness -> higher correlation -> longer groups
    gamma_s = 0.5 + 0.05 * spectrum_peakedness  # Approximate relationship
    gamma_s = np.clip(gamma_s, 0.3, 0.9)

    # Mean group length (waves per set)
    if gamma_s < 1:
        J_mean = np.pi / (2 * np.arccos(gamma_s))
    else:
        J_mean = 10  # Very long groups

    # Standard deviation of group length
    J_std = J_mean * 0.5  # Approximate

    return {
        'correlation': gamma_s,
        'mean_waves_per_set': J_mean,
        'std_waves_per_set': J_std,
        'spectral_width': spectral_width,
    }
```

---

## 5. Data Structures

### 5.1 Partition Input

```python
@dataclass
class SwanPartition:
    """Wave partition from SWAN output."""
    id: int
    Hs: float              # Significant wave height (m)
    Tp: float              # Peak period (s)
    direction: float       # Mean direction (degrees, nautical)
    directional_spread: float  # Directional spread (degrees)

    # Derived quantities
    @property
    def wavelength_deep(self) -> float:
        return 1.56 * self.Tp**2

    @property
    def crest_width(self) -> float:
        sigma = np.radians(self.directional_spread)
        return self.wavelength_deep / (2 * np.pi * max(sigma, 0.01))

    @property
    def groupiness(self) -> float:
        # Estimate from peakedness (would come from spectrum)
        return 0.7  # Default for swell
```

### 5.2 Ray Template

```python
@dataclass
class RayTemplate:
    """Pre-computed ray path for a partition."""
    partition_id: int
    boundary_x: float
    boundary_y: float
    crest_width: float

    # Path: list of steps from boundary to termination
    path: List[RayStep]

    # Pre-computed: cells affected by this ray
    cells_affected: List[Tuple[int, float, float]]  # (cell_idx, travel_time, distance)


@dataclass
class RayStep:
    """Single step along a ray path."""
    x: float
    y: float
    depth: float
    direction: float       # Current propagation direction
    travel_time: float     # Cumulative travel time from boundary
    shoaling_coef: float   # Cumulative shoaling coefficient
```

### 5.3 Wave Event

```python
@dataclass
class WaveEvent:
    """Individual wave arriving at a cell."""
    partition_id: int
    arrival_time: float    # Seconds from simulation start
    height: float          # Wave height at this location (shoaled)
    period: float
    direction: float
```

### 5.4 Cell Accumulator

```python
class CellAccumulator:
    """Accumulates wave arrivals at a single mesh cell."""

    def __init__(self, cell_idx: int, depth: float, slope: float):
        self.cell_idx = cell_idx
        self.depth = depth
        self.slope = slope
        self.arrivals: List[WaveEvent] = []

    def add_arrival(self, event: WaveEvent):
        self.arrivals.append(event)

    def compute_statistics(
        self,
        wind_speed: float,
        wind_direction: float
    ) -> 'CellResult':
        """Compute breaking statistics from accumulated arrivals."""
        if not self.arrivals:
            return CellResult.empty(self.cell_idx)

        # Sort by arrival time
        self.arrivals.sort(key=lambda e: e.arrival_time)

        # Combine overlapping waves and apply breaking criterion
        breaking_events = []
        all_events = []

        combined_events = self._combine_overlapping_waves()

        for event in combined_events:
            # Wind-modified breaking index
            C = compute_celerity(event.period, self.depth)
            gamma_b = compute_breaking_index(
                event.direction, wind_speed, wind_direction, C
            )

            H_break = gamma_b * self.depth

            all_events.append(event)
            if event.height >= H_break:
                breaking_events.append(event)

        # Compute statistics
        n_total = len(all_events)
        n_breaking = len(breaking_events)

        return CellResult(
            cell_idx=self.cell_idx,
            n_waves=n_total,
            n_breaking=n_breaking,
            P_break=n_breaking / n_total if n_total > 0 else 0,
            breaking_heights=[e.height for e in breaking_events],
            breaker_type=self._classify_dominant_breaker(breaking_events),
        )

    def _combine_overlapping_waves(self) -> List[WaveEvent]:
        """Combine waves that arrive within half-period of each other."""
        if not self.arrivals:
            return []

        combined = []
        current_group = [self.arrivals[0]]

        for event in self.arrivals[1:]:
            # Check if overlaps with current group
            time_gap = event.arrival_time - current_group[-1].arrival_time
            overlap_threshold = min(e.period for e in current_group) / 2

            if time_gap < overlap_threshold:
                current_group.append(event)
            else:
                # Finalize current group
                combined.append(self._merge_group(current_group))
                current_group = [event]

        # Don't forget last group
        combined.append(self._merge_group(current_group))

        return combined

    def _merge_group(self, group: List[WaveEvent]) -> WaveEvent:
        """Merge overlapping waves using RMS combination."""
        if len(group) == 1:
            return group[0]

        combined_height = np.sqrt(sum(e.height**2 for e in group))

        # Use energy-weighted average for other properties
        total_energy = sum(e.height**2 for e in group)
        avg_period = sum(e.height**2 * e.period for e in group) / total_energy
        avg_direction = self._circular_mean(
            [e.direction for e in group],
            [e.height**2 for e in group]
        )

        return WaveEvent(
            partition_id=-1,  # Mixed
            arrival_time=group[0].arrival_time,
            height=combined_height,
            period=avg_period,
            direction=avg_direction,
        )

    def _classify_dominant_breaker(self, events: List[WaveEvent]) -> str:
        """Classify dominant breaker type from breaking events."""
        if not events:
            return 'none'

        types = []
        for e in events:
            xi = compute_iribarren(self.slope, e.height, e.period)
            types.append(classify_breaker(xi))

        # Return most common
        from collections import Counter
        return Counter(types).most_common(1)[0][0]

    @staticmethod
    def _circular_mean(angles: List[float], weights: List[float]) -> float:
        """Compute weighted circular mean of angles in degrees."""
        angles_rad = np.radians(angles)
        weights = np.array(weights)

        x = np.sum(weights * np.cos(angles_rad))
        y = np.sum(weights * np.sin(angles_rad))

        return np.degrees(np.arctan2(y, x)) % 360
```

### 5.5 Cell Result

```python
@dataclass
class CellResult:
    """Breaking statistics for a single mesh cell."""
    cell_idx: int
    n_waves: int           # Total waves arriving
    n_breaking: int        # Number that broke
    P_break: float         # Breaking probability (0-1)
    breaking_heights: List[float]  # Heights of breaking waves
    breaker_type: str      # Dominant type: 'spilling', 'plunging', 'surging', 'none'

    @property
    def breaking_frequency(self) -> float:
        """Breaking waves per hour (assuming 1-hour simulation)."""
        return self.n_breaking  # If simulation_time != 1 hour, scale accordingly

    @property
    def mean_breaking_height(self) -> float:
        if not self.breaking_heights:
            return 0.0
        return np.mean(self.breaking_heights)

    @property
    def max_breaking_height(self) -> float:
        if not self.breaking_heights:
            return 0.0
        return np.max(self.breaking_heights)

    @classmethod
    def empty(cls, cell_idx: int) -> 'CellResult':
        return cls(
            cell_idx=cell_idx,
            n_waves=0,
            n_breaking=0,
            P_break=np.nan,
            breaking_heights=[],
            breaker_type='none',
        )
```

### 5.6 Breaking Field Output

```python
@dataclass
class BreakingField:
    """Complete breaking field for the surfzone."""
    # Mesh coordinates
    x: np.ndarray
    y: np.ndarray
    lon: np.ndarray
    lat: np.ndarray

    # Breaking statistics at each cell
    P_break: np.ndarray           # Breaking probability (0-1)
    breaking_frequency: np.ndarray # Waves breaking per hour
    breaker_type: np.ndarray      # String array: 'spilling', 'plunging', 'surging'
    mean_breaking_height: np.ndarray
    max_breaking_height: np.ndarray

    # Metadata
    simulation_time: float        # Seconds
    n_partitions: int
    wind_speed: float
    wind_direction: float

    # Coverage quality
    ray_coverage: np.ndarray      # Number of rays affecting each cell
```

---

## 6. Implementation Details

### 6.1 Phase 1: Pre-compute Ray Paths

```python
def precompute_ray_templates(
    mesh: SurfZoneMesh,
    partitions: List[SwanPartition],
    influence_radius_factor: float = 0.5,
) -> Dict[int, List[RayTemplate]]:
    """
    Pre-compute ray paths for all partitions.

    Ray paths depend only on bathymetry and wave period, not height.
    """
    templates = {}

    for partition in partitions:
        crest_width = partition.crest_width
        influence_radius = crest_width * influence_radius_factor

        # Generate boundary points spaced by crest width
        boundary_points = generate_boundary_points(
            mesh=mesh,
            spacing=crest_width,
            offshore_distance=2000,  # meters from shore
        )

        partition_templates = []

        for bx, by in boundary_points:
            # Trace ray
            ray_path = trace_ray(
                mesh=mesh,
                start_x=bx,
                start_y=by,
                period=partition.Tp,
                direction=partition.direction,
            )

            # Find cells affected by this ray
            cells_affected = find_cells_along_ray(
                mesh=mesh,
                ray_path=ray_path,
                influence_radius=influence_radius,
            )

            template = RayTemplate(
                partition_id=partition.id,
                boundary_x=bx,
                boundary_y=by,
                crest_width=crest_width,
                path=ray_path,
                cells_affected=cells_affected,
            )
            partition_templates.append(template)

        templates[partition.id] = partition_templates

    return templates
```

### 6.2 Phase 2: Generate Wave Time Series

```python
def generate_wave_time_series(
    partitions: List[SwanPartition],
    simulation_time: float,
) -> Dict[int, np.ndarray]:
    """
    Generate wave-by-wave height time series for each partition.
    """
    time_series = {}

    for partition in partitions:
        n_waves = int(simulation_time / partition.Tp)

        # Get groupiness parameters
        groupiness = compute_groupiness_parameters(
            spectrum_peakedness=3.3,  # Would come from SWAN spectrum
            spectral_width=0.4,
        )

        # Generate heights with temporal correlation
        heights = generate_wave_heights_markov(
            Hs=partition.Hs,
            n_waves=n_waves,
            correlation=groupiness['correlation'],
        )

        time_series[partition.id] = heights

    return time_series
```

### 6.3 Phase 3: Propagate and Accumulate

```python
def propagate_waves(
    mesh: SurfZoneMesh,
    ray_templates: Dict[int, List[RayTemplate]],
    wave_time_series: Dict[int, np.ndarray],
    partitions: List[SwanPartition],
) -> List[CellAccumulator]:
    """
    Propagate all waves and accumulate arrivals at mesh cells.
    """
    # Initialize accumulators for each cell
    accumulators = [
        CellAccumulator(
            cell_idx=i,
            depth=-mesh.elevation[i],
            slope=mesh.slope[i],
        )
        for i in range(len(mesh.points_x))
    ]

    for partition in partitions:
        templates = ray_templates[partition.id]
        heights = wave_time_series[partition.id]

        # For each wave (time step)
        for wave_idx, height in enumerate(heights):
            start_time = wave_idx * partition.Tp

            # For each ray template
            for template in templates:
                # Apply shoaling to wave height along path
                # (or use pre-computed shoaling at each cell)

                # Record arrivals at affected cells
                for cell_idx, travel_time, distance in template.cells_affected:
                    # Get shoaling coefficient at this cell
                    depth = -mesh.elevation[cell_idx]
                    Ks = compute_shoaling_coefficient(partition.Tp, depth)

                    # Shoaled wave height
                    H_local = height * Ks

                    # Weight by distance from ray center (optional)
                    # Could use Gaussian weighting based on distance

                    event = WaveEvent(
                        partition_id=partition.id,
                        arrival_time=start_time + travel_time,
                        height=H_local,
                        period=partition.Tp,
                        direction=partition.direction,  # Could evolve along ray
                    )

                    accumulators[cell_idx].add_arrival(event)

    return accumulators
```

### 6.4 Phase 4: Compute Breaking Statistics

```python
def compute_breaking_field(
    mesh: SurfZoneMesh,
    accumulators: List[CellAccumulator],
    wind_speed: float,
    wind_direction: float,
    simulation_time: float,
) -> BreakingField:
    """
    Compute breaking statistics at all cells.
    """
    n_cells = len(mesh.points_x)

    P_break = np.zeros(n_cells)
    breaking_freq = np.zeros(n_cells)
    breaker_type = np.empty(n_cells, dtype='U10')
    mean_H = np.zeros(n_cells)
    max_H = np.zeros(n_cells)
    ray_coverage = np.zeros(n_cells, dtype=int)

    for acc in accumulators:
        i = acc.cell_idx
        ray_coverage[i] = len(acc.arrivals)

        result = acc.compute_statistics(wind_speed, wind_direction)

        P_break[i] = result.P_break
        breaking_freq[i] = result.n_breaking * (3600 / simulation_time)  # per hour
        breaker_type[i] = result.breaker_type
        mean_H[i] = result.mean_breaking_height
        max_H[i] = result.max_breaking_height

    return BreakingField(
        x=mesh.points_x,
        y=mesh.points_y,
        lon=mesh.lon,
        lat=mesh.lat,
        P_break=P_break,
        breaking_frequency=breaking_freq,
        breaker_type=breaker_type,
        mean_breaking_height=mean_H,
        max_breaking_height=max_H,
        simulation_time=simulation_time,
        n_partitions=len(set(a.arrivals[0].partition_id for a in accumulators if a.arrivals)),
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        ray_coverage=ray_coverage,
    )
```

---

## 7. Computational Cost

### 7.1 Cost Breakdown by Phase

**Configuration assumptions:**
- Boundary length: 300 km
- Simulation time: 1 hour (3600s)
- Mesh resolution: 50m (yielding ~60,000 cells in surfzone)
- 4 partitions with varying crest widths

**Partition breakdown:**

| Partition | Period | Crest Width | Boundary Points | Waves | Total Rays |
|-----------|--------|-------------|-----------------|-------|------------|
| NW swell | 15s | 400m | 750 | 240 | 180,000 |
| W swell | 12s | 250m | 1,200 | 300 | 360,000 |
| SW swell | 10s | 150m | 2,000 | 360 | 720,000 |
| Wind | 7s | 50m | 6,000 | 514 | 3,084,000 |
| **Total** | | | | | **4,344,000** |

### 7.2 Phase 1: Pre-compute Ray Paths (ONE TIME)

```
Unique ray paths to trace: 750 + 1,200 + 2,000 + 6,000 = 9,950
Steps per ray: ~200
Total steps: 9,950 × 200 = 1,990,000
Time per step: ~15 μs

Pre-computation time: ~30 seconds
```

This is done ONCE and reused for all waves.

### 7.3 Phase 2: Generate Wave Time Series

```
Total waves: 240 + 300 + 360 + 514 = 1,414 per boundary point
Actually: just 4 time series, one per partition

Time: < 1 second (negligible)
```

### 7.4 Phase 3: Propagate and Accumulate (MAIN COST)

```
Total wave-ray combinations: 4,344,000
Cells affected per ray: ~12 (at 50m resolution, 100m influence radius)
Total cell updates: 4,344,000 × 12 = 52,128,000

Time per update: ~1 μs
Total time: ~52 seconds
```

But we also need to store arrivals. Memory consideration:
```
Events stored: 52,128,000
Bytes per event: ~40 (partition_id, time, height, period, direction)
Total memory: ~2 GB
```

### 7.5 Phase 4: Compute Statistics

```
Cells: 60,000
Operations per cell: sort arrivals, combine, compute breaking
Arrivals per cell: 52M / 60k ≈ 870 average

Time per cell: ~1 ms (dominated by sorting)
Total time: 60,000 × 1 ms = 60 seconds
```

### 7.6 Total Runtime Summary

| Phase | Time | Memory |
|-------|------|--------|
| Pre-compute rays | 30s | 100 MB |
| Generate time series | <1s | <1 MB |
| Propagate/accumulate | 52s | 2 GB |
| Compute statistics | 60s | 100 MB |
| **Total** | **~2.5 min** | **~2 GB peak** |

### 7.7 Scaling

**If reducing wind wave resolution (4x coarser):**
- Wind rays: 3,084,000 → 771,000
- Total rays: 2,031,000
- Runtime: ~1.5 min

**If using 10-minute simulation instead of 1 hour:**
- Total rays: ~700,000
- Runtime: ~30 seconds

**With 8-core parallelization:**
- Phase 3 parallelizable: 52s → 7s
- Phase 4 parallelizable: 60s → 8s
- Total: ~45 seconds

---

## 8. Optimizations

### 8.1 Skip Wind Waves for Wave-by-Wave

Wind waves have:
- Short crests (creates noise, not organized patterns)
- High frequency (many more waves to simulate)
- Less predictable breaking (chaotic)

**Recommendation:** Treat wind waves statistically (add to Hs_combined) rather than wave-by-wave.

```python
# For wind partition, use statistical contribution
wind_Hs_contribution = wind_partition.Hs

# At each cell, add to combined Hs
Hs_combined = sqrt(Hs_swell_combined² + wind_Hs_contribution²)
```

This reduces rays by ~70%.

### 8.2 Sparse Cell Storage

Most cells don't need full event history. Only store for cells in the active surfzone (depth < 10m, depth > 0.5m).

```python
# Filter cells
surfzone_mask = (mesh.elevation < -0.5) & (mesh.elevation > -10)
surfzone_cells = np.where(surfzone_mask)[0]

# Only accumulate for these cells
```

### 8.3 Chunked Processing

Instead of storing all arrivals, process in time chunks:

```python
chunk_duration = 300  # 5 minutes
for t_start in range(0, simulation_time, chunk_duration):
    t_end = min(t_start + chunk_duration, simulation_time)

    # Generate and process waves in this time window
    # Compute partial statistics
    # Aggregate
```

This reduces peak memory from 2 GB to ~200 MB.

### 8.4 Pre-compute Cell Lookups

Build spatial index once, reuse for all rays:

```python
from scipy.spatial import cKDTree

cell_tree = cKDTree(np.column_stack([mesh.points_x, mesh.points_y]))

# For each ray step
nearby = cell_tree.query_ball_point([x, y], r=influence_radius)
```

### 8.5 Vectorize Shoaling

Instead of computing shoaling per-wave, pre-compute shoaling grid:

```python
# Pre-compute Ks at each cell for each partition period
Ks_grid = {}
for partition in partitions:
    Ks = compute_shoaling_coefficient(partition.Tp, -mesh.elevation)
    Ks_grid[partition.id] = Ks

# At runtime: H_local = height * Ks_grid[partition.id][cell_idx]
```

---

## 9. Output Visualization

### 9.1 Breaking Probability Map

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

def plot_breaking_probability(field: BreakingField, region_bounds):
    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.PlateCarree()},
        figsize=(12, 10)
    )

    # Plot probability as colored scatter
    sc = ax.scatter(
        field.lon, field.lat,
        c=field.P_break,
        cmap='YlOrRd',
        s=1,
        vmin=0, vmax=1,
        transform=ccrs.PlateCarree()
    )

    ax.coastlines()
    ax.set_extent(region_bounds)
    plt.colorbar(sc, label='Breaking Probability')
    plt.title('Wave Breaking Probability')

    return fig
```

### 9.2 Breaker Type Map

```python
def plot_breaker_type(field: BreakingField, region_bounds):
    # Create numeric mapping
    type_map = {'spilling': 0, 'plunging': 1, 'surging': 2, 'none': -1}
    type_numeric = np.array([type_map.get(t, -1) for t in field.breaker_type])

    fig, ax = plt.subplots(
        subplot_kw={'projection': ccrs.PlateCarree()},
        figsize=(12, 10)
    )

    # Custom colormap
    colors = ['gray', 'blue', 'green', 'orange']  # none, spilling, plunging, surging
    cmap = plt.cm.colors.ListedColormap(colors)

    sc = ax.scatter(
        field.lon, field.lat,
        c=type_numeric + 1,  # Shift so 'none' is 0
        cmap=cmap,
        s=1,
        vmin=0, vmax=3,
        transform=ccrs.PlateCarree()
    )

    ax.coastlines()
    ax.set_extent(region_bounds)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker='o', color=c, linestyle='')
        for c in colors
    ]
    ax.legend(handles, ['None', 'Spilling', 'Plunging', 'Surging'])

    plt.title('Breaker Type')
    return fig
```

### 9.3 Combined Dashboard

```python
def create_breaking_dashboard(field: BreakingField, region_bounds):
    fig = plt.figure(figsize=(16, 12))

    # Breaking probability
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    # ... plot P_break

    # Breaker type
    ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
    # ... plot breaker_type

    # Breaking frequency
    ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
    # ... plot breaking_frequency

    # Mean breaking height
    ax4 = fig.add_subplot(2, 2, 4, projection=ccrs.PlateCarree())
    # ... plot mean_breaking_height

    plt.tight_layout()
    return fig
```

---

## References

1. **Kimura, A. (1980)**. "Statistical Properties of Random Wave Groups." Proceedings of 17th International Conference on Coastal Engineering, Sydney. [Link](https://icce-ojs-tamu.tdl.org/icce/article/view/3604)

2. **Longuet-Higgins, M.S. (1984)**. "Statistical Properties of Wave Groups in a Random Sea State." Philosophical Transactions of the Royal Society.

3. **Battjes, J.A. and Janssen, J.P.F.M. (1978)**. "Energy Loss and Set-Up Due to Breaking of Random Waves." Proceedings of 16th ICCE.

4. **USGS**. "Wave Groupiness Variations in the Nearshore." [Link](https://www.usgs.gov/publications/wave-groupiness-variations-nearshore)

5. **Coastal Wiki**. "Statistical Description of Wave Parameters." [Link](https://www.coastalwiki.org/wiki/Statistical_description_of_wave_parameters)
