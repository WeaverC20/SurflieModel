# Surfzone Wave Simulation: Backward Ray Tracing Approach

## Overview

This document describes the backward ray tracing approach for simulating wave propagation through the surfzone. Instead of tracing waves forward from the SWAN boundary to shore, we trace rays **backward** from each surfzone mesh point to the boundary (deep water).

---

## Table of Contents

1. [Key Concept](#1-key-concept)
2. [Algorithm Overview](#2-algorithm-overview)
3. [Physical Background](#3-physical-background)
4. [Data Structures](#4-data-structures)
5. [Implementation Details](#5-implementation-details)
6. [Computational Efficiency](#6-computational-efficiency)
7. [Output and Post-Processing](#7-output-and-post-processing)

---

## 1. Key Concept

### Forward vs Backward Ray Tracing

**Forward (inefficient):**
- Trace rays from SWAN boundary toward shore
- Hope rays pass through mesh points of interest
- Many rays miss important areas
- Difficult to get uniform coverage

**Backward (efficient):**
- Start at each mesh point where we need wave information
- Trace ray backward toward the SWAN boundary (deep water)
- Find which partition the ray connects to
- Compute wave height using accumulated shoaling

### Why Backward Works

The wave ray equations can be run in reverse by making two key changes:

1. **Negate direction**: Ray points away from shore (opposite of wave travel)
2. **Negate gradients**: Rays bend toward FASTER celerity (deeper water)

```python
# Forward: wave travels toward shore, bends toward slower C (shallow)
dx, dy = cos(theta), sin(theta)
dx, dy = update_ray_direction(dx, dy, C, dC_dx, dC_dy, ds)

# Backward: ray travels away from shore, bends toward faster C (deep)
dx, dy = -cos(theta), -sin(theta)  # NEGATED direction
dx, dy = update_ray_direction(dx, dy, C, -dC_dx, -dC_dy, ds)  # NEGATED gradients
```

---

## 2. Algorithm Overview

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUTS                                                          │
│  - SWAN boundary conditions (Hs, Tp, direction, spread per       │
│    partition at each boundary segment)                           │
│  - Surfzone mesh (bathymetry, triangulation, spatial index)     │
│  - Mesh points where wave information is needed                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  FOR EACH MESH POINT:                                            │
│                                                                  │
│  1. Select boundary segment ("to the left" mapping)              │
│  2. For each partition at that segment:                          │
│     a. Initial guess: θ_M = θ_partition                          │
│     b. Trace ray backward from mesh point toward boundary        │
│        - NEGATE direction (point away from shore)                │
│        - NEGATE gradients (bend toward deeper water)             │
│     c. Check if ray arrives with correct direction               │
│     d. If not converged, update θ_M using gradient descent       │
│     e. Repeat until converged (typically 5-10 iterations)        │
│  3. Compute wave height: H = H_boundary × K_shoaling             │
│                                                                  │
│  Output: Wave height, direction, period at mesh point            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  POST-PROCESSING (applied to results)                            │
│  - Apply breaking criteria                                       │
│  - Classify breaker types                                        │
│  - Compute statistics (probability, frequency, height dist.)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Physical Background

### 3.1 Wave Ray Equation

Waves refract as they propagate due to depth-dependent celerity.

**Forward tracing:**
```
dθ/ds = -(1/C) × ∂C/∂n
```
Rays bend toward **slower** celerity (shallower water).

**Backward tracing:**
```
dθ/ds = +(1/C) × ∂C/∂n
```
Rays bend toward **faster** celerity (deeper water).

We achieve backward behavior by **negating the celerity gradients** before passing to the standard update function:

```python
# In backward_ray_tracer.py
dC_dx, dC_dy = celerity_gradient_indexed(x, y, ...)
dx, dy = update_ray_direction(dx, dy, C, -dC_dx, -dC_dy, step_size)  # NEGATED
```

### 3.2 Shoaling

Wave height changes as waves propagate into shallower water:

```
H_local = H_deep × K_s × K_r
```

Where:
- `K_s = sqrt(Cg_deep / Cg_local)` - shoaling coefficient
- `K_r` = refraction coefficient (from ray spreading/convergence)

For backward tracing, we track Cg at start (mesh point) and end (boundary):

```
K_s = sqrt(Cg_boundary / Cg_mesh)
```

### 3.3 Direction Convergence

The iteration finds `θ_M` (direction at mesh point) such that when we trace backward, the ray arrives at the boundary with direction matching the partition:

**Convergence criterion:**
```
error = |θ_arrival - θ_partition| / Δθ_partition
converged = error < 0.10  (10% of directional spread)
```

**Gradient descent update:**
```
θ_M_new = θ_M - α × (θ_arrival - θ_partition)
```

Where `α = 0.5-0.7` is a relaxation factor.

---

## 4. Data Structures

### 4.1 Boundary Partition

```python
@dataclass
class BoundaryPartition:
    """Wave partition data at a boundary segment."""
    partition_id: int        # 0=wind sea, 1-3=swells
    Hs: float               # Significant wave height (m)
    Tp: float               # Peak period (s)
    direction: float        # Direction (degrees, nautical FROM)
    directional_spread: float  # Spread (degrees)
```

### 4.2 Mesh Point Result

```python
@dataclass
class MeshPointResult:
    """Complete wave information at a mesh point."""
    mesh_x: float
    mesh_y: float
    mesh_depth: float
    contributions: List[PartitionContribution]

    @property
    def total_Hs(self) -> float:
        """Combined Hs (RMS of contributions)."""
        return sqrt(sum(c.H**2 for c in self.contributions if c.converged))
```

---

## 5. Implementation Details

### 5.1 Core Backward Tracing Function

```python
@njit(cache=True)
def trace_backward_single(
    start_x, start_y,           # Mesh point (near shore)
    T,                          # Wave period
    theta_M_nautical,           # Direction guess at mesh point
    # Mesh arrays...
    boundary_depth_threshold,   # Depth threshold for "reached boundary"
    step_size=10.0,
    max_steps=500,
):
    """
    Trace a single ray backward from mesh point toward deep water boundary.
    """
    # Get deep water properties
    L0, C0, Cg0 = deep_water_properties(T)

    # Convert to math convention
    theta_M = nautical_to_math(theta_M_nautical)

    # BACKWARD: Negate direction (point away from shore toward ocean)
    dx = -cos(theta_M)
    dy = -sin(theta_M)

    # Ray marching loop
    for step in range(max_steps):
        # Check if reached boundary (deep water)
        if depth > boundary_depth_threshold:
            # Convert ray direction back to wave direction (negate again)
            theta_arrival = math_to_nautical(arctan2(-dy, -dx))
            return (x, y, theta_arrival, Cg_start, Cg_end, True)

        # Get local wave properties
        h = interpolate_depth_indexed(x, y, ...)
        L, k, C, n, Cg = local_wave_properties(L0, T, h)

        # BACKWARD: Negate gradients for backward refraction
        # Rays bend toward FASTER C (deeper water)
        dC_dx, dC_dy = celerity_gradient_indexed(x, y, ...)
        dx, dy = update_ray_direction(dx, dy, C, -dC_dx, -dC_dy, step_size)

        # Normal positive step (direction already points toward ocean)
        x += dx * step_size
        y += dy * step_size

    return (x, y, nan, Cg_start, Cg_end, False)
```

### 5.2 High-Level Interface

```python
class BackwardRayTracer:
    def __init__(self, mesh, boundary_depth_threshold):
        self.mesh = mesh
        self.boundary_depth_threshold = boundary_depth_threshold
        # Load mesh arrays and spatial index...

    def trace_mesh_point(self, x, y, partitions) -> MeshPointResult:
        """Trace all partitions for a single mesh point."""
        contributions = []
        for partition in partitions:
            result = backward_trace_with_convergence(x, y, partition, ...)
            contributions.append(result)
        return MeshPointResult(x, y, depth, contributions)
```

---

## 6. Computational Efficiency

### Why Backward is Faster

| Approach | Work Done | Typical Scale |
|----------|-----------|---------------|
| Forward | Trace N rays × M steps, then interpolate to mesh | N~10,000 rays, many miss |
| Backward | For each mesh point, trace 1-3 partitions × ~10 iterations | Every computation is useful |

Backward trades off:
- More total ray marching steps
- But: Every computation is useful (directly gives mesh point values)
- No wasted rays that miss areas of interest
- Trivially parallelizable (each mesh point is independent)

### Parallelization

```python
# Each mesh point is independent - perfect for parallel processing
from numba import prange

@njit(parallel=True)
def trace_all_parallel(mesh_x, mesh_y, ...):
    results = np.empty((len(mesh_x), n_partitions, n_fields))
    for i in prange(len(mesh_x)):
        results[i] = trace_mesh_point(mesh_x[i], mesh_y[i], ...)
    return results
```

---

## 7. Output and Post-Processing

### 7.1 Direct Outputs

For each mesh point:
- Combined Hs (RMS of partition contributions)
- Dominant direction (energy-weighted)
- Individual partition contributions

### 7.2 Breaking Analysis (Post-Processing)

Breaking criteria applied after backward tracing:

```python
def analyze_breaking(result: MeshPointResult, wind_speed, wind_direction):
    """Apply breaking criteria to mesh point result."""
    H = result.total_Hs
    h = result.mesh_depth
    T = weighted_mean([c.T for c in result.contributions])

    # Wind-modified breaking index
    gamma_b = compute_breaking_index(...)

    # Breaking check
    is_breaking = H >= gamma_b * h

    # Breaker classification
    if is_breaking:
        xi = iribarren_number(slope, H, T)
        breaker_type = classify_breaker(xi)

    return BreakingResult(is_breaking, breaker_type, H, h, ...)
```

---

## References

1. **Wave Ray Tracing**: Dean & Dalrymple, "Water Wave Mechanics for Engineers and Scientists"
2. **Shoaling & Refraction**: USACE Coastal Engineering Manual
3. **Fenton & McKee (1990)**: Wavelength approximation
4. **Breaking Criteria**: Battjes & Janssen (1978), Rattanapitikon & Shibayama (2000)
5. **Breaker Classification**: Galvin (1968) - Iribarren number classification
