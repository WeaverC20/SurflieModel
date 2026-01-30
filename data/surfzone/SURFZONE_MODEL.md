# Surfzone Wave Propagation Model

This document describes the backward ray tracing algorithm for propagating waves from the SWAN model output through the surfzone.

---

## Overview

The surfzone model uses **backward ray tracing** to determine wave conditions at mesh points. Instead of tracing waves forward from offshore to shore, we trace rays backward from each mesh point to the SWAN boundary (deep water).

**Key advantage**: We only compute wave properties at locations where we need them - no wasted rays.

---

## Directory Structure

```
data/surfzone/
├── SURFZONE_MODEL.md              # This documentation
├── mesh.py                        # SurfZoneMesh class (bathymetry + spatial index)
├── run_surfzone.py                # Main runner (WIP)
└── runner/
    ├── wave_physics.py            # Core physics (Numba-accelerated)
    ├── backward_ray_tracer.py     # Primary ray tracing engine
    ├── backward_ray_tracer_debug.py # Visualization/debug tool
    ├── ray_tracer.py              # Legacy forward tracer (unused)
    ├── swan_input_provider.py     # SWAN boundary conditions
    └── output_writer.py           # Results storage
```

---

## Backward Ray Tracing Physics

### Key Concept

Forward and backward ray tracing use the same physics equations, with one critical difference:

| Aspect | Forward Tracing | Backward Tracing |
|--------|-----------------|------------------|
| Direction | Toward shore (with wave) | Away from shore (opposite) |
| Refraction | Bends toward SLOWER celerity (shallow) | Bends toward FASTER celerity (deep) |
| Gradient sign | Normal | NEGATED |

### Core Equations

**1. Local wavelength (Fenton & McKee 1990):**
```
L = L₀ · [tanh((2πh/L₀)^0.75)]^(2/3)
```

**2. Local celerity:**
```
C = L / T
```

**3. Ray refraction (forward):**
```
dθ/ds = -(1/C) · ∂C/∂n
```
Rays bend toward slower celerity (shallower water).

**4. Ray refraction (backward):**
```
dθ/ds = +(1/C) · ∂C/∂n
```
Rays bend toward faster celerity (deeper water).

We achieve backward refraction by **negating the celerity gradients** before passing to the standard update function:
```python
dx, dy = update_ray_direction(dx, dy, C, -dC_dx, -dC_dy, step_size)
```

### Implementation

The backward tracer:
1. Starts at a near-shore mesh point
2. Converts wave direction to math convention
3. **Negates direction** (points away from shore toward ocean)
4. For each step:
   - Computes local depth and celerity
   - Computes celerity gradients
   - **Negates gradients** for backward refraction
   - Updates ray direction
   - Steps toward deeper water
5. Stops when reaching boundary depth threshold (e.g., 50m)

```python
# From backward_ray_tracer.py

# 1. Negate direction (point away from shore)
dx = -np.cos(theta_M)
dy = -np.sin(theta_M)

# 2. In the loop - negate gradients
dC_dx, dC_dy = celerity_gradient_indexed(...)
dx, dy = update_ray_direction(dx, dy, C, -dC_dx, -dC_dy, step_size)

# 3. Normal positive step
x += dx * step_size
y += dy * step_size
```

---

## Wave Physics Reference

All wave physics are in `wave_physics.py` using Numba for performance.

### Deep Water Properties (calculated once per wave)

| Property | Formula | Description |
|----------|---------|-------------|
| L₀ | g·T² / (2π) | Deep water wavelength |
| C₀ | L₀ / T | Deep water celerity |
| Cg₀ | C₀ / 2 | Deep water group velocity |

### Local Properties (at each step)

| Property | Formula | Description |
|----------|---------|-------------|
| L | Fenton-McKee | Local wavelength |
| k | 2π / L | Wavenumber |
| C | L / T | Local celerity |
| n | f(kh) | Group velocity ratio |
| Cg | n · C | Local group velocity |

### Shoaling

For backward tracing, shoaling coefficient relates boundary to mesh:
```
K_s = sqrt(Cg_boundary / Cg_mesh)
H_mesh = H_boundary × K_s
```

---

## Breaking Criteria

Breaking criteria are applied after wave heights are determined:

### Basic Criterion
```
Wave breaks when: H ≥ γb · h
```

### Breaker Index Options

1. **McCowan (1894)**: γb = 0.78 (constant)
2. **Rattanapitikon & Shibayama (2000)**: γb = 0.57 + 0.71·(H₀/L₀)^0.12·m^0.36
3. **Weggel (1972)**: Slope-dependent formula

### Wind Modification (Douglass 1990)
```
γb = γb,0 · (1 - Cw · Uw·cos(φ) / C)
```
- Onshore wind: Earlier breaking (lower γb)
- Offshore wind: Later breaking (higher γb)

### Iribarren Number (Breaker Type)
```
ξ = m / √(H/L₀)
```

| ξ Range | Breaker Type |
|---------|--------------|
| < 0.5 | Spilling |
| 0.5 - 3.3 | Plunging |
| 3.3 - 5.0 | Collapsing |
| > 5.0 | Surging |

---

## Debug/Visualization

Use `backward_ray_tracer_debug.py` to visualize ray paths:

```bash
cd /path/to/SurflieModel
venv/bin/python data/surfzone/runner/backward_ray_tracer_debug.py
```

This traces rays from shallow water points backward to the 50m boundary, showing:
- Left panel: Zoomed coastal view
- Middle panel: Full ray paths
- Right panel: Depth vs distance along rays

---

## References

1. **Fenton & McKee (1990)**: Wavelength approximation
2. **Dean & Dalrymple**: Water Wave Mechanics for Engineers and Scientists
3. **USACE Coastal Engineering Manual**: Shoaling and refraction
4. **Rattanapitikon & Shibayama (2000)**: Breaking index
5. **Douglass (1990)**: Wind effects on breaking
6. **Galvin (1968)**: Iribarren number classification
