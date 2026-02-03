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
│
├── meshes/                        # Generated surfzone meshes (by region)
│   ├── socal/
│   │   ├── socal_surfzone.npz     # Mesh data (points, depths, triangulation)
│   │   └── socal_surfzone.json    # Mesh metadata
│   ├── central/
│   └── norcal/
│
├── output/                        # Simulation results (by region)
│   ├── socal/
│   │   ├── primary_swell.npz      # Wave heights, convergence, etc.
│   │   └── primary_swell.json     # Result metadata
│   ├── central/
│   └── norcal/
│
└── runner/
    ├── run_simulation.py          # CLI entry point (--region argument)
    ├── surfzone_runner.py         # SurfzoneRunner orchestration class
    ├── backward_ray_tracer.py     # Primary ray tracing engine
    ├── forward_propagation.py     # Forward wave height propagation
    ├── wave_physics.py            # Core physics (Numba-accelerated)
    ├── swan_input_provider.py     # SWAN partition data interpolation
    ├── surfzone_result.py         # Result dataclasses
    ├── output_writer.py           # Results storage
    ├── backward_ray_tracer_debug.py # Visualization/debug tool
    └── ray_tracer.py              # Legacy forward tracer (unused)
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

---

## Multi-Region Support

The surfzone model supports three California regions:

| Region | Name | Latitude | Description |
|--------|------|----------|-------------|
| socal | Southern California | 32.0 - 35.0 | Mexico border to Point Conception |
| central | Central California | 34.5 - 39.0 | Point Conception to Point Reyes |
| norcal | Northern California | 38.5 - 42.0 | Point Reyes to Oregon border |

### Multi-Region Workflow

```bash
# 1. Generate surfzone mesh for a region
python scripts/generate_surfzone_mesh.py socal
python scripts/generate_surfzone_mesh.py central
python scripts/generate_surfzone_mesh.py norcal

# 2. Run simulation (requires SWAN data in data/swan/runs/{region}/)
python data/surfzone/runner/run_simulation.py --region socal
python data/surfzone/runner/run_simulation.py --region central --swan-resolution fine

# 3. View results
python scripts/dev/view_surfzone_result.py --region socal
python scripts/dev/view_surfzone_result.py --list-regions
```

### CLI Arguments

**run_simulation.py:**
```
--region REGION        Region name (socal, norcal, central) [required]
--list-regions         List available regions with status
--swan-resolution RES  SWAN resolution (coarse, fine, etc.) [auto-detect]
--min-depth FLOAT      Minimum depth filter (default: 0.0m)
--max-depth FLOAT      Maximum depth filter (default: 10.0m)
--sample-fraction F    Fraction of points to sample (e.g., 0.1 for 10%)
--sample-count N       Exact number of points to sample
--seed N               Random seed for reproducible sampling
```

---

## SurfzoneRunner Class

The `SurfzoneRunner` class orchestrates the complete simulation:

```python
from data.surfzone.mesh import SurfZoneMesh
from data.surfzone.runner.swan_input_provider import SwanInputProvider
from data.surfzone.runner.surfzone_runner import SurfzoneRunner, SurfzoneRunnerConfig

# Load mesh and SWAN data
mesh = SurfZoneMesh.load('data/surfzone/meshes/socal')
swan = SwanInputProvider('data/swan/runs/socal/coarse/latest')
boundary_conditions = swan.get_boundary_from_mesh(mesh)

# Configure runner
config = SurfzoneRunnerConfig(
    min_depth=0.0,
    max_depth=10.0,
    partition_id=1,  # Primary swell
    sample_fraction=0.1,  # Optional: sample 10% for testing
)

# Run simulation
runner = SurfzoneRunner(mesh, boundary_conditions, config)
result = runner.run(region_name="Southern California")

# Access results
print(result.summary())
print(f"Converged: {result.n_converged} / {result.n_sampled}")
```

### SurfzoneRunnerConfig Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_depth` | 0.0 | Minimum depth filter (m) |
| `max_depth` | 10.0 | Maximum depth filter (m) |
| `partition_id` | 1 | Wave partition (0=wind sea, 1-3=swell) |
| `boundary_depth_threshold` | 50.0 | Depth threshold for boundary detection (m) |
| `step_size` | 15.0 | Ray tracing step size (m) |
| `max_iterations` | 20 | Maximum convergence iterations |
| `convergence_tolerance` | 0.10 | Convergence tolerance (fraction) |
| `sample_fraction` | None | Fraction of points to sample |
| `sample_count` | None | Exact number of points to sample |

---

## References

1. **Fenton & McKee (1990)**: Wavelength approximation
2. **Dean & Dalrymple**: Water Wave Mechanics for Engineers and Scientists
3. **USACE Coastal Engineering Manual**: Shoaling and refraction
4. **Rattanapitikon & Shibayama (2000)**: Breaking index
5. **Douglass (1990)**: Wind effects on breaking
6. **Galvin (1968)**: Iribarren number classification
