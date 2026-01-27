# Surfzone Wave Propagation Model

This document describes the ray tracing algorithm for propagating waves from the SWAN model output (2.5km offshore) through the surfzone to the shore.

---

## Implementation Overview

### Architecture

```
data/surfzone/
├── SURFZONE_MODEL.md          # This documentation
├── mesh.py                    # Existing mesh class (SurfZoneMesh)
├── run_surfzone.py            # Main CLI entry point
└── runner/
    ├── __init__.py
    ├── wave_physics.py        # Core physics calculations (Numba-accelerated)
    ├── ray_tracer.py          # Ray tracing engine
    ├── swan_input_provider.py # Extract wave conditions from SWAN output
    ├── breaking_model.py      # Breaking criteria and classification
    └── output_writer.py       # Results storage
```

### Data Flow

```
SWAN Output (.mat files)
    ↓
swan_input_provider.py
    ↓ (interpolate to surfzone boundary at 2.5km offshore)
Wave partitions (1-4 per boundary point)
    ↓
ray_tracer.py + wave_physics.py
    ↓ (trace each partition shoreward)
Breaking locations + characteristics
    ↓
output_writer.py
    ↓
Continuous field (breaking density, heights, types)
```

### Key Design Decisions

1. **Input Source**: Read directly from SWAN `.mat` partition outputs
   - No intermediate format (simpler, faster iteration)
   - SWAN runs cached in `data/swan/runs/{region}/{resolution}/latest/`
   - Interpolate SWAN grid to surfzone boundary points (SWAN mesh is coarser)

2. **Ray Tracing**: Deterministic, not Monte Carlo
   - 1-4 rays per boundary point (one per SWAN wave partition: wind sea + up to 3 swells)
   - Each ray has: H₀, T, θ₀ from SWAN partition output

3. **Step Size**: Match mesh resolution (~20-50m near coast)
   - Surfzone mesh resolution: 20m (coast) to 300m (2.5km offshore)
   - Step size can be larger than wavelength for coarse initial runs
   - Risk: missing rapid depth changes, but mesh doesn't have finer detail anyway
   - For T=10s wave: L₀ ≈ 156m deep water, L ≈ 50m at h=5m depth

4. **Wind Data**: Use locally downloaded GFS data
   - Location: `data/downloaded_weather_data/wind/gfs_*.nc`
   - Reuse SWAN runner's WindProvider pattern for interpolation

5. **Output**: Continuous field for all of Southern California
   - Breaking density (rays breaking per area)
   - Breaking height Hb at each break point
   - Breaker type (spilling/plunging/collapsing/surging)
   - Format: `.npz` for fast loading (viewer to be built later)

6. **Performance**: Prioritize fast iteration
   - Numba JIT compilation for physics calculations
   - Coarse stepping for initial runs
   - Batch processing where possible

### Mesh Details

The surfzone mesh (`data/surfzone/meshes/socal/socal_surfzone.npz`) contains:
- ~321k points in UTM coordinates
- Delaunay triangulation for O(1) interpolation
- `get_numba_arrays()` method exports contiguous arrays for JIT

Mesh configuration:
- Offshore distance: 2500m (2.5km)
- Onshore buffer: 50m
- Resolution: 20m (coast) to 300m (offshore)
- Coastline density bias: 2.5 (more points near shore)

---

## Overview

The surfzone model takes wave conditions from the SWAN simulation at 2.5km offshore and traces wave rays shoreward through the high-resolution surfzone mesh. The model accounts for:
- **Shoaling**: Wave height changes due to depth-induced changes in group velocity
- **Refraction**: Wave direction changes due to depth gradients (Snell's Law)
- **Wind effects**: Energy input/extraction from local winds
- **Breaking**: Depth-limited wave breaking with slope and wind dependencies

---

## Input Variables

### User-Specified Inputs

| Symbol | Name | Typical Values | Units |
|--------|------|----------------|-------|
| H₀ | Deep water wave height | 0.5 - 5 | m |
| T | Wave period | 5 - 20 | s |
| θ₀ | Deep water wave angle | 0 - 45° | deg |
| Uᵥ | Wind speed (10m) | 0 - 25 | m/s |
| θᵥ | Wind direction | 0 - 360° | deg |
| h(x) | Bathymetry | varies | m |

### Physical Constants

| Symbol | Name | Value | Units |
|--------|------|-------|-------|
| g | Gravity | 9.81 | m/s² |
| ρw | Water density | 1025 | kg/m³ |
| ρa | Air density | 1.225 | kg/m³ |

### Empirical Coefficients

| Symbol | Name | Typical Range | Source |
|--------|------|---------------|--------|
| α | Wind energy coefficient | 0.01 - 0.05 | Calibration |
| β | Miles coefficient | ~0.025 | Cavaleri & Rizzoli (1981) |
| γw | Steepness modification | 0.1 - 0.2 | Douglass (1990) |
| Cw | Wind modification for γb | 0.1 - 0.2 | Douglass (1990) |

---

## Step-by-Step Algorithm

### Calculate Once (Deep Water Reference Values)

1. **Deep water wavelength**:
   ```
   L₀ = g·T² / (2π)
   ```

2. **Deep water wave celerity**:
   ```
   C₀ = L₀ / T
   ```

3. **Deep water group velocity**:
   ```
   Cg₀ = C₀ / 2
   ```

### At Each Position x (Marching from Offshore Toward Shore)

1. **Get local depth** h from bathymetry

2. **Calculate local wavelength** (Fenton & McKee approximation):
   ```
   L = L₀ · [tanh((2πh/L₀)^(3/4))]^(2/3)
   ```

3. **Calculate local wavenumber**:
   ```
   k = 2π / L
   ```

4. **Calculate local celerity**:
   ```
   C = L / T
   ```

5. **Calculate group velocity ratio**:
   ```
   n = (1/2) · [1 + 2kh / sinh(2kh)]
   ```

6. **Calculate local group velocity**:
   ```
   Cg = n · C
   ```

7. **Calculate shoaling coefficient**:
   ```
   Ks = √(Cg₀ / Cg)
   ```

8. **Calculate local wave angle** (Snell's Law):
   ```
   θ = arcsin((C/C₀) · sin(θ₀))
   ```

9. **Calculate refraction coefficient**:
   ```
   Kr = √(cos(θ₀) / cos(θ))
   ```

10. **Calculate relative wind-wave angle**:
    ```
    φ = θᵥ - θ
    ```

11. **Update wind modification factor**:
    ```
    Kw(x) = Kw(x-Δx) · exp[α · (Uᵥ·cos(φ) - C)/C · Δx/L]
    ```
    Initialize Kw = 1 at offshore boundary

12. **Calculate local wave height**:
    ```
    H(x) = H₀ · Ks · Kr · Kw
    ```

13. **Calculate local steepness** (for breaking check):
    ```
    H/L  (geometric steepness)
    (H/L)eff = (H/L) · [1 + γw · Uᵥ·cos(φ)/C]  (effective steepness)
    ```

14. **Check breaking criterion**:
    ```
    Is H ≥ γb · h ?
    ```
    If yes, this is the break point.

---

## Breaking Criteria Models

### Basic Breaking Criterion

Waves break when: `H / h ≥ γb`

Classic approximation: γb ≈ 0.78 (McCowan, 1894)

However, γb depends on beach slope, wave steepness, and wind conditions.

### Miche (1944) Criterion

```
Hb = min( 0.142·Lb·tanh(2πhb/Lb) , 0.78·hb )
```

- First term: steepness-limited breaking (waves too steep)
- Second term: depth-limited breaking (water too shallow)

### Weggel (1972) Formula

Incorporates beach slope explicitly:

```
γb = Hb/hb = b - a·(Hb / gT²)
```

Where a and b are slope-dependent coefficients:
```
a = 43.8 · (1 - e^(-19m))
b = 1.56 / (1 + e^(-19.5m))
```

| Variable | Name | Meaning |
|----------|------|---------|
| m | Beach slope | Rise over run (e.g., 0.02 for 1:50 slope) |
| a | Steepness coefficient | Controls how wave steepness affects γb |
| b | Maximum breaker index | Upper limit of γb as steepness → 0 |

**Behavior**:
- Steep slopes (m large): b → 1.56, a → 43.8 (higher γb)
- Mild slopes (m small): b → 0.78, a → 0 (lower γb)

### Rattanapitikon & Shibayama (2000)

Explicit empirical formulation (no iteration needed):

```
γb = 0.57 + 0.71 · (H₀/L₀)^0.12 · m^0.36
```

### Goda (2010) Formulation

Widely used in engineering:

```
γb = 0.17·(L₀/hb)·[1 - exp(-1.5π·hb/L₀·(1 + 15·m^(4/3)))]
```

---

## Wind Effects on Breaking

### Physical Mechanisms

**Onshore winds**:
- Push against the back of the wave, steepening the front face
- Waves break earlier (in deeper water)
- Lower effective γb

**Offshore winds**:
- Push against the front face, supporting it
- Waves can grow taller before breaking
- Higher effective γb
- Creates classic "offshore wind groomed" surf conditions

### Douglass (1990) Wind-Modified Breaking

```
γb = γb,0 · (1 - Cw · Uw·cos(φ) / C)
```

| Condition | cos(φ) | Effect on γb |
|-----------|--------|--------------|
| Onshore wind | Positive (~1) | Decreases γb (earlier breaking) |
| Offshore wind | Negative (~-1) | Increases γb (later breaking) |
| Cross-shore wind | ~0 | Minimal effect |
| Calm | Uw = 0 | γb = γb,0 |

### Comprehensive Wind-Modified Formula

Combining slope, steepness, and wind effects:

```
γb = [0.57 + 0.71·(H₀/L₀)^0.12·m^0.36] · (1 - Cw·Uw·cos(φ)/C)
```

---

## Iribarren Number (Surf Similarity Parameter)

### Definition

```
ξ = m / √(H/L₀)
```

Or using deep water wave height:
```
ξ₀ = m / √(H₀/L₀)
```

### Breaker Classification

| ξ Range | Breaker Type | Description |
|---------|--------------|-------------|
| ξ < 0.5 | Spilling | Gradual breaking; foam cascades down front face |
| 0.5 < ξ < 3.3 | Plunging | Classic "tube" or "barrel"; wave curls over |
| 3.3 < ξ < 5.0 | Collapsing | Lower part of wave front steepens and falls |
| ξ > 5.0 | Surging | Wave slides up beach without breaking completely |

### Visual Characteristics

| Breaker Type | Visual Signature | Typical Beach |
|--------------|------------------|---------------|
| Spilling | White foam at crest, rolling down | Gentle sandy |
| Plunging | Curling crest, air pocket | Moderate, reef breaks |
| Collapsing | Front face collapses | Steeper beaches |
| Surging | No distinct break, surges up | Very steep |

### Wind-Modified Iribarren Number

```
ξeff = m / √((H/L)eff)
ξeff = m / √( (H/L₀)·[1 + γw·Uw·cos(φ)/C] )
```

| Wind Condition | Effect on ξ | Breaker Tendency |
|----------------|-------------|------------------|
| Strong onshore | Decreases | Shifts to spilling |
| Offshore | Increases | Shifts to plunging |
| Calm | Base value | Slope-determined |

---

## Complete Breaking Analysis Algorithm

### Given (from propagation model at each point):
H, h, L, C, θ

### Given (inputs):
H₀, L₀, T, m (local slope), Uw, θw

### Step 1: Calculate relative wind angle
```
φ = θw - θ
```

### Step 2: Calculate base breaker index

**Option A — Rattanapitikon & Shibayama**:
```
γb,0 = 0.57 + 0.71·(H₀/L₀)^0.12·m^0.36
```

**Option B — Weggel**:
```
a = 43.8·(1 - e^(-19m))
b = 1.56 / (1 + e^(-19.5m))
γb,0 = b - a·(H / gT²)
```

### Step 3: Apply wind modification
```
γb = γb,0 · (1 - Cw·Uw·cos(φ)/C)
```
where Cw ≈ 0.1 to 0.2

### Step 4: Check breaking criterion
```
If H/h ≥ γb, wave breaks at this location
```

### Step 5: At break point, calculate effective steepness
```
(H/L)eff = (Hb/L₀)·[1 + γw·Uw·cos(φ)/Cb]
```

### Step 6: Calculate Iribarren number
```
ξ = m / √((H/L)eff)
```

### Step 7: Classify breaker type
| Condition | Breaker Type |
|-----------|--------------|
| ξ < 0.5 | Spilling |
| 0.5 ≤ ξ < 3.3 | Plunging |
| 3.3 ≤ ξ < 5.0 | Collapsing |
| ξ ≥ 5.0 | Surging |

---

## Practical Considerations

### Determining Local Slope m

```
m = Δh / Δx
```

**Recommendations**:
- Use slope averaged over 1-2 wavelengths shoreward of evaluation point
- For irregular bathymetry, use best-fit slope through surf zone
- Typical values: 0.01 (1:100, very mild) to 0.1 (1:10, steep)

### Iteration for Breaking Point

Since γb may depend on Hb, finding the exact break point may require iteration:
1. March shoreward checking H/h ≥ γb at each step
2. When criterion first satisfied, refine location
3. Interpolate to find exact hb where H = γb·h

### Coefficient Selection

| Condition | Recommended Cw | Recommended γw |
|-----------|----------------|----------------|
| Strong winds (Uw > 15 m/s) | 0.15 - 0.2 | 0.15 - 0.2 |
| Moderate winds (5-15 m/s) | 0.1 - 0.15 | 0.1 - 0.15 |
| Light winds (< 5 m/s) | 0.05 - 0.1 | 0.05 - 0.1 |

---

## Summary of All Variables

### Breaking Criterion Variables

| Symbol | Name | Typical Range | Units |
|--------|------|---------------|-------|
| γb | Breaker index | 0.6 - 1.2 | — |
| γb,0 | No-wind breaker index | 0.6 - 1.0 | — |
| Hb | Breaking wave height | varies | m |
| hb | Breaking depth | varies | m |
| m | Beach slope | 0.01 - 0.2 | — |

### Calculated Variables

| Symbol | Name | Calculated From |
|--------|------|-----------------|
| L₀ | Deep water wavelength | T, g |
| C₀ | Deep water celerity | L₀, T |
| Cg₀ | Deep water group velocity | C₀ |
| L | Local wavelength | L₀, h |
| k | Local wavenumber | L |
| C | Local celerity | L, T |
| n | Group velocity ratio | k, h |
| Cg | Local group velocity | n, C |
| Ks | Shoaling coefficient | Cg₀, Cg |
| θ | Local wave angle | C, C₀, θ₀ |
| Kr | Refraction coefficient | θ, θ₀ |
| φ | Relative wind angle | θᵥ, θ |
| Kw | Wind modification | Uᵥ, φ, C, L, Δx |
| H | Local wave height | H₀, Ks, Kr, Kw |

---

## Implementation Phases

### Phase 1: Basic Infrastructure

1. **`swan_input_provider.py`**
   - Read SWAN `.mat` partition outputs (phs0-3, ptp0-3, pdir0-3)
   - Interpolate from SWAN grid to surfzone boundary points
   - Return structured wave partition data (H, T, θ for each partition)

2. **`wave_physics.py`** (Numba-accelerated)
   - Deep water reference calculations (L₀, C₀, Cg₀)
   - Local wave properties (L, k, C, n, Cg)
   - Shoaling coefficient Ks
   - Refraction via Snell's Law (θ, Kr)
   - Wind modification factor Kw

3. **`ray_tracer.py`**
   - Single ray tracing function
   - Batch ray processing
   - Step size management (adaptive based on depth/wavelength)

### Phase 2: Runner Integration

1. **`breaking_model.py`**
   - Breaking criterion (γb calculation with slope/wind effects)
   - Iribarren number calculation
   - Breaker type classification

2. **`run_surfzone.py`**
   - CLI entry point (similar to SWAN runner)
   - Wind data integration (reuse WindProvider pattern)
   - Coordinate SWAN input → ray tracing → output

### Phase 3: Output and Visualization

1. **`output_writer.py`**
   - Aggregate breaking points into continuous field
   - Calculate breaking density per area
   - Store as `.npz` for fast loading

2. **Viewer** (future, separate task)
   - Visualize breaking density field
   - Show breaker types spatially
   - Overlay on bathymetry

---

## Runner Usage

```bash
# Basic run (uses latest SWAN output)
python data/surfzone/run_surfzone.py --region socal --mesh socal

# Specify SWAN resolution source
python data/surfzone/run_surfzone.py --region socal --mesh socal --swan-resolution coarse

# Dry run (validate inputs without tracing)
python data/surfzone/run_surfzone.py --region socal --mesh socal --dry-run

# Fast iteration mode (coarse stepping)
python data/surfzone/run_surfzone.py --region socal --mesh socal --fast
```

### Input Files Required

| File | Location | Description |
|------|----------|-------------|
| SWAN partitions | `data/swan/runs/socal/coarse/latest/phs*.mat` | Wave height per partition |
| SWAN partitions | `data/swan/runs/socal/coarse/latest/ptp*.mat` | Peak period per partition |
| SWAN partitions | `data/swan/runs/socal/coarse/latest/pdir*.mat` | Direction per partition |
| Surfzone mesh | `data/surfzone/meshes/socal/socal_surfzone.npz` | High-res bathymetry |
| Wind data | `data/downloaded_weather_data/wind/gfs_*.nc` | GFS wind fields |

### Output Files

| File | Description |
|------|-------------|
| `breaking_field.npz` | Breaking locations, heights, types, density |
| `run_metadata.json` | Run configuration and timing |

---

## Step Size Considerations

The step size Δx affects accuracy and performance:

| Step Size | Pros | Cons |
|-----------|------|------|
| Δx << L | High accuracy, captures all depth changes | Slow, many interpolations |
| Δx ≈ L | Good balance for production runs | May miss some features |
| Δx > L | Fast iteration, good for testing | May miss rapid depth changes |

**Current approach**: Use Δx ≈ mesh resolution (20-50m near coast)
- Mesh can't represent finer bathymetry detail anyway
- Acceptable for initial development and fast iteration
- Can reduce step size for production runs later

### Wavelength Reference

| Period T (s) | L₀ deep (m) | L at h=10m (m) | L at h=5m (m) |
|--------------|-------------|----------------|---------------|
| 6 | 56 | 47 | 38 |
| 10 | 156 | 99 | 68 |
| 14 | 306 | 139 | 91 |
| 18 | 505 | 171 | 109 |

---

## Coordinate Systems

- **SWAN output**: Lon/Lat (WGS84)
- **Surfzone mesh**: UTM Zone 11N (meters) for SoCal
- **Ray tracing**: UTM (all calculations in meters)
- **Output**: Both UTM and Lon/Lat stored for flexibility

The mesh provides coordinate conversion methods:
- `lon_lat_to_utm(lon, lat)` → (x, y)
- `utm_to_lon_lat(x, y)` → (lon, lat)
