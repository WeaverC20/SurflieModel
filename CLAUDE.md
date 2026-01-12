# SurflieModel - Claude Code Guidelines

This document provides critical context for Claude Code when working on this codebase.

## Project Overview

SurflieModel is a predictive surf forecast system that propagates NOAA WaveWatch III (WW3) spectral wave data through nested SWAN (Simulating WAves Nearshore) domains to predict surf conditions at specific beach locations.

### Pipeline Architecture

```
NOAA WW3 (25km offshore)
    ↓ [71 extraction points]
Outer SWAN Domain (14km resolution)
    ↓ [boundary conditions]
Inner SWAN Domain (1-2km resolution) [future]
    ↓ [boundary conditions]
Surf Spot Meshes (10-100m resolution) [future]
    ↓
Wave Breaking Prediction [future]
```

## Critical: Protected Structures

### DO NOT MODIFY WITHOUT EXPLICIT USER APPROVAL

The following structures are **immutable** and should never be changed without explicit user request:

1. **WW3 Extraction Points** (`regions/*/boundaries/ww3_extraction_points.json`)
   - These 71 points are "cemented" - they define where WW3 data feeds SWAN
   - The points are carefully positioned along the offshore boundary and around Channel Islands
   - Changing these points invalidates all historical runs and calibrations

2. **Mesh Versions** (files named `mesh_v*.grd`, `mesh_v*.nc`)
   - Meshes are versioned and immutable once created
   - To update a mesh, create a NEW version (e.g., `mesh_v2`)
   - Never overwrite existing mesh files

3. **Region Configuration Structure** (`regions/*/region.yaml`)
   - The hierarchical structure (region → domains → meshes) is intentional
   - Adding new regions is fine, but don't restructure existing ones

### Safe to Modify

- SWAN physics parameters in `domain.yaml`
- Adding new surf spots to `region.yaml`
- Creating new mesh versions
- Run scripts and processing pipelines
- Frontend/API code

## Directory Structure

```
SurflieModel/
├── regions/                    # PROTECTED - Region configurations
│   ├── california/            # California region
│   │   ├── region.yaml        # Master config (protected structure)
│   │   ├── boundaries/        # PROTECTED - Cemented boundary points
│   │   │   └── ww3_extraction_points.json
│   │   ├── domains/
│   │   │   ├── outer/         # Outer SWAN domain
│   │   │   │   └── domain.yaml
│   │   │   ├── inner/         # Inner domain (future)
│   │   │   └── spots/         # Surf spot meshes (future)
│   │   ├── bathymetry/        # Region-specific bathy products
│   │   └── runs/              # SWAN run outputs
│   └── _templates/            # Templates for new regions
│
├── data/                      # Data storage (legacy structure)
│   ├── grids/                 # Legacy mesh storage (being migrated)
│   ├── zarr/                  # Zarr data stores
│   └── swan_runs/             # Legacy run outputs
│
├── runs/                      # Execution scripts
│   ├── fetch_ww3.py          # Download WW3 data
│   └── run_swan.py           # Execute SWAN model
│
├── packages/python/           # Shared Python code
│   └── common/wave_forecast_common/
│       └── regions/           # Region data models
│
├── backend/api/              # FastAPI backend
├── apps/web/                 # Next.js frontend
└── ml/                       # Machine learning (future)
```

## Key Concepts

### Regions

A **Region** is the top-level geographic entity (e.g., "california"). Each region contains:
- Geographic bounds and sub-regions
- WW3 extraction point configuration (cemented)
- Domain hierarchy (outer → inner → spots)
- Bathymetry source priorities
- Surf spot definitions

### Domains

**Domains** are SWAN computational grids at different resolutions:
- **Outer**: Coarse mesh (~14km) bridging WW3 to nearshore
- **Inner**: Fine mesh (~1-2km) for nearshore propagation
- **Spots**: Very fine mesh (~10-100m) for surf spot prediction

### Extraction Points

The **71 WW3 extraction points** for California are:
- 39 points along the offshore boundary (~25km from coast)
- 32 points around 8 Channel Islands (4 per island)

These points are where WW3 spectral data is interpolated to provide SWAN boundary conditions.

## Working with This Codebase

### Adding a New Region

1. Copy `regions/_templates/` to `regions/new_region_name/`
2. Update `region.yaml` with geographic bounds
3. Generate extraction points using appropriate offshore distance
4. Create outer domain mesh using bathymetry pipeline
5. Register in API if needed

### Updating SWAN Physics

Safe to modify `regions/*/domains/*/domain.yaml`:
```yaml
swan_physics:
  breaking:
    gamma: 0.73  # Can adjust this
```

### Creating New Mesh Versions

1. Generate new mesh with updated bathymetry or resolution
2. Save as `mesh_v2.grd` and `mesh_v2.nc`
3. Add version entry to `domain.yaml` under `meshes:`
4. Update `active_mesh:` to point to new version
5. Keep old versions for reproducibility

## Python Usage

```python
from wave_forecast_common.regions import RegionRegistry

# Load a region
registry = RegionRegistry()
california = registry.load_region("california")

# Get extraction points
points = california.get_extraction_points_as_tuples()

# Get outer domain
outer = california.get_outer_domain()
mesh_path = outer.active_mesh.grd_file
```

## Common Tasks

### Run Daily Forecast

```bash
# 1. Fetch latest WW3 data
python runs/fetch_ww3.py --region california

# 2. Run SWAN with WW3 boundary conditions
python runs/run_swan.py --domain california_outer
```

### View Boundary Points

```bash
python -m data.pipelines.swan.ww3_boundary --points california_swan_14000m
```

## Spectral Configuration Reference

The wave spectrum is discretized into frequency and direction bins:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_frequencies` | 36 | Number of frequency bins |
| `freq_min_hz` | 0.04 | Minimum frequency (25s period) |
| `freq_max_hz` | 1.0 | Maximum frequency (1s period) |
| `n_directions` | 36 | Number of direction bins (10° each) |
| `spreading_degrees` | 25 | Directional spreading parameter |

## Notes for Claude

1. **Always check `regions/` first** when asked about configuration
2. **Never modify extraction points** without explicit approval
3. **Prefer creating new versions** over modifying existing meshes
4. **The 71 points are intentional** - don't try to "optimize" them
5. **Legacy structure in `data/grids/`** is being migrated to `regions/`
