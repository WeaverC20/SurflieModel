# SurflieModel - Claude Code Guidelines

## Project Overview

SurflieModel is a surf forecast system that uses NOAA WaveWatch III (WW3) data propagated through backward ray tracing to predict surf conditions at specific beach locations.

**Current Status**: Data fetching infrastructure is complete. Surfzone modeling uses backward ray tracing from mesh points to SWAN boundary.

## Directory Structure

```
SurflieModel/
├── apps/web/                  # Next.js frontend dashboard
│   └── src/
│       ├── app/               # Pages (4 heatmap panels)
│       └── components/map/    # Map visualization components
│
├── backend/api/               # FastAPI backend
│   └── app/
│       ├── main.py           # API entry point
│       └── routers/          # API endpoints
│           ├── buoys.py      # NDBC buoy data
│           ├── waves.py      # WaveWatch III data
│           ├── wind.py       # GFS wind data
│           ├── ocean_currents.py  # RTOFS current data
│           └── dev.py        # Development endpoints
│
├── data/
│   ├── raw/bathymetry/       # Raw bathymetry files
│   │   └── gebco_2024/
│   │       └── gebco_2024_california.nc  # GEBCO 2024 data
│   │
│   ├── bathymetry/           # Bathymetry processing (OOP)
│   │   └── gebco.py          # GEBCOBathymetry class
│   │
│   ├── regions/              # Geographic region definitions
│   │   └── region.py         # Region class (SOCAL, NORCAL, CENTRAL_CAL)
│   │
│   ├── meshes/               # SWAN computational meshes (by region)
│   │   ├── socal/            # coarse, medium, fine, ultrafine
│   │   ├── central/          # coarse, fine
│   │   └── norcal/           # coarse, fine
│   │
│   ├── swan/                 # SWAN model data
│   │   ├── runs/             # SWAN run outputs (by region)
│   │   │   ├── socal/{resolution}/latest/
│   │   │   ├── central/{resolution}/latest/
│   │   │   └── norcal/{resolution}/latest/
│   │   ├── ww3_endpoints/    # WW3 boundary extraction points
│   │   │   └── {region}/ww3_boundaries.json
│   │   └── run_swan.py       # SWAN runner script
│   │
│   ├── surfzone/             # Surfzone wave modeling
│   │   ├── mesh.py           # SurfZoneMesh class
│   │   ├── SURFZONE_MODEL.md # Technical documentation
│   │   ├── meshes/           # Generated surfzone meshes (by region)
│   │   │   └── {region}/     # socal/, central/, norcal/
│   │   ├── output/           # Simulation results (by region)
│   │   │   └── {region}/     # socal/, central/, norcal/
│   │   └── runner/           # Ray tracing engine
│   │       ├── run_simulation.py      # Main simulation CLI
│   │       ├── surfzone_runner.py     # SurfzoneRunner orchestration
│   │       ├── backward_ray_tracer.py # Primary backward tracer
│   │       ├── forward_propagation.py # Forward wave propagation
│   │       ├── wave_physics.py        # Numba wave physics
│   │       ├── swan_input_provider.py # SWAN boundary conditions
│   │       ├── surfzone_result.py     # Result dataclasses
│   │       ├── output_writer.py       # Results storage
│   │       ├── backward_ray_tracer_debug.py # Visualization tool
│   │       └── ray_tracer.py          # Legacy forward tracer (unused)
│   │
│   ├── pipelines/            # Data fetching pipelines
│   │   ├── noaa/             # NOAA data (tides, WW3, GFS)
│   │   ├── wave/             # WaveWatch III fetcher
│   │   ├── wind/             # GFS wind fetcher
│   │   ├── buoy/             # NDBC buoy fetcher
│   │   └── ocean_tiles/      # RTOFS ocean current fetcher
│   │
│   ├── storage/              # Data storage utilities
│   └── cache/                # Cached data
│
├── packages/python/common/    # Shared Python utilities
├── scripts/                   # Utility scripts
│   ├── generate_surfzone_mesh.py  # Generate surfzone mesh for a region
│   └── dev/
│       └── view_surfzone_result.py # Interactive result viewer
└── docs/                      # Documentation
    └── surfzone_wave_simulation_approach.md  # Detailed backward tracing docs
```

## Key Components

### GEBCO Bathymetry (`data/bathymetry/gebco.py`)

Object-oriented interface for GEBCO 2024 bathymetry data:

```python
from data.bathymetry import GEBCOBathymetry

# Load and view
gebco = GEBCOBathymetry()
gebco.view(lat_range=(32, 42), lon_range=(-126, -117))

# Or use quick function
from data.bathymetry.gebco import view_gebco
view_gebco(lat_range=(32, 42), lon_range=(-126, -117))
```

**GEBCO file location**: `data/raw/bathymetry/gebco_2024/gebco_2024_california.nc`

### Regions (`data/regions/`)

California is divided into three modeling regions for SWAN and surfzone simulations:

| Region | Name | Latitude | Longitude | Description |
|--------|------|----------|-----------|-------------|
| socal | Southern California | 32.0 - 35.0 | -121.0 to -117.0 | Mexico border to Point Conception |
| central | Central California | 34.5 - 39.0 | -124.0 to -120.0 | Point Conception to Point Reyes |
| norcal | Northern California | 38.5 - 42.0 | -126.0 to -122.0 | Point Reyes to Oregon border |

**Usage:**
```python
from data.regions.region import get_region, REGIONS

# Get a specific region
socal = get_region("socal")
print(f"{socal.display_name}: lat {socal.lat_range}, lon {socal.lon_range}")

# List all regions
for name in ['socal', 'central', 'norcal']:
    region = REGIONS[name]
    print(f"{name}: {region.display_name}")
```

### Data Pipelines

All data fetching is handled by pipeline modules in `data/pipelines/`:

- **NOAA** (`noaa/`): Tide predictions, WW3 waves, GFS wind
- **Buoy** (`buoy/`): NDBC real-time buoy observations
- **Wave** (`wave/`): WaveWatch III GRIB2 fetcher
- **Wind** (`wind/`): GFS wind GRIB2 fetcher
- **Ocean Tiles** (`ocean_tiles/`): RTOFS ocean currents

### Frontend Dashboard

The frontend displays 4 real-time heatmaps:
1. Wind Forecast (GFS)
2. Wave Forecast (WW3)
3. Ocean Currents (RTOFS)
4. Buoy Observations (NDBC)

### Surfzone Wave Modeling (`data/surfzone/`)

The surfzone module uses **backward ray tracing** to propagate waves from near-shore mesh points back to the SWAN boundary (deep water).

**Key Components:**

- **SurfZoneMesh** (`mesh.py`): Coastline-following mesh with bathymetry and spatial index
- **SurfzoneRunner** (`runner/surfzone_runner.py`): Main simulation orchestrator
- **BackwardRayTracer** (`runner/backward_ray_tracer.py`): Primary wave propagation engine
- **wave_physics.py**: Numba-accelerated physics (shoaling, refraction, breaking)
- **SwanInputProvider** (`runner/swan_input_provider.py`): SWAN partition data interpolation
- **backward_ray_tracer_debug.py**: Visualization tool for ray paths

**Backward Ray Tracing Physics:**

Rays are traced BACKWARD from near-shore points toward deep water. Key differences from forward tracing:

1. **Direction is NEGATED** - rays point away from shore (opposite of wave travel)
2. **Gradients are NEGATED** - rays bend toward FASTER celerity (deeper water)

```python
# Forward: bends toward slower C (shallow)
dθ/ds = -(1/C) · ∂C/∂n

# Backward: bends toward faster C (deep) - achieved by negating gradients
dx, dy = update_ray_direction(dx, dy, C, -dC_dx, -dC_dy, step_size)
```

**Running Simulations (Multi-Region):**

```bash
# List available regions and their status
python data/surfzone/runner/run_simulation.py --list-regions

# Run simulation for a region (auto-detects mesh and SWAN)
python data/surfzone/runner/run_simulation.py --region socal

# Specify SWAN resolution
python data/surfzone/runner/run_simulation.py --region socal --swan-resolution fine

# Sample subset for fast iteration
python data/surfzone/runner/run_simulation.py --region socal --sample-fraction 0.1

# View results
python scripts/dev/view_surfzone_result.py --region socal
python scripts/dev/view_surfzone_result.py --list-regions
```

**Visualization:**

```bash
venv/bin/python data/surfzone/runner/backward_ray_tracer_debug.py
```

See `data/surfzone/SURFZONE_MODEL.md` for detailed documentation.

## Development Notes

### What Exists
- Complete data fetching infrastructure (NOAA, WW3, GFS, RTOFS, NDBC)
- Frontend dashboard with 4 working heatmaps
- GEBCO bathymetry viewing capability
- Region definitions for socal, central, norcal (with overlapping boundaries)
- SWAN model runs by region with multiple resolutions
- Surfzone mesh generation with spatial indexing (multi-region support)
- Backward ray tracing for wave propagation (with correct physics)
- Wave physics (shoaling, refraction, breaking criteria)
- Convergence-based ray tracing with SWAN boundary lookup
- Forward wave propagation along traced paths
- Interactive result viewer with datashader

### What Will Be Built
- Breaking statistics and visualization
- Surf spot predictions at specific locations
- Meshes for central and norcal regions

### Code Style
- Use object-oriented programming for new modules
- Keep things modular and testable
- Build incrementally with user collaboration

## Running the Project

```bash
# Backend (from project root)
cd backend/api
uvicorn app.main:app --reload

# Frontend (from project root)
cd apps/web
npm run dev
```

### Surfzone Simulation Workflow

```bash
# 1. Generate surfzone mesh for a region
python scripts/generate_surfzone_mesh.py socal
python scripts/generate_surfzone_mesh.py --list-regions

# 2. Run SWAN model (requires WW3 boundary data)
python data/swan/run_swan.py --region socal --mesh coarse

# 3. Run surfzone simulation
python data/surfzone/runner/run_simulation.py --region socal
python data/surfzone/runner/run_simulation.py --list-regions

# 4. View results
python scripts/dev/view_surfzone_result.py --region socal
python scripts/dev/view_surfzone_result.py --list-regions

# Debug visualization
venv/bin/python data/surfzone/runner/backward_ray_tracer_debug.py
```

## Notes for Claude

1. **GEBCO data**: Located at `data/raw/bathymetry/gebco_2024/gebco_2024_california.nc`
2. **OOP approach**: Use classes for new functionality (see `data/bathymetry/gebco.py`)
3. **Build incrementally**: Work closely with user to design and implement features
4. **Data fetching is done**: Don't modify pipelines in `data/pipelines/` unless asked
5. **Frontend is stable**: The 4 heatmaps work - don't modify unless asked
6. **Regions**: Three California regions defined in `data/regions/region.py`:
   - `socal` (Southern California): 32.0-35.0°N
   - `central` (Central California): 34.5-39.0°N
   - `norcal` (Northern California): 38.5-42.0°N
   - Use `get_region(name)` to access region configuration
7. **Directory conventions**: All region-specific data follows `{base_path}/{region}/` pattern:
   - SWAN runs: `data/swan/runs/{region}/{resolution}/latest/`
   - Surfzone meshes: `data/surfzone/meshes/{region}/`
   - Surfzone output: `data/surfzone/output/{region}/`
8. **Backward ray tracing**: Use `BackwardRayTracer` for wave propagation
   - Rays trace from near-shore toward deep water boundary
   - Direction and gradients are NEGATED to make rays bend toward faster C (deeper water)
   - See `data/surfzone/SURFZONE_MODEL.md` for physics details
9. **Surfzone runner**: Use `run_simulation.py --region {name}` to run simulations
   - Auto-detects mesh and SWAN paths based on region
   - Results saved to `data/surfzone/output/{region}/`
10. **Wave physics**: Functions in `wave_physics.py` use standard formulas
    - `update_ray_direction()` uses forward formula: dθ/ds = -(1/C)·∂C/∂n
    - For backward tracing, pass NEGATED gradients to get correct behavior
11. **Legacy code**: `ray_tracer.py` is the old forward tracer - don't use
