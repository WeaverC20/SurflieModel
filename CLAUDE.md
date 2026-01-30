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
│   ├── surfzone/             # Surfzone wave modeling
│   │   ├── mesh.py           # SurfZoneMesh class
│   │   ├── SURFZONE_MODEL.md # Technical documentation
│   │   └── runner/           # Ray tracing engine
│   │       ├── wave_physics.py            # Numba wave physics
│   │       ├── backward_ray_tracer.py     # Primary backward tracer
│   │       ├── backward_ray_tracer_debug.py # Visualization tool
│   │       ├── ray_tracer.py              # Legacy forward tracer (unused)
│   │       ├── swan_input_provider.py     # SWAN boundary conditions
│   │       └── output_writer.py           # Results storage
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
- **BackwardRayTracer** (`runner/backward_ray_tracer.py`): Primary wave propagation engine
- **wave_physics.py**: Numba-accelerated physics (shoaling, refraction, breaking)
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

**Usage:**

```python
from data.surfzone.runner.backward_ray_tracer import BackwardRayTracer

tracer = BackwardRayTracer(mesh, boundary_depth_threshold=50.0)  # 50m depth
result = tracer.trace_mesh_point(x, y, partitions)
print(f"Total Hs: {result.total_Hs:.2f}m")
```

**Visualization:**

```bash
venv/bin/python data/surfzone/runner/backward_ray_tracer_debug.py
```

See `docs/surfzone_wave_simulation_approach.md` for detailed documentation.

## Development Notes

### What Exists
- Complete data fetching infrastructure (NOAA, WW3, GFS, RTOFS, NDBC)
- Frontend dashboard with 4 working heatmaps
- GEBCO bathymetry viewing capability
- Surfzone mesh generation with spatial indexing
- Backward ray tracing for wave propagation (with correct physics)
- Wave physics (shoaling, refraction, breaking criteria)

### What Will Be Built
- Integration with live SWAN boundary conditions
- Breaking statistics and visualization
- Surf spot predictions at specific locations

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

# Run backward ray tracer visualization
venv/bin/python data/surfzone/runner/backward_ray_tracer_debug.py
```

## Notes for Claude

1. **GEBCO data**: Located at `data/raw/bathymetry/gebco_2024/gebco_2024_california.nc`
2. **OOP approach**: Use classes for new functionality (see `data/bathymetry/gebco.py`)
3. **Build incrementally**: Work closely with user to design and implement features
4. **Data fetching is done**: Don't modify pipelines in `data/pipelines/` unless asked
5. **Frontend is stable**: The 4 heatmaps work - don't modify unless asked
6. **Backward ray tracing**: Use `BackwardRayTracer` for wave propagation
   - Rays trace from near-shore toward deep water boundary
   - Direction and gradients are NEGATED to make rays bend toward faster C (deeper water)
   - See `data/surfzone/SURFZONE_MODEL.md` for physics details
7. **Wave physics**: Functions in `wave_physics.py` use standard formulas
   - `update_ray_direction()` uses forward formula: dθ/ds = -(1/C)·∂C/∂n
   - For backward tracing, pass NEGATED gradients to get correct behavior
8. **Legacy code**: `ray_tracer.py` is the old forward tracer - don't use
