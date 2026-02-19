# SurflieModel - Claude Code Guidelines

## Project Overview

SurflieModel is a surf forecast system that uses NOAA WaveWatch III (WW3) data propagated through ray tracing to predict surf conditions at specific beach locations.

**Current Status**: Data fetching infrastructure is complete. Surfzone modeling uses forward ray tracing with energy deposition.

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
│   │   ├── meshes/           # Generated surfzone meshes (by region)
│   │   │   └── {region}/     # socal/, central/, norcal/
│   │   ├── output/           # Simulation results (by region)
│   │   │   └── {region}/     # socal/, central/, norcal/
│   │   └── runner/           # Ray tracing engine
│   │       ├── run_simulation.py      # Main simulation CLI
│   │       ├── surfzone_runner.py     # Runner orchestration
│   │       ├── forward_ray_tracer.py  # Forward ray tracer with energy deposition
│   │       ├── wave_physics.py        # Numba wave physics
│   │       ├── swan_input_provider.py # SWAN boundary conditions
│   │       ├── surfzone_result.py     # Result dataclasses
│   │       └── output_writer.py       # Results storage
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
├── data/spots/                # Surf spot definitions (JSON per region)
│   ├── spot.py               # SurfSpot/BoundingBox classes
│   └── {region}.json         # Spot configs (e.g., socal.json)
│
├── apps/mobile/               # Expo React Native mobile app
├── packages/python/common/    # Shared Python utilities
├── scripts/                   # Utility scripts
│   ├── generate_surfzone_mesh.py  # Generate surfzone mesh for a region
│   └── dev/
│       └── view_surfzone_result.py # Interactive result viewer
├── .github/workflows/         # CI/CD (test, deploy, scheduled fetch)
└── docs/                      # Documentation
```

## Key Components

### GEBCO Bathymetry (`data/bathymetry/gebco.py`)

Object-oriented interface for GEBCO 2024 bathymetry data:

```python
from data.bathymetry import GEBCOBathymetry

# Load and view
gebco = GEBCOBathymetry()
gebco.view(lat_range=(32, 42), lon_range=(-126, -117))
```

### Regions (`data/regions/`)

California is divided into three modeling regions:

| Region | Name | Latitude | Longitude |
|--------|------|----------|-----------|
| socal | Southern California | 32.0 - 35.0 | -121.0 to -117.0 |
| central | Central California | 34.5 - 39.0 | -124.0 to -120.0 |
| norcal | Northern California | 38.5 - 42.0 | -126.0 to -122.0 |

### Surfzone Wave Modeling (`data/surfzone/`)

The surfzone module uses **forward ray tracing with energy deposition** to propagate waves from the SWAN boundary into the surfzone.

**Key Components:**

- **SurfZoneMesh** (`mesh.py`): Coastline-following mesh with bathymetry and spatial index
- **ForwardRayTracer** (`runner/forward_ray_tracer.py`): Forward ray tracing with energy deposition
- **ForwardSurfzoneRunner** (`runner/surfzone_runner.py`): Simulation orchestrator
- **wave_physics.py**: Numba-accelerated physics (shoaling, refraction, breaking)

**Running Simulations:**

```bash
# List available regions
python data/surfzone/runner/run_simulation.py --list-regions

# Run forward ray tracing (recommended)
python data/surfzone/runner/run_simulation.py --region socal --mode forward

# Configure ray density
python data/surfzone/runner/run_simulation.py --region socal --mode forward --boundary-spacing 100 --rays-per-point 32

# View results
python scripts/dev/view_surfzone_result.py --region socal
```

## Data Flow

```
NOAA WaveWatch III (global)
  → WW3 boundary extraction (data/swan/ww3_endpoints/)
    → SWAN spectral model (data/swan/runs/{region}/{resolution}/)
      → Surfzone boundary conditions (SwanInputProvider)
        → Forward ray tracing with energy deposition (ForwardRayTracer)
          → Per-mesh-point wave heights (data/surfzone/output/{region}/)
            → API endpoints (backend/api/)
              → Frontend heatmaps (apps/web/)
```

## Surf Spot Configuration

Spots defined in `data/spots/{region}.json`. Each spot has name, display name, and bounding box. Classes: `SurfSpot` and `BoundingBox` in `data/spots/spot.py`. Load with `load_spots_config("socal")`.

## Surfzone Physics Conventions

**Direction conventions:**
- SWAN outputs: nautical "coming FROM" (0=N, 90=E, clockwise)
- Ray tracer internal: math convention (0=E, counterclockwise)
- Conversion: `nautical_to_math()` / `math_to_nautical()` in `wave_physics.py`

**Coordinate systems:**
- GEBCO bathymetry and spot configs: lat/lon (WGS84)
- SurfZoneMesh and ray tracer: UTM (meters), converted via pyproj

**Energy conservation:** Power P = E × Cg × W conserved along each ray. Local Hs = sqrt(8E / (ρg)).

**Numba constraints** (`@njit` in `wave_physics.py`, `forward_ray_tracer.py`):
- No Python objects in `@njit` functions - NumPy arrays and scalars only
- `prange` for parallel loops - no shared mutable state
- Delete `__pycache__` if Numba cache becomes stale

## Development Notes

### What Exists
- Complete data fetching infrastructure (NOAA, WW3, GFS, RTOFS, NDBC)
- Frontend dashboard with 4 working heatmaps
- GEBCO bathymetry viewing capability
- Region definitions for socal, central, norcal
- SWAN model runs by region with multiple resolutions
- Surfzone mesh generation with spatial indexing
- Forward ray tracing with energy deposition
- Wave physics (shoaling, refraction, breaking criteria)
- Interactive result viewer

### Code Style
- Use object-oriented programming for new modules
- Keep things modular and testable
- Build incrementally with user collaboration

### Testing
- Python: pytest, black (formatting), isort (imports), mypy (types)
- TypeScript: eslint, prettier
- CI: `.github/workflows/test.yml` runs on push/PR to main and develop
- Run locally: `pytest backend/ packages/python/`

## Running the Project

```bash
# Backend
cd backend/api && uvicorn app.main:app --reload

# Frontend
cd apps/web && npm run dev

# Mobile
cd apps/mobile && npx expo start

# Tests
pytest backend/ packages/python/
```

### Surfzone Simulation Workflow

```bash
# 1. Generate surfzone mesh for a region
python scripts/generate_surfzone_mesh.py socal

# 2. Run SWAN model (requires WW3 boundary data)
python data/swan/run_swan.py --region socal --mesh coarse

# 3. Run surfzone simulation (forward mode)
python data/surfzone/runner/run_simulation.py --region socal --mode forward

# 4. View results
python scripts/dev/view_surfzone_result.py --region socal
```

## Notes for Claude

1. **GEBCO data**: Located at `data/raw/bathymetry/gebco_2024/gebco_2024_california.nc`
2. **OOP approach**: Use classes for new functionality
3. **Build incrementally**: Work closely with user to design and implement features
4. **Data fetching is done**: Don't modify pipelines in `data/pipelines/` unless asked
5. **Frontend is stable**: The 4 heatmaps work - don't modify unless asked
6. **Regions**: Three California regions defined in `data/regions/region.py`
7. **Directory conventions**: All region-specific data follows `{base_path}/{region}/` pattern
8. **Forward ray tracing**: Use `ForwardRayTracer` for wave propagation with energy deposition
9. **Surfzone runner**: Use `run_simulation.py --region {name} --mode forward`
10. **Direction conventions**: SWAN uses nautical FROM, ray tracer uses math convention - always convert
11. **Coordinate systems**: Spots/GEBCO use lat/lon, surfzone mesh uses UTM meters
12. **Numba code**: `@njit` functions can't use Python objects - only NumPy arrays and scalars
13. **Energy conservation**: P = E × Cg × W must be conserved in ray tracing modifications

## Environment Setup

Required for local dev (see `.env.example`):
- `NEXT_PUBLIC_MAPBOX_TOKEN` - Map rendering
- `NEXT_PUBLIC_API_URL` - Backend URL (default `http://localhost:8000/api/v1`)

Optional:
- `NOAA_API_KEY` - Live NOAA data fetching
- `DATABASE_URL` - PostgreSQL (defaults to SQLite)
- `REDIS_URL` - Celery task queue

## Claude Code Skills Setup

Skills are gitignored (treated like dependencies). Install with:

```bash
# Vercel React/Next.js skills
npx skills add vercel-labs/agent-skills --skill vercel-react-best-practices -y
npx skills add vercel-labs/next-skills -y

# Scientific Python, FastAPI, Python quality (manual clone + copy to .claude/skills/)
# See: ianhi/scientific-python-skills, Jeffallan/claude-skills, honnibal/claude-skills
```

Custom skills (`run-sim`, `fetch-data`) and hooks are defined in `.claude/hooks/` and `.claude/settings.json` (committed).
