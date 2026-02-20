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
│           ├── ocean_currents.py  # WCOFS forecast + HF Radar observation currents
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
│   ├── swan/                 # SWAN model (see memory/data-pipelines.md)
│   ├── surfzone/             # Surfzone modeling (see memory/surfzone-physics.md)
│   ├── pipelines/            # Data fetching (see memory/data-pipelines.md)
│   ├── storage/              # Optional Zarr stores
│   └── cache/                # Cached GRIB/NetCDF data
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

Forward ray tracing with energy deposition. See `surfzone-physics.md` in memory for full details (components, commands, physics conventions).

## Data Flow

```
NOAA WaveWatch III (global)
  → WW3 boundary extraction (data/swan/ww3_endpoints/)
    → SWAN spectral model + WCOFS currents (data/swan/runs/{region}/{resolution}/)
      → Surfzone boundary conditions (SwanInputProvider)
        → Forward ray tracing with energy deposition (ForwardRayTracer)
          → Per-mesh-point wave heights (data/surfzone/output/{region}/)
            → API endpoints (backend/api/)
              → Frontend heatmaps (apps/web/)

Ocean Currents:
  WCOFS (ROMS, ~4km, 72hr forecast) → API /ocean-currents/grid → Frontend heatmap
  HF Radar (6km, hourly obs)        → API /ocean-currents/hfradar → Frontend heatmap
```

## Surf Spot Configuration

Spots defined in `data/spots/{region}.json`. Each spot has name, display name, and bounding box. Classes: `SurfSpot` and `BoundingBox` in `data/spots/spot.py`. Load with `load_spots_config("socal")`.

## Development Notes

### What Exists
- Complete data fetching infrastructure (NOAA, WW3, GFS, WCOFS, HF Radar, NDBC)
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

## Notes for Claude

1. **GEBCO data**: Located at `data/raw/bathymetry/gebco_2024/gebco_2024_california.nc`
2. **OOP approach**: Use classes for new functionality
3. **Build incrementally**: Work closely with user to design and implement features
4. **Data fetching is done**: Don't modify pipelines unless asked. See `data-pipelines.md` in memory for pipeline details
5. **Frontend is stable**: The 4 heatmaps work - don't modify unless asked
6. **Regions**: Three California regions defined in `data/regions/region.py`
7. **Directory conventions**: All region-specific data follows `{base_path}/{region}/` pattern
8. **Surfzone details**: Physics, conventions, simulation workflow, Numba constraints — see `surfzone-physics.md` in memory

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
