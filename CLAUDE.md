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
│   │   ├── statistics/       # Wave statistics (see memory/statistics.md)
│   │   │   ├── run_statistics.py   # CLI entry point
│   │   │   ├── runner.py           # StatisticsRunner orchestration
│   │   │   ├── registry.py         # Pluggable @register decorator
│   │   │   ├── spectral_utils.py   # Shared spectral moment reconstruction
│   │   │   └── {stat_name}.py      # Individual statistics (7 total)
│   │   └── ...
│   ├── pipelines/            # Data fetching (see memory/data-pipelines.md)
│   ├── storage/              # Optional Zarr stores
│   └── cache/                # Cached GRIB/NetCDF data
│
├── data/spots/                # Surf spot definitions (JSON per region)
│   ├── spot.py               # SurfSpot/BoundingBox classes
│   ├── spot_statistics.py    # SpotStatisticsAggregator (bbox → mesh point stats)
│   └── {region}.json         # Spot configs (e.g., socal.json)
│
├── tools/
│   └── viewer/                # Panel-based dev viewer (see memory/viewer.md)
│       ├── app.py            # DevViewerApp — main Panel application
│       ├── config.py         # Theme, colormaps, styling constants
│       ├── data_manager.py   # Centralized lazy-loading data cache
│       ├── components/       # Reusable UI components
│       │   ├── colorbar.py   # Matplotlib colorbar renderer
│       │   └── point_inspector.py  # KDTree click-to-inspect
│       └── views/            # View panels
│           ├── base.py       # BaseView abstract class
│           ├── swan_view.py  # SWAN output (Plotly heatmaps, buoy markers)
│           ├── mesh_view.py  # Surfzone mesh (Datashader, depth coloring)
│           └── result_view.py # Simulation results (Datashader, spot editing)
│
├── apps/mobile/               # Expo React Native mobile app
├── packages/python/common/    # Shared Python utilities
├── scripts/                   # Utility scripts
│   ├── generate_surfzone_mesh.py  # Generate surfzone mesh for a region
│   └── dev/
│       └── viewer.py         # CLI entry point for dev viewer
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

### Surfzone Statistics (`data/surfzone/statistics/`)

Pluggable registry of 7 vectorized statistics computed over all mesh points. Each statistic is a class registered via `@StatisticsRegistry.register`. Shared spectral moment reconstruction in `spectral_utils.py`. See `statistics.md` in memory for physics details.

```bash
# Run all statistics
python data/surfzone/statistics/run_statistics.py --region socal

# Run specific statistics
python data/surfzone/statistics/run_statistics.py --region socal --statistics set_period waves_per_set
```

Statistics: set_period, waves_per_set, set_duration, lull_duration, peakiness, groupiness_factor, height_amplification. Output: CSV + metadata JSON in `data/surfzone/output/{region}/`.

### Dev Viewer (`tools/viewer/`)

Panel-based interactive tool with 3 views for inspecting SWAN output, surfzone meshes, and simulation results. See `viewer.md` in memory for full details.

```bash
python scripts/dev/viewer.py --region socal
python scripts/dev/viewer.py --region socal --view "Surfzone Results" --lonlat
```

Key features: Datashader/Plotly heatmaps, KDTree point inspection, spot bounding box editing (persisted to JSON), buoy markers with spectral data, statistics overlay, ray path overlay, zoom/pan persistence across views.

## Data Flow

```
NOAA WaveWatch III (global)
  → WW3 boundary extraction (data/swan/ww3_endpoints/)
    → SWAN spectral model + WCOFS currents (data/swan/runs/{region}/{resolution}/)
      → Surfzone boundary conditions (SwanInputProvider)
        → Forward ray tracing with energy deposition (ForwardRayTracer)
          → Per-mesh-point wave heights (data/surfzone/output/{region}/)
            → Statistics (data/surfzone/statistics/) → CSV + metadata JSON
            → API endpoints (backend/api/)
              → Frontend heatmaps (apps/web/)

Dev Viewer (tools/viewer/):
  SWAN output + Surfzone mesh + Ray tracing results + Statistics CSV
    → Panel app (scripts/dev/viewer.py) → localhost:5007

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
- Surfzone wave statistics (7 pluggable statistics, vectorized computation)
- Spot-level statistics aggregation (`data/spots/spot_statistics.py`)
- Dev viewer tool (Panel, 3 views: SWAN/Mesh/Results, spot editing, buoy data)

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
9. **Statistics**: 7 pluggable wave statistics with spectral reconstruction — see `statistics.md` in memory
10. **Dev viewer**: Panel tool with 3 views, spot editing, buoy integration — see `viewer.md` in memory

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
