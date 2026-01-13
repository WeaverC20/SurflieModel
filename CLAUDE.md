# SurflieModel - Claude Code Guidelines

## Project Overview

SurflieModel is a surf forecast system that will use NOAA WaveWatch III (WW3) data propagated through SWAN (Simulating WAves Nearshore) models to predict surf conditions at specific beach locations.

**Current Status**: Data fetching infrastructure is complete. SWAN modeling is being rebuilt from scratch.

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

## Development Notes

### What Exists
- Complete data fetching infrastructure (NOAA, WW3, GFS, RTOFS, NDBC)
- Frontend dashboard with 4 working heatmaps
- GEBCO bathymetry viewing capability

### What Will Be Built (SWAN Modeling)
- SWAN domain creation from GEBCO bathymetry
- WW3 boundary condition extraction
- SWAN model execution
- Surf spot predictions

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

## Notes for Claude

1. **GEBCO data**: Located at `data/raw/bathymetry/gebco_2024/gebco_2024_california.nc`
2. **OOP approach**: Use classes for new functionality (see `data/bathymetry/gebco.py`)
3. **Build incrementally**: Work closely with user to design and implement features
4. **Data fetching is done**: Don't modify pipelines in `data/pipelines/` unless asked
5. **Frontend is stable**: The 4 heatmaps work - don't modify unless asked
