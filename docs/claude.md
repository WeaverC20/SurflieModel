# Wave Forecast App - Project Context for Claude

## Overview

A wave forecasting platform with web/mobile apps, custom ML models trained on NOAA data, and private performance analysis. The goal is to deploy a public-facing app that provides surf forecasts while maintaining private analysis comparing performance vs Surfline.

**Mission**: Provide accurate, accessible wave forecasts for surfers worldwide using custom ML models and open data sources.

## Tech Stack

### Frontend
- **Web**: Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS
- **Mobile**: Expo (React Native), Expo Router, TypeScript
- **Shared**: `@wave-forecast/api-client` package for API communication

### Backend
- **API**: FastAPI, Python 3.11+, Pydantic v2, SQLAlchemy
- **Worker**: Python scheduler for periodic data fetching and predictions
- **Database**: PostgreSQL (production), SQLite (local development)
- **Cache**: Redis (for forecast caching)

### ML/Data
- **Training**: PyTorch, scikit-learn, pandas, MLflow
- **Inference**: Lightweight module (`ml/inference/`) deployed with backend
- **Data Sources**: NOAA (Wave Watch III, GFS), NDBC buoys, (Surfline for private comparison only)

### Deployment
- **Web**: Vercel (auto-deploy from main branch)
- **Backend**: Railway or Fly.io (to be decided)
- **Mobile**: EAS (Expo Application Services) → App Store & Google Play
- **ML Models**: S3 or Google Cloud Storage (loaded at runtime, never in git)
- **Database**: Railway Postgres or Supabase

### Development
- **Monorepo**: pnpm workspaces (TypeScript), pip install -e (Python)
- **Version Control**: Git, GitHub
- **CI/CD**: GitHub Actions
- **No Docker**: Prefer native tooling for development

## Architecture Principles

### 1. Separation of Production vs Research Code

**Production Code** (deployed):
- `apps/` - Web and mobile apps
- `backend/api/` - FastAPI service
- `backend/worker/` - Background jobs
- `ml/inference/` - Prediction module
- `packages/` - Shared utilities

**Research/Development Code** (never deployed):
- `ml/training/` - Model training and experimentation
- `data/notebooks/` - Data exploration
- `data/analysis/` - **Private analysis (your performance vs Surfline)**
- `data/pipelines/surfline/` - Surfline fetching (for comparison only)

**CRITICAL**: Code in `data/analysis/` and `data/pipelines/surfline/` is NEVER imported by production code. This is your private workspace for validation and comparison.

### 2. ML Model Management

- **Models are never committed to git** (too large, change frequently)
- **Training**: Happens in `ml/training/`, models uploaded to S3
- **Inference**: `ml/inference/` loads models from S3 at runtime
- **Development**: Models can be saved to `ml/artifacts/` (gitignored) for local testing
- **Model URL format**: `s3://bucket-name/models/model_v1.pt` or `gs://bucket-name/models/model_v1.pt`

### 3. Data Flow

```
NOAA API → data/pipelines/noaa/ → Database
NDBC Buoys → data/pipelines/buoy/ → Database
                                      ↓
Backend Worker (scheduler) → Fetches data → Runs ML predictions → Cache (Redis)
                                                                       ↓
Backend API → Reads cache/DB → Returns forecasts → Frontend (web/mobile)
```

### 3a. Data Sources Strategy

#### Current Implementation (Phase 1)

**Tide Data**:
- **Source**: NOAA CO-OPS Harmonic Constituents
- **Coverage**: US only
- **License**: Free and open, no license required
- **Accuracy**: Highly accurate for US coastal waters
- **Future**: Consider TPXO / FES for global coverage (requires license, TPXO may be free after request)

**Swell Height and Direction**:
- **Source**: NOAA Wave Watch 3
- **Coverage**: US-focused but global data available
- **License**: Free and open
- **Forecast Range**: Up to 16 days
- **Status**: ✅ GRIB2 parsing implemented with cfgrib/xarray
- **NaN Handling**: Automatically falls back to nearest valid ocean point for coastal locations
- **Future**: Consider GEFS-WAVE for ensemble uncertainty forecasts (16-day with probabilistic outputs)

**Wind Data**:
- **Source**: NOAA GFS (Global Forecast System)
- **Coverage**: Global
- **License**: Free and open
- **Forecast Range**: Up to 16 days
- **Resolution**: 0.25 degree (~25km)
- **Status**: ✅ GRIB2 parsing implemented with cfgrib/xarray
- **Parsed Values**: Wind speed (m/s and knots), direction, gusts
- **Future**: Transition to GEFS for ensemble uncertainty

#### Future Enhancements (Phase 2+)

**Regional High-Resolution Wind Models**:
- NAM (U.S.) - 3km resolution
- HRRR (U.S.) - 3km hourly updates
- AROME (Europe) - 1.3km resolution
- HARMONIE-AROME (Nordic) - 2.5km resolution
- ACCESS (Australia) - Regional coverage

**Premium Data** (Optional, Paid):
- ECMWF - Superior global forecasts but requires paid subscription
- Only consider if forecasts demonstrate clear accuracy improvements

#### Data Validation Sources

**For Model Training and Accuracy Assessment** (Private use only):
- CDIP buoy data - High-quality directional wave data
- NDBC buoy data - Extensive network of observation buoys
- Surfline forecasts - For private comparison only (see `data/analysis/`)

**Important**: Buoy data and Surfline comparisons are for research and validation only. Never expose Surfline data through the public API.

### 4. Shared Code Strategy

**TypeScript**:
- Types and API client in `packages/typescript/api-client/`
- Imported by web and mobile apps
- Eventually auto-generate types from FastAPI OpenAPI schema

**Python**:
- Common utilities in `packages/python/common/`
- Installed with `pip install -e packages/python/common`
- Used by backend, worker, ML inference, and data pipelines

## Project Structure

```
wave-forecast-app/
├── apps/
│   ├── web/                    # Next.js web app → Vercel
│   └── mobile/                 # Expo mobile app → EAS
├── backend/
│   ├── api/                    # FastAPI REST API → Railway
│   │   ├── app/
│   │   │   ├── main.py        # Entry point
│   │   │   ├── routers/       # API endpoints
│   │   │   ├── models/        # Pydantic schemas
│   │   │   ├── services/      # Business logic
│   │   │   └── db/            # Database models
│   │   └── requirements.txt
│   └── worker/                 # Background tasks → Railway
│       ├── scheduler.py       # Main scheduler
│       └── tasks/             # Task definitions
├── ml/
│   ├── training/              # Model training (not deployed)
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── datasets/
│   │   ├── configs/
│   │   └── models/            # Model architectures
│   ├── inference/             # Production inference (deployed with backend)
│   │   ├── predictor.py       # Main API: predict()
│   │   └── utils.py
│   ├── artifacts/             # Local models (gitignored)
│   └── experiments/           # MLflow logs (gitignored)
├── data/
│   ├── pipelines/             # Data fetching ETL
│   │   ├── noaa/              # NOAA data fetching
│   │   ├── buoy/              # Buoy data fetching
│   │   └── surfline/          # Surfline (private use only!)
│   ├── notebooks/             # General exploration
│   └── analysis/              # **PRIVATE: Your vs Surfline analysis**
├── packages/
│   ├── typescript/
│   │   └── api-client/        # Shared TypeScript API client
│   └── python/
│       └── common/            # Shared Python utilities
├── .github/workflows/         # CI/CD
├── docs/
│   └── claude.md              # This file!
└── scripts/                   # Dev helper scripts
```

## Key Files and Locations

### When you need to...

**Add a new API endpoint**:
- Create router in `backend/api/app/routers/`
- Add Pydantic schemas in `backend/api/app/models/`
- Import router in `backend/api/app/main.py`

**Update forecast logic**:
- Business logic in `backend/api/app/services/`
- ML prediction calls `ml/inference/predictor.py`

**Add a new data source**:
- Create fetcher in `data/pipelines/<source>/fetcher.py`
- Add to worker tasks in `backend/worker/tasks/`
- Schedule in `backend/worker/scheduler.py`

**Train a new model**:
- Work in `ml/training/`
- Save best model to S3: `aws s3 cp model.pt s3://bucket/models/`
- Update model URL in backend config

**Add shared utility**:
- Python: `packages/python/common/wave_forecast_common/utils.py`
- TypeScript: `packages/typescript/api-client/src/`

**Private analysis (vs Surfline)**:
- Create notebook in `data/analysis/`
- Use Surfline fetcher from `data/pipelines/surfline/`
- **Never import this code in production!**

## Code Conventions

### Python
- **Formatter**: Black (line length 100)
- **Import sorting**: isort (black profile)
- **Type hints**: Required for all functions
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes
- **Docstrings**: Google style for public APIs
- **Async**: Prefer async/await for I/O operations

### TypeScript
- **Formatter**: Prettier
- **Linter**: ESLint (Next.js config)
- **Naming**: `camelCase` for functions/variables, `PascalCase` for components/types
- **Strict mode**: Enabled
- **React**: Functional components with hooks

### API Design
- **REST**: `/api/v1/` prefix for all endpoints
- **Versioning**: Include version in URL
- **Response format**: JSON with consistent structure
- **Error handling**: Proper HTTP status codes and error messages

## Environment Variables

See [.env.example](.env.example) for all required variables.

**Critical ones**:
- `DATABASE_URL` - PostgreSQL connection
- `NOAA_API_KEY` - NOAA API access (if required)
- `MODEL_STORAGE_BUCKET` - S3 bucket for ML models
- `REDIS_URL` - Redis for caching
- `NEXT_PUBLIC_API_URL` - API URL for frontend

## Development Workflow

### Initial Setup
```bash
./scripts/setup.sh  # One-time setup
```

### Daily Development
```bash
# Terminal 1: Backend API
cd backend/api
source ../../venv/bin/activate
uvicorn app.main:app --reload

# Terminal 2: Web app
pnpm dev:web

# Terminal 3: Worker (optional)
cd backend/worker
source ../../venv/bin/activate
python scheduler.py
```

Or use the helper script:
```bash
./scripts/dev.sh api      # Start API
./scripts/dev.sh web      # Start web
./scripts/dev.sh mobile   # Start mobile
```

### Before Committing
```bash
# Format code
./scripts/dev.sh format

# Run tests
./scripts/dev.sh test

# Check linting
./scripts/dev.sh lint
```

## Deployment

### Web (Vercel)
- Connect GitHub repo to Vercel
- Auto-deploys on push to `main`
- Root directory: `apps/web`

### Backend (Railway)
- Connect GitHub repo
- Deploy `backend/api` as a service
- Deploy `backend/worker` as a separate service
- Set environment variables in Railway dashboard

### Mobile (EAS)
```bash
cd apps/mobile
eas build --platform ios
eas build --platform android
eas submit --platform ios
eas submit --platform android
```

### ML Models
```bash
# After training
aws s3 cp ml/artifacts/model.pt s3://your-bucket/models/model_v1.pt

# Update backend config
# In Railway dashboard or .env:
MODEL_URL=s3://your-bucket/models/model_v1.pt
```

## Common Tasks & Patterns

### Adding a New Surf Spot

1. Add spot data to database (via migration or API)
2. Update spot types in `packages/python/common/wave_forecast_common/constants.py`
3. Frontend will automatically fetch from API

### Updating Forecast Logic

1. Modify `ml/inference/predictor.py` for ML predictions
2. Or modify `backend/api/app/services/forecast_service.py` for business logic
3. Update Pydantic schemas in `backend/api/app/models/`
4. Update TypeScript types in `packages/typescript/api-client/src/types.ts`

### Fetching Historical Data

```python
# In data/notebooks/ or data/pipelines/
from data.pipelines.noaa import NOAAFetcher
from data.pipelines.buoy import BuoyFetcher

fetcher = NOAAFetcher()
data = await fetcher.fetch_wavewatch_data(lat, lon, timestamp)
```

### Running Private Analysis

```python
# In data/analysis/my_analysis.ipynb
from data.pipelines.surfline import SurflineFetcher  # (when implemented)
from ml.inference import WaveForecastPredictor

# Compare your model vs Surfline
my_predictions = predictor.predict(...)
surfline_data = surfline_fetcher.fetch(...)

# Analyze differences
# This code NEVER goes to production!
```

## Important Reminders

### 🚨 Never Do This
- ❌ Commit ML model files (*.pt, *.pkl, *.h5) to git
- ❌ Import code from `data/analysis/` in production
- ❌ Use Surfline data in the public app
- ❌ Commit .env files or secrets
- ❌ Push to main without tests passing
- ❌ Use Docker (we prefer native tooling)

### ✅ Always Do This
- ✅ Load ML models from S3 at runtime
- ✅ Keep private analysis in `data/analysis/`
- ✅ Use type hints (Python) and strict mode (TypeScript)
- ✅ Write tests for new features
- ✅ Update this claude.md when making architectural decisions
- ✅ Use shared packages for common code

## Testing Strategy

### Backend
```bash
cd backend/api
pytest tests/ -v
```

### Frontend
```bash
pnpm test  # Run all tests
pnpm test:watch  # Watch mode
```

### ML Models
```bash
cd ml/training
python evaluate.py --model path/to/model --test-data path/to/test.csv
```

## Performance Considerations

- **Cache forecasts** in Redis (15-30 min TTL)
- **Batch predictions** for multiple spots/times
- **Use async** for all I/O operations
- **Preload ML models** on worker startup
- **Index database** on spot_id, timestamp

## Security Notes

- NOAA data is public and free to use
- NDBC buoy data is public
- Surfline data is for **private analysis only** - respect their ToS
- Never expose API keys in frontend code
- Use environment variables for all secrets
- Validate all user input in API

## Surf Zone Forecasting Architecture (SWAN Integration)

### Overview

The ultimate goal is to predict **actual surf conditions** at specific spots by propagating offshore wave energy through multiple model domains with increasing resolution, ultimately outputting **swell component breakdowns** (e.g., "1.9ft at 15s @ 250°, 0.7ft at 9s @ 180°").

### Multi-Resolution Wave Propagation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Wave Watch 3 (WW3) - NOAA Global Wave Model                                │
│  Resolution: ~0.25° (~25km)                                                 │
│  Domain: Global/Pacific basin                                               │
│  Output: Wave spectra at domain edges → Boundary conditions for SWAN        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SWAN Model - Regional Wave Propagation                                     │
│  Resolution: ~100-500m (variable)                                           │
│  Domain: ~50-200km offshore to ~1-2km from shore                            │
│  Purpose: Propagate wave spectra accounting for:                            │
│    - Refraction (bathymetry-induced bending)                                │
│    - Shoaling (wave height changes with depth)                              │
│    - Wind-wave generation                                                   │
│    - White-capping dissipation                                              │
│  Output: Nearshore wave spectra with swell component partitioning           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Surf Zone Physics Model - Nearshore Breaking                               │
│  Resolution: 1-50m (using USACE LiDAR bathymetry)                           │
│  Domain: ~1km from shore to beach                                           │
│  Purpose: Model actual wave breaking physics:                               │
│    - Breaking wave height (Hb)                                              │
│    - Surf zone width                                                        │
│    - Wave setup/setdown                                                     │
│    - Longshore currents                                                     │
│  Output: Surfable wave height, quality metrics, swell breakdown             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Swell Component Output Format

The final output should decompose total wave energy into discrete swell trains:

```python
{
    "spot_id": "huntington-pier",
    "timestamp": "2024-12-19T12:00:00Z",
    "swell_components": [
        {"height_ft": 1.9, "period_s": 15, "direction_deg": 250, "energy_pct": 65},
        {"height_ft": 0.7, "period_s": 9, "direction_deg": 180, "energy_pct": 25},
        {"height_ft": 0.3, "period_s": 6, "direction_deg": 220, "energy_pct": 10}
    ],
    "combined_height_ft": 2.1,
    "dominant_period_s": 15,
    "dominant_direction_deg": 250,
    "surf_quality_rating": 7.5,
    "breaking_wave_height_ft": 3.2  # Actual breaking height (differs from open ocean)
}
```

### Bathymetry Data Requirements

Three resolution tiers are needed:

| Tier | Domain | Resolution | Source | Purpose |
|------|--------|------------|--------|---------|
| **Offshore** | WW3 boundary → 50km offshore | ~500m-1km | GEBCO/ETOPO/NCEI CRM | SWAN outer domain |
| **Mid-range** | 50km → 2km offshore | ~100-200m | NCEI Coastal Relief Model | SWAN refinement zone |
| **Nearshore** | 2km → beach | 1-50m | USACE NCMP LiDAR (existing) | Surf zone physics |

### Data Storage Structure (Planned)

```
data/
├── raw/
│   └── bathymetry/
│       ├── USACE_CA_DEM_2009_9488/     # ✅ Existing: 1m nearshore LiDAR
│       ├── ncei_coastal_relief/         # TODO: 3 arc-second (~90m) CRM
│       └── gebco_2024/                  # TODO: 15 arc-second (~450m) global
├── processed/
│   └── bathymetry/
│       ├── swan_grids/                  # Rectified SWAN-format bathymetry
│       │   ├── socal_outer.grd          # ~500m resolution outer domain
│       │   └── socal_inner.grd          # ~100m resolution inner domain
│       └── surf_zone/                   # High-res nearshore grids
│           ├── huntington.grd
│           └── malibu.grd
└── zarr/
    └── static/
        └── bathymetry/
            └── multi_resolution.zarr    # Combined multi-res bathymetry store
```

### Multi-Resolution Mesh Visualization Tool

A key development tool for validating coverage and identifying gaps:

```python
# Concept: Visualize all three resolution tiers overlaid
class BathymetryMeshVisualizer:
    """
    Visualize multi-resolution bathymetry coverage with:
    - Color-coded resolution zones (offshore/mid/nearshore)
    - Grid cell boundaries showing mesh fidelity
    - Gap detection highlighting missing or low-quality areas
    - Interactive zoom to specific coordinates
    - Export snapshots for documentation
    """

    def plot_coverage_overview(self, region: str) -> Figure:
        """Show all resolution tiers with coverage boundaries."""
        pass

    def plot_mesh_fidelity(self, bounds: tuple) -> Figure:
        """Show actual grid cells/triangles in specified region."""
        pass

    def identify_gaps(self) -> List[GapRegion]:
        """Find areas with missing or insufficient resolution data."""
        pass

    def snapshot_for_coordinates(self, lat: float, lon: float, radius_km: float) -> Figure:
        """Generate zoomed visualization around specific point."""
        pass
```

### SWAN Model Integration Notes

**SWAN** (Simulating WAves Nearshore) is a third-generation spectral wave model:
- **Website**: https://swanmodel.sourceforge.io/
- **License**: GPL (free for research and commercial use)
- **Format**: Requires bathymetry in specific grid formats (.grd, .bot)
- **Boundary Conditions**: Can ingest WW3 spectral output
- **Output**: Full 2D wave spectra that can be partitioned into swell components

**Key SWAN Capabilities We'll Use**:
1. Wave refraction over complex bathymetry
2. Spectral partitioning (separating wind sea from swell)
3. Nested grids (coarse outer → fine inner)
4. Directional spreading and frequency resolution

### Input Data Integration

The surf zone model will combine:
- **SWAN output**: Nearshore wave spectra (partitioned swells)
- **Buoy data**: Real-time validation and data assimilation
- **Wind data**: Local wind effects on wave quality
- **Tide data**: Water level affects breaking location
- **Bathymetry**: Local beach/reef shape determines wave quality

### Implementation Phases

**Phase 1: Bathymetry Pipeline** (Current Focus)
- [ ] Download NCEI Coastal Relief Model for California
- [ ] Create bathymetry rectification/interpolation tools
- [ ] Build multi-resolution mesh visualization
- [ ] Define SWAN grid domains for target spots

**Phase 2: SWAN Model Setup**
- [ ] Install and configure SWAN model
- [ ] Create WW3 → SWAN boundary condition converter
- [ ] Set up nested grid configurations
- [ ] Validate against buoy observations

**Phase 3: Surf Zone Physics**
- [ ] Implement breaking wave height calculations
- [ ] Add wave quality metrics (shape, power)
- [ ] Integrate tide and wind effects
- [ ] ML model for surf quality prediction

**Phase 4: Production Integration**
- [ ] Automated SWAN runs in worker pipeline
- [ ] Swell component API endpoints
- [ ] Frontend visualization of swell breakdown
- [ ] Accuracy tracking vs actual conditions

## Roadmap & Future Ideas

These are ideas to consider, not requirements:

- [ ] Auto-generate TypeScript types from FastAPI OpenAPI schema
- [ ] Add user accounts and favorite spots
- [ ] Push notifications for good surf conditions
- [ ] Tide data integration
- [ ] Wind forecast visualization
- [ ] Spot reviews and conditions reports
- [ ] Historical accuracy tracking (public dashboard)
- [ ] A/B testing for model improvements

## Getting Help

- **FastAPI docs**: https://fastapi.tiangolo.com/
- **Next.js docs**: https://nextjs.org/docs
- **Expo docs**: https://docs.expo.dev/
- **NOAA API**: https://www.weather.gov/documentation/services-web-api
- **NDBC buoys**: https://www.ndbc.noaa.gov/

## Changelog of Architectural Decisions

Keep this updated as you make major decisions:

**2024-XX-XX**: Initial monorepo structure created
- Chose pnpm workspaces for TypeScript
- Chose Railway for backend hosting (pending final decision)
- Decided to keep Surfline analysis private in `data/analysis/`
- ML models stored in S3, loaded at runtime

**2024-12-04**: GRIB2 parsing implementation completed
- Implemented GRIB2 parsing using cfgrib + xarray + eccodes
- Added automatic NaN handling for coastal/land grid points in Wave Watch 3 data
- Wave fetcher now falls back to nearest valid ocean point when requested location is over land
- Wind and wave data now return parsed values (height, period, direction, speed, etc.)
- All GRIB parsing dependencies use pre-built binaries (no system library requirements)

**2024-12-19**: SWAN model integration architecture defined
- Designed multi-resolution wave propagation pipeline: WW3 → SWAN → Surf Zone
- Three-tier bathymetry strategy: offshore (~500m), mid-range (~100m), nearshore (1-50m existing LiDAR)
- Swell component decomposition output format defined (height/period/direction per swell train)
- NCEI Coastal Relief Model identified as primary source for SWAN domain bathymetry
- Multi-resolution mesh visualization tool concept documented for gap detection
- Implementation phased: Bathymetry Pipeline → SWAN Setup → Surf Zone Physics → Production

(Add future decisions here as they're made)

---

## Meta: How to Use This File

This file is your memory and guide. When working on this project:

1. **Reference it frequently** - It contains the "why" behind decisions
2. **Update it** when making architectural changes
3. **Be specific** - Include file paths, not just concepts
4. **Keep it practical** - Focus on what helps you write code correctly
5. **Explain non-obvious connections** - Like "worker calls ml/inference"

When you make a significant architectural decision, add it to the Changelog section above.
