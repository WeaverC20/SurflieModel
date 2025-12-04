# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Users                                    │
│                    (Web & Mobile)                                │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  │ HTTPS
                  ▼
         ┌────────────────┐
         │  Vercel (Web)  │
         │   Next.js App  │
         └────────┬───────┘
                  │
                  │ REST API
                  ▼
         ┌──────────────────────┐
         │   Backend API        │
         │   (FastAPI)          │◄────┐
         │   - Railway/Fly.io   │     │
         └──────┬───────────────┘     │
                │                     │
                │                     │ Load Models
         ┌──────┴───────┬─────────┐  │
         │              │         │  │
         ▼              ▼         ▼  │
    ┌────────┐    ┌─────────┐  ┌──────────┐
    │ Redis  │    │ Postgres│  │ ML       │
    │ Cache  │    │ Database│  │ Inference│
    └────────┘    └─────────┘  └─────┬────┘
                                      │
                                      │
                                 ┌────▼─────┐
                                 │ S3 / GCS │
                                 │  Models  │
                                 └──────────┘

         ┌──────────────────────┐
         │   Worker Service     │
         │   (Scheduler)        │
         │   - Railway/Fly.io   │
         └──────┬───────────────┘
                │
                │ Scheduled Jobs
                │
         ┌──────┴────────┬──────────────┐
         │               │              │
         ▼               ▼              ▼
    ┌────────┐     ┌─────────┐    ┌─────────┐
    │  NOAA  │     │  NDBC   │    │  Write  │
    │  API   │     │  Buoys  │    │  to DB  │
    └────────┘     └─────────┘    └─────────┘
```

## Data Flow

### Forecast Generation Flow

1. **Data Ingestion** (Worker, every 1-6 hours):
   ```
   NOAA API → Worker → Parse → Database
   NDBC API → Worker → Parse → Database
   ```

2. **Prediction Generation** (Worker, every 30 min):
   ```
   Worker → Load latest data from DB
         → Call ml/inference/predictor.py
         → Generate predictions
         → Cache in Redis (15-30 min TTL)
         → Store in Database
   ```

3. **User Request** (Real-time):
   ```
   User → Frontend → API
                  → Check Redis cache
                  → If cached: return
                  → If not: query DB → return
   ```

## Component Responsibilities

### Frontend (apps/)
- **Web** (Next.js): Desktop/tablet experience
- **Mobile** (Expo): iOS/Android apps
- **Responsibilities**:
  - User interface and experience
  - API client integration
  - Data visualization (charts, maps)
  - Responsive design

### Backend API (backend/api/)
- **Responsibilities**:
  - RESTful API endpoints
  - Authentication & authorization (future)
  - Request validation
  - Business logic
  - Database queries
  - Cache management
  - Call ML inference for on-demand predictions

### Worker Service (backend/worker/)
- **Responsibilities**:
  - Scheduled data fetching from NOAA/NDBC
  - Periodic prediction generation
  - Cache warming
  - Data cleanup/maintenance
  - No user-facing endpoints

### ML Inference (ml/inference/)
- **Responsibilities**:
  - Load models from S3
  - Preprocess input data
  - Run predictions
  - Post-process outputs
  - Simple API: `predict(spot_id, timestamp, data)`
- **Used by**: Backend API, Worker Service
- **NOT used by**: Frontend (models too large)

### ML Training (ml/training/)
- **Responsibilities**:
  - Model architecture development
  - Training pipelines
  - Hyperparameter tuning
  - Model evaluation
  - Upload best models to S3
- **Used by**: Developers, Data Scientists
- **NOT deployed** to production

### Data Pipelines (data/pipelines/)
- **Responsibilities**:
  - Fetch data from external sources
  - Parse and normalize data
  - Validate data quality
  - Store in database
- **Used by**: Worker Service, Training scripts

### Shared Packages (packages/)
- **TypeScript** (`api-client`):
  - API client
  - Type definitions
  - Shared utilities
  - Used by: Web, Mobile

- **Python** (`common`):
  - Constants and enums
  - Utility functions (unit conversion, distance calc)
  - Shared types
  - Used by: Backend, Worker, ML, Data Pipelines

## Database Schema (Conceptual)

```sql
-- Surf spots
spots (
  id, name, latitude, longitude,
  timezone, spot_type, description
)

-- Forecast data
forecasts (
  id, spot_id, timestamp,
  wave_height, wave_period, swell_direction,
  wind_speed, wind_direction,
  rating, confidence,
  created_at
)

-- Raw NOAA data
noaa_data (
  id, latitude, longitude, timestamp,
  model_data (jsonb), fetched_at
)

-- Buoy observations
buoy_observations (
  id, station_id, timestamp,
  wave_height, dominant_period, mean_direction,
  wind_speed, wind_direction, water_temp
)

-- Users (future)
users (
  id, email, password_hash, created_at
)

-- Favorite spots (future)
favorites (
  user_id, spot_id, created_at
)
```

## API Endpoints (Planned)

```
GET  /api/v1/health                          # Health check
GET  /api/v1/spots                           # List all spots
GET  /api/v1/spots/{spot_id}                 # Get spot details
GET  /api/v1/spots/search?lat=&lon=&radius=  # Search nearby spots
GET  /api/v1/forecast/{spot_id}              # Get forecast
GET  /api/v1/forecast/{spot_id}/current      # Get current conditions
GET  /api/v1/buoy/{station_id}               # Get buoy data

# Future: User endpoints
POST /api/v1/auth/register
POST /api/v1/auth/login
GET  /api/v1/users/me
POST /api/v1/favorites/{spot_id}
```

## Deployment Architecture

### Production Environment

```
Web (Vercel):
├── Auto-deploy from main branch
├── Environment: NEXT_PUBLIC_API_URL
└── CDN: Global edge network

Backend (Railway/Fly.io):
├── API Service (1+ instances)
│   ├── FastAPI app
│   ├── ml/inference module
│   └── Load models from S3 on startup
├── Worker Service (1 instance)
│   ├── Scheduler
│   ├── Periodic tasks
│   └── Cron jobs
├── PostgreSQL Database
└── Redis Cache

Storage (AWS/GCP):
├── S3/GCS: ML model artifacts
└── (Future) User uploads, images

Mobile (EAS):
├── Build on EAS
├── Submit to App Store / Google Play
└── OTA updates via Expo
```

### Development Environment

```
Local Machine:
├── Next.js dev server (port 3000)
├── Expo dev server (port 8081)
├── FastAPI dev server (port 8000)
├── Worker scheduler (background)
├── PostgreSQL (Docker or local)
└── Redis (Docker or local)
```

## Scaling Considerations

### Current (MVP)
- Single API instance
- Single worker instance
- Small database
- Redis cache

### Future (Growth)
- **Horizontal API scaling**: Multiple API instances behind load balancer
- **Worker scaling**: Multiple workers with distributed task queue (Celery)
- **Database**: Read replicas, connection pooling
- **Cache**: Redis cluster
- **CDN**: Serve static forecast data from CDN
- **ML**: Dedicated inference service, model versioning

## Security Architecture

### Authentication (Future)
- JWT tokens for API authentication
- Refresh tokens
- OAuth2 for social login

### Authorization (Future)
- Role-based access control (RBAC)
- Rate limiting per user/IP
- API key for third-party access

### Data Protection
- HTTPS everywhere
- Environment variables for secrets
- No secrets in git
- Database encryption at rest
- Input validation and sanitization

## Monitoring & Observability (Future)

- **Logging**: Structured logs (JSON)
- **Metrics**: Response times, error rates, cache hit rates
- **Tracing**: Distributed tracing for requests
- **Alerts**: Error rate spikes, API downtime, worker failures
- **Dashboard**: Grafana or similar for visualization

## Disaster Recovery

- **Database backups**: Daily automated backups
- **Model versioning**: Keep previous model versions in S3
- **Rollback strategy**: Deploy previous version if issues
- **Data retention**: Keep raw NOAA/buoy data for reprocessing
