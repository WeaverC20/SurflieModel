# Wave Forecast App

A comprehensive wave forecasting platform with custom ML models, NOAA data integration, and multi-platform support.

## Project Structure

- **apps/** - Frontend applications (web & mobile)
- **backend/** - FastAPI backend and worker services
- **ml/** - Machine learning training and inference
- **data/** - Data pipelines, notebooks, and analysis
- **packages/** - Shared code (TypeScript and Python)
- **docs/** - Project documentation

## Getting Started

### Prerequisites

- Node.js 18+ and pnpm
- Python 3.11+
- PostgreSQL (for production) or SQLite (for development)

### Installation

```bash
# Install frontend dependencies
pnpm install

# Set up Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
pip install -e backend/api
pip install -e backend/worker
pip install -e packages/python/common
pip install -e ml/inference
```

### Development

```bash
# Run web app
pnpm dev:web

# Run mobile app
pnpm dev:mobile

# Run backend API
cd backend/api
uvicorn app.main:app --reload

# Run worker
cd backend/worker
python scheduler.py
```

## Documentation

See [docs/claude.md](docs/claude.md) for detailed project context and conventions.

## Deployment

- **Web**: Vercel (auto-deploy from main branch)
- **Backend**: Railway or Fly.io
- **Mobile**: EAS (Expo Application Services)

## License

Private project - All rights reserved
