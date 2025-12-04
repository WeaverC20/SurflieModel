# Wave Forecast Worker

Background worker service for periodic data fetching and predictions.

## Setup

```bash
# Create virtual environment (if not already)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install shared packages
pip install -e ../../packages/python/common
pip install -e ../../ml/inference
pip install -e ../../data/pipelines
```

## Running

```bash
python scheduler.py
```

## Tasks

The worker runs these periodic tasks:

- **Fetch NOAA data**: Every 1 hour
- **Run ML predictions**: Every 30 minutes
- **Update cache**: Every 15 minutes

## Deployment

Can be deployed as:
- A long-running process on Railway/Fly.io
- GitHub Actions workflows (scheduled)
- Celery worker with Redis broker
