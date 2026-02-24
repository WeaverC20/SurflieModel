# Data

Data pipelines, exploration notebooks, and private analysis.

## Structure

- **pipelines/** - ETL scripts for data ingestion (NOAA, buoy data)
- **notebooks/** - Jupyter notebooks for data exploration and validation
- **analysis/** - Private analysis

## Data Sources

### NOAA
- Wave Watch III models
- GFS weather forecasts
- Coastal forecasts

### NDBC Buoys
- Real-time wave observations
- Historical buoy data

## Usage

Pipelines are designed to be:
1. Run by the worker service (scheduled)
2. Run manually for historical data collection
3. Imported by training scripts for dataset creation

## Important

**Private analysis stays private:**
- Never imported by production code (backend, ml/inference)
- Safe to experiment and explore here
