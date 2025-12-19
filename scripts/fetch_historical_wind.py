#!/usr/bin/env python3
"""
Fetch historical GFS wind data for model validation.

Usage:
    python scripts/fetch_historical_wind.py

This will fetch 3 months of historical GFS wind data for the California coast
and store it in data/zarr/historical/wind/gfs_historical.zarr
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.pipelines.wind.gfs_fetcher import GFSWindFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Fetch 3 months of historical GFS wind data."""

    # Calculate date range (last 3 months)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=90)

    logger.info(f"Fetching historical GFS wind data")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Region: California coast (32-42°N, 125-117°W)")
    logger.info(f"Resolution: 3-hourly")
    logger.info("")
    logger.info("This will download approximately 720 GRIB files (~15-30 MB each).")
    logger.info("Total download size: ~10-20 GB")
    logger.info("Estimated time: 1-2 hours depending on connection speed")
    logger.info("")

    fetcher = GFSWindFetcher()

    store_path = await fetcher.fetch_historical_range(
        start_date=start_date,
        end_date=end_date,
        resolution_hours=3,  # 3-hourly as requested
        # California coast bounds (from config)
        min_lat=32.0,
        max_lat=42.0,
        min_lon=-125.0,
        max_lon=-117.0,
    )

    if store_path:
        logger.info(f"Successfully stored historical data to: {store_path}")
        logger.info("")
        logger.info("To load the data:")
        logger.info("  import xarray as xr")
        logger.info(f"  ds = xr.open_zarr('{store_path}')")
        logger.info("  print(ds)")
    else:
        logger.error("Failed to fetch historical data")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
