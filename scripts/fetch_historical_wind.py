#!/usr/bin/env python3
"""
Fetch historical GFS wind data for model validation.

Usage:
    # Default: fetch last 3 days (for validation)
    python scripts/fetch_historical_wind.py

    # Fetch specific date range
    python scripts/fetch_historical_wind.py --start 2025-12-01 --end 2025-12-15

    # Fetch last N days
    python scripts/fetch_historical_wind.py --days 7

    # Append to existing dataset
    python scripts/fetch_historical_wind.py --start 2025-11-01 --end 2025-11-15 --append

Data is stored in data/zarr/historical/wind/gfs_historical.zarr
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fetch historical GFS wind data for model validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch last 3 days (default, good for testing)
  python scripts/fetch_historical_wind.py

  # Fetch specific date range
  python scripts/fetch_historical_wind.py --start 2025-12-01 --end 2025-12-15

  # Fetch last 30 days
  python scripts/fetch_historical_wind.py --days 30

  # Append new dates to existing dataset
  python scripts/fetch_historical_wind.py --start 2025-11-01 --end 2025-11-15 --append
        """
    )
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD format)'
    )
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD format). Defaults to today.'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=3,
        help='Number of days to fetch (from today backwards). Default: 3. Ignored if --start is provided.'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help='Append to existing dataset instead of overwriting'
    )
    parser.add_argument(
        '--resolution',
        type=int,
        choices=[3, 6],
        default=3,
        help='Time resolution in hours. Default: 3'
    )
    return parser.parse_args()


async def main():
    """Fetch historical GFS wind data."""
    args = parse_args()

    # Calculate date range
    end_date = datetime.now(timezone.utc).replace(tzinfo=None)
    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')

    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    else:
        start_date = end_date - timedelta(days=args.days)

    # Estimate download info
    total_days = (end_date - start_date).days + 1
    files_per_day = 8 if args.resolution == 3 else 4  # 4 cycles, plus forecast files for 3-hourly
    total_files = total_days * files_per_day
    estimated_size_gb = total_files * 0.02  # ~20MB per file after region extraction

    logger.info(f"Fetching historical GFS wind data")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()} ({total_days} days)")
    logger.info(f"Region: California coast (32-42°N, 125-117°W)")
    logger.info(f"Resolution: {args.resolution}-hourly")
    logger.info(f"Mode: {'Append' if args.append else 'Overwrite'}")
    logger.info("")
    logger.info(f"Estimated downloads: ~{total_files} GRIB files")
    logger.info(f"Estimated total size: ~{estimated_size_gb:.1f} GB")
    logger.info("")

    fetcher = GFSWindFetcher()

    store_path = await fetcher.fetch_historical_range(
        start_date=start_date,
        end_date=end_date,
        resolution_hours=args.resolution,
        min_lat=32.0,
        max_lat=42.0,
        min_lon=-125.0,
        max_lon=-117.0,
        append=args.append,
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
