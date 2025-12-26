#!/usr/bin/env python3
"""
Fetch WW3 Data

Fetches the latest WaveWatch III forecast data and stores it locally.
Run this script to update the local WW3 data cache before running SWAN.

Default: 16-day forecast (384 hours)
- 3-hourly for hours 0-48
- 24-hourly for hours 72-384

Usage:
    python runs/fetch_ww3.py
    python runs/fetch_ww3.py --region california
"""

import argparse
import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.pipelines.wave.wavewatch_fetcher import WaveWatchFetcher
from data.storage import WaveDataStore, get_storage_config
from data.storage.config import REGIONS


def get_forecast_hours() -> List[int]:
    """
    Get forecast hours matching frontend format.

    - 3-hourly for first 48 hours: 0, 3, 6, ..., 48
    - 24-hourly from 72 to 384 hours: 72, 96, 120, ..., 384

    Total: 31 forecast hours covering 16 days
    """
    hours = []

    # 3-hourly for first 48 hours
    for h in range(0, 49, 3):
        hours.append(h)

    # 24-hourly from 72 hours onwards (16 days = 384 hours)
    for h in range(72, 385, 24):
        hours.append(h)

    return hours


async def fetch_ww3(region: str = "california"):
    """
    Fetch WW3 data and store locally.

    Args:
        region: Region name from config
    """
    forecast_hours = get_forecast_hours()

    print("=" * 60)
    print("Fetching WaveWatch III Forecast Data")
    print("=" * 60)
    print(f"  Region: {region}")
    print(f"  Forecast hours: {len(forecast_hours)} time steps")
    print(f"    0-48h: 3-hourly ({len([h for h in forecast_hours if h <= 48])} steps)")
    print(f"    72-384h: 24-hourly ({len([h for h in forecast_hours if h >= 72])} steps)")
    print(f"  Time: {datetime.now(timezone.utc).isoformat()}")
    print()

    # Get region bounds
    if region not in REGIONS:
        print(f"Error: Unknown region '{region}'")
        print(f"Available: {list(REGIONS.keys())}")
        return False

    region_config = REGIONS[region]

    # Initialize fetcher and store
    fetcher = WaveWatchFetcher()
    store = WaveDataStore()

    # Ensure storage directories exist
    config = get_storage_config()
    config.ensure_directories()

    print(f"Fetching {len(forecast_hours)} forecast hours...")
    print()

    # Fetch data for each forecast hour
    data_list = []
    cycle_time = None

    for hour in forecast_hours:
        try:
            data = await fetcher.fetch_wave_grid(
                min_lat=region_config.min_lat,
                max_lat=region_config.max_lat,
                min_lon=region_config.min_lon,
                max_lon=region_config.max_lon,
                forecast_hour=hour,
            )

            if data:
                data_list.append(data)
                if cycle_time is None:
                    cycle_time = datetime.fromisoformat(data["cycle_time"])

                # Calculate mean Hs (handling 2D array with NaNs)
                import numpy as np
                hs_array = np.array(data["significant_wave_height"])
                hs_mean = np.nanmean(hs_array)

                # Format hour display like frontend
                if hour == 0:
                    hour_str = "Now"
                elif hour < 24:
                    hour_str = f"+{hour}h"
                else:
                    days = hour // 24
                    rem = hour % 24
                    hour_str = f"+{days}d" if rem == 0 else f"+{days}d {rem}h"

                print(f"  {hour_str:>8} (f{hour:03d}): Hs = {hs_mean:.2f}m")

        except Exception as e:
            print(f"  f{hour:03d}: FAILED - {e}")

    if not data_list:
        print("\nError: No data fetched successfully")
        return False

    # Store the data
    print(f"\nStoring {len(data_list)} forecast hours to local cache...")
    store.store_forecast_range(data_list, cycle_time)

    # Get storage path for user info
    store_path = config.get_wave_store_path()

    print()
    print("=" * 60)
    print("WW3 Fetch Complete")
    print("=" * 60)
    print(f"  Cycle time: {cycle_time.isoformat()}Z")
    print(f"  Hours fetched: {len(data_list)} of {len(forecast_hours)}")
    print(f"  Stored at: {store_path}")
    print()
    print("Next step: Run SWAN computation")
    print("  python runs/run_swan.py")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Fetch WW3 forecast data and store locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fetches 16-day WW3 forecast matching frontend display format:
  - 3-hourly for hours 0-48 (17 time steps)
  - 24-hourly for hours 72-384 (14 time steps)
  - Total: 31 forecast hours

Examples:
    python runs/fetch_ww3.py
    python runs/fetch_ww3.py --region southern_california
""",
    )

    parser.add_argument(
        "--region", "-r",
        type=str,
        default="california",
        help="Region to fetch (default: california)",
    )

    args = parser.parse_args()

    success = asyncio.run(fetch_ww3(region=args.region))

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
