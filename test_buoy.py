#!/usr/bin/env python3
"""Quick test script for buoy data fetching"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.pipelines.buoy import NDBCBuoyFetcher


async def test_ndbc_buoy():
    """Test NDBC buoy data fetching"""
    print("Testing NDBC Buoy Fetcher")
    print("=" * 50)

    fetcher = NDBCBuoyFetcher()

    # Test 1: Find nearby buoys for Huntington Beach
    print("\n1. Finding buoys near Huntington Beach (33.6595, -118.0007)...")
    nearby = fetcher.get_nearby_buoys(33.6595, -118.0007, max_distance_km=100)
    print(f"   Found {len(nearby)} buoys:")
    for buoy in nearby[:3]:
        print(f"   - {buoy['station_id']}: {buoy['name']} ({buoy['distance_km']} km)")

    # Test 2: Fetch data from nearest buoy
    if nearby:
        station_id = nearby[0]['station_id']
        print(f"\n2. Fetching latest observation from buoy {station_id}...")
        try:
            data = await fetcher.fetch_latest_observation(station_id)
            print(f"   Station: {data.get('station_id')}")
            print(f"   Timestamp: {data.get('timestamp')}")
            print(f"   Wave Height: {data.get('wave_height_m')} m ({data.get('wave_height_ft')} ft)")
            print(f"   Dominant Period: {data.get('dominant_wave_period_s')} s")
            print(f"   Wind Speed: {data.get('wind_speed_kts')} kts")
            print(f"   Water Temp: {data.get('water_temp_c')} °C")
            print("\n   All available fields:")
            for key, value in data.items():
                if key not in ['raw_data', 'data_source', 'station_id', 'timestamp']:
                    if value is not None:
                        print(f"      {key}: {value}")
        except Exception as e:
            print(f"   Error: {e}")

    print("\n" + "=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(test_ndbc_buoy())
