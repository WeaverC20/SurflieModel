#!/usr/bin/env python3
"""
Test NOAA data fetchers

Quick script to verify NOAA APIs are working and see the data structure.

Usage:
    python scripts/dev/test_noaa_fetch.py
    python scripts/dev/test_noaa_fetch.py --lat 33.6 --lon -117.9  # Newport Beach, CA
    python scripts/dev/test_noaa_fetch.py --all  # Test all fetchers
    python scripts/dev/test_noaa_fetch.py --tide-only
    python scripts/dev/test_noaa_fetch.py --wave-only
    python scripts/dev/test_noaa_fetch.py --wind-only
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.pipelines.noaa import (
    NOAAFetcher,
    NOAATideFetcher,
    NOAAWaveWatch3Fetcher,
    NOAAWindFetcher,
)


# Predefined test locations
TEST_LOCATIONS = {
    "huntington_beach": {
        "name": "Huntington Beach, CA",
        "lat": 33.6595,
        "lon": -118.0007,
        "tide_station": "9410660",  # Los Angeles
    },
    "ocean_beach_sf": {
        "name": "Ocean Beach, San Francisco",
        "lat": 37.7594,
        "lon": -122.5107,
        "tide_station": "9414290",  # San Francisco
    },
    "newport_beach": {
        "name": "Newport Beach, CA",
        "lat": 33.6,
        "lon": -117.9,
        "tide_station": "9410580",  # Newport Beach
    },
    "pipeline": {
        "name": "Pipeline, Oahu, HI",
        "lat": 21.6644,
        "lon": -158.0528,
        "tide_station": "1612340",  # Honolulu
    },
    "mavericks": {
        "name": "Mavericks, Half Moon Bay",
        "lat": 37.4947,
        "lon": -122.4969,
        "tide_station": "9414290",  # San Francisco (nearest)
    },
}


def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_section(title):
    """Print a section divider"""
    print(f"\n{'-'*70}")
    print(f"  {title}")
    print(f"{'-'*70}")


def format_json(data, max_items=5):
    """Format JSON data for display, limiting list lengths"""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if isinstance(value, list) and len(value) > max_items:
                result[key] = value[:max_items] + [f"... ({len(value) - max_items} more items)"]
            else:
                result[key] = value
        return json.dumps(result, indent=2, default=str)
    return json.dumps(data, indent=2, default=str)


async def test_tide_fetcher(lat, lon, station_id=None):
    """Test NOAA CO-OPS tide fetcher"""
    print_header("Testing NOAA CO-OPS Tide Fetcher")

    fetcher = NOAATideFetcher()

    # Find nearest station
    print(f"\nSearching for tide station near ({lat}, {lon})...")
    if not station_id:
        station = await fetcher.find_nearest_station(lat, lon)
        if not station:
            print("❌ No tide station found within 100km")
            return None

        print(f"✅ Found station: {station['name']}")
        print(f"   Station ID: {station['id']}")
        print(f"   Distance: {station['distance_km']:.2f} km")
        print(f"   Location: ({station['lat']}, {station['lon']})")
        station_id = station['id']
    else:
        print(f"✅ Using provided station: {station_id}")

    # Fetch tide predictions
    print_section("Fetching tide predictions (next 48 hours)")
    begin_date = datetime.utcnow()
    end_date = begin_date + timedelta(hours=48)

    try:
        tide_data = await fetcher.fetch_tide_predictions(
            station_id=station_id,
            begin_date=begin_date,
            end_date=end_date,
            interval='hilo'  # High/low only
        )

        print("✅ Successfully fetched tide data")
        print(f"\nData structure:")
        print(format_json(tide_data))

        # Display high/low tides
        if 'predictions' in tide_data:
            predictions = tide_data['predictions']
            print(f"\n📊 Tide predictions ({len(predictions)} events):")
            for pred in predictions[:10]:  # Show first 10
                time = pred.get('t', 'N/A')
                height = pred.get('v', 'N/A')
                tide_type = pred.get('type', 'N/A')
                print(f"   {time}: {height}m ({tide_type})")

        return tide_data

    except Exception as e:
        print(f"❌ Error fetching tide data: {e}")
        return None


async def test_wave_fetcher(lat, lon):
    """Test NOAA Wave Watch 3 fetcher"""
    print_header("Testing NOAA Wave Watch 3 Fetcher")

    fetcher = NOAAWaveWatch3Fetcher()

    print(f"\nFetching wave data for ({lat}, {lon})...")
    print("Forecast hour: 0 (current model run)")

    try:
        wave_data = await fetcher.fetch_wave_spectrum(lat, lon, forecast_hour=0)

        print("✅ Successfully fetched wave data")
        print(f"\nData structure:")
        print(format_json(wave_data))

        if wave_data.get('status') == 'success':
            print(f"\n📊 Wave data summary:")
            print(f"   Model run: {wave_data.get('model_time', 'N/A')}")
            print(f"   Valid time: {wave_data.get('valid_time', 'N/A')}")
            print(f"   Data size: {wave_data.get('data_size_bytes', 0)} bytes")
            print(f"   Note: {wave_data.get('note', '')}")

        return wave_data

    except Exception as e:
        print(f"❌ Error fetching wave data: {e}")
        return None


async def test_wind_fetcher(lat, lon):
    """Test NOAA GFS wind fetcher"""
    print_header("Testing NOAA GFS Wind Fetcher")

    fetcher = NOAAWindFetcher()

    print(f"\nFetching wind data for ({lat}, {lon})...")
    print("Forecast: Next 24 hours")

    try:
        wind_data = await fetcher.fetch_wind_timeseries(lat, lon, hours=24)

        print("✅ Successfully fetched wind data")
        print(f"\nData structure:")
        print(format_json(wind_data))

        if 'forecasts' in wind_data:
            forecasts = wind_data['forecasts']
            print(f"\n📊 Wind forecast summary:")
            print(f"   Number of forecast hours: {len(forecasts)}")
            print(f"   Forecast hours: {wind_data.get('forecast_hours', [])}")

            if forecasts and forecasts[0].get('status') == 'success':
                print(f"   Model run: {forecasts[0].get('model_time', 'N/A')}")
                print(f"   Data size (per hour): {forecasts[0].get('data_size_bytes', 0)} bytes")

        return wind_data

    except Exception as e:
        print(f"❌ Error fetching wind data: {e}")
        return None


async def test_complete_forecast(lat, lon, station_id=None):
    """Test complete forecast fetcher (all data sources)"""
    print_header("Testing Complete NOAA Forecast Fetcher")

    fetcher = NOAAFetcher()

    print(f"\nFetching complete forecast for ({lat}, {lon})...")
    print("This combines tide, wave, and wind data")

    try:
        forecast = await fetcher.fetch_complete_forecast(
            latitude=lat,
            longitude=lon,
            station_id=station_id,
            forecast_hours=48
        )

        print("✅ Successfully fetched complete forecast")
        print(f"\nData structure:")
        print(format_json(forecast, max_items=3))

        # Summary
        print(f"\n📊 Complete forecast summary:")

        if 'tide_station' in forecast:
            station = forecast['tide_station']
            print(f"   Tide station: {station.get('name', 'N/A')} ({station.get('distance_km', 0):.1f} km)")

        if 'tide' in forecast:
            tide_preds = forecast['tide'].get('predictions', [])
            print(f"   Tide predictions: {len(tide_preds)} events")
        elif 'tide_error' in forecast:
            print(f"   Tide error: {forecast['tide_error']}")

        if 'waves' in forecast:
            if forecast['waves'].get('status') == 'success':
                print(f"   Wave data: ✅ Retrieved")
            else:
                print(f"   Wave data: ❌ {forecast['waves'].get('error', 'Unknown error')}")
        elif 'wave_error' in forecast:
            print(f"   Wave error: {forecast['wave_error']}")

        if 'wind' in forecast:
            wind_forecasts = forecast['wind'].get('forecasts', [])
            print(f"   Wind forecasts: {len(wind_forecasts)} hours")
        elif 'wind_error' in forecast:
            print(f"   Wind error: {forecast['wind_error']}")

        return forecast

    except Exception as e:
        print(f"❌ Error fetching complete forecast: {e}")
        return None


async def main():
    parser = argparse.ArgumentParser(description="Test NOAA data fetchers")
    parser.add_argument('--lat', type=float, help='Latitude in decimal degrees')
    parser.add_argument('--lon', type=float, help='Longitude in decimal degrees')
    parser.add_argument('--station', type=str, help='NOAA tide station ID')
    parser.add_argument('--location', type=str, choices=list(TEST_LOCATIONS.keys()),
                       help='Use predefined test location')
    parser.add_argument('--tide-only', action='store_true', help='Test tide fetcher only')
    parser.add_argument('--wave-only', action='store_true', help='Test wave fetcher only')
    parser.add_argument('--wind-only', action='store_true', help='Test wind fetcher only')
    parser.add_argument('--all', action='store_true', help='Test all fetchers separately')
    parser.add_argument('--save', type=str, help='Save results to JSON file')

    args = parser.parse_args()

    # Determine location
    if args.location:
        location = TEST_LOCATIONS[args.location]
        lat = location['lat']
        lon = location['lon']
        station_id = location.get('tide_station')
        print(f"\n📍 Testing location: {location['name']}")
    elif args.lat and args.lon:
        lat = args.lat
        lon = args.lon
        station_id = args.station
        print(f"\n📍 Testing custom location: ({lat}, {lon})")
    else:
        # Default to Huntington Beach, CA
        location = TEST_LOCATIONS['huntington_beach']
        lat = location['lat']
        lon = location['lon']
        station_id = location.get('tide_station')
        print(f"\n📍 Using default location: {location['name']}")

    print(f"   Coordinates: ({lat}, {lon})")
    if station_id:
        print(f"   Tide station: {station_id}")

    # Run tests
    results = {}

    if args.tide_only:
        results['tide'] = await test_tide_fetcher(lat, lon, station_id)
    elif args.wave_only:
        results['wave'] = await test_wave_fetcher(lat, lon)
    elif args.wind_only:
        results['wind'] = await test_wind_fetcher(lat, lon)
    elif args.all:
        results['tide'] = await test_tide_fetcher(lat, lon, station_id)
        results['wave'] = await test_wave_fetcher(lat, lon)
        results['wind'] = await test_wind_fetcher(lat, lon)
    else:
        # Default: complete forecast
        results['complete'] = await test_complete_forecast(lat, lon, station_id)

    # Save results if requested
    if args.save:
        save_path = Path(args.save)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {save_path}")

    print_header("Test Complete")
    print("\nNext steps:")
    print("  1. Review the data structures above")
    print("  2. Implement GRIB2 parsing for wave/wind data (see requirements.txt)")
    print("  3. Add error handling for your use case")
    print("  4. Consider caching to reduce API calls")
    print("\nFor more locations, use: --location [huntington_beach|ocean_beach_sf|newport_beach|pipeline|mavericks]")


if __name__ == '__main__':
    asyncio.run(main())
