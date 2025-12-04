# Development Testing Scripts

Quick scripts for testing and debugging during development.

## NOAA Data Fetcher Test

Test the NOAA data fetchers (tide, wave, wind) to verify APIs are working and see data structure.

### Quick Start

```bash
# From project root
python scripts/dev/test_noaa_fetch.py
```

This will test Huntington Beach, CA by default with a complete forecast.

### Usage Examples

```bash
# Test specific location
python scripts/dev/test_noaa_fetch.py --location newport_beach

# Test custom coordinates
python scripts/dev/test_noaa_fetch.py --lat 33.6 --lon -117.9

# Test individual data sources
python scripts/dev/test_noaa_fetch.py --tide-only
python scripts/dev/test_noaa_fetch.py --wave-only
python scripts/dev/test_noaa_fetch.py --wind-only

# Test all sources separately (more detailed output)
python scripts/dev/test_noaa_fetch.py --all

# Save results to JSON
python scripts/dev/test_noaa_fetch.py --save results.json
```

### Available Test Locations

- `huntington_beach` - Huntington Beach, CA (default)
- `ocean_beach_sf` - Ocean Beach, San Francisco
- `newport_beach` - Newport Beach, CA
- `pipeline` - Pipeline, Oahu, HI
- `mavericks` - Mavericks, Half Moon Bay

### What It Tests

1. **Tide Data** (NOAA CO-OPS)
   - Finds nearest tide station
   - Fetches high/low tide predictions
   - Shows station info and distance

2. **Wave Data** (Wave Watch 3)
   - Fetches GRIB2 data for waves
   - Shows model run time and data size
   - Automatic fallback to previous model runs (handles 3-4 hour delay)
   - Note: GRIB2 parsing not yet implemented

3. **Wind Data** (GFS)
   - Fetches GRIB2 data for wind
   - Shows forecast hours available
   - Automatic fallback to previous model runs (handles 3-4 hour delay)
   - Note: GRIB2 parsing not yet implemented

### Example Output

```
======================================================================
  Testing NOAA CO-OPS Tide Fetcher
======================================================================

Searching for tide station near (33.6595, -118.0007)...
✅ Found station: Los Angeles
   Station ID: 9410660
   Distance: 12.45 km
   Location: (33.7201, -118.2731)

----------------------------------------------------------------------
  Fetching tide predictions (next 48 hours)
----------------------------------------------------------------------
✅ Successfully fetched tide data

📊 Tide predictions (12 events):
   2024-12-04 08:23: 1.83m (H)
   2024-12-04 14:45: -0.15m (L)
   2024-12-04 20:56: 1.52m (H)
   ...
```

## Requirements

Make sure you have the required dependencies:

```bash
cd data/pipelines
pip install -r requirements.txt
```

## Next Steps

After verifying the fetchers work:

1. Implement GRIB2 parsing (uncomment cfgrib in requirements.txt)
2. Set up database to store fetched data
3. Create scheduled jobs to fetch data regularly
4. Build ML models with the data
