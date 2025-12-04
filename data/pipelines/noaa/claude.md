# NOAA Data Pipeline - Implementation Guide

## Overview

This module fetches wave forecast data from NOAA (National Oceanic and Atmospheric Administration) services. All NOAA data used here is free, open, and requires no API key or license.

## Data Sources

### 1. NOAA CO-OPS (Tides)

**Purpose**: Tide predictions and harmonic constituents

**API**: https://api.tidesandcurrents.noaa.gov/api/prod/

**Coverage**: US coastal waters only

**Data Products**:
- Tide predictions (high/low, hourly, or 6-minute intervals)
- Harmonic constituents for accurate long-term predictions
- Water level observations
- Currents (where available)

**Key Features**:
- No API key required
- Highly accurate for US waters
- Station-based (must find nearest station to surf spot)
- Free and unrestricted use

**Limitations**:
- US only (for global coverage, would need TPXO or FES)
- Station coverage varies by region
- May not have stations for remote locations

**Example Station IDs**:
- 9414290 - San Francisco, CA
- 9410170 - San Diego, CA
- 8518750 - The Battery, NY

### 2. NOAA Wave Watch 3 (Swell)

**Purpose**: Swell height, period, and direction forecasts

**Data Access**: https://nomads.ncep.noaa.gov/

**Coverage**: Global, with focus on US waters

**Key Variables**:
- `HTSGW` - Significant height of combined wind waves and swell
- `PERPW` - Primary wave mean period
- `DIRPW` - Primary wave direction
- `SWELL` - Significant height of swell waves
- `SWPER` - Swell period
- `SWDIR` - Swell direction
- `WVHGT` - Significant height of wind waves

**Model Details**:
- Resolution: 0.25 degrees (~25km)
- Forecast range: 16 days (384 hours)
- Update frequency: Every 6 hours (00, 06, 12, 18 UTC)
- Data format: GRIB2

**Key Features**:
- Free and open
- Global coverage
- Good accuracy for offshore swell
- Includes wave partitioning (separate wind waves and swell)

**Future Enhancement**: GEFS-WAVE
- Ensemble forecasts with uncertainty estimates
- 16-day probabilistic forecasts
- Better for long-range planning

### 3. NOAA GFS (Wind)

**Purpose**: Wind speed and direction forecasts

**Data Access**: https://nomads.ncep.noaa.gov/

**Coverage**: Global

**Key Variables**:
- `UGRD` - U-component of wind (east-west)
- `VGRD` - V-component of wind (north-south)
- `GUST` - Wind gust speed

**Model Details**:
- Resolution: 0.25 degrees (~25km)
- Forecast range: 16 days (384 hours)
- Update frequency: Every 6 hours (00, 06, 12, 18 UTC)
- Output frequency: 3-hourly for first 240 hours, 12-hourly thereafter
- Standard level: 10m above ground
- Data format: GRIB2

**Key Features**:
- Free and open
- Global coverage
- Deterministic forecasts
- Good for 7-10 day forecasts

**Future Enhancement**: GEFS (ensemble)
- Probabilistic wind forecasts
- Better uncertainty quantification
- Same 16-day range

## Implementation Details

### Fetcher Classes

**NOAATideFetcher**:
- `fetch_tide_predictions()` - Get tide forecast for a station
- `fetch_harmonic_constituents()` - Get constituents for long-term predictions
- `find_nearest_station()` - Find closest tide station to coordinates

**NOAAWaveWatch3Fetcher**:
- `fetch_wave_data()` - Get wave forecast for specific hour
- `fetch_wave_spectrum()` - Get full spectrum with multiple swell components

**NOAAWindFetcher**:
- `fetch_wind_data()` - Get wind forecast for specific hour
- `fetch_wind_timeseries()` - Get time series of wind forecasts

**NOAAFetcher** (unified interface):
- `fetch_complete_forecast()` - Get tide, wave, and wind in one call
- Automatically finds nearest tide station
- Handles errors gracefully

### Data Format Notes

**GRIB2 Files**:
- Wave Watch 3 and GFS data come as GRIB2 binary files
- Current implementation fetches raw GRIB2 data
- For production, parse with:
  - `pygrib` (requires GRIB library installation)
  - `cfgrib` (pure Python, uses eccodes)
  - `xarray` + `cfgrib` (recommended for ease of use)

**Tide Data**:
- Returns JSON directly
- Easy to parse and use immediately
- Timestamps in GMT/UTC

### Example Usage

```python
from data.pipelines.noaa import NOAAFetcher
from datetime import datetime, timedelta

# Initialize fetcher
fetcher = NOAAFetcher()

# Get complete forecast for a location
forecast = await fetcher.fetch_complete_forecast(
    latitude=37.8,
    longitude=-122.4,  # San Francisco
    forecast_hours=168  # 7 days
)

# Access individual components
tide_data = forecast.get("tide")
wave_data = forecast.get("waves")
wind_data = forecast.get("wind")

# Or use specialized fetchers
tide_fetcher = fetcher.tide_fetcher
station = await tide_fetcher.find_nearest_station(37.8, -122.4)
print(f"Nearest station: {station['name']} ({station['distance_km']:.1f} km)")

tides = await tide_fetcher.fetch_tide_predictions(
    station_id=station['id'],
    begin_date=datetime.utcnow(),
    end_date=datetime.utcnow() + timedelta(days=7),
    interval='hilo'  # High/low tides only
)
```

## Development Roadmap

### Phase 1: Current Implementation ✅
- [x] NOAA CO-OPS tide fetcher
- [x] Wave Watch 3 GRIB2 fetcher (metadata only)
- [x] GFS wind GRIB2 fetcher (metadata only)
- [x] Unified interface with error handling

### Phase 2: GRIB Parsing
- [ ] Implement GRIB2 parsing with cfgrib/xarray
- [ ] Extract actual wave/wind values from binary data
- [ ] Add spatial interpolation for point locations
- [ ] Cache parsed data to reduce API calls

### Phase 3: Enhanced Features
- [ ] Add GEFS-WAVE for ensemble wave forecasts
- [ ] Add GEFS for ensemble wind forecasts
- [ ] Implement wave partitioning (separate swell components)
- [ ] Add metadata for data quality/confidence

### Phase 4: Regional High-Res Models
- [ ] NAM (U.S.) - 3km wind
- [ ] HRRR (U.S.) - 3km hourly wind
- [ ] Regional model selection based on location

### Phase 5: Global Expansion
- [ ] TPXO/FES for global tide data (requires license)
- [ ] AROME (Europe)
- [ ] HARMONIE-AROME (Nordic)
- [ ] ACCESS (Australia)

## Important Notes

### API Rate Limits
- NOAA APIs are generally rate-limited but generous
- CO-OPS: ~50 requests per 10 seconds
- NOMADS: No official limit, but be respectful
- Implement caching to avoid unnecessary requests

### Data Latency
- GFS/Wave Watch 3: ~3-4 hours after model run time
- Model runs: 00, 06, 12, 18 UTC
- Example: 00Z model available around 03-04Z

### Coordinate Systems
- NOAA GRIB files use 0-360 longitude (not -180 to 180)
- Fetchers automatically convert as needed
- Always use decimal degrees for coordinates

### GRIB2 Parsing Dependencies
To parse GRIB2 files in production, add to requirements:
```
cfgrib>=0.9.10
xarray>=2023.1.0
```

Or for pygrib (faster but requires system libraries):
```
pygrib>=2.1.4
```

### Best Practices

1. **Cache aggressively**: Model runs only update every 6 hours
2. **Batch requests**: Fetch multiple forecast hours in one go
3. **Handle errors**: Models can have missing data or outages
4. **Use async**: All fetchers are async for concurrent requests
5. **Log everything**: Track API calls for debugging

### Validation

Always validate NOAA data against buoy observations:
- NDBC buoys for wave/wind validation
- CDIP buoys for directional wave data
- Compare forecasts at different lead times
- See `data/analysis/` for validation notebooks

## References

- **NOAA CO-OPS API**: https://api.tidesandcurrents.noaa.gov/api/prod/
- **Wave Watch 3**: https://polar.ncep.noaa.gov/waves/
- **NOMADS Data Access**: https://nomads.ncep.noaa.gov/
- **GFS Documentation**: https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast
- **GRIB2 Format**: https://www.nco.ncep.noaa.gov/pmb/docs/grib2/

## Support

For issues with NOAA data:
1. Check NOMADS status page for outages
2. Verify coordinates are in valid range
3. Check model run times (data availability)
4. Review logs for API errors

For implementation questions, see the main project [docs/claude.md](../../../docs/claude.md).
