#!/usr/bin/env python3
"""
WW3 vs NDBC Buoy Comparison Script

Compares WaveWatch III model data with NDBC buoy observations at the same
timestamp and location. This helps validate model accuracy and understand
biases in the WW3 forecasts.

Run from project root:
    python scripts/dev/compare_ww3_ndbc.py

The script:
1. Fetches WW3 data for the latest available cycle (forecast hour 0 = analysis)
2. Fetches NDBC buoy observations for the matching timestamp
3. Interpolates WW3 grid to buoy locations
4. Compares wave height, period, and direction
"""

import asyncio
import gzip
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx
import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.pipelines.wave.wavewatch_fetcher import WaveWatchFetcher


# =============================================================================
# Configuration
# =============================================================================

# NDBC buoys with WAVE SENSORS (not all NDBC buoys measure waves!)
# These are confirmed to report WVHT, DPD, MWD in their standard met data
COMPARISON_BUOYS = {
    # Southern California
    "46222": {"name": "San Pedro", "lat": 33.618, "lon": -118.317},
    "46219": {"name": "San Nicolas Island", "lat": 33.221, "lon": -119.882},
    "46277": {"name": "Santa Rosa Island West", "lat": 34.045, "lon": -120.435},
    "46054": {"name": "Santa Barbara W", "lat": 34.274, "lon": -120.459},
    "46218": {"name": "Harvest", "lat": 34.454, "lon": -120.782},
    # Central California
    "46215": {"name": "Diablo Canyon", "lat": 35.204, "lon": -120.859},
    "46011": {"name": "Santa Maria", "lat": 34.868, "lon": -120.857},
    "46028": {"name": "Cape San Martin", "lat": 35.741, "lon": -121.884},
    "46239": {"name": "Point Sur", "lat": 36.340, "lon": -122.101},
    "46236": {"name": "Monterey Bay W", "lat": 36.761, "lon": -122.028},
    "46214": {"name": "Point Reyes", "lat": 37.946, "lon": -123.469},
    "46237": {"name": "San Francisco Bar", "lat": 37.786, "lon": -122.634},
    # Northern California
    "46013": {"name": "Bodega Bay", "lat": 38.242, "lon": -123.301},
    "46014": {"name": "Point Arena", "lat": 39.196, "lon": -123.969},
    "46213": {"name": "Cape Mendocino", "lat": 40.294, "lon": -124.738},
    "46022": {"name": "Eel River", "lat": 40.749, "lon": -124.577},
    "46027": {"name": "St Georges", "lat": 41.850, "lon": -124.381},
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class NDBCObservation:
    """Single NDBC buoy observation."""
    timestamp: datetime
    wave_height_m: Optional[float] = None
    dominant_period_s: Optional[float] = None
    average_period_s: Optional[float] = None
    mean_wave_direction_deg: Optional[float] = None
    wind_speed_ms: Optional[float] = None
    wind_direction_deg: Optional[float] = None


@dataclass
class WW3AtPoint:
    """WW3 data interpolated to a single point."""
    lat: float
    lon: float
    timestamp: datetime
    significant_wave_height_m: Optional[float] = None
    peak_period_s: Optional[float] = None
    mean_direction_deg: Optional[float] = None
    wind_wave_height_m: Optional[float] = None
    wind_wave_period_s: Optional[float] = None
    wind_wave_direction_deg: Optional[float] = None
    primary_swell_height_m: Optional[float] = None
    primary_swell_period_s: Optional[float] = None
    primary_swell_direction_deg: Optional[float] = None


@dataclass
class ComparisonResult:
    """Comparison between WW3 and NDBC at a single buoy."""
    station_id: str
    station_name: str
    lat: float
    lon: float

    # Timestamps
    ww3_timestamp: Optional[datetime] = None
    ndbc_timestamp: Optional[datetime] = None
    time_diff_minutes: Optional[float] = None

    # WW3 values
    ww3_hs_m: Optional[float] = None
    ww3_tp_s: Optional[float] = None
    ww3_dir_deg: Optional[float] = None

    # NDBC values
    ndbc_hs_m: Optional[float] = None
    ndbc_tp_s: Optional[float] = None
    ndbc_dir_deg: Optional[float] = None

    # Differences (WW3 - NDBC)
    hs_diff_m: Optional[float] = None
    hs_diff_pct: Optional[float] = None
    tp_diff_s: Optional[float] = None
    tp_diff_pct: Optional[float] = None
    dir_diff_deg: Optional[float] = None

    error: Optional[str] = None


# =============================================================================
# NDBC Historical Fetcher
# =============================================================================

class NDBCHistoricalFetcher:
    """Fetch and parse NDBC historical/realtime data."""

    def __init__(self):
        self.base_url = "https://www.ndbc.noaa.gov"

    def _parse_value(self, value: str) -> Optional[float]:
        """Parse NDBC value, handling missing data markers."""
        if value in ["MM", "999", "9999", "999.0", "9999.0", ""]:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    async def fetch_recent_observations(
        self,
        station_id: str,
        max_hours: int = 48
    ) -> List[NDBCObservation]:
        """
        Fetch recent observations from NDBC realtime2 files.

        The realtime2/*.txt files contain ~45 days of hourly data.

        Args:
            station_id: NDBC station ID
            max_hours: Maximum hours of data to return

        Returns:
            List of NDBCObservation sorted by timestamp (newest first)
        """
        observations = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{self.base_url}/data/realtime2/{station_id}.txt"

            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as e:
                print(f"  Error fetching {station_id}: {e}")
                return observations

            lines = response.text.strip().split("\n")
            if len(lines) < 3:
                return observations

            # Parse header to get column indices
            headers = lines[0].replace("#", "").split()

            # Find column indices
            col_map = {}
            for i, h in enumerate(headers):
                col_map[h] = i

            # Parse data lines (skip header and units lines)
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_hours)

            for line in lines[2:]:  # Skip headers and units
                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    # Parse timestamp (YY MM DD hh mm)
                    year = int(parts[0])
                    if year < 100:
                        year += 2000

                    timestamp = datetime(
                        year=year,
                        month=int(parts[1]),
                        day=int(parts[2]),
                        hour=int(parts[3]),
                        minute=int(parts[4]),
                        tzinfo=timezone.utc
                    )

                    # Skip if too old
                    if timestamp < cutoff_time:
                        break  # Data is sorted newest first, so we can stop

                    obs = NDBCObservation(timestamp=timestamp)

                    # Wave height (WVHT)
                    if "WVHT" in col_map and col_map["WVHT"] < len(parts):
                        obs.wave_height_m = self._parse_value(parts[col_map["WVHT"]])

                    # Dominant period (DPD)
                    if "DPD" in col_map and col_map["DPD"] < len(parts):
                        obs.dominant_period_s = self._parse_value(parts[col_map["DPD"]])

                    # Average period (APD)
                    if "APD" in col_map and col_map["APD"] < len(parts):
                        obs.average_period_s = self._parse_value(parts[col_map["APD"]])

                    # Mean wave direction (MWD)
                    if "MWD" in col_map and col_map["MWD"] < len(parts):
                        obs.mean_wave_direction_deg = self._parse_value(parts[col_map["MWD"]])

                    # Wind speed (WSPD)
                    if "WSPD" in col_map and col_map["WSPD"] < len(parts):
                        obs.wind_speed_ms = self._parse_value(parts[col_map["WSPD"]])

                    # Wind direction (WDIR)
                    if "WDIR" in col_map and col_map["WDIR"] < len(parts):
                        obs.wind_direction_deg = self._parse_value(parts[col_map["WDIR"]])

                    observations.append(obs)

                except (ValueError, IndexError) as e:
                    continue

        return observations

    def find_closest_observation(
        self,
        observations: List[NDBCObservation],
        target_time: datetime,
        max_diff_minutes: int = 60
    ) -> Optional[NDBCObservation]:
        """
        Find the observation closest to target time.

        Args:
            observations: List of observations
            target_time: Target timestamp to match
            max_diff_minutes: Maximum time difference allowed

        Returns:
            Closest observation or None if none within threshold
        """
        if not observations:
            return None

        # Ensure target_time is timezone-aware
        if target_time.tzinfo is None:
            target_time = target_time.replace(tzinfo=timezone.utc)

        closest = None
        min_diff = timedelta(minutes=max_diff_minutes + 1)

        for obs in observations:
            diff = abs(obs.timestamp - target_time)
            if diff < min_diff:
                min_diff = diff
                closest = obs

        if min_diff <= timedelta(minutes=max_diff_minutes):
            return closest
        return None


# =============================================================================
# WW3 Grid Interpolation
# =============================================================================

class WW3Interpolator:
    """Interpolate WW3 gridded data to point locations."""

    def __init__(self, ww3_data: Dict):
        """
        Initialize with WW3 grid data.

        Args:
            ww3_data: Dictionary returned by WaveWatchFetcher.fetch_wave_grid()
        """
        self.data = ww3_data
        self.lats = np.array(ww3_data["lat"])
        self.lons = np.array(ww3_data["lon"])

        # Parse timestamps (ensure timezone-aware)
        self.forecast_time = datetime.fromisoformat(ww3_data["forecast_time"])
        self.cycle_time = datetime.fromisoformat(ww3_data["cycle_time"])

        # Ensure timestamps are timezone-aware (WW3 times are UTC)
        if self.forecast_time.tzinfo is None:
            self.forecast_time = self.forecast_time.replace(tzinfo=timezone.utc)
        if self.cycle_time.tzinfo is None:
            self.cycle_time = self.cycle_time.replace(tzinfo=timezone.utc)

        # Create interpolators for each variable
        self._interpolators = {}

    def _get_interpolator(self, var_name: str) -> Optional[RegularGridInterpolator]:
        """Get or create interpolator for a variable."""
        if var_name in self._interpolators:
            return self._interpolators[var_name]

        if var_name not in self.data:
            return None

        values = np.array(self.data[var_name])

        # Handle NaN values by replacing with nearest valid
        if np.any(np.isnan(values)):
            # Simple approach: leave NaNs, interpolator will return NaN
            pass

        try:
            interp = RegularGridInterpolator(
                (self.lats, self.lons),
                values,
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            self._interpolators[var_name] = interp
            return interp
        except Exception as e:
            print(f"  Warning: Could not create interpolator for {var_name}: {e}")
            return None

    def interpolate_to_point(self, lat: float, lon: float) -> WW3AtPoint:
        """
        Interpolate WW3 data to a single point.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            WW3AtPoint with interpolated values
        """
        result = WW3AtPoint(
            lat=lat,
            lon=lon,
            timestamp=self.forecast_time
        )

        point = np.array([[lat, lon]])

        # Interpolate each variable
        var_mapping = {
            "significant_wave_height": "significant_wave_height_m",
            "peak_wave_period": "peak_period_s",
            "mean_wave_direction": "mean_direction_deg",
            "wind_wave_height": "wind_wave_height_m",
            "wind_wave_period": "wind_wave_period_s",
            "wind_wave_direction": "wind_wave_direction_deg",
            "primary_swell_height": "primary_swell_height_m",
            "primary_swell_period": "primary_swell_period_s",
            "primary_swell_direction": "primary_swell_direction_deg",
        }

        for ww3_var, result_attr in var_mapping.items():
            interp = self._get_interpolator(ww3_var)
            if interp is not None:
                try:
                    value = float(interp(point)[0])
                    if not np.isnan(value):
                        setattr(result, result_attr, value)
                except Exception:
                    pass

        return result


# =============================================================================
# Comparison Logic
# =============================================================================

def direction_difference(d1: Optional[float], d2: Optional[float]) -> Optional[float]:
    """
    Calculate the angular difference between two directions.
    Returns value in range [-180, 180].
    """
    if d1 is None or d2 is None:
        return None

    diff = d1 - d2
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff


async def compare_single_buoy(
    station_id: str,
    station_info: Dict,
    ww3_interpolator: WW3Interpolator,
    ndbc_fetcher: NDBCHistoricalFetcher,
) -> ComparisonResult:
    """Compare WW3 and NDBC for a single buoy."""

    result = ComparisonResult(
        station_id=station_id,
        station_name=station_info["name"],
        lat=station_info["lat"],
        lon=station_info["lon"],
    )

    # Get WW3 data at buoy location
    ww3_point = ww3_interpolator.interpolate_to_point(
        station_info["lat"],
        station_info["lon"]
    )

    result.ww3_timestamp = ww3_point.timestamp
    result.ww3_hs_m = ww3_point.significant_wave_height_m
    result.ww3_tp_s = ww3_point.peak_period_s
    result.ww3_dir_deg = ww3_point.mean_direction_deg

    # Fetch NDBC observations
    observations = await ndbc_fetcher.fetch_recent_observations(station_id)

    if not observations:
        result.error = "No NDBC data available"
        return result

    # Find closest observation to WW3 timestamp
    closest_obs = ndbc_fetcher.find_closest_observation(
        observations,
        ww3_point.timestamp,
        max_diff_minutes=60
    )

    if closest_obs is None:
        result.error = f"No NDBC observation within 60 min of {ww3_point.timestamp}"
        return result

    result.ndbc_timestamp = closest_obs.timestamp
    result.ndbc_hs_m = closest_obs.wave_height_m
    result.ndbc_tp_s = closest_obs.dominant_period_s
    result.ndbc_dir_deg = closest_obs.mean_wave_direction_deg

    # Check if buoy has wave data
    if closest_obs.wave_height_m is None:
        result.error = "Buoy has no wave sensor (met-only station)"
        return result

    # Calculate time difference
    time_diff = abs(ww3_point.timestamp - closest_obs.timestamp)
    result.time_diff_minutes = time_diff.total_seconds() / 60

    # Calculate differences (WW3 - NDBC)
    if result.ww3_hs_m is not None and result.ndbc_hs_m is not None:
        result.hs_diff_m = result.ww3_hs_m - result.ndbc_hs_m
        if result.ndbc_hs_m > 0:
            result.hs_diff_pct = 100 * result.hs_diff_m / result.ndbc_hs_m

    if result.ww3_tp_s is not None and result.ndbc_tp_s is not None:
        result.tp_diff_s = result.ww3_tp_s - result.ndbc_tp_s
        if result.ndbc_tp_s > 0:
            result.tp_diff_pct = 100 * result.tp_diff_s / result.ndbc_tp_s

    result.dir_diff_deg = direction_difference(result.ww3_dir_deg, result.ndbc_dir_deg)

    return result


# =============================================================================
# Display Functions
# =============================================================================

def print_comparison_table(results: List[ComparisonResult]):
    """Print comparison results as a formatted table."""

    print("\n" + "=" * 100)
    print("WW3 vs NDBC BUOY COMPARISON")
    print("=" * 100)

    # Header
    print(f"\n{'Station':<25} {'WW3 Hs':>8} {'NDBC Hs':>8} {'Diff':>8} {'Diff%':>7} | "
          f"{'WW3 Tp':>7} {'NDBC Tp':>7} {'Diff':>6} | "
          f"{'WW3 Dir':>7} {'NDBC Dir':>8} {'Diff':>6}")
    print(f"{'':<25} {'(m)':>8} {'(m)':>8} {'(m)':>8} {'':>7} | "
          f"{'(s)':>7} {'(s)':>7} {'(s)':>6} | "
          f"{'(deg)':>7} {'(deg)':>8} {'(deg)':>6}")
    print("-" * 100)

    # Results
    valid_results = []
    for r in results:
        if r.error:
            print(f"{r.station_name:<25} ERROR: {r.error}")
            continue

        valid_results.append(r)

        hs_ww3 = f"{r.ww3_hs_m:.2f}" if r.ww3_hs_m is not None else "--"
        hs_ndbc = f"{r.ndbc_hs_m:.2f}" if r.ndbc_hs_m is not None else "--"
        hs_diff = f"{r.hs_diff_m:+.2f}" if r.hs_diff_m is not None else "--"
        hs_pct = f"{r.hs_diff_pct:+.0f}%" if r.hs_diff_pct is not None else "--"

        tp_ww3 = f"{r.ww3_tp_s:.1f}" if r.ww3_tp_s is not None else "--"
        tp_ndbc = f"{r.ndbc_tp_s:.1f}" if r.ndbc_tp_s is not None else "--"
        tp_diff = f"{r.tp_diff_s:+.1f}" if r.tp_diff_s is not None else "--"

        dir_ww3 = f"{r.ww3_dir_deg:.0f}" if r.ww3_dir_deg is not None else "--"
        dir_ndbc = f"{r.ndbc_dir_deg:.0f}" if r.ndbc_dir_deg is not None else "--"
        dir_diff = f"{r.dir_diff_deg:+.0f}" if r.dir_diff_deg is not None else "--"

        print(f"{r.station_name:<25} {hs_ww3:>8} {hs_ndbc:>8} {hs_diff:>8} {hs_pct:>7} | "
              f"{tp_ww3:>7} {tp_ndbc:>7} {tp_diff:>6} | "
              f"{dir_ww3:>7} {dir_ndbc:>8} {dir_diff:>6}")

    # Summary statistics
    if valid_results:
        print("-" * 100)

        hs_diffs = [r.hs_diff_m for r in valid_results if r.hs_diff_m is not None]
        tp_diffs = [r.tp_diff_s for r in valid_results if r.tp_diff_s is not None]
        dir_diffs = [r.dir_diff_deg for r in valid_results if r.dir_diff_deg is not None]

        print(f"\nSUMMARY STATISTICS (WW3 - NDBC):")
        print(f"  Valid comparisons: {len(valid_results)}")

        if hs_diffs:
            print(f"\n  Wave Height (Hs):")
            print(f"    Mean bias:  {np.mean(hs_diffs):+.2f} m")
            print(f"    RMSE:       {np.sqrt(np.mean(np.array(hs_diffs)**2)):.2f} m")
            print(f"    Min/Max:    {min(hs_diffs):+.2f} / {max(hs_diffs):+.2f} m")

        if tp_diffs:
            print(f"\n  Peak Period (Tp):")
            print(f"    Mean bias:  {np.mean(tp_diffs):+.1f} s")
            print(f"    RMSE:       {np.sqrt(np.mean(np.array(tp_diffs)**2)):.1f} s")
            print(f"    Min/Max:    {min(tp_diffs):+.1f} / {max(tp_diffs):+.1f} s")

        if dir_diffs:
            print(f"\n  Direction:")
            print(f"    Mean bias:  {np.mean(dir_diffs):+.0f} deg")
            print(f"    RMSE:       {np.sqrt(np.mean(np.array(dir_diffs)**2)):.0f} deg")
            print(f"    Min/Max:    {min(dir_diffs):+.0f} / {max(dir_diffs):+.0f} deg")

    print("\n" + "=" * 100)


def print_detailed_comparison(result: ComparisonResult):
    """Print detailed comparison for a single buoy."""
    print(f"\n--- {result.station_name} ({result.station_id}) ---")
    print(f"Location: ({result.lat:.3f}, {result.lon:.3f})")

    if result.error:
        print(f"Error: {result.error}")
        return

    print(f"\nTimestamps:")
    print(f"  WW3:  {result.ww3_timestamp}")
    print(f"  NDBC: {result.ndbc_timestamp}")
    print(f"  Diff: {result.time_diff_minutes:.0f} minutes")

    print(f"\nWave Height (Hs):")
    print(f"  WW3:  {result.ww3_hs_m:.2f} m" if result.ww3_hs_m else "  WW3:  --")
    print(f"  NDBC: {result.ndbc_hs_m:.2f} m" if result.ndbc_hs_m else "  NDBC: --")
    if result.hs_diff_m is not None:
        print(f"  Diff: {result.hs_diff_m:+.2f} m ({result.hs_diff_pct:+.0f}%)")

    print(f"\nPeak Period (Tp):")
    print(f"  WW3:  {result.ww3_tp_s:.1f} s" if result.ww3_tp_s else "  WW3:  --")
    print(f"  NDBC: {result.ndbc_tp_s:.1f} s" if result.ndbc_tp_s else "  NDBC: --")
    if result.tp_diff_s is not None:
        print(f"  Diff: {result.tp_diff_s:+.1f} s ({result.tp_diff_pct:+.0f}%)")

    print(f"\nDirection:")
    print(f"  WW3:  {result.ww3_dir_deg:.0f} deg" if result.ww3_dir_deg else "  WW3:  --")
    print(f"  NDBC: {result.ndbc_dir_deg:.0f} deg" if result.ndbc_dir_deg else "  NDBC: --")
    if result.dir_diff_deg is not None:
        print(f"  Diff: {result.dir_diff_deg:+.0f} deg")


# =============================================================================
# Main
# =============================================================================

async def main():
    print("=" * 60)
    print("WW3 vs NDBC BUOY COMPARISON")
    print("=" * 60)

    # Step 1: Fetch WW3 data
    print("\n1. Fetching WaveWatch III data...")
    ww3_fetcher = WaveWatchFetcher()

    # Get bounding box that covers all buoys with some margin
    lats = [info["lat"] for info in COMPARISON_BUOYS.values()]
    lons = [info["lon"] for info in COMPARISON_BUOYS.values()]

    min_lat = min(lats) - 1
    max_lat = max(lats) + 1
    min_lon = min(lons) - 1
    max_lon = max(lons) + 1

    print(f"   Bounding box: ({min_lat:.1f}, {min_lon:.1f}) to ({max_lat:.1f}, {max_lon:.1f})")

    ww3_data = await ww3_fetcher.fetch_wave_grid(
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        forecast_hour=0  # Analysis/nowcast
    )

    print(f"   Cycle time:    {ww3_data['cycle_time']}")
    print(f"   Forecast time: {ww3_data['forecast_time']}")
    print(f"   Grid size:     {len(ww3_data['lat'])} x {len(ww3_data['lon'])}")

    # Check if this is synthetic data
    if "synthetic" in ww3_data.get("model", "").lower():
        print("\n   WARNING: Using synthetic fallback data - real WW3 not available!")
        print("   (NOAA NOMADS server may be down - try again later for real comparison)")

    # Step 2: Create interpolator
    print("\n2. Creating WW3 grid interpolator...")
    interpolator = WW3Interpolator(ww3_data)

    # Step 3: Fetch NDBC data and compare
    print("\n3. Fetching NDBC buoy data and comparing...")
    ndbc_fetcher = NDBCHistoricalFetcher()

    results = []
    for station_id, info in COMPARISON_BUOYS.items():
        print(f"   Processing {station_id} ({info['name']})...")
        result = await compare_single_buoy(
            station_id,
            info,
            interpolator,
            ndbc_fetcher
        )
        results.append(result)

    # Step 4: Display results
    print_comparison_table(results)

    # Optional: Print detailed results
    print("\nDETAILED RESULTS:")
    for result in results:
        print_detailed_comparison(result)


if __name__ == "__main__":
    asyncio.run(main())
