#!/usr/bin/env python3
"""
Weather Data Downloader

Downloads forecast data from NOAA and stores as NetCDF files for use with SWAN modeling.

Data sources:
    - WaveWatch III (waves): Significant wave height, period, direction, swell components
    - GFS (wind): 10m wind speed and direction
    - RTOFS (currents): Ocean surface currents
    - NDBC/CDIP (buoys): Real-time buoy observations

Usage:
    python download_weather_data.py                    # Download all data
    python download_weather_data.py --only waves wind  # Download specific data
    python download_weather_data.py --keep 3           # Keep only 3 most recent cycles

Output structure:
    data/downloaded_weather_data/
    ├── waves/ww3_YYYYMMDD_HHz.nc
    ├── wind/gfs_YYYYMMDD_HHz.nc
    ├── currents/rtofs_YYYYMMDD.nc
    ├── buoys/ndbc_YYYYMMDD.parquet
    ├── buoys/cdip_YYYYMMDD.parquet
    └── index.json
"""

import argparse
import asyncio
import json
import logging
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add pipelines to path
SCRIPT_DIR = Path(__file__).parent
PIPELINES_DIR = SCRIPT_DIR / "pipelines"
sys.path.insert(0, str(PIPELINES_DIR))

# Import fetchers
try:
    from wave.wavewatch_fetcher import WaveWatchFetcher
    WAVE_FETCHER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"WaveWatchFetcher not available: {e}")
    WAVE_FETCHER_AVAILABLE = False

try:
    from wind.gfs_fetcher import GFSWindFetcher
    WIND_FETCHER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"GFSWindFetcher not available: {e}")
    WIND_FETCHER_AVAILABLE = False

try:
    from ocean_tiles.rtofs_fetcher import RTOFSFetcher
    CURRENT_FETCHER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"RTOFSFetcher not available: {e}")
    CURRENT_FETCHER_AVAILABLE = False

try:
    from buoy.fetcher import NDBCBuoyFetcher, CDIPBuoyFetcher
    BUOY_FETCHER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Buoy fetchers not available: {e}")
    BUOY_FETCHER_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

# Default region (California coast)
DEFAULT_BOUNDS = {
    "min_lat": 32.0,
    "max_lat": 42.0,
    "min_lon": -125.0,
    "max_lon": -117.0
}

# Forecast hours to download
# 3-hourly from 0-48h, then 24-hourly out to 168h (7 days)
DEFAULT_FORECAST_HOURS = list(range(0, 49, 3)) + [72, 96, 120, 144, 168]

# RTOFS forecast hours (daily model, 8-day forecast)
RTOFS_FORECAST_HOURS = list(range(0, 49, 3)) + [72, 96, 120, 144, 168, 192]

# Output directory
OUTPUT_DIR = SCRIPT_DIR / "downloaded_weather_data"


# =============================================================================
# Base Downloader Class
# =============================================================================

class BaseDownloader(ABC):
    """Abstract base class for all weather data downloaders."""

    name: str = "Base"
    folder_name: str = "base"
    file_prefix: str = "data"

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        """Initialize downloader with output directory."""
        self.output_dir = Path(output_dir)
        self.data_dir = self.output_dir / self.folder_name
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    async def download(self) -> Optional[Path]:
        """Download data and save to file. Returns path to saved file or None."""
        pass

    def get_cycle_filename(self, cycle_time: datetime, extension: str = ".nc") -> str:
        """Generate filename for a model cycle (e.g., ww3_20240115_12z.nc)."""
        date_str = cycle_time.strftime("%Y%m%d")
        hour_str = f"{cycle_time.hour:02d}z"
        return f"{self.file_prefix}_{date_str}_{hour_str}{extension}"

    def get_daily_filename(self, date: datetime, extension: str = ".nc") -> str:
        """Generate filename for daily data (e.g., rtofs_20240115.nc)."""
        date_str = date.strftime("%Y%m%d")
        return f"{self.file_prefix}_{date_str}{extension}"

    def list_existing_files(self) -> List[Path]:
        """List existing data files, sorted by modification time (oldest first)."""
        files = list(self.data_dir.glob(f"{self.file_prefix}_*"))
        files.sort(key=lambda f: f.stat().st_mtime)
        return files

    def cleanup_old_files(self, keep: int = 3) -> List[Path]:
        """Remove old files, keeping only the most recent ones."""
        files = self.list_existing_files()
        deleted = []
        if len(files) > keep:
            for f in files[:-keep]:
                try:
                    f.unlink()
                    deleted.append(f)
                    self.log(f"Deleted old file: {f.name}")
                except Exception as e:
                    self.log(f"Failed to delete {f.name}: {e}", "warning")
        return deleted

    def log(self, message: str, level: str = "info"):
        """Print progress message with consistent formatting."""
        prefix = f"[{self.name}]"
        if level == "info":
            print(f"  {prefix} {message}")
        elif level == "warning":
            print(f"  {prefix} ⚠ {message}")
        elif level == "error":
            print(f"  {prefix} ✗ {message}")
        elif level == "success":
            print(f"  {prefix} ✓ {message}")

    def save_netcdf(self, ds: xr.Dataset, filepath: Path) -> bool:
        """Save xarray Dataset to NetCDF with compression."""
        try:
            encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
            ds.to_netcdf(filepath, encoding=encoding)
            return True
        except Exception as e:
            logger.error(f"Failed to save NetCDF: {e}")
            return False


# =============================================================================
# Wave Downloader (WaveWatch III)
# =============================================================================

class WaveDownloader(BaseDownloader):
    """Downloads WaveWatch III wave forecast data."""

    name = "Waves"
    folder_name = "waves"
    file_prefix = "ww3"

    def __init__(self, output_dir: Path = OUTPUT_DIR, bounds: Dict = None,
                 forecast_hours: List[int] = None):
        super().__init__(output_dir)
        self.bounds = bounds or DEFAULT_BOUNDS
        self.forecast_hours = forecast_hours or DEFAULT_FORECAST_HOURS
        self.fetcher = WaveWatchFetcher() if WAVE_FETCHER_AVAILABLE else None

    async def download(self) -> Optional[Path]:
        """Download WW3 wave forecast and save as NetCDF."""
        if not self.fetcher:
            self.log("WaveWatchFetcher not available", "error")
            return None

        self.log("Starting download...")

        # Fetch first hour to get cycle time and coordinates
        try:
            first_data = await self.fetcher.fetch_wave_grid(
                min_lat=self.bounds["min_lat"],
                max_lat=self.bounds["max_lat"],
                min_lon=self.bounds["min_lon"],
                max_lon=self.bounds["max_lon"],
                forecast_hour=0
            )
        except Exception as e:
            self.log(f"Failed to fetch initial data: {e}", "error")
            return None

        if not first_data:
            self.log("No data returned", "error")
            return None

        cycle_time = datetime.fromisoformat(first_data["cycle_time"])
        self.log(f"Model cycle: {cycle_time.strftime('%Y-%m-%d %H:%M UTC')}")

        lats = np.array(first_data["lat"])
        lons = np.array(first_data["lon"])

        # Variables to extract
        var_names = [
            "significant_wave_height", "peak_wave_period", "mean_wave_direction",
            "wind_wave_height", "wind_wave_period", "wind_wave_direction",
            "primary_swell_height", "primary_swell_period", "primary_swell_direction",
        ]

        # Initialize storage
        n_times = len(self.forecast_hours)
        data_arrays = {v: np.full((n_times, len(lats), len(lons)), np.nan, dtype=np.float32)
                       for v in var_names}
        times = []

        # Download each forecast hour
        successful = 0
        for i, hour in enumerate(self.forecast_hours):
            try:
                self.log(f"Fetching hour {hour:03d} ({i+1}/{n_times})...")

                if hour == 0:
                    data = first_data
                else:
                    data = await self.fetcher.fetch_wave_grid(
                        min_lat=self.bounds["min_lat"],
                        max_lat=self.bounds["max_lat"],
                        min_lon=self.bounds["min_lon"],
                        max_lon=self.bounds["max_lon"],
                        forecast_hour=hour
                    )

                if data:
                    times.append(datetime.fromisoformat(data["forecast_time"]))
                    for var in var_names:
                        if var in data and data[var]:
                            arr = np.array(data[var], dtype=np.float32)
                            data_arrays[var][i] = arr
                    successful += 1
                else:
                    times.append(cycle_time + timedelta(hours=hour))

            except Exception as e:
                logger.warning(f"Failed hour {hour}: {e}")
                times.append(cycle_time + timedelta(hours=hour))

        if successful == 0:
            self.log("No forecast hours downloaded", "error")
            return None

        self.log(f"Downloaded {successful}/{n_times} forecast hours")

        # Create NetCDF dataset
        ds = xr.Dataset(
            data_vars={
                "significant_wave_height": (["time", "lat", "lon"], data_arrays["significant_wave_height"],
                                            {"units": "m", "long_name": "Significant wave height"}),
                "peak_wave_period": (["time", "lat", "lon"], data_arrays["peak_wave_period"],
                                     {"units": "s", "long_name": "Peak wave period"}),
                "mean_wave_direction": (["time", "lat", "lon"], data_arrays["mean_wave_direction"],
                                        {"units": "degrees", "long_name": "Mean wave direction (from)"}),
                "wind_wave_height": (["time", "lat", "lon"], data_arrays["wind_wave_height"],
                                     {"units": "m", "long_name": "Wind wave height"}),
                "wind_wave_period": (["time", "lat", "lon"], data_arrays["wind_wave_period"],
                                     {"units": "s", "long_name": "Wind wave period"}),
                "wind_wave_direction": (["time", "lat", "lon"], data_arrays["wind_wave_direction"],
                                        {"units": "degrees", "long_name": "Wind wave direction (from)"}),
                "primary_swell_height": (["time", "lat", "lon"], data_arrays["primary_swell_height"],
                                         {"units": "m", "long_name": "Primary swell height"}),
                "primary_swell_period": (["time", "lat", "lon"], data_arrays["primary_swell_period"],
                                         {"units": "s", "long_name": "Primary swell period"}),
                "primary_swell_direction": (["time", "lat", "lon"], data_arrays["primary_swell_direction"],
                                            {"units": "degrees", "long_name": "Primary swell direction (from)"}),
            },
            coords={
                "time": pd.to_datetime(times),
                "lat": lats,
                "lon": lons,
            },
            attrs={
                "model": "WaveWatch III",
                "source": "NOAA NCEP",
                "cycle_time": cycle_time.isoformat(),
                "created": datetime.utcnow().isoformat(),
            }
        )

        # Save
        filename = self.get_cycle_filename(cycle_time)
        filepath = self.data_dir / filename

        if self.save_netcdf(ds, filepath):
            size_mb = filepath.stat().st_size / (1024 * 1024)
            self.log(f"Saved: {filename} ({size_mb:.1f} MB)", "success")
            return filepath
        else:
            self.log("Failed to save file", "error")
            return None


# =============================================================================
# Wind Downloader (GFS)
# =============================================================================

class WindDownloader(BaseDownloader):
    """Downloads GFS wind forecast data."""

    name = "Wind"
    folder_name = "wind"
    file_prefix = "gfs"

    def __init__(self, output_dir: Path = OUTPUT_DIR, bounds: Dict = None,
                 forecast_hours: List[int] = None):
        super().__init__(output_dir)
        self.bounds = bounds or DEFAULT_BOUNDS
        self.forecast_hours = forecast_hours or DEFAULT_FORECAST_HOURS
        self.fetcher = GFSWindFetcher() if WIND_FETCHER_AVAILABLE else None

    async def download(self) -> Optional[Path]:
        """Download GFS wind forecast and save as NetCDF."""
        if not self.fetcher:
            self.log("GFSWindFetcher not available", "error")
            return None

        self.log("Starting download...")

        # Fetch first hour
        try:
            first_data = await self.fetcher.fetch_wind_grid(
                min_lat=self.bounds["min_lat"],
                max_lat=self.bounds["max_lat"],
                min_lon=self.bounds["min_lon"],
                max_lon=self.bounds["max_lon"],
                forecast_hour=0
            )
        except Exception as e:
            self.log(f"Failed to fetch initial data: {e}", "error")
            return None

        if not first_data:
            self.log("No data returned", "error")
            return None

        cycle_time = datetime.fromisoformat(first_data["cycle_time"])
        self.log(f"Model cycle: {cycle_time.strftime('%Y-%m-%d %H:%M UTC')}")

        lats = np.array(first_data["lat"])
        lons = np.array(first_data["lon"])

        var_names = ["u_wind", "v_wind", "wind_speed", "wind_direction"]

        n_times = len(self.forecast_hours)
        data_arrays = {v: np.full((n_times, len(lats), len(lons)), np.nan, dtype=np.float32)
                       for v in var_names}
        times = []

        successful = 0
        for i, hour in enumerate(self.forecast_hours):
            try:
                self.log(f"Fetching hour {hour:03d} ({i+1}/{n_times})...")

                if hour == 0:
                    data = first_data
                else:
                    data = await self.fetcher.fetch_wind_grid(
                        min_lat=self.bounds["min_lat"],
                        max_lat=self.bounds["max_lat"],
                        min_lon=self.bounds["min_lon"],
                        max_lon=self.bounds["max_lon"],
                        forecast_hour=hour
                    )

                if data:
                    times.append(datetime.fromisoformat(data["forecast_time"]))
                    for var in var_names:
                        if var in data and data[var]:
                            data_arrays[var][i] = np.array(data[var], dtype=np.float32)
                    successful += 1
                else:
                    times.append(cycle_time + timedelta(hours=hour))

            except Exception as e:
                logger.warning(f"Failed hour {hour}: {e}")
                times.append(cycle_time + timedelta(hours=hour))

        if successful == 0:
            self.log("No forecast hours downloaded", "error")
            return None

        self.log(f"Downloaded {successful}/{n_times} forecast hours")

        ds = xr.Dataset(
            data_vars={
                "u_wind": (["time", "lat", "lon"], data_arrays["u_wind"],
                           {"units": "m/s", "long_name": "U-component of wind at 10m"}),
                "v_wind": (["time", "lat", "lon"], data_arrays["v_wind"],
                           {"units": "m/s", "long_name": "V-component of wind at 10m"}),
                "wind_speed": (["time", "lat", "lon"], data_arrays["wind_speed"],
                               {"units": "m/s", "long_name": "Wind speed at 10m"}),
                "wind_direction": (["time", "lat", "lon"], data_arrays["wind_direction"],
                                   {"units": "degrees", "long_name": "Wind direction (meteorological)"}),
            },
            coords={
                "time": pd.to_datetime(times),
                "lat": lats,
                "lon": lons,
            },
            attrs={
                "model": "GFS",
                "source": "NOAA NCEP",
                "cycle_time": cycle_time.isoformat(),
                "created": datetime.utcnow().isoformat(),
            }
        )

        filename = self.get_cycle_filename(cycle_time)
        filepath = self.data_dir / filename

        if self.save_netcdf(ds, filepath):
            size_mb = filepath.stat().st_size / (1024 * 1024)
            self.log(f"Saved: {filename} ({size_mb:.1f} MB)", "success")
            return filepath
        else:
            self.log("Failed to save file", "error")
            return None


# =============================================================================
# Current Downloader (RTOFS)
# =============================================================================

class CurrentDownloader(BaseDownloader):
    """Downloads RTOFS ocean current forecast data."""

    name = "Currents"
    folder_name = "currents"
    file_prefix = "rtofs"

    def __init__(self, output_dir: Path = OUTPUT_DIR, forecast_hours: List[int] = None):
        super().__init__(output_dir)
        self.forecast_hours = forecast_hours or RTOFS_FORECAST_HOURS
        self.fetcher = RTOFSFetcher() if CURRENT_FETCHER_AVAILABLE else None

    async def download(self) -> Optional[Path]:
        """Download RTOFS current forecast and save as NetCDF."""
        if not self.fetcher:
            self.log("RTOFSFetcher not available", "error")
            return None

        self.log("Starting download...")

        model_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

        # Fetch first hour to get grid structure
        try:
            first_data = await self.fetcher.fetch_current_grid(
                region_name="california",
                forecast_hour=0,
                model_date=model_date
            )
        except Exception as e:
            self.log(f"Failed to fetch initial data: {e}", "error")
            return None

        if not first_data:
            self.log("No data returned", "error")
            return None

        # RTOFS uses 2D lat/lon arrays (curvilinear grid)
        lats_2d = first_data["lats"]
        lons_2d = first_data["lons"]
        actual_model_date = datetime.fromisoformat(first_data["metadata"]["model_time"])

        self.log(f"Model date: {actual_model_date.strftime('%Y-%m-%d')}")
        self.log(f"Grid shape: {lats_2d.shape}")

        var_names = ["u_velocity", "v_velocity", "current_speed", "current_direction"]

        n_times = len(self.forecast_hours)
        grid_shape = lats_2d.shape
        data_arrays = {v: np.full((n_times, *grid_shape), np.nan, dtype=np.float32)
                       for v in var_names}
        times = []

        successful = 0
        for i, hour in enumerate(self.forecast_hours):
            try:
                self.log(f"Fetching hour {hour:03d} ({i+1}/{n_times})...")

                if hour == 0:
                    data = first_data
                else:
                    data = await self.fetcher.fetch_current_grid(
                        region_name="california",
                        forecast_hour=hour,
                        model_date=model_date
                    )

                if data:
                    valid_time = datetime.fromisoformat(data["metadata"]["valid_time"])
                    times.append(valid_time)
                    data_arrays["u_velocity"][i] = data["u_velocity"]
                    data_arrays["v_velocity"][i] = data["v_velocity"]
                    data_arrays["current_speed"][i] = data["current_speed"]
                    data_arrays["current_direction"][i] = data["current_direction"]
                    successful += 1
                else:
                    times.append(actual_model_date + timedelta(hours=hour))

            except Exception as e:
                logger.warning(f"Failed hour {hour}: {e}")
                times.append(actual_model_date + timedelta(hours=hour))

        if successful == 0:
            self.log("No forecast hours downloaded", "error")
            return None

        self.log(f"Downloaded {successful}/{n_times} forecast hours")

        # For RTOFS curvilinear grid, store lat/lon as 2D data variables
        ds = xr.Dataset(
            data_vars={
                "latitude": (["y", "x"], lats_2d, {"units": "degrees_north"}),
                "longitude": (["y", "x"], lons_2d, {"units": "degrees_east"}),
                "u_velocity": (["time", "y", "x"], data_arrays["u_velocity"],
                               {"units": "m/s", "long_name": "Eastward current velocity"}),
                "v_velocity": (["time", "y", "x"], data_arrays["v_velocity"],
                               {"units": "m/s", "long_name": "Northward current velocity"}),
                "current_speed": (["time", "y", "x"], data_arrays["current_speed"],
                                  {"units": "m/s", "long_name": "Current speed"}),
                "current_direction": (["time", "y", "x"], data_arrays["current_direction"],
                                      {"units": "degrees", "long_name": "Current direction (toward)"}),
            },
            coords={
                "time": pd.to_datetime(times),
                "y": np.arange(grid_shape[0]),
                "x": np.arange(grid_shape[1]),
            },
            attrs={
                "model": "RTOFS",
                "source": "NOAA NCEP",
                "model_date": actual_model_date.isoformat(),
                "created": datetime.utcnow().isoformat(),
                "grid_type": "curvilinear",
            }
        )

        filename = self.get_daily_filename(actual_model_date)
        filepath = self.data_dir / filename

        if self.save_netcdf(ds, filepath):
            size_mb = filepath.stat().st_size / (1024 * 1024)
            self.log(f"Saved: {filename} ({size_mb:.1f} MB)", "success")
            return filepath
        else:
            self.log("Failed to save file", "error")
            return None


# =============================================================================
# Buoy Downloader (NDBC + CDIP)
# =============================================================================

class BuoyDownloader(BaseDownloader):
    """Downloads buoy observations from NDBC and CDIP."""

    name = "Buoys"
    folder_name = "buoys"
    file_prefix = "buoys"

    # California buoys to fetch
    NDBC_STATIONS = ["46237", "46221", "46025", "46011", "46054", "46026", "46012", "46042"]
    CDIP_STATIONS = ["028", "045", "067", "071", "094", "100", "107", "132", "157", "168"]

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        super().__init__(output_dir)
        self.ndbc_fetcher = NDBCBuoyFetcher() if BUOY_FETCHER_AVAILABLE else None
        self.cdip_fetcher = CDIPBuoyFetcher() if BUOY_FETCHER_AVAILABLE else None

    async def download(self) -> Optional[Path]:
        """Download buoy observations and save as Parquet files."""
        if not BUOY_FETCHER_AVAILABLE:
            self.log("Buoy fetchers not available", "error")
            return None

        today = datetime.utcnow().date()
        ndbc_path = None
        cdip_path = None

        # Download NDBC data
        if self.ndbc_fetcher:
            self.log("Fetching NDBC buoy data...")
            ndbc_records = []

            for station in self.NDBC_STATIONS:
                try:
                    self.log(f"  Station {station}...")
                    obs = await self.ndbc_fetcher.fetch_latest_observation(station)
                    spec = await self.ndbc_fetcher.fetch_spectral_wave_data(station)

                    record = {
                        "station_id": station,
                        "timestamp": obs.get("timestamp"),
                        "wave_height_m": obs.get("wave_height_m"),
                        "dominant_period_s": obs.get("dominant_wave_period_s"),
                        "mean_direction_deg": obs.get("mean_wave_direction_deg"),
                        "water_temp_c": obs.get("water_temp_c"),
                        "wind_speed_ms": obs.get("wind_speed_ms"),
                        "wind_direction_deg": obs.get("wind_direction_deg"),
                    }

                    # Add spectral data if available
                    if spec and spec.get("swell"):
                        record["swell_height_m"] = spec["swell"].get("height_m")
                        record["swell_period_s"] = spec["swell"].get("period_s")
                        record["swell_direction_deg"] = spec["swell"].get("direction_deg")

                    if spec and spec.get("wind_waves"):
                        record["wind_wave_height_m"] = spec["wind_waves"].get("height_m")

                    ndbc_records.append(record)

                except Exception as e:
                    logger.warning(f"Failed to fetch NDBC {station}: {e}")

            if ndbc_records:
                df = pd.DataFrame(ndbc_records)
                ndbc_filename = f"ndbc_{today.strftime('%Y%m%d')}.csv"
                ndbc_path = self.data_dir / ndbc_filename
                df.to_csv(ndbc_path, index=False)
                self.log(f"Saved: {ndbc_filename} ({len(ndbc_records)} stations)", "success")

        # Download CDIP data
        if self.cdip_fetcher:
            self.log("Fetching CDIP buoy data...")
            cdip_records = []

            for station in self.CDIP_STATIONS:
                try:
                    self.log(f"  Station {station}...")
                    obs = await self.cdip_fetcher.fetch_latest_observation(station)

                    if obs.get("status") == "success":
                        record = {
                            "station_id": station,
                            "timestamp": obs.get("timestamp"),
                            "wave_height_m": obs.get("wave_height_m"),
                            "dominant_period_s": obs.get("dominant_period_s"),
                            "peak_direction_deg": obs.get("peak_direction_deg"),
                            "mean_direction_deg": obs.get("mean_direction_deg"),
                            "water_temp_c": obs.get("water_temp_c"),
                        }
                        cdip_records.append(record)

                except Exception as e:
                    logger.warning(f"Failed to fetch CDIP {station}: {e}")

            if cdip_records:
                df = pd.DataFrame(cdip_records)
                cdip_filename = f"cdip_{today.strftime('%Y%m%d')}.csv"
                cdip_path = self.data_dir / cdip_filename
                df.to_csv(cdip_path, index=False)
                self.log(f"Saved: {cdip_filename} ({len(cdip_records)} stations)", "success")

        return ndbc_path or cdip_path

    def cleanup_old_files(self, keep: int = 7) -> List[Path]:
        """Override to handle both NDBC and CDIP files separately."""
        deleted = []
        for prefix in ["ndbc", "cdip"]:
            files = sorted(self.data_dir.glob(f"{prefix}_*.csv"),
                           key=lambda f: f.stat().st_mtime)
            if len(files) > keep:
                for f in files[:-keep]:
                    try:
                        f.unlink()
                        deleted.append(f)
                        self.log(f"Deleted: {f.name}")
                    except Exception as e:
                        self.log(f"Failed to delete {f.name}: {e}", "warning")
        return deleted


# =============================================================================
# Download Manager
# =============================================================================

class DownloadManager:
    """Orchestrates downloading of all weather data."""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # =====================================================================
        # AVAILABLE DATA SOURCES
        # =====================================================================
        # All available downloaders (can be selected with --only flag)
        self.downloaders = {
            "waves": WaveDownloader(output_dir),
            "wind": WindDownloader(output_dir),
            "currents": CurrentDownloader(output_dir),
            "buoys": BuoyDownloader(output_dir),
        }

        # =====================================================================
        # DEFAULT DATA SOURCES (downloaded when no --only flag is specified)
        # =====================================================================
        # NOTE: "currents" (RTOFS) is excluded by default because it downloads
        # ~185 MB per forecast hour (no server-side filtering available).
        # This makes the default download take ~5 minutes instead of ~30+ minutes.
        #
        # To include currents by default, add "currents" to this list:
        # =====================================================================
        self.default_types = ["waves", "wind", "buoys"]
        # self.default_types = ["waves", "wind", "currents", "buoys"]  # With currents

        self.index_file = self.output_dir / "index.json"

    async def download_all(self, only: List[str] = None, keep: int = 3) -> Dict[str, Path]:
        """
        Download all (or selected) weather data.

        Args:
            only: List of data types to download (e.g., ["waves", "wind"]).
                  If None, downloads all.
            keep: Number of old cycles to keep. Set to 0 to keep all.

        Returns:
            Dictionary mapping data type to saved file path
        """
        results = {}
        types_to_download = only or self.default_types

        print("\n" + "=" * 60)
        print("  WEATHER DATA DOWNLOADER")
        print("=" * 60)
        print(f"  Output: {self.output_dir}")
        print(f"  Types:  {', '.join(types_to_download)}")
        print(f"  Keep:   {keep} cycles" if keep > 0 else "  Keep:   all")
        print("=" * 60 + "\n")

        for data_type in types_to_download:
            if data_type not in self.downloaders:
                print(f"  Unknown data type: {data_type}")
                continue

            downloader = self.downloaders[data_type]

            print(f"\n{'─' * 50}")
            print(f"  Downloading: {downloader.name}")
            print(f"{'─' * 50}")

            try:
                filepath = await downloader.download()
                if filepath:
                    results[data_type] = filepath

                    # Cleanup old files
                    if keep > 0:
                        downloader.cleanup_old_files(keep)

            except Exception as e:
                print(f"  [{downloader.name}] ✗ Error: {e}")
                logger.exception(f"Failed to download {data_type}")

        # Update index file
        self._update_index(results)

        # Print summary
        print(f"\n{'=' * 60}")
        print("  DOWNLOAD COMPLETE")
        print("=" * 60)
        for data_type, filepath in results.items():
            print(f"  ✓ {data_type}: {filepath.name}")
        if len(results) < len(types_to_download):
            failed = set(types_to_download) - set(results.keys())
            for data_type in failed:
                print(f"  ✗ {data_type}: FAILED")
        print("=" * 60 + "\n")

        return results

    def _update_index(self, results: Dict[str, Path]):
        """Update the index.json file with latest download info."""
        index = {
            "last_updated": datetime.utcnow().isoformat(),
            "latest": {},
            "available_cycles": {},
            "region": DEFAULT_BOUNDS,
        }

        for data_type, filepath in results.items():
            index["latest"][data_type] = filepath.name

        # List all available files for each type
        for data_type, downloader in self.downloaders.items():
            files = downloader.list_existing_files()
            index["available_cycles"][data_type] = [f.name for f in files]

        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download weather forecast data for surf modeling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_weather_data.py                     # Download all data
  python download_weather_data.py --only waves wind   # Download waves and wind only
  python download_weather_data.py --keep 5            # Keep 5 most recent cycles
  python download_weather_data.py --only waves --keep 1  # Latest waves only
        """
    )

    parser.add_argument(
        "--only",
        nargs="+",
        choices=["waves", "wind", "currents", "buoys"],
        help="Download only specific data types"
    )

    parser.add_argument(
        "--keep",
        type=int,
        default=3,
        help="Number of old cycles to keep (default: 3, use 0 to keep all)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )

    args = parser.parse_args()

    # Run the download
    manager = DownloadManager(output_dir=args.output)
    asyncio.run(manager.download_all(only=args.only, keep=args.keep))


if __name__ == "__main__":
    main()
