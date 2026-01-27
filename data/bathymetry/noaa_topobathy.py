#!/usr/bin/env python3
"""
NOAA 2009-2011 Topobathy DEM Handler

Accesses the NOAA Coastal California Topobathy DEM (2013 merge) via local VRT files.

Dataset: 2009-2011 Topobathy Elevation DEM (with voids): Coastal California (2013 merge)
- Resolution: ~1m
- Coverage: Coastal California
- CRS: UTM Zone 10 (EPSG:26910) and UTM Zone 11 (EPSG:26911)
- Convention: Elevation in meters (positive up, negative below sea level)

Data location: data/raw/bathymetry/noaa_topobathy_2011/California_Topobathy_DEM_2011_2616/
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import sys


class NOAATopobathy:
    """
    Handler for NOAA California Topobathy DEM via local VRT files.

    The dataset is split across two UTM zones:
    - Zone 10 (EPSG:26910): Northern/Central California
    - Zone 11 (EPSG:26911): Southern California
    """

    # Default data location relative to project root
    DEFAULT_DATA_DIR = Path("data/raw/bathymetry/noaa_topobathy_2011/California_Topobathy_DEM_2011_2616")

    # Approximate UTM zone boundary (longitude)
    ZONE_BOUNDARY_LON = -120.0  # West of this is Zone 10, East is Zone 11

    def __init__(self, data_dir: Optional[Path] = None, verbose: bool = True):
        """
        Initialize the NOAA Topobathy handler.

        Args:
            data_dir: Directory containing VRT files. If None, uses default.
            verbose: Print progress information
        """
        self.verbose = verbose
        self._ds_zone10 = None
        self._ds_zone11 = None
        self._transformer_to_utm10 = None
        self._transformer_to_utm11 = None
        self._transformer_from_utm10 = None
        self._transformer_from_utm11 = None

        # Resolve data directory
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / self.DEFAULT_DATA_DIR

        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"NOAA Topobathy data directory not found: {self.data_dir}")

        # VRT file paths
        self.vrt_zone10 = self.data_dir / "California_Topobathy_DEM_2011_EPSG-26910.vrt"
        self.vrt_zone11 = self.data_dir / "California_Topobathy_DEM_2011_EPSG-26911.vrt"

        if not self.vrt_zone10.exists():
            raise FileNotFoundError(f"VRT file not found: {self.vrt_zone10}")
        if not self.vrt_zone11.exists():
            raise FileNotFoundError(f"VRT file not found: {self.vrt_zone11}")

        if self.verbose:
            print("NOAA Topobathy DEM (2009-2011 Coastal California)")
            print(f"  Data dir: {self.data_dir}")
            print("  Resolution: ~1m")
            print("  Access: Local VRT files")

    def _init_transformers(self) -> None:
        """Initialize coordinate transformers."""
        from pyproj import Transformer

        if self._transformer_to_utm10 is None:
            self._transformer_to_utm10 = Transformer.from_crs(
                "EPSG:4326", "EPSG:26910", always_xy=True
            )
            self._transformer_from_utm10 = Transformer.from_crs(
                "EPSG:26910", "EPSG:4326", always_xy=True
            )

        if self._transformer_to_utm11 is None:
            self._transformer_to_utm11 = Transformer.from_crs(
                "EPSG:4326", "EPSG:26911", always_xy=True
            )
            self._transformer_from_utm11 = Transformer.from_crs(
                "EPSG:26911", "EPSG:4326", always_xy=True
            )

    def _get_dataset(self, zone: int):
        """Get rasterio dataset for specified UTM zone (lazy loaded)."""
        import rasterio

        if zone == 10:
            if self._ds_zone10 is None:
                if self.verbose:
                    print(f"  Opening VRT for UTM Zone 10...")
                    sys.stdout.flush()
                self._ds_zone10 = rasterio.open(str(self.vrt_zone10))
                if self.verbose:
                    print(f"    Size: {self._ds_zone10.width} x {self._ds_zone10.height}")
            return self._ds_zone10
        else:
            if self._ds_zone11 is None:
                if self.verbose:
                    print(f"  Opening VRT for UTM Zone 11...")
                    sys.stdout.flush()
                self._ds_zone11 = rasterio.open(str(self.vrt_zone11))
                if self.verbose:
                    print(f"    Size: {self._ds_zone11.width} x {self._ds_zone11.height}")
            return self._ds_zone11

    def _determine_utm_zone(self, lon: float) -> int:
        """Determine UTM zone for a longitude."""
        return 10 if lon < self.ZONE_BOUNDARY_LON else 11

    def lon_lat_to_utm(self, lon: np.ndarray, lat: np.ndarray,
                       zone: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convert lon/lat to UTM coordinates for specified zone."""
        self._init_transformers()

        if zone == 10:
            return self._transformer_to_utm10.transform(lon, lat)
        else:
            return self._transformer_to_utm11.transform(lon, lat)

    def utm_to_lon_lat(self, x: np.ndarray, y: np.ndarray,
                       zone: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convert UTM coordinates to lon/lat."""
        self._init_transformers()

        if zone == 10:
            return self._transformer_from_utm10.transform(x, y)
        else:
            return self._transformer_from_utm11.transform(x, y)

    def sample_point(self, lon: float, lat: float) -> Optional[float]:
        """
        Sample elevation at a single point.

        Args:
            lon: Longitude (degrees)
            lat: Latitude (degrees)

        Returns:
            Elevation in meters (positive up), or None if no data
        """
        zone = self._determine_utm_zone(lon)
        ds = self._get_dataset(zone)

        # Transform to UTM
        x, y = self.lon_lat_to_utm(np.array([lon]), np.array([lat]), zone)
        x, y = x[0], y[0]

        # Convert to pixel coordinates
        row, col = ds.index(x, y)

        # Check bounds
        if row < 0 or row >= ds.height or col < 0 or col >= ds.width:
            return None

        # Read single pixel
        try:
            value = ds.read(1, window=((row, row+1), (col, col+1)))[0, 0]
        except Exception:
            return None

        # Check for nodata
        if ds.nodata is not None and value == ds.nodata:
            return None

        return float(value)

    def sample_points(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """
        Sample elevation at multiple points efficiently.

        Uses windowed reads to minimize data transfer.

        Args:
            lons: Array of longitudes
            lats: Array of latitudes

        Returns:
            Array of elevations (NaN where no data)
        """
        lons = np.atleast_1d(lons).astype(np.float64)
        lats = np.atleast_1d(lats).astype(np.float64)

        if lons.shape != lats.shape:
            raise ValueError("lons and lats must have same shape")

        original_shape = lons.shape
        lons_flat = lons.ravel()
        lats_flat = lats.ravel()
        n_points = len(lons_flat)

        elevations = np.full(n_points, np.nan, dtype=np.float32)

        # Separate points by UTM zone
        zone10_mask = lons_flat < self.ZONE_BOUNDARY_LON
        zone11_mask = ~zone10_mask

        # Process each zone
        for zone, mask in [(10, zone10_mask), (11, zone11_mask)]:
            n_zone = np.sum(mask)
            if n_zone == 0:
                continue

            if self.verbose:
                print(f"    Processing {n_zone:,} points in UTM Zone {zone}...")
                sys.stdout.flush()

            indices = np.where(mask)[0]
            zone_lons = lons_flat[mask]
            zone_lats = lats_flat[mask]

            zone_elevs = self._sample_points_zone(zone_lons, zone_lats, zone)
            elevations[indices] = zone_elevs

        return elevations.reshape(original_shape)

    def _sample_points_zone(self, lons: np.ndarray, lats: np.ndarray,
                            zone: int) -> np.ndarray:
        """Sample points within a single UTM zone."""
        from rasterio.windows import Window
        import time

        n_points = len(lons)
        elevations = np.full(n_points, np.nan, dtype=np.float32)

        ds = self._get_dataset(zone)
        nodata = ds.nodata

        # Transform all points to UTM
        x_utm, y_utm = self.lon_lat_to_utm(lons, lats, zone)

        # Get inverse transform for pixel coordinates
        inv_transform = ~ds.transform

        # Convert to pixel coordinates
        cols, rows = inv_transform * (x_utm, y_utm)
        cols = cols.astype(np.int32)
        rows = rows.astype(np.int32)

        # Filter valid pixels
        valid = (
            (rows >= 0) & (rows < ds.height) &
            (cols >= 0) & (cols < ds.width)
        )

        n_valid = np.sum(valid)
        if n_valid == 0:
            if self.verbose:
                print(f"      No points within raster bounds!")
            return elevations

        valid_indices = np.where(valid)[0]
        valid_cols = cols[valid]
        valid_rows = rows[valid]

        if self.verbose:
            print(f"      {n_valid:,} points within raster bounds")

        # Find bounding box for all points
        col_min, col_max = int(valid_cols.min()), int(valid_cols.max())
        row_min, row_max = int(valid_rows.min()), int(valid_rows.max())
        width = col_max - col_min + 1
        height = row_max - row_min + 1

        start_time = time.time()

        # Local reads are fast, so we can read larger windows
        # Limit to ~400M pixels (~1.5GB for float32)
        max_window_size = 20000 * 20000

        if width * height <= max_window_size:
            # Read entire bounding box at once
            if self.verbose:
                print(f"      Reading {width}x{height} window ({width*height:,} pixels)...", end="")
                sys.stdout.flush()

            try:
                window = Window(col_min, row_min, width, height)
                data = ds.read(1, window=window)

                # Extract values at our points
                local_cols = valid_cols - col_min
                local_rows = valid_rows - row_min
                values = data[local_rows, local_cols]

                # Handle nodata
                if nodata is not None:
                    values = np.where(values == nodata, np.nan, values)

                elevations[valid_indices] = values

                valid_count = np.sum(~np.isnan(values))
                elapsed = time.time() - start_time

                if self.verbose:
                    print(f" OK ({valid_count:,} valid, {elapsed:.1f}s)")

            except Exception as e:
                if self.verbose:
                    print(f" ERROR: {e}")
        else:
            # Window too large, split into tiles
            if self.verbose:
                print(f"      Window {width}x{height} too large, splitting into tiles...")

            tile_size = 10000
            total_valid_count = 0

            for tile_row_start in range(row_min, row_max + 1, tile_size):
                for tile_col_start in range(col_min, col_max + 1, tile_size):
                    tile_row_end = min(tile_row_start + tile_size, row_max + 1)
                    tile_col_end = min(tile_col_start + tile_size, col_max + 1)

                    # Find points in this tile
                    in_tile = (
                        (valid_rows >= tile_row_start) & (valid_rows < tile_row_end) &
                        (valid_cols >= tile_col_start) & (valid_cols < tile_col_end)
                    )

                    if not np.any(in_tile):
                        continue

                    tile_width = tile_col_end - tile_col_start
                    tile_height = tile_row_end - tile_row_start

                    try:
                        window = Window(tile_col_start, tile_row_start, tile_width, tile_height)
                        data = ds.read(1, window=window)

                        tile_local_cols = valid_cols[in_tile] - tile_col_start
                        tile_local_rows = valid_rows[in_tile] - tile_row_start
                        values = data[tile_local_rows, tile_local_cols]

                        if nodata is not None:
                            values = np.where(values == nodata, np.nan, values)

                        elevations[valid_indices[in_tile]] = values
                        total_valid_count += np.sum(~np.isnan(values))

                    except Exception as e:
                        if self.verbose:
                            print(f"        Tile error: {e}")

            elapsed = time.time() - start_time
            if self.verbose:
                print(f"      Zone {zone} complete: {total_valid_count:,} valid values in {elapsed:.1f}s")

        return elevations

    def sample_grid_utm(self, x_min: float, x_max: float, y_min: float, y_max: float,
                        resolution_m: float, utm_zone: int) -> Dict:
        """
        Sample a regular grid in UTM coordinates.

        This is efficient for mesh generation since we're already in UTM.

        Args:
            x_min, x_max: UTM X (easting) bounds
            y_min, y_max: UTM Y (northing) bounds
            resolution_m: Grid spacing in meters
            utm_zone: UTM zone (10 or 11)

        Returns:
            Dict with keys: 'elevation', 'x', 'y', 'X', 'Y'
        """
        from rasterio.windows import Window

        if self.verbose:
            print(f"  Sampling grid in UTM Zone {utm_zone}...")
            print(f"    Bounds: X [{x_min:.0f}, {x_max:.0f}], Y [{y_min:.0f}, {y_max:.0f}]")
            print(f"    Resolution: {resolution_m}m")
            sys.stdout.flush()

        ds = self._get_dataset(utm_zone)
        nodata = ds.nodata

        # Create grid
        x_coords = np.arange(x_min, x_max + resolution_m, resolution_m)
        y_coords = np.arange(y_min, y_max + resolution_m, resolution_m)
        X, Y = np.meshgrid(x_coords, y_coords)

        n_cols, n_rows = len(x_coords), len(y_coords)
        total_points = n_cols * n_rows

        if self.verbose:
            print(f"    Grid size: {n_cols} x {n_rows} = {total_points:,} points")
            sys.stdout.flush()

        # Convert grid bounds to pixel coordinates
        inv_transform = ~ds.transform
        col_min_f, row_min_f = inv_transform * (x_min, y_max)  # Note: y_max for top
        col_max_f, row_max_f = inv_transform * (x_max, y_min)  # Note: y_min for bottom

        col_min = max(0, int(col_min_f))
        col_max = min(ds.width - 1, int(col_max_f))
        row_min = max(0, int(row_min_f))
        row_max = min(ds.height - 1, int(row_max_f))

        width = col_max - col_min + 1
        height = row_max - row_min + 1

        if self.verbose:
            print(f"    Reading window: {width} x {height} pixels...")
            sys.stdout.flush()

        # Read the data
        try:
            window = Window(col_min, row_min, width, height)
            data = ds.read(1, window=window)

            if data is None:
                raise RuntimeError("Failed to read raster data")

            # Resample to our grid resolution using interpolation
            from scipy.interpolate import RegularGridInterpolator

            # Create coordinate arrays for the raw data
            # Transform pixel centers to UTM
            raw_cols = np.arange(col_min, col_max + 1)
            raw_rows = np.arange(row_min, row_max + 1)
            raw_x = ds.transform[0] + (raw_cols + 0.5) * ds.transform[1]
            raw_y = ds.transform[3] + (raw_rows + 0.5) * ds.transform[5]

            # Handle nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data).astype(np.float32)

            # Interpolate
            interp = RegularGridInterpolator(
                (raw_y[::-1], raw_x),  # Flip y because row order
                data[::-1],  # Flip data to match
                method='linear',
                bounds_error=False,
                fill_value=np.nan
            )

            # Sample at our grid points
            points = np.column_stack([Y.ravel(), X.ravel()])
            elevation = interp(points).reshape(Y.shape)

            if self.verbose:
                valid = ~np.isnan(elevation)
                print(f"    Valid points: {np.sum(valid):,} ({100*np.sum(valid)/elevation.size:.1f}%)")

        except Exception as e:
            if self.verbose:
                print(f"    Error reading data: {e}")
            elevation = np.full(X.shape, np.nan, dtype=np.float32)

        return {
            'elevation': elevation,
            'x': x_coords,
            'y': y_coords,
            'X': X,
            'Y': Y,
            'resolution_m': resolution_m,
            'utm_zone': utm_zone,
        }

    @property
    def bounds(self) -> Dict[str, float]:
        """Get approximate coverage bounds in lon/lat."""
        # Approximate bounds for Coastal California
        return {
            'lon_min': -124.5,
            'lon_max': -117.0,
            'lat_min': 32.5,
            'lat_max': 42.0,
        }

    def find_tiles(self, lon_range: Tuple[float, float],
                   lat_range: Tuple[float, float]) -> List:
        """
        Check if we have coverage for a region (compatibility with USACELidar interface).

        Returns a non-empty list if the region is within coverage.
        """
        b = self.bounds
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range

        # Check overlap
        if (lon_max < b['lon_min'] or lon_min > b['lon_max'] or
            lat_max < b['lat_min'] or lat_min > b['lat_max']):
            return []

        # Return a placeholder to indicate coverage exists
        return [{'coverage': 'ok'}]

    def close(self) -> None:
        """Close rasterio datasets."""
        if self._ds_zone10 is not None:
            self._ds_zone10.close()
            self._ds_zone10 = None
        if self._ds_zone11 is not None:
            self._ds_zone11.close()
            self._ds_zone11 = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return (
            f"NOAATopobathy(\n"
            f"  dataset: 2009-2011 Topobathy DEM (Coastal California)\n"
            f"  data_dir: {self.data_dir}\n"
            f"  resolution: ~1m\n"
            f")"
        )


if __name__ == "__main__":
    # Test the handler
    print("Testing NOAA Topobathy DEM handler...\n")

    bathy = NOAATopobathy()
    print(f"\n{bathy}\n")

    # Test single point sampling (Huntington Beach)
    lon, lat = -117.9999, 33.6556
    print(f"Testing point: {lon}, {lat}")
    elev = bathy.sample_point(lon, lat)
    print(f"  Elevation: {elev}m" if elev is not None else "  No data")

    # Test a point in Zone 10 (Santa Cruz)
    lon2, lat2 = -122.0, 36.9
    print(f"\nTesting point: {lon2}, {lat2}")
    elev2 = bathy.sample_point(lon2, lat2)
    print(f"  Elevation: {elev2}m" if elev2 is not None else "  No data")

    bathy.close()
    print("\nDone!")
