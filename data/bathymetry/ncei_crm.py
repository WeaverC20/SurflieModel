#!/usr/bin/env python3
"""
NCEI Coastal Relief Model (CRM) Bathymetry Handler

NOAA NCEI Coastal Relief Model provides bathymetry data for US coastal waters.
This handler supports NetCDF format CRM tiles.

Data source: https://www.ncei.noaa.gov/products/coastal-relief-model
Resolution: 3 arc-second (~90m) or 1/3 arc-second (~10m) for some areas

Download tiles from:
https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ngdc.mgg.dem:348/html

For Southern California, download the "Southern California Coastal Relief Model" tile.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import xarray as xr


class NCECRM:
    """
    Handler for NCEI Coastal Relief Model bathymetry data.

    NCEI CRM uses elevation convention (same as GEBCO):
        - Negative values = below sea level (ocean depth)
        - Positive values = above sea level (land)

    Attributes:
        filepath: Path to the NCEI CRM NetCDF file
        elevation: Raw elevation data (negative = ocean)
        lats: Latitude coordinates
        lons: Longitude coordinates
    """

    # Default CRM file location relative to project root
    DEFAULT_CRM_PATH = Path("data/raw/bathymetry/ncei_crm/crm_california.nc")

    def __init__(self, filepath: Optional[Path] = None):
        """
        Initialize NCEI CRM bathymetry handler.

        Args:
            filepath: Path to NCEI CRM NetCDF file. If None, uses default.
        """
        if filepath is None:
            # Find project root
            project_root = Path(__file__).parent.parent.parent
            filepath = project_root / self.DEFAULT_CRM_PATH

        self.filepath = Path(filepath)

        if not self.filepath.exists():
            raise FileNotFoundError(
                f"NCEI CRM file not found: {self.filepath}\n"
                f"Download from: https://www.ncei.noaa.gov/products/coastal-relief-model"
            )

        # Load data
        self._ds = None
        self._elevation = None
        self._lats = None
        self._lons = None
        self._load_data()

    def _load_data(self) -> None:
        """Load NCEI CRM data from NetCDF file."""
        print(f"Loading NCEI CRM data from: {self.filepath}")

        self._ds = xr.open_dataset(self.filepath)

        # NCEI CRM files can have different variable names
        # Common names: 'Band1', 'z', 'elevation', 'topo'
        elev_var = None
        for name in ['Band1', 'z', 'elevation', 'topo', '__xarray_dataarray_variable__']:
            if name in self._ds:
                elev_var = name
                break

        if elev_var is None:
            # Try to find the main data variable
            data_vars = list(self._ds.data_vars)
            if len(data_vars) == 1:
                elev_var = data_vars[0]
            else:
                raise ValueError(f"Could not find elevation variable. Available: {data_vars}")

        self._elevation = self._ds[elev_var].values

        # Handle coordinate names (can be 'lat'/'lon', 'y'/'x', etc.)
        if 'lat' in self._ds.coords:
            self._lats = self._ds['lat'].values
            self._lons = self._ds['lon'].values
        elif 'y' in self._ds.coords:
            self._lats = self._ds['y'].values
            self._lons = self._ds['x'].values
        else:
            coord_names = list(self._ds.coords)
            raise ValueError(f"Could not find lat/lon coordinates. Available: {coord_names}")

        # Ensure lats/lons are sorted ascending
        if self._lats[0] > self._lats[-1]:
            self._lats = self._lats[::-1]
            self._elevation = self._elevation[::-1, :]
        if self._lons[0] > self._lons[-1]:
            self._lons = self._lons[::-1]
            self._elevation = self._elevation[:, ::-1]

        print(f"  Shape: {self._elevation.shape}")
        print(f"  Lat range: {self._lats.min():.4f} to {self._lats.max():.4f}")
        print(f"  Lon range: {self._lons.min():.4f} to {self._lons.max():.4f}")

        # Calculate resolution
        lat_res = abs(self._lats[1] - self._lats[0])
        lon_res = abs(self._lons[1] - self._lons[0])
        center_lat = (self._lats.min() + self._lats.max()) / 2
        res_m = lat_res * 111000 * np.cos(np.radians(center_lat))
        print(f"  Resolution: {lat_res:.6f}° (~{res_m:.0f}m)")

        # Stats
        valid = ~np.isnan(self._elevation)
        if np.any(valid):
            ocean = self._elevation[valid] < 0
            if np.any(ocean):
                depths = -self._elevation[valid][ocean]
                print(f"  Depth range: {depths.min():.1f}m to {depths.max():.1f}m")

    @property
    def bounds(self) -> Dict:
        """Get geographic bounds of the data."""
        return {
            'lat_min': float(self._lats.min()),
            'lat_max': float(self._lats.max()),
            'lon_min': float(self._lons.min()),
            'lon_max': float(self._lons.max()),
        }

    @property
    def resolution_deg(self) -> float:
        """Get approximate resolution in degrees."""
        return float(np.abs(self._lats[1] - self._lats[0]))

    @property
    def resolution_m(self) -> float:
        """Get approximate resolution in meters (at center latitude)."""
        center_lat = (self._lats.min() + self._lats.max()) / 2
        return self.resolution_deg * 111000 * np.cos(np.radians(center_lat))

    def sample_point(self, lon: float, lat: float) -> Optional[float]:
        """
        Sample elevation at a single point using nearest neighbor.

        Args:
            lon: Longitude
            lat: Latitude

        Returns:
            Elevation in meters (negative = below sea level), or None if outside bounds
        """
        bounds = self.bounds
        if (lon < bounds['lon_min'] or lon > bounds['lon_max'] or
            lat < bounds['lat_min'] or lat > bounds['lat_max']):
            return None

        # Find nearest indices
        lat_idx = np.argmin(np.abs(self._lats - lat))
        lon_idx = np.argmin(np.abs(self._lons - lon))

        value = self._elevation[lat_idx, lon_idx]

        if np.isnan(value):
            return None

        return float(value)

    def sample_points(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        """
        Sample elevation at multiple points using bilinear interpolation.

        Args:
            lons: Array of longitudes
            lats: Array of latitudes

        Returns:
            Array of elevations (NaN where no data or outside bounds)
        """
        lons = np.atleast_1d(lons)
        lats = np.atleast_1d(lats)

        if lons.shape != lats.shape:
            raise ValueError("lons and lats must have same shape")

        original_shape = lons.shape
        lons_flat = lons.ravel()
        lats_flat = lats.ravel()
        n_points = len(lons_flat)

        elevations = np.full(n_points, np.nan, dtype=np.float32)

        bounds = self.bounds

        # Filter points within bounds
        in_bounds = ((lons_flat >= bounds['lon_min']) & (lons_flat <= bounds['lon_max']) &
                     (lats_flat >= bounds['lat_min']) & (lats_flat <= bounds['lat_max']))

        if not np.any(in_bounds):
            return elevations.reshape(original_shape)

        # Get indices for in-bounds points
        valid_lons = lons_flat[in_bounds]
        valid_lats = lats_flat[in_bounds]

        # Calculate grid indices (for bilinear interpolation)
        lat_spacing = self._lats[1] - self._lats[0]
        lon_spacing = self._lons[1] - self._lons[0]

        lat_idx_float = (valid_lats - self._lats[0]) / lat_spacing
        lon_idx_float = (valid_lons - self._lons[0]) / lon_spacing

        lat_idx_0 = np.floor(lat_idx_float).astype(int)
        lon_idx_0 = np.floor(lon_idx_float).astype(int)

        # Clamp to valid range
        lat_idx_0 = np.clip(lat_idx_0, 0, len(self._lats) - 2)
        lon_idx_0 = np.clip(lon_idx_0, 0, len(self._lons) - 2)

        lat_idx_1 = lat_idx_0 + 1
        lon_idx_1 = lon_idx_0 + 1

        # Fractional position within cell
        lat_frac = lat_idx_float - lat_idx_0
        lon_frac = lon_idx_float - lon_idx_0

        # Bilinear interpolation
        # Get four corner values
        v00 = self._elevation[lat_idx_0, lon_idx_0]
        v01 = self._elevation[lat_idx_0, lon_idx_1]
        v10 = self._elevation[lat_idx_1, lon_idx_0]
        v11 = self._elevation[lat_idx_1, lon_idx_1]

        # Interpolate
        v0 = v00 * (1 - lon_frac) + v01 * lon_frac
        v1 = v10 * (1 - lon_frac) + v11 * lon_frac
        values = v0 * (1 - lat_frac) + v1 * lat_frac

        elevations[in_bounds] = values

        return elevations.reshape(original_shape)

    def close(self) -> None:
        """Close the dataset."""
        if self._ds is not None:
            self._ds.close()
            self._ds = None

    def __repr__(self) -> str:
        return (
            f"NCECRM(\n"
            f"  file='{self.filepath.name}',\n"
            f"  shape={self._elevation.shape},\n"
            f"  bounds={self.bounds},\n"
            f"  resolution={self.resolution_deg:.6f}° (~{self.resolution_m:.0f}m)\n"
            f")"
        )

    def __del__(self):
        self.close()