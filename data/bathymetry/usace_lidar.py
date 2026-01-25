#!/usr/bin/env python3
"""
USACE Lidar Bathymetry Handler

Manages the USACE California DEM 2009 dataset - high-resolution (1m) coastal
bathymetry and topography from Lidar surveys.

Dataset: NCMP CA DEM 2009 (ID: 9488)
- 140 GeoTiff tiles covering California coast
- Resolution: ~1m (0.00001 degrees)
- Coverage: Lat 32.53-36.66, Lon -121.96 to -117.12
- CRS: EPSG:5498 (NAD83 / geographic with meters height)
- Convention: Elevation (positive up, negative below sea level)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import rasterio


@dataclass
class TileInfo:
    """Metadata for a single USACE Lidar tile."""
    path: Path
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def contains(self, lon: float, lat: float) -> bool:
        """Check if a point is within this tile's bounds."""
        return (self.lon_min <= lon <= self.lon_max and
                self.lat_min <= lat <= self.lat_max)

    def overlaps(self, lon_range: Tuple[float, float],
                 lat_range: Tuple[float, float]) -> bool:
        """Check if this tile overlaps with a bounding box."""
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range
        return not (self.lon_max < lon_min or self.lon_min > lon_max or
                    self.lat_max < lat_min or self.lat_min > lat_max)


class USACELidar:
    """
    Handler for USACE California Lidar DEM tiles.

    Efficiently manages multiple GeoTiff tiles and provides sampling
    at arbitrary (lon, lat) points.

    Attributes:
        data_dir: Directory containing the GeoTiff tiles
        tiles: List of TileInfo objects with bounds for each tile
    """

    # Default data location relative to project root
    DEFAULT_DATA_DIR = Path("data/raw/bathymetry/USACE_CA_DEM_2009_9488")

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the USACE Lidar handler.

        Args:
            data_dir: Directory containing GeoTiff tiles. If None, uses default.
        """
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent
            data_dir = project_root / self.DEFAULT_DATA_DIR

        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"USACE data directory not found: {self.data_dir}")

        self.tiles: List[TileInfo] = []
        self._tile_cache: Dict[Path, 'rasterio.DatasetReader'] = {}

        self._index_tiles()

    def _index_tiles(self) -> None:
        """Build spatial index of all tiles."""
        import rasterio

        tif_files = sorted(self.data_dir.glob("*.tif"))

        if not tif_files:
            raise FileNotFoundError(f"No .tif files found in {self.data_dir}")

        print(f"Indexing {len(tif_files)} USACE Lidar tiles...")

        for tif_path in tif_files:
            with rasterio.open(tif_path) as src:
                bounds = src.bounds
                self.tiles.append(TileInfo(
                    path=tif_path,
                    lon_min=bounds.left,
                    lon_max=bounds.right,
                    lat_min=bounds.bottom,
                    lat_max=bounds.top,
                ))

        # Calculate overall coverage
        self._lon_min = min(t.lon_min for t in self.tiles)
        self._lon_max = max(t.lon_max for t in self.tiles)
        self._lat_min = min(t.lat_min for t in self.tiles)
        self._lat_max = max(t.lat_max for t in self.tiles)

        print(f"  Coverage: Lon [{self._lon_min:.4f}, {self._lon_max:.4f}]")
        print(f"            Lat [{self._lat_min:.4f}, {self._lat_max:.4f}]")
        print(f"  Resolution: ~1m")

    @property
    def bounds(self) -> Dict[str, float]:
        """Get overall coverage bounds."""
        return {
            'lon_min': self._lon_min,
            'lon_max': self._lon_max,
            'lat_min': self._lat_min,
            'lat_max': self._lat_max,
        }

    def find_tiles(self, lon_range: Tuple[float, float],
                   lat_range: Tuple[float, float]) -> List[TileInfo]:
        """
        Find all tiles that overlap a given bounding box.

        Args:
            lon_range: (lon_min, lon_max)
            lat_range: (lat_min, lat_max)

        Returns:
            List of TileInfo objects that overlap the region
        """
        return [t for t in self.tiles if t.overlaps(lon_range, lat_range)]

    def find_tile_for_point(self, lon: float, lat: float) -> Optional[TileInfo]:
        """Find the tile containing a specific point."""
        for tile in self.tiles:
            if tile.contains(lon, lat):
                return tile
        return None

    def _get_raster(self, tile: TileInfo) -> 'rasterio.DatasetReader':
        """Get rasterio dataset for a tile (cached)."""
        import rasterio

        if tile.path not in self._tile_cache:
            self._tile_cache[tile.path] = rasterio.open(tile.path)
        return self._tile_cache[tile.path]

    def sample_point(self, lon: float, lat: float) -> Optional[float]:
        """
        Sample elevation at a single point.

        Args:
            lon: Longitude (degrees)
            lat: Latitude (degrees)

        Returns:
            Elevation in meters (positive up), or None if no data
        """
        tile = self.find_tile_for_point(lon, lat)
        if tile is None:
            return None

        src = self._get_raster(tile)

        # Convert lon/lat to pixel coordinates
        row, col = src.index(lon, lat)

        # Check bounds
        if row < 0 or row >= src.height or col < 0 or col >= src.width:
            return None

        # Read single pixel value
        value = src.read(1, window=((row, row+1), (col, col+1)))[0, 0]

        # Check for nodata
        if src.nodata is not None and value == src.nodata:
            return None

        return float(value)

    def sample_points(self, lons: np.ndarray, lats: np.ndarray,
                       chunk_size: int = 100000) -> np.ndarray:
        """
        Sample elevation at multiple points efficiently.

        Uses vectorized operations per tile for speed.

        Args:
            lons: Array of longitudes
            lats: Array of latitudes
            chunk_size: Process this many points at a time (memory management)

        Returns:
            Array of elevations (NaN where no data)
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

        # Process tiles that overlap with our query bounds
        query_lon_range = (lons_flat.min(), lons_flat.max())
        query_lat_range = (lats_flat.min(), lats_flat.max())
        relevant_tiles = self.find_tiles(query_lon_range, query_lat_range)

        if not relevant_tiles:
            return elevations.reshape(original_shape)

        print(f"      Sampling from {len(relevant_tiles)} tiles...")

        # Vectorized tile membership check
        for tile_idx, tile in enumerate(relevant_tiles):
            # Vectorized bounds check
            in_tile = ((lons_flat >= tile.lon_min) & (lons_flat <= tile.lon_max) &
                       (lats_flat >= tile.lat_min) & (lats_flat <= tile.lat_max))

            if not np.any(in_tile):
                continue

            indices = np.where(in_tile)[0]
            tile_lons = lons_flat[indices]
            tile_lats = lats_flat[indices]

            src = self._get_raster(tile)

            # Convert all lon/lat to pixel coordinates at once
            # rasterio transform: (col, row) = ~transform * (x, y)
            inv_transform = ~src.transform
            cols, rows = inv_transform * (tile_lons, tile_lats)
            cols = cols.astype(np.int32)
            rows = rows.astype(np.int32)

            # Bounds check
            valid = ((rows >= 0) & (rows < src.height) &
                     (cols >= 0) & (cols < src.width))

            if not np.any(valid):
                continue

            valid_indices = indices[valid]
            valid_rows = rows[valid]
            valid_cols = cols[valid]

            # Read full tile data if we need many points from it
            # (faster than many small reads)
            if len(valid_rows) > 1000:
                # Read entire tile
                data = src.read(1)
                values = data[valid_rows, valid_cols]
            else:
                # Read individual pixels for small counts
                values = np.array([
                    src.read(1, window=((r, r+1), (c, c+1)))[0, 0]
                    for r, c in zip(valid_rows, valid_cols)
                ])

            # Handle nodata
            if src.nodata is not None:
                values = np.where(values == src.nodata, np.nan, values)

            elevations[valid_indices] = values

            if (tile_idx + 1) % 10 == 0:
                print(f"        Processed {tile_idx + 1}/{len(relevant_tiles)} tiles...")

        return elevations.reshape(original_shape)

    def sample_grid(self, lon_range: Tuple[float, float],
                    lat_range: Tuple[float, float],
                    resolution_m: float = 10.0) -> Dict:
        """
        Sample a regular grid over a region.

        Args:
            lon_range: (lon_min, lon_max)
            lat_range: (lat_min, lat_max)
            resolution_m: Grid spacing in meters

        Returns:
            Dict with keys: 'elevation', 'lons', 'lats', 'depth'
        """
        lon_min, lon_max = lon_range
        lat_min, lat_max = lat_range

        # Convert resolution to degrees (approximate)
        center_lat = (lat_min + lat_max) / 2
        meters_per_deg_lat = 111000
        meters_per_deg_lon = 111000 * np.cos(np.radians(center_lat))

        dlon = resolution_m / meters_per_deg_lon
        dlat = resolution_m / meters_per_deg_lat

        # Create grid
        lons = np.arange(lon_min, lon_max + dlon, dlon)
        lats = np.arange(lat_min, lat_max + dlat, dlat)

        LON, LAT = np.meshgrid(lons, lats)

        # Sample
        elevation = self.sample_points(LON, LAT)

        # Convert to depth (positive = below sea level)
        depth = -elevation
        depth[depth <= 0] = np.nan  # Mask land

        return {
            'elevation': elevation,
            'depth': depth,
            'lons': lons,
            'lats': lats,
            'LON': LON,
            'LAT': LAT,
            'resolution_m': resolution_m,
        }

    def close(self) -> None:
        """Close all cached rasterio datasets."""
        for src in self._tile_cache.values():
            src.close()
        self._tile_cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __repr__(self) -> str:
        return (
            f"USACELidar(\n"
            f"  tiles={len(self.tiles)},\n"
            f"  coverage: Lon [{self._lon_min:.4f}, {self._lon_max:.4f}],\n"
            f"            Lat [{self._lat_min:.4f}, {self._lat_max:.4f}]\n"
            f")"
        )


if __name__ == "__main__":
    # Example usage
    lidar = USACELidar()
    print(lidar)

    # Test sampling at Huntington Beach Pier
    lon, lat = -117.9999, 33.6556
    elev = lidar.sample_point(lon, lat)
    print(f"\nHuntington Beach Pier ({lon}, {lat}):")
    print(f"  Elevation: {elev}m" if elev is not None else "  No data")

    lidar.close()
