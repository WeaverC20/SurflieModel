#!/usr/bin/env python3
"""
Surf Zone Mesh for Ray Tracing

High-resolution nearshore bathymetry mesh for wave ray tracing.
Uses contour extraction for coastline detection and offset curves
for variable-density grid generation.

Designed for use with Numba-accelerated ray tracing algorithms.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List, TYPE_CHECKING
import json
import numpy as np
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

if TYPE_CHECKING:
    from data.regions.region import Region
    from data.bathymetry.usace_lidar import USACELidar
    from data.bathymetry.ncei_crm import NCECRM
    from data.bathymetry.noaa_topobathy import NOAATopobathy
    from typing import Union
    BathymetrySource = Union[USACELidar, NOAATopobathy]


@dataclass
class SurfZoneMeshConfig:
    """Configuration for surf zone mesh generation."""

    # Spatial extent from coastline
    offshore_distance_m: float = 2500.0   # How far offshore (meters)
    onshore_distance_m: float = 50.0      # How far onshore for tidal buffer (meters)

    # Resolution settings
    min_resolution_m: float = 20.0        # Finest resolution at coastline
    max_resolution_m: float = 300.0       # Coarsest resolution at max offshore distance

    # Coastline detection
    coastline_sample_res_m: float = 50.0  # Resolution for initial elevation sampling

    # Land filtering
    max_land_elevation_m: float = 5.0     # Exclude land points above this elevation

    # Point density bias (higher = more points near coastline)
    # 1.0 = linear, 2.0 = quadratic, 3.0 = cubic bias towards coast
    coastline_density_bias: float = 2.5

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'offshore_distance_m': self.offshore_distance_m,
            'onshore_distance_m': self.onshore_distance_m,
            'min_resolution_m': self.min_resolution_m,
            'max_resolution_m': self.max_resolution_m,
            'coastline_sample_res_m': self.coastline_sample_res_m,
            'max_land_elevation_m': self.max_land_elevation_m,
            'coastline_density_bias': self.coastline_density_bias,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'SurfZoneMeshConfig':
        """Create from dictionary."""
        return cls(**d)


class SurfZoneMesh:
    """
    Coastline-following surf zone mesh for ray tracing.

    Uses:
    1. Contour extraction to find exact coastline (elevation = 0)
    2. Offset curves parallel to coastline with log-spaced distances
    3. Points sampled along each offset curve

    Coordinate System:
    - All coordinates in UTM (meters)
    - Elevation convention: positive = above sea level, negative = below
    """

    def __init__(
        self,
        region_name: str,
        utm_zone: int,
        utm_hemisphere: str = 'N',
        config: Optional[SurfZoneMeshConfig] = None,
    ):
        self.region_name = region_name
        self.utm_zone = utm_zone
        self.utm_hemisphere = utm_hemisphere
        self.config = config or SurfZoneMeshConfig()

        # Coastline as ordered polylines
        self.coastlines: Optional[List[np.ndarray]] = None  # List of (N, 2) arrays

        # All mesh points (irregular point cloud)
        self.points_x: Optional[np.ndarray] = None
        self.points_y: Optional[np.ndarray] = None
        self.elevation: Optional[np.ndarray] = None

        # Reference bounds
        self.lon_range: Optional[Tuple[float, float]] = None
        self.lat_range: Optional[Tuple[float, float]] = None

        # Interpolation (lazy loaded)
        self._triangulation: Optional[Delaunay] = None
        self._interpolator: Optional[LinearNDInterpolator] = None

        # Transformers (lazy loaded)
        self._transformer_to_utm = None
        self._transformer_from_utm = None

    def _init_transformers(self) -> None:
        """Initialize coordinate transformers."""
        from pyproj import Transformer
        utm_crs = f"+proj=utm +zone={self.utm_zone} +{'north' if self.utm_hemisphere == 'N' else 'south'} +datum=WGS84"
        self._transformer_to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        self._transformer_from_utm = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    def lon_lat_to_utm(self, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert lon/lat to UTM coordinates."""
        if self._transformer_to_utm is None:
            self._init_transformers()
        return self._transformer_to_utm.transform(lon, lat)

    def utm_to_lon_lat(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert UTM coordinates to lon/lat."""
        if self._transformer_from_utm is None:
            self._init_transformers()
        return self._transformer_from_utm.transform(x, y)

    @property
    def depth(self) -> Optional[np.ndarray]:
        """Water depth (positive = below sea level)."""
        if self.elevation is None:
            return None
        return -self.elevation

    @classmethod
    def from_region(
        cls,
        region: 'Region',
        bathymetry: 'BathymetrySource',
        config: Optional[SurfZoneMeshConfig] = None,
        utm_zone: Optional[int] = None,
        fallback_bathy: Optional['NCECRM'] = None,
    ) -> 'SurfZoneMesh':
        """
        Create a surf zone mesh for a region.

        Args:
            region: Region definition
            bathymetry: Primary bathymetry data source (USACELidar or NOAATopobathy)
                       Must implement find_tiles() and sample_points()
            config: Mesh configuration
            utm_zone: UTM zone (auto-detected if None)
            fallback_bathy: Optional fallback bathymetry source (e.g., NCEI CRM)
                           Used for points without primary coverage

        Algorithm:
        1. Sample elevation on regular grid
        2. Extract coastline using contour at elevation = 0
        3. Generate offset curves at log-spaced distances
        4. Sample points along each offset curve
        5. Sample elevation at all points (primary source, fallback for gaps)
        """
        config = config or SurfZoneMeshConfig()

        # Auto-detect UTM zone
        if utm_zone is None:
            center_lon = (region.lon_range[0] + region.lon_range[1]) / 2
            utm_zone = int((center_lon + 180) / 6) + 1

        center_lat = (region.lat_range[0] + region.lat_range[1]) / 2
        utm_hemisphere = 'N' if center_lat >= 0 else 'S'

        mesh = cls(
            region_name=region.name,
            utm_zone=utm_zone,
            utm_hemisphere=utm_hemisphere,
            config=config,
        )
        mesh.lon_range = region.lon_range
        mesh.lat_range = region.lat_range

        mesh._build_mesh(region, bathymetry, fallback_bathy)
        return mesh

    def _build_mesh(
        self,
        region: 'Region',
        bathymetry: 'BathymetrySource',
        fallback_bathy: Optional['NCECRM'] = None,
    ) -> None:
        """Build the mesh using contour extraction and offset curves."""
        cfg = self.config
        print(f"Building surf zone mesh for {region.display_name}...")
        print(f"  Min resolution: {cfg.min_resolution_m}m (at coast)")
        print(f"  Max resolution: {cfg.max_resolution_m}m (at {cfg.offshore_distance_m}m offshore)")
        print(f"  Max land elevation: {cfg.max_land_elevation_m}m")
        print(f"  Coastline density bias: {cfg.coastline_density_bias}")
        if fallback_bathy:
            print(f"  Fallback bathymetry: {getattr(fallback_bathy, 'filepath', fallback_bathy)}")

        # Step 1: Check coverage
        tiles = bathymetry.find_tiles(region.lon_range, region.lat_range)
        if not tiles:
            raise ValueError(f"No bathymetry data found for region {region.name}")

        # Use region bounds (bathymetry coverage may be larger)
        bathy_lon_min = region.lon_range[0]
        bathy_lon_max = region.lon_range[1]
        bathy_lat_min = region.lat_range[0]
        bathy_lat_max = region.lat_range[1]

        print(f"\n  Step 1: Sampling elevation grid...")
        print(f"    Region bounds: Lon [{bathy_lon_min:.4f}, {bathy_lon_max:.4f}]")
        print(f"                   Lat [{bathy_lat_min:.4f}, {bathy_lat_max:.4f}]")

        # Convert to UTM
        corners_x, corners_y = self.lon_lat_to_utm(
            np.array([bathy_lon_min, bathy_lon_max, bathy_lon_max, bathy_lon_min]),
            np.array([bathy_lat_min, bathy_lat_min, bathy_lat_max, bathy_lat_max])
        )
        x_min, x_max = corners_x.min(), corners_x.max()
        y_min, y_max = corners_y.min(), corners_y.max()

        # Sample elevation on regular grid for contour extraction
        sample_x = np.arange(x_min, x_max, cfg.coastline_sample_res_m)
        sample_y = np.arange(y_min, y_max, cfg.coastline_sample_res_m)
        X, Y = np.meshgrid(sample_x, sample_y)

        print(f"    Grid size: {len(sample_x)} x {len(sample_y)} = {X.size:,} points")
        print(f"    (This step fetches elevation data remotely - may take several minutes)")

        lon_grid, lat_grid = self.utm_to_lon_lat(X, Y)
        print(f"    Fetching elevation data...")
        elev_grid = bathymetry.sample_points(lon_grid, lat_grid)

        valid_count = np.sum(~np.isnan(elev_grid))
        print(f"    Valid elevation samples: {valid_count:,} ({100*valid_count/elev_grid.size:.1f}%)")

        # Step 2: Extract coastline contours
        print(f"\n  Step 2: Extracting coastline contours...")
        coastlines = self._extract_coastline_contours(
            sample_x, sample_y, elev_grid
        )
        self.coastlines = coastlines

        total_coastline_length = sum(self._polyline_length(c) for c in coastlines)
        print(f"    Found {len(coastlines)} coastline segments")
        print(f"    Total coastline length: {total_coastline_length/1000:.1f} km")

        # Step 3: Generate offset distances (biased towards coastline)
        print(f"\n  Step 3: Generating offset distances...")
        offshore_distances = self._generate_offset_distances(
            cfg.min_resolution_m, cfg.max_resolution_m, cfg.offshore_distance_m,
            density_bias=cfg.coastline_density_bias
        )
        # Onshore uses lower bias (more uniform) since it's a small area
        onshore_distances = self._generate_offset_distances(
            cfg.min_resolution_m, cfg.min_resolution_m * 2, cfg.onshore_distance_m,
            density_bias=1.5
        )

        print(f"    Offshore: {len(offshore_distances)} layers from 0 to {offshore_distances[-1]:.0f}m")
        print(f"    Onshore: {len(onshore_distances)} layers from 0 to {onshore_distances[-1]:.0f}m")

        # Step 4: Generate points along offset curves
        print(f"\n  Step 4: Generating mesh points along offset curves...")
        all_x, all_y = self._generate_offset_curve_points(
            coastlines, offshore_distances, onshore_distances,
            cfg.min_resolution_m, cfg.max_resolution_m, cfg.offshore_distance_m
        )

        print(f"    Total mesh points: {len(all_x):,}")

        # Step 5: Sample elevation at mesh points
        print(f"\n  Step 5: Sampling elevation at mesh points...")
        print(f"    (This step fetches elevation data remotely - may take several minutes)")
        all_lon, all_lat = self.utm_to_lon_lat(all_x, all_y)

        # Primary source: bathymetry
        print(f"    Fetching elevation data for {len(all_x):,} mesh points...")
        all_elev = bathymetry.sample_points(all_lon, all_lat)
        primary_valid = ~np.isnan(all_elev)
        n_primary = np.sum(primary_valid)
        print(f"    From primary source: {n_primary:,} points ({100*n_primary/len(all_x):.1f}%)")

        # Fallback source: NCEI CRM (if provided)
        if fallback_bathy is not None:
            missing = np.isnan(all_elev)
            n_missing = np.sum(missing)
            if n_missing > 0:
                print(f"    Filling {n_missing:,} missing points from fallback...")
                fallback_elev = fallback_bathy.sample_points(
                    all_lon[missing], all_lat[missing]
                )
                all_elev[missing] = fallback_elev
                n_fallback = np.sum(~np.isnan(fallback_elev))
                print(f"    From fallback: {n_fallback:,} points ({100*n_fallback/n_missing:.1f}% of missing)")

        # Remove points with no elevation data
        valid = ~np.isnan(all_elev)

        # Filter out land points above max elevation threshold
        max_land_elev = cfg.max_land_elevation_m
        too_high = all_elev > max_land_elev
        n_too_high = np.sum(valid & too_high)
        if n_too_high > 0:
            print(f"    Excluding {n_too_high:,} points above {max_land_elev}m elevation")
            valid = valid & ~too_high

        self.points_x = all_x[valid]
        self.points_y = all_y[valid]
        self.elevation = all_elev[valid]

        print(f"    Total valid: {len(self.points_x):,} ({100*len(self.points_x)/len(all_x):.1f}%)")

        # Step 6: Build triangulation
        print(f"\n  Step 6: Building triangulation...")
        self._build_interpolator()

        # Stats
        ocean = self.elevation < 0
        land = self.elevation >= 0

        print(f"\n  Results:")
        print(f"    Total points: {len(self.points_x):,}")
        print(f"    Ocean points: {np.sum(ocean):,}")
        print(f"    Land points:  {np.sum(land):,}")

        if np.any(ocean):
            depths = -self.elevation[ocean]
            print(f"    Depth range: {depths.min():.1f}m to {depths.max():.1f}m")
        if np.any(land):
            heights = self.elevation[land]
            print(f"    Land height: {heights.min():.1f}m to {heights.max():.1f}m")

    def _extract_coastline_contours(
        self,
        x: np.ndarray,
        y: np.ndarray,
        elevation: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Extract coastline as contours at elevation = 0.

        Uses matplotlib's contour algorithm (marching squares).
        Returns list of (N, 2) arrays, each representing a coastline segment.
        """
        import matplotlib.pyplot as plt

        # Replace NaN with a value that won't create false contours
        elev_filled = np.where(np.isnan(elevation), 1000, elevation)

        # Use matplotlib's contour to extract the 0-level contour
        fig, ax = plt.subplots()
        cs = ax.contour(x, y, elev_filled, levels=[0])
        plt.close(fig)

        # Extract contour paths (compatible with newer matplotlib versions)
        coastlines = []
        # allsegs[level_index] contains list of (N, 2) arrays for that level
        if hasattr(cs, 'allsegs') and len(cs.allsegs) > 0:
            for segment in cs.allsegs[0]:  # Level 0 (our only level)
                if len(segment) >= 2:
                    coastlines.append(segment.copy())
        else:
            # Fallback for older matplotlib
            for collection in cs.collections:
                for path in collection.get_paths():
                    vertices = path.vertices
                    if len(vertices) >= 2:
                        coastlines.append(vertices.copy())

        return coastlines

    def _polyline_length(self, points: np.ndarray) -> float:
        """Calculate total length of a polyline."""
        if len(points) < 2:
            return 0.0
        diffs = np.diff(points, axis=0)
        lengths = np.sqrt(np.sum(diffs**2, axis=1))
        return np.sum(lengths)

    def _generate_offset_distances(
        self,
        min_spacing: float,
        max_spacing: float,
        max_distance: float,
        density_bias: float = 2.5,
    ) -> np.ndarray:
        """
        Generate offset distances with bias towards coastline.

        Uses a power function to concentrate more offset layers near the coast.
        Higher density_bias = more layers near coastline, fewer far offshore.

        Args:
            min_spacing: Minimum spacing between offsets (at coast)
            max_spacing: Maximum spacing between offsets (at max distance)
            max_distance: Maximum offset distance
            density_bias: Power factor for density bias (1.0=linear, 2.0=quadratic, etc.)
        """
        # Estimate number of layers based on average spacing
        avg_spacing = (min_spacing + max_spacing) / 2
        n_layers = max(5, int(max_distance / avg_spacing))

        # Generate normalized positions [0, 1] with uniform spacing
        t = np.linspace(0, 1, n_layers)

        # Apply power function to concentrate values near 0 (coastline)
        # t_biased will have more values near 0, fewer near 1
        t_biased = t ** density_bias

        # Scale to actual distances
        distances = t_biased * max_distance

        # Ensure 0 is included
        if distances[0] != 0:
            distances = np.concatenate([[0], distances])

        return distances

    def _generate_offset_curve_points(
        self,
        coastlines: List[np.ndarray],
        offshore_distances: np.ndarray,
        onshore_distances: np.ndarray,
        min_spacing: float,
        max_spacing: float,
        max_offshore: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate points along offset curves parallel to coastline.

        For each coastline segment:
        1. Compute normals at each vertex
        2. Offset the curve by each distance
        3. Sample points along the offset curve with spacing proportional to distance
        """
        all_x = []
        all_y = []

        for coast_idx, coastline in enumerate(coastlines):
            if len(coastline) < 2:
                continue

            # Compute normals at each point (perpendicular to local tangent)
            normals = self._compute_polyline_normals(coastline)

            # Generate points for offshore distances
            for dist in offshore_distances:
                # Spacing along curve proportional to distance from shore
                if max_offshore > 0:
                    t = dist / max_offshore  # 0 at coast, 1 at max offshore
                else:
                    t = 0
                along_spacing = min_spacing + t * (max_spacing - min_spacing)

                # Offset the coastline
                offset_points = coastline + normals * dist

                # Sample points along the offset curve
                sampled = self._sample_polyline(offset_points, along_spacing)
                all_x.extend(sampled[:, 0])
                all_y.extend(sampled[:, 1])

            # Generate points for onshore distances (negative offset)
            for dist in onshore_distances[1:]:  # Skip 0 (already done with offshore)
                along_spacing = min_spacing  # Keep dense onshore

                # Offset the coastline in opposite direction
                offset_points = coastline - normals * dist

                sampled = self._sample_polyline(offset_points, along_spacing)
                all_x.extend(sampled[:, 0])
                all_y.extend(sampled[:, 1])

        return np.array(all_x), np.array(all_y)

    def _compute_polyline_normals(self, points: np.ndarray) -> np.ndarray:
        """
        Compute outward-pointing normals at each polyline vertex.

        Normal is perpendicular to local tangent, pointing "left" of the
        curve direction. We assume coastline is traced with ocean on the left.
        """
        n = len(points)
        normals = np.zeros_like(points)

        for i in range(n):
            # Compute tangent using neighbors
            if i == 0:
                tangent = points[1] - points[0]
            elif i == n - 1:
                tangent = points[-1] - points[-2]
            else:
                tangent = points[i + 1] - points[i - 1]

            # Normalize tangent
            length = np.sqrt(tangent[0]**2 + tangent[1]**2)
            if length > 0:
                tangent = tangent / length

            # Normal is perpendicular (rotate 90 degrees left)
            normals[i] = np.array([-tangent[1], tangent[0]])

        return normals

    def _sample_polyline(self, points: np.ndarray, spacing: float) -> np.ndarray:
        """
        Sample points along a polyline at approximately uniform spacing.
        """
        if len(points) < 2:
            return points

        # Compute cumulative distance along polyline
        diffs = np.diff(points, axis=0)
        segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        cumulative_dist = np.zeros(len(points))
        cumulative_dist[1:] = np.cumsum(segment_lengths)

        total_length = cumulative_dist[-1]
        if total_length < spacing:
            return points

        # Generate target distances
        n_samples = max(2, int(total_length / spacing) + 1)
        target_dists = np.linspace(0, total_length, n_samples)

        # Interpolate x and y
        sampled_x = np.interp(target_dists, cumulative_dist, points[:, 0])
        sampled_y = np.interp(target_dists, cumulative_dist, points[:, 1])

        return np.column_stack([sampled_x, sampled_y])

    def _build_interpolator(self) -> None:
        """Build Delaunay triangulation and interpolator."""
        if self.points_x is None or len(self.points_x) < 3:
            return

        points = np.column_stack([self.points_x, self.points_y])
        self._triangulation = Delaunay(points)
        self._interpolator = LinearNDInterpolator(
            self._triangulation, self.elevation, fill_value=np.nan
        )

    # =========================================================================
    # Query methods
    # =========================================================================

    def get_elevation_at_point(self, x: float, y: float) -> float:
        """Get elevation at UTM coordinate using triangulated interpolation."""
        if self._interpolator is None:
            self._build_interpolator()
        if self._interpolator is None:
            return np.nan
        return float(self._interpolator(x, y))

    def get_depth_at_point(self, x: float, y: float) -> float:
        """Get depth at UTM coordinate (positive = below sea level)."""
        elev = self.get_elevation_at_point(x, y)
        return -elev if not np.isnan(elev) else np.nan

    def get_elevation_at_points(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Get elevation at multiple UTM coordinates."""
        if self._interpolator is None:
            self._build_interpolator()
        if self._interpolator is None:
            return np.full(len(x), np.nan)
        points = np.column_stack([x, y])
        return self._interpolator(points)

    def get_depth_gradient(self, x: float, y: float, h: float = 10.0) -> Tuple[float, float]:
        """Get depth gradient (∂d/∂x, ∂d/∂y) at a point using finite differences."""
        d_xp = self.get_depth_at_point(x + h, y)
        d_xm = self.get_depth_at_point(x - h, y)
        d_yp = self.get_depth_at_point(x, y + h)
        d_ym = self.get_depth_at_point(x, y - h)

        dz_dx = (d_xp - d_xm) / (2 * h) if not (np.isnan(d_xp) or np.isnan(d_xm)) else 0.0
        dz_dy = (d_yp - d_ym) / (2 * h) if not (np.isnan(d_yp) or np.isnan(d_ym)) else 0.0

        return dz_dx, dz_dy

    # =========================================================================
    # Numba-Compatible Data Export
    # =========================================================================

    def get_numba_arrays(self) -> Dict[str, np.ndarray]:
        """Get arrays formatted for Numba ray tracing."""
        if self._triangulation is None:
            self._build_interpolator()

        return {
            'points_x': np.ascontiguousarray(self.points_x, dtype=np.float64),
            'points_y': np.ascontiguousarray(self.points_y, dtype=np.float64),
            'elevation': np.ascontiguousarray(self.elevation, dtype=np.float64),
            'depth': np.ascontiguousarray(-self.elevation, dtype=np.float64),
            'triangles': np.ascontiguousarray(self._triangulation.simplices, dtype=np.int32),
        }

    # =========================================================================
    # Save / Load
    # =========================================================================

    def save(self, directory: Path) -> Path:
        """Save mesh to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        npz_path = directory / f"{self.region_name}_surfzone.npz"
        save_dict = {
            'points_x': self.points_x,
            'points_y': self.points_y,
            'elevation': self.elevation,
        }

        # Save coastlines as separate arrays
        if self.coastlines:
            for i, coastline in enumerate(self.coastlines):
                save_dict[f'coastline_{i}'] = coastline

        np.savez_compressed(npz_path, **save_dict)
        print(f"Saved: {npz_path}")

        json_path = directory / f"{self.region_name}_surfzone.json"
        metadata = {
            'region_name': self.region_name,
            'utm_zone': self.utm_zone,
            'utm_hemisphere': self.utm_hemisphere,
            'config': self.config.to_dict(),
            'lon_range': list(self.lon_range) if self.lon_range else None,
            'lat_range': list(self.lat_range) if self.lat_range else None,
            'n_points': len(self.points_x) if self.points_x is not None else 0,
            'n_coastline_segments': len(self.coastlines) if self.coastlines else 0,
            'stats': self._compute_stats(),
        }
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved: {json_path}")

        return directory

    def _compute_stats(self) -> Dict:
        """Compute statistics for metadata."""
        if self.elevation is None:
            return {}

        ocean = self.elevation < 0
        land = self.elevation >= 0

        stats = {
            'n_points': int(len(self.elevation)),
            'n_ocean': int(np.sum(ocean)),
            'n_land': int(np.sum(land)),
        }

        if np.any(ocean):
            depths = -self.elevation[ocean]
            stats['depth_min'] = float(depths.min())
            stats['depth_max'] = float(depths.max())

        if np.any(land):
            heights = self.elevation[land]
            stats['land_min'] = float(heights.min())
            stats['land_max'] = float(heights.max())

        return stats

    @classmethod
    def load(cls, directory: Path) -> 'SurfZoneMesh':
        """Load mesh from directory."""
        directory = Path(directory)

        json_files = list(directory.glob("*_surfzone.json"))
        if not json_files:
            raise FileNotFoundError(f"No surfzone metadata found in {directory}")

        json_path = json_files[0]
        region_name = json_path.stem.replace("_surfzone", "")
        npz_path = directory / f"{region_name}_surfzone.npz"

        with open(json_path, 'r') as f:
            metadata = json.load(f)

        config = SurfZoneMeshConfig.from_dict(metadata['config'])
        mesh = cls(
            region_name=metadata['region_name'],
            utm_zone=metadata['utm_zone'],
            utm_hemisphere=metadata['utm_hemisphere'],
            config=config,
        )

        mesh.lon_range = tuple(metadata['lon_range']) if metadata.get('lon_range') else None
        mesh.lat_range = tuple(metadata['lat_range']) if metadata.get('lat_range') else None

        data = np.load(npz_path)
        mesh.points_x = data['points_x']
        mesh.points_y = data['points_y']
        mesh.elevation = data['elevation']

        # Load coastlines
        mesh.coastlines = []
        i = 0
        while f'coastline_{i}' in data:
            mesh.coastlines.append(data[f'coastline_{i}'])
            i += 1

        print(f"Loaded surf zone mesh: {region_name}")
        print(f"  Points: {len(mesh.points_x):,}")

        return mesh

    def summary(self) -> str:
        """Return a summary string."""
        lines = [
            f"SurfZoneMesh: {self.region_name}",
            f"  UTM Zone: {self.utm_zone}{self.utm_hemisphere}",
        ]

        if self.points_x is not None:
            lines.append(f"  Total points: {len(self.points_x):,}")

            if self.coastlines:
                total_len = sum(self._polyline_length(c) for c in self.coastlines)
                lines.append(f"  Coastline segments: {len(self.coastlines)}")
                lines.append(f"  Coastline length: {total_len/1000:.1f} km")

            ocean = self.elevation < 0
            land = self.elevation >= 0

            lines.append(f"  Ocean points: {np.sum(ocean):,}")
            lines.append(f"  Land points:  {np.sum(land):,}")

            if np.any(ocean):
                depths = -self.elevation[ocean]
                lines.append(f"  Depth range: {depths.min():.1f}m to {depths.max():.1f}m")
            if np.any(land):
                heights = self.elevation[land]
                lines.append(f"  Land height: {heights.min():.1f}m to {heights.max():.1f}m")

        return '\n'.join(lines)

    def __repr__(self) -> str:
        n_points = len(self.points_x) if self.points_x is not None else 0
        return f"SurfZoneMesh(region='{self.region_name}', points={n_points})"
