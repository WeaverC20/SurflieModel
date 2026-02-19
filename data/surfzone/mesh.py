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
    min_resolution_m: float = 5.0         # Finest resolution at coastline (meters)
    max_resolution_m: float = 200.0       # Coarsest resolution at max offshore distance
    n_uniform_layers: int = 10            # Number of layers at min_resolution before coarsening

    # Coastline detection
    coastline_sample_res_m: float = 50.0  # Resolution for initial elevation sampling

    # Land filtering
    max_land_elevation_m: float = 5.0     # Exclude land points above this elevation

    # Coastline filtering
    min_coastline_length_m: float = 500.0  # Minimum coastline segment length (filters small features)

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
            'n_uniform_layers': self.n_uniform_layers,
            'coastline_sample_res_m': self.coastline_sample_res_m,
            'max_land_elevation_m': self.max_land_elevation_m,
            'min_coastline_length_m': self.min_coastline_length_m,
            'coastline_density_bias': self.coastline_density_bias,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'SurfZoneMeshConfig':
        """Create from dictionary (with backwards compatibility)."""
        # Filter to only known fields for backwards compatibility
        known_fields = {
            'offshore_distance_m', 'onshore_distance_m', 'min_resolution_m',
            'max_resolution_m', 'n_uniform_layers', 'coastline_sample_res_m',
            'max_land_elevation_m', 'min_coastline_length_m', 'coastline_density_bias'
        }
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


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

        # Offshore boundary as ordered polylines (at offshore_distance_m from coastline)
        self.offshore_boundary: Optional[List[np.ndarray]] = None  # List of (N, 2) arrays

        # All mesh points (irregular point cloud)
        self.points_x: Optional[np.ndarray] = None
        self.points_y: Optional[np.ndarray] = None
        self.elevation: Optional[np.ndarray] = None
        self.coast_distance: Optional[np.ndarray] = None  # Distance from coastline (m)

        # Reference bounds
        self.lon_range: Optional[Tuple[float, float]] = None
        self.lat_range: Optional[Tuple[float, float]] = None

        # Interpolation (lazy loaded)
        self._triangulation: Optional[Delaunay] = None
        self._interpolator: Optional[LinearNDInterpolator] = None

        # Spatial index for fast triangle lookup (built once, saved with mesh)
        self._spatial_index: Optional[Dict[str, np.ndarray]] = None

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
        print(f"  Uniform layers: {cfg.n_uniform_layers} (first {cfg.min_resolution_m * cfg.n_uniform_layers:.0f}m)")
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

        # Step 3: Generate offset distances (uniform near coast, then biased)
        print(f"\n  Step 3: Generating offset distances...")
        offshore_distances = self._generate_offset_distances(
            cfg.min_resolution_m, cfg.max_resolution_m, cfg.offshore_distance_m,
            density_bias=cfg.coastline_density_bias,
            n_uniform_layers=cfg.n_uniform_layers,
        )
        # Onshore uses lower bias and fewer uniform layers since it's a small area
        onshore_distances = self._generate_offset_distances(
            cfg.min_resolution_m, cfg.min_resolution_m * 2, cfg.onshore_distance_m,
            density_bias=1.5,
            n_uniform_layers=3,
        )

        uniform_zone = cfg.min_resolution_m * cfg.n_uniform_layers
        print(f"    Offshore: {len(offshore_distances)} layers (uniform to {uniform_zone:.0f}m, then coarsening to {offshore_distances[-1]:.0f}m)")
        print(f"    Onshore: {len(onshore_distances)} layers from 0 to {onshore_distances[-1]:.0f}m")

        # Step 4: Generate points along offset curves
        print(f"\n  Step 4: Generating mesh points along offset curves...")
        all_x, all_y = self._generate_offset_curve_points(
            coastlines, offshore_distances, onshore_distances,
            cfg.min_resolution_m, cfg.max_resolution_m, cfg.offshore_distance_m,
            sample_x, sample_y, elev_grid
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

        # Step 5b: Interpolate from primary source for nearby gaps
        missing = np.isnan(all_elev)
        n_missing = np.sum(missing)
        if n_missing > 0:
            from scipy.spatial import cKDTree
            from scipy.interpolate import LinearNDInterpolator

            valid_mask = ~missing
            valid_pts = np.column_stack([all_x[valid_mask], all_y[valid_mask]])
            valid_elev = all_elev[valid_mask]
            missing_pts = np.column_stack([all_x[missing], all_y[missing]])

            # Build KDTree for fast neighbor lookup
            tree = cKDTree(valid_pts)

            # Find distance to nearest valid point
            distances, _ = tree.query(missing_pts, k=1)

            # Only interpolate points within max_interp_distance (200m)
            max_interp_dist = 200.0  # meters
            can_interpolate = distances <= max_interp_dist

            if np.any(can_interpolate):
                # Use LinearNDInterpolator for smooth interpolation
                interp = LinearNDInterpolator(valid_pts, valid_elev)
                interp_elev = interp(missing_pts[can_interpolate])

                # Fill in the values
                missing_indices = np.where(missing)[0]
                all_elev[missing_indices[can_interpolate]] = interp_elev

                n_filled = np.sum(~np.isnan(interp_elev))
                print(f"    Interpolated from primary: {n_filled:,} points (within {max_interp_dist:.0f}m of valid data)")

        # Step 5c: Fallback source — NCEI CRM (if provided) for remaining gaps
        if fallback_bathy is not None:
            missing = np.isnan(all_elev)
            n_missing = np.sum(missing)
            if n_missing > 0:
                print(f"    Filling {n_missing:,} remaining points from fallback (NCEI CRM)...")
                fallback_elev = fallback_bathy.sample_points(
                    all_lon[missing], all_lat[missing]
                )
                all_elev[missing] = fallback_elev
                n_fallback = np.sum(~np.isnan(fallback_elev))
                print(f"    From fallback: {n_fallback:,} points ({100*n_fallback/n_missing:.1f}% of missing)")

        # Report remaining gaps
        still_missing = np.sum(np.isnan(all_elev))
        if still_missing > 0:
            print(f"    Remaining gaps: {still_missing:,} points (no data from any source)")

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

        # Step 7: Compute offshore boundary and coast distance
        print(f"\n  Step 7: Computing offshore boundary and coast distance...")
        self.compute_offshore_boundary()
        self.compute_coast_distance()

    def _extract_coastline_contours(
        self,
        x: np.ndarray,
        y: np.ndarray,
        elevation: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Extract coastline as contours at the land/water boundary.

        Uses matplotlib's contour algorithm (marching squares) on a BINARY land mask.
        This prevents false coastlines at data void boundaries (NaN regions).

        Returns list of (N, 2) arrays, each representing a coastline segment.
        """
        import matplotlib.pyplot as plt
        from scipy import ndimage

        # Create binary land mask: 1 = land (elev >= 0), 0 = water OR no data
        # By treating NaN as water, we avoid creating false coastlines at data voids
        valid_data = ~np.isnan(elevation)
        land_mask = np.where(valid_data, elevation >= 0, False)

        # Note: Morphological closing was removed to preserve accurate coastline geometry.
        # The previous 150m closing kernel (3x3 at 50m resolution) was filling in small bays
        # and concave features, causing holes in near-coast offset contours.
        land_mask = land_mask.astype(float)

        # Use matplotlib's contour to extract the 0.5-level contour of the binary mask
        # This finds the boundary between land (1) and water (0)
        fig, ax = plt.subplots()
        cs = ax.contour(x, y, land_mask, levels=[0.5])
        plt.close(fig)

        # Extract contour paths (compatible with newer matplotlib versions)
        raw_coastlines = []
        if hasattr(cs, 'allsegs') and len(cs.allsegs) > 0:
            for segment in cs.allsegs[0]:
                if len(segment) >= 2:
                    raw_coastlines.append(segment.copy())
        else:
            # Fallback for older matplotlib
            for collection in cs.collections:
                for path in collection.get_paths():
                    vertices = path.vertices
                    if len(vertices) >= 2:
                        raw_coastlines.append(vertices.copy())

        # Filter out short segments (lagoons, harbors, small islands)
        min_segment_length = self.config.min_coastline_length_m
        coastlines = []
        for segment in raw_coastlines:
            if self._polyline_length(segment) >= min_segment_length:
                coastlines.append(segment)

        n_filtered = len(raw_coastlines) - len(coastlines)
        if n_filtered > 0:
            print(f"    Filtered {n_filtered} short coastline segments (<{min_segment_length/1000:.1f}km)")

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
        n_uniform_layers: int = 10,
    ) -> np.ndarray:
        """
        Generate offset distances with uniform spacing near coast, then coarsening.

        First generates n_uniform_layers at min_spacing, then uses a power function
        to concentrate more layers near that transition zone.

        Args:
            min_spacing: Minimum spacing between offsets (at coast)
            max_spacing: Maximum spacing between offsets (at max distance)
            max_distance: Maximum offset distance
            density_bias: Power factor for density bias in coarsening zone
            n_uniform_layers: Number of layers at uniform min_spacing before coarsening

        Returns:
            Array of offset distances from coastline
        """
        distances = [0.0]

        # Phase 1: Uniform spacing near coastline
        uniform_zone_end = min_spacing * n_uniform_layers
        for i in range(1, n_uniform_layers + 1):
            d = min_spacing * i
            if d <= max_distance:
                distances.append(d)

        if uniform_zone_end >= max_distance:
            return np.array(distances)

        # Phase 2: Biased spacing from uniform zone to max distance
        remaining_distance = max_distance - uniform_zone_end

        # Estimate number of coarsening layers
        avg_spacing = (min_spacing + max_spacing) / 2
        n_coarse_layers = max(5, int(remaining_distance / avg_spacing))

        # Generate normalized positions [0, 1] with uniform spacing
        t = np.linspace(0, 1, n_coarse_layers + 1)[1:]  # Skip 0 (already have uniform_zone_end)

        # Apply power function to concentrate values near 0 (transition zone)
        t_biased = t ** density_bias

        # Scale to remaining distance and add offset
        coarse_distances = uniform_zone_end + t_biased * remaining_distance

        distances.extend(coarse_distances.tolist())

        return np.array(distances)

    def _generate_offset_curve_points(
        self,
        coastlines: List[np.ndarray],
        offshore_distances: np.ndarray,
        onshore_distances: np.ndarray,
        min_spacing: float,
        max_spacing: float,
        max_offshore: float,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        elevation_grid: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate points along iso-distance contours from the coastline.

        Uses a signed distance field approach:
        1. Compute distance transform from land mask boundary
        2. Extract iso-contours at each desired offset distance
        3. Sample points along those unified contours

        This treats all coastline segments as one unified feature, avoiding
        overlapping offset curves from individual segments.
        """
        from scipy import ndimage

        all_x = []
        all_y = []

        # Get grid resolution (should be uniform)
        dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else self.config.coastline_sample_res_m

        # Create binary land mask (same as in _extract_coastline_contours)
        valid_data = ~np.isnan(elevation_grid)
        land_mask = np.where(valid_data, elevation_grid >= 0, False)

        # Note: Morphological closing removed to match _extract_coastline_contours()
        # and preserve accurate coastline geometry for offset curve generation.

        # Compute distance transforms (in grid cells, then convert to meters)
        # Distance from ocean cells to nearest land cell
        dist_from_ocean = ndimage.distance_transform_edt(~land_mask) * dx
        # Distance from land cells to nearest ocean cell
        dist_from_land = ndimage.distance_transform_edt(land_mask) * dx

        # Create signed distance field: positive = ocean (distance from shore into water)
        #                               negative = land (distance from shore into land)
        signed_distance = np.where(land_mask, -dist_from_land, dist_from_ocean)

        print(f"    Distance field range: {signed_distance.min():.0f}m to {signed_distance.max():.0f}m")

        # Add coastline points (distance = 0) - sample from extracted coastlines
        for coastline in coastlines:
            sampled = self._sample_polyline(coastline, min_spacing)
            all_x.extend(sampled[:, 0])
            all_y.extend(sampled[:, 1])
        n_coastline = len(all_x)
        print(f"    Coastline points: {n_coastline:,}")

        # Extract iso-contours for offshore distances (positive = into ocean)
        n_offshore = 0
        for dist in offshore_distances:
            if dist <= 0:
                continue  # Skip 0, already added coastline points

            # Spacing along curve proportional to distance from shore
            t = dist / max_offshore if max_offshore > 0 else 0
            along_spacing = min_spacing + t * (max_spacing - min_spacing)

            # Extract iso-contour at this distance
            contour_points = self._extract_iso_contour(
                x_grid, y_grid, signed_distance, dist, along_spacing
            )
            if len(contour_points) > 0:
                all_x.extend(contour_points[:, 0])
                all_y.extend(contour_points[:, 1])
                n_offshore += len(contour_points)

        print(f"    Offshore points: {n_offshore:,} ({len(offshore_distances)-1} contours)")

        # Extract iso-contours for onshore distances (negative = into land)
        n_onshore = 0
        for dist in onshore_distances:
            if dist <= 0:
                continue

            along_spacing = min_spacing  # Keep dense onshore

            # Negative distance for land side
            contour_points = self._extract_iso_contour(
                x_grid, y_grid, signed_distance, -dist, along_spacing
            )
            if len(contour_points) > 0:
                all_x.extend(contour_points[:, 0])
                all_y.extend(contour_points[:, 1])
                n_onshore += len(contour_points)

        print(f"    Onshore points: {n_onshore:,} ({len(onshore_distances)-1} contours)")

        return np.array(all_x), np.array(all_y)

    def _extract_iso_contour(
        self,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        distance_field: np.ndarray,
        target_distance: float,
        spacing: float,
    ) -> np.ndarray:
        """
        Extract points along an iso-distance contour.

        Args:
            x_grid: 1D array of x coordinates
            y_grid: 1D array of y coordinates
            distance_field: 2D signed distance field
            target_distance: Distance level to extract (positive=ocean, negative=land)
            spacing: Point spacing along the contour

        Returns:
            (N, 2) array of sampled points along the iso-contour
        """
        import matplotlib.pyplot as plt

        # Use matplotlib contour to extract iso-line
        fig, ax = plt.subplots()
        cs = ax.contour(x_grid, y_grid, distance_field, levels=[target_distance])
        plt.close(fig)

        all_points = []

        # Extract contour paths (compatible with newer matplotlib versions)
        if hasattr(cs, 'allsegs') and len(cs.allsegs) > 0:
            segments = cs.allsegs[0]
        else:
            segments = []
            for collection in cs.collections:
                for path in collection.get_paths():
                    if len(path.vertices) >= 2:
                        segments.append(path.vertices.copy())

        # Sample points along each segment
        for segment in segments:
            if len(segment) >= 2:
                sampled = self._sample_polyline(segment, spacing)
                all_points.append(sampled)

        if not all_points:
            return np.array([]).reshape(0, 2)

        return np.vstack(all_points)

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

    def compute_offshore_boundary(
        self,
        grid_resolution: float = 50.0,
        boundary_spacing: float = 100.0,
    ) -> List[np.ndarray]:
        """
        Compute the offshore boundary as polylines at offshore_distance_m from coastline.

        Uses a signed distance field approach:
        1. Create a grid covering the mesh extent
        2. Compute signed distance from coastline points
        3. Extract iso-contour at offshore_distance_m

        This can be called on an existing mesh to compute/recompute the boundary.

        Args:
            grid_resolution: Resolution of the distance field grid (meters)
            boundary_spacing: Point spacing along the extracted boundary (meters)

        Returns:
            List of (N, 2) arrays representing boundary polylines
        """
        from scipy import ndimage
        import matplotlib.pyplot as plt

        if self.coastlines is None or len(self.coastlines) == 0:
            print("Warning: No coastlines available to compute offshore boundary")
            return []

        if self.points_x is None or len(self.points_x) == 0:
            print("Warning: No mesh points available")
            return []

        print(f"  Computing offshore boundary at {self.config.offshore_distance_m}m...")

        # Create a grid covering the mesh extent with padding
        padding = self.config.offshore_distance_m * 1.5
        x_min, x_max = self.points_x.min() - padding, self.points_x.max() + padding
        y_min, y_max = self.points_y.min() - padding, self.points_y.max() + padding

        x_grid = np.arange(x_min, x_max, grid_resolution)
        y_grid = np.arange(y_min, y_max, grid_resolution)

        # Rasterize coastline points onto grid to create land mask
        # First, collect all coastline points
        coastline_points = []
        for coastline in self.coastlines:
            # Sample coastline at grid resolution for accurate rasterization
            sampled = self._sample_polyline(coastline, grid_resolution / 2)
            coastline_points.append(sampled)

        if not coastline_points:
            print("Warning: No coastline points found")
            return []

        all_coast_pts = np.vstack(coastline_points)

        # Create binary mask: 1 = near coastline, 0 = far from coastline
        # We'll use distance transform to get signed distance
        coast_mask = np.zeros((len(y_grid), len(x_grid)), dtype=bool)

        # Mark grid cells containing coastline points
        for pt in all_coast_pts:
            ix = int((pt[0] - x_min) / grid_resolution)
            iy = int((pt[1] - y_min) / grid_resolution)
            if 0 <= ix < len(x_grid) and 0 <= iy < len(y_grid):
                coast_mask[iy, ix] = True

        # Dilate coastline mask slightly to ensure connectivity
        from scipy.ndimage import binary_dilation
        coast_mask = binary_dilation(coast_mask, iterations=2)

        # Compute distance from coastline for all grid cells
        # Distance transform gives distance to nearest True cell
        dist_from_coast = ndimage.distance_transform_edt(~coast_mask) * grid_resolution

        print(f"    Distance field range: 0 to {dist_from_coast.max():.0f}m")

        # Extract iso-contour at offshore_distance_m
        target_distance = self.config.offshore_distance_m

        fig, ax = plt.subplots()
        cs = ax.contour(x_grid, y_grid, dist_from_coast, levels=[target_distance])
        plt.close(fig)

        boundary_segments = []

        # Extract contour paths
        if hasattr(cs, 'allsegs') and len(cs.allsegs) > 0:
            segments = cs.allsegs[0]
        else:
            segments = []
            for collection in cs.collections:
                for path in collection.get_paths():
                    if len(path.vertices) >= 2:
                        segments.append(path.vertices.copy())

        # Sample points along each segment
        for segment in segments:
            if len(segment) >= 2:
                # Filter to only keep segments within reasonable bounds of mesh
                sampled = self._sample_polyline(segment, boundary_spacing)
                if len(sampled) >= 2:
                    boundary_segments.append(sampled)

        print(f"    Extracted {len(boundary_segments)} boundary segments")
        total_points = sum(len(seg) for seg in boundary_segments)
        print(f"    Total boundary points: {total_points:,}")

        self.offshore_boundary = boundary_segments
        return boundary_segments

    def compute_coast_distance(self) -> np.ndarray:
        """
        Compute distance from coastline for each mesh point.

        Uses the stored coastlines to compute the minimum distance from each
        mesh point to any coastline segment. This is used by the ray tracer
        to determine when rays have reached the offshore boundary.

        Returns:
            Array of distances (m) from coastline for each mesh point
        """
        from scipy.spatial import cKDTree

        if self.coastlines is None or len(self.coastlines) == 0:
            print("Warning: No coastlines available to compute coast distance")
            return np.array([])

        if self.points_x is None or len(self.points_x) == 0:
            print("Warning: No mesh points available")
            return np.array([])

        print(f"  Computing coast distance for {len(self.points_x):,} mesh points...")

        # Collect all coastline points (sample densely for accuracy)
        coast_pts = []
        for coastline in self.coastlines:
            # Sample coastline at fine resolution
            sampled = self._sample_polyline(coastline, 10.0)  # 10m spacing
            coast_pts.append(sampled)

        all_coast_pts = np.vstack(coast_pts)
        print(f"    Using {len(all_coast_pts):,} coastline points")

        # Build KD-tree for fast nearest neighbor lookup
        tree = cKDTree(all_coast_pts)

        # Query distance for each mesh point
        mesh_pts = np.column_stack([self.points_x, self.points_y])
        distances, _ = tree.query(mesh_pts)

        print(f"    Distance range: {distances.min():.1f}m to {distances.max():.1f}m")

        self.coast_distance = distances.astype(np.float64)
        return self.coast_distance

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
    # Spatial Index for Fast Triangle Lookup
    # =========================================================================

    def build_spatial_index(self, cell_size: float = 50.0) -> None:
        """
        Build a grid-based spatial index for fast triangle lookup.

        This index is saved with the mesh and loaded on subsequent runs,
        so it only needs to be built once per mesh.

        Args:
            cell_size: Size of grid cells in meters (default 50m)
        """
        import time
        t0 = time.time()

        if self._triangulation is None:
            self._build_interpolator()

        points_x = self.points_x
        points_y = self.points_y
        triangles = self._triangulation.simplices

        # Compute bounding box with padding
        x_min = points_x.min() - cell_size
        x_max = points_x.max() + cell_size
        y_min = points_y.min() - cell_size
        y_max = points_y.max() + cell_size

        n_cells_x = int(np.ceil((x_max - x_min) / cell_size))
        n_cells_y = int(np.ceil((y_max - y_min) / cell_size))
        n_cells = n_cells_x * n_cells_y
        n_triangles = triangles.shape[0]

        print(f"  Building spatial index: {n_cells_x}x{n_cells_y} = {n_cells:,} cells, cell_size={cell_size}m")

        # Vectorized: get all triangle vertex coordinates
        v0_x = points_x[triangles[:, 0]]
        v0_y = points_y[triangles[:, 0]]
        v1_x = points_x[triangles[:, 1]]
        v1_y = points_y[triangles[:, 1]]
        v2_x = points_x[triangles[:, 2]]
        v2_y = points_y[triangles[:, 2]]

        # Vectorized: compute bounding box for all triangles
        tri_x_min = np.minimum(np.minimum(v0_x, v1_x), v2_x)
        tri_x_max = np.maximum(np.maximum(v0_x, v1_x), v2_x)
        tri_y_min = np.minimum(np.minimum(v0_y, v1_y), v2_y)
        tri_y_max = np.maximum(np.maximum(v0_y, v1_y), v2_y)

        # Vectorized: compute cell ranges for all triangles
        cx_min = np.clip(((tri_x_min - x_min) / cell_size).astype(np.int32), 0, n_cells_x - 1)
        cx_max = np.clip(((tri_x_max - x_min) / cell_size).astype(np.int32), 0, n_cells_x - 1)
        cy_min = np.clip(((tri_y_min - y_min) / cell_size).astype(np.int32), 0, n_cells_y - 1)
        cy_max = np.clip(((tri_y_max - y_min) / cell_size).astype(np.int32), 0, n_cells_y - 1)

        # Count cells per triangle and total entries
        cells_per_tri = (cx_max - cx_min + 1) * (cy_max - cy_min + 1)
        total_entries = int(cells_per_tri.sum())

        print(f"  Total cell entries: {total_entries:,} (avg {total_entries/n_triangles:.1f} per triangle)")

        # Build the index using pure Python (simpler, still fast enough for one-time build)
        cell_counts = np.zeros(n_cells, dtype=np.int32)

        # First pass: count
        for tri_idx in range(n_triangles):
            for cy in range(cy_min[tri_idx], cy_max[tri_idx] + 1):
                for cx in range(cx_min[tri_idx], cx_max[tri_idx] + 1):
                    cell_idx = cy * n_cells_x + cx
                    cell_counts[cell_idx] += 1

        # Compute start indices
        cell_starts = np.zeros(n_cells + 1, dtype=np.int32)
        cell_starts[1:] = np.cumsum(cell_counts)

        # Second pass: fill
        grid_triangles = np.empty(total_entries, dtype=np.int32)
        cell_fill = np.zeros(n_cells, dtype=np.int32)

        for tri_idx in range(n_triangles):
            for cy in range(cy_min[tri_idx], cy_max[tri_idx] + 1):
                for cx in range(cx_min[tri_idx], cx_max[tri_idx] + 1):
                    cell_idx = cy * n_cells_x + cx
                    pos = cell_starts[cell_idx] + cell_fill[cell_idx]
                    grid_triangles[pos] = tri_idx
                    cell_fill[cell_idx] += 1

        self._spatial_index = {
            'grid_x_min': np.float64(x_min),
            'grid_y_min': np.float64(y_min),
            'grid_cell_size': np.float64(cell_size),
            'grid_n_cells_x': np.int32(n_cells_x),
            'grid_n_cells_y': np.int32(n_cells_y),
            'grid_cell_starts': cell_starts.astype(np.int32),
            'grid_cell_counts': cell_counts.astype(np.int32),
            'grid_triangles': grid_triangles.astype(np.int32),
        }

        print(f"  Spatial index built in {time.time() - t0:.1f}s")

    def has_spatial_index(self) -> bool:
        """Check if spatial index has been built."""
        return self._spatial_index is not None

    # =========================================================================
    # Numba-Compatible Data Export
    # =========================================================================

    def get_numba_arrays(self) -> Dict[str, np.ndarray]:
        """
        Get arrays formatted for Numba ray tracing.

        Includes spatial index if available.
        """
        if self._triangulation is None:
            self._build_interpolator()

        result = {
            'points_x': np.ascontiguousarray(self.points_x, dtype=np.float64),
            'points_y': np.ascontiguousarray(self.points_y, dtype=np.float64),
            'elevation': np.ascontiguousarray(self.elevation, dtype=np.float64),
            'depth': np.ascontiguousarray(-self.elevation, dtype=np.float64),
            'triangles': np.ascontiguousarray(self._triangulation.simplices, dtype=np.int32),
        }

        # Include coast_distance if available (for boundary detection)
        if self.coast_distance is not None:
            result['coast_distance'] = np.ascontiguousarray(self.coast_distance, dtype=np.float64)

        # Include offshore_distance_m from config (for boundary threshold)
        result['offshore_distance_m'] = self.config.offshore_distance_m

        # Include spatial index if available
        if self._spatial_index is not None:
            result.update(self._spatial_index)

        return result

    # =========================================================================
    # Save / Load
    # =========================================================================

    def save(self, directory: Path, build_spatial_index: bool = True) -> Path:
        """
        Save mesh to directory.

        Args:
            directory: Output directory
            build_spatial_index: If True, build spatial index before saving
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Build spatial index if requested and not already built
        if build_spatial_index and self._spatial_index is None:
            print("\n  Step 8: Building spatial index for ray tracing...")
            self.build_spatial_index()

        npz_path = directory / f"{self.region_name}_surfzone.npz"
        save_dict = {
            'points_x': self.points_x,
            'points_y': self.points_y,
            'elevation': self.elevation,
        }

        # Save coast_distance if available
        if self.coast_distance is not None:
            save_dict['coast_distance'] = self.coast_distance

        # Save coastlines as separate arrays
        if self.coastlines:
            for i, coastline in enumerate(self.coastlines):
                save_dict[f'coastline_{i}'] = coastline

        # Save offshore boundary as separate arrays
        if self.offshore_boundary:
            for i, boundary in enumerate(self.offshore_boundary):
                save_dict[f'offshore_boundary_{i}'] = boundary

        # Save spatial index if available
        if self._spatial_index is not None:
            for key, value in self._spatial_index.items():
                save_dict[f'spatial_{key}'] = value

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

        # Load coast_distance if available
        if 'coast_distance' in data:
            mesh.coast_distance = data['coast_distance']
        else:
            mesh.coast_distance = None

        # Load coastlines
        mesh.coastlines = []
        i = 0
        while f'coastline_{i}' in data:
            mesh.coastlines.append(data[f'coastline_{i}'])
            i += 1

        # Load offshore boundary if available
        mesh.offshore_boundary = []
        i = 0
        while f'offshore_boundary_{i}' in data:
            mesh.offshore_boundary.append(data[f'offshore_boundary_{i}'])
            i += 1
        if not mesh.offshore_boundary:
            mesh.offshore_boundary = None  # No boundary saved

        # Load spatial index if available
        if 'spatial_grid_x_min' in data:
            mesh._spatial_index = {
                'grid_x_min': float(data['spatial_grid_x_min']),
                'grid_y_min': float(data['spatial_grid_y_min']),
                'grid_cell_size': float(data['spatial_grid_cell_size']),
                'grid_n_cells_x': int(data['spatial_grid_n_cells_x']),
                'grid_n_cells_y': int(data['spatial_grid_n_cells_y']),
                'grid_cell_starts': data['spatial_grid_cell_starts'],
                'grid_cell_counts': data['spatial_grid_cell_counts'],
                'grid_triangles': data['spatial_grid_triangles'],
            }
            print(f"Loaded surf zone mesh: {region_name}")
            print(f"  Points: {len(mesh.points_x):,}")
            print(f"  Spatial index: loaded from disk")
        else:
            print(f"Loaded surf zone mesh: {region_name}")
            print(f"  Points: {len(mesh.points_x):,}")
            print(f"  Spatial index: not found (will build on first use)")

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
