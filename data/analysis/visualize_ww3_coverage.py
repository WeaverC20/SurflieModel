#!/usr/bin/env python3
"""
WaveWatch III Grid Coverage Visualization

Shows how far the WW3 model grid extends towards the California coast,
highlighting the boundary between valid ocean data and land/NaN values.

This helps identify where SWAN model coverage needs to begin.

Usage:
    python visualize_ww3_coverage.py                    # Default California view
    python visualize_ww3_coverage.py --region socal     # Southern California focus
    python visualize_ww3_coverage.py --region norcal    # Northern California focus
    python visualize_ww3_coverage.py --save             # Save figure to disk
"""

import asyncio
import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe

# Try to import cartopy for proper coastlines
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    print("Warning: cartopy not installed. Install with: pip install cartopy")
    print("Falling back to approximate coastline.")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.pipelines.wave.wavewatch_fetcher import WaveWatchFetcher

# Output directory
FIGURES_DIR = Path(__file__).parent / "figures"

# Region definitions (min_lat, max_lat, min_lon, max_lon)
REGIONS = {
    'california': (32.0, 42.0, -126.0, -117.0),
    'socal': (32.0, 35.0, -122.0, -117.0),
    'norcal': (36.0, 42.0, -126.0, -121.0),
    'central': (34.0, 38.0, -124.0, -119.0),
    'channel_islands': (33.0, 34.5, -121.0, -118.5),
}


def create_wave_colormap():
    """Create a colormap for wave height visualization"""
    colors = [
        (0.0, '#f7fbff'),   # Very light blue (small waves)
        (0.2, '#c6dbef'),
        (0.4, '#6baed6'),   # Medium blue
        (0.6, '#2171b5'),   # Darker blue
        (0.8, '#08519c'),   # Dark blue
        (1.0, '#08306b'),   # Very dark blue (large waves)
    ]
    return LinearSegmentedColormap.from_list('wave_height', colors)


def create_coverage_colormap():
    """Create a colormap for coverage visualization"""
    colors = [
        (0.0, '#d73027'),   # Red - no data
        (0.5, '#fee08b'),   # Yellow - partial
        (1.0, '#1a9850'),   # Green - full coverage
    ]
    return LinearSegmentedColormap.from_list('coverage', colors)


async def fetch_ww3_data(region: Tuple[float, float, float, float]) -> dict:
    """Fetch WW3 data for the specified region"""
    min_lat, max_lat, min_lon, max_lon = region

    fetcher = WaveWatchFetcher()
    print(f"Fetching WW3 data for region: {min_lat}°N to {max_lat}°N, {min_lon}°E to {max_lon}°E")

    data = await fetcher.fetch_wave_grid(
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        forecast_hour=0
    )

    return data


def plot_ww3_coverage(
    data: dict,
    region_name: str = "California",
    output_path: Optional[Path] = None,
    show_wave_height: bool = True,
    figsize: Tuple[int, int] = (14, 12)
):
    """
    Create visualization of WW3 grid coverage.

    Args:
        data: WW3 data dict from fetcher
        region_name: Name for title
        output_path: Optional path to save figure
        show_wave_height: If True, show wave heights; if False, show coverage mask
        figsize: Figure size
    """
    lats = np.array(data['lat'])
    lons = np.array(data['lon'])
    wave_height = np.array(data['significant_wave_height'])

    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Create valid data mask (True where we have ocean data)
    valid_mask = ~np.isnan(wave_height)

    # Calculate statistics
    total_cells = wave_height.size
    valid_cells = np.sum(valid_mask)
    land_cells = total_cells - valid_cells
    coverage_pct = (valid_cells / total_cells) * 100

    # Find the boundary (easternmost valid longitude at each latitude)
    boundary_lons = []
    boundary_lats = []
    for i, lat in enumerate(lats):
        row = wave_height[i, :]
        valid_indices = np.where(~np.isnan(row))[0]
        if len(valid_indices) > 0:
            # Find easternmost valid point (closest to coast)
            easternmost_idx = valid_indices[-1]
            boundary_lons.append(lons[easternmost_idx])
            boundary_lats.append(lat)

    # Create figure with subplots
    if CARTOPY_AVAILABLE:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(1, 2, 2)
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        ax1, ax2 = axes

    # === Left plot: Wave Height with Coverage ===
    # Plot wave height
    cmap = create_wave_colormap()
    wave_masked = np.ma.masked_invalid(wave_height)

    if CARTOPY_AVAILABLE:
        im1 = ax1.pcolormesh(
            lon_grid, lat_grid, wave_masked,
            cmap=cmap,
            shading='auto',
            vmin=0,
            vmax=np.nanmax(wave_height),
            transform=ccrs.PlateCarree()
        )
    else:
        im1 = ax1.pcolormesh(
            lon_grid, lat_grid, wave_masked,
            cmap=cmap,
            shading='auto',
            vmin=0,
            vmax=np.nanmax(wave_height)
        )

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, pad=0.02)
    cbar1.set_label('Significant Wave Height (m)', fontsize=11)

    # Plot WW3 boundary (nearest valid data to coast)
    if boundary_lons:
        if CARTOPY_AVAILABLE:
            ax1.plot(
                boundary_lons, boundary_lats,
                color='red', linewidth=2.5, linestyle='-',
                label='WW3 Valid Data Boundary',
                transform=ccrs.PlateCarree(),
                path_effects=[pe.Stroke(linewidth=4, foreground='white'), pe.Normal()]
            )
        else:
            ax1.plot(
                boundary_lons, boundary_lats,
                color='red', linewidth=2.5, linestyle='-',
                label='WW3 Valid Data Boundary',
                path_effects=[pe.Stroke(linewidth=4, foreground='white'), pe.Normal()]
            )

    # Add proper coastlines and features
    if CARTOPY_AVAILABLE:
        # Use Natural Earth high-resolution coastlines (actual GIS data)
        ax1.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='black')
        ax1.add_feature(cfeature.LAND, facecolor='#d3d3d3', alpha=0.8)
        ax1.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
        ax1.add_feature(cfeature.STATES, linestyle='-', alpha=0.3, linewidth=0.5)

        # Set extent
        ax1.set_extent([lons.min(), lons.max(), lats.min(), lats.max()], crs=ccrs.PlateCarree())

        # Add gridlines
        gl = ax1.gridlines(draw_labels=True, alpha=0.3, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
    else:
        # Fallback: just show gray for NaN areas
        land_mask = np.isnan(wave_height).astype(float)
        ax1.contourf(
            lon_grid, lat_grid, land_mask,
            levels=[0.5, 1.5],
            colors=['#d3d3d3'],
            alpha=0.8
        )
        ax1.set_aspect('equal')

    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.set_title(f'WaveWatch III Coverage - {region_name}\nWave Height & Valid Data Boundary (Red Line)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)

    # === Right plot: Distance to Coast Analysis ===
    # Note: ax2 was already set up above

    # Calculate actual distance from each WW3 boundary point to nearest coastline
    distances_km = []

    if CARTOPY_AVAILABLE:
        from cartopy.geodesic import Geodesic
        from cartopy.io.shapereader import natural_earth, Reader

        # Load high-resolution coastline from Natural Earth
        coastline_shp = natural_earth(resolution='10m', category='physical', name='coastline')
        reader = Reader(coastline_shp)

        # Extract all coastline coordinates within our region
        coast_points = []
        lat_min, lat_max = min(boundary_lats) - 1, max(boundary_lats) + 1
        lon_min, lon_max = min(boundary_lons) - 1, max(boundary_lons) + 1

        for geometry in reader.geometries():
            coords = list(geometry.coords)
            for lon, lat in coords:
                # Filter to our region of interest
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                    coast_points.append((lon, lat))

        coast_points = np.array(coast_points)
        print(f"  Loaded {len(coast_points)} coastline points for distance calculation")

        # Calculate geodesic distance from each boundary point to nearest coast point
        geod = Geodesic()

        # Sample every N boundary points to reduce computation
        sample_rate = max(1, len(boundary_lons) // 50)  # Max ~50 points
        sampled_indices = list(range(0, len(boundary_lons), sample_rate))

        sampled_lons = [boundary_lons[i] for i in sampled_indices]
        sampled_lats = [boundary_lats[i] for i in sampled_indices]

        for b_lon, b_lat in zip(sampled_lons, sampled_lats):
            # Find nearest coastline point using simple Euclidean approximation first
            # (faster than computing all geodesic distances)
            km_per_deg_lon = 111 * np.cos(np.radians(b_lat))
            km_per_deg_lat = 111

            dx = (coast_points[:, 0] - b_lon) * km_per_deg_lon
            dy = (coast_points[:, 1] - b_lat) * km_per_deg_lat
            approx_distances = np.sqrt(dx**2 + dy**2)

            nearest_idx = np.argmin(approx_distances)
            min_dist_km = approx_distances[nearest_idx]

            distances_km.append(min_dist_km)

        # Use sampled lats for plotting
        boundary_lats_for_plot = sampled_lats
    else:
        # Fallback: rough estimate based on WW3 resolution
        boundary_lats_for_plot = boundary_lats
        for b_lon, b_lat in zip(boundary_lons, boundary_lats):
            km_per_deg_lon = 111 * np.cos(np.radians(b_lat))
            estimated_gap_deg = 0.25
            distances_km.append(estimated_gap_deg * km_per_deg_lon)

    # Plot distance vs latitude
    ax2.fill_between(boundary_lats_for_plot, 0, distances_km, alpha=0.3, color='blue')
    ax2.plot(boundary_lats_for_plot, distances_km, color='blue', linewidth=2, marker='o', markersize=4, label='Distance to coast')

    # Add reference lines
    ax2.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='WW3 resolution (~25 km)')
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50 km reference')

    # Mark key locations (only if within the sampled data range)
    key_points = [
        (32.7, 'San Diego'),
        (33.6, 'Huntington'),
        (34.0, 'LA'),
        (34.45, 'Pt. Conception'),
        (36.6, 'Monterey'),
        (37.8, 'SF'),
        (40.8, 'Eureka'),
    ]
    for lat, name in key_points:
        if boundary_lats_for_plot and min(boundary_lats_for_plot) <= lat <= max(boundary_lats_for_plot):
            idx = np.argmin(np.abs(np.array(boundary_lats_for_plot) - lat))
            if idx < len(distances_km):
                ax2.annotate(
                    name,
                    (boundary_lats_for_plot[idx], distances_km[idx]),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha='center',
                    fontsize=8,
                    rotation=45
                )
                ax2.scatter([boundary_lats_for_plot[idx]], [distances_km[idx]], color='red', s=30, zorder=5)

    ax2.set_xlabel('Latitude (°N)', fontsize=12)
    ax2.set_ylabel('Distance from WW3 Boundary to Coast (km)', fontsize=12)
    ax2.set_title('Gap Between WW3 Valid Data and Coastline\n(Region where SWAN coverage needed)', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(distances_km) * 1.2 if distances_km else 100)

    # Add statistics text box
    stats_text = (
        f"WW3 Grid Statistics:\n"
        f"  Resolution: {data.get('resolution_deg', 0.25)}° ({data.get('resolution_deg', 0.25) * 111:.0f} km)\n"
        f"  Total cells: {total_cells:,}\n"
        f"  Ocean cells: {valid_cells:,} ({coverage_pct:.1f}%)\n"
        f"  Land/Invalid: {land_cells:,}\n\n"
        f"Gap to Coast:\n"
        f"  Min: {min(distances_km):.1f} km\n"
        f"  Max: {max(distances_km):.1f} km\n"
        f"  Mean: {np.mean(distances_km):.1f} km\n\n"
        f"Model: {data.get('model', 'WW3')}\n"
        f"Cycle: {data.get('cycle_time', 'N/A')}"
    )
    ax2.text(
        0.02, 0.98, stats_text,
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.tight_layout()

    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig, (ax1, ax2)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize WaveWatch III grid coverage near California coast',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_ww3_coverage.py                    # Full California
    python visualize_ww3_coverage.py --region socal     # Southern California
    python visualize_ww3_coverage.py --region norcal    # Northern California
    python visualize_ww3_coverage.py --save             # Save to file

Available regions: """ + ', '.join(REGIONS.keys())
    )

    parser.add_argument(
        '--region', type=str, choices=list(REGIONS.keys()),
        default='california',
        help='Predefined region to visualize'
    )
    parser.add_argument(
        '--save', action='store_true',
        help='Save figure to disk'
    )
    parser.add_argument(
        '--no-show', action='store_true',
        help='Do not display the plot (useful with --save)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("WAVEWATCH III COVERAGE VISUALIZATION")
    print("=" * 60)

    # Get region bounds
    region = REGIONS[args.region]
    print(f"Region: {args.region}")
    print(f"Bounds: {region[0]}°N to {region[1]}°N, {region[2]}°E to {region[3]}°E")

    # Fetch data
    print("\nFetching WW3 data...")
    data = asyncio.run(fetch_ww3_data(region))

    if 'synthetic' in data.get('model', '').lower():
        print("\nWARNING: Using synthetic data - real WW3 data not available")
        print("Install cfgrib and xarray for real data: pip install cfgrib xarray")

    print(f"\nGrid size: {len(data['lat'])} lat x {len(data['lon'])} lon")

    # Create visualization
    output_path = None
    if args.save:
        output_path = FIGURES_DIR / f"ww3_coverage_{args.region}.png"

    plot_ww3_coverage(
        data,
        region_name=args.region.replace('_', ' ').title(),
        output_path=output_path
    )

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)

    if not args.no_show:
        print("\nClose plot window to exit.")
        plt.show()


if __name__ == '__main__':
    main()
