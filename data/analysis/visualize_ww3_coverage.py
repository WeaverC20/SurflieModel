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

# Approximate California coastline points for visualization
# (lon, lat) pairs tracing the coast from south to north
CA_COASTLINE = [
    (-117.13, 32.53),   # Mexican border
    (-117.17, 32.72),   # San Diego
    (-117.26, 32.87),   # La Jolla
    (-117.39, 33.20),   # Oceanside
    (-117.60, 33.40),   # San Clemente
    (-117.87, 33.60),   # Dana Point
    (-117.96, 33.63),   # Huntington Beach
    (-118.00, 33.72),   # Newport Beach
    (-118.11, 33.77),   # Laguna Beach
    (-118.41, 33.95),   # Long Beach
    (-118.47, 33.98),   # San Pedro
    (-118.52, 33.89),   # Palos Verdes
    (-118.67, 33.97),   # Redondo Beach
    (-118.82, 33.95),   # Marina del Rey
    (-118.88, 34.01),   # Santa Monica
    (-119.00, 34.05),   # Malibu
    (-119.27, 34.27),   # Point Mugu
    (-119.43, 34.35),   # Ventura
    (-119.69, 34.41),   # Santa Barbara
    (-120.47, 34.45),   # Point Conception
    (-120.65, 34.68),   # Lompoc
    (-120.86, 35.12),   # Pismo Beach
    (-120.87, 35.37),   # Morro Bay
    (-121.17, 35.64),   # San Simeon
    (-121.80, 36.30),   # Big Sur
    (-121.90, 36.62),   # Monterey
    (-122.00, 36.95),   # Santa Cruz
    (-122.38, 37.50),   # Half Moon Bay
    (-122.51, 37.76),   # San Francisco
    (-122.68, 37.90),   # Bolinas
    (-122.96, 38.31),   # Point Reyes
    (-123.02, 38.44),   # Bodega Bay
    (-123.35, 38.91),   # Fort Ross
    (-123.80, 39.43),   # Point Arena
    (-123.82, 39.75),   # Mendocino
    (-124.08, 40.34),   # Cape Mendocino
    (-124.16, 40.80),   # Eureka
    (-124.20, 41.46),   # Trinidad
    (-124.20, 41.76),   # Crescent City
]

# Channel Islands approximate outlines (for context)
CHANNEL_ISLANDS = {
    'San Clemente': [(-118.60, 32.80), (-118.35, 33.05), (-118.30, 32.95), (-118.55, 32.75)],
    'Santa Catalina': [(-118.60, 33.30), (-118.30, 33.50), (-118.25, 33.40), (-118.55, 33.25)],
    'San Nicolas': [(-119.60, 33.20), (-119.40, 33.30), (-119.35, 33.25), (-119.55, 33.15)],
    'Santa Barbara': [(-119.45, 33.45), (-119.35, 33.50), (-119.30, 33.45), (-119.40, 33.40)],
    'San Miguel': [(-120.45, 34.00), (-120.30, 34.08), (-120.25, 34.02), (-120.40, 33.95)],
    'Santa Rosa': [(-120.25, 33.90), (-119.95, 34.05), (-119.85, 33.95), (-120.15, 33.85)],
    'Santa Cruz': [(-120.05, 33.95), (-119.55, 34.08), (-119.50, 34.00), (-119.95, 33.90)],
    'Anacapa': [(-119.45, 34.00), (-119.35, 34.02), (-119.32, 34.00), (-119.42, 33.98)],
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
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # === Left plot: Wave Height with Coverage ===
    ax1 = axes[0]

    # Plot wave height
    cmap = create_wave_colormap()
    wave_masked = np.ma.masked_invalid(wave_height)

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

    # Overlay land/NaN areas with gray
    land_mask = np.isnan(wave_height).astype(float)
    ax1.contourf(
        lon_grid, lat_grid, land_mask,
        levels=[0.5, 1.5],
        colors=['#d3d3d3'],
        alpha=0.8
    )

    # Plot WW3 boundary (nearest valid data to coast)
    if boundary_lons:
        ax1.plot(
            boundary_lons, boundary_lats,
            color='red', linewidth=2.5, linestyle='-',
            label='WW3 Valid Data Boundary',
            path_effects=[pe.Stroke(linewidth=4, foreground='white'), pe.Normal()]
        )

    # Plot California coastline
    coast_lons, coast_lats = zip(*CA_COASTLINE)
    ax1.plot(
        coast_lons, coast_lats,
        color='black', linewidth=1.5,
        label='California Coastline',
        path_effects=[pe.Stroke(linewidth=3, foreground='white'), pe.Normal()]
    )

    # Plot Channel Islands
    for island_name, points in CHANNEL_ISLANDS.items():
        island_lons, island_lats = zip(*points)
        ax1.fill(island_lons, island_lats, color='#808080', alpha=0.7)

    # Add grid showing WW3 resolution
    for lon in lons[::4]:  # Every 4th longitude (1 degree)
        ax1.axvline(x=lon, color='gray', alpha=0.2, linewidth=0.5)
    for lat in lats[::4]:  # Every 4th latitude
        ax1.axhline(y=lat, color='gray', alpha=0.2, linewidth=0.5)

    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.set_title(f'WaveWatch III Coverage - {region_name}\nWave Height & Valid Data Boundary', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.set_aspect('equal')

    # === Right plot: Distance to Coast Analysis ===
    ax2 = axes[1]

    # Calculate approximate distance from WW3 boundary to coast
    # Using simple lon difference (rough approximation)
    distances_km = []
    for i, (b_lon, b_lat) in enumerate(zip(boundary_lons, boundary_lats)):
        # Find nearest coastline point
        min_dist = float('inf')
        for c_lon, c_lat in CA_COASTLINE:
            # Rough distance calculation (1 degree lon ≈ 85-111 km depending on lat)
            km_per_deg_lon = 111 * np.cos(np.radians(b_lat))
            km_per_deg_lat = 111
            dist = np.sqrt(((b_lon - c_lon) * km_per_deg_lon)**2 + ((b_lat - c_lat) * km_per_deg_lat)**2)
            min_dist = min(min_dist, dist)
        distances_km.append(min_dist)

    # Plot distance vs latitude
    ax2.fill_between(boundary_lats, 0, distances_km, alpha=0.3, color='blue')
    ax2.plot(boundary_lats, distances_km, color='blue', linewidth=2, label='Distance to coast')

    # Add reference lines
    ax2.axhline(y=25, color='red', linestyle='--', alpha=0.7, label='WW3 resolution (~25 km)')
    ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='50 km reference')

    # Mark key locations
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
        if min(boundary_lats) <= lat <= max(boundary_lats):
            idx = np.argmin(np.abs(np.array(boundary_lats) - lat))
            ax2.annotate(
                name,
                (lat, distances_km[idx]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8,
                rotation=45
            )
            ax2.scatter([lat], [distances_km[idx]], color='red', s=30, zorder=5)

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

    return fig, axes


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
