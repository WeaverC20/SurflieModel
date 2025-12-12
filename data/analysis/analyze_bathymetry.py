#!/usr/bin/env python3
"""
California Bathymetry Analysis

Loads USACE NCMP Topobathy Lidar DEM tiles and creates depth heatmap visualizations.
Supports viewing individual tiles, merging multiple tiles, or subsetting by region.

Usage:
    python analyze_bathymetry.py                    # Plot all tiles overview
    python analyze_bathymetry.py --tile 01         # Plot single tile
    python analyze_bathymetry.py --region socal    # Plot Southern California
    python analyze_bathymetry.py --bounds 33.5 34.0 -118.5 -117.5  # Custom bounds

Downsample Option (-d / --downsample):
    Use this for faster plotting of multi-tile regions. The value specifies
    the resolution reduction factor (e.g., -d 20 means 20m resolution instead of 1m).

    Examples:
        python analyze_bathymetry.py --region san_diego -d 20   # Fast, 20m resolution
        python analyze_bathymetry.py --region socal -d 50       # Very fast, 50m resolution
        python analyze_bathymetry.py --region huntington -d 1   # Full 1m resolution (slow)

    Recommended values:
        -d 1   : 1m  (original) - Very slow, use for small areas or detailed analysis
        -d 10  : 10m - Fast, good detail
        -d 20  : 20m - Very fast, recommended for most region plots
        -d 50  : 50m - Instant, good for large area overviews

Available Regions:
    socal, la, san_diego, huntington, malibu, santa_barbara
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = Path(__file__).parent.parent / "raw" / "bathymetry" / "USACE_CA_DEM_2009_9488"
FIGURES_DIR = Path(__file__).parent / "figures"

# Predefined regions (min_lat, max_lat, min_lon, max_lon)
REGIONS = {
    'socal': (32.5, 34.5, -120.5, -117.0),      # Southern California
    'la': (33.7, 34.1, -118.6, -118.1),          # Los Angeles area
    'san_diego': (32.5, 33.0, -117.4, -117.0),   # San Diego
    'huntington': (33.62, 33.68, -118.02, -117.95),  # Huntington Beach
    'malibu': (33.95, 34.05, -118.85, -118.65),  # Malibu
    'santa_barbara': (34.35, 34.45, -119.75, -119.60),  # Santa Barbara
}
# ============================================================================


def find_tiles_for_bounds(
    data_dir: Path,
    bounds: Tuple[float, float, float, float]
) -> List[Path]:
    """
    Find tiles that intersect with given bounds.

    Args:
        data_dir: Directory containing TIF files
        bounds: (min_lat, max_lat, min_lon, max_lon)

    Returns:
        List of paths to intersecting tiles
    """
    min_lat, max_lat, min_lon, max_lon = bounds
    matching_tiles = []

    for tif_path in sorted(data_dir.glob("*.tif")):
        try:
            with rasterio.open(tif_path) as src:
                tile_bounds = src.bounds
                # Check if tile intersects with requested bounds
                if (tile_bounds.left <= max_lon and
                    tile_bounds.right >= min_lon and
                    tile_bounds.bottom <= max_lat and
                    tile_bounds.top >= min_lat):
                    matching_tiles.append(tif_path)
        except Exception as e:
            print(f"  Warning: Could not read {tif_path.name}: {e}")

    return matching_tiles


def load_and_merge_tiles(
    tile_paths: List[Path],
    downsample: int = 1
) -> Tuple[np.ndarray, dict]:
    """
    Load and merge multiple tiles into a single array.

    Args:
        tile_paths: List of paths to TIF files
        downsample: Downsample factor (e.g., 10 = read every 10th pixel)

    Returns:
        Tuple of (merged array, metadata dict)
    """
    if not tile_paths:
        raise ValueError("No tiles provided")

    if len(tile_paths) == 1:
        # Single tile - just load it
        with rasterio.open(tile_paths[0]) as src:
            if downsample > 1:
                # Read at reduced resolution
                data = src.read(
                    1,
                    out_shape=(src.height // downsample, src.width // downsample),
                    resampling=Resampling.average
                )
                # Adjust transform for downsampled data
                transform = src.transform * src.transform.scale(downsample, downsample)
            else:
                data = src.read(1)
                transform = src.transform

            meta = {
                'bounds': src.bounds,
                'crs': src.crs,
                'nodata': src.nodata,
                'transform': transform,
                'resolution': (src.res[0] * downsample, src.res[1] * downsample)
            }
            return data, meta

    # Multiple tiles - load each at reduced resolution, then combine
    print(f"  Loading {len(tile_paths)} tiles (downsample={downsample}x)...")

    all_data = []
    all_bounds = []
    nodata = None
    crs = None

    for i, path in enumerate(tile_paths):
        with rasterio.open(path) as src:
            if nodata is None:
                nodata = src.nodata
                crs = src.crs

            if downsample > 1:
                data = src.read(
                    1,
                    out_shape=(src.height // downsample, src.width // downsample),
                    resampling=Resampling.average
                )
            else:
                data = src.read(1)

            all_data.append({
                'data': data,
                'bounds': src.bounds,
                'shape': data.shape
            })
            all_bounds.append(src.bounds)

        if (i + 1) % 10 == 0:
            print(f"    Loaded {i + 1}/{len(tile_paths)} tiles...")

    # Calculate merged bounds
    min_lon = min(b.left for b in all_bounds)
    max_lon = max(b.right for b in all_bounds)
    min_lat = min(b.bottom for b in all_bounds)
    max_lat = max(b.top for b in all_bounds)

    # Calculate pixel size (approximate, assuming uniform resolution)
    with rasterio.open(tile_paths[0]) as src:
        pixel_size_x = src.res[0] * downsample
        pixel_size_y = src.res[1] * downsample

    # Create output array
    out_width = int((max_lon - min_lon) / pixel_size_x) + 1
    out_height = int((max_lat - min_lat) / pixel_size_y) + 1

    print(f"  Creating merged array: {out_height} x {out_width} pixels")
    merged = np.full((out_height, out_width), nodata, dtype=np.float32)

    # Place each tile's data into the merged array
    for tile_info in all_data:
        data = tile_info['data']
        bounds = tile_info['bounds']

        # Calculate position in output array
        col_start = int((bounds.left - min_lon) / pixel_size_x)
        row_start = int((max_lat - bounds.top) / pixel_size_y)

        # Handle edge cases where tile extends beyond calculated bounds
        row_end = min(row_start + data.shape[0], out_height)
        col_end = min(col_start + data.shape[1], out_width)
        data_rows = row_end - row_start
        data_cols = col_end - col_start

        if row_start >= 0 and col_start >= 0 and data_rows > 0 and data_cols > 0:
            merged[row_start:row_end, col_start:col_end] = data[:data_rows, :data_cols]

    meta = {
        'bounds': (min_lon, min_lat, max_lon, max_lat),
        'crs': crs,
        'nodata': nodata,
        'transform': None,
        'resolution': (pixel_size_x, pixel_size_y)
    }

    return merged, meta


def create_depth_colormap():
    """
    Create a custom colormap for bathymetry visualization.
    Blue (deep) -> Cyan (shallow water) -> Green/Yellow (land)
    """
    colors = [
        (0.0, '#000033'),   # Deep ocean - very dark blue
        (0.2, '#000066'),   # Deep - dark blue
        (0.35, '#0066cc'),  # Mid depth - blue
        (0.45, '#00ccff'),  # Shallow - cyan
        (0.5, '#99ffff'),   # Very shallow - light cyan
        (0.52, '#ffffcc'),  # Beach/intertidal - light yellow
        (0.6, '#99cc66'),   # Low land - green
        (0.8, '#666633'),   # Higher land - brown
        (1.0, '#996633'),   # High land - darker brown
    ]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'bathymetry',
        [(pos, color) for pos, color in colors]
    )
    return cmap


def plot_bathymetry_heatmap(
    data: np.ndarray,
    meta: dict,
    title: str = "California Coastal Bathymetry",
    output_path: Optional[Path] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: bool = True,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Create a heatmap visualization of bathymetry data.

    Args:
        data: 2D array of elevation values
        meta: Metadata dict with bounds, nodata, etc.
        title: Plot title
        output_path: Optional path to save figure
        vmin, vmax: Color scale limits
        show_colorbar: Whether to show colorbar
        figsize: Figure size in inches
    """
    # Mask nodata values
    nodata = meta.get('nodata', -3.4028234663852886e+38)
    masked_data = np.ma.masked_where(
        (data == nodata) | (data < -1000) | np.isnan(data),
        data
    )

    # Calculate bounds for extent
    if isinstance(meta['bounds'], tuple) and len(meta['bounds']) == 4:
        left, bottom, right, top = meta['bounds']
    else:
        bounds = meta['bounds']
        left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top

    # Auto-calculate vmin/vmax if not provided
    if vmin is None:
        vmin = max(np.nanpercentile(masked_data.compressed(), 1), -50)
    if vmax is None:
        vmax = min(np.nanpercentile(masked_data.compressed(), 99), 30)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create custom colormap
    cmap = create_depth_colormap()

    # Normalize around sea level (0)
    # Ensure 0 is at roughly 50% of the colormap
    if vmin < 0 and vmax > 0:
        # Find where 0 should be in the normalized range
        zero_point = -vmin / (vmax - vmin)
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Plot heatmap
    im = ax.imshow(
        masked_data,
        extent=[left, right, bottom, top],
        cmap=cmap,
        norm=norm,
        aspect='auto',
        interpolation='nearest'
    )

    # Add colorbar
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Elevation (m, NAVD88)', fontsize=11)

        # Add reference lines on colorbar
        cbar.ax.axhline(y=0, color='white', linewidth=1.5, linestyle='-')

    # Labels and title
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Format axis labels
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}°'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.2f}°'))

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add data info
    valid_data = masked_data.compressed()
    info_text = (
        f"Depth range: {valid_data.min():.1f}m to {valid_data.max():.1f}m\n"
        f"Resolution: ~1m | Tiles: {meta.get('n_tiles', 1)}"
    )
    ax.text(
        0.02, 0.02, info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()

    # Save or show
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig, ax


def create_overview_plot(data_dir: Path, output_path: Optional[Path] = None):
    """
    Create an overview showing all tile locations.
    """
    print("Creating tile overview...")

    fig, ax = plt.subplots(figsize=(12, 14))

    all_bounds = []
    for tif_path in sorted(data_dir.glob("*.tif")):
        try:
            with rasterio.open(tif_path) as src:
                b = src.bounds
                all_bounds.append((b.left, b.bottom, b.right, b.top, tif_path.stem))
        except:
            continue

    if not all_bounds:
        print("  No tiles found!")
        return

    # Plot each tile as a rectangle
    cmap = plt.cm.viridis
    for i, (left, bottom, right, top, name) in enumerate(all_bounds):
        color = cmap(i / len(all_bounds))
        rect = plt.Rectangle(
            (left, bottom), right - left, top - bottom,
            linewidth=0.5, edgecolor='black', facecolor=color, alpha=0.6
        )
        ax.add_patch(rect)

    # Set axis limits
    all_lons = [b[0] for b in all_bounds] + [b[2] for b in all_bounds]
    all_lats = [b[1] for b in all_bounds] + [b[3] for b in all_bounds]

    ax.set_xlim(min(all_lons) - 0.1, max(all_lons) + 0.1)
    ax.set_ylim(min(all_lats) - 0.1, max(all_lats) + 0.1)

    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(
        f'USACE California Bathymetry Tiles ({len(all_bounds)} tiles)',
        fontsize=14, fontweight='bold'
    )
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description='Analyze California bathymetry data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_bathymetry.py                          # Tile overview
    python analyze_bathymetry.py --tile 01                # Single tile
    python analyze_bathymetry.py --region huntington      # Huntington Beach
    python analyze_bathymetry.py --bounds 33.5 34.0 -118.5 -117.5

Available regions: """ + ', '.join(REGIONS.keys())
    )

    parser.add_argument(
        '--tile', type=str,
        help='Plot specific tile number (e.g., "01", "42")'
    )
    parser.add_argument(
        '--region', type=str, choices=list(REGIONS.keys()),
        help='Plot predefined region'
    )
    parser.add_argument(
        '--bounds', nargs=4, type=float,
        metavar=('MIN_LAT', 'MAX_LAT', 'MIN_LON', 'MAX_LON'),
        help='Custom bounds to plot'
    )
    parser.add_argument(
        '--overview', action='store_true',
        help='Show tile coverage overview'
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help='Do not save figures to disk'
    )
    parser.add_argument(
        '--vmin', type=float, default=None,
        help='Minimum elevation for color scale'
    )
    parser.add_argument(
        '--vmax', type=float, default=None,
        help='Maximum elevation for color scale'
    )
    parser.add_argument(
        '--downsample', '-d', type=int, default=1,
        help='Downsample factor for faster plotting (e.g., 10 = 10m resolution, 50 = 50m)'
    )

    args = parser.parse_args()

    # Check data directory
    if not DATA_DIR.exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        print("\nPlease ensure bathymetry data has been downloaded.")
        sys.exit(1)

    print("=" * 60)
    print("CALIFORNIA BATHYMETRY ANALYSIS")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")

    output_path = None if args.no_save else FIGURES_DIR

    # Handle different modes
    if args.overview or (not args.tile and not args.region and not args.bounds):
        # Show overview
        out = output_path / "bathymetry_tile_overview.png" if output_path else None
        create_overview_plot(DATA_DIR, out)

    if args.tile:
        # Single tile
        print(f"\nLoading tile {args.tile}...")
        tile_pattern = f"*_{args.tile}_*.tif"
        tiles = list(DATA_DIR.glob(tile_pattern))

        if not tiles:
            # Try with leading zeros
            tile_pattern = f"*_CA_{args.tile.zfill(2)}_*.tif"
            tiles = list(DATA_DIR.glob(tile_pattern))

        if not tiles:
            print(f"  Error: No tile found matching '{args.tile}'")
            sys.exit(1)

        print(f"  Found: {tiles[0].name}")
        data, meta = load_and_merge_tiles(tiles, downsample=args.downsample)
        meta['n_tiles'] = 1

        out = output_path / f"bathymetry_tile_{args.tile}.png" if output_path else None
        plot_bathymetry_heatmap(
            data, meta,
            title=f"Bathymetry - Tile {args.tile}",
            output_path=out,
            vmin=args.vmin, vmax=args.vmax
        )

    elif args.region:
        # Predefined region
        bounds = REGIONS[args.region]
        print(f"\nLoading region: {args.region}")
        print(f"  Bounds: {bounds}")

        tiles = find_tiles_for_bounds(DATA_DIR, bounds)
        print(f"  Found {len(tiles)} intersecting tiles")

        if not tiles:
            print("  Error: No tiles found for this region")
            sys.exit(1)

        data, meta = load_and_merge_tiles(tiles, downsample=args.downsample)
        meta['n_tiles'] = len(tiles)

        out = output_path / f"bathymetry_{args.region}.png" if output_path else None
        plot_bathymetry_heatmap(
            data, meta,
            title=f"Bathymetry - {args.region.replace('_', ' ').title()}",
            output_path=out,
            vmin=args.vmin, vmax=args.vmax
        )

    elif args.bounds:
        # Custom bounds
        bounds = tuple(args.bounds)
        print(f"\nLoading custom bounds: {bounds}")

        tiles = find_tiles_for_bounds(DATA_DIR, bounds)
        print(f"  Found {len(tiles)} intersecting tiles")

        if not tiles:
            print("  Error: No tiles found for these bounds")
            sys.exit(1)

        data, meta = load_and_merge_tiles(tiles, downsample=args.downsample)
        meta['n_tiles'] = len(tiles)

        out = output_path / "bathymetry_custom_bounds.png" if output_path else None
        plot_bathymetry_heatmap(
            data, meta,
            title="Bathymetry - Custom Region",
            output_path=out,
            vmin=args.vmin, vmax=args.vmax
        )

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    if output_path:
        print(f"Figures saved to: {output_path}")
    print("\nClose plot windows to exit.")
    plt.show()


if __name__ == '__main__':
    main()
