#!/usr/bin/env python3
"""
Plot California Subregions on GEBCO Bathymetry

Visualizes the three California subregions (NorCal, Central, SoCal)
overlaid on GEBCO bathymetry data.

Usage:
    python scripts/plot_regions.py
    python scripts/plot_regions.py --save output.png
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from data.bathymetry.gebco import GEBCOBathymetry
from data.regions import CALIFORNIA, CA_SUBREGIONS


def plot_regions(
    save_path: Path = None,
    show: bool = True,
    depth_max: float = 4000,
    figsize: tuple = (14, 12),
) -> None:
    """
    Plot GEBCO bathymetry with California subregion boundaries.

    Args:
        save_path: Path to save the figure (optional)
        show: Whether to display the plot
        depth_max: Maximum depth for colorbar
        figsize: Figure size (width, height)
    """
    # Load GEBCO data
    print("Loading GEBCO bathymetry...")
    gebco = GEBCOBathymetry()

    # Get California bounds for the view
    lat_range = CALIFORNIA.lat_range
    lon_range = CALIFORNIA.lon_range

    # Subset data to California
    lat_mask = (gebco.lats >= lat_range[0]) & (gebco.lats <= lat_range[1])
    lon_mask = (gebco.lons >= lon_range[0]) & (gebco.lons <= lon_range[1])

    depth_subset = gebco.depth[np.ix_(lat_mask, lon_mask)]
    lats_subset = gebco.lats[lat_mask]
    lons_subset = gebco.lons[lon_mask]

    LON, LAT = np.meshgrid(lons_subset, lats_subset)

    # Create figure
    print("Creating plot...")
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    # Plot bathymetry
    im = ax.pcolormesh(
        LON, LAT, depth_subset,
        cmap='Blues',
        vmin=0, vmax=depth_max,
        shading='auto',
        transform=ccrs.PlateCarree()
    )

    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, edgecolor='black')
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                  crs=ccrs.PlateCarree())

    # Gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # Plot subregion boundaries
    print("Adding subregion boundaries...")
    for region in CA_SUBREGIONS:
        region.plot_bounds(
            ax,
            linewidth=3.0,
            fill=True,
            alpha=0.15,
            label=True,
            transform=ccrs.PlateCarree()
        )
        print(f"  Added: {region.display_name} - {region.bounds}")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
    cbar.set_label('Depth (m)', fontsize=12)

    # Title
    ax.set_title(
        'California Coast - SWAN Modeling Subregions\n'
        'GEBCO 2024 Bathymetry',
        fontsize=14,
        fontweight='bold'
    )

    # Add legend for regions
    legend_elements = []
    for region in CA_SUBREGIONS:
        from matplotlib.patches import Patch
        legend_elements.append(
            Patch(facecolor=region.color, alpha=0.3,
                  edgecolor=region.color, linewidth=2,
                  label=region.display_name)
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()

    # Save
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    # Show
    if show:
        plt.show()
    else:
        plt.close()

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description='Plot California subregions on GEBCO bathymetry'
    )
    parser.add_argument(
        '--save', '-s',
        type=str,
        help='Path to save the figure (e.g., output.png)'
    )
    parser.add_argument(
        '--no-show',
        action='store_true',
        help='Do not display the plot (useful for saving only)'
    )
    parser.add_argument(
        '--depth-max',
        type=float,
        default=4000,
        help='Maximum depth for colorbar (default: 4000)'
    )

    args = parser.parse_args()

    plot_regions(
        save_path=args.save,
        show=not args.no_show,
        depth_max=args.depth_max,
    )


if __name__ == "__main__":
    main()
