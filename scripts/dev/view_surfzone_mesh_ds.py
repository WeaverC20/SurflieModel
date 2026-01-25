#!/usr/bin/env python3
"""
Interactive Surf Zone Mesh Viewer (Datashader)

View surf zone meshes with all points using datashader for fast rendering.
Handles millions of points with smooth zoom/pan.

Usage:
    python scripts/dev/view_surfzone_mesh_ds.py socal
    python scripts/dev/view_surfzone_mesh_ds.py socal --lonlat
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import holoviews as hv
from holoviews.operation.datashader import datashade, spread
import datashader as ds
import panel as pn

hv.extension('bokeh')
pn.extension()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_mesh(region_name: str):
    """Load a surf zone mesh."""
    from data.surfzone.mesh import SurfZoneMesh

    mesh_dir = project_root / "data" / "surfzone" / "meshes" / region_name
    if not mesh_dir.exists():
        print(f"Error: Mesh not found at {mesh_dir}")
        print(f"Generate it first with: python scripts/generate_surfzone_mesh.py {region_name}")
        sys.exit(1)

    return SurfZoneMesh.load(mesh_dir)


def create_colorbar(cmap, vmin, vmax, title, width=80, height=300):
    """Create a vertical colorbar using HoloViews."""
    gradient = np.linspace(vmax, vmin, 100).reshape(-1, 1)

    img = hv.Image(
        gradient,
        bounds=(0, vmin, 1, vmax),
        kdims=['x', 'y'],
    ).opts(
        cmap=cmap,
        xaxis=None,
        yaxis='right',
        ylabel=title,
        width=width,
        height=height,
        toolbar=None,
        default_tools=[],
    )

    return img


def view_mesh(
    region_name: str,
    depth_max: float = 30.0,
    land_max: float = 5.0,
    use_lonlat: bool = False,
):
    """Create interactive mesh viewer using datashader."""

    print(f"Loading mesh for {region_name}...")
    mesh = load_mesh(region_name)
    print(mesh.summary())
    print()

    # Get point cloud data
    x = mesh.points_x.copy()
    y = mesh.points_y.copy()
    elevation = mesh.elevation.copy()

    n_points = len(x)
    print(f"Preparing {n_points:,} points for visualization...")

    # Convert to lon/lat if requested
    if use_lonlat:
        x, y = mesh.utm_to_lon_lat(x, y)
        x_label = "Longitude"
        y_label = "Latitude"
    else:
        x_label = "UTM Easting (m)"
        y_label = "UTM Northing (m)"

    # Separate ocean and land
    ocean_mask = elevation < 0
    land_mask = elevation >= 0

    n_ocean = np.sum(ocean_mask)
    n_land = np.sum(land_mask)
    print(f"  Ocean points: {n_ocean:,}")
    print(f"  Land points: {n_land:,}")

    # Get actual depth range for colorbar
    depths = -elevation[ocean_mask]
    actual_depth_max = min(depths.max(), depth_max)

    # Get actual land range
    heights = elevation[land_mask]
    actual_land_max = min(heights.max(), land_max) if len(heights) > 0 else land_max

    # Create DataFrames for ocean and land
    ocean_df = pd.DataFrame({
        'x': x[ocean_mask],
        'y': y[ocean_mask],
        'depth': np.clip(-elevation[ocean_mask], 0, depth_max),
    })

    land_df = pd.DataFrame({
        'x': x[land_mask],
        'y': y[land_mask],
        'height': np.clip(elevation[land_mask], 0, land_max),
    })

    # Create HoloViews Points objects
    ocean_points = hv.Points(ocean_df, kdims=['x', 'y'], vdims=['depth'])
    land_points = hv.Points(land_df, kdims=['x', 'y'], vdims=['height'])

    # Custom colormaps
    ocean_cmap = ['#e0f7ff', '#b3ecff', '#80dfff', '#4dd2ff', '#1ac6ff', '#00b3e6', '#0099cc', '#007399', '#004d66']
    land_cmap = ['#f5f5dc', '#e6daa6', '#c4b454', '#a89932', '#8b7d32', '#6b5b2a', '#4a3f1d', '#2d2612', '#1a1609']

    # Create datashaded plots with spread for larger points
    ocean_shaded = spread(
        datashade(
            ocean_points,
            aggregator=ds.mean('depth'),
            cmap=ocean_cmap,
            cnorm='linear',
        ),
        px=3,  # Fixed pixel spread
    )

    land_shaded = spread(
        datashade(
            land_points,
            aggregator=ds.mean('height'),
            cmap=land_cmap,
            cnorm='linear',
        ),
        px=3,  # Fixed pixel spread
    )

    # Combine ocean and land - use fixed size with aspect='equal'
    combined = (ocean_shaded * land_shaded).opts(
        width=1400,
        height=900,
        xlabel=x_label,
        ylabel=y_label,
        tools=['wheel_zoom', 'pan', 'reset', 'box_zoom'],
        active_tools=['wheel_zoom', 'pan'],
        bgcolor='#e8e8e8',
        aspect='equal',  # This ensures top-down orthographic view
    )

    # Create colorbars
    ocean_colorbar = create_colorbar(
        ocean_cmap, 0, actual_depth_max,
        f'Depth (m)', width=60, height=400
    )

    land_colorbar = create_colorbar(
        land_cmap, 0, actual_land_max,
        f'Land (m)', width=60, height=200
    )

    # Create layout with colorbars on the right
    colorbar_col = pn.Column(
        pn.pane.Markdown("### Ocean Depth"),
        pn.pane.HoloViews(ocean_colorbar, sizing_mode='fixed', width=100, height=420),
        pn.Spacer(height=20),
        pn.pane.Markdown("### Land Elevation"),
        pn.pane.HoloViews(land_colorbar, sizing_mode='fixed', width=100, height=220),
        width=140,
    )

    # Title
    title = pn.pane.Markdown(
        f"# Surf Zone Mesh: {region_name.upper()} ({n_points:,} points)",
        sizing_mode='stretch_width',
    )

    # Main layout
    main_row = pn.Row(
        pn.pane.HoloViews(combined, sizing_mode='fixed'),
        colorbar_col,
    )

    layout = pn.Column(
        title,
        main_row,
        sizing_mode='stretch_width',
    )

    print("Opening in browser...")
    pn.serve(layout, show=True, title=f'Surf Zone Mesh: {region_name}')


def main():
    parser = argparse.ArgumentParser(
        description="Interactive surf zone mesh viewer (datashader)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        'region',
        help="Region name (e.g., 'socal')"
    )

    parser.add_argument(
        '--depth-max',
        type=float,
        default=30.0,
        help="Maximum depth for color scale (default: 30)"
    )

    parser.add_argument(
        '--land-max',
        type=float,
        default=5.0,
        help="Maximum land elevation for color scale (default: 5)"
    )

    parser.add_argument(
        '--lonlat',
        action='store_true',
        help="Use longitude/latitude coordinates instead of UTM"
    )

    args = parser.parse_args()

    view_mesh(
        args.region,
        depth_max=args.depth_max,
        land_max=args.land_max,
        use_lonlat=args.lonlat,
    )


if __name__ == "__main__":
    main()
