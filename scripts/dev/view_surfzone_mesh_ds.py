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
import io
import base64

import numpy as np
import pandas as pd
import holoviews as hv
from holoviews.operation.datashader import datashade, spread
import datashader as ds
import panel as pn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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


def create_matplotlib_colorbar(vmin, vmax, label, cmap_colors, height=400):
    """Create a colorbar image using matplotlib."""
    fig, ax = plt.subplots(figsize=(1.2, height/100), dpi=100)

    # Create custom colormap
    cmap = mcolors.LinearSegmentedColormap.from_list('custom', cmap_colors, N=256)

    # Create colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='vertical'
    )
    cb.set_label(label, fontsize=10)
    cb.ax.tick_params(labelsize=8)

    # Style for dark theme
    ax.tick_params(colors='white')
    cb.ax.yaxis.set_tick_params(color='white')
    cb.outline.set_edgecolor('white')
    cb.ax.yaxis.label.set_color('white')
    for label in cb.ax.get_yticklabels():
        label.set_color('white')

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                facecolor='#1a1a2e', edgecolor='none')
    plt.close(fig)
    buf.seek(0)

    # Convert to base64 for Panel
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_data}" />'


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

    # Create DataFrames
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

    # Colormaps - vibrant colors that show well on dark background
    # Ocean: cyan/teal for shallow -> deep blue for deep
    ocean_cmap = ['#00ffff', '#00e5e5', '#00cccc', '#00b3b3', '#0099cc', '#0080b3', '#006699', '#004d80', '#003366']
    # Land: yellow/gold for low -> orange/red for high
    land_cmap = ['#ffff00', '#ffdd00', '#ffbb00', '#ff9900', '#ff7700', '#ff5500', '#e64400', '#cc3300', '#aa2200']

    # Prepare coastline segments for overlay
    coastline_paths = []
    if mesh.coastlines:
        print(f"  Coastline segments: {len(mesh.coastlines)}")
        for coastline in mesh.coastlines:
            if use_lonlat:
                cl_x, cl_y = mesh.utm_to_lon_lat(coastline[:, 0], coastline[:, 1])
            else:
                cl_x, cl_y = coastline[:, 0], coastline[:, 1]
            coastline_paths.append(np.column_stack([cl_x, cl_y]))

    # Create datashaded plots
    ocean_shaded = spread(
        datashade(
            ocean_points,
            aggregator=ds.mean('depth'),
            cmap=ocean_cmap,
            cnorm='linear',
        ),
        px=3,
    )

    land_shaded = spread(
        datashade(
            land_points,
            aggregator=ds.mean('height'),
            cmap=land_cmap,
            cnorm='linear',
        ),
        px=3,
    )

    # Create coastline overlay (magenta, stands out against cyan/yellow)
    if coastline_paths:
        coastline_overlay = hv.Path(coastline_paths).opts(
            color='#ff00ff',  # Magenta
            line_width=2,
        )
        plot = (ocean_shaded * land_shaded * coastline_overlay).opts(
            width=1400,
            height=900,
            xlabel=x_label,
            ylabel=y_label,
            tools=['wheel_zoom', 'pan', 'reset', 'box_zoom'],
            active_tools=['wheel_zoom', 'pan'],
            bgcolor='#1a1a2e',
            aspect='equal',
        )
    else:
        plot = (ocean_shaded * land_shaded).opts(
            width=1400,
            height=900,
            xlabel=x_label,
            ylabel=y_label,
            tools=['wheel_zoom', 'pan', 'reset', 'box_zoom'],
            active_tools=['wheel_zoom', 'pan'],
            bgcolor='#1a1a2e',
            aspect='equal',
        )

    # Create matplotlib colorbars as images
    ocean_colorbar_html = create_matplotlib_colorbar(
        0, depth_max, 'Ocean Depth (m)', ocean_cmap, height=300
    )
    land_colorbar_html = create_matplotlib_colorbar(
        0, land_max, 'Land Elevation (m)', land_cmap, height=150
    )

    # Layout with colorbars
    colorbar_col = pn.Column(
        pn.pane.HTML(ocean_colorbar_html, width=120, height=350),
        pn.Spacer(height=20),
        pn.pane.HTML(land_colorbar_html, width=120, height=200),
        width=140,
    )

    title = pn.pane.Markdown(
        f"# Surf Zone Mesh: {region_name.upper()} ({n_points:,} points)",
        sizing_mode='stretch_width',
    )

    main_row = pn.Row(
        pn.pane.HoloViews(plot, sizing_mode='fixed'),
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
