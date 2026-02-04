#!/usr/bin/env python3
"""
Interactive Surfzone Statistics Viewer

View surfzone wave statistics results with wave height coloring and
hover tooltips showing all statistics for each point.

Usage:
    python scripts/dev/view_statistics.py --region socal
    python scripts/dev/view_statistics.py --region norcal --lonlat
    python scripts/dev/view_statistics.py --list-regions
"""

import argparse
import json
import sys
from pathlib import Path
import io
import base64
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import holoviews as hv
from holoviews import streams
from holoviews.operation.datashader import datashade, spread
import datashader as ds
import panel as pn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree

hv.extension('bokeh')
pn.extension()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.regions.region import REGIONS
from data.surfzone.statistics import StatisticsResult

# Partition file names (same order as run_statistics.py)
PARTITION_FILES = ['wind_sea', 'primary_swell', 'secondary_swell', 'tertiary_swell']


def list_regions_with_statistics():
    """List regions with their statistics status."""
    print("\nRegions with statistics results:")
    print("-" * 70)

    for name in ['socal', 'central', 'norcal']:
        region = REGIONS[name]

        # Check for statistics
        output_dir = project_root / "data" / "surfzone" / "output" / name
        has_stats = output_dir.exists() and (output_dir / "statistics_latest.csv").exists()

        # Check for partition files
        partitions_found = []
        if output_dir.exists():
            for pname in PARTITION_FILES:
                if (output_dir / f"{pname}.npz").exists():
                    partitions_found.append(pname)

        print(f"  {name:12} - {region.display_name}")
        print(f"               Statistics: {'Yes' if has_stats else 'No'}")
        print(f"               Partitions: {', '.join(partitions_found) if partitions_found else 'None'}")
        print()


def load_statistics(region_name: str) -> StatisticsResult:
    """Load statistics result for a region."""
    output_dir = project_root / "data" / "surfzone" / "output" / region_name
    stats_path = output_dir / "statistics_latest.csv"

    if not stats_path.exists():
        raise FileNotFoundError(
            f"No statistics found for region '{region_name}'. "
            f"Run: python data/surfzone/statistics/run_statistics.py --region {region_name}"
        )

    return StatisticsResult.load(output_dir)


def load_combined_wave_height(region_name: str) -> np.ndarray:
    """
    Load and combine wave heights from all partition files.

    Returns combined Hs = sqrt(sum(Hs_i^2)) for all partitions.
    """
    output_dir = project_root / "data" / "surfzone" / "output" / region_name

    # Accumulate squared Hs values
    hs_squared_sum = None
    n_points = None

    for pname in PARTITION_FILES:
        npz_path = output_dir / f"{pname}.npz"
        if not npz_path.exists():
            continue

        data = np.load(npz_path, allow_pickle=True)

        # Get wave height at mesh points (after propagation)
        if 'H_at_mesh' in data:
            hs = data['H_at_mesh']
            converged = data['converged']

            if n_points is None:
                n_points = len(hs)
                hs_squared_sum = np.zeros(n_points)

            # Only add converged points
            valid_hs = np.where(converged, hs, 0)
            valid_hs = np.nan_to_num(valid_hs, nan=0.0)
            hs_squared_sum += valid_hs**2

    if hs_squared_sum is None:
        raise FileNotFoundError(
            f"No partition files found for region '{region_name}'. "
            f"Run surfzone simulation first."
        )

    # Combined Hs (energy addition)
    combined_hs = np.sqrt(hs_squared_sum)
    return combined_hs


def load_mesh(region_name: str):
    """Load a surf zone mesh."""
    from data.surfzone.mesh import SurfZoneMesh

    mesh_dir = project_root / "data" / "surfzone" / "meshes" / region_name
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Mesh not found at {mesh_dir}")

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
    for lbl in cb.ax.get_yticklabels():
        lbl.set_color('white')

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                facecolor='#1a1a2e', edgecolor='none')
    plt.close(fig)
    buf.seek(0)

    # Convert to base64 for Panel
    img_data = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_data}" />'


def format_stat_value(value, name: str) -> str:
    """Format a statistic value for display."""
    if pd.isna(value) or np.isnan(value):
        return "N/A"

    # Format based on statistic type
    if 'period' in name or 'duration' in name:
        return f"{value:.1f} s"
    elif 'wavelength' in name:
        return f"{value:.1f} m"
    elif 'steepness' in name:
        return f"{value:.4f}"
    elif name == 'waves_per_set':
        return f"{value:.1f}"
    elif 'factor' in name or 'amplification' in name:
        return f"{value:.3f}"
    else:
        return f"{value:.2f}"


def view_statistics(
    region_name: str,
    use_lonlat: bool = False,
    h_max: float = None,
):
    """Create interactive statistics viewer with hover tooltips."""

    print(f"Loading data for {region_name}...")

    # Load mesh for coordinate conversion and coastlines
    mesh = load_mesh(region_name)
    print(f"  Mesh: {len(mesh.points_x):,} points")

    # Load statistics
    stats = load_statistics(region_name)
    print(f"  Statistics: {stats.num_points:,} points")

    # Load combined wave height
    combined_hs = load_combined_wave_height(region_name)
    print(f"  Combined Hs loaded: {np.sum(combined_hs > 0):,} points with waves")

    # Get coordinates from statistics (lat/lon)
    stats_df = stats.df.copy()
    lats = stats_df['lat'].values
    lons = stats_df['lon'].values

    # Convert to UTM for display if not using lonlat
    if use_lonlat:
        x_coords = lons
        y_coords = lats
        x_label = "Longitude"
        y_label = "Latitude"
    else:
        # Convert from lon/lat to UTM
        x_coords, y_coords = mesh.lon_lat_to_utm(lons, lats)
        x_label = "UTM Easting (m)"
        y_label = "UTM Northing (m)"

    # Add combined Hs to dataframe
    stats_df['combined_hs'] = combined_hs
    stats_df['x'] = x_coords
    stats_df['y'] = y_coords

    # Auto-determine wave height max
    valid_hs = combined_hs[combined_hs > 0]
    if h_max is None:
        if len(valid_hs) > 0:
            h_max = float(np.nanpercentile(valid_hs, 98))
            h_max = max(h_max, 0.5)  # At least 0.5m
        else:
            h_max = 2.0

    print(f"  Wave height color scale: 0 - {h_max:.1f}m")

    # Get statistic columns (exclude base columns)
    base_cols = ['point_id', 'lat', 'lon', 'depth', 'combined_hs', 'x', 'y']
    stat_cols = [c for c in stats_df.columns if c not in base_cols]
    print(f"  Statistics: {', '.join(stat_cols)}")

    # === Create visualization layers ===

    # Split points by whether they have wave data
    has_waves = combined_hs > 0
    no_waves = ~has_waves

    # Layer 1: Points without waves (gray)
    if np.sum(no_waves) > 0:
        no_wave_df = pd.DataFrame({
            'x': x_coords[no_waves],
            'y': y_coords[no_waves],
            'val': np.ones(np.sum(no_waves)),
        })
        no_wave_points = hv.Points(no_wave_df, kdims=['x', 'y'], vdims=['val'])
        no_wave_cmap = ['#444455', '#444455']
    else:
        no_wave_points = None

    # Layer 2: Points with waves - colored by combined Hs
    wave_df = pd.DataFrame({
        'x': x_coords[has_waves],
        'y': y_coords[has_waves],
        'H': np.clip(combined_hs[has_waves], 0, h_max),
    })
    wave_points = hv.Points(wave_df, kdims=['x', 'y'], vdims=['H'])
    wave_cmap = ['#0044aa', '#0066cc', '#0088ee', '#00aaff', '#00cccc',
                 '#00ee88', '#44ff44', '#aaff00', '#ffff00']

    # Create datashaded plots
    plot = None

    if no_wave_points is not None:
        no_wave_shaded = spread(
            datashade(
                no_wave_points,
                aggregator=ds.mean('val'),
                cmap=no_wave_cmap,
            ),
            px=2,
        )
        plot = no_wave_shaded

    wave_shaded = spread(
        datashade(
            wave_points,
            aggregator=ds.mean('H'),
            cmap=wave_cmap,
            cnorm='linear',
        ),
        px=4,
    )

    if plot is None:
        plot = wave_shaded
    else:
        plot = plot * wave_shaded

    # Coastline overlay
    if mesh.coastlines:
        coastline_paths = []
        for coastline in mesh.coastlines:
            if use_lonlat:
                cl_x, cl_y = mesh.utm_to_lon_lat(coastline[:, 0], coastline[:, 1])
            else:
                cl_x, cl_y = coastline[:, 0], coastline[:, 1]
            coastline_paths.append(np.column_stack([cl_x, cl_y]))

        coastline_overlay = hv.Path(coastline_paths).opts(
            color='#ff00ff',
            line_width=2,
        )
        plot = plot * coastline_overlay

    # === Create interactive point info panel ===

    # Build KDTree for fast nearest-point lookup
    coords = np.column_stack([x_coords, y_coords])
    tree = cKDTree(coords)

    # Create a Tap stream for click detection
    tap = streams.SingleTap(x=None, y=None)

    # Function to create a marker at clicked location
    def click_marker(x, y):
        if x is None or y is None:
            return hv.Points([]).opts(size=0)
        # Show a marker at clicked location
        return hv.Points([(x, y)]).opts(
            size=15,
            color='red',
            marker='circle',
            line_color='white',
            line_width=2,
        )

    # Create dynamic marker overlay
    marker_dmap = hv.DynamicMap(click_marker, streams=[tap])

    # Combine plot with marker
    plot = plot * marker_dmap

    # Apply plot options
    plot = plot.opts(
        width=1200,
        height=800,
        xlabel=x_label,
        ylabel=y_label,
        tools=['wheel_zoom', 'pan', 'reset', 'box_zoom', 'tap'],
        active_tools=['wheel_zoom', 'pan'],
        bgcolor='#1a1a2e',
        data_aspect=1,
    )

    # Panel for displaying point info - using a reactive approach
    def get_point_info_html(x, y):
        """Generate HTML for point info based on click coordinates."""
        if x is None or y is None:
            return "<div style='color: white; padding: 10px;'><b>Click on a point to see statistics</b></div>"

        # Find nearest point
        dist, idx = tree.query([x, y])

        # Get data for this point
        row = stats_df.iloc[idx]

        # Build HTML content
        html = f"""
        <div style="color: white; font-size: 11px; padding: 10px; background: #2a2a3e; border-radius: 5px; max-height: 700px; overflow-y: auto;">
            <b style="font-size: 13px;">Point Statistics</b><br>
            <hr style="border-color: #444;">

            <b>Location</b><br>
            Lat: {row['lat']:.5f}<br>
            Lon: {row['lon']:.5f}<br>
            Depth: {row['depth']:.1f} m<br>
            <br>

            <b>Combined Wave Height</b><br>
            Hs: {row['combined_hs']:.2f} m<br>
            <hr style="border-color: #444;">

            <b>Wave Statistics</b><br>
        """

        for col in stat_cols:
            val = row[col]
            formatted = format_stat_value(val, col)
            # Make column name more readable
            display_name = col.replace('_', ' ').title()
            html += f"{display_name}: {formatted}<br>"

        html += f"""
            <hr style="border-color: #444;">
            <span style="color: #888; font-size: 10px;">Point ID: {int(row['point_id'])}</span>
        </div>
        """

        return html

    # Create reactive HTML pane that updates when tap coordinates change
    point_info_pane = pn.pane.HTML(
        pn.bind(get_point_info_html, tap.param.x, tap.param.y),
        width=280,
        sizing_mode='stretch_height',
    )

    # Create colorbar
    colorbar_html = create_matplotlib_colorbar(
        0, h_max, 'Combined Hs (m)', wave_cmap, height=300
    )

    # Legend HTML
    legend_html = """
    <div style="color: white; font-size: 12px; padding: 10px; background: #1a1a2e;">
        <b>Legend</b><br><br>
        <span style="color: #444455;">●</span> No wave data<br>
        <span style="color: #00cccc;">●</span> Wave height (see colorbar)<br>
        <span style="color: #ff00ff;">—</span> Coastline<br>
        <br>
        <b>Interaction</b><br>
        Click on a point to see<br>
        all statistics for that location.
    </div>
    """

    # Summary stats HTML
    summary_html = f"""
    <div style="color: white; font-size: 11px; padding: 10px; background: #2a2a3e; border-radius: 5px;">
        <b>Summary</b><br><br>
        Region: {region_name}<br>
        Points: {stats.num_points:,}<br>
        With waves: {np.sum(has_waves):,}<br>
        <br>
        <b>Combined Hs Range</b><br>
        Min: {np.nanmin(valid_hs):.2f}m<br>
        Max: {np.nanmax(valid_hs):.2f}m<br>
        Mean: {np.nanmean(valid_hs):.2f}m<br>
    </div>
    """

    # Layout
    sidebar = pn.Column(
        pn.pane.HTML(legend_html, width=180),
        pn.Spacer(height=10),
        pn.pane.HTML(colorbar_html, width=120, height=350),
        pn.Spacer(height=10),
        pn.pane.HTML(summary_html, width=180),
        width=200,
    )

    info_panel = pn.Column(
        pn.pane.Markdown("### Point Details"),
        point_info_pane,
        width=300,
    )

    title = pn.pane.Markdown(
        f"# Surfzone Statistics: {region_name}",
        sizing_mode='stretch_width',
    )

    main_row = pn.Row(
        pn.pane.HoloViews(plot, sizing_mode='fixed'),
        sidebar,
        info_panel,
    )

    layout = pn.Column(
        title,
        main_row,
        sizing_mode='stretch_width',
    )

    print("\nOpening in browser...")
    print("Click on any point to see detailed statistics.")
    pn.serve(layout, show=True, title=f'Surfzone Statistics: {region_name}')


def main():
    parser = argparse.ArgumentParser(
        description="Interactive surfzone statistics viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/dev/view_statistics.py --region socal
    python scripts/dev/view_statistics.py --region norcal --lonlat
    python scripts/dev/view_statistics.py --list-regions
        """,
    )

    parser.add_argument(
        '--region', '-r',
        type=str,
        choices=['socal', 'norcal', 'central'],
        help="Region to view"
    )

    parser.add_argument(
        '--list-regions',
        action='store_true',
        help="List regions with available statistics"
    )

    parser.add_argument(
        '--h-max',
        type=float,
        default=None,
        help="Maximum wave height for color scale (default: auto)"
    )

    parser.add_argument(
        '--lonlat',
        action='store_true',
        help="Use longitude/latitude coordinates instead of UTM"
    )

    args = parser.parse_args()

    if args.list_regions:
        list_regions_with_statistics()
        return

    if not args.region:
        parser.print_help()
        print("\nError: --region is required (unless using --list-regions)")
        sys.exit(1)

    try:
        view_statistics(
            region_name=args.region,
            use_lonlat=args.lonlat,
            h_max=args.h_max,
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
