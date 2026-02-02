#!/usr/bin/env python3
"""
Interactive Surfzone Simulation Result Viewer (Datashader)

View surfzone wave propagation results overlaid on the mesh.
Shows wave heights, convergence status, and depth filtering.

Usage:
    python scripts/dev/view_surfzone_result.py
    python scripts/dev/view_surfzone_result.py --result-file path/to/result.npz
    python scripts/dev/view_surfzone_result.py --lonlat
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


def load_mesh(region_name: str = "socal"):
    """Load a surf zone mesh."""
    from data.surfzone.mesh import SurfZoneMesh

    mesh_dir = project_root / "data" / "surfzone" / "meshes" / region_name
    if not mesh_dir.exists():
        print(f"Error: Mesh not found at {mesh_dir}")
        sys.exit(1)

    return SurfZoneMesh.load(mesh_dir)


def load_result(result_path: Path = None):
    """Load surfzone simulation result."""
    from data.surfzone.runner.output_writer import load_surfzone_result

    if result_path is None:
        # Default to latest result
        result_path = project_root / "data" / "surfzone" / "output" / "primary_swell.npz"

    if not result_path.exists():
        print(f"Error: Result not found at {result_path}")
        print("Run the simulation first: python data/surfzone/runner/run_simulation.py")
        sys.exit(1)

    return load_surfzone_result(result_path)


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


def view_result(
    result_path: Path = None,
    mesh_region: str = "socal",
    use_lonlat: bool = False,
    h_max: float = None,
):
    """Create interactive result viewer using datashader."""

    print(f"Loading mesh for {mesh_region}...")
    mesh = load_mesh(mesh_region)
    print(mesh.summary())
    print()

    print("Loading simulation result...")
    result = load_result(result_path)
    print(result.summary())
    print()

    # Get mesh data
    mesh_x_all = mesh.points_x.copy()
    mesh_y_all = mesh.points_y.copy()
    elevation = mesh.elevation.copy()
    depth_all = np.where(elevation < 0, -elevation, 0)  # Positive depth for water

    n_mesh = len(mesh_x_all)
    print(f"Mesh points: {n_mesh:,}")

    # Identify depth ranges
    depth_range = result.depth_range
    in_range_mask = (depth_all >= depth_range[0]) & (depth_all <= depth_range[1])
    n_in_range = np.sum(in_range_mask)
    print(f"Points in depth range [{depth_range[0]}-{depth_range[1]}m]: {n_in_range:,}")
    print(f"Total filtered points: {result.n_points}")
    print(f"Sampled: {result.n_sampled} ({100*result.sample_rate:.1f}%)")
    print(f"Converged: {result.n_converged} ({100*result.convergence_rate:.1f}% of sampled)")

    # Convert coordinates if needed
    if use_lonlat:
        mesh_x_all, mesh_y_all = mesh.utm_to_lon_lat(mesh_x_all, mesh_y_all)
        result_x, result_y = mesh.utm_to_lon_lat(result.mesh_x.copy(), result.mesh_y.copy())
        x_label = "Longitude"
        y_label = "Latitude"
    else:
        result_x = result.mesh_x.copy()
        result_y = result.mesh_y.copy()
        x_label = "UTM Easting (m)"
        y_label = "UTM Northing (m)"

    # Auto-determine wave height max
    if h_max is None:
        if result.n_converged > 0:
            h_max = float(np.nanpercentile(result.H_at_mesh[result.converged], 98))
            h_max = max(h_max, 0.5)  # At least 0.5m
        else:
            h_max = 2.0

    print(f"Wave height color scale: 0 - {h_max:.1f}m")

    # === Create layers ===

    # Layer 1: Background - all mesh points not in computation range (dark gray)
    out_of_range_mask = ~in_range_mask
    bg_df = pd.DataFrame({
        'x': mesh_x_all[out_of_range_mask],
        'y': mesh_y_all[out_of_range_mask],
        'val': np.ones(np.sum(out_of_range_mask)),  # Dummy value
    })
    bg_points = hv.Points(bg_df, kdims=['x', 'y'], vdims=['val'])
    bg_cmap = ['#333344', '#333344']  # Dark gray

    # Layer 2: Not sampled points (light gray - in depth range but not processed)
    not_sampled_mask = ~result.sampled
    n_not_sampled = np.sum(not_sampled_mask)
    if n_not_sampled > 0:
        ns_df = pd.DataFrame({
            'x': result_x[not_sampled_mask],
            'y': result_y[not_sampled_mask],
            'val': np.ones(n_not_sampled),
        })
        ns_points = hv.Points(ns_df, kdims=['x', 'y'], vdims=['val'])
        ns_cmap = ['#666677', '#666677']  # Light gray
    else:
        ns_points = None

    # Layer 3: Non-converged points (orange/amber - sampled but didn't converge)
    non_converged_mask = result.sampled & ~result.converged
    n_non_converged = np.sum(non_converged_mask)
    if n_non_converged > 0:
        nc_df = pd.DataFrame({
            'x': result_x[non_converged_mask],
            'y': result_y[non_converged_mask],
            'val': np.ones(n_non_converged),
        })
        nc_points = hv.Points(nc_df, kdims=['x', 'y'], vdims=['val'])
        nc_cmap = ['#ff8800', '#ff8800']  # Orange/amber
    else:
        nc_points = None

    # Layer 4: Converged points - colored by wave height
    # Using blue-cyan-green colormap (avoids orange/red used for non-converged)
    converged_mask = result.converged
    if result.n_converged > 0:
        conv_df = pd.DataFrame({
            'x': result_x[converged_mask],
            'y': result_y[converged_mask],
            'H': np.clip(result.H_at_mesh[converged_mask], 0, h_max),
        })
        conv_points = hv.Points(conv_df, kdims=['x', 'y'], vdims=['H'])
        # Wave height colormap: dark blue (low) -> cyan -> green -> yellow (high)
        wave_cmap = ['#0044aa', '#0066cc', '#0088ee', '#00aaff', '#00cccc', '#00ee88', '#44ff44', '#aaff00', '#ffff00']
    else:
        conv_points = None

    # Create datashaded plots
    bg_shaded = spread(
        datashade(
            bg_points,
            aggregator=ds.mean('val'),
            cmap=bg_cmap,
        ),
        px=2,
    )

    # Start combining with * operator (like mesh viewer does)
    plot = bg_shaded

    if ns_points is not None:
        ns_shaded = spread(
            datashade(
                ns_points,
                aggregator=ds.mean('val'),
                cmap=ns_cmap,
            ),
            px=2,
        )
        plot = plot * ns_shaded

    if nc_points is not None:
        nc_shaded = spread(
            datashade(
                nc_points,
                aggregator=ds.mean('val'),
                cmap=nc_cmap,
            ),
            px=3,
        )
        plot = plot * nc_shaded

    if conv_points is not None:
        conv_shaded = spread(
            datashade(
                conv_points,
                aggregator=ds.mean('H'),
                cmap=wave_cmap,
                cnorm='linear',
            ),
            px=4,
        )
        plot = plot * conv_shaded

    # Coastline overlay
    coastline_paths = []
    if mesh.coastlines:
        for coastline in mesh.coastlines:
            if use_lonlat:
                cl_x, cl_y = mesh.utm_to_lon_lat(coastline[:, 0], coastline[:, 1])
            else:
                cl_x, cl_y = coastline[:, 0], coastline[:, 1]
            coastline_paths.append(np.column_stack([cl_x, cl_y]))

    if coastline_paths:
        coastline_overlay = hv.Path(coastline_paths).opts(
            color='#ff00ff',  # Magenta
            line_width=2,
        )
        plot = plot * coastline_overlay

    # Apply final opts (like mesh viewer does)
    plot = plot.opts(
        width=1400,
        height=900,
        xlabel=x_label,
        ylabel=y_label,
        tools=['wheel_zoom', 'pan', 'reset', 'box_zoom'],
        active_tools=['wheel_zoom', 'pan'],
        bgcolor='#1a1a2e',
    )

    # Create colorbars
    wave_colorbar_html = create_matplotlib_colorbar(
        0, h_max, 'Wave Height (m)', wave_cmap if conv_points else ['#888888'], height=300
    )

    # Legend HTML
    legend_html = """
    <div style="color: white; font-size: 12px; padding: 10px; background: #1a1a2e;">
        <b>Legend</b><br><br>
        <span style="color: #333344;">●</span> Outside depth range<br>
        <span style="color: #666677;">●</span> Not sampled (skipped)<br>
        <span style="color: #ff8800;">●</span> Did not converge<br>
        <span style="color: #00cccc;">●</span> Converged (see colorbar)<br>
        <span style="color: #ff00ff;">—</span> Coastline<br>
    </div>
    """

    # Stats HTML
    stats_html = f"""
    <div style="color: white; font-size: 11px; padding: 10px; background: #2a2a3e; border-radius: 5px;">
        <b>Statistics</b><br><br>
        Region: {result.region_name}<br>
        Depth range: {depth_range[0]:.1f} - {depth_range[1]:.1f}m<br>
        Partition: {result.partition_id} (primary swell)<br><br>
        Total mesh: {n_mesh:,}<br>
        In range: {n_in_range:,}<br>
        Filtered: {result.n_points:,}<br>
        Sampled: {result.n_sampled:,} ({100*result.sample_rate:.1f}%)<br>
        Converged: {result.n_converged:,} ({100*result.convergence_rate:.1f}% of sampled)<br><br>
    """
    if result.n_converged > 0:
        H_conv = result.H_at_mesh[result.converged]
        K_conv = result.K_shoaling[result.converged]
        stats_html += f"""
        <b>Wave Height (converged)</b><br>
        Min: {np.nanmin(H_conv):.2f}m<br>
        Max: {np.nanmax(H_conv):.2f}m<br>
        Mean: {np.nanmean(H_conv):.2f}m<br><br>
        <b>Shoaling Coeff</b><br>
        Min: {np.nanmin(K_conv):.2f}<br>
        Max: {np.nanmax(K_conv):.2f}<br>
        Mean: {np.nanmean(K_conv):.2f}<br>
        """
    stats_html += "</div>"

    # Layout (match mesh viewer pattern)
    sidebar = pn.Column(
        pn.pane.HTML(legend_html, width=180),
        pn.Spacer(height=20),
        pn.pane.HTML(wave_colorbar_html, width=120, height=350),
        pn.Spacer(height=20),
        pn.pane.HTML(stats_html, width=180),
        width=200,
    )

    # Match mesh viewer layout exactly
    title = pn.pane.Markdown(
        f"# Surfzone Simulation Result: {result.region_name}",
        sizing_mode='stretch_width',
    )

    main_row = pn.Row(
        pn.pane.HoloViews(plot, sizing_mode='fixed'),
        sidebar,
    )

    layout = pn.Column(
        title,
        main_row,
        sizing_mode='stretch_width',
    )

    print("Opening in browser...")
    pn.serve(layout, show=True, title=f'Surfzone Result: {result.region_name}')


def main():
    parser = argparse.ArgumentParser(
        description="Interactive surfzone simulation result viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--result-file',
        type=Path,
        default=None,
        help="Path to result .npz file (default: data/surfzone/output/primary_swell.npz)"
    )

    parser.add_argument(
        '--mesh',
        type=str,
        default="socal",
        help="Mesh region name (default: socal)"
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

    view_result(
        result_path=args.result_file,
        mesh_region=args.mesh,
        use_lonlat=args.lonlat,
        h_max=args.h_max,
    )


if __name__ == "__main__":
    main()
