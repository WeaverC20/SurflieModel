#!/usr/bin/env python3
"""
Interactive Surfzone Simulation Result Viewer (Datashader)

View forward ray tracing results overlaid on the mesh.
Shows wave heights, coverage, and energy statistics.

Usage:
    python scripts/dev/view_surfzone_result.py --region socal
    python scripts/dev/view_surfzone_result.py --region norcal --result-file path/to/result.npz
    python scripts/dev/view_surfzone_result.py --list-regions
    python scripts/dev/view_surfzone_result.py --region socal --lonlat
"""

import argparse
import sys
from pathlib import Path
import io
import base64
from typing import Optional

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

from data.regions.region import REGIONS


def list_regions_with_results():
    """List regions with their simulation results status."""
    print("\nRegions with simulation results:")
    print("-" * 70)

    for name in ['socal', 'central', 'norcal']:
        region = REGIONS[name]

        # Check for surfzone mesh
        mesh_dir = project_root / "data" / "surfzone" / "meshes" / name
        mesh_exists = mesh_dir.exists() and any(mesh_dir.glob("*.npz"))

        # Check for results in region directory
        result_dir = project_root / "data" / "surfzone" / "output" / name
        results = []
        if result_dir.exists():
            results = sorted(result_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)

        print(f"  {name:12} - {region.display_name}")
        print(f"               Mesh: {'Yes' if mesh_exists else 'No'}")

        if results:
            print(f"               Results:")
            for r in results[:3]:  # Show first 3
                print(f"                 - {r.name}")
            if len(results) > 3:
                print(f"                 ... and {len(results) - 3} more")
        else:
            print(f"               Results: None")
        print()


def find_result_file(region_name: str, result_file: Optional[Path] = None) -> Path:
    """
    Find result file for a region.

    Args:
        region_name: Region identifier (socal, norcal, central)
        result_file: Explicit path (if provided, used directly)

    Returns:
        Path to result .npz file

    Raises:
        FileNotFoundError: If no results found
    """
    if result_file is not None:
        if not result_file.exists():
            raise FileNotFoundError(f"Result file not found: {result_file}")
        return result_file

    # Try region-specific directory
    region_dir = project_root / "data" / "surfzone" / "output" / region_name
    if region_dir.exists():
        # Default filename for forward results
        default_result = region_dir / "forward_energy.npz"
        if default_result.exists():
            return default_result

        # Fall back to most recent .npz in directory
        npz_files = sorted(region_dir.glob("*.npz"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if npz_files:
            return npz_files[0]

    raise FileNotFoundError(
        f"No results found for region '{region_name}'. "
        f"Run simulation first: python data/surfzone/runner/run_simulation.py --region {region_name}"
    )


def load_mesh(region_name: str = "socal"):
    """Load a surf zone mesh."""
    from data.surfzone.mesh import SurfZoneMesh

    mesh_dir = project_root / "data" / "surfzone" / "meshes" / region_name
    if not mesh_dir.exists():
        print(f"Error: Mesh not found at {mesh_dir}")
        sys.exit(1)

    return SurfZoneMesh.load(mesh_dir)


def load_result(result_path: Path):
    """Load forward ray tracing result."""
    from data.surfzone.runner.output_writer import load_forward_result

    if not result_path.exists():
        print(f"Error: Result not found at {result_path}")
        print("Run the simulation first: python data/surfzone/runner/run_simulation.py --region <region>")
        sys.exit(1)

    return load_forward_result(result_path)


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

    # Get coordinates
    result_x = result.mesh_x.copy()
    result_y = result.mesh_y.copy()

    n_points = result.n_points
    print(f"Result points: {n_points:,}")

    # Coverage mask
    covered_mask = result.ray_count > 0
    not_covered_mask = ~covered_mask
    n_covered = result.n_covered

    print(f"Covered: {n_covered:,} ({100*result.coverage_rate:.1f}%)")
    print(f"Not covered: {np.sum(not_covered_mask):,}")

    # Convert coordinates if needed
    if use_lonlat:
        result_x, result_y = mesh.utm_to_lon_lat(result_x, result_y)
        x_label = "Longitude"
        y_label = "Latitude"
    else:
        x_label = "UTM Easting (m)"
        y_label = "UTM Northing (m)"

    # Auto-determine wave height max
    if h_max is None:
        if n_covered > 0:
            h_max = float(np.nanpercentile(result.H_at_mesh[covered_mask], 98))
            h_max = max(h_max, 0.5)  # At least 0.5m
        else:
            h_max = 2.0

    print(f"Wave height color scale: 0 - {h_max:.1f}m")

    # === Create layers ===

    # Layer 1: Not covered points (dark gray)
    if np.sum(not_covered_mask) > 0:
        nc_df = pd.DataFrame({
            'x': result_x[not_covered_mask],
            'y': result_y[not_covered_mask],
            'val': np.ones(np.sum(not_covered_mask)),
        })
        nc_points = hv.Points(nc_df, kdims=['x', 'y'], vdims=['val'])
        nc_cmap = ['#444455', '#444455']  # Dark gray
    else:
        nc_points = None

    # Layer 2: Covered points - colored by wave height
    wave_cmap = ['#0044aa', '#0066cc', '#0088ee', '#00aaff', '#00cccc',
                 '#00ee88', '#44ff44', '#aaff00', '#ffff00']
    if n_covered > 0:
        cov_df = pd.DataFrame({
            'x': result_x[covered_mask],
            'y': result_y[covered_mask],
            'H': np.clip(result.H_at_mesh[covered_mask], 0, h_max),
        })
        cov_points = hv.Points(cov_df, kdims=['x', 'y'], vdims=['H'])
    else:
        cov_points = None

    # Create datashaded plots
    plot = None

    if nc_points is not None:
        nc_shaded = spread(
            datashade(
                nc_points,
                aggregator=ds.mean('val'),
                cmap=nc_cmap,
            ),
            px=2,
        )
        plot = nc_shaded

    if cov_points is not None:
        cov_shaded = spread(
            datashade(
                cov_points,
                aggregator=ds.mean('H'),
                cmap=wave_cmap,
                cnorm='linear',
            ),
            px=4,
        )
        if plot is None:
            plot = cov_shaded
        else:
            plot = plot * cov_shaded

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
        if plot is not None:
            plot = plot * coastline_overlay

    # Apply final opts
    if plot is not None:
        plot = plot.opts(
            width=1400,
            height=900,
            xlabel=x_label,
            ylabel=y_label,
            tools=['wheel_zoom', 'pan', 'reset', 'box_zoom'],
            active_tools=['wheel_zoom', 'pan'],
            bgcolor='#1a1a2e',
            data_aspect=1,  # Equal scaling on X and Y axes
        )

    # Create colorbars
    wave_colorbar_html = create_matplotlib_colorbar(
        0, h_max, 'Wave Height (m)', wave_cmap, height=300
    )

    # Legend HTML
    legend_html = """
    <div style="color: white; font-size: 12px; padding: 10px; background: #1a1a2e;">
        <b>Legend</b><br><br>
        <span style="color: #444455;">&#9632;</span> Not covered (no rays)<br>
        <span style="color: #00cccc;">&#9632;</span> Covered (see colorbar)<br>
        <span style="color: #ff00ff;">&mdash;</span> Coastline<br>
    </div>
    """

    # Stats HTML
    stats_html = f"""
    <div style="color: white; font-size: 11px; padding: 10px; background: #2a2a3e; border-radius: 5px;">
        <b>Forward Ray Tracing Statistics</b><br><br>
        Region: {result.region_name}<br>
        Partitions: {result.n_partitions}<br><br>
        Total points: {result.n_points:,}<br>
        Covered: {result.n_covered:,} ({100*result.coverage_rate:.1f}%)<br>
        Rays traced: {result.n_rays_total:,}<br><br>
    """
    if n_covered > 0:
        H_cov = result.H_at_mesh[covered_mask]
        energy_cov = result.energy[covered_mask]
        ray_cov = result.ray_count[covered_mask]
        stats_html += f"""
        <b>Wave Height (covered)</b><br>
        Min: {np.nanmin(H_cov):.2f}m<br>
        Max: {np.nanmax(H_cov):.2f}m<br>
        Mean: {np.nanmean(H_cov):.2f}m<br><br>
        <b>Energy</b><br>
        Min: {np.nanmin(energy_cov):.1f} J/m<br>
        Max: {np.nanmax(energy_cov):.1f} J/m<br>
        Mean: {np.nanmean(energy_cov):.1f} J/m<br><br>
        <b>Rays per point</b><br>
        Min: {ray_cov.min()}<br>
        Max: {ray_cov.max()}<br>
        Mean: {ray_cov.mean():.1f}<br>
        """
    stats_html += "</div>"

    # Layout
    sidebar = pn.Column(
        pn.pane.HTML(legend_html, width=180),
        pn.Spacer(height=20),
        pn.pane.HTML(wave_colorbar_html, width=120, height=350),
        pn.Spacer(height=20),
        pn.pane.HTML(stats_html, width=180),
        width=200,
    )

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
        epilog="""
Examples:
    python scripts/dev/view_surfzone_result.py --region socal
    python scripts/dev/view_surfzone_result.py --region norcal --lonlat
    python scripts/dev/view_surfzone_result.py --list-regions
    python scripts/dev/view_surfzone_result.py --region socal --result-file path/to/result.npz
        """,
    )

    # Region selection
    parser.add_argument(
        '--region',
        type=str,
        default=None,
        help="Region name (socal, norcal, central). Auto-detects mesh and results."
    )

    parser.add_argument(
        '--list-regions',
        action='store_true',
        help="List regions with available results and exit"
    )

    parser.add_argument(
        '--result-file',
        type=Path,
        default=None,
        help="Path to result .npz file (default: auto-detect for region)"
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

    # Handle --list-regions
    if args.list_regions:
        list_regions_with_results()
        return

    # Determine region
    region_name = args.region
    if region_name is None:
        region_name = "socal"  # Default
        print(f"Note: Using default region '{region_name}'. Use --region to specify.")

    # Find result file
    try:
        result_path = find_result_file(region_name, args.result_file)
        print(f"Using result file: {result_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    view_result(
        result_path=result_path,
        mesh_region=region_name,
        use_lonlat=args.lonlat,
        h_max=args.h_max,
    )


if __name__ == "__main__":
    main()
