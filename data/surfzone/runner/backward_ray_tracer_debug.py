#!/usr/bin/env python3
"""
Debug visualization for backward wave ray tracing.

Uses the ACTUAL physics from backward_ray_tracer.py - no duplicate code.
This ensures what you see in the visualization is exactly what the
production code does.

Usage:
    python backward_ray_tracer_debug.py [--zoom CENTER_X CENTER_Y RANGE_KM]

Example:
    python backward_ray_tracer_debug.py --zoom 475000 3660000 5

Interactive controls (in plot window):
    - Scroll wheel: Zoom in/out (centered on mouse)
    - Click and drag: Pan
    - 'h' or Home: Reset to original view
    - 'p': Toggle pan mode
    - 'o': Toggle zoom-to-rectangle mode
    - 's': Save figure
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('MacOSX')  # Native macOS backend with interactive support
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the ACTUAL functions we want to debug - no duplication
from data.surfzone.runner.backward_ray_tracer import (
    trace_backward_single,
    compute_initial_direction_blended,
)
from data.surfzone.runner.wave_physics import deep_water_properties


# =============================================================================
# Visualization
# =============================================================================

def plot_backward_rays(
    mesh,
    rays_data,
    title="Backward Wave Propagation Debug",
    output_path=None,
    zoom_center=None,
    zoom_range_km=15.0,
):
    """
    Plot backward ray paths with depth coloring.

    Args:
        mesh: SurfZoneMesh object
        rays_data: List of ray result dicts
        title: Plot title
        output_path: Optional path to save figure
        zoom_center: (x, y) center point for zoom, or None for auto
        zoom_range_km: Zoom box size in km (default 15)
    """
    # Collect ray data
    all_x = np.concatenate([r['path_x'] for r in rays_data if len(r['path_x']) > 0])
    all_y = np.concatenate([r['path_y'] for r in rays_data if len(r['path_y']) > 0])

    # Get depths from mesh for coloring
    arrays = mesh.get_numba_arrays()
    all_depths = []
    for ray in rays_data:
        if len(ray['path_x']) > 0:
            # Use the depths we stored during tracing
            all_depths.extend(ray.get('path_depth', []))

    if all_depths:
        vmin, vmax = min(all_depths), max(all_depths)
    else:
        vmin, vmax = 0, 50

    termination_colors = {
        'boundary': 'green',   # reached boundary (success)
        'domain': 'red',       # left domain
        'max_steps': 'orange', # max_steps
    }

    # Get start points for zooming
    start_x = [r['path_x'][0] for r in rays_data if len(r['path_x']) > 0]
    start_y = [r['path_y'][0] for r in rays_data if len(r['path_y']) > 0]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    def plot_rays_on_axis(ax, xlim=None, ylim=None):
        """Helper to plot rays on an axis."""
        # Plot coastlines
        if hasattr(mesh, 'coastlines') and mesh.coastlines:
            for coastline in mesh.coastlines:
                ax.plot(coastline[:, 0], coastline[:, 1], 'k-', linewidth=2, alpha=0.9)

        # Plot each ray
        for ray in rays_data:
            path_x = ray['path_x']
            path_y = ray['path_y']
            path_depth = ray.get('path_depth', np.zeros(len(path_x)))
            term = ray['termination']

            if len(path_x) < 2:
                continue

            # Create line segments for coloring
            points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Color by depth
            if len(path_depth) > 1:
                colors = path_depth[:-1]
            else:
                colors = np.zeros(len(segments))

            lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(vmin, vmax))
            lc.set_array(colors)
            lc.set_linewidth(3)
            ax.add_collection(lc)

            # Mark start point (near shore) with red dot
            ax.plot(path_x[0], path_y[0], 'ro', markersize=8, alpha=0.9, zorder=10)

            # Mark end point
            end_color = termination_colors.get(term, 'gray')
            ax.plot(path_x[-1], path_y[-1], 's', color=end_color, markersize=8, alpha=0.9, zorder=10)

        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        ax.set_aspect('equal')
        ax.set_xlabel('UTM X (m)')
        ax.set_ylabel('UTM Y (m)')

    # Determine zoom center
    if zoom_center is None:
        x_center = np.mean(start_x)
        y_center = np.mean(start_y)
    else:
        x_center, y_center = zoom_center

    zoom_range = zoom_range_km * 1000  # Convert km to m

    # LEFT: Zoomed view
    ax1 = axes[0]
    plot_rays_on_axis(
        ax1,
        xlim=(x_center - zoom_range, x_center + zoom_range),
        ylim=(y_center - zoom_range, y_center + zoom_range)
    )
    ax1.set_title(f'ZOOMED: {zoom_range_km*2:.0f}km box\n(Red dots = ray start)')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cbar1 = plt.colorbar(sm, ax=ax1, shrink=0.8)
    cbar1.set_label('Depth (m)')

    # MIDDLE: Full view
    ax2 = axes[1]
    x_pad = (all_x.max() - all_x.min()) * 0.1
    y_pad = (all_y.max() - all_y.min()) * 0.1
    plot_rays_on_axis(
        ax2,
        xlim=(all_x.min() - x_pad, all_x.max() + x_pad),
        ylim=(all_y.min() - y_pad, all_y.max() + y_pad)
    )
    ax2.set_title('FULL VIEW: All Ray Paths\n(Shore -> Boundary)')

    # Legend
    for term, color in termination_colors.items():
        label = {'boundary': 'Reached boundary', 'domain': 'Left domain', 'max_steps': 'Max steps'}[term]
        ax2.plot([], [], 's', color=color, markersize=10, label=label)
    ax2.legend(loc='upper left')

    # RIGHT: Depth vs distance
    ax3 = axes[2]
    for ray in rays_data:
        path_x = ray['path_x']
        path_y = ray['path_y']
        path_depth = ray.get('path_depth', [])
        term = ray['termination']

        if len(path_x) < 2:
            continue

        # Calculate distance along ray
        dx = np.diff(path_x)
        dy = np.diff(path_y)
        distances = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])

        color = termination_colors.get(term, 'gray')
        if len(path_depth) == len(distances):
            ax3.plot(distances, path_depth, '-', color=color, alpha=0.6, linewidth=2)

    ax3.set_xlabel('Distance along ray (m)')
    ax3.set_ylabel('Water Depth (m)')
    ax3.set_title('Depth vs Distance\n(Rays: shallow -> deep)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=50, color='blue', linestyle='--', linewidth=2, label='Boundary (50m)')
    ax3.legend(loc='lower right')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_path}")

    # Add scroll wheel zoom for all axes
    def on_scroll(event):
        """Zoom in/out with scroll wheel, centered on mouse position."""
        if event.inaxes is None:
            return

        ax = event.inaxes
        scale_factor = 1.2 if event.button == 'down' else 1/1.2

        # Get current limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Get mouse position in data coordinates
        xdata = event.xdata
        ydata = event.ydata

        # Calculate new limits centered on mouse
        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor

        # Keep mouse position at same relative location
        rel_x = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        rel_y = (ydata - ylim[0]) / (ylim[1] - ylim[0])

        new_xlim = [xdata - new_width * rel_x, xdata + new_width * (1 - rel_x)]
        new_ylim = [ydata - new_height * rel_y, ydata + new_height * (1 - rel_y)]

        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    print("\n" + "=" * 50)
    print("INTERACTIVE CONTROLS:")
    print("  Scroll wheel : Zoom in/out (centered on mouse)")
    print("  Click + drag : Pan")
    print("  'h' or Home  : Reset view")
    print("  'o'          : Zoom to rectangle mode")
    print("  'p'          : Pan mode")
    print("  's'          : Save figure")
    print("=" * 50)

    plt.show(block=True)


def trace_rays_for_debug(
    mesh,
    n_rays: int = 100,
    T: float = 12.0,
    partition_direction: float = 285.0,
    boundary_depth: float = 50.0,
    step_size: float = 15.0,
    max_steps: int = 3000,
    deep_weight: float = 0.8,
    depth_range: tuple = (2, 10),
    seed: int = 42,
):
    """
    Trace rays using the ACTUAL backward_ray_tracer functions.

    This calls the same trace_backward_single and compute_initial_direction_blended
    that the production code uses.

    Args:
        mesh: SurfZoneMesh object
        n_rays: Number of rays to trace
        T: Wave period (s)
        partition_direction: Swell direction at boundary (nautical degrees)
        boundary_depth: Depth threshold for "reached boundary" (m)
        step_size: Ray marching step size (m)
        max_steps: Maximum steps per ray
        deep_weight: Blend weight for depth gradient (0-1)
        depth_range: (min, max) depth range for selecting start points
        seed: Random seed for reproducibility

    Returns:
        List of ray data dicts with path coordinates and termination status
    """
    from data.surfzone.runner.ray_tracer import interpolate_depth_indexed

    arrays = mesh.get_numba_arrays()
    points_x = arrays['points_x']
    points_y = arrays['points_y']
    depth = arrays['depth']
    triangles = arrays['triangles']
    grid_x_min = arrays['grid_x_min']
    grid_y_min = arrays['grid_y_min']
    grid_cell_size = arrays['grid_cell_size']
    grid_n_cells_x = arrays['grid_n_cells_x']
    grid_n_cells_y = arrays['grid_n_cells_y']
    grid_cell_starts = arrays['grid_cell_starts']
    grid_cell_counts = arrays['grid_cell_counts']
    grid_triangles = arrays['grid_triangles']

    # Find shallow water start points
    min_depth, max_depth = depth_range
    shallow_mask = (depth > min_depth) & (depth < max_depth)
    shallow_indices = np.where(shallow_mask)[0]

    if len(shallow_indices) == 0:
        print(f"No points found in depth range {depth_range}!")
        return []

    print(f"Found {len(shallow_indices)} points in depth range {min_depth}-{max_depth}m")

    # Sample points
    n_rays = min(n_rays, len(shallow_indices))
    np.random.seed(seed)
    sample_indices = np.random.choice(shallow_indices, size=n_rays, replace=False)

    print(f"Tracing {n_rays} rays using ACTUAL backward_ray_tracer functions...")
    print(f"  - trace_backward_single()")
    print(f"  - compute_initial_direction_blended(deep_weight={deep_weight})")

    rays_data = []
    termination_counts = {'boundary': 0, 'domain': 0, 'max_steps': 0}

    for i, idx in enumerate(sample_indices):
        start_x = points_x[idx]
        start_y = points_y[idx]

        # Use the ACTUAL blended initial direction function
        initial_direction = compute_initial_direction_blended(
            start_x, start_y, partition_direction,
            points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles,
            deep_weight=deep_weight,
        )

        # Use the ACTUAL trace_backward_single function
        (
            end_x, end_y, theta_arrival, Cg_start, Cg_end, reached_boundary,
            path_x, path_y
        ) = trace_backward_single(
            start_x, start_y, T, initial_direction,
            points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles,
            boundary_depth,
            step_size, max_steps,
        )

        # Determine termination reason
        if reached_boundary:
            termination = 'boundary'
        elif np.isnan(end_x):
            termination = 'domain'
        else:
            termination = 'max_steps'

        termination_counts[termination] += 1

        # Get depths along path for visualization
        path_depth = np.array([
            interpolate_depth_indexed(
                px, py, points_x, points_y, depth, triangles,
                grid_x_min, grid_y_min, grid_cell_size,
                grid_n_cells_x, grid_n_cells_y,
                grid_cell_starts, grid_cell_counts, grid_triangles
            )
            for px, py in zip(path_x, path_y)
        ])

        rays_data.append({
            'path_x': path_x,
            'path_y': path_y,
            'path_depth': path_depth,
            'termination': termination,
            'initial_direction': initial_direction,
            'theta_arrival': theta_arrival,
            'Cg_start': Cg_start,
            'Cg_end': Cg_end,
        })

        if (i + 1) % 20 == 0:
            print(f"  Traced {i + 1}/{n_rays} rays")

    print(f"\nResults:")
    print(f"  Reached boundary: {termination_counts['boundary']}")
    print(f"  Left domain: {termination_counts['domain']}")
    print(f"  Max steps: {termination_counts['max_steps']}")

    return rays_data


def run_debug(args):
    """Run the backward ray tracing debug visualization."""
    from data.surfzone.mesh import SurfZoneMesh

    print("=" * 60)
    print("BACKWARD RAY TRACER DEBUG")
    print("=" * 60)
    print("\nThis visualization uses the ACTUAL functions from")
    print("backward_ray_tracer.py - no duplicate physics code.")

    # Load mesh
    print("\nLoading mesh...")
    mesh_dir = PROJECT_ROOT / "data" / "surfzone" / "meshes" / "socal"
    mesh = SurfZoneMesh.load(mesh_dir)

    # Wave conditions
    T = 12.0  # seconds
    partition_direction = 285.0  # degrees nautical (FROM WNW)

    print(f"\nWave conditions:")
    print(f"  T = {T} s")
    print(f"  Partition direction = {partition_direction}deg (FROM)")

    # Deep water properties for reference
    L0, C0, Cg0 = deep_water_properties(T)
    print(f"\nDeep water properties:")
    print(f"  L0 = {L0:.1f} m")
    print(f"  C0 = {C0:.1f} m/s")
    print(f"  Cg0 = {Cg0:.1f} m/s")

    # Trace rays using ACTUAL functions
    rays_data = trace_rays_for_debug(
        mesh,
        n_rays=100,
        T=T,
        partition_direction=partition_direction,
        boundary_depth=50.0,
        step_size=15.0,
        max_steps=3000,
        deep_weight=0.8,  # 80% depth gradient, 20% partition
        depth_range=(2, 10),
    )

    if not rays_data:
        return

    # Depth statistics for successful rays
    successful_rays = [r for r in rays_data if r['termination'] == 'boundary']
    if successful_rays:
        start_depths = [r['path_depth'][0] for r in successful_rays if len(r['path_depth']) > 0]
        end_depths = [r['path_depth'][-1] for r in successful_rays if len(r['path_depth']) > 0]
        if start_depths and end_depths:
            print(f"\nDepth change (successful rays):")
            print(f"  Start: {np.mean(start_depths):.1f} +/- {np.std(start_depths):.1f} m")
            print(f"  End: {np.mean(end_depths):.1f} +/- {np.std(end_depths):.1f} m")

    # Parse zoom arguments
    zoom_center = None
    zoom_range_km = 15.0
    if args.zoom:
        zoom_center = (args.zoom[0], args.zoom[1])
        zoom_range_km = args.zoom[2]
        print(f"\nZoom: center=({zoom_center[0]:.0f}, {zoom_center[1]:.0f}), range={zoom_range_km}km")

    # Plot
    print("\nGenerating visualization...")
    output_path = PROJECT_ROOT / "data" / "surfzone" / "debug_backward_rays.png"
    plot_backward_rays(
        mesh, rays_data,
        title=f"Backward Ray Tracer Debug: T={T}s, Dir={partition_direction}deg, deep_weight=0.8",
        output_path=output_path,
        zoom_center=zoom_center,
        zoom_range_km=zoom_range_km,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug backward ray tracing visualization")
    parser.add_argument(
        '--zoom',
        nargs=3,
        type=float,
        metavar=('CENTER_X', 'CENTER_Y', 'RANGE_KM'),
        help='Zoom to specific area: CENTER_X CENTER_Y RANGE_KM (e.g., 475000 3660000 5)'
    )
    args = parser.parse_args()
    run_debug(args)
