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
        'land': 'red',         # hit land (failure)
        'domain': 'blue',      # left domain
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
        boundary_depth: Depth threshold for "reached boundary" (m).
                        Set to 0 to use mesh boundary instead of depth threshold.
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

    # Coast distance boundary (if available)
    coast_distance = arrays.get('coast_distance', np.array([], dtype=np.float64))
    offshore_distance_m = arrays.get('offshore_distance_m', 0.0)

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
    termination_counts = {'boundary': 0, 'domain': 0, 'land': 0, 'max_steps': 0}

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
            boundary_depth,  # depth threshold (0 = disabled)
            coast_distance, offshore_distance_m,  # coast distance boundary
            step_size, max_steps,
        )

        # Determine termination reason
        # - boundary: ray reached offshore boundary (coast_distance >= threshold)
        # - domain: ray left mesh domain (NaN position)
        # - land: ray hit land (depth <= 0), returns reached_boundary=False, theta=NaN
        # - max_steps: ray exceeded max iterations without reaching boundary
        if reached_boundary:
            termination = 'boundary'
        elif np.isnan(end_x):
            termination = 'domain'
        elif np.isnan(theta_arrival):
            termination = 'land'
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
    print(f"  Hit land: {termination_counts['land']}")
    print(f"  Left domain: {termination_counts['domain']}")
    print(f"  Max steps: {termination_counts['max_steps']}")

    return rays_data


def trace_rays_with_convergence(
    mesh,
    n_rays: int = 20,
    T: float = 12.0,
    partition_direction: float = 285.0,
    directional_spread: float = 30.0,
    boundary_depth: float = 0.0,
    step_size: float = 15.0,
    max_steps: int = 3000,
    deep_weight: float = 0.8,
    depth_range: tuple = (2, 10),
    seed: int = 42,
    # Convergence parameters
    alpha: float = 0.6,
    max_iterations: int = 20,
    tolerance: float = 0.05,  # 5% of directional spread = converged
):
    """
    Trace rays with convergence iteration to match partition direction at boundary.

    For each mesh point, iteratively adjusts θ_M (direction at mesh point) until
    the ray arrives at the boundary with direction matching the partition direction.

    Convergence uses gradient descent:
        θ_M_new = θ_M - α × (θ_arrival - θ_partition)

    Convergence criterion:
        |θ_arrival - θ_partition| < tolerance × directional_spread

    Args:
        mesh: SurfZoneMesh object
        n_rays: Number of rays to trace
        T: Wave period (s)
        partition_direction: Target swell direction at boundary (nautical degrees)
        directional_spread: Directional spread of the partition (degrees)
        boundary_depth: Depth threshold (0 = use coast_distance boundary)
        step_size: Ray marching step size (m)
        max_steps: Maximum steps per ray trace
        deep_weight: Blend weight for initial direction guess
        depth_range: (min, max) depth range for selecting start points
        seed: Random seed for reproducibility
        alpha: Relaxation factor for gradient descent (0.5-0.7 typical)
        max_iterations: Maximum convergence iterations
        tolerance: Convergence tolerance as fraction of directional_spread

    Returns:
        List of ray data dicts with convergence information
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

    # Coast distance boundary
    coast_distance = arrays.get('coast_distance', np.array([], dtype=np.float64))
    offshore_distance_m = arrays.get('offshore_distance_m', 0.0)

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

    # Convergence threshold in degrees
    convergence_threshold = tolerance * directional_spread
    print(f"\nConvergence settings:")
    print(f"  Target direction: {partition_direction}° (nautical)")
    print(f"  Directional spread: {directional_spread}°")
    print(f"  Tolerance: {tolerance*100:.0f}% of spread = {convergence_threshold:.1f}°")
    print(f"  Alpha (relaxation): {alpha}")
    print(f"  Max iterations: {max_iterations}")

    print(f"\nTracing {n_rays} rays with convergence iteration...")

    rays_data = []
    convergence_stats = {
        'converged': 0,
        'not_converged': 0,
        'failed_to_reach': 0,
        'iterations': [],
    }

    for i, idx in enumerate(sample_indices):
        start_x = points_x[idx]
        start_y = points_y[idx]
        start_depth = depth[idx]

        # Initial guess: blend of depth gradient and partition direction
        theta_M = compute_initial_direction_blended(
            start_x, start_y, partition_direction,
            points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles,
            deep_weight=deep_weight,
        )

        # Track iteration history
        iteration_history = []
        final_path_x = None
        final_path_y = None
        final_path_depth = None
        converged = False
        failed_to_reach = False

        # Track best solution found
        best_error = float('inf')
        best_theta_M = theta_M
        best_path_x = None
        best_path_y = None

        # Adaptive alpha
        current_alpha = alpha
        prev_error = None
        consecutive_failures = 0

        for iteration in range(max_iterations):
            # Trace ray backward
            (
                end_x, end_y, theta_arrival, Cg_start, Cg_end, reached_boundary,
                path_x, path_y
            ) = trace_backward_single(
                start_x, start_y, T, theta_M,
                points_x, points_y, depth, triangles,
                grid_x_min, grid_y_min, grid_cell_size,
                grid_n_cells_x, grid_n_cells_y,
                grid_cell_starts, grid_cell_counts, grid_triangles,
                boundary_depth,
                coast_distance, offshore_distance_m,
                step_size, max_steps,
            )

            # Record iteration data
            iteration_data = {
                'iteration': iteration,
                'theta_M': theta_M,
                'theta_arrival': theta_arrival,
                'reached_boundary': reached_boundary,
                'alpha': current_alpha,
                'path_x': path_x.copy(),
                'path_y': path_y.copy(),
            }

            if not reached_boundary:
                # Ray didn't reach boundary (hit land or max steps)
                iteration_data['error'] = np.nan
                iteration_history.append(iteration_data)

                consecutive_failures += 1
                if consecutive_failures >= 3:
                    # Too many failures, give up
                    failed_to_reach = True
                    break

                # Backtrack: revert toward best known direction with smaller step
                current_alpha *= 0.5
                if prev_error is not None and best_theta_M is not None:
                    # Move back toward best direction
                    theta_M = best_theta_M
                continue

            consecutive_failures = 0  # Reset on success

            # Compute angle error (handle wrap-around)
            angle_diff = theta_arrival - partition_direction
            while angle_diff > 180:
                angle_diff -= 360
            while angle_diff < -180:
                angle_diff += 360

            iteration_data['error'] = angle_diff
            iteration_history.append(iteration_data)

            # Track best solution
            if abs(angle_diff) < best_error:
                best_error = abs(angle_diff)
                best_theta_M = theta_M
                best_path_x = path_x.copy()
                best_path_y = path_y.copy()

            # Store current path
            final_path_x = path_x
            final_path_y = path_y

            # Check convergence
            if abs(angle_diff) < convergence_threshold:
                converged = True
                break

            # Adaptive alpha: reduce if error increased
            if prev_error is not None and abs(angle_diff) > abs(prev_error) * 1.1:
                current_alpha *= 0.7  # Reduce step size
                current_alpha = max(current_alpha, 0.1)  # Don't go below 0.1

            prev_error = angle_diff

            # Gradient descent update
            theta_M = theta_M - current_alpha * angle_diff

            # Keep in valid range
            while theta_M > 360:
                theta_M -= 360
            while theta_M < 0:
                theta_M += 360

        # If not converged but we have a good solution, use it
        if not converged and not failed_to_reach and best_path_x is not None:
            final_path_x = best_path_x
            final_path_y = best_path_y

        # Get depths along final path
        if final_path_x is not None and len(final_path_x) > 0:
            final_path_depth = np.array([
                interpolate_depth_indexed(
                    px, py, points_x, points_y, depth, triangles,
                    grid_x_min, grid_y_min, grid_cell_size,
                    grid_n_cells_x, grid_n_cells_y,
                    grid_cell_starts, grid_cell_counts, grid_triangles
                )
                for px, py in zip(final_path_x, final_path_y)
            ])
        else:
            final_path_depth = np.array([])

        # Update stats
        if failed_to_reach:
            convergence_stats['failed_to_reach'] += 1
        elif converged:
            convergence_stats['converged'] += 1
            convergence_stats['iterations'].append(len(iteration_history))
        else:
            convergence_stats['not_converged'] += 1
            convergence_stats['iterations'].append(max_iterations)

        # Store ray data
        rays_data.append({
            'start_x': start_x,
            'start_y': start_y,
            'start_depth': start_depth,
            'path_x': final_path_x if final_path_x is not None else np.array([]),
            'path_y': final_path_y if final_path_y is not None else np.array([]),
            'path_depth': final_path_depth,
            'converged': converged,
            'failed_to_reach': failed_to_reach,
            'n_iterations': len(iteration_history),
            'iteration_history': iteration_history,
            'initial_direction': iteration_history[0]['theta_M'] if iteration_history else np.nan,
            'final_direction': iteration_history[-1]['theta_M'] if iteration_history else np.nan,
            'final_arrival': iteration_history[-1]['theta_arrival'] if iteration_history else np.nan,
            'final_error': iteration_history[-1].get('error', np.nan) if iteration_history else np.nan,
            'Cg_start': Cg_start if 'Cg_start' in dir() else np.nan,
            'Cg_end': Cg_end if 'Cg_end' in dir() else np.nan,
        })

        if (i + 1) % 10 == 0:
            print(f"  Traced {i + 1}/{n_rays} rays")

    # Print summary
    print(f"\nConvergence Results:")
    print(f"  Converged: {convergence_stats['converged']}/{n_rays}")
    print(f"  Not converged: {convergence_stats['not_converged']}/{n_rays}")
    print(f"  Failed to reach boundary: {convergence_stats['failed_to_reach']}/{n_rays}")

    if convergence_stats['iterations']:
        iters = convergence_stats['iterations']
        print(f"\nIterations to converge:")
        print(f"  Min: {min(iters)}, Max: {max(iters)}, Mean: {np.mean(iters):.1f}")

    return rays_data


def plot_convergence_rays(
    mesh,
    rays_data,
    partition_direction: float,
    directional_spread: float,
    title="Backward Ray Convergence Debug",
    output_path=None,
    zoom_center=None,
    zoom_range_km=15.0,
):
    """
    Plot ray paths with convergence information.

    Shows:
    - Final converged ray paths (colored by convergence status)
    - Convergence iteration statistics
    - Direction error vs iteration for sample rays
    """
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 12))

    # Layout: 2x2 grid
    ax_map = fig.add_subplot(2, 2, 1)       # Map view of rays
    ax_zoom = fig.add_subplot(2, 2, 2)      # Zoomed map
    ax_errors = fig.add_subplot(2, 2, 3)    # Error vs iteration
    ax_stats = fig.add_subplot(2, 2, 4)     # Statistics

    # Get mesh data for coastline plotting
    coastlines = mesh.coastlines if hasattr(mesh, 'coastlines') and mesh.coastlines else []

    # Collect ray endpoints for auto-zoom
    converged_rays = [r for r in rays_data if r['converged']]
    failed_rays = [r for r in rays_data if r['failed_to_reach']]
    not_converged_rays = [r for r in rays_data if not r['converged'] and not r['failed_to_reach']]

    # --- Map View (ax_map) ---
    ax_map.set_title("Ray Paths (colored by convergence)", fontsize=11)

    # Plot coastlines
    for coastline in coastlines:
        ax_map.plot(coastline[:, 0], coastline[:, 1], 'k-', linewidth=0.5, alpha=0.5)

    # Plot rays
    for ray in converged_rays:
        if len(ray['path_x']) > 0:
            ax_map.plot(ray['path_x'], ray['path_y'], 'g-', linewidth=0.8, alpha=0.6)
            ax_map.plot(ray['path_x'][0], ray['path_y'][0], 'go', markersize=3)

    for ray in not_converged_rays:
        if len(ray['path_x']) > 0:
            ax_map.plot(ray['path_x'], ray['path_y'], 'orange', linewidth=0.8, alpha=0.6)
            ax_map.plot(ray['path_x'][0], ray['path_y'][0], 'o', color='orange', markersize=3)

    for ray in failed_rays:
        if len(ray['path_x']) > 0:
            ax_map.plot(ray['path_x'], ray['path_y'], 'r-', linewidth=0.8, alpha=0.6)
            ax_map.plot(ray['path_x'][0], ray['path_y'][0], 'ro', markersize=3)

    ax_map.set_xlabel('Easting (m)')
    ax_map.set_ylabel('Northing (m)')
    ax_map.set_aspect('equal')
    ax_map.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='g', label=f'Converged ({len(converged_rays)})'),
        Line2D([0], [0], color='orange', label=f'Not converged ({len(not_converged_rays)})'),
        Line2D([0], [0], color='r', label=f'Failed to reach ({len(failed_rays)})'),
    ]
    ax_map.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # --- Zoomed View (ax_zoom) ---
    ax_zoom.set_title("Zoomed View (start points)", fontsize=11)

    # Plot coastlines
    for coastline in coastlines:
        ax_zoom.plot(coastline[:, 0], coastline[:, 1], 'k-', linewidth=1, alpha=0.7)

    # Plot rays with iteration count coloring
    all_iterations = [r['n_iterations'] for r in rays_data if r['n_iterations'] > 0]
    if all_iterations:
        max_iter = max(all_iterations)
        cmap = plt.cm.viridis

        for ray in rays_data:
            if len(ray['path_x']) > 0 and ray['n_iterations'] > 0:
                color = cmap(ray['n_iterations'] / max(max_iter, 1))
                ax_zoom.plot(ray['path_x'], ray['path_y'], '-', color=color, linewidth=1.2, alpha=0.7)
                ax_zoom.plot(ray['path_x'][0], ray['path_y'][0], 'o', color=color, markersize=4)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, max_iter))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_zoom, label='Iterations to converge')

    # Set zoom limits
    if zoom_center:
        half_range = zoom_range_km * 1000 / 2
        ax_zoom.set_xlim(zoom_center[0] - half_range, zoom_center[0] + half_range)
        ax_zoom.set_ylim(zoom_center[1] - half_range, zoom_center[1] + half_range)
    else:
        # Auto-zoom to ray start points
        start_xs = [r['start_x'] for r in rays_data]
        start_ys = [r['start_y'] for r in rays_data]
        if start_xs and start_ys:
            cx, cy = np.mean(start_xs), np.mean(start_ys)
            half_range = 10000  # 10km default
            ax_zoom.set_xlim(cx - half_range, cx + half_range)
            ax_zoom.set_ylim(cy - half_range, cy + half_range)

    ax_zoom.set_xlabel('Easting (m)')
    ax_zoom.set_ylabel('Northing (m)')
    ax_zoom.set_aspect('equal')
    ax_zoom.grid(True, alpha=0.3)

    # --- Error vs Iteration (ax_errors) ---
    ax_errors.set_title("Direction Error vs Iteration", fontsize=11)

    # Plot error curves for each ray
    convergence_threshold = 0.05 * directional_spread
    for i, ray in enumerate(rays_data):
        if ray['iteration_history']:
            iterations = [h['iteration'] for h in ray['iteration_history']]
            errors = [h.get('error', np.nan) for h in ray['iteration_history']]

            # Color by convergence status
            if ray['converged']:
                color = 'green'
                alpha = 0.5
            elif ray['failed_to_reach']:
                color = 'red'
                alpha = 0.3
            else:
                color = 'orange'
                alpha = 0.5

            ax_errors.plot(iterations, errors, '-o', color=color, alpha=alpha,
                          markersize=3, linewidth=1)

    # Add convergence threshold lines
    ax_errors.axhline(y=convergence_threshold, color='g', linestyle='--',
                     label=f'+{convergence_threshold:.1f}° (converged)', alpha=0.7)
    ax_errors.axhline(y=-convergence_threshold, color='g', linestyle='--',
                     label=f'-{convergence_threshold:.1f}°', alpha=0.7)
    ax_errors.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)

    ax_errors.set_xlabel('Iteration')
    ax_errors.set_ylabel('Direction Error (θ_arrival - θ_partition) [degrees]')
    ax_errors.grid(True, alpha=0.3)
    ax_errors.legend(loc='upper right', fontsize=9)

    # --- Statistics (ax_stats) ---
    ax_stats.axis('off')

    # Compute statistics
    converged_count = len(converged_rays)
    not_converged_count = len(not_converged_rays)
    failed_count = len(failed_rays)
    total = len(rays_data)

    iterations_to_converge = [r['n_iterations'] for r in converged_rays]
    final_errors = [r['final_error'] for r in rays_data if not np.isnan(r['final_error'])]

    stats_text = [
        "CONVERGENCE STATISTICS",
        "=" * 40,
        f"",
        f"Total rays: {total}",
        f"  Converged: {converged_count} ({100*converged_count/total:.0f}%)",
        f"  Not converged: {not_converged_count} ({100*not_converged_count/total:.0f}%)",
        f"  Failed to reach: {failed_count} ({100*failed_count/total:.0f}%)",
        f"",
        f"Target direction: {partition_direction}° (nautical)",
        f"Directional spread: {directional_spread}°",
        f"Convergence threshold: ±{convergence_threshold:.1f}°",
        f"",
    ]

    if iterations_to_converge:
        stats_text.extend([
            "Iterations to converge:",
            f"  Min: {min(iterations_to_converge)}",
            f"  Max: {max(iterations_to_converge)}",
            f"  Mean: {np.mean(iterations_to_converge):.1f}",
            f"  Median: {np.median(iterations_to_converge):.0f}",
            f"",
        ])

    if final_errors:
        stats_text.extend([
            "Final direction error:",
            f"  Mean: {np.mean(final_errors):.2f}°",
            f"  Std: {np.std(final_errors):.2f}°",
            f"  |Max|: {max(abs(e) for e in final_errors):.2f}°",
        ])

    ax_stats.text(0.1, 0.95, '\n'.join(stats_text), transform=ax_stats.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_path}")

    # Print interactive help
    print("\n" + "=" * 50)
    print("INTERACTIVE CONTROLS")
    print("=" * 50)
    print("  Scroll wheel : Zoom in/out")
    print("  Click+drag   : Pan")
    print("  'h' or Home  : Reset view")
    print("  's'          : Save figure")
    print("=" * 50)

    plt.show(block=True)


def plot_detailed_convergence(
    mesh,
    rays_data,
    partition_direction: float,
    title="Detailed Convergence: All Iteration Paths",
    output_path=None,
):
    """
    Plot detailed convergence showing ALL attempted paths for each ray.

    Creates a grid of subplots, one per ray, showing:
    - All iteration attempts colored by iteration number
    - Start point (green dot)
    - Final path (thick line)
    - Coastline for context
    """
    n_rays = len(rays_data)
    if n_rays == 0:
        print("No rays to plot!")
        return

    # Determine grid layout
    n_cols = min(3, n_rays)
    n_rows = (n_rays + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows))

    # Handle single row/col case
    if n_rays == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Get coastlines
    coastlines = mesh.coastlines if hasattr(mesh, 'coastlines') and mesh.coastlines else []

    # Color map for iterations
    cmap = plt.cm.plasma

    for idx, ray in enumerate(rays_data):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Get iteration history
        history = ray.get('iteration_history', [])
        n_iters = len(history)

        if n_iters == 0:
            ax.text(0.5, 0.5, 'No iterations', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            ax.set_title(f"Ray {idx + 1}: No data")
            continue

        # Collect all path coordinates for zoom calculation
        all_x = []
        all_y = []
        for h in history:
            if len(h['path_x']) > 0:
                all_x.extend(h['path_x'])
                all_y.extend(h['path_y'])

        if not all_x:
            ax.text(0.5, 0.5, 'No paths', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            ax.set_title(f"Ray {idx + 1}: No paths")
            continue

        # Calculate zoom bounds (with padding)
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        x_range = max(x_max - x_min, 500)  # Minimum 500m range
        y_range = max(y_max - y_min, 500)
        padding = 0.2
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        half_range = max(x_range, y_range) / 2 * (1 + padding)

        ax.set_xlim(x_center - half_range, x_center + half_range)
        ax.set_ylim(y_center - half_range, y_center + half_range)

        # Plot coastlines (clipped to view)
        for coastline in coastlines:
            ax.plot(coastline[:, 0], coastline[:, 1], 'k-', linewidth=1, alpha=0.5)

        # Plot each iteration's path
        for i, h in enumerate(history):
            path_x = h['path_x']
            path_y = h['path_y']
            if len(path_x) == 0:
                continue

            # Color by iteration (early = blue, late = red/yellow)
            color = cmap(i / max(n_iters - 1, 1))

            # Line width: thinner for early iterations, thicker for later
            lw = 0.8 + 1.5 * (i / max(n_iters - 1, 1))
            alpha = 0.4 + 0.5 * (i / max(n_iters - 1, 1))

            ax.plot(path_x, path_y, '-', color=color, linewidth=lw, alpha=alpha)

            # Mark end point with small circle
            ax.plot(path_x[-1], path_y[-1], 'o', color=color, markersize=3, alpha=alpha)

        # Mark start point (green, prominent)
        start_x = ray['start_x']
        start_y = ray['start_y']
        ax.plot(start_x, start_y, 'go', markersize=10, markeredgecolor='black',
               markeredgewidth=1, zorder=10, label='Start')

        # Highlight final path (if converged or best attempt)
        if history:
            final_path = history[-1]
            if len(final_path['path_x']) > 0:
                ax.plot(final_path['path_x'], final_path['path_y'], 'g-',
                       linewidth=2.5, alpha=0.9, label='Final')

        # Title with convergence info
        status = "✓ Converged" if ray['converged'] else ("✗ Failed" if ray['failed_to_reach'] else "○ Not converged")
        final_err = ray.get('final_error', np.nan)
        err_str = f"{final_err:+.1f}°" if not np.isnan(final_err) else "N/A"

        ax.set_title(
            f"Ray {idx + 1}: {status}\n"
            f"{n_iters} iters, final err: {err_str}",
            fontsize=10
        )
        ax.set_xlabel('Easting (m)', fontsize=9)
        ax.set_ylabel('Northing (m)', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Add iteration color legend text
        ax.text(0.02, 0.98, f"Iter 0→{n_iters-1}",
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Hide empty subplots
    for idx in range(n_rays, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    # Add colorbar for iteration colors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Iteration (early → late)', fontsize=10)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['First', 'Middle', 'Last'])

    plt.suptitle(f"{title}\nTarget direction: {partition_direction}°", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_path}")

    # Print iteration details
    print("\n" + "=" * 70)
    print("DETAILED ITERATION HISTORY")
    print("=" * 70)
    for idx, ray in enumerate(rays_data):
        history = ray.get('iteration_history', [])
        status = "CONVERGED" if ray['converged'] else ("FAILED" if ray['failed_to_reach'] else "NOT CONVERGED")
        print(f"\nRay {idx + 1} [{status}] - {len(history)} iterations:")
        print(f"  Start: ({ray['start_x']:.0f}, {ray['start_y']:.0f}), depth={ray['start_depth']:.1f}m")
        print(f"  Iteration progression:")
        for h in history:
            err = h.get('error', np.nan)
            alpha_used = h.get('alpha', np.nan)
            reached = "→boundary" if h['reached_boundary'] else "→FAILED"
            err_str = f"err={err:+6.1f}°" if not np.isnan(err) else "err=  N/A "
            print(f"    [{h['iteration']:2d}] θ_M={h['theta_M']:6.1f}° {reached}, {err_str}, α={alpha_used:.2f}")

    print("\n" + "=" * 50)
    print("INTERACTIVE CONTROLS")
    print("=" * 50)
    print("  Scroll wheel : Zoom in/out")
    print("  Click+drag   : Pan")
    print("  'h' or Home  : Reset view")
    print("  's'          : Save figure")
    print("=" * 50)

    plt.show(block=True)


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
    T = args.period
    partition_direction = args.direction
    directional_spread = args.spread

    print(f"\nWave conditions:")
    print(f"  T = {T} s")
    print(f"  Partition direction = {partition_direction}° (nautical FROM)")
    print(f"  Directional spread = {directional_spread}°")

    # Deep water properties for reference
    L0, C0, Cg0 = deep_water_properties(T)
    print(f"\nDeep water properties:")
    print(f"  L0 = {L0:.1f} m")
    print(f"  C0 = {C0:.1f} m/s")
    print(f"  Cg0 = {Cg0:.1f} m/s")

    # Parse zoom arguments
    zoom_center = None
    zoom_range_km = 15.0
    if args.zoom:
        zoom_center = (args.zoom[0], args.zoom[1])
        zoom_range_km = args.zoom[2]
        print(f"\nZoom: center=({zoom_center[0]:.0f}, {zoom_center[1]:.0f}), range={zoom_range_km}km")

    if args.convergence or args.detailed:
        # ===== CONVERGENCE MODE =====
        n_rays_to_trace = 5 if args.detailed else args.n_rays

        print("\n" + "=" * 60)
        if args.detailed:
            print("DETAILED CONVERGENCE MODE (showing all iteration paths)")
        else:
            print("CONVERGENCE MODE")
        print("=" * 60)
        print("\nIteratively adjusting θ_M until θ_arrival matches partition direction.")
        print(f"Update rule: θ_M_new = θ_M - α × (θ_arrival - θ_partition)")

        rays_data = trace_rays_with_convergence(
            mesh,
            n_rays=n_rays_to_trace,
            T=T,
            partition_direction=partition_direction,
            directional_spread=directional_spread,
            boundary_depth=0.0,  # Use coast_distance boundary
            step_size=15.0,
            max_steps=3000,
            deep_weight=0.8,
            depth_range=(2, 10),
            alpha=args.alpha,
            max_iterations=args.max_iter,
            tolerance=args.tolerance,
        )

        if not rays_data:
            return

        if args.detailed:
            # Plot detailed view showing all iteration paths
            print("\nGenerating detailed iteration visualization...")
            output_path = PROJECT_ROOT / "data" / "surfzone" / "debug_detailed_convergence.png"
            plot_detailed_convergence(
                mesh, rays_data,
                partition_direction=partition_direction,
                title=f"Detailed Convergence: T={T}s, α={args.alpha}",
                output_path=output_path,
            )
        else:
            # Plot summary convergence results
            print("\nGenerating convergence visualization...")
            output_path = PROJECT_ROOT / "data" / "surfzone" / "debug_convergence_rays.png"
            plot_convergence_rays(
                mesh, rays_data,
                partition_direction=partition_direction,
                directional_spread=directional_spread,
                title=f"Ray Convergence: T={T}s, Dir={partition_direction}°, α={args.alpha}",
                output_path=output_path,
                zoom_center=zoom_center,
                zoom_range_km=zoom_range_km,
            )

    else:
        # ===== SINGLE-TRACE MODE (original) =====
        rays_data = trace_rays_for_debug(
            mesh,
            n_rays=args.n_rays,
            T=T,
            partition_direction=partition_direction,
            boundary_depth=0.0,  # Use coast_distance boundary
            step_size=15.0,
            max_steps=3000,
            deep_weight=0.8,
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

        # Plot
        print("\nGenerating visualization...")
        output_path = PROJECT_ROOT / "data" / "surfzone" / "debug_backward_rays.png"
        plot_backward_rays(
            mesh, rays_data,
            title=f"Backward Ray Tracer Debug: T={T}s, Dir={partition_direction}°, deep_weight=0.8",
            output_path=output_path,
            zoom_center=zoom_center,
            zoom_range_km=zoom_range_km,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug backward ray tracing visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic single-trace mode
  python backward_ray_tracer_debug.py

  # Convergence mode (iteratively refine direction)
  python backward_ray_tracer_debug.py --convergence

  # Convergence with custom parameters
  python backward_ray_tracer_debug.py --convergence --alpha 0.5 --tolerance 0.05

  # Custom wave conditions
  python backward_ray_tracer_debug.py --convergence -T 10 -D 270 --spread 25

  # Zoomed view
  python backward_ray_tracer_debug.py --convergence --zoom 400000 3720000 10

  # Detailed view: 5 rays showing ALL iteration attempts
  python backward_ray_tracer_debug.py --detailed
        """
    )

    # Mode selection
    parser.add_argument(
        '--convergence', '-c',
        action='store_true',
        help='Enable convergence mode (iteratively adjust direction until arrival matches partition)'
    )
    parser.add_argument(
        '--detailed', '-d',
        action='store_true',
        help='Detailed view: trace 5 rays and show ALL iteration paths for each'
    )

    # Wave conditions
    parser.add_argument(
        '--period', '-T',
        type=float,
        default=12.0,
        help='Wave period in seconds (default: 12.0)'
    )
    parser.add_argument(
        '--direction', '-D',
        type=float,
        default=285.0,
        help='Partition direction in degrees nautical (default: 285.0)'
    )
    parser.add_argument(
        '--spread',
        type=float,
        default=30.0,
        help='Directional spread in degrees (default: 30.0)'
    )

    # Ray tracing parameters
    parser.add_argument(
        '--n-rays', '-n',
        type=int,
        default=50,
        help='Number of rays to trace (default: 50)'
    )

    # Convergence parameters
    parser.add_argument(
        '--alpha', '-a',
        type=float,
        default=0.6,
        help='Relaxation factor for gradient descent (default: 0.6)'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=20,
        help='Maximum convergence iterations (default: 20)'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=0.05,
        help='Convergence tolerance as fraction of directional spread (default: 0.05 = 5%%)'
    )

    # View options
    parser.add_argument(
        '--zoom',
        nargs=3,
        type=float,
        metavar=('CENTER_X', 'CENTER_Y', 'RANGE_KM'),
        help='Zoom to specific area: CENTER_X CENTER_Y RANGE_KM (e.g., 475000 3660000 5)'
    )

    args = parser.parse_args()
    run_debug(args)
