#!/usr/bin/env python3
"""
Debug script for visualizing BACKWARD wave propagation physics.

Traces waves BACKWARD from near-shore points toward the open ocean,
showing how rays bend toward FASTER celerity (deeper water).

Physics (reversed from forward):
1. Local wavelength: L = L₀ · [tanh((2πh/L₀)^0.75)]^(2/3)  (same)
2. Local celerity: C = L / T  (same)
3. Ray refraction: dθ/ds = +(1/C) · ∂C/∂n  (FLIPPED SIGN - bends toward faster C / deeper water)
4. Start near shore, trace toward boundary
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from numba import njit

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.surfzone.runner.wave_physics import (
    deep_water_properties,
    local_wave_properties,
    nautical_to_math,
    update_ray_direction,
)
from data.surfzone.runner.ray_tracer import (
    interpolate_depth_indexed,
    celerity_gradient_indexed,
)


# =============================================================================
# Core Forward Ray Tracing with Clear Physics
# =============================================================================

@njit(cache=True)
def trace_ray_backward(
    start_x: float,
    start_y: float,
    T: float,
    direction_nautical: float,
    # Mesh data
    points_x: np.ndarray,
    points_y: np.ndarray,
    depth: np.ndarray,
    triangles: np.ndarray,
    # Spatial index
    grid_x_min: float,
    grid_y_min: float,
    grid_cell_size: float,
    grid_n_cells_x: int,
    grid_n_cells_y: int,
    grid_cell_starts: np.ndarray,
    grid_cell_counts: np.ndarray,
    grid_triangles: np.ndarray,
    # Config
    step_size: float = 10.0,
    max_steps: int = 2000,
    boundary_depth: float = 50.0,  # Stop when we reach this depth (deep water)
):
    """
    Trace a single wave ray BACKWARD from near-shore toward open ocean.

    Physics (TIME-REVERSED):
    1. Wavelength: L = f(L₀, h) - same equation
    2. Celerity: C = L/T - same equation
    3. Ray bends toward FASTER C (deeper water): dθ/ds = +(1/C)·∂C/∂n
       This is achieved by NEGATING the celerity gradient.
    4. Step AWAY from shore (opposite of wave travel direction)

    Returns:
        path_x, path_y: Ray path coordinates
        path_depth: Water depth along path
        path_celerity: Celerity along path
        reached_boundary: True if ray reached deep water
        termination: 0=boundary, 1=left_domain, 2=max_steps
    """
    # Allocate path storage
    path_x = np.empty(max_steps, dtype=np.float64)
    path_y = np.empty(max_steps, dtype=np.float64)
    path_depth = np.empty(max_steps, dtype=np.float64)
    path_celerity = np.empty(max_steps, dtype=np.float64)

    # Deep water reference properties
    L0, C0, Cg0 = deep_water_properties(T)

    # Convert direction to math convention
    # This gives the direction the wave WAS traveling (toward shore)
    theta = nautical_to_math(direction_nautical)

    # BACKWARD: Start with direction pointing AWAY from shore (opposite of wave travel)
    # We negate the direction to trace backward in time
    dx = -np.cos(theta)  # NEGATED for backward tracing
    dy = -np.sin(theta)  # NEGATED for backward tracing

    x, y = start_x, start_y
    termination = 2  # 0=reached_boundary, 1=left_domain, 2=max_steps

    for step in range(max_steps):
        # Store current position
        path_x[step] = x
        path_y[step] = y

        # Get depth at current position
        h = interpolate_depth_indexed(
            x, y, points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles
        )

        # Check if we left the domain
        if np.isnan(h) or h <= 0:
            path_x = path_x[:step]
            path_y = path_y[:step]
            path_depth = path_depth[:step]
            path_celerity = path_celerity[:step]
            termination = 1
            return path_x, path_y, path_depth, path_celerity, termination

        path_depth[step] = h

        # =====================================================================
        # PHYSICS EQUATION 1: Local wave properties from depth
        # L = L₀ · [tanh((2πh/L₀)^0.75)]^(2/3)  (Fenton & McKee 1990)
        # C = L / T
        # =====================================================================
        L, k, C, n, Cg = local_wave_properties(L0, T, h)
        path_celerity[step] = C

        # =====================================================================
        # Check if we reached the boundary (deep water)
        # =====================================================================
        if h >= boundary_depth:
            termination = 0  # Reached boundary
            path_x = path_x[:step + 1]
            path_y = path_y[:step + 1]
            path_depth = path_depth[:step + 1]
            path_celerity = path_celerity[:step + 1]
            return path_x, path_y, path_depth, path_celerity, termination

        # =====================================================================
        # PHYSICS EQUATION 2: Ray refraction (BACKWARD)
        #
        # Forward: dθ/ds = -(1/C) · ∂C/∂n  (bends toward slower C)
        # Backward: dθ/ds = +(1/C) · ∂C/∂n  (bends toward FASTER C)
        #
        # We achieve this by NEGATING the gradient before passing to
        # update_ray_direction, which keeps the standard formula.
        # =====================================================================
        dC_dx, dC_dy = celerity_gradient_indexed(
            x, y, T, L0, points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles
        )

        # NEGATE gradients for backward tracing - this makes rays bend toward
        # FASTER celerity (deeper water) instead of slower celerity
        dx, dy = update_ray_direction(dx, dy, C, -dC_dx, -dC_dy, step_size)

        # =====================================================================
        # BACKWARD STEP: Move away from shore (toward deep water)
        # =====================================================================
        x += dx * step_size
        y += dy * step_size

    # Max steps reached
    path_x = path_x[:max_steps]
    path_y = path_y[:max_steps]
    path_depth = path_depth[:max_steps]
    path_celerity = path_celerity[:max_steps]
    return path_x, path_y, path_depth, path_celerity, termination


# =============================================================================
# Visualization
# =============================================================================

def plot_backward_rays(
    mesh,
    rays_data,
    title="Backward Wave Propagation - Physics Debug",
    output_path=None,
):
    """
    Plot backward ray paths with depth coloring.
    Creates both a full view and a zoomed coastal view.
    """
    # Collect ray data
    all_x = np.concatenate([r['path_x'] for r in rays_data if len(r['path_x']) > 0])
    all_y = np.concatenate([r['path_y'] for r in rays_data if len(r['path_y']) > 0])
    all_depths = []
    for ray in rays_data:
        if len(ray['path_depth']) > 0:
            all_depths.extend(ray['path_depth'])

    if all_depths:
        vmin, vmax = min(all_depths), max(all_depths)
    else:
        vmin, vmax = 0, 50

    termination_colors = {
        0: 'green',    # reached boundary (success)
        1: 'red',      # left domain
        2: 'orange',   # max_steps
    }

    # Get start points for zooming to coastal area
    start_x = [r['path_x'][0] for r in rays_data if len(r['path_x']) > 0]
    start_y = [r['path_y'][0] for r in rays_data if len(r['path_y']) > 0]

    # Create figure with 3 subplots: zoomed coast, full view, depth profile
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Helper function to plot rays on an axis
    def plot_rays_on_axis(ax, xlim=None, ylim=None):
        # Plot coastlines
        if mesh.coastlines:
            for coastline in mesh.coastlines:
                ax.plot(coastline[:, 0], coastline[:, 1], 'k-', linewidth=2, alpha=0.9)

        # Plot each ray
        for ray in rays_data:
            path_x = ray['path_x']
            path_y = ray['path_y']
            path_depth = ray['path_depth']
            term = ray['termination']

            if len(path_x) < 2:
                continue

            # Create line segments for coloring
            points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Color by depth
            colors = path_depth[:-1]

            lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(vmin, vmax))
            lc.set_array(colors)
            lc.set_linewidth(3)  # Thicker lines
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

    # LEFT PLOT: Zoomed view of coastal area (near start points)
    ax1 = axes[0]
    # Zoom to area around start points with some padding
    x_center = np.mean(start_x)
    y_center = np.mean(start_y)
    zoom_range = 15000  # 15km box
    plot_rays_on_axis(
        ax1,
        xlim=(x_center - zoom_range, x_center + zoom_range),
        ylim=(y_center - zoom_range, y_center + zoom_range)
    )
    ax1.set_title('ZOOMED: Coastal Area\n(Red dots = ray start points)')

    # Add colorbar to zoomed plot
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cbar1 = plt.colorbar(sm, ax=ax1, shrink=0.8)
    cbar1.set_label('Depth (m)')

    # MIDDLE PLOT: Full view
    ax2 = axes[1]
    x_pad = (all_x.max() - all_x.min()) * 0.1
    y_pad = (all_y.max() - all_y.min()) * 0.1
    plot_rays_on_axis(
        ax2,
        xlim=(all_x.min() - x_pad, all_x.max() + x_pad),
        ylim=(all_y.min() - y_pad, all_y.max() + y_pad)
    )
    ax2.set_title('FULL VIEW: All Ray Paths\n(Shore → Boundary)')

    # Legend for termination
    for term, color in termination_colors.items():
        label = {0: 'Reached boundary', 1: 'Left domain', 2: 'Max steps'}[term]
        ax2.plot([], [], 's', color=color, markersize=10, label=label)
    ax2.legend(loc='upper left')

    # RIGHT PLOT: Depth vs distance
    ax3 = axes[2]
    for ray in rays_data:
        path_x = ray['path_x']
        path_y = ray['path_y']
        path_depth = ray['path_depth']
        term = ray['termination']

        if len(path_x) < 2:
            continue

        # Calculate distance along ray
        dx = np.diff(path_x)
        dy = np.diff(path_y)
        distances = np.concatenate([[0], np.cumsum(np.sqrt(dx**2 + dy**2))])

        color = termination_colors.get(term, 'gray')
        ax3.plot(distances, path_depth, '-', color=color, alpha=0.6, linewidth=2)

    ax3.set_xlabel('Distance along ray (m)')
    ax3.set_ylabel('Water Depth (m)')
    ax3.set_title('Depth vs Distance\n(Rays trend shallow → deep)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=50, color='blue', linestyle='--', linewidth=2, label='Boundary (50m)')
    ax3.legend(loc='lower right')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {output_path}")

    # Show interactive plot - user can zoom/pan
    print("\n[Interactive plot window opened - use toolbar to zoom/pan]")
    plt.show(block=True)


def run_debug():
    """Run the backward ray tracing debug visualization."""
    from data.surfzone.mesh import SurfZoneMesh

    print("=" * 60)
    print("BACKWARD WAVE PROPAGATION - PHYSICS DEBUG")
    print("=" * 60)
    print("\nTracing rays BACKWARD from shore toward open ocean.")
    print("Rays should bend toward FASTER celerity (deeper water).")

    # Load mesh
    print("\nLoading mesh...")
    mesh_dir = PROJECT_ROOT / "data" / "surfzone" / "meshes" / "socal"
    mesh = SurfZoneMesh.load(mesh_dir)

    # Get numba arrays
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

    # Define test wave conditions
    T = 12.0  # seconds (period)
    direction = 285.0  # degrees nautical (FROM) - wave was coming from WNW

    print(f"\nWave conditions:")
    print(f"  T = {T} s (period)")
    print(f"  Direction = {direction}° (wave was coming FROM WNW)")

    # Calculate deep water properties for reference
    L0, C0, Cg0 = deep_water_properties(T)
    print(f"\nDeep water properties:")
    print(f"  L0 = {L0:.1f} m (wavelength)")
    print(f"  C0 = {C0:.1f} m/s (celerity)")
    print(f"  Cg0 = {Cg0:.1f} m/s (group velocity)")

    # Find NEAR-SHORE points (shallow water) to start backward tracing
    # Use points where depth is 2-10m (typical breaking/surfzone depths)
    shallow_mask = (depth > 2) & (depth < 10)
    shallow_indices = np.where(shallow_mask)[0]

    if len(shallow_indices) == 0:
        print("No suitable shallow points found!")
        return

    print(f"\nFound {len(shallow_indices)} shallow water points (2-10m depth)")

    # Sample shallow points for ray tracing
    n_rays = min(100, len(shallow_indices))
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(shallow_indices, size=n_rays, replace=False)

    print(f"Tracing {n_rays} rays BACKWARD from shallow water toward boundary...")

    rays_data = []
    termination_counts = {0: 0, 1: 0, 2: 0}

    for i, idx in enumerate(sample_indices):
        start_x = points_x[idx]
        start_y = points_y[idx]

        path_x, path_y, path_depth, path_celerity, termination = trace_ray_backward(
            start_x, start_y, T, direction,
            points_x, points_y, depth, triangles,
            grid_x_min, grid_y_min, grid_cell_size,
            grid_n_cells_x, grid_n_cells_y,
            grid_cell_starts, grid_cell_counts, grid_triangles,
            step_size=15.0,
            max_steps=3000,
            boundary_depth=50.0,
        )

        rays_data.append({
            'path_x': path_x,
            'path_y': path_y,
            'path_depth': path_depth,
            'path_celerity': path_celerity,
            'termination': termination,
        })

        termination_counts[termination] += 1

        if (i + 1) % 20 == 0:
            print(f"  Traced {i + 1}/{n_rays} rays")

    # Print summary
    print(f"\nResults:")
    print(f"  Reached boundary (50m depth): {termination_counts[0]}")
    print(f"  Left domain: {termination_counts[1]}")
    print(f"  Max steps: {termination_counts[2]}")

    # Depth statistics for successful rays
    successful_rays = [r for r in rays_data if r['termination'] == 0]
    if successful_rays:
        start_depths = [r['path_depth'][0] for r in successful_rays]
        end_depths = [r['path_depth'][-1] for r in successful_rays]
        print(f"\nDepth change (successful rays):")
        print(f"  Start depth: {np.mean(start_depths):.1f} ± {np.std(start_depths):.1f} m")
        print(f"  End depth: {np.mean(end_depths):.1f} ± {np.std(end_depths):.1f} m")
        print(f"  Rays correctly moved from shallow → deep water!")

    # Plot
    print("\nGenerating visualization...")
    output_path = PROJECT_ROOT / "data" / "surfzone" / "debug_backward_rays.png"
    plot_backward_rays(
        mesh, rays_data,
        title=f"Backward Wave Propagation: T={T}s, Original Dir={direction}°",
        output_path=output_path,
    )


if __name__ == "__main__":
    run_debug()
