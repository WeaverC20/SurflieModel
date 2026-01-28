#!/usr/bin/env python3
"""
Debug script for investigating ray tracing behavior.

Run after run_surfzone.py to analyze why rays aren't breaking.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.surfzone.mesh import SurfZoneMesh
from data.surfzone.runner.swan_input_provider import SwanInputProvider
from data.surfzone.runner.ray_tracer import RayTracer
from data.surfzone.runner.output_writer import load_breaking_field
from data.surfzone.run_surfzone import (
    get_surfzone_mesh_dir, get_swan_run_dir, get_wind_data,
    SurfzoneRunner
)
from data.regions.region import get_region


def analyze_termination_reasons(results):
    """Count all termination reasons."""
    reasons = {}
    for r in results:
        reason = r.termination_reason
        reasons[reason] = reasons.get(reason, 0) + 1

    print("\nTermination Reasons:")
    print("-" * 40)
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(results)
        print(f"  {reason}: {count} ({pct:.1f}%)")
    return reasons


def analyze_wave_directions(boundary, mesh):
    """Check if wave directions point shoreward."""
    print("\nWave Direction Analysis:")
    print("-" * 40)

    for p in boundary.partitions:
        if not np.any(p.is_valid):
            continue

        valid_dirs = p.direction[p.is_valid]

        # SWAN directions are nautical (FROM direction, clockwise from N)
        # For SoCal, waves coming FROM ~270° (W) to ~315° (NW) are typical
        print(f"\n  {p.label}:")
        print(f"    Direction range: {valid_dirs.min():.0f}° - {valid_dirs.max():.0f}°")
        print(f"    Mean direction: {np.mean(valid_dirs):.0f}°")

        # Check if directions are reasonable for SoCal
        # Waves should generally come from W-NW (225-315°)
        onshore_mask = (valid_dirs > 180) & (valid_dirs < 360)
        pct_onshore = 100 * np.sum(onshore_mask) / len(valid_dirs)
        print(f"    From offshore (180-360°): {pct_onshore:.0f}%")


def trace_sample_rays_with_details(runner, boundary, n_samples=5):
    """Trace a few rays with detailed logging."""
    print("\nSample Ray Details:")
    print("-" * 40)

    # Get wind
    wind_speed, wind_dir = get_wind_data(runner.mesh)

    # Initialize tracer with path storage
    tracer = RayTracer(
        runner.mesh,
        step_size=runner.step_size,
        max_steps=runner.max_steps,
        min_depth=runner.min_depth,
    )

    # Find valid boundary points with wave data
    for p in boundary.partitions:
        if not np.any(p.is_valid):
            continue

        valid_indices = np.where(p.is_valid)[0][:n_samples]

        print(f"\n{p.label} - tracing {len(valid_indices)} sample rays:")

        for idx in valid_indices:
            x0, y0 = boundary.x[idx], boundary.y[idx]
            H0, T, theta = p.hs[idx], p.tp[idx], p.direction[idx]

            result = tracer.trace_single(
                x0, y0, H0, T, theta,
                wind_speed, wind_dir,
                partition_id=p.partition_id,
                store_path=True,
            )

            print(f"\n  Ray from ({x0:.0f}, {y0:.0f}):")
            print(f"    Initial: H={H0:.2f}m, T={T:.1f}s, θ={theta:.0f}°")
            print(f"    Termination: {result.termination_reason}")

            if result.did_break:
                print(f"    BROKE at depth {result.break_depth:.2f}m, H={result.break_height:.2f}m")

            if result.path_x is not None and len(result.path_x) > 1:
                # Calculate total distance traveled
                dx = np.diff(result.path_x)
                dy = np.diff(result.path_y)
                total_dist = np.sum(np.sqrt(dx**2 + dy**2))
                print(f"    Distance traveled: {total_dist:.0f}m")

                # Direction of travel
                net_dx = result.path_x[-1] - result.path_x[0]
                net_dy = result.path_y[-1] - result.path_y[0]
                travel_dir = np.degrees(np.arctan2(net_dx, net_dy)) % 360
                print(f"    Net travel direction: {travel_dir:.0f}° (N=0, E=90)")

        break  # Just first partition for now


def plot_boundary_and_mesh(runner, boundary, results=None):
    """Visualize boundary points on the mesh."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get region bounds and convert to UTM for axis limits
    region = get_region(runner.region)
    lon_min, lon_max = region.lon_range
    lat_min, lat_max = region.lat_range

    # Convert corners to UTM
    x_min, y_min = runner.mesh.lon_lat_to_utm(
        np.array([lon_min]), np.array([lat_min])
    )
    x_max, y_max = runner.mesh.lon_lat_to_utm(
        np.array([lon_max]), np.array([lat_max])
    )
    x_min, y_min = x_min[0], y_min[0]
    x_max, y_max = x_max[0], y_max[0]

    # Plot mesh outline (coastlines)
    if runner.mesh.coastlines:
        for coastline in runner.mesh.coastlines:
            ax.plot(coastline[:, 0], coastline[:, 1], 'k-', linewidth=0.5, alpha=0.5)

    # Plot boundary points
    ax.scatter(boundary.x, boundary.y, c='blue', s=20, alpha=0.7, label='Boundary points')

    # Plot breaking points if we have them
    if results:
        break_x = [r.break_x for r in results if r.did_break]
        break_y = [r.break_y for r in results if r.did_break]
        if break_x:
            ax.scatter(break_x, break_y, c='red', s=50, marker='*', label='Breaking points')

    # Add wave direction arrows at boundary
    for p in boundary.partitions:
        if not np.any(p.is_valid):
            continue

        valid = p.is_valid
        # Convert nautical direction (FROM) to math direction (TO)
        # Nautical: 0=N, 90=E, clockwise, FROM
        # Math: 0=E, counterclockwise, TO
        # If wave is FROM 270° (W), it travels TO 90° (E)
        math_dir = np.radians(90 - (p.direction[valid] + 180))

        # Arrow components
        dx = np.cos(math_dir)
        dy = np.sin(math_dir)

        # Scale by wave height for visibility
        scale = p.hs[valid] * 500

        ax.quiver(
            boundary.x[valid], boundary.y[valid],
            dx * scale, dy * scale,
            alpha=0.3, color='green',
            scale=1, scale_units='xy'
        )
        break  # Just first partition

    # Set axis limits to region bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal')
    ax.set_xlabel('UTM X (m)')
    ax.set_ylabel('UTM Y (m)')
    ax.set_title(f'Surfzone Boundary Points and Wave Directions - {region.display_name}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'data' / 'surfzone' / 'debug_boundary.png', dpi=150)
    print(f"\nSaved: data/surfzone/debug_boundary.png")
    plt.close()


def plot_sample_ray_paths(runner, boundary, n_rays=20):
    """Plot actual ray paths."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Get region bounds and convert to UTM for axis limits
    region = get_region(runner.region)
    lon_min, lon_max = region.lon_range
    lat_min, lat_max = region.lat_range

    # Convert corners to UTM
    x_min, y_min = runner.mesh.lon_lat_to_utm(
        np.array([lon_min]), np.array([lat_min])
    )
    x_max, y_max = runner.mesh.lon_lat_to_utm(
        np.array([lon_max]), np.array([lat_max])
    )
    x_min, y_min = x_min[0], y_min[0]
    x_max, y_max = x_max[0], y_max[0]

    # Plot coastlines
    if runner.mesh.coastlines:
        for coastline in runner.mesh.coastlines:
            ax.plot(coastline[:, 0], coastline[:, 1], 'k-', linewidth=1)

    # Get wind
    wind_speed, wind_dir = get_wind_data(runner.mesh)

    # Initialize tracer
    tracer = RayTracer(
        runner.mesh,
        step_size=runner.step_size,
        max_steps=runner.max_steps,
        min_depth=runner.min_depth,
    )

    colors = {'broke': 'red', 'reached_shore': 'orange', 'left_domain': 'gray', 'max_steps': 'blue'}

    ray_count = 0
    for p in boundary.partitions:
        if not np.any(p.is_valid):
            continue

        valid_indices = np.where(p.is_valid)[0]
        # Sample evenly across valid indices
        step = max(1, len(valid_indices) // n_rays)
        sample_indices = valid_indices[::step][:n_rays]

        for idx in sample_indices:
            x0, y0 = boundary.x[idx], boundary.y[idx]
            H0, T, theta = p.hs[idx], p.tp[idx], p.direction[idx]

            result = tracer.trace_single(
                x0, y0, H0, T, theta,
                wind_speed, wind_dir,
                partition_id=p.partition_id,
                store_path=True,
            )

            if result.path_x is not None and len(result.path_x) > 1:
                color = colors.get(result.termination_reason, 'purple')
                ax.plot(result.path_x, result.path_y, '-', color=color, alpha=0.6, linewidth=1)

                # Mark start
                ax.plot(result.path_x[0], result.path_y[0], 'o', color=color, markersize=4)

                # Mark breaking point
                if result.did_break:
                    ax.plot(result.break_x, result.break_y, '*', color='red', markersize=10)

            ray_count += 1
            if ray_count >= n_rays:
                break

        if ray_count >= n_rays:
            break

    # Legend
    for reason, color in colors.items():
        ax.plot([], [], '-', color=color, label=reason)
    ax.plot([], [], '*', color='red', markersize=10, label='breaking point')

    # Set axis limits to region bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal')
    ax.set_xlabel('UTM X (m)')
    ax.set_ylabel('UTM Y (m)')
    ax.set_title(f'Sample Ray Paths ({ray_count} rays) - {region.display_name}')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / 'data' / 'surfzone' / 'debug_ray_paths.png', dpi=150)
    print(f"Saved: data/surfzone/debug_ray_paths.png")
    plt.close()


def main():
    region = "socal"

    print("=" * 60)
    print("SURFZONE RAY TRACING DEBUG")
    print("=" * 60)

    # Initialize runner
    runner = SurfzoneRunner(
        region=region,
        swan_resolution="fine",
        step_size=50.0,  # Fast mode
        max_steps=500,
    )

    # Get boundary conditions
    boundary = runner.get_boundary_conditions(boundary_spacing_m=1000.0)

    # Analyze wave directions
    analyze_wave_directions(boundary, runner.mesh)

    # Trace sample rays with details
    trace_sample_rays_with_details(runner, boundary, n_samples=3)

    # Full run to get all results
    print("\n" + "=" * 60)
    print("Running full trace for termination analysis...")
    print("=" * 60)

    wind_speed, wind_dir = get_wind_data(runner.mesh)
    tracer = RayTracer(
        runner.mesh,
        step_size=runner.step_size,
        max_steps=runner.max_steps,
        min_depth=runner.min_depth,
    )

    results = tracer.trace_from_boundary(
        boundary,
        U_wind=wind_speed,
        wind_direction=wind_dir,
        store_paths=False,
    )

    analyze_termination_reasons(results)

    # Visualizations
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    plot_boundary_and_mesh(runner, boundary, results)
    plot_sample_ray_paths(runner, boundary, n_rays=30)

    print("\nDone! Check the PNG files in data/surfzone/")


if __name__ == "__main__":
    main()
