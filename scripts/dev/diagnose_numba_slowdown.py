#!/usr/bin/env python3
"""Diagnose Numba slowdown issues in surfzone simulation."""

import sys
import gc
import time
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Pin thread count before importing numba
os.environ.setdefault('NUMBA_NUM_THREADS', '4')

import psutil
import numpy as np


def check_numba_signatures():
    """Check for multiple Numba specializations (recompilation)."""
    print("\n=== Checking Numba Signatures ===")

    from data.surfzone.runner.backward_ray_tracer import (
        trace_backward_single,
        trace_all_parallel,
        _trace_single_partition_converge,
        compute_initial_direction_blended,
        find_nearest_boundary_direction,
    )
    from data.surfzone.runner.wave_physics import (
        local_wave_properties,
        update_ray_direction,
        deep_water_properties,
    )
    from data.surfzone.runner.ray_tracer import (
        interpolate_depth_indexed,
        celerity_gradient_indexed,
    )

    funcs = [
        ("trace_backward_single", trace_backward_single),
        ("trace_all_parallel", trace_all_parallel),
        ("_trace_single_partition_converge", _trace_single_partition_converge),
        ("compute_initial_direction_blended", compute_initial_direction_blended),
        ("find_nearest_boundary_direction", find_nearest_boundary_direction),
        ("local_wave_properties", local_wave_properties),
        ("update_ray_direction", update_ray_direction),
        ("deep_water_properties", deep_water_properties),
        ("interpolate_depth_indexed", interpolate_depth_indexed),
        ("celerity_gradient_indexed", celerity_gradient_indexed),
    ]

    for name, func in funcs:
        try:
            sigs = func.signatures
            n_sigs = len(sigs) if sigs else 0
            status = "⚠️  RECOMPILATION" if n_sigs > 1 else "✓" if n_sigs == 1 else "(not compiled)"
            print(f"  {name}: {n_sigs} signature(s) {status}")
            if n_sigs > 1:
                for sig in sigs:
                    print(f"    - {sig}")
        except AttributeError:
            print(f"  {name}: (not yet compiled)")


def check_nrt_allocations():
    """Check Numba Runtime allocations."""
    print("\n=== Checking NRT Allocations ===")
    try:
        from numba.core.runtime import nrt
        stats = nrt.rtsys.get_allocation_stats()
        print(f"  Allocations: {stats}")
    except Exception as e:
        print(f"  Could not get NRT stats: {e}")


def check_memory():
    """Check current memory usage."""
    process = psutil.Process()
    mem = process.memory_info()
    print(f"\n=== Memory Usage ===")
    print(f"  RSS: {mem.rss / 1024 / 1024:.1f} MB")
    print(f"  VMS: {mem.vms / 1024 / 1024:.1f} MB")
    return mem.rss


def check_numba_config():
    """Check Numba configuration."""
    print("\n=== Numba Configuration ===")
    import numba
    from numba import config
    print(f"  Numba version: {numba.__version__}")
    print(f"  NUMBA_NUM_THREADS: {os.environ.get('NUMBA_NUM_THREADS', 'not set')}")
    print(f"  Actual threads: {numba.get_num_threads()}")
    print(f"  Cache enabled: {config.CACHE_DIR}")
    print(f"  Debug mode: {config.DEBUG}")


def run_mini_simulation():
    """Run a small simulation and track performance across batches."""
    print("\n=== Running Mini Simulation ===")

    from data.surfzone.mesh import SurfZoneMesh
    from data.surfzone.runner.swan_input_provider import SwanInputProvider, BoundaryConditions, WavePartition
    from data.surfzone.runner.surfzone_runner import SurfzoneRunner, SurfzoneRunnerConfig

    # Load data
    mesh_dir = PROJECT_ROOT / "data" / "surfzone" / "meshes" / "socal"
    swan_dir = PROJECT_ROOT / "data" / "swan" / "runs" / "socal" / "coarse" / "latest"

    if not mesh_dir.exists():
        print(f"  ERROR: Mesh not found at {mesh_dir}")
        return
    if not swan_dir.exists():
        print(f"  ERROR: SWAN data not found at {swan_dir}")
        return

    print(f"  Loading mesh from {mesh_dir}...")
    mesh = SurfZoneMesh.load(mesh_dir)
    print(f"    Loaded {len(mesh.points_x)} mesh points")

    print(f"  Loading SWAN from {swan_dir}...")
    swan = SwanInputProvider(swan_dir)

    # Get boundary conditions
    arrays = mesh.get_numba_arrays()
    coast_distance = arrays.get('coast_distance', None)
    offshore_distance_m = arrays.get('offshore_distance_m', 0.0)

    if coast_distance is None:
        print("  ERROR: Mesh missing coast_distance")
        return

    boundary_threshold = offshore_distance_m * 0.90
    boundary_mask = coast_distance >= boundary_threshold
    boundary_indices = np.where(boundary_mask)[0][:500]  # Just 500 points

    boundary_x = arrays['points_x'][boundary_indices].copy()
    boundary_y = arrays['points_y'][boundary_indices].copy()
    lons, lats = mesh.utm_to_lon_lat(boundary_x, boundary_y)

    print(f"  Sampling SWAN at {len(lons)} boundary points...")
    sampled = swan.sample_at_points(lons, lats, use_nearest_fallback=True)

    partitions = []
    for part_id, data in sampled.items():
        hs = data['hs']
        tp = data['tp']
        direction = data['dir']
        is_valid = ~np.isnan(hs) & (hs > 0) & ~np.isnan(tp) & (tp > 0)
        partitions.append(WavePartition(
            hs=hs, tp=tp, direction=direction,
            partition_id=part_id, is_valid=is_valid,
        ))

    boundary_conditions = BoundaryConditions(
        x=boundary_x, y=boundary_y, lon=lons, lat=lats, partitions=partitions,
    )

    # Run with timing per batch
    config = SurfzoneRunnerConfig(
        min_depth=0.0,
        max_depth=10.0,
        partition_id=1,
        sample_count=1500,  # 1500 points for quick test
        random_seed=42,
    )

    runner = SurfzoneRunner(mesh, boundary_conditions, config)

    print("\n  Running simulation with batch timing...")
    mem_before = check_memory()
    check_nrt_allocations()

    # Get filtered points
    x, y, depths, _ = runner.get_filtered_points()

    np.random.seed(42)
    n_sample = min(1500, len(x))
    sample_idx = np.random.choice(len(x), n_sample, replace=False)

    batch_size = 150
    batch_times = []
    batch_memory = []

    print(f"\n  Processing {n_sample} points in batches of {batch_size}...")

    for batch_num, batch_start in enumerate(range(0, len(sample_idx), batch_size)):
        batch_end = min(batch_start + batch_size, len(sample_idx))
        batch_idx = sample_idx[batch_start:batch_end]

        gc.collect()  # Force GC before timing
        mem_start = psutil.Process().memory_info().rss
        t_start = time.perf_counter()

        for i in batch_idx:
            runner.run_single_point(x[i], y[i], depths[i])

        elapsed = time.perf_counter() - t_start
        mem_end = psutil.Process().memory_info().rss

        rate = len(batch_idx) / elapsed
        mem_delta = (mem_end - mem_start) / 1024 / 1024

        batch_times.append(rate)
        batch_memory.append(mem_delta)

        print(f"    Batch {batch_num + 1:2d}: {rate:6.1f} pts/sec, mem delta: {mem_delta:+.1f} MB")

    # Check for degradation
    print("\n  === Performance Analysis ===")

    if len(batch_times) >= 4:
        first_quarter = np.mean(batch_times[:len(batch_times)//4])
        last_quarter = np.mean(batch_times[-len(batch_times)//4:])
        degradation = (first_quarter - last_quarter) / first_quarter * 100

        print(f"  First quarter avg:  {first_quarter:.1f} pts/sec")
        print(f"  Last quarter avg:   {last_quarter:.1f} pts/sec")
        print(f"  Degradation:        {degradation:+.1f}%")

        if degradation > 15:
            print("  ⚠️  SIGNIFICANT DEGRADATION DETECTED")
        elif degradation > 5:
            print("  ⚠️  Minor degradation detected")
        else:
            print("  ✓ Performance stable within run")

    # Memory analysis
    total_mem_growth = sum(batch_memory)
    print(f"\n  Total memory growth: {total_mem_growth:+.1f} MB over {n_sample} points")
    print(f"  Memory per point:    {total_mem_growth * 1024 / n_sample:.2f} KB/point")

    print("\n  After simulation:")
    mem_after = check_memory()
    print(f"  Total memory growth: {(mem_after - mem_before) / 1024 / 1024:.1f} MB")

    check_nrt_allocations()
    check_numba_signatures()


def main():
    print("=" * 60)
    print("Numba Slowdown Diagnostic")
    print("=" * 60)

    check_numba_config()
    check_memory()
    check_numba_signatures()
    check_nrt_allocations()

    run_mini_simulation()

    print("\n" + "=" * 60)
    print("Diagnostic Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
