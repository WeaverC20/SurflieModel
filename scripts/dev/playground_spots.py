#!/usr/bin/env python3
"""
Playground script for testing surf spot forecast extraction.

Run from project root:
    python scripts/dev/playground_spots.py
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.spots import SPOTS, SOCAL_SPOTS, get_spot
from data.swan.analysis.output_reader import SwanOutputReader


def main():
    # Load the latest SWAN output
    run_dir = project_root / "data" / "swan" / "runs" / "socal" / "coarse" / "latest"

    print("=" * 60)
    print("SURF SPOT FORECAST EXTRACTION TEST")
    print("=" * 60)
    print(f"\nLoading SWAN output from: {run_dir}")

    reader = SwanOutputReader(run_dir)
    swan_output = reader.read()

    print(f"\nSWAN Output Summary:")
    print(f"  Grid size: {swan_output.hsig.shape} (ny, nx)")
    print(f"  Lon range: [{swan_output.lons[0]:.2f}, {swan_output.lons[-1]:.2f}] ({len(swan_output.lons)} points)")
    print(f"  Lat range: [{swan_output.lats[0]:.2f}, {swan_output.lats[-1]:.2f}] ({len(swan_output.lats)} points)")
    print(f"  Exception value (land): {swan_output.exception_value}")
    print(f"  Partitions available: {len(swan_output.partitions)}")
    for p in swan_output.partitions:
        print(f"    - {p.label} (id={p.partition_id})")

    print("\n" + "=" * 60)
    print("AVAILABLE SPOTS")
    print("=" * 60)
    for spot in SOCAL_SPOTS:
        print(f"  {spot.name}: {spot.display_name} ({spot.lat:.4f}, {spot.lon:.4f})")

    print("\n" + "=" * 60)
    print("EXTRACTING FORECASTS (DETAILED)")
    print("=" * 60)

    for spot in SOCAL_SPOTS:
        print(f"\n{'='*50}")
        print(f"  {spot.display_name}")
        print(f"{'='*50}")
        print(f"  Spot location: ({spot.lat:.4f}, {spot.lon:.4f})")

        # Find nearest grid point manually for diagnostics
        i_nearest = np.argmin(np.abs(swan_output.lons - spot.lon))
        j_nearest = np.argmin(np.abs(swan_output.lats - spot.lat))
        nearest_lon = swan_output.lons[i_nearest]
        nearest_lat = swan_output.lats[j_nearest]
        nearest_hsig = swan_output.hsig[j_nearest, i_nearest]

        print(f"\n  Nearest grid point: i={i_nearest}, j={j_nearest}")
        print(f"  Nearest location: ({nearest_lat:.4f}, {nearest_lon:.4f})")
        print(f"  Nearest Hsig: {nearest_hsig:.3f} m {'(LAND/NaN)' if np.isnan(nearest_hsig) else ''}")

        # Get forecast using the new nearby-search method
        forecast = spot.get_forecast(swan_output)

        # Check if we had to move from nearest
        if forecast.grid_i != i_nearest or forecast.grid_j != j_nearest:
            used_lon = swan_output.lons[forecast.grid_i]
            used_lat = swan_output.lats[forecast.grid_j]
            print(f"\n  -> Moved to valid point: i={forecast.grid_i}, j={forecast.grid_j}")
            print(f"     Valid location: ({used_lat:.4f}, {used_lon:.4f})")
            print(f"     Distance from spot: {forecast.distance_km:.2f} km")
        else:
            print(f"  -> Using nearest point (valid)")

        # Show integrated parameters at the used point
        i, j = forecast.grid_i, forecast.grid_j
        hsig_val = swan_output.hsig[j, i]
        tps_val = swan_output.tps[j, i]
        dir_val = swan_output.dir[j, i]

        print(f"\n  Integrated parameters at used point:")
        print(f"    Hsig: {hsig_val:.3f} m")
        print(f"    Tps:  {tps_val:.3f} s")
        print(f"    Dir:  {dir_val:.1f}°")

        # Show ALL partitions
        print(f"\n  Partition data:")
        for hs, tp, direction, label in forecast.partitions:
            # Determine status
            if np.isnan(hs) or hs == -99.0:
                status = "LAND/INVALID"
            elif hs <= 0:
                status = "NO ENERGY"
            else:
                status = "OK"

            print(f"    {label}:")
            print(f"      Hs={hs:.3f}m, Tp={tp:.2f}s, Dir={direction:.1f}° [{status}]")


if __name__ == "__main__":
    main()
