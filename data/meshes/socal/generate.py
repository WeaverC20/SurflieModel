#!/usr/bin/env python3
"""
Southern California Mesh Generator

Generates SWAN meshes for the Southern California region.
This script defines SoCal-specific parameters and calls the
generalized mesh generator.

Usage:
    python data/meshes/socal/generate.py
    python data/meshes/socal/generate.py --plot
    python data/meshes/socal/generate.py --resolution 2.5

Region Bounds:
    Latitude:  32.0°N to 34.5°N (Mexico border to Point Conception)
    Longitude: 121.0°W to 117.0°W

Available Meshes:
    - socal_coarse: 5km resolution (default)
"""

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
from data.meshes.generate_mesh import generate_mesh, load_mesh


# =============================================================================
# SoCal Configuration
# =============================================================================

REGION_NAME = "socal"
OUTPUT_DIR = Path(__file__).parent  # data/meshes/socal/

# Mesh configurations
MESH_CONFIGS = {
    "coarse": {
        "name": "socal_coarse",
        "resolution_km": 5.0,
        "description": "Coarse mesh for initial testing and rapid iteration",
    },
    # Future mesh configurations can be added here:
    # "medium": {
    #     "name": "socal_medium",
    #     "resolution_km": 2.5,
    #     "description": "Medium resolution for production runs",
    # },
    # "fine": {
    #     "name": "socal_fine",
    #     "resolution_km": 1.0,
    #     "description": "Fine resolution for detailed nearshore modeling",
    # },
}


def generate_socal_mesh(
    mesh_type: str = "coarse",
    resolution_km: float = None,
    plot: bool = False,
    save_plot: bool = True,
):
    """
    Generate a SoCal mesh.

    Args:
        mesh_type: Type of mesh ("coarse", "medium", "fine")
        resolution_km: Override resolution (optional)
        plot: Whether to display the plot
        save_plot: Whether to save the plot

    Returns:
        Generated Mesh object
    """
    if mesh_type not in MESH_CONFIGS:
        available = list(MESH_CONFIGS.keys())
        raise ValueError(f"Unknown mesh type: {mesh_type}. Available: {available}")

    config = MESH_CONFIGS[mesh_type]

    # Allow resolution override
    resolution = resolution_km if resolution_km is not None else config["resolution_km"]

    # Generate mesh name based on resolution if overridden
    if resolution_km is not None and resolution_km != config["resolution_km"]:
        name = f"socal_{resolution_km}km"
    else:
        name = config["name"]

    print(f"=" * 60)
    print(f"Generating SoCal Mesh: {name}")
    print(f"Description: {config['description']}")
    print(f"=" * 60)

    mesh = generate_mesh(
        name=name,
        region_name=REGION_NAME,
        resolution_km=resolution,
        output_dir=OUTPUT_DIR,
        plot=plot,
        save_plot=save_plot,
    )

    return mesh


def load_socal_mesh(name: str = None):
    """
    Load a previously generated SoCal mesh.

    Args:
        name: Mesh name (e.g., "socal_coarse"). If None, loads the only mesh
              in the directory or raises an error if multiple exist.

    Returns:
        Loaded Mesh object
    """
    return load_mesh(OUTPUT_DIR, name)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SWAN mesh for Southern California"
    )
    parser.add_argument(
        "--type", "-t",
        choices=list(MESH_CONFIGS.keys()),
        default="coarse",
        help="Mesh type to generate (default: coarse)"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=float,
        help="Override resolution in km (e.g., 2.5)"
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Display the mesh plot"
    )
    parser.add_argument(
        "--no-save-plot",
        action="store_true",
        help="Don't save the plot image"
    )
    parser.add_argument(
        "--load", "-l",
        action="store_true",
        help="Load existing mesh instead of generating"
    )
    parser.add_argument(
        "--name", "-n",
        help="Mesh name for loading (optional)"
    )

    args = parser.parse_args()

    if args.load:
        # Load existing mesh
        mesh = load_socal_mesh(args.name)
        print(f"\n{mesh.summary()}")
        if args.plot:
            mesh.plot(show=True)
    else:
        # Generate new mesh
        mesh = generate_socal_mesh(
            mesh_type=args.type,
            resolution_km=args.resolution,
            plot=args.plot,
            save_plot=not args.no_save_plot,
        )

    print("\nDone!")
    return mesh


if __name__ == "__main__":
    main()