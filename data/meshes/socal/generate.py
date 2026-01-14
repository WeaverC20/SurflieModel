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

Output Structure:
    data/meshes/socal/
    ├── generate.py              # This script
    └── coarse/                  # Coarse mesh (5km)
        ├── socal_coarse.bot     # SWAN bathymetry
        ├── socal_coarse.json    # Mesh metadata
        └── socal_coarse.png     # Visualization

Available Meshes:
    - coarse: 5km resolution (default)
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
BASE_DIR = Path(__file__).parent  # data/meshes/socal/

# Mesh configurations - each gets its own subfolder
MESH_CONFIGS = {
    "coarse": {
        "name": "socal_coarse",
        "resolution_km": 5.0,
        "description": "Coarse mesh for initial testing and rapid iteration",
        "folder": "coarse",
    },
    # Future mesh configurations can be added here:
    # "medium": {
    #     "name": "socal_medium",
    #     "resolution_km": 2.5,
    #     "description": "Medium resolution for production runs",
    #     "folder": "medium",
    # },
    # "fine": {
    #     "name": "socal_fine",
    #     "resolution_km": 1.0,
    #     "description": "Fine resolution for detailed nearshore modeling",
    #     "folder": "fine",
    # },
}

def get_mesh_dir(mesh_type: str) -> Path:
    """Get the output directory for a mesh type."""
    if mesh_type not in MESH_CONFIGS:
        raise ValueError(f"Unknown mesh type: {mesh_type}")
    return BASE_DIR / MESH_CONFIGS[mesh_type]["folder"]


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
    output_dir = get_mesh_dir(mesh_type)

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
    print(f"Output: {output_dir}")
    print(f"=" * 60)

    mesh = generate_mesh(
        name=name,
        region_name=REGION_NAME,
        resolution_km=resolution,
        output_dir=output_dir,
        plot=plot,
        save_plot=save_plot,
    )

    return mesh


def load_socal_mesh(mesh_type: str = "coarse", name: str = None):
    """
    Load a previously generated SoCal mesh.

    Args:
        mesh_type: Type of mesh ("coarse", "medium", "fine")
        name: Mesh name (optional, uses config name by default)

    Returns:
        Loaded Mesh object
    """
    if mesh_type not in MESH_CONFIGS:
        raise ValueError(f"Unknown mesh type: {mesh_type}")

    mesh_dir = get_mesh_dir(mesh_type)

    # Use configured name if not specified
    if name is None:
        name = MESH_CONFIGS[mesh_type]["name"]

    return load_mesh(mesh_dir, name)


def list_meshes() -> dict:
    """List all available meshes and their status."""
    status = {}
    for mesh_type, config in MESH_CONFIGS.items():
        mesh_dir = get_mesh_dir(mesh_type)
        json_files = list(mesh_dir.glob("*.json")) if mesh_dir.exists() else []
        mesh_files = [f for f in json_files if not f.name.startswith("ww3_")]

        status[mesh_type] = {
            "config": config,
            "exists": len(mesh_files) > 0,
            "path": mesh_dir,
            "files": [f.name for f in mesh_dir.iterdir()] if mesh_dir.exists() else [],
        }
    return status


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
        "--list",
        action="store_true",
        help="List available mesh configurations"
    )

    args = parser.parse_args()

    if args.list:
        print("SoCal Mesh Configurations:")
        print("=" * 60)
        for mesh_type, info in list_meshes().items():
            status = "EXISTS" if info["exists"] else "not generated"
            print(f"\n{mesh_type}:")
            print(f"  Name: {info['config']['name']}")
            print(f"  Resolution: {info['config']['resolution_km']} km")
            print(f"  Description: {info['config']['description']}")
            print(f"  Status: {status}")
            print(f"  Path: {info['path']}")
            if info['files']:
                print(f"  Files: {', '.join(info['files'])}")
        return

    if args.load:
        # Load existing mesh
        mesh = load_socal_mesh(args.type)
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