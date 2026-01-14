#!/usr/bin/env python3
"""
Generalized Mesh Generator

Creates SWAN meshes by sampling from GEBCO bathymetry data.
This module provides the core generation functionality used by
region-specific generator scripts.

Usage:
    from data.meshes.generate_mesh import generate_mesh

    mesh = generate_mesh(
        name="socal_coarse",
        region_name="socal",
        resolution_km=5.0,
        output_dir="data/meshes/socal",
    )
"""

import sys
from pathlib import Path
from typing import Optional, Union

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data.bathymetry.gebco import GEBCOBathymetry
from data.regions import Mesh, get_region, Region


def generate_mesh(
    name: str,
    region_name: Optional[str] = None,
    region: Optional[Region] = None,
    resolution_km: float = 5.0,
    output_dir: Optional[Union[str, Path]] = None,
    gebco: Optional[GEBCOBathymetry] = None,
    plot: bool = False,
    save_plot: bool = False,
) -> Mesh:
    """
    Generate a SWAN mesh by sampling from GEBCO bathymetry.

    Args:
        name: Name for the mesh (e.g., "socal_coarse")
        region_name: Name of predefined region (e.g., "socal", "norcal", "central")
        region: Region object (alternative to region_name)
        resolution_km: Grid resolution in kilometers (default 5.0)
        output_dir: Directory to save mesh files (optional)
        gebco: GEBCOBathymetry instance (loads if not provided)
        plot: Whether to display the mesh plot
        save_plot: Whether to save the plot to output_dir

    Returns:
        Generated Mesh object

    Example:
        mesh = generate_mesh(
            name="socal_coarse",
            region_name="socal",
            resolution_km=5.0,
            output_dir="data/meshes/socal",
        )
    """
    # Get region
    if region is None and region_name is None:
        raise ValueError("Must provide either region_name or region")

    if region is None:
        region = get_region(region_name)

    # Load GEBCO if not provided
    if gebco is None:
        print("Loading GEBCO bathymetry...")
        gebco = GEBCOBathymetry()

    # Create mesh
    print(f"\nGenerating mesh '{name}' for {region.display_name}...")
    mesh = Mesh(
        name=name,
        region=region,
        resolution_km=resolution_km,
    )

    # Sample from GEBCO
    mesh.from_gebco(gebco)

    # Print summary
    print(f"\n{mesh.summary()}")

    # Print SWAN commands
    print(f"\nSWAN Commands:")
    print(f"  {mesh.generate_inpgrid_command()}")
    print(f"  {mesh.generate_readinp_command(f'{name}.bot')}")

    # Save if output directory specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        mesh.save(output_dir)

        # Save plot if requested
        if save_plot:
            plot_path = output_dir / f"{name}.png"
            mesh.plot(save_path=plot_path, show=False)

    # Show plot if requested
    if plot:
        mesh.plot(show=True)

    return mesh


def load_mesh(directory: Union[str, Path], name: Optional[str] = None) -> Mesh:
    """
    Load a previously saved mesh.

    Args:
        directory: Directory containing mesh files
        name: Mesh name (optional if only one mesh in directory)

    Returns:
        Loaded Mesh object
    """
    return Mesh.load(directory, name)


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Generate a SWAN mesh from GEBCO")
    parser.add_argument("--name", "-n", required=True, help="Mesh name")
    parser.add_argument("--region", "-r", required=True, help="Region name (socal, norcal, central)")
    parser.add_argument("--resolution", "-res", type=float, default=5.0, help="Resolution in km")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--plot", "-p", action="store_true", help="Display plot")
    parser.add_argument("--save-plot", action="store_true", help="Save plot to output dir")

    args = parser.parse_args()

    mesh = generate_mesh(
        name=args.name,
        region_name=args.region,
        resolution_km=args.resolution,
        output_dir=args.output,
        plot=args.plot,
        save_plot=args.save_plot,
    )