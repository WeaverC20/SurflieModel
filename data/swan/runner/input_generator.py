"""
SWAN Input File Generator

Generates SWAN INPUT files for stationary wave model runs.
Combines mesh bathymetry, WW3 boundary conditions, and physics settings.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Meters per degree of latitude (approximately constant)
METERS_PER_DEG_LAT = 111_000


@dataclass
class PhysicsSettings:
    """
    Physics settings for SWAN simulation.

    These are reasonable defaults for nearshore wave modeling.
    Can be customized per-run if needed.
    """
    # Bottom friction
    friction: str = "JONSWAP"
    friction_coefficient: float = 0.067

    # Wave breaking
    breaking: bool = True
    breaking_alpha: float = 1.0
    breaking_gamma: float = 0.73

    # Whitecapping
    whitecapping: bool = True

    # Triads (shallow water nonlinear interactions)
    triads: bool = True

    def to_swan_commands(self) -> List[str]:
        """Generate SWAN physics command strings."""
        commands = []

        # Friction
        if self.friction.upper() == "JONSWAP":
            commands.append(f"FRICTION JONSWAP {self.friction_coefficient}")
        elif self.friction.upper() == "COLLINS":
            commands.append(f"FRICTION COLLINS {self.friction_coefficient}")
        elif self.friction.upper() == "MADSEN":
            commands.append(f"FRICTION MADSEN {self.friction_coefficient}")

        # Breaking
        if self.breaking:
            commands.append(f"BREAKING CONSTANT {self.breaking_alpha} {self.breaking_gamma}")

        # Whitecapping
        if self.whitecapping:
            commands.append("WHITECAPPING KOMEN")

        # Triads
        if self.triads:
            commands.append("TRIAD")

        return commands


@dataclass
class SwanInputGenerator:
    """
    Generates SWAN INPUT files for stationary runs.

    Combines:
    - Mesh (CGRID, INPGRID, READINP for bathymetry)
    - Boundary conditions (BOUNDSPEC with TPAR files)
    - Physics settings

    Example usage:
        generator = SwanInputGenerator(
            mesh_dir="data/meshes/socal/coarse",
            boundary_file="data/swan/ww3_endpoints/socal/ww3_boundary_west.json"
        )
        generator.generate(
            output_dir="data/swan/runs/socal/latest",
            tpar_files=["west_000.tpar", "west_001.tpar", ...]
        )
    """

    mesh_dir: str
    boundary_file: str
    physics: PhysicsSettings = field(default_factory=PhysicsSettings)

    # Loaded data (populated by load())
    mesh_metadata: Optional[Dict] = None
    boundary_metadata: Optional[Dict] = None

    def __post_init__(self):
        self.mesh_dir = Path(self.mesh_dir)
        self.boundary_file = Path(self.boundary_file)
        self._load()

    def _load(self) -> None:
        """Load mesh and boundary metadata."""
        # Find and load mesh JSON
        json_files = list(self.mesh_dir.glob("*.json"))
        # Filter out ww3 boundary files
        mesh_jsons = [f for f in json_files if not f.name.startswith("ww3_")]
        if not mesh_jsons:
            raise FileNotFoundError(f"No mesh metadata found in {self.mesh_dir}")

        with open(mesh_jsons[0]) as f:
            self.mesh_metadata = json.load(f)

        logger.info(f"Loaded mesh: {self.mesh_metadata['name']}")

        # Load boundary metadata
        with open(self.boundary_file) as f:
            self.boundary_metadata = json.load(f)

        logger.info(f"Loaded boundary: {self.boundary_metadata['boundary']['side']} side, "
                   f"{self.boundary_metadata['n_points']} points")

    def _get_boundary_side_code(self) -> str:
        """Get SWAN boundary side code (N/S/E/W)."""
        side = self.boundary_metadata["boundary"]["side"].upper()
        if side in ["NORTH", "N"]:
            return "N"
        elif side in ["SOUTH", "S"]:
            return "S"
        elif side in ["EAST", "E"]:
            return "E"
        elif side in ["WEST", "W"]:
            return "W"
        else:
            raise ValueError(f"Unknown boundary side: {side}")

    def _calculate_boundary_distances(self) -> List[float]:
        """
        Calculate cumulative distances along boundary for each point.

        SWAN requires distances in meters from the first corner of the side.

        Returns:
            List of distances in meters
        """
        points = self.boundary_metadata["points"]
        side = self._get_boundary_side_code()

        distances = []
        first_point = points[0]

        for point in points:
            if side in ["W", "E"]:
                # Vertical boundary - distance is in latitude direction
                delta_lat = point[1] - first_point[1]
                distance = abs(delta_lat) * METERS_PER_DEG_LAT
            else:
                # Horizontal boundary - distance is in longitude direction
                # Need to account for longitude spacing varying with latitude
                center_lat = (point[1] + first_point[1]) / 2
                meters_per_deg_lon = METERS_PER_DEG_LAT * np.cos(np.radians(center_lat))
                delta_lon = point[0] - first_point[0]
                distance = abs(delta_lon) * meters_per_deg_lon

            distances.append(distance)

        return distances

    def generate_boundspec_command(self, tpar_files: List[str]) -> List[str]:
        """
        Generate BOUNDSPEC command with VARIABLE TPAR files.

        Args:
            tpar_files: List of TPAR filenames (one per boundary point)

        Returns:
            List of SWAN command lines for boundary specification
        """
        if len(tpar_files) != self.boundary_metadata["n_points"]:
            raise ValueError(
                f"Number of TPAR files ({len(tpar_files)}) doesn't match "
                f"boundary points ({self.boundary_metadata['n_points']})"
            )

        side = self._get_boundary_side_code()
        distances = self._calculate_boundary_distances()

        # Build BOUNDSPEC command
        # BOUNDSPEC SIDE <side> CCW VARIABLE FILE &
        #     <dist1> '<file1>' &
        #     <dist2> '<file2>' ...
        lines = [f"BOUNDSPEC SIDE {side} CCW VARIABLE FILE &"]

        for i, (dist, tpar_file) in enumerate(zip(distances, tpar_files)):
            # Last line doesn't need continuation
            if i == len(tpar_files) - 1:
                lines.append(f"    {dist:.1f} '{tpar_file}'")
            else:
                lines.append(f"    {dist:.1f} '{tpar_file}' &")

        return lines

    def generate(
        self,
        output_dir: str | Path,
        tpar_files: List[str],
        project_name: Optional[str] = None,
        run_id: str = "001"
    ) -> Path:
        """
        Generate complete SWAN INPUT file.

        Args:
            output_dir: Directory to write INPUT file
            tpar_files: List of TPAR boundary condition filenames
            project_name: Project name (default: mesh name)
            run_id: Run identifier (default: "001")

        Returns:
            Path to generated INPUT file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if project_name is None:
            project_name = self.mesh_metadata["name"]

        input_path = output_dir / "INPUT"

        lines = []

        # Header
        lines.append(f"$ SWAN INPUT file generated {datetime.now().isoformat()}")
        lines.append(f"$ Mesh: {self.mesh_metadata['name']}")
        lines.append(f"$ Region: {self.mesh_metadata['region_name']}")
        lines.append("$")
        lines.append("")

        # Project
        lines.append(f"PROJECT '{project_name}' '{run_id}'")
        lines.append("")

        # Mode - stationary
        lines.append("$ Stationary mode")
        lines.append("MODE STAT")
        lines.append("")

        # Coordinates - spherical
        lines.append("$ Spherical coordinates (lat/lon)")
        lines.append("COORD SPHE")
        lines.append("")

        # Computational grid
        lines.append("$ Computational grid")
        lines.append(self.mesh_metadata["swan_commands"]["cgrid"])
        lines.append("")

        # Bathymetry input
        lines.append("$ Bathymetry")
        lines.append(self.mesh_metadata["swan_commands"]["inpgrid"])
        lines.append(self.mesh_metadata["swan_commands"]["readinp"])
        lines.append("")

        # Boundary conditions
        lines.append("$ Boundary conditions (WW3 TPAR format)")
        boundspec_lines = self.generate_boundspec_command(tpar_files)
        lines.extend(boundspec_lines)
        lines.append("")

        # Physics
        lines.append("$ Physics")
        physics_commands = self.physics.to_swan_commands()
        lines.extend(physics_commands)
        lines.append("")

        # Numerical settings
        lines.append("$ Numerical settings")
        lines.append("NUM ACCUR 0.02 0.02 0.02 95 STAT 50")
        lines.append("")

        # Output - table output at all grid points
        lines.append("$ Output")
        lines.append("OUTPUT OPTIONS BLOCK 4")
        lines.append("BLOCK 'COMPGRID' NOHEAD 'hsig.mat' LAY 4 HSIG 1.")
        lines.append("BLOCK 'COMPGRID' NOHEAD 'tps.mat' LAY 4 TPS 1.")
        lines.append("BLOCK 'COMPGRID' NOHEAD 'dir.mat' LAY 4 DIR 1.")
        lines.append("")

        # Compute
        lines.append("$ Compute")
        lines.append("COMPUTE STAT")
        lines.append("")

        # Stop
        lines.append("STOP")
        lines.append("")

        # Write file
        with open(input_path, 'w') as f:
            f.write('\n'.join(lines))

        logger.info(f"Generated SWAN INPUT file: {input_path}")
        return input_path

    def summary(self) -> str:
        """Return summary of input configuration."""
        lines = [
            "SWAN Input Generator",
            f"  Mesh: {self.mesh_metadata['name']}",
            f"  Region: {self.mesh_metadata['region_name']}",
            f"  Grid: {self.mesh_metadata['nx']} x {self.mesh_metadata['ny']}",
            f"  Resolution: {self.mesh_metadata['resolution_km']} km",
            f"  Boundary: {self.boundary_metadata['boundary']['side']} side",
            f"  Boundary points: {self.boundary_metadata['n_points']}",
            "",
            "Physics:",
            f"  Friction: {self.physics.friction} ({self.physics.friction_coefficient})",
            f"  Breaking: {self.physics.breaking}",
            f"  Whitecapping: {self.physics.whitecapping}",
            f"  Triads: {self.physics.triads}",
        ]
        return '\n'.join(lines)


def generate_swan_input(
    mesh_dir: str | Path,
    boundary_file: str | Path,
    output_dir: str | Path,
    tpar_files: List[str],
    physics: Optional[PhysicsSettings] = None
) -> Path:
    """
    Convenience function to generate SWAN INPUT file.

    Args:
        mesh_dir: Directory containing mesh files
        boundary_file: Path to boundary JSON file
        output_dir: Directory for output
        tpar_files: List of TPAR boundary condition filenames
        physics: Physics settings (uses defaults if None)

    Returns:
        Path to generated INPUT file
    """
    if physics is None:
        physics = PhysicsSettings()

    generator = SwanInputGenerator(
        mesh_dir=str(mesh_dir),
        boundary_file=str(boundary_file),
        physics=physics
    )

    return generator.generate(output_dir, tpar_files)


# CLI for testing
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Generate SWAN INPUT file")
    parser.add_argument("mesh_dir", help="Directory containing mesh files")
    parser.add_argument("boundary_file", help="Path to boundary JSON file")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory")
    parser.add_argument("--tpar-prefix", default="west", help="TPAR file prefix")
    parser.add_argument("--n-points", type=int, default=11, help="Number of boundary points")

    args = parser.parse_args()

    # Generate TPAR file list
    tpar_files = [f"{args.tpar_prefix}_{i:03d}.tpar" for i in range(args.n_points)]

    generator = SwanInputGenerator(
        mesh_dir=args.mesh_dir,
        boundary_file=args.boundary_file
    )

    print(generator.summary())
    print()

    input_path = generator.generate(args.output_dir, tpar_files)
    print(f"\nGenerated: {input_path}")