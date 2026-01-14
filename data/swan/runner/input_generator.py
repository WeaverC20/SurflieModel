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
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BoundaryWaveParams:
    """Wave parameters at a single boundary point."""
    distance: float  # Distance along boundary (degrees for spherical coords)
    hs: float        # Significant wave height (m)
    tp: float        # Peak period (s)
    dir: float       # Wave direction (degrees, nautical convention)
    spread: float    # Directional spreading (degrees)


@dataclass
class PhysicsSettings:
    """
    Physics settings for SWAN simulation.

    Defaults are for wind-driven nearshore wave modeling:
    - GEN3 WESTH for wind-wave generation (van der Westhuysen formulation)
    - Bottom friction enabled (JONSWAP)
    - Depth-limited breaking enabled
    - Triads disabled (for swell propagation)

    The WESTH formulation is preferred over KOMEN for nearshore applications
    as it better handles mixed sea states (swell + wind sea).
    """
    # Bottom friction
    friction: str = "JONSWAP"
    friction_coefficient: float = 0.067

    # Wave breaking
    breaking: bool = True
    breaking_alpha: float = 1.0
    breaking_gamma: float = 0.73

    # GEN3 formulation: "WESTH" (recommended) or "KOMEN" (classic)
    gen3_formulation: str = "WESTH"

    # Triads (shallow water nonlinear interactions)
    triads: bool = False

    def to_swan_commands(self) -> List[str]:
        """Generate SWAN physics command strings."""
        commands = []

        # GEN3 enables wind-wave generation, whitecapping, and quadruplets
        # WESTH = van der Westhuysen formulation (better for mixed sea states)
        # KOMEN = classic Komen formulation
        commands.append(f"GEN3 {self.gen3_formulation.upper()}")

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
    - Boundary conditions (BOUNDSPEC with PAR parametric values)
    - Physics settings

    Example usage:
        generator = SwanInputGenerator(
            mesh_dir="data/meshes/socal/coarse",
            boundary_file="data/swan/ww3_endpoints/socal/ww3_boundary_west.json"
        )
        wave_params = [BoundaryWaveParams(...), ...]
        generator.generate(
            output_dir="data/swan/runs/socal/latest",
            wave_params=wave_params
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

        For spherical coordinates (COORD SPHE), SWAN expects distances
        in degrees along the boundary side.

        Returns:
            List of distances in degrees
        """
        points = self.boundary_metadata["points"]
        side = self._get_boundary_side_code()

        distances = []
        first_point = points[0]

        for point in points:
            if side in ["W", "E"]:
                # Vertical boundary - distance is latitude difference in degrees
                delta_lat = point[1] - first_point[1]
                distance = abs(delta_lat)
            else:
                # Horizontal boundary - distance is longitude difference in degrees
                delta_lon = point[0] - first_point[0]
                distance = abs(delta_lon)

            distances.append(distance)

        return distances

    def _generate_boundspec_command(self, wave_params: List[BoundaryWaveParams]) -> List[str]:
        """
        Generate BOUNDSPEC command with PAR (parametric values) for stationary mode.

        This embeds wave parameters directly in the INPUT file.
        Required for stationary mode in SWAN.

        Args:
            wave_params: List of BoundaryWaveParams (one per boundary point)

        Returns:
            List of SWAN command lines for boundary specification
        """
        if len(wave_params) != self.boundary_metadata["n_points"]:
            raise ValueError(
                f"Number of wave params ({len(wave_params)}) doesn't match "
                f"boundary points ({self.boundary_metadata['n_points']})"
            )

        side = self._get_boundary_side_code()

        # Build BOUNDSPEC command with PAR
        # BOUNDSPEC SIDE <side> CCW VARIABLE PAR &
        #     <dist1> <hs1> <per1> <dir1> <dd1> &
        #     <dist2> <hs2> <per2> <dir2> <dd2> ...
        lines = [f"BOUNDSPEC SIDE {side} CCW VARIABLE PAR &"]

        for i, params in enumerate(wave_params):
            # Distance in degrees needs more precision than meters
            param_str = f"    {params.distance:.4f} {params.hs:.2f} {params.tp:.1f} {params.dir:.1f} {params.spread:.1f}"
            # Last line doesn't need continuation
            if i == len(wave_params) - 1:
                lines.append(param_str)
            else:
                lines.append(f"{param_str} &")

        return lines

    def generate(
        self,
        output_dir: str | Path,
        wave_params: List[BoundaryWaveParams],
        wind_commands: Optional[List[str]] = None,
        project_name: Optional[str] = None,
        run_id: str = "001"
    ) -> Path:
        """
        Generate SWAN INPUT file for stationary mode.

        Args:
            output_dir: Directory to write INPUT file
            wave_params: List of BoundaryWaveParams (one per boundary point)
            wind_commands: List of SWAN wind commands (INPGRID WIND, READINP WIND)
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

        # Wind input (if provided)
        if wind_commands:
            lines.append("$ Wind forcing (GFS)")
            lines.extend(wind_commands)
            lines.append("")

        # Boundary conditions - using PAR syntax for stationary mode
        lines.append("$ Boundary conditions (WW3 parametric)")
        boundspec_lines = self._generate_boundspec_command(wave_params)
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

        # Output - ASCII format (LAY 3) for easy reading with numpy
        lines.append("$ Output")
        lines.append("BLOCK 'COMPGRID' NOHEAD 'hsig.mat' LAY 3 HSIG 1.")
        lines.append("BLOCK 'COMPGRID' NOHEAD 'tps.mat' LAY 3 TPS 1.")
        lines.append("BLOCK 'COMPGRID' NOHEAD 'dir.mat' LAY 3 DIR 1.")
        lines.append("")

        # Compute - just COMPUTE for stationary mode
        lines.append("$ Compute")
        lines.append("COMPUTE")
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
            f"  GEN3: {self.physics.gen3_formulation}",
            f"  Friction: {self.physics.friction} ({self.physics.friction_coefficient})",
            f"  Breaking: {self.physics.breaking}",
            f"  Triads: {self.physics.triads}",
        ]
        return '\n'.join(lines)