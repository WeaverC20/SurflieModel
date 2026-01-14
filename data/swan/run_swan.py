#!/usr/bin/env python3
"""
SWAN Run Orchestrator

Executes SWAN wave model simulations for specified regions and meshes.
Handles the complete workflow:
1. Fetch WW3 boundary conditions
2. Generate SWAN INPUT file
3. Copy required files to run directory
4. Execute SWAN
5. Report results

Usage:
    python data/swan/run_swan.py --region socal --mesh coarse
    python data/swan/run_swan.py --region socal --mesh coarse --forecast-hour 24
    python data/swan/run_swan.py --region socal --mesh coarse --dry-run
"""

import argparse
import asyncio
import logging
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.swan.runner import (
    WW3BoundaryFetcher,
    BoundaryPoint,
    SwanInputGenerator,
    PhysicsSettings,
    BoundaryWaveParams,
    WindProvider,
    WindData,
)

logger = logging.getLogger(__name__)

# Directory structure
DATA_DIR = PROJECT_ROOT / "data"
MESHES_DIR = DATA_DIR / "meshes"
WW3_ENDPOINTS_DIR = DATA_DIR / "swan" / "ww3_endpoints"
RUNS_DIR = DATA_DIR / "swan" / "runs"


def get_mesh_dir(region: str, mesh: str) -> Path:
    """Get path to mesh directory."""
    mesh_dir = MESHES_DIR / region / mesh
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")
    return mesh_dir


def get_boundary_file(region: str, side: str = "west") -> Path:
    """Get path to WW3 boundary file."""
    boundary_file = WW3_ENDPOINTS_DIR / region / f"ww3_boundary_{side}.json"
    if not boundary_file.exists():
        raise FileNotFoundError(f"Boundary file not found: {boundary_file}")
    return boundary_file


def get_run_dir(region: str, mesh: str) -> Path:
    """Get path to run directory (latest)."""
    return RUNS_DIR / region / mesh / "latest"


class SwanRunner:
    """
    Orchestrates SWAN simulations.

    Handles the complete workflow from fetching boundary conditions
    to executing SWAN and collecting results.
    """

    def __init__(
        self,
        region: str,
        mesh: str,
        boundary_side: str = "west",
        forecast_hours: Optional[List[int]] = None,
        physics: Optional[PhysicsSettings] = None,
    ):
        """
        Initialize SWAN runner.

        Args:
            region: Region name (e.g., "socal")
            mesh: Mesh name (e.g., "coarse")
            boundary_side: Boundary side for WW3 data (default: "west")
            forecast_hours: Forecast hours to fetch (default: [0] for stationary)
            physics: Physics settings (uses defaults if None)
        """
        self.region = region
        self.mesh = mesh
        self.boundary_side = boundary_side
        self.forecast_hours = forecast_hours or [0]  # Single hour for stationary
        self.physics = physics or PhysicsSettings()

        # Resolve paths
        self.mesh_dir = get_mesh_dir(region, mesh)
        self.boundary_file = get_boundary_file(region, boundary_side)
        self.run_dir = get_run_dir(region, mesh)

        # Find mesh files
        self.mesh_name = self._find_mesh_name()
        self.bot_file = self.mesh_dir / f"{self.mesh_name}.bot"

        if not self.bot_file.exists():
            raise FileNotFoundError(f"Bathymetry file not found: {self.bot_file}")

        logger.info(f"SwanRunner initialized for {region}/{mesh}")

    def _find_mesh_name(self) -> str:
        """Find the mesh name from directory contents."""
        json_files = [f for f in self.mesh_dir.glob("*.json") if not f.name.startswith("ww3_")]
        if not json_files:
            raise FileNotFoundError(f"No mesh metadata found in {self.mesh_dir}")
        return json_files[0].stem

    async def fetch_boundary_conditions(self) -> List[BoundaryPoint]:
        """
        Fetch WW3 boundary conditions.

        Returns:
            List of BoundaryPoint objects with wave data
        """
        logger.info(f"Fetching WW3 boundary conditions for hours: {self.forecast_hours}")

        fetcher = WW3BoundaryFetcher()
        metadata, points = await fetcher.fetch_boundary(
            self.boundary_file,
            forecast_hours=self.forecast_hours
        )

        # Log summary of fetched data
        if points and points[0].hs:
            avg_hs = sum(p.hs[0] for p in points) / len(points)
            avg_tp = sum(p.tp[0] for p in points) / len(points)
            avg_dir = sum(p.dir[0] for p in points) / len(points)
            logger.info(f"Boundary conditions: avg Hs={avg_hs:.2f}m, Tp={avg_tp:.1f}s, Dir={avg_dir:.0f}°")

        return points

    def prepare_wind_data(self) -> WindData:
        """
        Prepare wind data from downloaded GFS files.

        Extracts wind for the mesh region and interpolates to mesh resolution.

        Returns:
            WindData object with interpolated wind
        """
        logger.info("Preparing wind data from GFS")

        # Load mesh metadata
        import json
        mesh_json = self.mesh_dir / f"{self.mesh_name}.json"
        with open(mesh_json) as f:
            mesh_metadata = json.load(f)

        # Extract and interpolate wind
        wind_provider = WindProvider()
        wind_data = wind_provider.extract_for_mesh(
            mesh_metadata,
            forecast_hour=self.forecast_hours[0]
        )

        # Write wind files to run directory
        wind_provider.write_swan_files(wind_data, self.run_dir)

        # Log summary
        logger.info(wind_provider.summary(wind_data))

        return wind_data

    def generate_input_file(
        self,
        boundary_points: List[BoundaryPoint],
        wind_data: Optional[WindData] = None
    ) -> Path:
        """
        Generate SWAN INPUT file for stationary mode.

        Args:
            boundary_points: List of BoundaryPoint objects with wave data
            wind_data: Optional WindData object for wind forcing

        Returns:
            Path to generated INPUT file
        """
        logger.info("Generating SWAN INPUT file")

        generator = SwanInputGenerator(
            mesh_dir=str(self.mesh_dir),
            boundary_file=str(self.boundary_file),
            physics=self.physics
        )

        # Calculate distances along boundary
        distances = generator._calculate_boundary_distances()

        # Convert boundary points to BoundaryWaveParams
        # Use first timestep (index 0) since we're in stationary mode
        wave_params = []
        for i, (point, distance) in enumerate(zip(boundary_points, distances)):
            params = BoundaryWaveParams(
                distance=distance,
                hs=point.hs[0],      # First timestep
                tp=point.tp[0],
                dir=point.dir[0],
                spread=point.spread[0]
            )
            wave_params.append(params)

        # Generate wind commands if wind data provided
        wind_commands = None
        if wind_data is not None:
            wind_provider = WindProvider()
            wind_commands = [
                wind_provider.generate_inpgrid_command(wind_data),
                wind_provider.generate_readinp_command()
            ]

        # Generate INPUT file (PAR syntax for stationary mode)
        input_path = generator.generate(
            output_dir=self.run_dir,
            wave_params=wave_params,
            wind_commands=wind_commands,
            project_name=f"{self.region}_{self.mesh}",
            run_id="001"
        )

        return input_path

    def copy_bathymetry(self) -> Path:
        """
        Copy bathymetry file to run directory.

        Returns:
            Path to copied file
        """
        dest_path = self.run_dir / self.bot_file.name
        shutil.copy2(self.bot_file, dest_path)
        logger.info(f"Copied bathymetry: {dest_path.name}")
        return dest_path

    def execute_swan(self) -> subprocess.CompletedProcess:
        """
        Execute SWAN in the run directory.

        Returns:
            CompletedProcess result
        """
        logger.info(f"Executing SWAN in {self.run_dir}")

        # Run SWAN (assumes 'swan' or 'swanrun' is in PATH)
        # Common SWAN executables: swan.exe, swanrun, swan
        result = subprocess.run(
            ["swan"],
            cwd=self.run_dir,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            logger.info("SWAN completed successfully")
        else:
            logger.error(f"SWAN failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")

        return result

    async def run(self, dry_run: bool = False) -> bool:
        """
        Execute complete SWAN workflow.

        Args:
            dry_run: If True, prepare files but don't execute SWAN

        Returns:
            True if successful, False otherwise
        """
        start_time = datetime.now()
        logger.info(f"Starting SWAN run for {self.region}/{self.mesh}")
        logger.info(f"Run directory: {self.run_dir}")

        # Clean and create run directory
        if self.run_dir.exists():
            shutil.rmtree(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Fetch boundary conditions
            boundary_points = await self.fetch_boundary_conditions()

            # Step 2: Prepare wind data
            wind_data = self.prepare_wind_data()

            # Step 3: Generate INPUT file (with wind)
            input_path = self.generate_input_file(boundary_points, wind_data)

            # Step 4: Copy bathymetry
            self.copy_bathymetry()

            # Step 5: Execute SWAN (unless dry run)
            if dry_run:
                logger.info("Dry run - skipping SWAN execution")
                logger.info(f"Run directory ready: {self.run_dir}")
                self._print_run_contents()
                return True

            result = self.execute_swan()

            # Step 6: Check results
            elapsed = (datetime.now() - start_time).total_seconds()

            if result.returncode == 0:
                logger.info(f"SWAN run completed in {elapsed:.1f}s")
                self._print_output_files()
                return True
            else:
                logger.error(f"SWAN run failed after {elapsed:.1f}s")
                return False

        except Exception as e:
            logger.error(f"Run failed: {e}")
            raise

    def _print_run_contents(self) -> None:
        """Print contents of run directory."""
        print("\nRun directory contents:")
        for f in sorted(self.run_dir.iterdir()):
            size = f.stat().st_size
            print(f"  {f.name} ({size:,} bytes)")

    def _print_output_files(self) -> None:
        """Print output files from SWAN run."""
        output_files = ["hsig.mat", "tps.mat", "dir.mat", "u.wind", "v.wind", "PRINT", "Errfile"]
        print("\nOutput files:")
        for name in output_files:
            path = self.run_dir / name
            if path.exists():
                size = path.stat().st_size
                print(f"  {name} ({size:,} bytes)")


async def run_swan(
    region: str,
    mesh: str,
    forecast_hours: Optional[List[int]] = None,
    dry_run: bool = False
) -> bool:
    """
    Convenience function to run SWAN.

    Args:
        region: Region name
        mesh: Mesh name
        forecast_hours: Forecast hours to fetch
        dry_run: If True, prepare but don't execute

    Returns:
        True if successful
    """
    runner = SwanRunner(
        region=region,
        mesh=mesh,
        forecast_hours=forecast_hours
    )
    return await runner.run(dry_run=dry_run)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run SWAN wave model simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data/swan/run_swan.py --region socal --mesh coarse
  python data/swan/run_swan.py --region socal --mesh coarse --forecast-hour 24
  python data/swan/run_swan.py --region socal --mesh coarse --dry-run
        """
    )

    parser.add_argument(
        "--region", "-r",
        required=True,
        help="Region name (e.g., socal, norcal, central)"
    )
    parser.add_argument(
        "--mesh", "-m",
        required=True,
        help="Mesh name (e.g., coarse, fine)"
    )
    parser.add_argument(
        "--forecast-hour", "-f",
        type=int,
        default=0,
        help="Forecast hour to run (default: 0 = current)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Prepare files but don't execute SWAN"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # Run
    try:
        success = asyncio.run(run_swan(
            region=args.region,
            mesh=args.mesh,
            forecast_hours=[args.forecast_hour],
            dry_run=args.dry_run
        ))

        sys.exit(0 if success else 1)

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()