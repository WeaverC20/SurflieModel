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
from typing import List, Optional

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.swan.runner import WW3BoundaryFetcher, SwanInputGenerator, PhysicsSettings

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

    async def fetch_boundary_conditions(self) -> List[Path]:
        """
        Fetch WW3 boundary conditions and write TPAR files.

        Returns:
            List of paths to generated TPAR files
        """
        logger.info(f"Fetching WW3 boundary conditions for hours: {self.forecast_hours}")

        fetcher = WW3BoundaryFetcher()
        metadata, points = await fetcher.fetch_boundary(
            self.boundary_file,
            forecast_hours=self.forecast_hours
        )

        # Write TPAR files to run directory
        tpar_files = fetcher.write_tpar_files(
            points,
            self.run_dir,
            filename_prefix=self.boundary_side
        )

        logger.info(f"Wrote {len(tpar_files)} TPAR files to {self.run_dir}")
        return tpar_files

    def generate_input_file(self, tpar_files: List[Path]) -> Path:
        """
        Generate SWAN INPUT file.

        Args:
            tpar_files: List of TPAR file paths

        Returns:
            Path to generated INPUT file
        """
        logger.info("Generating SWAN INPUT file")

        generator = SwanInputGenerator(
            mesh_dir=str(self.mesh_dir),
            boundary_file=str(self.boundary_file),
            physics=self.physics
        )

        # Use just filenames (not full paths) for INPUT file
        tpar_filenames = [f.name for f in tpar_files]

        input_path = generator.generate(
            output_dir=self.run_dir,
            tpar_files=tpar_filenames,
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
            tpar_files = await self.fetch_boundary_conditions()

            # Step 2: Generate INPUT file
            input_path = self.generate_input_file(tpar_files)

            # Step 3: Copy bathymetry
            self.copy_bathymetry()

            # Step 4: Execute SWAN (unless dry run)
            if dry_run:
                logger.info("Dry run - skipping SWAN execution")
                logger.info(f"Run directory ready: {self.run_dir}")
                self._print_run_contents()
                return True

            result = self.execute_swan()

            # Step 5: Check results
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
        output_files = ["hsig.mat", "tps.mat", "dir.mat", "PRINT", "Errfile"]
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