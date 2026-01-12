"""
Region Registry

Central registry for discovering and loading region configurations.
"""

from pathlib import Path
from typing import Dict, List, Optional
import yaml

from .models import (
    Region,
    GeoBounds,
    WW3SourceConfig,
    WW3ExtractionPoints,
    ChannelIsland,
    BathymetrySource,
    SurfSpot,
    Domain,
    DomainType,
    DomainMesh,
    GridType,
    SpectralConfig,
    SwanPhysics,
)


# Default regions directory relative to project root
DEFAULT_REGIONS_DIR = Path("regions")


class RegionRegistry:
    """
    Central registry for all configured regions.

    Singleton pattern ensures consistent region access across the application.

    Usage:
        registry = RegionRegistry()
        california = registry.load_region("california")
    """

    _instance: Optional["RegionRegistry"] = None
    _regions: Dict[str, Region]
    _base_path: Path

    def __new__(cls, base_path: Optional[Path] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._regions = {}
            cls._instance._base_path = base_path or cls._find_project_root() / DEFAULT_REGIONS_DIR
        return cls._instance

    @staticmethod
    def _find_project_root() -> Path:
        """Find project root by looking for regions/ directory."""
        current = Path.cwd()
        while current != current.parent:
            if (current / "regions").exists():
                return current
            current = current.parent
        # Fallback to cwd
        return Path.cwd()

    def discover_regions(self) -> List[str]:
        """
        Find all valid region directories.

        A valid region has a region.yaml file.
        """
        regions = []
        if not self._base_path.exists():
            return regions

        for path in self._base_path.iterdir():
            if path.is_dir() and path.name != "_templates":
                if (path / "region.yaml").exists():
                    regions.append(path.name)
        return sorted(regions)

    def load_region(self, name: str, force_reload: bool = False) -> Region:
        """
        Load and cache a region configuration.

        Args:
            name: Region name (directory name under regions/)
            force_reload: If True, reload even if cached

        Returns:
            Loaded Region object

        Raises:
            FileNotFoundError: If region directory or config doesn't exist
            ValueError: If region configuration is invalid
        """
        if name in self._regions and not force_reload:
            return self._regions[name]

        region_path = self._base_path / name
        config_path = region_path / "region.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Region config not found: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        region = self._parse_region(name, region_path, config)
        self._regions[name] = region
        return region

    def _parse_region(
        self, name: str, region_path: Path, config: Dict
    ) -> Region:
        """Parse region configuration from YAML dict."""

        # Parse bounds
        bounds_data = config["bounds"]
        bounds = GeoBounds(
            lat_min=bounds_data["lat_min"],
            lat_max=bounds_data["lat_max"],
            lon_min=bounds_data["lon_min"],
            lon_max=bounds_data["lon_max"],
        )

        # Parse sub-regions
        sub_regions = {}
        for sr_name, sr_data in config.get("sub_regions", {}).items():
            sr_bounds = sr_data.get("bounds", {})
            sub_regions[sr_name] = GeoBounds(
                lat_min=sr_bounds.get("lat_min", bounds.lat_min),
                lat_max=sr_bounds.get("lat_max", bounds.lat_max),
                lon_min=sr_bounds.get("lon_min", bounds.lon_min),
                lon_max=sr_bounds.get("lon_max", bounds.lon_max),
            )

        # Parse WW3 source
        ww3_source = WW3SourceConfig.from_dict(config.get("ww3_source", {}))

        # Parse Channel Islands
        channel_islands = [
            ChannelIsland.from_dict(ci)
            for ci in config.get("channel_islands", [])
        ]

        # Load extraction points
        extraction_points = None
        boundaries_config = config.get("boundaries", {})
        if boundaries_config.get("ww3_extraction"):
            points_path = region_path / boundaries_config["ww3_extraction"]
            if points_path.exists():
                extraction_points = WW3ExtractionPoints.load(points_path)
                # Validate point count
                expected_count = boundaries_config.get("extraction_point_count")
                if expected_count and len(extraction_points.points) != expected_count:
                    raise ValueError(
                        f"Extraction point count mismatch: "
                        f"expected {expected_count}, got {len(extraction_points.points)}"
                    )

        # Parse bathymetry sources
        bathymetry_sources = [
            BathymetrySource.from_dict(bs)
            for bs in config.get("bathymetry_sources", [])
        ]

        # Parse surf spots
        surf_spots = [
            SurfSpot.from_dict(spot)
            for spot in config.get("surf_spots", [])
        ]

        # Parse domains
        domains = {}
        domains_config = config.get("domains", {})
        for domain_type_str, domain_meta in domains_config.items():
            domain_type = DomainType(domain_type_str)
            domain_path = region_path / domain_meta.get("path", f"domains/{domain_type_str}")
            domain_config_path = domain_path / "domain.yaml"

            if domain_config_path.exists():
                domain = self._parse_domain(domain_type, domain_path, domain_config_path)
                domains[domain_type] = domain

        return Region(
            name=name,
            display_name=config.get("display_name", name.title()),
            version=config.get("version", "1.0.0"),
            base_path=region_path,
            bounds=bounds,
            sub_regions=sub_regions,
            ww3_source=ww3_source,
            channel_islands=channel_islands,
            extraction_points=extraction_points,
            domains=domains,
            bathymetry_sources=bathymetry_sources,
            surf_spots=surf_spots,
        )

    def _parse_domain(
        self,
        domain_type: DomainType,
        domain_path: Path,
        config_path: Path,
    ) -> Domain:
        """Parse domain configuration from YAML file."""

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        grid = config.get("grid", {})
        bounds = GeoBounds(
            lat_min=grid.get("lat_min", 0),
            lat_max=grid.get("lat_max", 0),
            lon_min=grid.get("lon_min", 0),
            lon_max=grid.get("lon_max", 0),
        )

        # Parse meshes
        meshes = {}
        for mesh_version, mesh_data in config.get("meshes", {}).items():
            meshes[mesh_version] = DomainMesh.from_dict(mesh_version, mesh_data)

        # Parse spectral config
        boundaries = config.get("boundaries", {})
        spectral_data = boundaries.get("spectral_config", {})
        spectral_config = SpectralConfig(
            n_frequencies=spectral_data.get("n_frequencies", 36),
            freq_min_hz=spectral_data.get("freq_min_hz", 0.04),
            freq_max_hz=spectral_data.get("freq_max_hz", 1.0),
            n_directions=spectral_data.get("n_directions", 36),
            spreading_degrees=spectral_data.get("spreading_degrees", 25.0),
            spectral_shape=spectral_data.get("spectral_shape", "JONSWAP"),
        )

        # Parse SWAN physics
        swan_physics = SwanPhysics.from_dict(config.get("swan_physics", {}))

        return Domain(
            name=config.get("name", ""),
            region_name=config.get("region", ""),
            domain_type=domain_type,
            version=config.get("version", "1.0.0"),
            description=config.get("description", ""),
            bounds=bounds,
            grid_type=GridType(grid.get("type", "regular")),
            resolution_m=grid.get("resolution_m", 0),
            n_lat=grid.get("n_lat", 0),
            n_lon=grid.get("n_lon", 0),
            n_cells=grid.get("n_cells", 0),
            n_wet_cells=grid.get("n_wet_cells", 0),
            meshes=meshes,
            active_mesh_version=config.get("active_mesh"),
            spectral_config=spectral_config,
            swan_physics=swan_physics,
            boundary_source=boundaries.get("source", "ww3"),
            extraction_points_ref=boundaries.get("extraction_points_ref"),
        )

    def get_all_regions(self) -> Dict[str, Region]:
        """Load and return all discovered regions."""
        for name in self.discover_regions():
            if name not in self._regions:
                self.load_region(name)
        return self._regions.copy()

    def clear_cache(self):
        """Clear the region cache."""
        self._regions.clear()

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (mainly for testing)."""
        cls._instance = None
