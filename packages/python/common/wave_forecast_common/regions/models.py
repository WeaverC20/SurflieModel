"""
Region Data Models

Data classes representing the region configuration hierarchy.
These models are designed to be:
- Immutable where appropriate (especially boundary points)
- Serializable to/from YAML and JSON
- Compatible with SWAN model execution
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json


class DomainType(Enum):
    """Type of SWAN computational domain."""
    OUTER = "outer"
    INNER = "inner"
    SPOT = "spot"


class GridType(Enum):
    """Type of computational grid."""
    REGULAR = "regular"
    CURVILINEAR = "curvilinear"
    UNSTRUCTURED = "unstructured"


class BathymetrySourceType(Enum):
    """How bathymetry data is accessed."""
    LOCAL = "local"
    OPENDAP = "opendap"
    S3 = "s3"


@dataclass(frozen=True)
class GeoBounds:
    """
    Geographic bounding box in WGS84 coordinates.

    Immutable to prevent accidental modification of region boundaries.
    """
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def contains(self, lat: float, lon: float) -> bool:
        """Check if a point is within these bounds."""
        return (
            self.lat_min <= lat <= self.lat_max and
            self.lon_min <= lon <= self.lon_max
        )

    def contains_bounds(self, other: "GeoBounds") -> bool:
        """Check if another bounding box is fully contained."""
        return (
            self.lat_min <= other.lat_min and
            self.lat_max >= other.lat_max and
            self.lon_min <= other.lon_min and
            self.lon_max >= other.lon_max
        )

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Return as (lat_min, lat_max, lon_min, lon_max) tuple."""
        return (self.lat_min, self.lat_max, self.lon_min, self.lon_max)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "GeoBounds":
        """Create from dictionary."""
        return cls(
            lat_min=data["lat_min"],
            lat_max=data["lat_max"],
            lon_min=data["lon_min"],
            lon_max=data["lon_max"],
        )


@dataclass(frozen=True)
class BoundaryPoint:
    """
    A single WW3 extraction point for SWAN boundary conditions.

    Immutable - these points are "cemented" and should not change
    without explicit versioning.
    """
    index: int
    lat: float
    lon: float
    depth_m: float
    segment: str  # "western_offshore" | "channel_islands"
    island: Optional[str] = None  # Only for channel_islands segment

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "index": self.index,
            "lat": self.lat,
            "lon": self.lon,
            "depth_m": self.depth_m,
            "segment": self.segment,
        }
        if self.island:
            d["island"] = self.island
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundaryPoint":
        """Create from dictionary."""
        return cls(
            index=data["index"],
            lat=data["lat"],
            lon=data["lon"],
            depth_m=data["depth_m"],
            segment=data["segment"],
            island=data.get("island"),
        )


@dataclass
class WW3ExtractionPoints:
    """
    Collection of WW3 extraction points for a region.

    These points define where WW3 data is interpolated to provide
    boundary conditions for SWAN. The points are IMMUTABLE once set.
    """
    version: str
    region: str
    description: str
    total_points: int
    points: List[BoundaryPoint]
    segments: Dict[str, Any]
    validation: Dict[str, Any]
    changelog: List[Dict[str, str]]
    immutable: bool = True

    @classmethod
    def load(cls, path: Path) -> "WW3ExtractionPoints":
        """Load extraction points from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        points = [BoundaryPoint.from_dict(p) for p in data["points"]]

        return cls(
            version=data["version"],
            region=data["region"],
            description=data["description"],
            total_points=data["total_points"],
            points=points,
            segments=data["segments"],
            validation=data["validation"],
            changelog=data.get("changelog", []),
            immutable=data.get("immutable", True),
        )

    def validate(self) -> bool:
        """Validate point count matches expected."""
        if len(self.points) != self.total_points:
            return False
        # Validate indices are sequential
        for i, pt in enumerate(self.points):
            if pt.index != i:
                return False
        return True

    def get_offshore_points(self) -> List[BoundaryPoint]:
        """Get only offshore (western) boundary points."""
        return [p for p in self.points if p.segment == "western_offshore"]

    def get_island_points(self, island: Optional[str] = None) -> List[BoundaryPoint]:
        """Get island boundary points, optionally filtered by island."""
        island_pts = [p for p in self.points if p.segment == "channel_islands"]
        if island:
            island_pts = [p for p in island_pts if p.island == island]
        return island_pts

    def to_lat_lon_list(self) -> List[Tuple[float, float]]:
        """Return list of (lat, lon) tuples for all points."""
        return [(p.lat, p.lon) for p in self.points]


@dataclass
class DomainMesh:
    """
    A versioned, immutable mesh for a SWAN domain.

    Once generated, meshes should not be modified. Instead, create
    a new version if changes are needed.
    """
    version: str
    description: str
    grd_file: str  # SWAN .grd file (relative path)
    nc_file: str   # NetCDF file for analysis (relative path)
    swn_template: Optional[str]  # SWAN input template (relative path)
    generated_at: str
    bathymetry_sources: List[str]
    quality_metrics: Dict[str, float]
    immutable: bool = True
    is_active: bool = False

    @classmethod
    def from_dict(cls, version: str, data: Dict[str, Any]) -> "DomainMesh":
        """Create from dictionary (as stored in domain.yaml)."""
        return cls(
            version=version,
            description=data.get("description", ""),
            grd_file=data["grd_file"],
            nc_file=data["nc_file"],
            swn_template=data.get("swn_template"),
            generated_at=data.get("generated_at", ""),
            bathymetry_sources=data.get("bathymetry_sources", []),
            quality_metrics=data.get("quality_metrics", {}),
            immutable=data.get("immutable", True),
            is_active=data.get("is_active", False),
        )


@dataclass
class SpectralConfig:
    """Configuration for wave spectral discretization."""
    n_frequencies: int = 36
    freq_min_hz: float = 0.04
    freq_max_hz: float = 1.0
    n_directions: int = 36
    spreading_degrees: float = 25.0
    spectral_shape: str = "JONSWAP"  # JONSWAP | PM


@dataclass
class SwanPhysics:
    """SWAN physics configuration for a domain."""
    generation_model: str = "WESTH"
    whitecapping: bool = True
    breaking_model: str = "BKD"
    breaking_alpha: float = 1.0
    breaking_gamma: float = 0.73
    friction_model: str = "JONSWAP"
    friction_coefficient: float = 0.067
    triads_enabled: bool = False
    quadruplets_enabled: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SwanPhysics":
        """Create from nested physics dict in domain.yaml."""
        gen = data.get("generation", {})
        brk = data.get("breaking", {})
        fric = data.get("friction", {})
        tri = data.get("triads", {})
        quad = data.get("quadruplets", {})

        return cls(
            generation_model=gen.get("model", "WESTH"),
            whitecapping=gen.get("whitecapping", True),
            breaking_model=brk.get("model", "BKD"),
            breaking_alpha=brk.get("alpha", 1.0),
            breaking_gamma=brk.get("gamma", 0.73),
            friction_model=fric.get("model", "JONSWAP"),
            friction_coefficient=fric.get("coefficient", 0.067),
            triads_enabled=tri.get("enabled", False),
            quadruplets_enabled=quad.get("enabled", True),
        )


@dataclass
class Domain:
    """
    A SWAN computational domain (outer, inner, or spot-level).
    """
    name: str
    region_name: str
    domain_type: DomainType
    version: str
    description: str
    bounds: GeoBounds
    grid_type: GridType
    resolution_m: float
    n_lat: int
    n_lon: int
    n_cells: int
    n_wet_cells: int
    meshes: Dict[str, DomainMesh]
    active_mesh_version: Optional[str]
    spectral_config: SpectralConfig
    swan_physics: SwanPhysics
    boundary_source: str  # "ww3" for outer, parent domain name for inner
    extraction_points_ref: Optional[str]

    @property
    def active_mesh(self) -> Optional[DomainMesh]:
        """Get the currently active mesh."""
        if self.active_mesh_version and self.active_mesh_version in self.meshes:
            return self.meshes[self.active_mesh_version]
        return None

    def get_swan_input_template(self, base_path: Path) -> Optional[str]:
        """Load SWAN input template content if available."""
        mesh = self.active_mesh
        if mesh and mesh.swn_template:
            template_path = base_path / mesh.swn_template
            if template_path.exists():
                return template_path.read_text()
        return None


@dataclass
class BathymetrySource:
    """Configuration for a bathymetry data source."""
    name: str
    description: str
    source_type: BathymetrySourceType
    path_or_endpoint: str
    resolution_m: float
    priority: int  # Lower = higher priority
    use_for: List[DomainType]
    coverage_notes: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BathymetrySource":
        """Create from dictionary."""
        use_for = [DomainType(t) for t in data.get("use_for", [])]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            source_type=BathymetrySourceType(data["type"]),
            path_or_endpoint=data.get("path") or data.get("endpoint", ""),
            resolution_m=data["resolution_m"],
            priority=data["priority"],
            use_for=use_for,
            coverage_notes=data.get("coverage_notes"),
        )


@dataclass
class ChannelIsland:
    """Definition of a Channel Island for boundary point generation."""
    id: str
    name: str
    center_lat: float
    center_lon: float
    radius_km: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChannelIsland":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            center_lat=data["center_lat"],
            center_lon=data["center_lon"],
            radius_km=data["radius_km"],
        )


@dataclass
class SurfSpot:
    """Definition of a surf spot within a region."""
    id: str
    name: str
    lat: float
    lon: float
    sub_region: Optional[str] = None
    inner_domain: Optional[str] = None  # Which inner domain covers this spot
    spot_mesh: Optional[str] = None     # Fine-resolution mesh version

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SurfSpot":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            lat=data["lat"],
            lon=data["lon"],
            sub_region=data.get("sub_region"),
            inner_domain=data.get("inner_domain"),
            spot_mesh=data.get("spot_mesh"),
        )


@dataclass
class WW3SourceConfig:
    """Configuration for WW3 data source."""
    description: str
    endpoint: str
    model: str
    native_resolution_deg: float
    variables: List[str]
    fetch_bounds: GeoBounds

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WW3SourceConfig":
        """Create from dictionary."""
        fetch_bounds_data = data.get("fetch_bounds", {})
        fetch_bounds = GeoBounds(
            lat_min=fetch_bounds_data.get("lat_min", 30.0),
            lat_max=fetch_bounds_data.get("lat_max", 44.0),
            lon_min=fetch_bounds_data.get("lon_min", -130.0),
            lon_max=fetch_bounds_data.get("lon_max", -115.0),
        )
        return cls(
            description=data.get("description", ""),
            endpoint=data["endpoint"],
            model=data["model"],
            native_resolution_deg=data.get("native_resolution_deg", 0.25),
            variables=data.get("variables", []),
            fetch_bounds=fetch_bounds,
        )


@dataclass
class Region:
    """
    Master region configuration.

    This is the top-level entity that contains all configuration
    for a geographic region including domains, boundaries, and spots.
    """
    name: str
    display_name: str
    version: str
    base_path: Path
    bounds: GeoBounds
    sub_regions: Dict[str, GeoBounds]
    ww3_source: WW3SourceConfig
    channel_islands: List[ChannelIsland]
    extraction_points: Optional[WW3ExtractionPoints]
    domains: Dict[DomainType, Domain]
    bathymetry_sources: List[BathymetrySource]
    surf_spots: List[SurfSpot]

    def get_domain(self, domain_type: str | DomainType) -> Optional[Domain]:
        """Get domain by type."""
        if isinstance(domain_type, str):
            domain_type = DomainType(domain_type)
        return self.domains.get(domain_type)

    def get_outer_domain(self) -> Optional[Domain]:
        """Convenience method to get outer domain."""
        return self.domains.get(DomainType.OUTER)

    def get_extraction_points(self) -> List[BoundaryPoint]:
        """Get all WW3 extraction points."""
        if self.extraction_points:
            return self.extraction_points.points
        return []

    def get_extraction_points_as_tuples(self) -> List[Tuple[float, float]]:
        """Get extraction points as (lat, lon) tuples for SWAN."""
        if self.extraction_points:
            return self.extraction_points.to_lat_lon_list()
        return []

    def get_surf_spot(self, spot_id: str) -> Optional[SurfSpot]:
        """Get a surf spot by ID."""
        for spot in self.surf_spots:
            if spot.id == spot_id:
                return spot
        return None

    def get_bathymetry_sources_for_domain(
        self, domain_type: DomainType
    ) -> List[BathymetrySource]:
        """Get bathymetry sources applicable to a domain type, sorted by priority."""
        sources = [s for s in self.bathymetry_sources if domain_type in s.use_for]
        return sorted(sources, key=lambda s: s.priority)
