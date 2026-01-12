# Regions Directory - Claude Code Guidelines

**CRITICAL: This directory contains protected configuration structures. Read this entire document before making any changes.**

## Purpose

The `regions/` directory contains geographic region configurations for the surf forecasting system. Each region encapsulates everything needed to run SWAN wave models for that geographic area.

## Directory Structure

```
regions/
├── CLAUDE.md              # This file - READ FIRST
├── _templates/            # Templates for creating new regions
│   ├── region.yaml.template
│   └── domain.yaml.template
│
├── california/            # Example region
│   ├── region.yaml        # Master region configuration
│   ├── boundaries/
│   │   └── ww3_extraction_points.json  # IMMUTABLE - DO NOT MODIFY
│   ├── domains/
│   │   ├── outer/
│   │   │   ├── domain.yaml
│   │   │   ├── mesh_v1.grd     # IMMUTABLE versions
│   │   │   ├── mesh_v1.nc
│   │   │   └── mesh_v1.swn
│   │   ├── inner/              # Future
│   │   └── spots/              # Future
│   ├── bathymetry/             # Region-specific bathymetry products
│   └── runs/                   # SWAN run outputs (timestamped)
│
└── [other_regions]/       # Same structure
```

## PROTECTED FILES - DO NOT MODIFY

### 1. WW3 Extraction Points (`boundaries/ww3_extraction_points.json`)

```
⚠️  IMMUTABLE - NEVER MODIFY WITHOUT EXPLICIT USER APPROVAL
```

These files contain "cemented" boundary condition extraction points. They define:
- Where WW3 spectral data is interpolated for SWAN input
- The exact lat/lon/depth of each extraction point
- Segment classification (offshore vs. island)

**Why protected:**
- Historical runs depend on these exact points
- Model calibration is tied to these locations
- Changing points invalidates all previous validation work

**If you need to modify:**
1. ASK THE USER FIRST
2. Create a new version file (e.g., `ww3_extraction_points_v2.json`)
3. Update `region.yaml` to reference the new version
4. Add changelog entry to the new file
5. Keep the old file for reference

### 2. Mesh Files (`mesh_v*.grd`, `mesh_v*.nc`)

```
⚠️  IMMUTABLE VERSIONS - CREATE NEW VERSIONS INSTEAD
```

Mesh files are versioned (v1, v2, etc.) and should never be overwritten.

**To update a mesh:**
```yaml
# In domain.yaml, add new version:
meshes:
  v1:
    description: "Original 14km mesh"
    # ... existing config
    immutable: true
  v2:
    description: "Updated with improved coastline"
    grd_file: "mesh_v2.grd"
    nc_file: "mesh_v2.nc"
    generated_at: "2026-01-15T00:00:00Z"
    immutable: true
    is_active: true

active_mesh: "v2"  # Switch to new version
```

### 3. Region Structure

The hierarchical structure is intentional:
```
region → boundaries (immutable)
       → domains → outer (active)
                 → inner (future)
                 → spots (future)
```

Don't flatten or restructure this hierarchy.

## SAFE TO MODIFY

### region.yaml

Safe to add/modify:
- `surf_spots` list (add new spots)
- `sub_regions` definitions
- `bathymetry_sources` (add new sources)
- `display_name` and cosmetic fields

**Do NOT modify:**
- `boundaries.extraction_point_count` (must match JSON)
- `bounds` (defines region, affects all domains)

### domain.yaml

Safe to modify:
- `swan_physics` parameters
- `spectral_config` (frequency/direction bins)
- `outputs` specification
- `description` and `notes`

**Do NOT modify without creating new version:**
- `grid` specification (resolution, bounds, cell counts)
- `meshes.*.grd_file` paths for existing versions

### Adding New Regions

Creating a new region is always safe:
```bash
cp -r regions/_templates regions/new_region_name
# Then customize the configuration
```

## File Format Reference

### region.yaml

```yaml
name: region_id                    # Lowercase, underscores
display_name: "Human Readable"
version: "1.0.0"                   # Semantic versioning

bounds:                            # Geographic bounds (WGS84)
  lat_min: 32.0
  lat_max: 42.0
  lon_min: -126.0
  lon_max: -117.0

boundaries:
  ww3_extraction: "boundaries/ww3_extraction_points.json"
  extraction_point_count: 71       # MUST match actual count

domains:
  outer:
    path: "domains/outer"
    active_mesh: "v1"
  inner:
    path: "domains/inner"
    active_mesh: null              # Not implemented yet
  spots:
    path: "domains/spots"
    active_meshes: {}              # Map of spot_id -> mesh version

surf_spots:
  - id: spot_id
    name: "Spot Name"
    lat: 33.655
    lon: -117.999
```

### ww3_extraction_points.json

```json
{
  "version": "1.0.0",
  "immutable": true,               // ALWAYS TRUE
  "total_points": 71,              // Must match array length

  "segments": {
    "western_offshore": { "point_count": 39 },
    "channel_islands": { "point_count": 32 }
  },

  "points": [
    {
      "index": 0,
      "segment": "western_offshore",
      "lat": 32.252,
      "lon": -117.171,
      "depth_m": 77.87
    }
    // ... more points
  ],

  "changelog": [
    {
      "version": "1.0.0",
      "date": "2026-01-12",
      "description": "Initial cemented configuration"
    }
  ]
}
```

### domain.yaml

```yaml
name: "region_outer"
domain_type: "outer"               # outer | inner | spot

grid:
  type: "regular"                  # regular | curvilinear | unstructured
  resolution_m: 14000
  n_lat: 80
  n_lon: 72

meshes:
  v1:
    grd_file: "mesh_v1.grd"
    nc_file: "mesh_v1.nc"
    immutable: true
    is_active: true

active_mesh: "v1"

swan_physics:                      # Safe to modify
  breaking:
    gamma: 0.73
  friction:
    coefficient: 0.067
```

## Validation Rules

Before any modification, verify:

1. **Point count matches**: `extraction_point_count` in region.yaml == length of `points` array in JSON
2. **Active mesh exists**: `active_mesh` value is a key in `meshes` dict
3. **File references valid**: All `*_file` paths exist
4. **Bounds contain domains**: Region bounds contain all domain bounds

## Python API

Load regions programmatically:

```python
from wave_forecast_common.regions import RegionRegistry

registry = RegionRegistry()
california = registry.load_region("california")

# Access protected data (read-only)
points = california.extraction_points.points
assert len(points) == 71  # Cemented count

# Access domain
outer = california.get_outer_domain()
mesh = outer.active_mesh
```

## Common Errors

### "Extraction point count mismatch"

```
ValueError: Extraction point count mismatch: expected 71, got 70
```

The `extraction_point_count` in region.yaml doesn't match the JSON. Either:
- Someone accidentally deleted a point (restore from git)
- The count was incorrectly set (fix region.yaml)

### "Active mesh not found"

```
KeyError: 'v2' not found in meshes
```

The `active_mesh` references a version that doesn't exist. Check `domain.yaml`.

## Migration Notes

The legacy structure in `data/grids/` is being migrated to this `regions/` structure. During transition:

- Old: `data/grids/swan_domains/california_swan_14000m/`
- New: `regions/california/domains/outer/`

The legacy structure will be deprecated once migration is complete.
