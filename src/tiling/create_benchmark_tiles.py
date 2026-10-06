from pathlib import Path
import json
import shutil

import numpy as np


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INTERMEDIATE = ROOT / "data" / "intermediate"
BENCHMARK = ROOT / "data" / "benchmark"

TILES_DIR = BENCHMARK / "tiles"
TILES_DIR.mkdir(parents=True, exist_ok=True)

GRID_METADATA_PATH = (
    INTERMEDIATE
    / "registration"
    / "common_grid_metadata.json"
)


# ============================================================
# Load common-grid metadata
# ============================================================

with open(GRID_METADATA_PATH, "r", encoding="utf-8") as f:
    grid = json.load(f)

height = grid["shape"]["height"]
width = grid["shape"]["width"]

pixels_per_degree = grid["resolution"]["pixels_per_degree"]
resolution_m = grid["resolution"]["nominal_m_per_pixel"]

lat_min = grid["extent"]["latitude_min_deg"]
lat_max = grid["extent"]["latitude_max_deg"]
lon_min = grid["extent"]["longitude_min_deg"]
lon_max = grid["extent"]["longitude_max_deg"]


# ============================================================
# Benchmark tile configuration
# ============================================================

TILE_SIZE = 64
OVERLAP = 0

stride = TILE_SIZE - OVERLAP

if height % stride != 0 or width % stride != 0:
    raise ValueError(
        f"Common grid {height}x{width} is not compatible "
        f"with tile size {TILE_SIZE} and overlap {OVERLAP}."
    )

rows = (height - OVERLAP) // stride
cols = (width - OVERLAP) // stride

expected_tiles = 16


# ============================================================
# Input products
# ============================================================

inputs = {
    "optical": (
        INTERMEDIATE
        / "optical"
        / "wac_common_grid.npy"
    ),

    "optical_coverage": (
        INTERMEDIATE
        / "optical"
        / "wac_common_coverage.npy"
    ),

    "dem": (
        INTERMEDIATE
        / "dem"
        / "lola_dem_common_grid.npy"
    ),

    "dem_valid_mask": (
        INTERMEDIATE
        / "dem"
        / "lola_dem_common_valid_mask.npy"
    ),

    "dem_coverage": (
        INTERMEDIATE
        / "dem"
        / "lola_dem_common_coverage.npy"
    ),

    "slope": (
        INTERMEDIATE
        / "dem"
        / "lola_common_slope_deg.npy"
    ),

    "hillshade": (
        INTERMEDIATE
        / "dem"
        / "lola_common_hillshade.npy"
    ),

    "thermal": (
        INTERMEDIATE
        / "thermal"
        / "diviner_tbol_sam_common_grid.npy"
    ),

    "thermal_valid_mask": (
        INTERMEDIATE
        / "thermal"
        / "diviner_tbol_sam_common_valid_mask.npy"
    ),

    "thermal_coverage": (
        INTERMEDIATE
        / "thermal"
        / "diviner_tbol_sam_common_coverage.npy"
    )
}


# ============================================================
# Verify inputs
# ============================================================

print("=" * 70)
print("BENCHMARK TILE CONSTRUCTION")
print("=" * 70)

print("\nChecking input products...")

arrays = {}

for name, path in inputs.items():

    if not path.exists():
        raise FileNotFoundError(
            f"Required input does not exist:\n{path}"
        )

    arr = np.load(path)

    if arr.shape != (height, width):
        raise ValueError(
            f"{name} has shape {arr.shape}; "
            f"expected {(height, width)}."
        )

    arrays[name] = arr

    print(
        f"  {name:22s}: "
        f"shape={arr.shape}, dtype={arr.dtype}"
    )


# ============================================================
# Remove previous generated tiles
# ============================================================

existing_tiles = sorted(
    p for p in TILES_DIR.iterdir()
    if p.is_dir() and p.name.startswith("tile_")
)

if existing_tiles:

    print(
        f"\nRemoving {len(existing_tiles)} previously generated "
        "benchmark tile directories..."
    )

    for tile_dir in existing_tiles:
        shutil.rmtree(tile_dir)


# ============================================================
# Coordinate arrays
# ============================================================

longitudes = np.load(
    INTERMEDIATE
    / "registration"
    / "common_grid_longitudes.npy"
)

latitudes = np.load(
    INTERMEDIATE
    / "registration"
    / "common_grid_latitudes.npy"
)

if longitudes.shape != (width,):
    raise ValueError(
        f"Unexpected longitude array shape: {longitudes.shape}"
    )

if latitudes.shape != (height,):
    raise ValueError(
        f"Unexpected latitude array shape: {latitudes.shape}"
    )


# ============================================================
# Create tiles
# ============================================================

tile_records = []

tile_counter = 0

for row in range(rows):

    row_start = row * stride
    row_end = row_start + TILE_SIZE

    for col in range(cols):

        col_start = col * stride
        col_end = col_start + TILE_SIZE

        tile_id = f"tile_{row:02d}_{col:02d}"

        tile_dir = TILES_DIR / tile_id
        tile_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------------------------------
        # Extract synchronized spatial region
        # ----------------------------------------------------

        tile_arrays = {}

        for name, arr in arrays.items():

            tile = arr[
                row_start:row_end,
                col_start:col_end
            ]

            if tile.shape != (TILE_SIZE, TILE_SIZE):
                raise ValueError(
                    f"{name} in {tile_id} has shape {tile.shape}"
                )

            tile_arrays[name] = tile

            np.save(
                tile_dir / f"{name}.npy",
                tile
            )

        # ----------------------------------------------------
        # Geographic extent
        # ----------------------------------------------------

        tile_lons = longitudes[col_start:col_end]
        tile_lats = latitudes[row_start:row_end]

        tile_metadata = {
            "tile_id": tile_id,

            "tile_index": {
                "row": row,
                "column": col
            },

            "pixel_window": {
                "row_start": row_start,
                "row_end_exclusive": row_end,
                "column_start": col_start,
                "column_end_exclusive": col_end
            },

            "shape": {
                "height": TILE_SIZE,
                "width": TILE_SIZE
            },

            "spatial_resolution": {
                "pixels_per_degree": pixels_per_degree,
                "nominal_m_per_pixel": resolution_m
            },

            "geographic_extent": {
                "longitude_min_deg": float(tile_lons.min()),
                "longitude_max_deg": float(tile_lons.max()),
                "latitude_min_deg": float(tile_lats.min()),
                "latitude_max_deg": float(tile_lats.max())
            },

            "pixel_center_coordinates": {
                "first_pixel": {
                    "longitude_deg": float(tile_lons[0]),
                    "latitude_deg": float(tile_lats[0])
                },

                "last_pixel": {
                    "longitude_deg": float(tile_lons[-1]),
                    "latitude_deg": float(tile_lats[-1])
                }
            },

            "common_grid": {
                "name": grid["name"],
                "projection": grid["projection"]
            },

            "modalities": {
                "optical": {
                    "file": "optical.npy",
                    "source": "LROC WAC"
                },

                "dem": {
                    "file": "dem.npy",
                    "source": "LRO LOLA"
                },

                "thermal": {
                    "file": "thermal.npy",
                    "source": "LRO Diviner"
                }
            },

            "derived_products": {
                "slope": "slope.npy",
                "hillshade": "hillshade.npy"
            },

            "validity_products": {
                "optical_coverage": "optical_coverage.npy",
                "dem_valid_mask": "dem_valid_mask.npy",
                "dem_coverage": "dem_coverage.npy",
                "thermal_valid_mask": "thermal_valid_mask.npy",
                "thermal_coverage": "thermal_coverage.npy"
            },

            "tile_generation": {
                "tile_size_pixels": TILE_SIZE,
                "overlap_pixels": OVERLAP,
                "pixel_correspondence_preserved": True
            }
        }

        # ----------------------------------------------------
        # Save tile metadata
        # ----------------------------------------------------

        with open(
            tile_dir / "metadata.json",
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                tile_metadata,
                f,
                indent=2
            )

        tile_records.append(tile_metadata)

        tile_counter += 1

        print(
            f"Created {tile_id} "
            f"({row_start}:{row_end}, {col_start}:{col_end})"
        )


# ============================================================
# Dataset-level tile index
# ============================================================

index = {
    "dataset": "Multimodal Lunar Benchmark",

    "version": "1.0.0",

    "pilot_region": {
        "name": "Lacus Mortis - Burg Pilot",
        "latitude_min_deg": lat_min,
        "latitude_max_deg": lat_max,
        "longitude_min_deg": lon_min,
        "longitude_max_deg": lon_max
    },

    "common_grid": {
        "height": height,
        "width": width,
        "pixels_per_degree": pixels_per_degree,
        "nominal_m_per_pixel": resolution_m
    },

    "tiling": {
        "tile_size_pixels": TILE_SIZE,
        "overlap_pixels": OVERLAP,
        "rows": rows,
        "columns": cols,
        "total_tiles": tile_counter
    },

    "tiles": tile_records
}

index_path = BENCHMARK / "metadata" / "tile_index.json"

with open(index_path, "w", encoding="utf-8") as f:
    json.dump(index, f, indent=2)


# ============================================================
# Final checks
# ============================================================

if tile_counter != expected_tiles:
    raise RuntimeError(
        f"Expected {expected_tiles} tiles, "
        f"but generated {tile_counter}."
    )


print("\n" + "=" * 70)
print("TILE CONSTRUCTION COMPLETE")
print("=" * 70)

print(f"Common grid      : {height} x {width}")
print(f"Tile size        : {TILE_SIZE} x {TILE_SIZE}")
print(f"Rows             : {rows}")
print(f"Columns          : {cols}")
print(f"Total tiles      : {tile_counter}")
print(f"Overlap          : {OVERLAP} pixels")
print(f"Tile index       : {index_path}")
print(f"Tiles directory  : {TILES_DIR}")
print("=" * 70)