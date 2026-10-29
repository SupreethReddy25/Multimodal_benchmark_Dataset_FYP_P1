"""Step 7 — Partition the 256x256 common grid into 16 synchronized 64x64 tiles.

All 10 layers (optical, optical_coverage, dem, dem_valid_mask, dem_coverage,
slope, hillshade, thermal, thermal_valid_mask, thermal_coverage) are sliced
with identical pixel windows, guaranteeing pixel-wise correspondence.

Produces:
    data/benchmark/{region_id}/tiles/tile_{row:02d}_{col:02d}/
        optical.npy, dem.npy, thermal.npy, slope.npy, hillshade.npy,
        optical_coverage.npy, dem_valid_mask.npy, dem_coverage.npy,
        thermal_valid_mask.npy, thermal_coverage.npy, metadata.json
    data/benchmark/{region_id}/metadata/tile_index.json

Usage
-----
    python -m src.pipeline.create_tiles --region posidonius
"""

import argparse
import json
import shutil

import numpy as np

from .config import load_region

TILE_SIZE = 64
OVERLAP = 0
EXPECTED_TILES = 16


def run(region_id: str) -> None:
    region = load_region(region_id)

    inter = region["intermediate_dir"]
    bench = region["benchmark_dir"]
    tiles_dir = bench / "tiles"
    meta_dir = bench / "metadata"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Already done?
    existing = sorted(
        p for p in tiles_dir.iterdir()
        if p.is_dir() and p.name.startswith("tile_")
    )
    if len(existing) == EXPECTED_TILES:
        print(f"  [tiles] {EXPECTED_TILES} tiles already exist — skipping.")
        return
    elif existing:
        print(f"  [tiles] Removing {len(existing)} incomplete tiles...")
        for td in existing:
            shutil.rmtree(td)

    # Load common-grid metadata
    grid_meta_path = inter / "registration" / "common_grid_metadata.json"
    with open(grid_meta_path, "r", encoding="utf-8") as fh:
        grid = json.load(fh)

    height = grid["shape"]["height"]          # 256
    width = grid["shape"]["width"]            # 256
    ppd = grid["resolution"]["pixels_per_degree"]
    resolution_m = grid["resolution"]["nominal_m_per_pixel"]
    lat_min = grid["extent"]["latitude_min_deg"]
    lat_max = grid["extent"]["latitude_max_deg"]
    lon_min = grid["extent"]["longitude_min_deg"]
    lon_max = grid["extent"]["longitude_max_deg"]

    stride = TILE_SIZE - OVERLAP
    if height % stride != 0 or width % stride != 0:
        raise ValueError(
            f"Grid {height}x{width} not divisible by tile stride {stride}."
        )
    rows = (height - OVERLAP) // stride      # 4
    cols = (width - OVERLAP) // stride       # 4

    # Load all input arrays
    def _load(sub, name):
        path = inter / sub / name
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")
        arr = np.load(path)
        if arr.shape != (height, width):
            raise ValueError(
                f"{name} shape {arr.shape} != expected {(height, width)}"
            )
        return arr

    arrays = {
        "optical":            _load("optical", "wac_common_grid.npy"),
        "optical_coverage":   _load("optical", "wac_common_coverage.npy"),
        "dem":                _load("dem", "lola_dem_common_grid.npy"),
        "dem_valid_mask":     _load("dem", "lola_dem_common_valid_mask.npy"),
        "dem_coverage":       _load("dem", "lola_dem_common_coverage.npy"),
        "slope":              _load("dem", "lola_common_slope_deg.npy"),
        "hillshade":          _load("dem", "lola_common_hillshade.npy"),
        "thermal":            _load("thermal", "diviner_tbol_sam_common_grid.npy"),
        "thermal_valid_mask": _load("thermal", "diviner_tbol_sam_common_valid_mask.npy"),
        "thermal_coverage":   _load("thermal", "diviner_tbol_sam_common_coverage.npy"),
    }

    longitudes = np.load(inter / "registration" / "common_grid_longitudes.npy")
    latitudes = np.load(inter / "registration" / "common_grid_latitudes.npy")

    tile_records = []
    tile_counter = 0

    for row in range(rows):
        row_start = row * stride
        row_end = row_start + TILE_SIZE

        for col in range(cols):
            col_start = col * stride
            col_end = col_start + TILE_SIZE
            tile_id = f"tile_{row:02d}_{col:02d}"
            tile_dir = tiles_dir / tile_id
            tile_dir.mkdir(parents=True, exist_ok=True)

            # Extract and save each layer with identical spatial window
            for name, arr in arrays.items():
                tile = arr[row_start:row_end, col_start:col_end]
                if tile.shape != (TILE_SIZE, TILE_SIZE):
                    raise ValueError(
                        f"{name} in {tile_id}: shape {tile.shape}"
                    )
                np.save(tile_dir / f"{name}.npy", tile)

            # Geographic extents of this tile
            tile_lons = longitudes[col_start:col_end]
            tile_lats = latitudes[row_start:row_end]

            tile_meta = {
                "tile_id": tile_id,
                "region_id": region_id,
                "tile_index": {"row": row, "column": col},
                "pixel_window": {
                    "row_start": row_start,
                    "row_end_exclusive": row_end,
                    "column_start": col_start,
                    "column_end_exclusive": col_end,
                },
                "shape": {"height": TILE_SIZE, "width": TILE_SIZE},
                "spatial_resolution": {
                    "pixels_per_degree": ppd,
                    "nominal_m_per_pixel": resolution_m,
                },
                "geographic_extent": {
                    "longitude_min_deg": float(tile_lons.min()),
                    "longitude_max_deg": float(tile_lons.max()),
                    "latitude_min_deg": float(tile_lats.min()),
                    "latitude_max_deg": float(tile_lats.max()),
                },
                "pixel_center_coordinates": {
                    "first_pixel": {
                        "longitude_deg": float(tile_lons[0]),
                        "latitude_deg": float(tile_lats[0]),
                    },
                    "last_pixel": {
                        "longitude_deg": float(tile_lons[-1]),
                        "latitude_deg": float(tile_lats[-1]),
                    },
                },
                "common_grid": {
                    "name": grid["name"],
                    "projection": grid["projection"],
                },
                "modalities": {
                    "optical": {"file": "optical.npy", "source": "LROC WAC"},
                    "dem": {"file": "dem.npy", "source": "LRO LOLA"},
                    "thermal": {"file": "thermal.npy", "source": "LRO Diviner"},
                },
                "derived_products": {
                    "slope": "slope.npy",
                    "hillshade": "hillshade.npy",
                },
                "validity_products": {
                    "optical_coverage": "optical_coverage.npy",
                    "dem_valid_mask": "dem_valid_mask.npy",
                    "dem_coverage": "dem_coverage.npy",
                    "thermal_valid_mask": "thermal_valid_mask.npy",
                    "thermal_coverage": "thermal_coverage.npy",
                },
                "tile_generation": {
                    "tile_size_pixels": TILE_SIZE,
                    "overlap_pixels": OVERLAP,
                    "pixel_correspondence_preserved": True,
                },
            }

            with open(tile_dir / "metadata.json", "w", encoding="utf-8") as fh:
                json.dump(tile_meta, fh, indent=2)

            tile_records.append(tile_meta)
            tile_counter += 1

    if tile_counter != EXPECTED_TILES:
        raise RuntimeError(
            f"Expected {EXPECTED_TILES} tiles, generated {tile_counter}."
        )

    # Dataset-level tile index
    index = {
        "dataset": "Multimodal Lunar Benchmark",
        "version": "1.0.0",
        "region_id": region_id,
        "region_name": region["name"],
        "pilot_region": {
            "latitude_min_deg": lat_min,
            "latitude_max_deg": lat_max,
            "longitude_min_deg": lon_min,
            "longitude_max_deg": lon_max,
        },
        "common_grid": {
            "height": height,
            "width": width,
            "pixels_per_degree": ppd,
            "nominal_m_per_pixel": resolution_m,
        },
        "tiling": {
            "tile_size_pixels": TILE_SIZE,
            "overlap_pixels": OVERLAP,
            "rows": rows,
            "columns": cols,
            "total_tiles": tile_counter,
        },
        "tiles": tile_records,
    }

    index_path = meta_dir / "tile_index.json"
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)

    print(f"  [tiles] {region_id}: {tile_counter} tiles "
          f"({rows}x{cols} @ {TILE_SIZE}x{TILE_SIZE} px)  "
          f"written to {tiles_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    args = ap.parse_args()
    run(args.region)
