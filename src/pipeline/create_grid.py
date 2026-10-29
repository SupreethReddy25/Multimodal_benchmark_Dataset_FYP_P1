"""Step 4 — Define the common cartographic grid for a region.

Generates the pixel-center coordinate vectors (latitudes, longitudes)
and projected (x, y) arrays for the 256x256 common grid, then saves
them as .npy files and a JSON metadata document.

Usage
-----
    python -m src.pipeline.create_grid --region posidonius
"""

import argparse
import json
import math
import numpy as np

from .config import load_region

LUNAR_RADIUS_M = 1_737_400.0


def run(region_id: str) -> None:
    region = load_region(region_id)

    out_dir = region["intermediate_dir"] / "registration"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "common_grid_metadata.json"
    if meta_path.exists():
        print(f"  [grid] common_grid_metadata.json already exists — skipping.")
        return

    lat_min = region["lat_min"]
    lat_max = region["lat_max"]
    lon_min = region["lon_min"]
    lon_max = region["lon_max"]
    ppd = region["pixels_per_degree"]        # 128
    height = region["grid_height"]           # 256
    width = region["grid_width"]             # 256
    resolution_m = region["resolution_m"]

    # Pixel-center coordinate vectors
    longitudes = (
        lon_min + (np.arange(width) + 0.5) / ppd
    ).astype(np.float64)

    latitudes = (
        lat_max - (np.arange(height) + 0.5) / ppd
    ).astype(np.float64)

    # Projected (x, y) in metres for equirectangular moon
    x_centers = (LUNAR_RADIUS_M * np.deg2rad(longitudes)).astype(np.float64)
    y_centers = (LUNAR_RADIUS_M * np.deg2rad(latitudes)).astype(np.float64)

    # Save arrays
    np.save(out_dir / "common_grid_longitudes.npy", longitudes)
    np.save(out_dir / "common_grid_latitudes.npy", latitudes)
    np.save(out_dir / "common_grid_x_m.npy", x_centers)
    np.save(out_dir / "common_grid_y_m.npy", y_centers)

    # Save metadata JSON
    grid_meta = {
        "name": f"{region_id}_common_grid_v1",
        "region_id": region_id,
        "projection": {
            "type": "Equirectangular",
            "central_meridian_deg": 180.0,
            "standard_parallel_deg": 0.0,
            "lunar_radius_m": LUNAR_RADIUS_M,
        },
        "extent": {
            "latitude_min_deg": lat_min,
            "latitude_max_deg": lat_max,
            "longitude_min_deg": lon_min,
            "longitude_max_deg": lon_max,
        },
        "resolution": {
            "pixels_per_degree": ppd,
            "nominal_m_per_pixel": resolution_m,
        },
        "shape": {
            "height": height,
            "width": width,
        },
        "pixel_convention": {
            "origin": "upper_left",
            "coordinates": "pixel_centers",
            "latitude_direction": "north_to_south",
            "longitude_direction": "west_to_east",
        },
        "center_coordinates": {
            "first_pixel": {
                "longitude_deg": float(longitudes[0]),
                "latitude_deg": float(latitudes[0]),
                "x_m": float(x_centers[0]),
                "y_m": float(y_centers[0]),
            },
            "last_pixel": {
                "longitude_deg": float(longitudes[-1]),
                "latitude_deg": float(latitudes[-1]),
                "x_m": float(x_centers[-1]),
                "y_m": float(y_centers[-1]),
            },
        },
    }

    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(grid_meta, fh, indent=2)

    print(f"  [grid] {region_id}: {height}x{width} @ {ppd} PPD "
          f"({resolution_m:.2f} m/px)  saved to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    args = ap.parse_args()
    run(args.region)
