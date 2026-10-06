from pathlib import Path

import json
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "registration"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Common-grid definition
# ============================================================

LON_MIN = 27.0
LON_MAX = 29.0

LAT_MIN = 44.0
LAT_MAX = 46.0

LUNAR_RADIUS_M = 1_737_400.0

TARGET_PPD = 128
TARGET_RESOLUTION_M = 236.9011751886678


# ============================================================
# Grid dimensions
# ============================================================

width = int(
    round(
        (LON_MAX - LON_MIN)
        * TARGET_PPD
    )
)

height = int(
    round(
        (LAT_MAX - LAT_MIN)
        * TARGET_PPD
    )
)


# ============================================================
# Pixel-center coordinates
# ============================================================

longitudes = (
    LON_MIN
    + (np.arange(width) + 0.5)
    / TARGET_PPD
)

latitudes = (
    LAT_MAX
    - (np.arange(height) + 0.5)
    / TARGET_PPD
)


# ============================================================
# Projected coordinates
# ============================================================

x_centers = (
    LUNAR_RADIUS_M
    * np.deg2rad(longitudes)
)

y_centers = (
    LUNAR_RADIUS_M
    * np.deg2rad(latitudes)
)


# ============================================================
# Grid metadata
# ============================================================

grid = {
    "name": "pilot_common_grid_v1",

    "projection": {
        "type": "Equirectangular",
        "central_meridian_deg": 180.0,
        "standard_parallel_deg": 0.0,
        "lunar_radius_m": LUNAR_RADIUS_M,
    },

    "extent": {
        "longitude_min_deg": LON_MIN,
        "longitude_max_deg": LON_MAX,
        "latitude_min_deg": LAT_MIN,
        "latitude_max_deg": LAT_MAX,
    },

    "resolution": {
        "pixels_per_degree": TARGET_PPD,
        "nominal_m_per_pixel": TARGET_RESOLUTION_M,
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


# ============================================================
# Save coordinate arrays
# ============================================================

np.save(
    OUTPUT_DIR / "common_grid_longitudes.npy",
    longitudes.astype(np.float64),
)

np.save(
    OUTPUT_DIR / "common_grid_latitudes.npy",
    latitudes.astype(np.float64),
)

np.save(
    OUTPUT_DIR / "common_grid_x_m.npy",
    x_centers.astype(np.float64),
)

np.save(
    OUTPUT_DIR / "common_grid_y_m.npy",
    y_centers.astype(np.float64),
)


# ============================================================
# Save metadata
# ============================================================

metadata_path = (
    OUTPUT_DIR
    / "common_grid_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        grid,
        f,
        indent=2,
    )


# ============================================================
# Report
# ============================================================

print("=" * 60)
print("COMMON GRID DEFINITION")
print("=" * 60)

print(f"Longitude : {LON_MIN}° → {LON_MAX}°")
print(f"Latitude  : {LAT_MIN}° → {LAT_MAX}°")

print(f"PPD       : {TARGET_PPD}")
print(
    f"Resolution: "
    f"{TARGET_RESOLUTION_M:.12f} m/pixel"
)

print(f"Shape     : {height} × {width}")

print("\nFirst pixel center:")
print(
    f"  Lon: {longitudes[0]:.6f}°"
)
print(
    f"  Lat: {latitudes[0]:.6f}°"
)

print("\nLast pixel center:")
print(
    f"  Lon: {longitudes[-1]:.6f}°"
)
print(
    f"  Lat: {latitudes[-1]:.6f}°"
)

print("\nSaved:")
print(metadata_path)