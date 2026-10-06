from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRID_DIR = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "registration"
)

# ------------------------------------------------------------
# Common grid
# ------------------------------------------------------------

common_lon = np.load(
    GRID_DIR / "common_grid_longitudes.npy"
)

common_lat = np.load(
    GRID_DIR / "common_grid_latitudes.npy"
)

# ------------------------------------------------------------
# Diviner geometry from PDS label
# ------------------------------------------------------------

DIVINER_PPD = 128

DIVINER_LAT_TOP = 70.0
DIVINER_LON_ORIGIN = 0.0

ROW_START = 3072
ROW_END = 3328

COL_START = 3456
COL_END = 3712

# Pixel-center coordinates of the extracted Diviner window.
#
# The first pixel center is half a pixel inside the geographic
# boundary.

diviner_rows = np.arange(
    ROW_START,
    ROW_END
)

diviner_cols = np.arange(
    COL_START,
    COL_END
)

diviner_lat = (
    DIVINER_LAT_TOP
    - (diviner_rows + 0.5) / DIVINER_PPD
)

diviner_lon = (
    DIVINER_LON_ORIGIN
    + (diviner_cols + 0.5) / DIVINER_PPD
)


# ------------------------------------------------------------
# Compare
# ------------------------------------------------------------

lon_difference = (
    common_lon - diviner_lon
)

lat_difference = (
    common_lat - diviner_lat
)

print("=" * 60)
print("COMMON GRID vs DIVINER NATIVE GRID")
print("=" * 60)

print(
    f"Common grid longitude range: "
    f"{common_lon[0]:.9f} → {common_lon[-1]:.9f}"
)

print(
    f"Diviner longitude range:     "
    f"{diviner_lon[0]:.9f} → {diviner_lon[-1]:.9f}"
)

print(
    f"\nCommon grid latitude range:  "
    f"{common_lat[0]:.9f} → {common_lat[-1]:.9f}"
)

print(
    f"Diviner latitude range:      "
    f"{diviner_lat[0]:.9f} → {diviner_lat[-1]:.9f}"
)

print("\nMaximum center-coordinate differences:")

print(
    f"Longitude: "
    f"{np.max(np.abs(lon_difference)):.12f}°"
)

print(
    f"Latitude : "
    f"{np.max(np.abs(lat_difference)):.12f}°"
)

print("\nArray equality:")

print(
    "Longitude:",
    np.array_equal(
        common_lon,
        diviner_lon
    )
)

print(
    "Latitude :",
    np.array_equal(
        common_lat,
        diviner_lat
    )
)

print("\nResult:")

if (
    np.allclose(common_lon, diviner_lon)
    and np.allclose(common_lat, diviner_lat)
):
    print(
        "PASS — the common grid has the same "
        "pixel-center geometry as the Diviner pilot grid."
    )
else:
    print(
        "FAIL — common-grid and Diviner coordinates differ."
    )