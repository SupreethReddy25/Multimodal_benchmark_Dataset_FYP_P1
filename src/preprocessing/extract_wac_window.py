from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WAC_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "optical"
    / "WAC_GLOBAL_E300N0450_100M.tif"
)

OUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "optical"
    / "wac_pilot.npy"
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Pilot AOI
# --------------------------------------------------
LAT_MIN = 44.0
LAT_MAX = 46.0

LON_MIN = 27.0
LON_MAX = 29.0

# WAC lunar equirectangular radius
MOON_RADIUS = 1_737_400.0


def lon_to_x(lon):
    return MOON_RADIUS * np.deg2rad(lon)


def lat_to_y(lat):
    return MOON_RADIUS * np.deg2rad(lat)


# --------------------------------------------------
# Geographic AOI -> WAC projected coordinates
# --------------------------------------------------
left = lon_to_x(LON_MIN)
right = lon_to_x(LON_MAX)

bottom = lat_to_y(LAT_MIN)
top = lat_to_y(LAT_MAX)


print("WAC source:")
print(WAC_PATH)

print("\nProjected AOI:")
print(f"Left   : {left:.3f} m")
print(f"Right  : {right:.3f} m")
print(f"Bottom : {bottom:.3f} m")
print(f"Top    : {top:.3f} m")


# --------------------------------------------------
# Read native-resolution window
# --------------------------------------------------
with rasterio.open(WAC_PATH) as src:

    print("\nSource raster:")
    print(f"CRS      : {src.crs}")
    print(f"Resolution: {src.res}")
    print(f"Size     : {src.width} x {src.height}")

    window = from_bounds(
        left,
        bottom,
        right,
        top,
        transform=src.transform
    )

    # Round outward so the complete AOI is retained
    window = window.round_offsets().round_lengths()

    data = src.read(1, window=window)

    transform = src.window_transform(window)

    print("\nExtracted window:")
    print(f"Rows : {window.row_off} -> "
          f"{window.row_off + window.height}")
    print(f"Cols : {window.col_off} -> "
          f"{window.col_off + window.width}")
    print(f"Shape: {data.shape}")

    print("\nExtracted statistics:")
    print(f"Dtype : {data.dtype}")
    print(f"Min   : {data.min()}")
    print(f"Max   : {data.max()}")
    print(f"Mean  : {data.mean():.3f}")
    print(f"Std   : {data.std():.3f}")

np.save(OUT_PATH, data)

print("\nSaved:")
print(OUT_PATH)