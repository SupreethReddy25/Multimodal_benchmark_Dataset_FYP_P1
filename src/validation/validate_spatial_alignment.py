from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WAC_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "optical"
    / "WAC_GLOBAL_E300N0450_100M.tif"
)

# --------------------------------------------------
# Pilot AOI
# --------------------------------------------------
LAT_MIN = 44.0
LAT_MAX = 46.0
LON_MIN = 27.0
LON_MAX = 29.0

MOON_RADIUS = 1_737_400.0

# LOLA source geometry
LOLA_PPD = 256
LOLA_LINE_OFFSET = 23039.5
LOLA_SAMPLE_OFFSET = 46079.5
LOLA_CENTER_LON = 180.0

# Extracted LOLA window
LOLA_ROW_START = 11264
LOLA_COL_START = 6912

LOLA_HEIGHT = 512
LOLA_WIDTH = 512


def projected_to_lat(y):
    return np.rad2deg(y / MOON_RADIUS)


def projected_to_lon(x):
    return np.rad2deg(x / MOON_RADIUS)


print("=" * 60)
print("MULTIMODAL SPATIAL CORRESPONDENCE CHECK")
print("=" * 60)

# --------------------------------------------------
# WAC
# --------------------------------------------------
with rasterio.open(WAC_PATH) as src:

    transform = src.transform

    print("\nWAC native geometry")
    print("-------------------")
    print(f"Resolution: {src.res}")

    # Pixel centers of the extracted window
    wac_row_start = 4245
    wac_col_start = 8187

    wac_row_end = wac_row_start + 606 - 1
    wac_col_end = wac_col_start + 606 - 1

    x1, y1 = rasterio.transform.xy(
        transform,
        wac_row_start,
        wac_col_start,
        offset="center"
    )

    x2, y2 = rasterio.transform.xy(
        transform,
        wac_row_end,
        wac_col_end,
        offset="center"
    )

    print(f"First pixel center:")
    print(f"  lon = {projected_to_lon(x1):.6f}°")
    print(f"  lat = {projected_to_lat(y1):.6f}°")

    print(f"Last pixel center:")
    print(f"  lon = {projected_to_lon(x2):.6f}°")
    print(f"  lat = {projected_to_lat(y2):.6f}°")


# --------------------------------------------------
# LOLA
# --------------------------------------------------
print("\nLOLA native geometry")
print("--------------------")
print("Resolution: 256 pixels/degree")
print("Ground spacing: 118.4505876 m/pixel")

# Convert zero-based raster indices back to PDS image coordinates.
#
# PDS line/sample coordinates are represented by the projection
# offsets supplied in the source label.
#
# We evaluate the centers of the first and last extracted pixels.

row_first = LOLA_ROW_START
row_last = LOLA_ROW_START + LOLA_HEIGHT - 1

col_first = LOLA_COL_START
col_last = LOLA_COL_START + LOLA_WIDTH - 1

line_first = row_first + 1
line_last = row_last + 1

sample_first = col_first + 1
sample_last = col_last + 1

lat_first = (
    LOLA_LINE_OFFSET - line_first
) / LOLA_PPD

# --------------------------------------------------
# LOLA
# --------------------------------------------------
print("\nLOLA native geometry")
print("--------------------")
print("Resolution: 256 pixels/degree")
print("Ground spacing: 118.4505876 m/pixel")

row_first = LOLA_ROW_START
row_last = LOLA_ROW_START + LOLA_HEIGHT - 1

col_first = LOLA_COL_START
col_last = LOLA_COL_START + LOLA_WIDTH - 1

# Zero-based raster indices correspond directly to the
# projection-offset coordinate system used here.
lat_first = (
    LOLA_LINE_OFFSET - row_first
) / LOLA_PPD

lat_last = (
    LOLA_LINE_OFFSET - row_last
) / LOLA_PPD

lon_first = (
    LOLA_CENTER_LON
    + (col_first - LOLA_SAMPLE_OFFSET) / LOLA_PPD
)

lon_last = (
    LOLA_CENTER_LON
    + (col_last - LOLA_SAMPLE_OFFSET) / LOLA_PPD
)

print("First pixel center:")
print(f"  lon = {lon_first:.6f}°")
print(f"  lat = {lat_first:.6f}°")

print("Last pixel center:")
print(f"  lon = {lon_last:.6f}°")
print(f"  lat = {lat_last:.6f}°")

# --------------------------------------------------
# Expected geographic extent
# --------------------------------------------------
print("\nRequested AOI")
print("-------------")
print(f"Longitude: {LON_MIN}° to {LON_MAX}°")
print(f"Latitude : {LAT_MIN}° to {LAT_MAX}°")

print("\nInterpretation:")
print("The pixel-center ranges should fall just inside the")
print("requested AOI boundaries because the extracted windows")
print("contain complete pixels.")