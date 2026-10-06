from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

WAC_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "optical"
    / "WAC_GLOBAL_E300N0450_100M.tif"
)

GRID_DIR = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "registration"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "optical"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Common-grid geometry
# ============================================================

LON_MIN = 27.0
LON_MAX = 29.0

LAT_MIN = 44.0
LAT_MAX = 46.0

MOON_RADIUS_M = 1_737_400.0

TARGET_PPD = 128

TARGET_WIDTH = 256
TARGET_HEIGHT = 256


# Pixel size of the common grid in the lunar
# equirectangular projection.
TARGET_PIXEL_SIZE = (
    MOON_RADIUS_M
    * np.deg2rad(1.0 / TARGET_PPD)
)


# Upper-left pixel corner.
TARGET_LEFT = (
    MOON_RADIUS_M
    * np.deg2rad(LON_MIN)
)

TARGET_TOP = (
    MOON_RADIUS_M
    * np.deg2rad(LAT_MAX)
)


TARGET_TRANSFORM = Affine(
    TARGET_PIXEL_SIZE,
    0.0,
    TARGET_LEFT,
    0.0,
    -TARGET_PIXEL_SIZE,
    TARGET_TOP,
)


# ============================================================
# Output paths
# ============================================================

WAC_OUTPUT = (
    OUTPUT_DIR
    / "wac_common_grid.npy"
)

COVERAGE_OUTPUT = (
    OUTPUT_DIR
    / "wac_common_coverage.npy"
)


# ============================================================
# Open source WAC
# ============================================================

print("=" * 60)
print("WAC → COMMON GRID HARMONIZATION")
print("=" * 60)

print("\nSource:")
print(WAC_PATH)

with rasterio.open(WAC_PATH) as src:

    print("\nSource geometry:")
    print(f"CRS       : {src.crs}")
    print(f"Resolution: {src.res}")
    print(f"Size      : {src.width} × {src.height}")
    print(f"Dtype     : {src.dtypes[0]}")
    print(f"NoData    : {src.nodata}")

    # --------------------------------------------------------
    # Destination array
    # --------------------------------------------------------

    wac_common = np.empty(
        (TARGET_HEIGHT, TARGET_WIDTH),
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Area-weighted aggregation
    # --------------------------------------------------------

    reproject(
        source=rasterio.band(src, 1),
        destination=wac_common,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=TARGET_TRANSFORM,
        dst_crs=src.crs,
        resampling=Resampling.average,
    )


# ============================================================
# Coverage calculation
# ============================================================

# A source-validity raster is used to quantify how much
# source support contributes to each common-grid cell.

with rasterio.open(WAC_PATH) as src:

    if src.nodata is None:

        source_valid = np.ones(
            (src.height, src.width),
            dtype=np.float32,
        )

    else:

        source_valid = np.ones(
            (src.height, src.width),
            dtype=np.float32,
        )

        source_data = src.read(
            1,
            masked=False,
        )

        source_valid[
            source_data == src.nodata
        ] = 0.0

    coverage = np.zeros(
        (TARGET_HEIGHT, TARGET_WIDTH),
        dtype=np.float32,
    )

    reproject(
        source=source_valid,
        destination=coverage,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=TARGET_TRANSFORM,
        dst_crs=src.crs,
        resampling=Resampling.average,
    )


# ============================================================
# Save
# ============================================================

np.save(
    WAC_OUTPUT,
    wac_common,
)

np.save(
    COVERAGE_OUTPUT,
    coverage,
)


# ============================================================
# Statistics
# ============================================================

finite = np.isfinite(wac_common)

print("\nCommon-grid WAC:")
print(f"Shape : {wac_common.shape}")
print(f"Dtype : {wac_common.dtype}")
print(
    f"Finite pixels: "
    f"{finite.sum()} / {wac_common.size}"
)

if finite.any():

    values = wac_common[finite]

    print(
        f"Min   : {values.min():.6f}"
    )

    print(
        f"Max   : {values.max():.6f}"
    )

    print(
        f"Mean  : {values.mean():.6f}"
    )

    print(
        f"Std   : {values.std():.6f}"
    )


print("\nCoverage:")
print(
    f"Minimum: {coverage.min():.6f}"
)

print(
    f"Maximum: {coverage.max():.6f}"
)

print(
    f"Mean   : {coverage.mean():.6f}"
)

print(
    "Cells with full coverage:",
    np.count_nonzero(
        coverage >= 0.999
    ),
    "/",
    coverage.size,
)


print("\nTarget transform:")
print(TARGET_TRANSFORM)

print("\nSaved:")
print(WAC_OUTPUT)
print(COVERAGE_OUTPUT)