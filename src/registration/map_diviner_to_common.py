from pathlib import Path
import json

import numpy as np


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    r"D:\S7\Project Phase 1\Multimodal_Lunar_Benchmark"
)

THERMAL_INPUT = (
    ROOT
    / "data/intermediate/thermal/"
    "diviner_tbol_sam_pilot.npy"
)

GRID_META = (
    ROOT
    / "data/intermediate/registration/"
    "common_grid_metadata.json"
)

COMMON_LON = (
    ROOT
    / "data/intermediate/registration/"
    "common_grid_longitudes.npy"
)

COMMON_LAT = (
    ROOT
    / "data/intermediate/registration/"
    "common_grid_latitudes.npy"
)

THERMAL_OUTPUT = (
    ROOT
    / "data/intermediate/thermal/"
    "diviner_tbol_sam_common_grid.npy"
)

VALID_OUTPUT = (
    ROOT
    / "data/intermediate/thermal/"
    "diviner_tbol_sam_common_valid_mask.npy"
)

COVERAGE_OUTPUT = (
    ROOT
    / "data/intermediate/thermal/"
    "diviner_tbol_sam_common_coverage.npy"
)

META_OUTPUT = (
    ROOT
    / "data/intermediate/registration/"
    "diviner_common_grid_metadata.json"
)


# ============================================================
# Load Diviner thermal data
# ============================================================

thermal = np.load(THERMAL_INPUT)

print("Native Diviner pilot")
print("Shape:", thermal.shape)
print("Dtype:", thermal.dtype)

print(
    "Finite:",
    np.isfinite(thermal).sum(),
    "/",
    thermal.size
)

print(
    "NaN:",
    np.isnan(thermal).sum()
)


# ============================================================
# Load common-grid metadata
# ============================================================

with open(GRID_META, "r") as f:
    grid_meta = json.load(f)


# ============================================================
# Expected common-grid shape
# ============================================================

expected_shape = (
    grid_meta["shape"]["height"],
    grid_meta["shape"]["width"]
)


if thermal.shape != expected_shape:
    raise ValueError(
        f"Unexpected Diviner shape: {thermal.shape}. "
        f"Expected {expected_shape}."
    )


# ============================================================
# Load common-grid coordinates
# ============================================================

common_lon = np.load(COMMON_LON)
common_lat = np.load(COMMON_LAT)

print("\nCoordinate arrays")
print("Longitude shape:", common_lon.shape)
print("Latitude shape:", common_lat.shape)


# ============================================================
# Support either 1-D coordinate vectors or 2-D grids
# ============================================================

if common_lon.ndim == 1 and common_lat.ndim == 1:

    if len(common_lon) != expected_shape[1]:
        raise ValueError(
            "Longitude vector length does not match "
            "common-grid width."
        )

    if len(common_lat) != expected_shape[0]:
        raise ValueError(
            "Latitude vector length does not match "
            "common-grid height."
        )

    common_lon_2d, common_lat_2d = np.meshgrid(
        common_lon,
        common_lat
    )

    coordinate_storage = "1-D coordinate vectors"

elif common_lon.ndim == 2 and common_lat.ndim == 2:

    if common_lon.shape != expected_shape:
        raise ValueError(
            f"Longitude grid shape {common_lon.shape} "
            f"does not match common grid "
            f"{expected_shape}."
        )

    if common_lat.shape != expected_shape:
        raise ValueError(
            f"Latitude grid shape {common_lat.shape} "
            f"does not match common grid "
            f"{expected_shape}."
        )

    common_lon_2d = common_lon
    common_lat_2d = common_lat

    coordinate_storage = "2-D coordinate grids"

else:

    raise ValueError(
        "Longitude and latitude arrays must both be "
        "either 1-D or 2-D."
    )


print(
    "Coordinate representation:",
    coordinate_storage
)

print(
    "2-D longitude shape:",
    common_lon_2d.shape
)

print(
    "2-D latitude shape:",
    common_lat_2d.shape
)


# ============================================================
# Diviner / common-grid geometry
# ============================================================

DIVINER_RESOLUTION_M = 236.9011751886678

COMMON_RESOLUTION_M = (
    grid_meta["resolution"]["nominal_m_per_pixel"]
)

DIVINER_PPD = 128

COMMON_PPD = (
    grid_meta["resolution"]["pixels_per_degree"]
)


print("\nGeometry")

print(
    "Diviner resolution:",
    DIVINER_RESOLUTION_M,
    "m/pixel"
)

print(
    "Common resolution:",
    COMMON_RESOLUTION_M,
    "m/pixel"
)

print(
    "Diviner pixels/degree:",
    DIVINER_PPD
)

print(
    "Common pixels/degree:",
    COMMON_PPD
)


# ============================================================
# Resolution validation
# ============================================================

if not np.isclose(
    DIVINER_RESOLUTION_M,
    COMMON_RESOLUTION_M,
    rtol=1e-9,
    atol=1e-9
):
    raise ValueError(
        "Diviner and common-grid resolutions do not match."
    )


if DIVINER_PPD != COMMON_PPD:
    raise ValueError(
        "Diviner and common-grid pixels-per-degree differ."
    )


# ============================================================
# Pixel-center validation
# ============================================================

expected_first_lon = (
    grid_meta["center_coordinates"]
    ["first_pixel"]
    ["longitude_deg"]
)

expected_first_lat = (
    grid_meta["center_coordinates"]
    ["first_pixel"]
    ["latitude_deg"]
)

expected_last_lon = (
    grid_meta["center_coordinates"]
    ["last_pixel"]
    ["longitude_deg"]
)

expected_last_lat = (
    grid_meta["center_coordinates"]
    ["last_pixel"]
    ["latitude_deg"]
)


actual_first_lon = common_lon_2d[0, 0]
actual_first_lat = common_lat_2d[0, 0]

actual_last_lon = common_lon_2d[-1, -1]
actual_last_lat = common_lat_2d[-1, -1]


first_lon_diff = abs(
    actual_first_lon - expected_first_lon
)

first_lat_diff = abs(
    actual_first_lat - expected_first_lat
)

last_lon_diff = abs(
    actual_last_lon - expected_last_lon
)

last_lat_diff = abs(
    actual_last_lat - expected_last_lat
)


print("\nPixel-center geometry")

print(
    "First longitude difference:",
    first_lon_diff
)

print(
    "First latitude difference:",
    first_lat_diff
)

print(
    "Last longitude difference:",
    last_lon_diff
)

print(
    "Last latitude difference:",
    last_lat_diff
)


# ============================================================
# Pixel-center tolerance
# ============================================================

GEOMETRY_TOLERANCE_DEG = 1e-10

if first_lon_diff > GEOMETRY_TOLERANCE_DEG:
    raise ValueError(
        "First-pixel longitude does not match "
        "common-grid metadata."
    )

if first_lat_diff > GEOMETRY_TOLERANCE_DEG:
    raise ValueError(
        "First-pixel latitude does not match "
        "common-grid metadata."
    )

if last_lon_diff > GEOMETRY_TOLERANCE_DEG:
    raise ValueError(
        "Last-pixel longitude does not match "
        "common-grid metadata."
    )

if last_lat_diff > GEOMETRY_TOLERANCE_DEG:
    raise ValueError(
        "Last-pixel latitude does not match "
        "common-grid metadata."
    )


# ============================================================
# Angular spacing validation
# ============================================================

lon_step = np.diff(
    common_lon_2d[0, :]
)

lat_step = np.diff(
    common_lat_2d[:, 0]
)


print("\nAngular spacing")

print(
    "Longitude step:",
    lon_step.min(),
    "to",
    lon_step.max()
)

print(
    "Latitude step:",
    lat_step.min(),
    "to",
    lat_step.max()
)


EXPECTED_LON_STEP = 1.0 / 128.0
EXPECTED_LAT_STEP = -1.0 / 128.0


if not np.allclose(
    lon_step,
    EXPECTED_LON_STEP,
    atol=1e-10
):
    raise ValueError(
        "Unexpected longitude spacing."
    )


if not np.allclose(
    lat_step,
    EXPECTED_LAT_STEP,
    atol=1e-10
):
    raise ValueError(
        "Unexpected latitude spacing."
    )


# ============================================================
# Verify regular coordinate mesh
# ============================================================

# Longitude should be constant along each row.
lon_row_difference = np.max(
    np.abs(
        common_lon_2d
        - common_lon_2d[0, :][None, :]
    )
)

# Latitude should be constant along each column.
lat_column_difference = np.max(
    np.abs(
        common_lat_2d
        - common_lat_2d[:, 0][:, None]
    )
)


print("\nGrid regularity")

print(
    "Maximum longitude row difference:",
    lon_row_difference
)

print(
    "Maximum latitude column difference:",
    lat_column_difference
)


if lon_row_difference > GEOMETRY_TOLERANCE_DEG:
    raise ValueError(
        "Longitude grid is not a regular row-wise grid."
    )

if lat_column_difference > GEOMETRY_TOLERANCE_DEG:
    raise ValueError(
        "Latitude grid is not a regular column-wise grid."
    )


# ============================================================
# Direct native mapping
# ============================================================

# IMPORTANT:
#
# No interpolation is performed.
#
# Diviner already has:
#
#     236.901175 m/pixel
#     128 pixels/degree
#
# which exactly matches the common grid.
#
# Therefore each Diviner pixel maps directly to the
# corresponding common-grid pixel.

thermal_common = thermal.astype(
    np.float32,
    copy=True
)


# ============================================================
# Validity mask
# ============================================================

valid_mask = np.isfinite(
    thermal_common
)


# ============================================================
# Coverage
# ============================================================

coverage = valid_mask.astype(
    np.float32
)


# ============================================================
# Save harmonized thermal product
# ============================================================

np.save(
    THERMAL_OUTPUT,
    thermal_common
)

np.save(
    VALID_OUTPUT,
    valid_mask
)

np.save(
    COVERAGE_OUTPUT,
    coverage
)


# ============================================================
# Metadata
# ============================================================

metadata = {

    "source":
        "LRO Diviner Global High-Resolution Mosaic",

    "source_file":
        "dghrm_tbol_sam_70s70n_img.img",

    "source_label":
        "dghrm_tbol_sam_70s70n_img.xml",

    "native_resolution_m_per_pixel":
        DIVINER_RESOLUTION_M,

    "common_grid_resolution_m_per_pixel":
        COMMON_RESOLUTION_M,

    "native_pixels_per_degree":
        DIVINER_PPD,

    "common_grid_pixels_per_degree":
        COMMON_PPD,

    "mapping_method":
        "direct native-grid mapping",

    "resampling":
        "none",

    "interpolation":
        "none",

    "pixel_center_alignment":
        "exact",

    "coordinate_storage":
        coordinate_storage,

    "native_shape":
        list(thermal.shape),

    "common_grid_shape":
        list(expected_shape),

    "source_dtype":
        str(thermal.dtype),

    "output_dtype":
        str(thermal_common.dtype),

    "validity_definition":
        "Finite Diviner source value",

    "coverage_definition":
        "1 for finite source pixel, 0 otherwise",

    "physical_unit":
        (
            "Not specified in the downloaded PDS label; "
            "source values are preserved without unit conversion."
        ),

    "native_product_preserved":
        True
}


with open(
    META_OUTPUT,
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=2
    )


# ============================================================
# Final statistics
# ============================================================

print("\n========================================")
print("DIVINER COMMON-GRID MAPPING COMPLETE")
print("========================================")

print("\nOutput:")
print(THERMAL_OUTPUT)

print("\nShape:")
print(thermal_common.shape)

print("Dtype:")
print(thermal_common.dtype)

print("\nStatistics:")

print(
    "Finite:",
    np.isfinite(thermal_common).sum(),
    "/",
    thermal_common.size
)

print(
    "NaN:",
    np.isnan(thermal_common).sum()
)

print(
    "Min:",
    np.nanmin(thermal_common)
)

print(
    "Max:",
    np.nanmax(thermal_common)
)

print(
    "Mean:",
    np.nanmean(thermal_common)
)

print(
    "Std:",
    np.nanstd(thermal_common)
)

print("\nCoverage:")

print(
    "Min:",
    coverage.min()
)

print(
    "Max:",
    coverage.max()
)

print(
    "Mean:",
    coverage.mean()
)

print(
    "Fully covered cells:",
    np.sum(coverage >= 0.999),
    "/",
    coverage.size
)

print("\nGeometry validation: PASS")

print("\nSaved:")
print(THERMAL_OUTPUT)
print(VALID_OUTPUT)
print(COVERAGE_OUTPUT)
print(META_OUTPUT)