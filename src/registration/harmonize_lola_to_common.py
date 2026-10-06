from pathlib import Path
import json
import numpy as np


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    r"D:\S7\Project Phase 1\Multimodal_Lunar_Benchmark"
)

INPUT_DEM = (
    ROOT / "data/intermediate/dem/lola_ldem256_pilot.npy"
)

GRID_META = (
    ROOT / "data/intermediate/registration/"
    "common_grid_metadata.json"
)

OUTPUT_DEM = (
    ROOT / "data/intermediate/dem/"
    "lola_dem_common_grid.npy"
)

OUTPUT_VALID = (
    ROOT / "data/intermediate/dem/"
    "lola_dem_common_valid_mask.npy"
)

OUTPUT_COVERAGE = (
    ROOT / "data/intermediate/dem/"
    "lola_dem_common_coverage.npy"
)

OUTPUT_META = (
    ROOT / "data/intermediate/registration/"
    "lola_common_grid_metadata.json"
)


# ============================================================
# Load native LOLA DEM
# ============================================================

dem = np.load(INPUT_DEM)

print("Native LOLA DEM")
print("Shape:", dem.shape)
print("Dtype:", dem.dtype)
print(
    "Finite:",
    np.isfinite(dem).sum(),
    "/",
    dem.size
)


# ============================================================
# Load common-grid metadata
# ============================================================

with open(GRID_META, "r") as f:
    grid_meta = json.load(f)


# ============================================================
# Read actual common-grid metadata structure
# ============================================================

target_height = grid_meta["shape"]["height"]
target_width = grid_meta["shape"]["width"]

target_shape = (
    target_height,
    target_width
)

target_resolution = (
    grid_meta["resolution"]["nominal_m_per_pixel"]
)

target_ppd = (
    grid_meta["resolution"]["pixels_per_degree"]
)

native_resolution = 118.4505876


print("\nCommon grid")
print("Shape:", target_shape)
print(
    "Resolution:",
    target_resolution,
    "m/pixel"
)
print(
    "Pixels per degree:",
    target_ppd
)


# ============================================================
# Validate native DEM
# ============================================================

expected_native_shape = (
    target_height * 2,
    target_width * 2
)

if dem.shape != expected_native_shape:
    raise ValueError(
        f"Unexpected LOLA shape: {dem.shape}. "
        f"Expected {expected_native_shape}."
    )

if not np.all(np.isfinite(dem)):
    raise ValueError(
        "LOLA DEM contains non-finite values."
    )


# ============================================================
# Validate resolution relationship
# ============================================================

ratio = (
    target_resolution /
    native_resolution
)

print(
    "\nResolution ratio:",
    ratio
)

if not np.isclose(
    ratio,
    2.0,
    rtol=1e-6
):
    raise ValueError(
        "Common-grid resolution is not exactly "
        "2x the native LOLA resolution."
    )


# ============================================================
# Validate spatial-grid relationship
# ============================================================

# Native LOLA:
#     256 pixels/degree
#
# Common grid:
#     128 pixels/degree
#
# Therefore:
#     2 x 2 native LOLA cells
#     correspond to
#     1 common-grid cell.

expected_native_ppd = 256

if expected_native_ppd != target_ppd * 2:
    raise ValueError(
        "Unexpected pixels-per-degree relationship."
    )


# ============================================================
# Reshape into 2 x 2 native blocks
# ============================================================

native_valid = np.isfinite(dem)

dem_blocks = dem.reshape(
    target_height,
    2,
    target_width,
    2
)

valid_blocks = native_valid.reshape(
    target_height,
    2,
    target_width,
    2
)


# ============================================================
# Aggregate native LOLA values
# ============================================================

valid_count = valid_blocks.sum(
    axis=(1, 3)
)

dem_sum = np.where(
    valid_blocks,
    dem_blocks,
    0.0
).sum(
    axis=(1, 3)
)


common_dem = np.full(
    target_shape,
    np.nan,
    dtype=np.float32
)

valid_cells = (
    valid_count > 0
)

common_dem[valid_cells] = (
    dem_sum[valid_cells] /
    valid_count[valid_cells]
).astype(np.float32)


# ============================================================
# Coverage
# ============================================================

coverage = (
    valid_count.astype(np.float32) /
    4.0
)

valid_mask = (
    coverage > 0
)


# ============================================================
# Save outputs
# ============================================================

np.save(
    OUTPUT_DEM,
    common_dem
)

np.save(
    OUTPUT_VALID,
    valid_mask
)

np.save(
    OUTPUT_COVERAGE,
    coverage
)


# ============================================================
# Save processing metadata
# ============================================================

metadata = {

    "source": "LRO LOLA GDRDEM",

    "source_file":
        "ldem_256_00n_90n_000_180_float.img",

    "native_resolution_m_per_pixel":
        native_resolution,

    "common_grid_resolution_m_per_pixel":
        float(target_resolution),

    "native_pixels_per_degree":
        expected_native_ppd,

    "common_grid_pixels_per_degree":
        int(target_ppd),

    "aggregation":
        "2x2 native-pixel arithmetic mean",

    "aggregation_basis":
        "aligned equirectangular angular grid",

    "native_shape":
        list(dem.shape),

    "common_grid_shape":
        list(target_shape),

    "source_value_dtype":
        str(dem.dtype),

    "output_dtype":
        str(common_dem.dtype),

    "source_value_semantics":
        (
            "LOLA stored HEIGHT relative to the "
            "lunar reference radius; not converted "
            "to conventional elevation in this step."
        ),

    "source_scale":
        1.0,

    "source_offset_km":
        1737.4,

    "validity_definition":
        (
            "Common-grid cell is valid when at least "
            "one of its four native LOLA pixels "
            "is finite."
        ),

    "coverage_definition":
        (
            "Fraction of the four native LOLA pixels "
            "that are finite."
        ),

    "native_product_preserved":
        True
}


with open(
    OUTPUT_META,
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=2
    )


# ============================================================
# Report
# ============================================================

print("\n========================================")
print("LOLA COMMON-GRID HARMONIZATION COMPLETE")
print("========================================")

print("\nOutput DEM:")
print(OUTPUT_DEM)

print("\nShape:", common_dem.shape)
print("Dtype:", common_dem.dtype)

print("\nStatistics:")
print(
    "Finite:",
    np.isfinite(common_dem).sum(),
    "/",
    common_dem.size
)

print(
    "Min:",
    np.nanmin(common_dem)
)

print(
    "Max:",
    np.nanmax(common_dem)
)

print(
    "Mean:",
    np.nanmean(common_dem)
)

print(
    "Std:",
    np.nanstd(common_dem)
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

print(
    "Valid cells:",
    np.sum(valid_mask),
    "/",
    valid_mask.size
)

print("\nSaved:")
print(OUTPUT_DEM)
print(OUTPUT_VALID)
print(OUTPUT_COVERAGE)
print(OUTPUT_META)