from pathlib import Path
import json
import hashlib
import platform
import sys

import numpy as np
import rasterio


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERMEDIATE = ROOT / "data" / "intermediate"
DATA_BENCHMARK = ROOT / "data" / "benchmark"
OUTPUT_DIR = DATA_BENCHMARK / "metadata"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = OUTPUT_DIR / "benchmark_manifest.json"


# ============================================================
# Utility functions
# ============================================================

def sha256_file(path, chunk_size=1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def file_info(path):
    path = Path(path)

    if not path.exists():
        return {
            "path": str(path.relative_to(ROOT)),
            "exists": False
        }

    return {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path)
    }


def numpy_info(path):
    path = Path(path)

    if not path.exists():
        return {
            "path": str(path.relative_to(ROOT)),
            "exists": False
        }

    arr = np.load(path, mmap_mode="r")

    finite = np.isfinite(arr)

    info = {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite_values": int(np.count_nonzero(finite)),
        "total_values": int(arr.size),
    }

    if np.issubdtype(arr.dtype, np.number) and np.any(finite):
        info.update({
            "min": float(np.nanmin(arr)),
            "max": float(np.nanmax(arr)),
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr))
        })

    return info


def raster_info(path):
    path = Path(path)

    if not path.exists():
        return {
            "path": str(path.relative_to(ROOT)),
            "exists": False
        }

    with rasterio.open(path) as src:
        return {
            "path": str(path.relative_to(ROOT)),
            "exists": True,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": str(src.crs),
            "bounds": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top
            },
            "resolution": list(src.res),
            "nodata": src.nodata,
            "transform": list(src.transform)
        }


# ============================================================
# Load common-grid metadata
# ============================================================

common_grid_path = (
    DATA_INTERMEDIATE
    / "registration"
    / "common_grid_metadata.json"
)

with open(common_grid_path, "r", encoding="utf-8") as f:
    common_grid = json.load(f)


# ============================================================
# Source information
# ============================================================

wac_source = DATA_RAW / "optical" / "WAC_GLOBAL_E300N0450_100M.tif"

lola_label = (
    DATA_RAW
    / "dem"
    / "LDEM_256_00N_90N_000_180_FLOAT.LBL"
)

lola_img = (
    DATA_RAW
    / "dem"
    / "LDEM_256_00N_90N_000_180_FLOAT.IMG"
)

thermal_label = (
    DATA_RAW
    / "thermal"
    / "dghrm_tbol_sam_70s70n_img.xml"
)


# ============================================================
# Intermediate products
# ============================================================

products = {
    "optical_native_pilot": numpy_info(
        DATA_INTERMEDIATE / "optical" / "wac_pilot.npy"
    ),

    "optical_common_grid": numpy_info(
        DATA_INTERMEDIATE / "optical" / "wac_common_grid.npy"
    ),

    "optical_common_coverage": numpy_info(
        DATA_INTERMEDIATE / "optical" / "wac_common_coverage.npy"
    ),

    "dem_native_pilot": numpy_info(
        DATA_INTERMEDIATE / "dem" / "lola_ldem256_pilot.npy"
    ),

    "dem_native_slope": numpy_info(
        DATA_INTERMEDIATE / "dem" / "lola_slope_deg.npy"
    ),

    "dem_native_hillshade": numpy_info(
        DATA_INTERMEDIATE / "dem" / "lola_hillshade.npy"
    ),

    "dem_common_grid": numpy_info(
        DATA_INTERMEDIATE / "dem" / "lola_dem_common_grid.npy"
    ),

    "dem_common_valid_mask": numpy_info(
        DATA_INTERMEDIATE / "dem" / "lola_dem_common_valid_mask.npy"
    ),

    "dem_common_coverage": numpy_info(
        DATA_INTERMEDIATE / "dem" / "lola_dem_common_coverage.npy"
    ),

    "dem_common_slope": numpy_info(
        DATA_INTERMEDIATE / "dem" / "lola_common_slope_deg.npy"
    ),

    "dem_common_hillshade": numpy_info(
        DATA_INTERMEDIATE / "dem" / "lola_common_hillshade.npy"
    ),

    "thermal_native_pilot": numpy_info(
        DATA_INTERMEDIATE / "thermal" / "diviner_tbol_sam_pilot.npy"
    ),

    "thermal_common_grid": numpy_info(
        DATA_INTERMEDIATE / "thermal" / "diviner_tbol_sam_common_grid.npy"
    ),

    "thermal_common_valid_mask": numpy_info(
        DATA_INTERMEDIATE
        / "thermal"
        / "diviner_tbol_sam_common_valid_mask.npy"
    ),

    "thermal_common_coverage": numpy_info(
        DATA_INTERMEDIATE
        / "thermal"
        / "diviner_tbol_sam_common_coverage.npy"
    )
}


# ============================================================
# Validation outputs
# ============================================================

validation_files = {}

validation_dir = ROOT / "outputs" / "validation"

for path in sorted(validation_dir.glob("*")):
    if path.is_file():
        validation_files[path.name] = file_info(path)


# ============================================================
# Software environment
# ============================================================

environment = {
    "python_version": sys.version,
    "platform": platform.platform(),
    "rasterio_version": rasterio.__version__,
    "numpy_version": np.__version__
}


# ============================================================
# Benchmark manifest
# ============================================================

manifest = {

    "dataset": {
        "name": "Multimodal Lunar Benchmark",
        "version": "1.0.0",
        "phase": "Phase 1",
        "purpose": (
            "Standardized multimodal lunar terrain benchmark construction "
            "for autonomous landing hazard analysis."
        )
    },

    "pilot_region": {
        "name": "Lacus Mortis - Burg Pilot",
        "latitude_min_deg": 44.0,
        "latitude_max_deg": 46.0,
        "longitude_min_deg": 27.0,
        "longitude_max_deg": 29.0,
        "center_latitude_deg": 45.0,
        "center_longitude_deg": 28.0
    },

    "modalities": [
        "optical",
        "dem",
        "thermal"
    ],

    "common_grid": common_grid,

    "source_datasets": {

        "optical": {
            "product": "LROC WAC Global Morphology Mosaic 100 m",
            "source_agency": "USGS / LROC",
            "source_file": file_info(wac_source),
            "native_resolution_m_per_pixel": 100.0,
            "native_projection": "Lunar Equirectangular",
            "latitude_convention": "planetocentric",
            "longitude_domain": "-180 to +180 degrees",
            "bands": 1,
            "data_type": "uint8",
            "harmonization": {
                "method": "Reprojection and area averaging",
                "target_resolution_m_per_pixel": (
                    common_grid["resolution"]["nominal_m_per_pixel"]
                )
            },
            "validity_definition": (
                "Source metadata-defined NoData value of 0 was retained "
                "as the source validity condition during harmonization."
            ),
            "native_product_preserved": True
        },

        "dem": {
            "product": "LRO LOLA Global DEM / GDRDEM",
            "source_agency": "NASA / USGS / PDS",
            "label_file": file_info(lola_label),
            "data_file": file_info(lola_img),
            "native_resolution_m_per_pixel": 118.4505876,
            "native_resolution_pixels_per_degree": 256,
            "native_projection": "Simple Cylindrical",
            "latitude_convention": "planetocentric",
            "longitude_domain": "0 to 180 degrees for selected source tile",
            "data_type": "32-bit floating point",
            "stored_quantity": (
                "Source-relative height values as defined by the PDS "
                "product metadata; not relabeled as conventional elevation."
            ),
            "harmonization": {
                "method": "Exact 2x2 aggregation from 256 to 128 pixels/degree",
                "target_resolution_m_per_pixel": (
                    common_grid["resolution"]["nominal_m_per_pixel"]
                )
            },
            "derived_products": [
                "slope_degrees",
                "hillshade"
            ],
            "native_product_preserved": True
        },

        "thermal": {
            "product": (
                "LRO Diviner Global High-Resolution Mosaic of "
                "Bolometric Temperature at Slope-Adjusted Midnight"
            ),
            "source_agency": "NASA / PDS Geosciences",
            "label_file": file_info(thermal_label),
            "native_resolution_m_per_pixel": 236.90117518866782,
            "native_resolution_pixels_per_degree": 128,
            "native_projection": "Equirectangular",
            "latitude_convention": "planetocentric",
            "longitude_domain": "0 to 360 degrees",
            "data_type": "IEEE 754 32-bit floating point",
            "source_missing_value": "NaN",
            "physical_unit": (
                "Not specified in the downloaded PDS label; "
                "source values are preserved without assigning a unit."
            ),
            "harmonization": {
                "method": "Direct native-grid mapping",
                "interpolation": False,
                "target_resolution_m_per_pixel": (
                    common_grid["resolution"]["nominal_m_per_pixel"]
                )
            },
            "native_product_preserved": True
        }
    },

    "products": products,

    "validation": {
        "geometric": [
            "common_grid_definition",
            "coordinate_range_validation",
            "native_grid correspondence",
            "Diviner_common_grid_identity_validation"
        ],
        "coverage": [
            "WAC coverage mask",
            "LOLA validity mask",
            "Diviner coverage mask"
        ],
        "visual": [
            "WAC common-grid QC",
            "LOLA common-grid QC",
            "common-grid multimodal QC",
            "native multimodal overlay"
        ],
        "quantitative": [
            "cross-modal mutual information",
            "cross-modal normalized mutual information",
            "shift diagnostic",
            "cross-modal edge correspondence"
        ]
    },

    "validation_outputs": validation_files,

    "processing_environment": environment,

    "provenance": {
        "native_products_preserved": True,
        "source_metadata_preserved": True,
        "coverage_masks_preserved": True,
        "processing_principle": (
            "Native-resolution products are preserved and transformed "
            "onto an explicitly defined common spatial framework."
        )
    }
}


# ============================================================
# Write manifest
# ============================================================

with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("=" * 70)
print("BENCHMARK MANIFEST GENERATED")
print("=" * 70)
print(f"Output: {MANIFEST_PATH}")
print()
print("Dataset:", manifest["dataset"]["name"])
print("Version:", manifest["dataset"]["version"])
print("Pilot:", manifest["pilot_region"]["name"])
print(
    "Common grid:",
    common_grid["shape"]["height"],
    "x",
    common_grid["shape"]["width"]
)
print(
    "Resolution:",
    common_grid["resolution"]["nominal_m_per_pixel"],
    "m/pixel"
)
print()
print("Modalities:")
for modality in manifest["modalities"]:
    print(f"  - {modality}")

print()
print("Validation files recorded:", len(validation_files))
print("=" * 70)