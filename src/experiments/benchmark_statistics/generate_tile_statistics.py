from pathlib import Path
import csv
import json

import numpy as np


# ============================================================
# Project paths
# ============================================================

# Script location:
# src/experiments/benchmark_statistics/generate_tile_statistics.py
#
# parents[0] = benchmark_statistics
# parents[1] = experiments
# parents[2] = src
# parents[3] = project root

ROOT = Path(__file__).resolve().parents[3]

TILES_DIR = ROOT / "data" / "benchmark" / "tiles"
OUTPUT_DIR = ROOT / "outputs" / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "benchmark_tile_statistics.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "benchmark_tile_statistics_summary.json"


# ============================================================
# Expected benchmark structure
# ============================================================

EXPECTED_TILES = 16
TILE_SIZE = 64

EXPECTED_ARRAYS = {
    "optical": "optical.npy",
    "optical_coverage": "optical_coverage.npy",

    "dem": "dem.npy",
    "dem_valid_mask": "dem_valid_mask.npy",
    "dem_coverage": "dem_coverage.npy",

    "slope": "slope.npy",
    "hillshade": "hillshade.npy",

    "thermal": "thermal.npy",
    "thermal_valid_mask": "thermal_valid_mask.npy",
    "thermal_coverage": "thermal_coverage.npy",
}


# ============================================================
# Utility functions
# ============================================================

def finite_stats(arr):
    """
    Calculate descriptive statistics over finite numerical values.
    """

    arr = np.asarray(arr)

    if not np.issubdtype(arr.dtype, np.number):
        return {
            "count": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
        }

    finite = np.isfinite(arr)

    if not np.any(finite):
        return {
            "count": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
        }

    values = arr[finite].astype(np.float64)

    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
    }


def coverage_statistics(arr):
    """
    Calculate coverage statistics.

    For boolean validity masks:
        True = valid.

    For floating-point coverage arrays:
        values represent coverage fractions.

    Returns:
        mean_fraction
        min_fraction
        max_fraction
        full_coverage_percent

    A pixel is considered fully covered when its coverage
    fraction is >= 0.999.
    """

    arr = np.asarray(arr)

    # Boolean validity mask
    if arr.dtype == bool:

        fraction = arr.astype(np.float64)

        return {
            "mean_fraction": float(np.mean(fraction)),
            "min_fraction": float(np.min(fraction)),
            "max_fraction": float(np.max(fraction)),
            "full_coverage_percent": float(
                100.0
                * np.count_nonzero(fraction >= 0.999)
                / fraction.size
            ),
        }

    # Floating-point coverage array
    finite = np.isfinite(arr)

    if not np.any(finite):
        return {
            "mean_fraction": np.nan,
            "min_fraction": np.nan,
            "max_fraction": np.nan,
            "full_coverage_percent": np.nan,
        }

    fraction = arr[finite].astype(np.float64)

    return {
        "mean_fraction": float(np.mean(fraction)),
        "min_fraction": float(np.min(fraction)),
        "max_fraction": float(np.max(fraction)),
        "full_coverage_percent": float(
            100.0
            * np.count_nonzero(fraction >= 0.999)
            / fraction.size
        ),
    }


def column_values(rows, name):
    """
    Return finite values for a column from all tile records.
    """

    values = [
        row[name]
        for row in rows
        if name in row and np.isfinite(row[name])
    ]

    return np.asarray(values, dtype=np.float64)


# ============================================================
# Locate benchmark tiles
# ============================================================

if not TILES_DIR.exists():
    raise FileNotFoundError(
        f"Benchmark tiles directory does not exist:\n{TILES_DIR}"
    )


tile_dirs = sorted(
    p
    for p in TILES_DIR.iterdir()
    if p.is_dir() and p.name.startswith("tile_")
)


print("=" * 70)
print("BENCHMARK TILE STATISTICS")
print("=" * 70)

print(f"\nTiles found: {len(tile_dirs)}")


if len(tile_dirs) != EXPECTED_TILES:
    raise RuntimeError(
        f"Expected {EXPECTED_TILES} tiles, "
        f"found {len(tile_dirs)}."
    )


# ============================================================
# Process tiles
# ============================================================

rows = []


for tile_dir in tile_dirs:

    tile_id = tile_dir.name

    print(f"\nProcessing {tile_id}...")

    # --------------------------------------------------------
    # Load tile metadata
    # --------------------------------------------------------

    metadata_path = tile_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file:\n{metadata_path}"
        )

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as f:
        metadata = json.load(f)

    # --------------------------------------------------------
    # Pixel and geographic metadata
    # --------------------------------------------------------

    pixel_window = metadata["pixel_window"]
    geographic_extent = metadata["geographic_extent"]

    row = {
        "tile_id": tile_id,

        "tile_row": metadata["tile_index"]["row"],
        "tile_column": metadata["tile_index"]["column"],

        "row_start": pixel_window["row_start"],
        "row_end": pixel_window["row_end_exclusive"],

        "column_start": pixel_window["column_start"],
        "column_end": pixel_window["column_end_exclusive"],

        "longitude_min_deg":
            geographic_extent["longitude_min_deg"],

        "longitude_max_deg":
            geographic_extent["longitude_max_deg"],

        "latitude_min_deg":
            geographic_extent["latitude_min_deg"],

        "latitude_max_deg":
            geographic_extent["latitude_max_deg"],

        "tile_height_pixels": TILE_SIZE,
        "tile_width_pixels": TILE_SIZE,
    }

    # --------------------------------------------------------
    # Load all required arrays
    # --------------------------------------------------------

    arrays = {}

    for name, filename in EXPECTED_ARRAYS.items():

        path = tile_dir / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Missing required file for {tile_id}:\n{path}"
            )

        arr = np.load(path)

        if arr.shape != (TILE_SIZE, TILE_SIZE):
            raise ValueError(
                f"{tile_id}/{filename} has shape {arr.shape}; "
                f"expected {(TILE_SIZE, TILE_SIZE)}."
            )

        arrays[name] = arr

    # --------------------------------------------------------
    # Assign arrays
    # --------------------------------------------------------

    optical = arrays["optical"]
    optical_coverage = arrays["optical_coverage"]

    dem = arrays["dem"]
    dem_valid_mask = arrays["dem_valid_mask"]
    dem_coverage = arrays["dem_coverage"]

    slope = arrays["slope"]
    hillshade = arrays["hillshade"]

    thermal = arrays["thermal"]
    thermal_valid_mask = arrays["thermal_valid_mask"]
    thermal_coverage = arrays["thermal_coverage"]

    # ========================================================
    # COVERAGE STATISTICS
    # ========================================================

    optical_cov = coverage_statistics(
        optical_coverage
    )

    dem_cov = coverage_statistics(
        dem_coverage
    )

    thermal_cov = coverage_statistics(
        thermal_coverage
    )

    dem_valid_cov = coverage_statistics(
        dem_valid_mask
    )

    thermal_valid_cov = coverage_statistics(
        thermal_valid_mask
    )

    # --------------------------------------------------------
    # Optical coverage
    # --------------------------------------------------------

    row["optical_mean_coverage_fraction"] = (
        optical_cov["mean_fraction"]
    )

    row["optical_min_coverage_fraction"] = (
        optical_cov["min_fraction"]
    )

    row["optical_max_coverage_fraction"] = (
        optical_cov["max_fraction"]
    )

    row["optical_full_coverage_percent"] = (
        optical_cov["full_coverage_percent"]
    )

    # --------------------------------------------------------
    # DEM coverage
    # --------------------------------------------------------

    row["dem_mean_coverage_fraction"] = (
        dem_cov["mean_fraction"]
    )

    row["dem_min_coverage_fraction"] = (
        dem_cov["min_fraction"]
    )

    row["dem_max_coverage_fraction"] = (
        dem_cov["max_fraction"]
    )

    row["dem_full_coverage_percent"] = (
        dem_cov["full_coverage_percent"]
    )

    # --------------------------------------------------------
    # DEM validity mask
    # --------------------------------------------------------

    row["dem_valid_percent"] = (
        dem_valid_cov["full_coverage_percent"]
    )

    # --------------------------------------------------------
    # Thermal coverage
    # --------------------------------------------------------

    row["thermal_mean_coverage_fraction"] = (
        thermal_cov["mean_fraction"]
    )

    row["thermal_min_coverage_fraction"] = (
        thermal_cov["min_fraction"]
    )

    row["thermal_max_coverage_fraction"] = (
        thermal_cov["max_fraction"]
    )

    row["thermal_full_coverage_percent"] = (
        thermal_cov["full_coverage_percent"]
    )

    # --------------------------------------------------------
    # Thermal validity mask
    # --------------------------------------------------------

    row["thermal_valid_percent"] = (
        thermal_valid_cov["full_coverage_percent"]
    )

    # ========================================================
    # OPTICAL STATISTICS
    # ========================================================

    stats = finite_stats(optical)

    row["optical_min"] = stats["min"]
    row["optical_max"] = stats["max"]
    row["optical_mean"] = stats["mean"]
    row["optical_std"] = stats["std"]
    row["optical_median"] = stats["median"]

    # ========================================================
    # DEM STATISTICS
    # ========================================================

    stats = finite_stats(dem)

    row["dem_min"] = stats["min"]
    row["dem_max"] = stats["max"]
    row["dem_mean"] = stats["mean"]
    row["dem_std"] = stats["std"]
    row["dem_median"] = stats["median"]

    # ========================================================
    # SLOPE STATISTICS
    # ========================================================

    stats = finite_stats(slope)

    row["slope_min_deg"] = stats["min"]
    row["slope_max_deg"] = stats["max"]
    row["slope_mean_deg"] = stats["mean"]
    row["slope_std_deg"] = stats["std"]
    row["slope_median_deg"] = stats["median"]

    # ========================================================
    # HILLSHADE STATISTICS
    # ========================================================

    stats = finite_stats(hillshade)

    row["hillshade_min"] = stats["min"]
    row["hillshade_max"] = stats["max"]
    row["hillshade_mean"] = stats["mean"]
    row["hillshade_std"] = stats["std"]
    row["hillshade_median"] = stats["median"]

    # ========================================================
    # THERMAL STATISTICS
    # ========================================================

    stats = finite_stats(thermal)

    row["thermal_min"] = stats["min"]
    row["thermal_max"] = stats["max"]
    row["thermal_mean"] = stats["mean"]
    row["thermal_std"] = stats["std"]
    row["thermal_median"] = stats["median"]

    # ========================================================
    # TERRAIN INDICATORS
    # ========================================================

    slope_finite = np.isfinite(slope)

    if np.any(slope_finite):

        slope_values = slope[slope_finite]

        row["slope_gt_5deg_percent"] = float(
            100.0
            * np.count_nonzero(slope_values > 5.0)
            / slope_values.size
        )

        row["slope_gt_10deg_percent"] = float(
            100.0
            * np.count_nonzero(slope_values > 10.0)
            / slope_values.size
        )

        row["slope_gt_20deg_percent"] = float(
            100.0
            * np.count_nonzero(slope_values > 20.0)
            / slope_values.size
        )

    else:

        row["slope_gt_5deg_percent"] = np.nan
        row["slope_gt_10deg_percent"] = np.nan
        row["slope_gt_20deg_percent"] = np.nan

    # ========================================================
    # Save row
    # ========================================================

    rows.append(row)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print(
        f"  Optical coverage : "
        f"mean={row['optical_mean_coverage_fraction']:.4f}, "
        f"full={row['optical_full_coverage_percent']:.2f}%"
    )

    print(
        f"  DEM coverage     : "
        f"mean={row['dem_mean_coverage_fraction']:.4f}, "
        f"full={row['dem_full_coverage_percent']:.2f}%"
    )

    print(
        f"  Thermal coverage : "
        f"mean={row['thermal_mean_coverage_fraction']:.4f}, "
        f"full={row['thermal_full_coverage_percent']:.2f}%"
    )

    print(
        f"  Mean slope       : "
        f"{row['slope_mean_deg']:.3f} deg"
    )


# ============================================================
# Write CSV
# ============================================================

fieldnames = list(rows[0].keys())

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# Dataset-level summary
# ============================================================

summary = {

    "tile_count": len(rows),

    "optical_coverage": {

        "mean_of_tile_mean_fraction":
            float(np.mean(
                column_values(
                    rows,
                    "optical_mean_coverage_fraction"
                )
            )),

        "minimum_tile_mean_fraction":
            float(np.min(
                column_values(
                    rows,
                    "optical_mean_coverage_fraction"
                )
            )),

        "maximum_tile_mean_fraction":
            float(np.max(
                column_values(
                    rows,
                    "optical_mean_coverage_fraction"
                )
            )),

        "mean_full_coverage_percent":
            float(np.mean(
                column_values(
                    rows,
                    "optical_full_coverage_percent"
                )
            ))
    },

    "dem_coverage": {

        "mean_of_tile_mean_fraction":
            float(np.mean(
                column_values(
                    rows,
                    "dem_mean_coverage_fraction"
                )
            )),

        "minimum_tile_mean_fraction":
            float(np.min(
                column_values(
                    rows,
                    "dem_mean_coverage_fraction"
                )
            )),

        "maximum_tile_mean_fraction":
            float(np.max(
                column_values(
                    rows,
                    "dem_mean_coverage_fraction"
                )
            )),

        "mean_full_coverage_percent":
            float(np.mean(
                column_values(
                    rows,
                    "dem_full_coverage_percent"
                )
            ))
    },

    "thermal_coverage": {

        "mean_of_tile_mean_fraction":
            float(np.mean(
                column_values(
                    rows,
                    "thermal_mean_coverage_fraction"
                )
            )),

        "minimum_tile_mean_fraction":
            float(np.min(
                column_values(
                    rows,
                    "thermal_mean_coverage_fraction"
                )
            )),

        "maximum_tile_mean_fraction":
            float(np.max(
                column_values(
                    rows,
                    "thermal_mean_coverage_fraction"
                )
            )),

        "mean_full_coverage_percent":
            float(np.mean(
                column_values(
                    rows,
                    "thermal_full_coverage_percent"
                )
            ))
    },

    "slope": {

        "mean_tile_mean_deg":
            float(np.mean(
                column_values(
                    rows,
                    "slope_mean_deg"
                )
            )),

        "minimum_tile_mean_deg":
            float(np.min(
                column_values(
                    rows,
                    "slope_mean_deg"
                )
            )),

        "maximum_tile_mean_deg":
            float(np.max(
                column_values(
                    rows,
                    "slope_mean_deg"
                )
            ))
    }
}


# ============================================================
# Write summary JSON
# ============================================================

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 70)
print("BENCHMARK TILE STATISTICS COMPLETE")
print("=" * 70)

print(f"Tiles processed       : {len(rows)}")
print(f"CSV output            : {OUTPUT_CSV}")
print(f"Summary output        : {OUTPUT_SUMMARY}")

print("\nDataset-level coverage:")

print(
    f"  Optical mean coverage fraction : "
    f"{summary['optical_coverage']['mean_of_tile_mean_fraction']:.4f}"
)

print(
    f"  Optical full coverage           : "
    f"{summary['optical_coverage']['mean_full_coverage_percent']:.2f}%"
)

print(
    f"  DEM mean coverage fraction      : "
    f"{summary['dem_coverage']['mean_of_tile_mean_fraction']:.4f}"
)

print(
    f"  DEM full coverage               : "
    f"{summary['dem_coverage']['mean_full_coverage_percent']:.2f}%"
)

print(
    f"  Thermal mean coverage fraction  : "
    f"{summary['thermal_coverage']['mean_of_tile_mean_fraction']:.4f}"
)

print(
    f"  Thermal full coverage            : "
    f"{summary['thermal_coverage']['mean_full_coverage_percent']:.2f}%"
)

print(
    f"\nMean of tile mean slopes : "
    f"{summary['slope']['mean_tile_mean_deg']:.3f} deg"
)

print("=" * 70)