from pathlib import Path
import json

import numpy as np


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TILES_DIR = ROOT / "data" / "benchmark" / "tiles"
METADATA_DIR = ROOT / "data" / "benchmark" / "metadata"
GRID_METADATA_PATH = (
    ROOT
    / "data"
    / "intermediate"
    / "registration"
    / "common_grid_metadata.json"
)

OUTPUT_DIR = ROOT / "outputs" / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = OUTPUT_DIR / "benchmark_tile_validation.txt"
JSON_PATH = OUTPUT_DIR / "benchmark_tile_validation.json"


# ============================================================
# Expected benchmark structure
# ============================================================

TILE_SIZE = 64
EXPECTED_ROWS = 4
EXPECTED_COLS = 4
EXPECTED_TILES = 16

EXPECTED_ARRAYS = [
    "optical.npy",
    "optical_coverage.npy",
    "dem.npy",
    "dem_valid_mask.npy",
    "dem_coverage.npy",
    "slope.npy",
    "hillshade.npy",
    "thermal.npy",
    "thermal_valid_mask.npy",
    "thermal_coverage.npy",
]

EXPECTED_METADATA = "metadata.json"


# ============================================================
# Load common-grid metadata
# ============================================================

with open(GRID_METADATA_PATH, "r", encoding="utf-8") as f:
    grid = json.load(f)

grid_height = grid["shape"]["height"]
grid_width = grid["shape"]["width"]

expected_grid_pixels = grid_height * grid_width


# ============================================================
# Validation state
# ============================================================

errors = []
warnings = []

tile_results = []

coverage_counter = np.zeros(
    (grid_height, grid_width),
    dtype=np.int16
)


# ============================================================
# Find tile directories
# ============================================================

tile_dirs = sorted(
    p for p in TILES_DIR.iterdir()
    if p.is_dir() and p.name.startswith("tile_")
)

print("=" * 70)
print("BENCHMARK TILE VALIDATION")
print("=" * 70)

print(f"\nTile directories found: {len(tile_dirs)}")


if len(tile_dirs) != EXPECTED_TILES:
    errors.append(
        f"Expected {EXPECTED_TILES} tile directories, "
        f"found {len(tile_dirs)}."
    )


# ============================================================
# Validate individual tiles
# ============================================================

for tile_dir in tile_dirs:

    tile_id = tile_dir.name

    print(f"\nChecking {tile_id}...")

    tile_errors = []

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    for filename in EXPECTED_ARRAYS + [EXPECTED_METADATA]:

        path = tile_dir / filename

        if not path.exists():
            tile_errors.append(
                f"Missing file: {filename}"
            )

    if tile_errors:
        errors.extend(
            [f"{tile_id}: {e}" for e in tile_errors]
        )

        tile_results.append({
            "tile_id": tile_id,
            "status": "FAIL",
            "errors": tile_errors
        })

        print("  FAIL")
        continue

    # --------------------------------------------------------
    # Read metadata
    # --------------------------------------------------------

    try:
        with open(
            tile_dir / "metadata.json",
            "r",
            encoding="utf-8"
        ) as f:
            metadata = json.load(f)
    except Exception as exc:

        message = f"Could not read metadata: {exc}"

        errors.append(
            f"{tile_id}: {message}"
        )

        tile_results.append({
            "tile_id": tile_id,
            "status": "FAIL",
            "errors": [message]
        })

        print("  FAIL")
        continue

    # --------------------------------------------------------
    # Validate tile ID
    # --------------------------------------------------------

    if metadata.get("tile_id") != tile_id:

        message = (
            f"Metadata tile_id is "
            f"{metadata.get('tile_id')!r}, "
            f"expected {tile_id!r}."
        )

        tile_errors.append(message)

    # --------------------------------------------------------
    # Validate arrays
    # --------------------------------------------------------

    loaded_arrays = {}

    for filename in EXPECTED_ARRAYS:

        path = tile_dir / filename
        name = filename.replace(".npy", "")

        try:
            arr = np.load(path)
            loaded_arrays[name] = arr

        except Exception as exc:

            tile_errors.append(
                f"Could not load {filename}: {exc}"
            )
            continue

        if arr.shape != (TILE_SIZE, TILE_SIZE):

            tile_errors.append(
                f"{filename}: shape {arr.shape}, "
                f"expected {(TILE_SIZE, TILE_SIZE)}."
            )

    # --------------------------------------------------------
    # Check synchronized shapes
    # --------------------------------------------------------

    shapes = {
        name: arr.shape
        for name, arr in loaded_arrays.items()
    }

    unique_shapes = set(shapes.values())

    if unique_shapes and unique_shapes != {
        (TILE_SIZE, TILE_SIZE)
    }:

        tile_errors.append(
            f"Not all arrays have the expected synchronized "
            f"shape: {shapes}"
        )

    # --------------------------------------------------------
    # Validate tile pixel window
    # --------------------------------------------------------

    pixel_window = metadata.get("pixel_window", {})

    row_start = pixel_window.get("row_start")
    row_end = pixel_window.get("row_end_exclusive")
    col_start = pixel_window.get("column_start")
    col_end = pixel_window.get("column_end_exclusive")

    if None in (
        row_start,
        row_end,
        col_start,
        col_end
    ):

        tile_errors.append(
            "Incomplete pixel_window metadata."
        )

    else:

        if row_end - row_start != TILE_SIZE:

            tile_errors.append(
                "Metadata row window does not equal "
                f"{TILE_SIZE} pixels."
            )

        if col_end - col_start != TILE_SIZE:

            tile_errors.append(
                "Metadata column window does not equal "
                f"{TILE_SIZE} pixels."
            )

        # ----------------------------------------------------
        # Check that tile lies inside the common grid
        # ----------------------------------------------------

        if (
            row_start < 0
            or row_end > grid_height
            or col_start < 0
            or col_end > grid_width
        ):

            tile_errors.append(
                "Tile pixel window lies outside the "
                "common grid."
            )

        else:

            # ------------------------------------------------
            # Record spatial coverage
            # ------------------------------------------------

            coverage_counter[
                row_start:row_end,
                col_start:col_end
            ] += 1

    # --------------------------------------------------------
    # Validate tile metadata shape
    # --------------------------------------------------------

    metadata_shape = metadata.get("shape", {})

    if (
        metadata_shape.get("height") != TILE_SIZE
        or metadata_shape.get("width") != TILE_SIZE
    ):

        tile_errors.append(
            "Metadata shape does not match 64x64."
        )

    # --------------------------------------------------------
    # Validate tile geographic metadata
    # --------------------------------------------------------

    geographic_extent = metadata.get(
        "geographic_extent"
    )

    if not geographic_extent:

        tile_errors.append(
            "Missing geographic_extent metadata."
        )

    # --------------------------------------------------------
    # Validate correspondence declaration
    # --------------------------------------------------------

    tile_generation = metadata.get(
        "tile_generation",
        {}
    )

    if (
        tile_generation.get(
            "pixel_correspondence_preserved"
        )
        is not True
    ):

        tile_errors.append(
            "pixel_correspondence_preserved is not True."
        )

    # --------------------------------------------------------
    # Record result
    # --------------------------------------------------------

    if tile_errors:

        status = "FAIL"

        errors.extend(
            [f"{tile_id}: {e}" for e in tile_errors]
        )

        print("  FAIL")

    else:

        status = "PASS"

        print("  PASS")

    tile_results.append({
        "tile_id": tile_id,
        "status": status,
        "errors": tile_errors
    })


# ============================================================
# Global spatial coverage validation
# ============================================================

print("\n" + "-" * 70)
print("Checking global spatial coverage...")

coverage_min = int(coverage_counter.min())
coverage_max = int(coverage_counter.max())

covered_pixels = int(
    np.count_nonzero(coverage_counter)
)

duplicated_pixels = int(
    np.count_nonzero(coverage_counter > 1)
)

missing_pixels = int(
    np.count_nonzero(coverage_counter == 0)
)


if covered_pixels != expected_grid_pixels:

    errors.append(
        f"Only {covered_pixels}/{expected_grid_pixels} "
        "common-grid pixels are covered."
    )

else:

    print(
        f"  Covered pixels: "
        f"{covered_pixels}/{expected_grid_pixels}"
    )


if duplicated_pixels > 0:

    errors.append(
        f"{duplicated_pixels} common-grid pixels are covered "
        "by more than one tile."
    )

else:

    print("  Duplicate pixels: 0")


if missing_pixels > 0:

    errors.append(
        f"{missing_pixels} common-grid pixels are not "
        "covered by any tile."
    )

else:

    print("  Missing pixels: 0")


print(f"  Coverage minimum: {coverage_min}")
print(f"  Coverage maximum: {coverage_max}")


# ============================================================
# Validate expected 4x4 tile arrangement
# ============================================================

expected_tile_ids = {
    f"tile_{row:02d}_{col:02d}"
    for row in range(EXPECTED_ROWS)
    for col in range(EXPECTED_COLS)
}

actual_tile_ids = {
    p.name for p in tile_dirs
}

missing_tile_ids = sorted(
    expected_tile_ids - actual_tile_ids
)

unexpected_tile_ids = sorted(
    actual_tile_ids - expected_tile_ids
)

if missing_tile_ids:

    errors.append(
        "Missing expected tile IDs: "
        + ", ".join(missing_tile_ids)
    )

if unexpected_tile_ids:

    errors.append(
        "Unexpected tile IDs: "
        + ", ".join(unexpected_tile_ids)
    )


# ============================================================
# Validate tile index
# ============================================================

tile_index_path = (
    METADATA_DIR
    / "tile_index.json"
)

if not tile_index_path.exists():

    errors.append(
        "tile_index.json does not exist."
    )

else:

    try:

        with open(
            tile_index_path,
            "r",
            encoding="utf-8"
        ) as f:
            tile_index = json.load(f)

        indexed_tiles = tile_index.get(
            "tiles",
            []
        )

        if len(indexed_tiles) != EXPECTED_TILES:

            errors.append(
                f"tile_index.json contains "
                f"{len(indexed_tiles)} tiles; "
                f"expected {EXPECTED_TILES}."
            )

    except Exception as exc:

        errors.append(
            f"Could not read tile_index.json: {exc}"
        )


# ============================================================
# Final status
# ============================================================

overall_status = (
    "PASS"
    if not errors
    else "FAIL"
)


# ============================================================
# Machine-readable report
# ============================================================

report = {

    "status": overall_status,

    "benchmark": {
        "expected_tiles": EXPECTED_TILES,
        "actual_tiles": len(tile_dirs),
        "tile_size_pixels": TILE_SIZE,
        "expected_grid": [
            grid_height,
            grid_width
        ]
    },

    "spatial_coverage": {
        "expected_pixels": expected_grid_pixels,
        "covered_pixels": covered_pixels,
        "missing_pixels": missing_pixels,
        "duplicated_pixels": duplicated_pixels,
        "coverage_min": coverage_min,
        "coverage_max": coverage_max
    },

    "tiles": tile_results,

    "errors": errors,

    "warnings": warnings
}


with open(
    JSON_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# ============================================================
# Human-readable report
# ============================================================

lines = []

lines.append("=" * 70)
lines.append("BENCHMARK TILE VALIDATION REPORT")
lines.append("=" * 70)
lines.append("")
lines.append(f"Overall status       : {overall_status}")
lines.append(f"Expected tiles       : {EXPECTED_TILES}")
lines.append(f"Actual tiles         : {len(tile_dirs)}")
lines.append(
    f"Tile size            : "
    f"{TILE_SIZE} x {TILE_SIZE}"
)
lines.append(
    f"Common grid          : "
    f"{grid_height} x {grid_width}"
)
lines.append("")
lines.append("Spatial coverage")
lines.append("-" * 70)
lines.append(
    f"Expected pixels      : {expected_grid_pixels}"
)
lines.append(
    f"Covered pixels       : {covered_pixels}"
)
lines.append(
    f"Missing pixels       : {missing_pixels}"
)
lines.append(
    f"Duplicated pixels    : {duplicated_pixels}"
)
lines.append(
    f"Coverage minimum     : {coverage_min}"
)
lines.append(
    f"Coverage maximum     : {coverage_max}"
)
lines.append("")
lines.append("Tile results")
lines.append("-" * 70)

for result in tile_results:

    lines.append(
        f"{result['tile_id']:12s} : "
        f"{result['status']}"
    )

lines.append("")
lines.append("Errors")
lines.append("-" * 70)

if errors:

    for error in errors:
        lines.append(f"- {error}")

else:

    lines.append("None")

lines.append("")
lines.append("Warnings")
lines.append("-" * 70)

if warnings:

    for warning in warnings:
        lines.append(f"- {warning}")

else:

    lines.append("None")

lines.append("")
lines.append("=" * 70)

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(lines))


# ============================================================
# Console summary
# ============================================================

print("\n" + "=" * 70)
print("BENCHMARK TILE VALIDATION COMPLETE")
print("=" * 70)

print(f"Overall status    : {overall_status}")
print(f"Expected tiles    : {EXPECTED_TILES}")
print(f"Actual tiles      : {len(tile_dirs)}")
print(f"Covered pixels    : {covered_pixels}/{expected_grid_pixels}")
print(f"Missing pixels    : {missing_pixels}")
print(f"Duplicated pixels : {duplicated_pixels}")
print(f"Errors            : {len(errors)}")
print(f"Warnings          : {len(warnings)}")

print(f"\nReport:")
print(REPORT_PATH)

print("\nJSON report:")
print(JSON_PATH)

print("=" * 70)

if errors:
    raise SystemExit(1)