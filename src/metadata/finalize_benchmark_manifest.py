from pathlib import Path
import json
import hashlib


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    ROOT
    / "data"
    / "benchmark"
    / "metadata"
    / "benchmark_manifest.json"
)

TILE_INDEX_PATH = (
    ROOT
    / "data"
    / "benchmark"
    / "metadata"
    / "tile_index.json"
)

STATISTICS_CSV = (
    ROOT
    / "outputs"
    / "tables"
    / "benchmark_tile_statistics.csv"
)

STATISTICS_SUMMARY = (
    ROOT
    / "outputs"
    / "tables"
    / "benchmark_tile_statistics_summary.json"
)

QUALITY_FIGURE = (
    ROOT
    / "outputs"
    / "figures"
    / "benchmark_quality_overview.png"
)

VALIDATION_REPORT = (
    ROOT
    / "outputs"
    / "validation"
    / "benchmark_tile_validation.txt"
)

VALIDATION_JSON = (
    ROOT
    / "outputs"
    / "validation"
    / "benchmark_tile_validation.json"
)


# ============================================================
# Utility
# ============================================================

def file_record(path):
    """
    Record existence, size, and SHA-256 checksum.
    """

    if not path.exists():
        return {
            "path": str(path.relative_to(ROOT)),
            "exists": False
        }

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256.hexdigest()
    }


# ============================================================
# Check required files
# ============================================================

required_files = [
    MANIFEST_PATH,
    TILE_INDEX_PATH,
    STATISTICS_CSV,
    STATISTICS_SUMMARY,
    QUALITY_FIGURE,
    VALIDATION_REPORT,
    VALIDATION_JSON
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required benchmark artifact does not exist:\n{path}"
        )


# ============================================================
# Load existing manifest
# ============================================================

with open(
    MANIFEST_PATH,
    "r",
    encoding="utf-8"
) as f:

    manifest = json.load(f)


# ============================================================
# Load tile index
# ============================================================

with open(
    TILE_INDEX_PATH,
    "r",
    encoding="utf-8"
) as f:

    tile_index = json.load(f)


# ============================================================
# Load statistics summary
# ============================================================

with open(
    STATISTICS_SUMMARY,
    "r",
    encoding="utf-8"
) as f:

    statistics_summary = json.load(f)


# ============================================================
# Load tile validation report
# ============================================================

with open(
    VALIDATION_JSON,
    "r",
    encoding="utf-8"
) as f:

    validation = json.load(f)


# ============================================================
# Benchmark tile package
# ============================================================

manifest["benchmark_package"] = {

    "status": "pilot_benchmark_constructed",

    "tile_configuration": {
        "tile_size_pixels": 64,
        "tile_overlap_pixels": 0,
        "rows": 4,
        "columns": 4,
        "total_tiles": 16
    },

    "spatial_partition": {
        "common_grid_height": 256,
        "common_grid_width": 256,
        "pixels_per_degree": 128,
        "tile_span_degrees": 0.5,
        "non_overlapping": True,
        "complete_spatial_coverage": True
    },

    "tile_index": file_record(
        TILE_INDEX_PATH
    ),

    "tile_count": len(
        tile_index.get("tiles", [])
    ),

    "tile_directory": (
        "data/benchmark/tiles"
    )
}


# ============================================================
# Benchmark quality statistics
# ============================================================

manifest["benchmark_statistics"] = {

    "statistics_table": file_record(
        STATISTICS_CSV
    ),

    "statistics_summary": file_record(
        STATISTICS_SUMMARY
    ),

    "summary": statistics_summary
}


# ============================================================
# Benchmark structural validation
# ============================================================

manifest["benchmark_validation"] = {

    "structural_validation_report": file_record(
        VALIDATION_REPORT
    ),

    "structural_validation_json": file_record(
        VALIDATION_JSON
    ),

    "result": validation.get(
        "status",
        "UNKNOWN"
    ),

    "expected_tiles": validation.get(
        "benchmark",
        {}
    ).get(
        "expected_tiles"
    ),

    "actual_tiles": validation.get(
        "benchmark",
        {}
    ).get(
        "actual_tiles"
    ),

    "covered_pixels": validation.get(
        "spatial_coverage",
        {}
    ).get(
        "covered_pixels"
    ),

    "missing_pixels": validation.get(
        "spatial_coverage",
        {}
    ).get(
        "missing_pixels"
    ),

    "duplicated_pixels": validation.get(
        "spatial_coverage",
        {}
    ).get(
        "duplicated_pixels"
    )
}


# ============================================================
# Benchmark quality visualization
# ============================================================

manifest["benchmark_quality_visualization"] = {

    "file": file_record(
        QUALITY_FIGURE
    ),

    "description": (
        "Four-by-four pilot tile overview showing optical "
        "coverage, optical full-coverage percentage, DEM "
        "coverage, thermal coverage, mean tile slope, and "
        "tile spatial layout."
    )
}


# ============================================================
# Phase 1 status
# ============================================================

manifest["phase_1_status"] = {

    "status": "pilot_benchmark_complete",

    "scope": [
        "native source preservation",
        "multimodal spatial harmonization",
        "common-grid construction",
        "terrain derivative generation",
        "coverage and validity handling",
        "cross-modal validation",
        "provenance tracking",
        "synchronized benchmark tiling",
        "tile structural validation",
        "tile-level quality statistics"
    ],

    "phase_2_started": False,

    "phase_2_components_not_included": [
        "uncertainty estimation",
        "adaptive landing decision",
        "landing-site prediction",
        "temporal multimodal fusion",
        "hazard decision model",
        "reinforcement learning"
    ]
}


# ============================================================
# Write finalized manifest
# ============================================================

with open(
    MANIFEST_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        manifest,
        f,
        indent=2
    )


# ============================================================
# Console output
# ============================================================

print("=" * 70)
print("FINAL BENCHMARK MANIFEST UPDATED")
print("=" * 70)

print(f"Manifest:")
print(MANIFEST_PATH)

print()
print("Benchmark tiles:")
print(
    manifest["benchmark_package"]["tile_count"]
)

print(
    "Structural validation:",
    manifest["benchmark_validation"]["result"]
)

print(
    "Covered pixels:",
    manifest["benchmark_validation"]["covered_pixels"]
)

print(
    "Missing pixels:",
    manifest["benchmark_validation"]["missing_pixels"]
)

print(
    "Duplicated pixels:",
    manifest["benchmark_validation"]["duplicated_pixels"]
)

print()
print("Phase 1 status:")
print(
    manifest["phase_1_status"]["status"]
)

print(
    "Phase 2 started:",
    manifest["phase_1_status"]["phase_2_started"]
)

print("=" * 70)