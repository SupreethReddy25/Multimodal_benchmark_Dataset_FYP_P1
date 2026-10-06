from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = (
    ROOT
    / "outputs"
    / "tables"
    / "benchmark_tile_statistics.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "benchmark_quality_overview.png"
)


# ============================================================
# Load statistics
# ============================================================

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Benchmark statistics file not found:\n{CSV_PATH}"
    )


rows = []

with open(
    CSV_PATH,
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:
        rows.append(row)


if len(rows) != 16:
    raise ValueError(
        f"Expected 16 tile records, found {len(rows)}."
    )


# ============================================================
# Build 4 x 4 matrices
# ============================================================

optical_coverage = np.full(
    (4, 4),
    np.nan,
    dtype=np.float64
)

optical_full = np.full(
    (4, 4),
    np.nan,
    dtype=np.float64
)

dem_coverage = np.full(
    (4, 4),
    np.nan,
    dtype=np.float64
)

thermal_coverage = np.full(
    (4, 4),
    np.nan,
    dtype=np.float64
)

mean_slope = np.full(
    (4, 4),
    np.nan,
    dtype=np.float64
)


for row in rows:

    tile_row = int(row["tile_row"])
    tile_col = int(row["tile_column"])

    optical_coverage[
        tile_row,
        tile_col
    ] = float(
        row["optical_mean_coverage_fraction"]
    )

    optical_full[
        tile_row,
        tile_col
    ] = float(
        row["optical_full_coverage_percent"]
    )

    dem_coverage[
        tile_row,
        tile_col
    ] = float(
        row["dem_mean_coverage_fraction"]
    )

    thermal_coverage[
        tile_row,
        tile_col
    ] = float(
        row["thermal_mean_coverage_fraction"]
    )

    mean_slope[
        tile_row,
        tile_col
    ] = float(
        row["slope_mean_deg"]
    )


# ============================================================
# Print matrices
# ============================================================

print("=" * 70)
print("BENCHMARK QUALITY MAP")
print("=" * 70)

print("\nOptical mean coverage:")
print(np.round(optical_coverage, 4))

print("\nOptical full coverage (%):")
print(np.round(optical_full, 2))

print("\nDEM mean coverage:")
print(np.round(dem_coverage, 4))

print("\nThermal mean coverage:")
print(np.round(thermal_coverage, 4))

print("\nMean slope (degrees):")
print(np.round(mean_slope, 3))


# ============================================================
# Create figure
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(15, 9)
)


# ------------------------------------------------------------
# Panel 1: Optical mean coverage
# ------------------------------------------------------------

im1 = axes[0, 0].imshow(
    optical_coverage,
    vmin=0,
    vmax=1,
    interpolation="nearest"
)

axes[0, 0].set_title(
    "Optical Mean Coverage Fraction"
)

axes[0, 0].set_xlabel("Tile Column")
axes[0, 0].set_ylabel("Tile Row")

fig.colorbar(
    im1,
    ax=axes[0, 0],
    fraction=0.046,
    pad=0.04
)


# ------------------------------------------------------------
# Panel 2: Optical full coverage
# ------------------------------------------------------------

im2 = axes[0, 1].imshow(
    optical_full,
    vmin=0,
    vmax=100,
    interpolation="nearest"
)

axes[0, 1].set_title(
    "Optical Fully Covered Pixels (%)"
)

axes[0, 1].set_xlabel("Tile Column")
axes[0, 1].set_ylabel("Tile Row")

fig.colorbar(
    im2,
    ax=axes[0, 1],
    fraction=0.046,
    pad=0.04
)


# ------------------------------------------------------------
# Panel 3: Mean slope
# ------------------------------------------------------------

im3 = axes[0, 2].imshow(
    mean_slope,
    interpolation="nearest"
)

axes[0, 2].set_title(
    "Mean Slope (degrees)"
)

axes[0, 2].set_xlabel("Tile Column")
axes[0, 2].set_ylabel("Tile Row")

fig.colorbar(
    im3,
    ax=axes[0, 2],
    fraction=0.046,
    pad=0.04
)


# ------------------------------------------------------------
# Panel 4: DEM coverage
# ------------------------------------------------------------

im4 = axes[1, 0].imshow(
    dem_coverage,
    vmin=0,
    vmax=1,
    interpolation="nearest"
)

axes[1, 0].set_title(
    "DEM Mean Coverage Fraction"
)

axes[1, 0].set_xlabel("Tile Column")
axes[1, 0].set_ylabel("Tile Row")

fig.colorbar(
    im4,
    ax=axes[1, 0],
    fraction=0.046,
    pad=0.04
)


# ------------------------------------------------------------
# Panel 5: Thermal coverage
# ------------------------------------------------------------

im5 = axes[1, 1].imshow(
    thermal_coverage,
    vmin=0,
    vmax=1,
    interpolation="nearest"
)

axes[1, 1].set_title(
    "Thermal Mean Coverage Fraction"
)

axes[1, 1].set_xlabel("Tile Column")
axes[1, 1].set_ylabel("Tile Row")

fig.colorbar(
    im5,
    ax=axes[1, 1],
    fraction=0.046,
    pad=0.04
)


# ------------------------------------------------------------
# Panel 6: Tile IDs
# ------------------------------------------------------------

axes[1, 2].set_xlim(-0.5, 3.5)
axes[1, 2].set_ylim(3.5, -0.5)

axes[1, 2].set_xticks(range(4))
axes[1, 2].set_yticks(range(4))

axes[1, 2].set_xlabel("Tile Column")
axes[1, 2].set_ylabel("Tile Row")

axes[1, 2].set_title(
    "Benchmark Tile Layout"
)

for r in range(4):

    for c in range(4):

        tile_id = f"tile_{r:02d}_{c:02d}"

        axes[1, 2].text(
            c,
            r,
            tile_id,
            ha="center",
            va="center"
        )


# ============================================================
# Figure formatting
# ============================================================

fig.suptitle(
    "Multimodal Lunar Benchmark — Pilot Tile Quality Overview",
    fontsize=16
)

plt.tight_layout()

fig.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 70)
print("BENCHMARK QUALITY MAP GENERATED")
print("=" * 70)

print(f"Output:")
print(OUTPUT_PATH)

print("=" * 70)