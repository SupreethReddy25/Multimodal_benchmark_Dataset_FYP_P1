from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    r"D:\S7\Project Phase 1\Multimodal_Lunar_Benchmark"
)

WAC_PATH = (
    ROOT
    / "data/intermediate/optical/"
    "wac_common_grid.npy"
)

WAC_COVERAGE_PATH = (
    ROOT
    / "data/intermediate/optical/"
    "wac_common_coverage.npy"
)

DEM_PATH = (
    ROOT
    / "data/intermediate/dem/"
    "lola_dem_common_grid.npy"
)

SLOPE_PATH = (
    ROOT
    / "data/intermediate/dem/"
    "lola_common_slope_deg.npy"
)

THERMAL_PATH = (
    ROOT
    / "data/intermediate/thermal/"
    "diviner_tbol_sam_common_grid.npy"
)

THERMAL_COVERAGE_PATH = (
    ROOT
    / "data/intermediate/thermal/"
    "diviner_tbol_sam_common_coverage.npy"
)

OUTPUT = (
    ROOT
    / "outputs/validation/"
    "common_grid_multimodal_qc.png"
)


# ============================================================
# Load
# ============================================================

wac = np.load(WAC_PATH)
wac_coverage = np.load(WAC_COVERAGE_PATH)

dem = np.load(DEM_PATH)

slope = np.load(SLOPE_PATH)

thermal = np.load(THERMAL_PATH)
thermal_coverage = np.load(
    THERMAL_COVERAGE_PATH
)


# ============================================================
# Validate shapes
# ============================================================

expected_shape = (256, 256)

arrays = {
    "WAC": wac,
    "WAC coverage": wac_coverage,
    "LOLA DEM": dem,
    "LOLA slope": slope,
    "Diviner": thermal,
    "Diviner coverage": thermal_coverage
}

print("========================================")
print("COMMON-GRID MULTIMODAL QC")
print("========================================")

for name, array in arrays.items():

    print(
        f"{name:20s}: "
        f"shape={array.shape}, "
        f"dtype={array.dtype}"
    )

    if array.shape != expected_shape:
        raise ValueError(
            f"{name} has unexpected shape "
            f"{array.shape}."
        )


# ============================================================
# Finite-value checks
# ============================================================

print("\nFinite-value checks")

for name, array in arrays.items():

    finite_count = np.isfinite(array).sum()

    print(
        f"{name:20s}: "
        f"{finite_count}/{array.size}"
    )


# ============================================================
# Coverage checks
# ============================================================

print("\nCoverage")

print(
    "WAC mean coverage:",
    wac_coverage.mean()
)

print(
    "WAC full coverage:",
    np.sum(wac_coverage >= 0.999),
    "/",
    wac_coverage.size
)

print(
    "Diviner mean coverage:",
    thermal_coverage.mean()
)

print(
    "Diviner full coverage:",
    np.sum(thermal_coverage >= 0.999),
    "/",
    thermal_coverage.size
)


# ============================================================
# Statistics
# ============================================================

print("\nData statistics")

print(
    "WAC:",
    "min=", np.nanmin(wac),
    "max=", np.nanmax(wac),
    "mean=", np.nanmean(wac)
)

print(
    "LOLA DEM:",
    "min=", np.nanmin(dem),
    "max=", np.nanmax(dem),
    "mean=", np.nanmean(dem)
)

print(
    "Slope:",
    "min=", np.nanmin(slope),
    "max=", np.nanmax(slope),
    "mean=", np.nanmean(slope)
)

print(
    "Diviner:",
    "min=", np.nanmin(thermal),
    "max=", np.nanmax(thermal),
    "mean=", np.nanmean(thermal)
)


# ============================================================
# Create figure
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)


# ============================================================
# WAC
# ============================================================

im0 = axes[0, 0].imshow(
    wac,
    origin="upper"
)

axes[0, 0].set_title(
    "Optical — WAC Common Grid"
)

axes[0, 0].set_xlabel(
    "Common-grid column"
)

axes[0, 0].set_ylabel(
    "Common-grid row"
)

plt.colorbar(
    im0,
    ax=axes[0, 0],
    label="WAC intensity"
)


# ============================================================
# LOLA DEM
# ============================================================

im1 = axes[0, 1].imshow(
    dem,
    origin="upper"
)

axes[0, 1].set_title(
    "Elevation — LOLA Common Grid"
)

axes[0, 1].set_xlabel(
    "Common-grid column"
)

axes[0, 1].set_ylabel(
    "Common-grid row"
)

plt.colorbar(
    im1,
    ax=axes[0, 1],
    label="Stored LOLA height (km)"
)


# ============================================================
# Slope
# ============================================================

im2 = axes[1, 0].imshow(
    slope,
    origin="upper"
)

axes[1, 0].set_title(
    "Terrain — Slope"
)

axes[1, 0].set_xlabel(
    "Common-grid column"
)

axes[1, 0].set_ylabel(
    "Common-grid row"
)

plt.colorbar(
    im2,
    ax=axes[1, 0],
    label="Slope (degrees)"
)


# ============================================================
# Diviner
# ============================================================

im3 = axes[1, 1].imshow(
    thermal,
    origin="upper"
)

axes[1, 1].set_title(
    "Thermal — Diviner Common Grid"
)

axes[1, 1].set_xlabel(
    "Common-grid column"
)

axes[1, 1].set_ylabel(
    "Common-grid row"
)

plt.colorbar(
    im3,
    ax=axes[1, 1],
    label="Diviner source value"
)


# ============================================================
# Save
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.tight_layout()

plt.savefig(
    OUTPUT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Final report
# ============================================================

print("\n========================================")
print("MULTIMODAL COMMON-GRID QC COMPLETE")
print("========================================")

print("\nOutput:")
print(OUTPUT)