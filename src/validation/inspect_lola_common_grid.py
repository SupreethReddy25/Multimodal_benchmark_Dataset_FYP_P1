from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(
    r"D:\S7\Project Phase 1\Multimodal_Lunar_Benchmark"
)

DEM_PATH = (
    ROOT / "data/intermediate/dem/"
    "lola_dem_common_grid.npy"
)

COVERAGE_PATH = (
    ROOT / "data/intermediate/dem/"
    "lola_dem_common_coverage.npy"
)

OUTPUT = (
    ROOT / "outputs/validation/"
    "lola_common_grid_qc.png"
)


# ============================================================
# Load
# ============================================================

dem = np.load(DEM_PATH)
coverage = np.load(COVERAGE_PATH)


# ============================================================
# Basic checks
# ============================================================

print("LOLA common-grid DEM")
print("Shape:", dem.shape)
print("Dtype:", dem.dtype)

print(
    "Finite:",
    np.isfinite(dem).sum(),
    "/",
    dem.size
)

print("\nCoverage")
print("Shape:", coverage.shape)
print("Min:", coverage.min())
print("Max:", coverage.max())
print("Mean:", coverage.mean())


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


# ------------------------------------------------------------
# DEM
# ------------------------------------------------------------

im0 = axes[0].imshow(
    dem,
    origin="upper"
)

axes[0].set_title(
    "LOLA DEM — Common Grid"
)

axes[0].set_xlabel(
    "Longitude direction"
)

axes[0].set_ylabel(
    "Latitude direction"
)

plt.colorbar(
    im0,
    ax=axes[0],
    label="Stored LOLA height (km)"
)


# ------------------------------------------------------------
# Coverage
# ------------------------------------------------------------

im1 = axes[1].imshow(
    coverage,
    origin="upper",
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "LOLA Common-Grid Coverage"
)

axes[1].set_xlabel(
    "Longitude direction"
)

axes[1].set_ylabel(
    "Latitude direction"
)

plt.colorbar(
    im1,
    ax=axes[1],
    label="Coverage fraction"
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

print("\nSaved:")
print(OUTPUT)