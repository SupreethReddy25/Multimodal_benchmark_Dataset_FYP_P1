from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INTERMEDIATE = ROOT / "data" / "intermediate"
OUTPUT_DIR = ROOT / "outputs" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "final_three_modality_pilot.png"
)


# ============================================================
# Load common-grid data
# ============================================================

optical = np.load(
    INTERMEDIATE
    / "optical"
    / "wac_common_grid.npy"
)

dem = np.load(
    INTERMEDIATE
    / "dem"
    / "lola_dem_common_grid.npy"
)

thermal = np.load(
    INTERMEDIATE
    / "thermal"
    / "diviner_tbol_sam_common_grid.npy"
)

longitudes = np.load(
    INTERMEDIATE
    / "registration"
    / "common_grid_longitudes.npy"
)

latitudes = np.load(
    INTERMEDIATE
    / "registration"
    / "common_grid_latitudes.npy"
)


# ============================================================
# Validate dimensions
# ============================================================

expected_shape = (256, 256)

for name, arr in [
    ("Optical", optical),
    ("DEM", dem),
    ("Thermal", thermal),
]:

    if arr.shape != expected_shape:
        raise ValueError(
            f"{name} has shape {arr.shape}; "
            f"expected {expected_shape}."
        )


if longitudes.shape != (256,):
    raise ValueError(
        f"Unexpected longitude shape: {longitudes.shape}"
    )

if latitudes.shape != (256,):
    raise ValueError(
        f"Unexpected latitude shape: {latitudes.shape}"
    )


# ============================================================
# Coordinate extent
# ============================================================

lon_min = float(longitudes.min())
lon_max = float(longitudes.max())

lat_min = float(latitudes.min())
lat_max = float(latitudes.max())

# imshow extent expects:
# left, right, bottom, top
extent = [
    lon_min,
    lon_max,
    lat_min,
    lat_max
]


# ============================================================
# Mask invalid values
# ============================================================

optical_plot = np.where(
    np.isfinite(optical),
    optical,
    np.nan
)

dem_plot = np.where(
    np.isfinite(dem),
    dem,
    np.nan
)

thermal_plot = np.where(
    np.isfinite(thermal),
    thermal,
    np.nan
)


# ============================================================
# Create figure
# ============================================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6.5),
    constrained_layout=True
)


# ============================================================
# Optical panel
# ============================================================

im1 = axes[0].imshow(
    optical_plot,
    extent=extent,
    origin="upper",
    interpolation="nearest",
    aspect="equal"
)

axes[0].set_title(
    "Optical Lunar Surface\nLROC WAC 100 m"
)

axes[0].set_xlabel(
    "Longitude (°E)"
)

axes[0].set_ylabel(
    "Latitude (°N)"
)

cbar1 = fig.colorbar(
    im1,
    ax=axes[0],
    fraction=0.046,
    pad=0.04
)

cbar1.set_label(
    "WAC radiometric value"
)


# ============================================================
# DEM panel
# ============================================================

im2 = axes[1].imshow(
    dem_plot,
    extent=extent,
    origin="upper",
    interpolation="nearest",
    aspect="equal"
)

axes[1].set_title(
    "Lunar Topography\nLRO LOLA DEM"
)

axes[1].set_xlabel(
    "Longitude (°E)"
)

axes[1].set_ylabel(
    "Latitude (°N)"
)

cbar2 = fig.colorbar(
    im2,
    ax=axes[1],
    fraction=0.046,
    pad=0.04
)

cbar2.set_label(
    "LOLA source-relative height (km)"
)


# ============================================================
# Thermal panel
# ============================================================

im3 = axes[2].imshow(
    thermal_plot,
    extent=extent,
    origin="upper",
    interpolation="nearest",
    aspect="equal"
)

axes[2].set_title(
    "Thermal Surface Information\nLRO Diviner GHRM"
)

axes[2].set_xlabel(
    "Longitude (°E)"
)

axes[2].set_ylabel(
    "Latitude (°N)"
)

cbar3 = fig.colorbar(
    im3,
    ax=axes[2],
    fraction=0.046,
    pad=0.04
)

cbar3.set_label(
    "Diviner bolometric-temperature value\n"
    "(unit not specified in downloaded PDS label)"
)


# ============================================================
# Overall title
# ============================================================

fig.suptitle(
    "Standardized Multimodal Lunar Terrain Representation — "
    "Lacus Mortis–Bürg Pilot",
    fontsize=17
)


# ============================================================
# Save
# ============================================================

fig.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Report
# ============================================================

print("=" * 70)
print("FINAL THREE-MODALITY FIGURE GENERATED")
print("=" * 70)

print(f"Output:")
print(OUTPUT_PATH)

print()
print("Common grid:")
print("  Shape       :", optical.shape)
print(
    "  Longitude   : "
    f"{lon_min:.6f}°E to {lon_max:.6f}°E"
)
print(
    "  Latitude    : "
    f"{lat_min:.6f}°N to {lat_max:.6f}°N"
)

print()
print("Optical:")
print(
    f"  Range       : "
    f"{np.nanmin(optical):.4f} – "
    f"{np.nanmax(optical):.4f}"
)

print("DEM:")
print(
    f"  Range       : "
    f"{np.nanmin(dem):.6f} – "
    f"{np.nanmax(dem):.6f} km"
)

print("Thermal:")
print(
    f"  Range       : "
    f"{np.nanmin(thermal):.6f} – "
    f"{np.nanmax(thermal):.6f}"
)

print()
print("All three panels use the same 256 × 256 common-grid geometry.")
print("=" * 70)