from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# Project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INTERMEDIATE = ROOT / "data" / "intermediate"
OUTPUT_DIR = ROOT / "outputs" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "final_multimodal_pilot_composite.png"
)


# ============================================================
# Load harmonized common-grid products
# ============================================================

optical = np.load(
    INTERMEDIATE
    / "optical"
    / "wac_common_grid.npy"
).astype(np.float32)

dem = np.load(
    INTERMEDIATE
    / "dem"
    / "lola_dem_common_grid.npy"
).astype(np.float32)

thermal = np.load(
    INTERMEDIATE
    / "thermal"
    / "diviner_tbol_sam_common_grid.npy"
).astype(np.float32)

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
# Validation
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

extent = [
    float(longitudes.min()),
    float(longitudes.max()),
    float(latitudes.min()),
    float(latitudes.max())
]


# ============================================================
# Prepare plotting arrays
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
# Robust normalization for RGB visualization
# ============================================================

def robust_normalize(arr, lower=2.0, upper=98.0):
    """
    Normalize an array to [0, 1] using percentile limits.

    This normalization is ONLY for visualization.
    The original benchmark arrays are not modified.
    """

    finite = np.isfinite(arr)

    if not np.any(finite):
        raise ValueError(
            "Array contains no finite values."
        )

    values = arr[finite]

    low = np.percentile(values, lower)
    high = np.percentile(values, upper)

    if high <= low:
        raise ValueError(
            "Invalid normalization range."
        )

    normalized = (
        (arr - low)
        / (high - low)
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0
    )

    normalized[~finite] = np.nan

    return normalized, float(low), float(high)


optical_norm, optical_low, optical_high = (
    robust_normalize(optical_plot)
)

dem_norm, dem_low, dem_high = (
    robust_normalize(dem_plot)
)

thermal_norm, thermal_low, thermal_high = (
    robust_normalize(thermal_plot)
)


# ============================================================
# Construct multimodal RGB composite
# ============================================================

# Red   = Optical
# Green = DEM / Topography
# Blue  = Thermal

rgb = np.stack(
    [
        optical_norm,
        dem_norm,
        thermal_norm
    ],
    axis=-1
)


# ============================================================
# Handle pixels where one or more modalities are unavailable
# ============================================================

valid_rgb = (
    np.isfinite(optical_norm)
    & np.isfinite(dem_norm)
    & np.isfinite(thermal_norm)
)

rgb_display = rgb.copy()

# Use neutral black for pixels where complete
# three-modality information is unavailable.
rgb_display[~valid_rgb] = 0.0


# ============================================================
# Create figure
# ============================================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 12),
    constrained_layout=True
)


# ============================================================
# Panel 1 — Optical
# ============================================================

im1 = axes[0, 0].imshow(
    optical_plot,
    extent=extent,
    origin="upper",
    interpolation="nearest",
    aspect="equal"
)

axes[0, 0].set_title(
    "Optical Surface\nLROC WAC 100 m",
    fontsize=13
)

axes[0, 0].set_xlabel(
    "Longitude (°E)"
)

axes[0, 0].set_ylabel(
    "Latitude (°N)"
)

cbar1 = fig.colorbar(
    im1,
    ax=axes[0, 0],
    fraction=0.046,
    pad=0.04
)

cbar1.set_label(
    "WAC radiometric value"
)


# ============================================================
# Panel 2 — DEM
# ============================================================

im2 = axes[0, 1].imshow(
    dem_plot,
    extent=extent,
    origin="upper",
    interpolation="nearest",
    aspect="equal"
)

axes[0, 1].set_title(
    "Topography\nLRO LOLA DEM",
    fontsize=13
)

axes[0, 1].set_xlabel(
    "Longitude (°E)"
)

axes[0, 1].set_ylabel(
    "Latitude (°N)"
)

cbar2 = fig.colorbar(
    im2,
    ax=axes[0, 1],
    fraction=0.046,
    pad=0.04
)

cbar2.set_label(
    "LOLA source-relative height (km)"
)


# ============================================================
# Panel 3 — Thermal
# ============================================================

im3 = axes[1, 0].imshow(
    thermal_plot,
    extent=extent,
    origin="upper",
    interpolation="nearest",
    aspect="equal"
)

axes[1, 0].set_title(
    "Thermal Surface Information\nLRO Diviner GHRM",
    fontsize=13
)

axes[1, 0].set_xlabel(
    "Longitude (°E)"
)

axes[1, 0].set_ylabel(
    "Latitude (°N)"
)

cbar3 = fig.colorbar(
    im3,
    ax=axes[1, 0],
    fraction=0.046,
    pad=0.04
)

cbar3.set_label(
    "Diviner bolometric-temperature value\n"
    "(unit not specified in downloaded PDS label)"
)


# ============================================================
# Panel 4 — Multimodal composite
# ============================================================

axes[1, 1].imshow(
    rgb_display,
    extent=extent,
    origin="upper",
    interpolation="nearest",
    aspect="equal"
)

axes[1, 1].set_title(
    "Multimodal False-Color Composite",
    fontsize=13
)

axes[1, 1].set_xlabel(
    "Longitude (°E)"
)

axes[1, 1].set_ylabel(
    "Latitude (°N)"
)


# ============================================================
# Composite legend
# ============================================================

legend_elements = [

    Patch(
        facecolor="red",
        edgecolor="black",
        label="Red — Optical WAC"
    ),

    Patch(
        facecolor="green",
        edgecolor="black",
        label="Green — LOLA Topography"
    ),

    Patch(
        facecolor="blue",
        edgecolor="black",
        label="Blue — Diviner Thermal"
    )
]

axes[1, 1].legend(
    handles=legend_elements,
    loc="lower left",
    fontsize=9,
    framealpha=0.9
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
# Save figure
# ============================================================

fig.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# Console report
# ============================================================

print("=" * 70)
print("FINAL MULTIMODAL COMPOSITE GENERATED")
print("=" * 70)

print(f"Output:")
print(OUTPUT_PATH)

print()
print("Common grid:")
print("  Shape:", optical.shape)

print(
    f"  Longitude: "
    f"{extent[0]:.6f}°E to {extent[1]:.6f}°E"
)

print(
    f"  Latitude: "
    f"{extent[2]:.6f}°N to {extent[3]:.6f}°N"
)

print()
print("RGB channel mapping:")
print("  Red   = Optical WAC")
print("  Green = LOLA DEM / topography")
print("  Blue  = Diviner thermal information")

print()
print("Visualization normalization:")
print(
    f"  Optical: {optical_low:.4f} to "
    f"{optical_high:.4f}"
)

print(
    f"  DEM: {dem_low:.6f} to "
    f"{dem_high:.6f} km"
)

print(
    f"  Thermal: {thermal_low:.6f} to "
    f"{thermal_high:.6f}"
)

print()
print(
    "Complete three-modality pixels: "
    f"{np.count_nonzero(valid_rgb)}/{valid_rgb.size}"
)

print()
print(
    "NOTE: RGB normalization is only for visualization. "
    "Original benchmark arrays remain unchanged."
)

print("=" * 70)