from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    r"D:\S7\Project Phase 1\Multimodal_Lunar_Benchmark"
)

DEM_PATH = (
    ROOT / "data/intermediate/dem/"
    "lola_dem_common_grid.npy"
)

SLOPE_OUTPUT = (
    ROOT / "data/intermediate/dem/"
    "lola_common_slope_deg.npy"
)

HILLSHADE_OUTPUT = (
    ROOT / "data/intermediate/dem/"
    "lola_common_hillshade.npy"
)

FIGURE_OUTPUT = (
    ROOT / "outputs/validation/"
    "lola_common_terrain_derivatives.png"
)


# ============================================================
# Common-grid geometry
# ============================================================

# Common grid:
# 128 pixels / degree
# Lunar reference radius:
# 1,737,400 m

PIXELS_PER_DEGREE = 128.0
LUNAR_RADIUS_M = 1_737_400.0

DEG_PER_PIXEL = 1.0 / PIXELS_PER_DEGREE
DEG_TO_RAD = np.pi / 180.0


# ============================================================
# Load common-grid DEM
# ============================================================

dem = np.load(DEM_PATH).astype(np.float64)

print("Common-grid DEM")
print("Shape:", dem.shape)
print("Dtype:", dem.dtype)
print(
    "Finite:",
    np.isfinite(dem).sum(),
    "/",
    dem.size
)


# ============================================================
# Validate
# ============================================================

if dem.shape != (256, 256):
    raise ValueError(
        f"Unexpected DEM shape: {dem.shape}. "
        "Expected (256, 256)."
    )

if not np.all(np.isfinite(dem)):
    raise ValueError(
        "Common-grid DEM contains non-finite values."
    )


# ============================================================
# Pixel spacing
# ============================================================

# North-south spacing is constant in an equirectangular
# projection.

dy_m = (
    LUNAR_RADIUS_M
    * DEG_TO_RAD
    * DEG_PER_PIXEL
)


# Longitude spacing depends on latitude:
#
# dx = R cos(latitude) d_lambda

lat_centers = (
    46.0
    - (np.arange(dem.shape[0]) + 0.5)
    * DEG_PER_PIXEL
)

lat_rad = lat_centers * DEG_TO_RAD

dx_m = (
    LUNAR_RADIUS_M
    * np.cos(lat_rad)
    * DEG_TO_RAD
    * DEG_PER_PIXEL
)


print("\nPixel spacing")
print(
    "North-south:",
    dy_m,
    "m/pixel"
)

print(
    "East-west range:",
    dx_m.min(),
    "to",
    dx_m.max(),
    "m/pixel"
)


# ============================================================
# Gradient
# ============================================================

# DEM is stored in kilometres.
# Convert to metres before calculating gradients.

dem_m = dem * 1000.0


# np.gradient returns:
# gradient along latitude/row direction
# gradient along longitude/column direction

dz_drow, dz_dcol = np.gradient(
    dem_m
)


# ------------------------------------------------------------
# Convert gradients to physical derivatives
# ------------------------------------------------------------

# Row direction corresponds to north-south.
dz_dy = dz_drow / dy_m

# Longitude direction has latitude-dependent spacing.
dz_dx = (
    dz_dcol
    / dx_m[:, None]
)


# ============================================================
# Slope
# ============================================================

slope_rad = np.arctan(
    np.sqrt(
        dz_dx ** 2
        +
        dz_dy ** 2
    )
)

slope_deg = np.degrees(
    slope_rad
).astype(np.float32)


# ============================================================
# Hillshade
# ============================================================

# Standard visualization geometry:
# azimuth = 315 degrees
# altitude = 45 degrees

azimuth_deg = 315.0
altitude_deg = 45.0

azimuth_rad = np.radians(
    azimuth_deg
)

altitude_rad = np.radians(
    altitude_deg
)

# Surface orientation

aspect_rad = np.arctan2(
    -dz_dx,
    dz_dy
)

hillshade = (
    np.sin(altitude_rad)
    * np.cos(slope_rad)
    +
    np.cos(altitude_rad)
    * np.sin(slope_rad)
    * np.cos(
        azimuth_rad - aspect_rad
    )
)

hillshade = np.clip(
    hillshade,
    0,
    1
).astype(np.float32)


# ============================================================
# Save
# ============================================================

np.save(
    SLOPE_OUTPUT,
    slope_deg
)

np.save(
    HILLSHADE_OUTPUT,
    hillshade
)


# ============================================================
# Statistics
# ============================================================

print("\nSlope statistics")
print(
    "Min:",
    np.min(slope_deg)
)

print(
    "Max:",
    np.max(slope_deg)
)

print(
    "Mean:",
    np.mean(slope_deg)
)

print(
    "Median:",
    np.median(slope_deg)
)

print(
    "Std:",
    np.std(slope_deg)
)


print("\nHillshade statistics")
print(
    "Min:",
    np.min(hillshade)
)

print(
    "Max:",
    np.max(hillshade)
)

print(
    "Mean:",
    np.mean(hillshade)
)

print(
    "Std:",
    np.std(hillshade)
)


# ============================================================
# QC figure
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)


# ------------------------------------------------------------
# Slope
# ------------------------------------------------------------

im0 = axes[0].imshow(
    slope_deg,
    origin="upper"
)

axes[0].set_title(
    "LOLA Common-Grid Slope"
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
    label="Slope (degrees)"
)


# ------------------------------------------------------------
# Hillshade
# ------------------------------------------------------------

im1 = axes[1].imshow(
    hillshade,
    origin="upper",
    vmin=0,
    vmax=1
)

axes[1].set_title(
    "LOLA Common-Grid Hillshade"
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
    label="Relative illumination"
)


plt.tight_layout()

FIGURE_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

plt.savefig(
    FIGURE_OUTPUT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Final report
# ============================================================

print("\n========================================")
print("COMMON-GRID TERRAIN DERIVATION COMPLETE")
print("========================================")

print("\nSaved:")
print(SLOPE_OUTPUT)
print(HILLSHADE_OUTPUT)
print(FIGURE_OUTPUT)