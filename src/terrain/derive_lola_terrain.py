from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEM_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "dem"
    / "lola_ldem256_pilot.npy"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "dem"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOLA geometry
# ============================================================

MOON_RADIUS_M = 1_737_400.0

PIXELS_PER_DEGREE = 256

LAT_MAX = 46.0
LON_MIN = 27.0

ROWS = 512
COLS = 512


# ============================================================
# Load DEM
# ============================================================

dem_km = np.load(DEM_PATH)

if dem_km.shape != (ROWS, COLS):
    raise RuntimeError(
        f"Unexpected DEM shape: {dem_km.shape}"
    )

print("DEM loaded")
print(f"Shape: {dem_km.shape}")
print(f"Dtype: {dem_km.dtype}")


# ============================================================
# Coordinate grid
# ============================================================

# Pixel-center coordinates.
#
# The extracted DEM starts at approximately:
# 46 deg N, 27 deg E.
#
# Latitude decreases as image row increases.

row = np.arange(ROWS)
col = np.arange(COLS)

lat_deg = (
    LAT_MAX
    - row / PIXELS_PER_DEGREE
)

lon_deg = (
    LON_MIN
    + col / PIXELS_PER_DEGREE
)

lat_rad = np.deg2rad(lat_deg)


# ============================================================
# Local ground spacing
# ============================================================

# Angular spacing of one DEM pixel.
dlat_rad = np.deg2rad(1.0 / PIXELS_PER_DEGREE)
dlon_rad = np.deg2rad(1.0 / PIXELS_PER_DEGREE)

# North-south ground spacing.
dy_m = MOON_RADIUS_M * dlat_rad

# East-west spacing varies with latitude.
dx_m = (
    MOON_RADIUS_M
    * np.cos(lat_rad)
    * dlon_rad
)

print("\nGround spacing")
print(f"North-south: {dy_m:.6f} m/pixel")
print(
    f"East-west range: "
    f"{dx_m.min():.6f} - {dx_m.max():.6f} m/pixel"
)


# ============================================================
# Convert DEM from km to metres
# ============================================================

dem_m = dem_km * 1000.0

valid = np.isfinite(dem_m)

# Work with NaN-safe representation.
dem_work = dem_m.copy()
dem_work[~valid] = np.nan


# ============================================================
# DEM gradients
# ============================================================

# Gradient along image rows.
#
# Image rows increase toward decreasing latitude.
# The sign therefore corresponds to the image direction,
# but slope magnitude is unaffected.

dz_drow = np.gradient(
    dem_work,
    axis=0
)

dz_dcol = np.gradient(
    dem_work,
    axis=1
)

# Convert pixel gradients to physical gradients.
#
# dx varies with latitude, therefore divide every row
# by its corresponding east-west ground spacing.

dz_dy = dz_drow / dy_m

dz_dx = (
    dz_dcol
    / dx_m[:, np.newaxis]
)


# ============================================================
# Slope
# ============================================================

slope_rad = np.arctan(
    np.sqrt(
        dz_dx ** 2
        + dz_dy ** 2
    )
)

slope_deg = np.rad2deg(slope_rad)

slope_deg[~valid] = np.nan


# ============================================================
# Hillshade
# ============================================================

# Convert derivatives to local terrain orientation.
#
# dz_dx = east-west derivative
# dz_dy = image-row derivative.
#
# For hillshade we only need a visually useful local relief
# representation, so the gradient magnitude and orientation
# are derived from the physical gradients.

aspect_rad = np.arctan2(
    -dz_dx,
    dz_dy
)

# Illumination geometry.
azimuth_deg = 315.0
altitude_deg = 45.0

azimuth_rad = np.deg2rad(azimuth_deg)
altitude_rad = np.deg2rad(altitude_deg)

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

hillshade = (
    hillshade - hillshade.min()
) / (
    hillshade.max() - hillshade.min()
)

hillshade[~valid] = np.nan


# ============================================================
# Save numerical products
# ============================================================

slope_path = (
    OUTPUT_DIR
    / "lola_slope_deg.npy"
)

hillshade_path = (
    OUTPUT_DIR
    / "lola_hillshade.npy"
)

validity_path = (
    OUTPUT_DIR
    / "lola_dem_valid_mask.npy"
)

np.save(
    slope_path,
    slope_deg.astype(np.float32)
)

np.save(
    hillshade_path,
    hillshade.astype(np.float32)
)

np.save(
    validity_path,
    valid
)


# ============================================================
# Statistics
# ============================================================

slope_valid = slope_deg[np.isfinite(slope_deg)]

print("\nTerrain derivatives")
print("-------------------")
print(f"Valid pixels: {slope_valid.size} / {slope_deg.size}")

print(
    f"Slope minimum: "
    f"{slope_valid.min():.6f} deg"
)

print(
    f"Slope maximum: "
    f"{slope_valid.max():.6f} deg"
)

print(
    f"Slope mean: "
    f"{slope_valid.mean():.6f} deg"
)

print(
    f"Slope median: "
    f"{np.median(slope_valid):.6f} deg"
)

print(
    f"Slope std: "
    f"{slope_valid.std():.6f} deg"
)


# ============================================================
# Diagnostic figure
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)

# Slope
im1 = axes[0].imshow(
    slope_deg,
    origin="upper"
)

axes[0].set_title(
    "LOLA Terrain Slope"
)

axes[0].set_xlabel(
    "Pixel column"
)

axes[0].set_ylabel(
    "Pixel row"
)

plt.colorbar(
    im1,
    ax=axes[0],
    label="Slope (degrees)"
)


# Hillshade
im2 = axes[1].imshow(
    hillshade,
    cmap="gray",
    origin="upper"
)

axes[1].set_title(
    "LOLA Hillshade"
)

axes[1].set_xlabel(
    "Pixel column"
)

axes[1].set_ylabel(
    "Pixel row"
)

plt.colorbar(
    im2,
    ax=axes[1],
    label="Normalized illumination"
)


plt.tight_layout()

figure_path = (
    FIGURE_DIR
    / "lola_terrain_derivatives.png"
)

plt.savefig(
    figure_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


print("\nSaved products:")
print(slope_path)
print(hillshade_path)
print(validity_path)
print(figure_path)