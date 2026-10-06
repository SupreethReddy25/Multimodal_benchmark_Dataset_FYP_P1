from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WAC_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "optical"
    / "wac_pilot.npy"
)

LOLA_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "dem"
    / "lola_ldem256_pilot.npy"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "wac_lola_overlay.png"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load native-resolution products
# --------------------------------------------------
wac = np.load(WAC_PATH)
lola = np.load(LOLA_PATH)

print("WAC shape :", wac.shape)
print("LOLA shape:", lola.shape)

# --------------------------------------------------
# Geographic extent
# --------------------------------------------------
lon_min = 27.0
lon_max = 29.0

lat_min = 44.0
lat_max = 46.0

# Both arrays cover this same geographic footprint.
wac_extent = [
    lon_min,
    lon_max,
    lat_min,
    lat_max,
]

lola_extent = [
    lon_min,
    lon_max,
    lat_min,
    lat_max,
]

# --------------------------------------------------
# Create overlay
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

ax.imshow(
    wac,
    cmap="gray",
    extent=wac_extent,
    origin="upper",
    interpolation="nearest",
)

# DEM contours
levels = np.linspace(
    np.nanmin(lola),
    np.nanmax(lola),
    15,
)

contours = ax.contour(
    np.linspace(lon_min, lon_max, lola.shape[1]),
    np.linspace(lat_max, lat_min, lola.shape[0]),
    lola,
    levels=levels,
    linewidths=0.7,
)

ax.clabel(
    contours,
    inline=True,
    fontsize=7,
    fmt="%.1f",
)

ax.set_title(
    "WAC Optical + LOLA DEM Spatial Overlay\n"
    "Lacus Mortis–Bürg Pilot | 44–46°N, 27–29°E"
)

ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(f"Saved overlay: {OUTPUT_PATH}")