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

THERMAL_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "thermal"
    / "diviner_tbol_sam_pilot.npy"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "multimodal_native_overlay.png"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load native products
# ------------------------------------------------------------

wac = np.load(WAC_PATH)
lola = np.load(LOLA_PATH)
thermal = np.load(THERMAL_PATH)

print("Native shapes:")
print(f"WAC     : {wac.shape}")
print(f"LOLA    : {lola.shape}")
print(f"Thermal : {thermal.shape}")


# ------------------------------------------------------------
# Common geographic extent
# ------------------------------------------------------------

lon_min = 27.0
lon_max = 29.0

lat_min = 44.0
lat_max = 46.0


# ------------------------------------------------------------
# Figure
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 9))


# ------------------------------------------------------------
# WAC background
# ------------------------------------------------------------

ax.imshow(
    wac,
    cmap="gray",
    extent=[
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    ],
    origin="upper",
    interpolation="nearest",
)


# ------------------------------------------------------------
# Thermal overlay
# ------------------------------------------------------------

thermal_masked = np.ma.masked_invalid(thermal)

thermal_image = ax.imshow(
    thermal_masked,
    extent=[
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    ],
    origin="upper",
    alpha=0.42,
    interpolation="nearest",
)


# ------------------------------------------------------------
# LOLA contours
# ------------------------------------------------------------

levels = np.linspace(
    np.nanmin(lola),
    np.nanmax(lola),
    12,
)

contours = ax.contour(
    np.linspace(
        lon_min,
        lon_max,
        lola.shape[1],
    ),
    np.linspace(
        lat_max,
        lat_min,
        lola.shape[0],
    ),
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


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

ax.set_title(
    "Native-Grid Multimodal Lunar Overlay\n"
    "WAC + Diviner tbol_sam + LOLA | "
    "44–46°N, 27–29°E"
)

ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)


# ------------------------------------------------------------
# Thermal colorbar
# ------------------------------------------------------------

cbar = fig.colorbar(
    thermal_image,
    ax=ax,
    pad=0.02,
)

cbar.set_label(
    "Diviner tbol_sam source value"
)


plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(f"Saved: {OUTPUT_PATH}")