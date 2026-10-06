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

HILLSHADE_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "dem"
    / "lola_hillshade.npy"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "wac_lola_hillshade_overlay.png"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

wac = np.load(WAC_PATH)
hillshade = np.load(HILLSHADE_PATH)

fig, ax = plt.subplots(figsize=(10, 8))

ax.imshow(
    wac,
    cmap="gray",
    extent=[27, 29, 44, 46],
    origin="upper"
)

ax.imshow(
    hillshade,
    cmap="gray",
    extent=[27, 29, 44, 46],
    origin="upper",
    alpha=0.35
)

ax.set_title(
    "WAC Optical + LOLA Hillshade Overlay\n"
    "Lacus Mortis–Bürg Pilot | 44–46°N, 27–29°E"
)

ax.set_xlabel("Longitude (°E)")
ax.set_ylabel("Latitude (°N)")

ax.set_xlim(27, 29)
ax.set_ylim(44, 46)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"WAC shape: {wac.shape}")
print(f"Hillshade shape: {hillshade.shape}")
print(f"Saved: {OUTPUT_PATH}")