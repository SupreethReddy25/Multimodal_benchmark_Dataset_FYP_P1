from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
    / "diviner_tbol_sam_pilot.png"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

thermal = np.load(THERMAL_PATH)

finite = np.isfinite(thermal)

plt.figure(figsize=(8, 8))

im = plt.imshow(
    thermal,
    origin="upper"
)

plt.title(
    "Diviner Bolometric Temperature\n"
    "Slope-Adjusted Midnight | 44–46°N, 27–29°E"
)

plt.xlabel("Pixel column")
plt.ylabel("Pixel row")

plt.colorbar(
    im,
    label="Source value"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"Shape: {thermal.shape}")
print(f"Finite pixels: {finite.sum()} / {thermal.size}")
print(f"Saved: {OUTPUT_PATH}")