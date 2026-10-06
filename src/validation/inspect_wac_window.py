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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "wac_pilot.png"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

wac = np.load(WAC_PATH)

plt.figure(figsize=(8, 8))

plt.imshow(
    wac,
    cmap="gray",
    origin="upper"
)

plt.title("LROC WAC Pilot Optical Image\n44–46°N, 27–29°E")
plt.xlabel("Pixel column")
plt.ylabel("Pixel row")

plt.colorbar(label="WAC intensity")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200
)

plt.close()

print(f"Shape: {wac.shape}")
print(f"Saved: {OUTPUT_PATH}")