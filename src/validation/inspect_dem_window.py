from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEM_PATH = (
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
    / "lola_ldem256_pilot.png"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

dem = np.load(DEM_PATH)

plt.figure(figsize=(8, 8))
plt.imshow(dem, origin="upper")
plt.title("LOLA LDEM256 Pilot DEM\n44–46°N, 27–29°E")
plt.xlabel("Pixel column")
plt.ylabel("Pixel row")
plt.colorbar(label="Stored height relative to reference radius (km)")
plt.tight_layout()

plt.savefig(OUTPUT_PATH, dpi=200)
plt.close()

print(f"DEM shape: {dem.shape}")
print(f"Saved: {OUTPUT_PATH}")