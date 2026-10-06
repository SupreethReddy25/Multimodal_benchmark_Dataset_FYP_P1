from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

WAC_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "optical"
    / "wac_common_grid.npy"
)

COVERAGE_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "optical"
    / "wac_common_coverage.npy"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "validation"
    / "wac_common_grid_qc.png"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

wac = np.load(WAC_PATH)
coverage = np.load(COVERAGE_PATH)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 6)
)

# ------------------------------------------------------------
# Harmonized WAC
# ------------------------------------------------------------

im1 = axes[0].imshow(
    wac,
    cmap="gray",
    origin="upper",
)

axes[0].set_title(
    "WAC on Common 236.9 m Grid"
)

axes[0].set_xlabel("Common-grid column")
axes[0].set_ylabel("Common-grid row")

fig.colorbar(
    im1,
    ax=axes[0],
    label="Aggregated WAC intensity",
)


# ------------------------------------------------------------
# Coverage
# ------------------------------------------------------------

im2 = axes[1].imshow(
    coverage,
    origin="upper",
    vmin=0,
    vmax=1,
)

axes[1].set_title(
    "WAC Source Coverage per Common Cell"
)

axes[1].set_xlabel("Common-grid column")
axes[1].set_ylabel("Common-grid row")

fig.colorbar(
    im2,
    ax=axes[1],
    label="Source coverage fraction",
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(f"WAC shape: {wac.shape}")
print(f"Coverage shape: {coverage.shape}")
print(f"Saved: {OUTPUT_PATH}")