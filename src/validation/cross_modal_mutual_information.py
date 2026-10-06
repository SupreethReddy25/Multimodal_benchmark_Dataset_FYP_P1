from pathlib import Path

import numpy as np


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    r"D:\S7\Project Phase 1\Multimodal_Lunar_Benchmark"
)

WAC_PATH = (
    ROOT
    / "data/intermediate/optical/"
    "wac_common_grid.npy"
)

WAC_COVERAGE_PATH = (
    ROOT
    / "data/intermediate/optical/"
    "wac_common_coverage.npy"
)

DEM_PATH = (
    ROOT
    / "data/intermediate/dem/"
    "lola_dem_common_grid.npy"
)

SLOPE_PATH = (
    ROOT
    / "data/intermediate/dem/"
    "lola_common_slope_deg.npy"
)

THERMAL_PATH = (
    ROOT
    / "data/intermediate/thermal/"
    "diviner_tbol_sam_common_grid.npy"
)

OUTPUT = (
    ROOT
    / "outputs/validation/"
    "cross_modal_mutual_information.txt"
)


# ============================================================
# Load
# ============================================================

wac = np.load(WAC_PATH).astype(np.float64)
wac_coverage = np.load(WAC_COVERAGE_PATH)

dem = np.load(DEM_PATH).astype(np.float64)

slope = np.load(SLOPE_PATH).astype(np.float64)

thermal = np.load(THERMAL_PATH).astype(np.float64)


# ============================================================
# Validate
# ============================================================

arrays = {
    "WAC": wac,
    "LOLA DEM": dem,
    "Slope": slope,
    "Diviner": thermal,
}

for name, array in arrays.items():

    if array.shape != (256, 256):
        raise ValueError(
            f"{name} has shape {array.shape}, "
            "expected (256, 256)."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains non-finite values."
        )


# ============================================================
# Quantization
# ============================================================

def quantize(values, bins=64):
    """
    Convert continuous values into equally spaced
    histogram bins.
    """

    vmin = np.min(values)
    vmax = np.max(values)

    if np.isclose(vmin, vmax):
        return np.zeros(
            values.shape,
            dtype=np.int32
        )

    scaled = (
        (values - vmin)
        / (vmax - vmin)
    )

    result = np.floor(
        scaled * bins
    ).astype(np.int32)

    result = np.clip(
        result,
        0,
        bins - 1
    )

    return result


# ============================================================
# Mutual information
# ============================================================

def mutual_information(
    x,
    y,
    bins=64
):
    """
    Histogram-based mutual information.
    """

    xq = quantize(x, bins)
    yq = quantize(y, bins)

    joint = np.zeros(
        (bins, bins),
        dtype=np.float64
    )

    np.add.at(
        joint,
        (xq, yq),
        1
    )

    joint /= joint.sum()

    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0

    for i in range(bins):
        for j in range(bins):

            pxy = joint[i, j]

            if pxy <= 0:
                continue

            if px[i] <= 0 or py[j] <= 0:
                continue

            mi += (
                pxy
                * np.log(
                    pxy
                    / (px[i] * py[j])
                )
            )

    return mi


def entropy(values, bins=64):

    q = quantize(values, bins)

    counts = np.bincount(
        q,
        minlength=bins
    ).astype(np.float64)

    probabilities = (
        counts / counts.sum()
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    return -np.sum(
        probabilities
        * np.log(probabilities)
    )


def normalized_mi(
    x,
    y,
    bins=64
):

    mi = mutual_information(
        x,
        y,
        bins
    )

    hx = entropy(
        x,
        bins
    )

    hy = entropy(
        y,
        bins
    )

    denominator = np.sqrt(
        hx * hy
    )

    if denominator == 0:
        return 0.0

    return mi / denominator


# ============================================================
# Common valid mask
# ============================================================

# We require complete WAC source coverage for the
# cross-modal comparison.

mask = (
    (wac_coverage >= 0.999)
    &
    np.isfinite(wac)
    &
    np.isfinite(dem)
    &
    np.isfinite(slope)
    &
    np.isfinite(thermal)
)


n_valid = np.sum(mask)

print("========================================")
print("CROSS-MODAL MUTUAL INFORMATION")
print("========================================")

print(
    "Common evaluation pixels:",
    n_valid,
    "/",
    mask.size
)

print(
    "Fraction:",
    n_valid / mask.size
)


# ============================================================
# Extract paired samples
# ============================================================

wac_v = wac[mask]
dem_v = dem[mask]
slope_v = slope[mask]
thermal_v = thermal[mask]


# ============================================================
# Calculate MI
# ============================================================

pairs = {

    "WAC vs LOLA DEM":
        (wac_v, dem_v),

    "WAC vs LOLA slope":
        (wac_v, slope_v),

    "WAC vs Diviner":
        (wac_v, thermal_v),

    "LOLA DEM vs Diviner":
        (dem_v, thermal_v),
}


results = []


for name, (x, y) in pairs.items():

    mi = mutual_information(
        x,
        y,
        bins=64
    )

    nmi = normalized_mi(
        x,
        y,
        bins=64
    )

    results.append(
        (
            name,
            mi,
            nmi
        )
    )


# ============================================================
# Print results
# ============================================================

print("\nResults")
print("----------------------------------------")

for name, mi, nmi in results:

    print(
        f"{name:25s} "
        f"MI={mi:.6f} "
        f"NMI={nmi:.6f}"
    )


# ============================================================
# Save results
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT,
    "w"
) as f:

    f.write(
        "Cross-Modal Mutual Information Validation\n"
    )

    f.write(
        "========================================\n\n"
    )

    f.write(
        f"Total common-grid pixels: "
        f"{mask.size}\n"
    )

    f.write(
        f"Evaluation pixels: "
        f"{n_valid}\n"
    )

    f.write(
        f"Evaluation fraction: "
        f"{n_valid / mask.size:.6f}\n\n"
    )

    f.write(
        "Method:\n"
    )

    f.write(
        "Histogram-based mutual information "
        "with 64 bins.\n"
    )

    f.write(
        "Evaluation uses common-grid cells with "
        "WAC source coverage >= 0.999.\n\n"
    )

    f.write(
        "Results:\n"
    )

    for name, mi, nmi in results:

        f.write(
            f"{name}: "
            f"MI={mi:.6f}, "
            f"NMI={nmi:.6f}\n"
        )


print("\nSaved:")
print(OUTPUT)