from pathlib import Path

import numpy as np


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    r"D:\S7\Project Phase 1\Multimodal_Lunar_Benchmark"
)

WAC_PATH = (
    ROOT / "data/intermediate/optical/"
    "wac_common_grid.npy"
)

WAC_COVERAGE_PATH = (
    ROOT / "data/intermediate/optical/"
    "wac_common_coverage.npy"
)

DEM_PATH = (
    ROOT / "data/intermediate/dem/"
    "lola_dem_common_grid.npy"
)

SLOPE_PATH = (
    ROOT / "data/intermediate/dem/"
    "lola_common_slope_deg.npy"
)

THERMAL_PATH = (
    ROOT / "data/intermediate/thermal/"
    "diviner_tbol_sam_common_grid.npy"
)

OUTPUT = (
    ROOT / "outputs/validation/"
    "cross_modal_edge_validation.txt"
)


# ============================================================
# Load
# ============================================================

wac = np.load(WAC_PATH).astype(np.float64)

wac_coverage = np.load(
    WAC_COVERAGE_PATH
)

dem = np.load(
    DEM_PATH
).astype(np.float64)

slope = np.load(
    SLOPE_PATH
).astype(np.float64)

thermal = np.load(
    THERMAL_PATH
).astype(np.float64
)


# ============================================================
# Validate
# ============================================================

expected_shape = (256, 256)

arrays = {
    "WAC": wac,
    "LOLA DEM": dem,
    "LOLA slope": slope,
    "Diviner": thermal,
}

for name, array in arrays.items():

    if array.shape != expected_shape:
        raise ValueError(
            f"{name} has shape {array.shape}; "
            f"expected {expected_shape}."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains non-finite values."
        )


# ============================================================
# Gradient magnitude
# ============================================================

def gradient_magnitude(image):

    gy, gx = np.gradient(image)

    magnitude = np.sqrt(
        gx ** 2 +
        gy ** 2
    )

    return magnitude


gradient_maps = {

    "WAC":
        gradient_magnitude(wac),

    "LOLA DEM":
        gradient_magnitude(dem),

    "LOLA slope":
        gradient_magnitude(slope),

    "Diviner":
        gradient_magnitude(thermal),
}


# ============================================================
# Normalize gradient maps
# ============================================================

def normalize_gradient(values):

    low = np.percentile(
        values,
        1
    )

    high = np.percentile(
        values,
        99
    )

    if high <= low:
        return np.zeros_like(
            values
        )

    normalized = (
        values - low
    ) / (
        high - low
    )

    return np.clip(
        normalized,
        0,
        1
    )


normalized_gradients = {}

for name, gradient in gradient_maps.items():

    normalized_gradients[name] = (
        normalize_gradient(gradient)
    )


# ============================================================
# Edge threshold
# ============================================================

EDGE_PERCENTILE = 85.0


def make_edge_mask(
    gradient,
    percentile=EDGE_PERCENTILE
):

    threshold = np.percentile(
        gradient,
        percentile
    )

    return gradient >= threshold


edge_masks = {

    name:
        make_edge_mask(gradient)
    for name, gradient
    in normalized_gradients.items()
}


# ============================================================
# Metrics
# ============================================================

def edge_metrics(
    reference_edges,
    moving_edges,
    valid_mask
):

    ref = (
        reference_edges
        & valid_mask
    )

    mov = (
        moving_edges
        & valid_mask
    )

    intersection = np.sum(
        ref & mov
    )

    ref_count = np.sum(ref)
    mov_count = np.sum(mov)

    if ref_count == 0:
        precision = 0.0
        recall = 0.0
    else:
        precision = (
            intersection /
            ref_count
        )

    if mov_count == 0:
        moving_precision = 0.0
    else:
        moving_precision = (
            intersection /
            mov_count
        )

    denominator = (
        ref_count +
        mov_count
    )

    if denominator == 0:
        dice = 0.0
    else:
        dice = (
            2.0 * intersection /
            denominator
        )

    return (
        dice,
        precision,
        moving_precision,
        intersection,
        ref_count,
        mov_count
    )


# ============================================================
# Evaluation mask
# ============================================================

# WAC has partial source coverage.
# Use only fully covered WAC cells whenever WAC participates.

wac_valid = (
    wac_coverage >= 0.999
)

full_valid = np.ones(
    expected_shape,
    dtype=bool
)


# ============================================================
# Modality pairs
# ============================================================

pairs = {

    "WAC vs LOLA DEM": (
        "WAC",
        "LOLA DEM",
        wac_valid
    ),

    "WAC vs LOLA slope": (
        "WAC",
        "LOLA slope",
        wac_valid
    ),

    "WAC vs Diviner": (
        "WAC",
        "Diviner",
        wac_valid
    ),

    "LOLA DEM vs Diviner": (
        "LOLA DEM",
        "Diviner",
        full_valid
    ),
}


# ============================================================
# Calculate metrics
# ============================================================

results = []


print("========================================")
print("CROSS-MODAL EDGE VALIDATION")
print("========================================")

print(
    "Edge threshold percentile:",
    EDGE_PERCENTILE
)

print(
    "WAC evaluation pixels:",
    np.sum(wac_valid),
    "/",
    wac_valid.size
)


for pair_name, (
    reference_name,
    moving_name,
    valid_mask
) in pairs.items():

    reference_edges = (
        edge_masks[reference_name]
    )

    moving_edges = (
        edge_masks[moving_name]
    )

    metrics = edge_metrics(
        reference_edges,
        moving_edges,
        valid_mask
    )

    (
        dice,
        precision,
        recall,
        intersection,
        reference_count,
        moving_count
    ) = metrics

    results.append(
        (
            pair_name,
            dice,
            precision,
            recall,
            intersection,
            reference_count,
            moving_count
        )
    )

    print(
        f"\n{pair_name}"
    )

    print(
        "Edge Dice:",
        f"{dice:.6f}"
    )

    print(
        "Reference-edge overlap:",
        f"{precision:.6f}"
    )

    print(
        "Moving-edge overlap:",
        f"{recall:.6f}"
    )

    print(
        "Common edge pixels:",
        intersection
    )

    print(
        "Reference edges:",
        reference_count
    )

    print(
        "Moving edges:",
        moving_count
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
        "Cross-Modal Edge Validation\n"
    )

    f.write(
        "===========================\n\n"
    )

    f.write(
        "Method:\n"
    )

    f.write(
        "Gradient magnitude computed on each "
        "common-grid modality.\n"
    )

    f.write(
        f"Strong edges defined using the "
        f"{EDGE_PERCENTILE:.1f}th percentile "
        "of normalized gradient magnitude.\n"
    )

    f.write(
        "WAC comparisons use cells with "
        "WAC coverage >= 0.999.\n\n"
    )

    f.write(
        "Results:\n"
    )

    for (
        pair_name,
        dice,
        precision,
        recall,
        intersection,
        reference_count,
        moving_count
    ) in results:

        f.write(
            f"{pair_name}\n"
        )

        f.write(
            f"Edge Dice: {dice:.6f}\n"
        )

        f.write(
            f"Reference-edge overlap: "
            f"{precision:.6f}\n"
        )

        f.write(
            f"Moving-edge overlap: "
            f"{recall:.6f}\n"
        )

        f.write(
            f"Common edge pixels: "
            f"{intersection}\n"
        )

        f.write(
            f"Reference edges: "
            f"{reference_count}\n"
        )

        f.write(
            f"Moving edges: "
            f"{moving_count}\n\n"
        )


print("\n========================================")
print("EDGE VALIDATION COMPLETE")
print("========================================")

print("Saved:")
print(OUTPUT)