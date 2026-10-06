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
    "cross_modal_shift_validation.txt"
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
# Validation
# ============================================================

expected_shape = (256, 256)

for name, arr in {
    "WAC": wac,
    "WAC coverage": wac_coverage,
    "LOLA DEM": dem,
    "Slope": slope,
    "Diviner": thermal,
}.items():

    if arr.shape != expected_shape:
        raise ValueError(
            f"{name} has shape {arr.shape}; "
            f"expected {expected_shape}."
        )


# ============================================================
# Quantization
# ============================================================

def quantize(values, bins=64):

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

    q = np.floor(
        scaled * bins
    ).astype(np.int32)

    return np.clip(
        q,
        0,
        bins - 1
    )


# ============================================================
# Mutual information
# ============================================================

def mutual_information(x, y, bins=64):

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

            denominator = px[i] * py[j]

            if denominator <= 0:
                continue

            mi += (
                pxy
                * np.log(
                    pxy / denominator
                )
            )

    return mi


def entropy(values, bins=64):

    q = quantize(values, bins)

    counts = np.bincount(
        q,
        minlength=bins
    ).astype(np.float64)

    p = counts / counts.sum()

    p = p[p > 0]

    return -np.sum(
        p * np.log(p)
    )


def normalized_mi(x, y, bins=64):

    mi = mutual_information(
        x,
        y,
        bins
    )

    hx = entropy(x, bins)
    hy = entropy(y, bins)

    denominator = np.sqrt(
        hx * hy
    )

    if denominator == 0:
        return 0.0

    return mi / denominator


# ============================================================
# Shift helper
# ============================================================

def shifted_overlap(
    reference,
    moving,
    dy,
    dx
):
    """
    Shift 'moving' relative to 'reference'.

    Positive dx:
        moving shifted right.

    Positive dy:
        moving shifted down.

    Only the overlapping region is returned.
    """

    h, w = reference.shape

    ref_y0 = max(0, dy)
    ref_y1 = min(h, h + dy)

    ref_x0 = max(0, dx)
    ref_x1 = min(w, w + dx)

    mov_y0 = max(0, -dy)
    mov_y1 = min(h, h - dy)

    mov_x0 = max(0, -dx)
    mov_x1 = min(w, w - dx)

    ref_part = reference[
        ref_y0:ref_y1,
        ref_x0:ref_x1
    ]

    moving_part = moving[
        mov_y0:mov_y1,
        mov_x0:mov_x1
    ]

    return ref_part, moving_part


# ============================================================
# Shift range
# ============================================================

SHIFT_RANGE = range(-5, 6)

BINS = 64


# ============================================================
# Evaluation mask
# ============================================================

# Use cells where WAC has complete source coverage.
#
# The mask is additionally applied after shifting so that
# only valid overlapping pixels are evaluated.

wac_valid = (
    wac_coverage >= 0.999
)


# ============================================================
# Modality pairs
# ============================================================

pairs = {

    "WAC vs LOLA DEM": (
        wac,
        dem
    ),

    "WAC vs LOLA slope": (
        wac,
        slope
    ),

    "WAC vs Diviner": (
        wac,
        thermal
    ),

    "LOLA DEM vs Diviner": (
        dem,
        thermal
    ),
}


# ============================================================
# Run shift analysis
# ============================================================

all_results = {}


print("========================================")
print("CROSS-MODAL SHIFT VALIDATION")
print("========================================")

print(
    "Shift range:",
    "-5 to +5 pixels"
)

print(
    "Histogram bins:",
    BINS
)

print(
    "WAC full-coverage mask used:",
    np.sum(wac_valid),
    "/",
    wac_valid.size
)


for pair_name, (reference, moving) in pairs.items():

    print(
        f"\n{pair_name}"
    )

    pair_results = []

    for dy in SHIFT_RANGE:

        for dx in SHIFT_RANGE:

            ref_part, mov_part = shifted_overlap(
                reference,
                moving,
                dy,
                dx
            )

            # Corresponding WAC mask must be shifted
            # in exactly the same way when WAC is
            # involved as the reference.
            if pair_name.startswith("WAC"):

                mask_ref, _ = shifted_overlap(
                    wac_valid,
                    wac_valid,
                    dy,
                    dx
                )

                mask = mask_ref

            else:

                # For DEM vs Diviner there is no WAC
                # coverage limitation.
                mask = np.ones(
                    ref_part.shape,
                    dtype=bool
                )

            ref_values = ref_part[mask]
            mov_values = mov_part[mask]

            if len(ref_values) < 100:

                continue

            nmi = normalized_mi(
                ref_values,
                mov_values,
                bins=BINS
            )

            pair_results.append(
                (
                    dy,
                    dx,
                    nmi,
                    len(ref_values)
                )
            )

    all_results[pair_name] = pair_results

    # Find maximum NMI.
    best = max(
        pair_results,
        key=lambda x: x[2]
    )

    zero_shift = [
        r for r in pair_results
        if r[0] == 0 and r[1] == 0
    ][0]

    print(
        "Zero-shift NMI:",
        f"{zero_shift[2]:.6f}"
    )

    print(
        "Zero-shift pixels:",
        zero_shift[3]
    )

    print(
        "Maximum NMI:",
        f"{best[2]:.6f}"
    )

    print(
        "Maximum at:",
        f"dy={best[0]}, dx={best[1]}"
    )


# ============================================================
# Save
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
        "Cross-Modal Shift Validation\n"
    )

    f.write(
        "============================\n\n"
    )

    f.write(
        "Purpose:\n"
        "Evaluate normalized mutual information "
        "around the zero-shift common-grid alignment.\n\n"
    )

    f.write(
        "Shift range: -5 to +5 pixels\n"
    )

    f.write(
        f"Histogram bins: {BINS}\n\n"
    )

    for pair_name, results in all_results.items():

        zero = [
            r for r in results
            if r[0] == 0 and r[1] == 0
        ][0]

        best = max(
            results,
            key=lambda x: x[2]
        )

        f.write(
            f"{pair_name}\n"
        )

        f.write(
            f"Zero shift: "
            f"dy={zero[0]}, "
            f"dx={zero[1]}, "
            f"NMI={zero[2]:.6f}, "
            f"N={zero[3]}\n"
        )

        f.write(
            f"Maximum: "
            f"dy={best[0]}, "
            f"dx={best[1]}, "
            f"NMI={best[2]:.6f}, "
            f"N={best[3]}\n\n"
        )

        f.write(
            "All shifts:\n"
        )

        for dy, dx, nmi, n in sorted(
            results
        ):

            f.write(
                f"dy={dy:+d}, "
                f"dx={dx:+d}, "
                f"NMI={nmi:.6f}, "
                f"N={n}\n"
            )

        f.write("\n")


print("\n========================================")
print("SHIFT VALIDATION COMPLETE")
print("========================================")

print("Saved:")
print(OUTPUT)