"""Step 8 — Validate benchmark tiles and generate QC figures.

Three validation checks:
  1. Structural integrity  — all files present, shapes correct, no NaN
  2. Cross-modal mutual information  — histogram MI across modality pairs
  3. Cross-modal edge alignment      — gradient magnitude + Dice similarity

QC figures (composite 2x5 panel, MI bar chart, edge overlay) saved to
outputs/{region_id}/validation/.

Usage
-----
    python -m src.pipeline.validate --region posidonius
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .config import load_region

# Required files in each tile directory
REQUIRED_TILE_FILES = [
    "optical.npy", "optical_coverage.npy",
    "dem.npy", "dem_valid_mask.npy", "dem_coverage.npy",
    "slope.npy", "hillshade.npy",
    "thermal.npy", "thermal_valid_mask.npy", "thermal_coverage.npy",
    "metadata.json",
]

EXPECTED_SHAPE = (64, 64)
EXPECTED_TILES = 16


# ===========================================================================
# Structural validation
# ===========================================================================

def _validate_structure(region: dict) -> dict:
    bench = region["benchmark_dir"]
    tiles_dir = bench / "tiles"

    results = {"pass": True, "errors": [], "tile_summaries": []}

    tile_dirs = sorted(
        p for p in tiles_dir.iterdir()
        if p.is_dir() and p.name.startswith("tile_")
    )

    if len(tile_dirs) != EXPECTED_TILES:
        results["errors"].append(
            f"Expected {EXPECTED_TILES} tile dirs, found {len(tile_dirs)}."
        )
        results["pass"] = False

    for td in tile_dirs:
        tile_result = {"tile_id": td.name, "errors": []}

        # Check all required files exist
        for fname in REQUIRED_TILE_FILES:
            fp = td / fname
            if not fp.exists():
                tile_result["errors"].append(f"MISSING: {fname}")

        # Check array shapes and no NaN in primary layers
        for layer in ["optical", "dem", "slope", "hillshade", "thermal"]:
            fp = td / f"{layer}.npy"
            if fp.exists():
                arr = np.load(fp)
                if arr.shape != EXPECTED_SHAPE:
                    tile_result["errors"].append(
                        f"{layer}: shape {arr.shape} != {EXPECTED_SHAPE}"
                    )
                if not np.all(np.isfinite(arr)):
                    nan_count = np.count_nonzero(~np.isfinite(arr))
                    tile_result["errors"].append(
                        f"{layer}: {nan_count} non-finite values"
                    )

        # Check mask consistency: gap-filled pixels are finite in the data
        # layer but flagged False in the mask, so require mask => finite data
        # and a boolean mask of the right shape.
        for mask_name, data_name in [
            ("dem_valid_mask.npy", "dem.npy"),
            ("thermal_valid_mask.npy", "thermal.npy"),
        ]:
            mask_fp = td / mask_name
            data_fp = td / data_name
            if mask_fp.exists() and data_fp.exists():
                mask = np.load(mask_fp)
                data = np.load(data_fp)
                if mask.dtype != np.bool_ or mask.shape != data.shape:
                    tile_result["errors"].append(
                        f"{mask_name}: dtype/shape {mask.dtype}/{mask.shape} invalid"
                    )
                elif not np.all(np.isfinite(data[mask])):
                    tile_result["errors"].append(
                        f"{mask_name}: marks non-finite {data_name} pixels valid"
                    )

        if tile_result["errors"]:
            results["pass"] = False

        results["tile_summaries"].append(tile_result)

    return results


# ===========================================================================
# Mutual information
# ===========================================================================

def _mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Compute histogram-based MI in nats."""
    a_norm = (a - a.min()) / (np.ptp(a) + 1e-12)
    b_norm = (b - b.min()) / (np.ptp(b) + 1e-12)

    joint, _, _ = np.histogram2d(a_norm.ravel(), b_norm.ravel(), bins=bins)
    joint = joint / joint.sum()

    p_a = joint.sum(axis=1, keepdims=True)
    p_b = joint.sum(axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(joint > 0, np.log(joint / (p_a * p_b + 1e-300)), 0.0)

    return float(np.sum(joint * log_term))


def _cross_modal_mi(region: dict) -> dict:
    inter = region["intermediate_dir"]

    modalities = {
        "optical": inter / "optical" / "wac_common_grid.npy",
        "dem": inter / "dem" / "lola_dem_common_grid.npy",
        "thermal": inter / "thermal" / "diviner_tbol_sam_common_grid.npy",
        "slope": inter / "dem" / "lola_common_slope_deg.npy",
        "hillshade": inter / "dem" / "lola_common_hillshade.npy",
    }

    arrays = {}
    for name, path in modalities.items():
        if path.exists():
            arr = np.load(path).astype(np.float64)
            valid = np.isfinite(arr)
            if valid.any():
                arrays[name] = np.where(valid, arr, np.nanmean(arr))

    pairs = [
        ("optical", "dem"),
        ("optical", "thermal"),
        ("dem", "thermal"),
        ("slope", "thermal"),
        ("hillshade", "thermal"),
    ]

    mi_results = {}
    for a, b in pairs:
        if a in arrays and b in arrays:
            mi_results[f"{a}_vs_{b}"] = _mutual_information(arrays[a], arrays[b])

    return mi_results


# ===========================================================================
# Edge alignment (Dice similarity on salient gradients)
# ===========================================================================

def _edge_dice(a: np.ndarray, b: np.ndarray, percentile: float = 90.0) -> float:
    """Dice similarity of salient-edge binary masks."""
    grad_a = np.sqrt(np.gradient(a)[0] ** 2 + np.gradient(a)[1] ** 2)
    grad_b = np.sqrt(np.gradient(b)[0] ** 2 + np.gradient(b)[1] ** 2)

    thr_a = np.percentile(grad_a, percentile)
    thr_b = np.percentile(grad_b, percentile)

    edge_a = grad_a > thr_a
    edge_b = grad_b > thr_b

    intersection = np.logical_and(edge_a, edge_b).sum()
    denom = edge_a.sum() + edge_b.sum()
    if denom == 0:
        return 0.0
    return float(2 * intersection / denom)


def _cross_modal_edges(region: dict) -> dict:
    inter = region["intermediate_dir"]

    optical = np.load(inter / "optical" / "wac_common_grid.npy").astype(np.float64)
    dem = np.load(inter / "dem" / "lola_dem_common_grid.npy").astype(np.float64)
    thermal = np.load(inter / "thermal" / "diviner_tbol_sam_common_grid.npy").astype(np.float64)

    # Replace NaN with interpolated mean for gradient computation
    for arr in [optical, dem, thermal]:
        if not np.all(np.isfinite(arr)):
            arr[~np.isfinite(arr)] = np.nanmean(arr)

    return {
        "optical_vs_dem_dice": _edge_dice(optical, dem),
        "optical_vs_thermal_dice": _edge_dice(optical, thermal),
        "dem_vs_thermal_dice": _edge_dice(dem, thermal),
    }


# ===========================================================================
# QC figures
# ===========================================================================

def _save_qc_figures(region: dict, mi: dict, edges: dict) -> None:
    inter = region["intermediate_dir"]
    out_dir = region["output_dir"] / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _load(sub, name):
        p = inter / sub / name
        return np.load(p) if p.exists() else None

    optical = _load("optical", "wac_common_grid.npy")
    dem = _load("dem", "lola_dem_common_grid.npy")
    slope = _load("dem", "lola_common_slope_deg.npy")
    hillshade = _load("dem", "lola_common_hillshade.npy")
    thermal = _load("thermal", "diviner_tbol_sam_common_grid.npy")

    # --- Composite 2x3 panel ---
    layers = [
        (optical, "Optical (WAC)", "gray"),
        (dem, "DEM (LOLA, km)", "terrain"),
        (slope, "Slope (deg)", "YlOrRd"),
        (hillshade, "Hillshade", "gray"),
        (thermal, "Thermal (Diviner, K)", "inferno"),
    ]
    layers = [(arr, title, cmap) for arr, title, cmap in layers if arr is not None]

    fig, axes = plt.subplots(1, len(layers), figsize=(4 * len(layers), 4))
    if len(layers) == 1:
        axes = [axes]

    for ax, (arr, title, cmap) in zip(axes, layers):
        im = ax.imshow(arr, cmap=cmap, interpolation="none")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Multimodal Common Grid — {region['name']}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "common_grid_composite.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- MI bar chart ---
    if mi:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        labels = list(mi.keys())
        values = list(mi.values())
        bars = ax.barh(labels, values, color="#4A90E2")
        ax.set_xlabel("Mutual Information (nats)")
        ax.set_title(f"Cross-modal MI — {region['name']}", fontweight="bold")
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=8)
        plt.tight_layout()
        fig.savefig(out_dir / "cross_modal_mi.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  [validate] QC figures saved to {out_dir}")


# ===========================================================================
# Public entry point
# ===========================================================================

def run(region_id: str) -> None:
    region = load_region(region_id)
    bench = region["benchmark_dir"]
    out_dir = region["output_dir"] / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [validate] Structural check ...")
    structural = _validate_structure(region)
    if structural["pass"]:
        print(f"  [validate] PASS — all {EXPECTED_TILES} tiles structurally valid.")
    else:
        print(f"  [validate] FAIL — errors found:")
        for e in structural["errors"]:
            print(f"             {e}")
        for ts in structural["tile_summaries"]:
            for err in ts["errors"]:
                print(f"             [{ts['tile_id']}] {err}")

    print(f"  [validate] Cross-modal mutual information ...")
    mi = _cross_modal_mi(region)
    for pair, val in mi.items():
        print(f"             MI({pair}) = {val:.6f} nats")

    print(f"  [validate] Cross-modal edge alignment ...")
    edges = _cross_modal_edges(region)
    for pair, val in edges.items():
        print(f"             Dice({pair}) = {val:.4f}")

    # Save full validation report
    report = {
        "region_id": region_id,
        "region_name": region["name"],
        "structural_validation": structural,
        "mutual_information_nats": mi,
        "edge_dice_scores": edges,
    }
    report_path = out_dir / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=bool)

    _save_qc_figures(region, mi, edges)

    print(f"  [validate] Report saved to {report_path}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    args = ap.parse_args()
    run(args.region)
