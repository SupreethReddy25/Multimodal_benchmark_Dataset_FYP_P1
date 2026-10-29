"""Step 9 — Update the global benchmark manifest.

Reads each processed region's tile_index.json and validation_report.json
and writes a consolidated global_manifest.json to data/benchmark/.

Usage
-----
    python -m src.pipeline.manifest [--region posidonius]
    python -m src.pipeline.manifest --all
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import load_region, list_regions

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "data" / "benchmark" / "global_manifest.json"


def _load_existing() -> dict:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {
        "dataset": "Multimodal Lunar Benchmark Dataset",
        "version": "1.0.0",
        "description": (
            "A multimodal, spatially-synchronized benchmark dataset of the "
            "lunar surface comprising optical, elevation, thermal, slope, and "
            "hillshade layers partitioned into 64x64 px tiles."
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "regions": {},
    }


def _update_region(manifest: dict, region_id: str) -> None:
    region = load_region(region_id)
    bench = region["benchmark_dir"]
    out_dir = region["output_dir"] / "validation"

    tile_index_path = bench / "metadata" / "tile_index.json"
    validation_path = out_dir / "validation_report.json"

    if not tile_index_path.exists():
        print(f"  [manifest] {region_id}: no tile_index.json — skip.")
        return

    with open(tile_index_path, "r", encoding="utf-8") as fh:
        tile_index = json.load(fh)

    validation = {}
    if validation_path.exists():
        with open(validation_path, "r", encoding="utf-8") as fh:
            validation = json.load(fh)

    # Compute aggregate statistics across all tiles
    tiles_dir = bench / "tiles"
    tile_dirs = sorted(
        p for p in tiles_dir.iterdir()
        if p.is_dir() and p.name.startswith("tile_")
    )

    coverage_stats = {}
    for modality in ["optical", "dem", "thermal"]:
        cov_files = {
            "optical": "optical_coverage.npy",
            "dem": "dem_coverage.npy",
            "thermal": "thermal_coverage.npy",
        }
        vals = []
        for td in tile_dirs:
            fp = td / cov_files[modality]
            if fp.exists():
                cov = float(np.load(fp).mean())
                vals.append(cov)
        if vals:
            coverage_stats[modality] = {
                "mean": float(np.mean(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    manifest["regions"][region_id] = {
        "id": region_id,
        "name": region["name"],
        "description": region.get("description", ""),
        "extent": {
            "lat_min": region["lat_min"],
            "lat_max": region["lat_max"],
            "lon_min": region["lon_min"],
            "lon_max": region["lon_max"],
        },
        "tiling": tile_index.get("tiling", {}),
        "common_grid": tile_index.get("common_grid", {}),
        "coverage_stats": coverage_stats,
        "validation": {
            "structural_pass": validation.get(
                "structural_validation", {}
            ).get("pass", None),
            "mutual_information_nats": validation.get("mutual_information_nats", {}),
            "edge_dice_scores": validation.get("edge_dice_scores", {}),
        },
        "benchmark_dir": str(bench),
    }

    manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
    print(f"  [manifest] Updated entry for '{region_id}'.")


def run(region_ids: list) -> None:
    manifest = _load_existing()

    for region_id in region_ids:
        _update_region(manifest, region_id)

    # Summary stats
    n = len(manifest["regions"])
    manifest["summary"] = {
        "total_regions": n,
        "total_tiles": n * 16,
        "region_ids": sorted(manifest["regions"].keys()),
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"  [manifest] global_manifest.json updated — {n} region(s).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--region", help="Update a single region.")
    grp.add_argument("--all", action="store_true",
                     help="Update all regions in regions.yaml.")
    args = ap.parse_args()

    if args.all:
        run(list_regions())
    else:
        run([args.region])
