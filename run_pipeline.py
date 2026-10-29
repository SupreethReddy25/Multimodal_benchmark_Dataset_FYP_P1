#!/usr/bin/env python
"""
run_pipeline.py  --  Multimodal Lunar Benchmark Pipeline Orchestrator
======================================================================

Runs the full 9-step pipeline for one or more regions, in sequence.
Each step is idempotent: if its output already exists it is skipped.

Usage
-----
  # Single region
  python run_pipeline.py --region posidonius

  # Multiple regions
  python run_pipeline.py --region posidonius --region montes_apenninus

  # All regions in configs/regions.yaml
  python run_pipeline.py --all

  # Force re-run all steps (ignore cached outputs)
  python run_pipeline.py --all --force

  # List all available regions
  python run_pipeline.py --list

Pipeline Steps (in order)
--------------------------
  1  extract_thermal   HTTP-stream Diviner 256x256 window
  2  extract_dem       HTTP-stream LOLA 512x512 window  (~1 MB, no 4.2 GB dl)
  3  extract_optical   Download WAC GeoTIFF once, extract AOI window
  4  create_grid       Define common cartographic grid (256x256, 128 PPD)
  5  harmonize         Resample/aggregate all three modalities to common grid
  6  derive_terrain    Slope (deg) + hillshade from common-grid DEM
  7  create_tiles      Partition into 16 synchronized 64x64 tiles
  8  validate          Structural + MI + edge-Dice + QC figures
  9  manifest          Update global_manifest.json
"""

import argparse
import sys
import time
from pathlib import Path
import traceback

# Ensure the project root is on sys.path so `src` is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline.config import list_regions, load_region
import src.pipeline.extract_thermal as step_thermal
import src.pipeline.extract_dem     as step_dem
import src.pipeline.extract_optical as step_optical
import src.pipeline.create_grid     as step_grid
import src.pipeline.harmonize       as step_harmonize
import src.pipeline.derive_terrain  as step_terrain
import src.pipeline.create_tiles    as step_tiles
import src.pipeline.validate        as step_validate
import src.pipeline.manifest        as step_manifest


STEPS = [
    ("extract_thermal",   step_thermal.run,   "HTTP-stream Diviner window"),
    ("extract_dem",       step_dem.run,        "HTTP-stream LOLA window"),
    ("extract_optical",   step_optical.run,    "Extract WAC optical AOI"),
    ("create_grid",       step_grid.run,       "Define common grid"),
    ("harmonize",         step_harmonize.run,  "Harmonize all modalities"),
    ("derive_terrain",    step_terrain.run,    "Derive slope + hillshade"),
    ("create_tiles",      step_tiles.run,      "Create 16 synchronized tiles"),
    ("validate",          step_validate.run,   "Validate + QC figures"),
]

DIVIDER = "=" * 72


def _print_header(region_id: str, region: dict) -> None:
    print(DIVIDER)
    print(f"  REGION : {region_id}")
    print(f"  NAME   : {region['name']}")
    print(f"  EXTENT : {region['lat_min']}-{region['lat_max']}N  "
          f"{region['lon_min']}-{region['lon_max']}E")
    print(DIVIDER)


def _run_region(region_id: str, force: bool = False) -> bool:
    """Run all pipeline steps for a single region.

    Returns True on success, False on failure.
    """
    region = load_region(region_id)
    _print_header(region_id, region)

    # If --force, remove existing intermediate and benchmark outputs
    if force:
        import shutil
        inter = region["intermediate_dir"]
        bench = region["benchmark_dir"]
        for d in [inter, bench]:
            if d.exists():
                print(f"  [force] Removing {d}")
                shutil.rmtree(d)

    t_region_start = time.perf_counter()

    for step_num, (step_name, step_fn, step_desc) in enumerate(STEPS, start=1):
        print(f"\nStep {step_num}/{len(STEPS)}: {step_name}  —  {step_desc}")
        t_step_start = time.perf_counter()
        try:
            step_fn(region_id)
        except Exception as exc:
            print(f"\n  [ERROR] Step '{step_name}' FAILED for region '{region_id}':")
            traceback.print_exc()
            return False
        elapsed = time.perf_counter() - t_step_start
        print(f"  done in {elapsed:.1f}s")

    t_total = time.perf_counter() - t_region_start
    print(f"\n{DIVIDER}")
    print(f"  Region '{region_id}' completed in {t_total:.1f}s")
    print(DIVIDER)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Multimodal Lunar Benchmark — full pipeline runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--region", action="append", dest="regions", metavar="REGION_ID",
        help="Region ID to process (may be repeated).",
    )
    grp.add_argument(
        "--all", action="store_true",
        help="Process all regions defined in configs/regions.yaml.",
    )
    grp.add_argument(
        "--list", action="store_true",
        help="List all available regions and exit.",
    )
    ap.add_argument(
        "--force", action="store_true",
        help="Delete cached intermediate outputs and re-run all steps.",
    )
    ap.add_argument(
        "--skip-lacus-mortis", action="store_true",
        help="When using --all, skip lacus_mortis (pilot already processed).",
    )
    args = ap.parse_args()

    if args.list:
        all_regions = list_regions()
        print("Available regions:")
        for r in all_regions:
            region = load_region(r)
            print(f"  {r:<22}  {region['name']}")
        return 0

    if args.all:
        target_regions = list_regions()
        if args.skip_lacus_mortis and "lacus_mortis" in target_regions:
            target_regions.remove("lacus_mortis")
            print("Skipping lacus_mortis (pilot, already processed).\n")
    else:
        target_regions = args.regions

    print(f"\nProcessing {len(target_regions)} region(s): {target_regions}\n")

    successes = []
    failures = []

    for region_id in target_regions:
        ok = _run_region(region_id, force=args.force)
        (successes if ok else failures).append(region_id)

    # Update global manifest for all successful regions
    if successes:
        print(f"\nUpdating global manifest for: {successes}")
        step_manifest.run(successes)

    print(f"\n{DIVIDER}")
    print(f"  SUMMARY")
    print(f"  Success : {successes}")
    if failures:
        print(f"  Failure : {failures}")
    print(DIVIDER)

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
