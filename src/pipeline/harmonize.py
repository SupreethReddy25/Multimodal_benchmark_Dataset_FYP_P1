"""Step 5 — Harmonize all three modalities to the common grid.

WAC Optical   : reprojected with area-weighted resampling (rasterio).
LOLA DEM      : exact 2x2 block aggregation (native 512x512 -> 256x256).
Diviner Therm.: direct native-grid mapping  (already 256x256 @ 128 PPD).

Produces:
  optical/  wac_common_grid.npy, wac_common_coverage.npy
  dem/      lola_dem_common_grid.npy, lola_dem_common_valid_mask.npy,
            lola_dem_common_coverage.npy
  thermal/  diviner_tbol_sam_common_grid.npy,
            diviner_tbol_sam_common_valid_mask.npy,
            diviner_tbol_sam_common_coverage.npy

Usage
-----
    python -m src.pipeline.harmonize --region posidonius
"""

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds

from .config import load_region
from .io_utils import fill_gaps, save_npy_atomic

MOON_RADIUS_M = 1_737_400.0


# ===========================================================================
# WAC optical harmonization
# ===========================================================================

def _harmonize_optical(region: dict) -> None:
    inter = region["intermediate_dir"]
    out_dir = inter / "optical"
    out_dir.mkdir(parents=True, exist_ok=True)

    wac_out = out_dir / "wac_common_grid.npy"
    cov_out = out_dir / "wac_common_coverage.npy"

    if wac_out.exists() and cov_out.exists():
        print("  [harmonize] optical already exists — skipping.")
        return

    wac_raw = Path(__file__).resolve().parents[2] / "data" / "raw" / "optical"
    local_name = region["wac_local_filename"]
    wac_path = wac_raw / local_name

    if not wac_path.exists():
        raise FileNotFoundError(
            f"WAC GeoTIFF not found at {wac_path}. "
            "Run extract_optical first."
        )

    ppd = region["pixels_per_degree"]
    lat_min = region["lat_min"]
    lat_max = region["lat_max"]
    lon_min = region["lon_min"]
    h = region["grid_height"]
    w = region["grid_width"]

    import math
    pixel_size = MOON_RADIUS_M * math.radians(1.0 / ppd)
    left_m = MOON_RADIUS_M * math.radians(lon_min)
    top_m = MOON_RADIUS_M * math.radians(lat_max)
    target_transform = Affine(pixel_size, 0.0, left_m, 0.0, -pixel_size, top_m)

    wac_common = np.zeros((h, w), dtype=np.float32)
    coverage = np.zeros((h, w), dtype=np.float32)

    with rasterio.open(wac_path) as src:
        # Read only the AOI (+ a small margin so edge cells average fully)
        # instead of the whole ~500 MB global raster.
        margin = 4
        right_m = left_m + pixel_size * w
        bottom_m = top_m - pixel_size * h
        win = from_bounds(left_m, bottom_m, right_m, top_m, transform=src.transform)
        col0 = max(int(np.floor(win.col_off)) - margin, 0)
        row0 = max(int(np.floor(win.row_off)) - margin, 0)
        col1 = min(int(np.ceil(win.col_off + win.width)) + margin, src.width)
        row1 = min(int(np.ceil(win.row_off + win.height)) + margin, src.height)
        window = Window(col0, row0, col1 - col0, row1 - row0)

        src_data = src.read(1, window=window)
        src_transform = src.window_transform(window)
        nodata = src.nodata

        # Area-averaged resampling to the common grid
        reproject(
            source=src_data,
            destination=wac_common,
            src_transform=src_transform,
            src_crs=src.crs,
            src_nodata=nodata,
            dst_transform=target_transform,
            dst_crs=src.crs,
            dst_nodata=nodata if nodata is not None else 0,
            resampling=Resampling.average,
        )

        # Coverage: proportion of valid source pixels per common cell
        source_valid = np.ones(src_data.shape, dtype=np.float32)
        if nodata is not None:
            source_valid[src_data == nodata] = 0.0
        reproject(
            source=source_valid,
            destination=coverage,
            src_transform=src_transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=src.crs,
            resampling=Resampling.average,
        )

    if coverage.max() <= 0.0:
        raise RuntimeError(
            "WAC window has zero valid coverage — region lies outside the "
            "downloaded WAC tile or is entirely nodata."
        )

    save_npy_atomic(wac_out, wac_common)
    save_npy_atomic(cov_out, coverage)
    print(f"  [harmonize] optical  shape={wac_common.shape}  "
          f"coverage_mean={coverage.mean():.4f}")


# ===========================================================================
# LOLA DEM harmonization  (exact 2x2 block averaging)
# ===========================================================================

def _harmonize_dem(region: dict) -> None:
    inter = region["intermediate_dir"]
    in_path = inter / "dem" / "lola_ldem256_pilot.npy"
    out_dir = inter / "dem"

    dem_out = out_dir / "lola_dem_common_grid.npy"
    mask_out = out_dir / "lola_dem_common_valid_mask.npy"
    cov_out = out_dir / "lola_dem_common_coverage.npy"

    if dem_out.exists() and mask_out.exists() and cov_out.exists():
        print("  [harmonize] DEM already exists — skipping.")
        return

    if not in_path.exists():
        raise FileNotFoundError(
            f"Native LOLA DEM not found at {in_path}. "
            "Run extract_dem first."
        )

    h = region["grid_height"]   # 256
    w = region["grid_width"]    # 256

    dem = np.load(in_path)  # 512x512

    if dem.shape != (h * 2, w * 2):
        raise ValueError(
            f"Native LOLA DEM shape {dem.shape} != expected {(h*2, w*2)}"
        )

    # Reshape into 2x2 native blocks and average
    native_valid = np.isfinite(dem)
    dem_blocks = dem.reshape(h, 2, w, 2)
    valid_blocks = native_valid.reshape(h, 2, w, 2)

    valid_count = valid_blocks.sum(axis=(1, 3)).astype(np.float32)
    sum_blocks = np.where(valid_blocks, dem_blocks, 0.0).sum(axis=(1, 3))

    with np.errstate(invalid="ignore"):
        dem_common = np.where(valid_count > 0, sum_blocks / valid_count, np.nan)

    dem_common = dem_common.astype(np.float32)
    dem_common, valid_mask = fill_gaps(dem_common)   # no-op when complete
    coverage = (valid_count / 4.0).astype(np.float32)

    save_npy_atomic(dem_out, dem_common)
    save_npy_atomic(mask_out, valid_mask)
    save_npy_atomic(cov_out, coverage)
    print(f"  [harmonize] DEM  shape={dem_common.shape}  "
          f"valid={valid_mask.sum()}/{valid_mask.size}  "
          f"min={np.nanmin(dem_common):.4f} km  max={np.nanmax(dem_common):.4f} km")


# ===========================================================================
# Diviner thermal harmonization  (direct native-grid mapping)
# ===========================================================================

def _harmonize_thermal(region: dict) -> None:
    inter = region["intermediate_dir"]
    in_path = inter / "thermal" / "diviner_tbol_sam_pilot.npy"
    out_dir = inter / "thermal"

    therm_out = out_dir / "diviner_tbol_sam_common_grid.npy"
    mask_out = out_dir / "diviner_tbol_sam_common_valid_mask.npy"
    cov_out = out_dir / "diviner_tbol_sam_common_coverage.npy"

    if therm_out.exists() and mask_out.exists() and cov_out.exists():
        print("  [harmonize] thermal already exists — skipping.")
        return

    if not in_path.exists():
        raise FileNotFoundError(
            f"Native Diviner array not found at {in_path}. "
            "Run extract_thermal first."
        )

    thermal = np.load(in_path)

    expected = (region["grid_height"], region["grid_width"])
    if thermal.shape != expected:
        raise ValueError(
            f"Diviner shape {thermal.shape} != expected {expected}. "
            "Diviner native resolution must match common grid (128 PPD)."
        )

    # Validate pixel-center geometry
    grid_meta_path = (
        inter / "registration" / "common_grid_metadata.json"
    )
    if grid_meta_path.exists():
        with open(grid_meta_path, "r") as fh:
            gm = json.load(fh)
        common_lons = np.load(inter / "registration" / "common_grid_longitudes.npy")
        common_lats = np.load(inter / "registration" / "common_grid_latitudes.npy")

        ppd = region["pixels_per_degree"]
        div_row_start = region["diviner_row_start"]
        div_col_start = region["diviner_col_start"]
        diviner_lats = 70.0 - (np.arange(div_row_start, div_row_start + thermal.shape[0]) + 0.5) / ppd
        diviner_lons = (np.arange(div_col_start, div_col_start + thermal.shape[1]) + 0.5) / ppd

        if not (np.allclose(common_lons, diviner_lons, atol=1e-9) and
                np.allclose(common_lats, diviner_lats, atol=1e-9)):
            raise ValueError(
                "Diviner pixel centers do not align with the common grid. "
                "Check region coordinates."
            )

    # Isolated NaN pixels are filled from the nearest observation so that
    # downstream arrays are finite; valid_mask / coverage stay 0 there.
    thermal_common, valid_mask = fill_gaps(thermal)
    coverage = valid_mask.astype(np.float32)
    if not valid_mask.all():
        print(f"  [harmonize] thermal: gap-filled "
              f"{int((~valid_mask).sum())} non-finite pixel(s)")

    save_npy_atomic(therm_out, thermal_common)
    save_npy_atomic(mask_out, valid_mask)
    save_npy_atomic(cov_out, coverage)
    print(f"  [harmonize] thermal  shape={thermal_common.shape}  "
          f"valid={valid_mask.sum()}/{valid_mask.size}  "
          f"min={np.nanmin(thermal_common):.2f}  max={np.nanmax(thermal_common):.2f}")


# ===========================================================================
# Public entry point
# ===========================================================================

def run(region_id: str) -> None:
    region = load_region(region_id)
    _harmonize_optical(region)
    _harmonize_dem(region)
    _harmonize_thermal(region)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    args = ap.parse_args()
    run(args.region)
