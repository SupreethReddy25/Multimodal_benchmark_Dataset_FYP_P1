"""Step 6 — Derive slope (degrees) and hillshade from common-grid DEM.

Slope uses central differences (numpy.gradient), as in the original pilot
scripts; it is not Horn's 3x3 weighted method.  Hillshade is computed with
azimuth=315 deg (NW illumination) and altitude=45 deg.

DEM values are in kilometres; converted to metres before differencing.
East-west pixel spacing is latitude-dependent (equirectangular grid).

Usage
-----
    python -m src.pipeline.derive_terrain --region posidonius
"""

import argparse
import numpy as np

from .config import load_region

LUNAR_RADIUS_M = 1_737_400.0
DEG_TO_RAD = np.pi / 180.0


def run(region_id: str) -> None:
    region = load_region(region_id)

    inter = region["intermediate_dir"]
    dem_path = inter / "dem" / "lola_dem_common_grid.npy"
    slope_out = inter / "dem" / "lola_common_slope_deg.npy"
    hillshade_out = inter / "dem" / "lola_common_hillshade.npy"

    if slope_out.exists() and hillshade_out.exists():
        print("  [terrain] slope + hillshade already exist — skipping.")
        return

    if not dem_path.exists():
        raise FileNotFoundError(
            f"Common-grid DEM not found at {dem_path}. "
            "Run harmonize first."
        )

    dem = np.load(dem_path).astype(np.float64)

    if not np.all(np.isfinite(dem)):
        raise ValueError("Common-grid DEM contains non-finite values.")

    ppd = region["pixels_per_degree"]          # 128
    lat_max = region["lat_max"]                # e.g. 46.0
    deg_per_pixel = 1.0 / ppd

    # North-south pixel spacing (constant in equirectangular projection)
    dy_m = LUNAR_RADIUS_M * DEG_TO_RAD * deg_per_pixel

    # East-west pixel spacing (latitude-dependent)
    lat_centers = lat_max - (np.arange(dem.shape[0]) + 0.5) * deg_per_pixel
    lat_rad = lat_centers * DEG_TO_RAD
    dx_m = LUNAR_RADIUS_M * np.cos(lat_rad) * DEG_TO_RAD * deg_per_pixel

    # Convert DEM from km to m
    dem_m = dem * 1000.0

    # Compute gradients in pixel units, then convert to physical
    dz_drow, dz_dcol = np.gradient(dem_m)
    dz_dy = dz_drow / dy_m
    dz_dx = dz_dcol / dx_m[:, None]

    # -----------------------------------------------------------------------
    # Slope (degrees)
    # -----------------------------------------------------------------------
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.degrees(slope_rad).astype(np.float32)

    # -----------------------------------------------------------------------
    # Hillshade  (NW illumination: azimuth=315, altitude=45)
    # -----------------------------------------------------------------------
    azimuth_deg = 315.0
    altitude_deg = 45.0

    azimuth_rad = np.radians(azimuth_deg)
    altitude_rad = np.radians(altitude_deg)

    # Surface normal in geographic convention
    zenith_rad = np.pi / 2.0 - altitude_rad

    # Aspect (bearing of steepest ascent from North, clockwise)
    aspect_rad = np.arctan2(-dz_dy, dz_dx) - np.pi / 2.0

    hillshade = (
        np.cos(zenith_rad) * np.cos(slope_rad)
        + np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad)
    )
    hillshade = np.clip(hillshade, 0.0, 1.0).astype(np.float32)

    np.save(slope_out, slope_deg)
    np.save(hillshade_out, hillshade)

    print(f"  [terrain] slope    mean={slope_deg.mean():.3f} deg  "
          f"max={slope_deg.max():.3f} deg")
    print(f"  [terrain] hillshade mean={hillshade.mean():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    args = ap.parse_args()
    run(args.region)
