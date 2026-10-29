"""Pipeline configuration loader.

Reads configs/regions.yaml and returns a fully-populated region dict
including derived geometry indices for Diviner and LOLA HTTP extraction.
"""

from pathlib import Path
import yaml

# Project root is three levels above src/pipeline/config.py
ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Constants shared across all regions / instruments
# ---------------------------------------------------------------------------

# Common benchmark grid
TARGET_PPD = 128                    # pixels per degree
GRID_SIZE_DEG = 2.0                 # region spans 2deg x 2deg
GRID_PIXELS = int(GRID_SIZE_DEG * TARGET_PPD)   # 256
TARGET_TILE_SIZE = 64
TARGET_TILES = 16                   # 4 x 4
TARGET_RESOLUTION_M = 1_737_400.0 * (3.14159265358979 / 180.0) / TARGET_PPD

# LOLA source geometry: four 256-PPD tiles (N/S hemisphere x 0-180 / 180-360 E).
# In every tile line 0 is the tile's maximum latitude and sample 0 its
# westernmost longitude (verified against the PDS labels).
LOLA_PPD = 256
LOLA_TOTAL_COLS = 46_080            # 180deg * 256
LOLA_RECORD_BYTES = 184_320         # 46080 * 4

# Diviner source geometry (128 PPD, 70S-70N, 0-360E)
DIVINER_PPD = 128
DIVINER_TOTAL_COLS = 46_080         # 360deg * 128
DIVINER_LAT_TOP = 70.0              # file top latitude
DIVINER_LON_ORIGIN = 0.0            # file left longitude

# WAC 100 m mosaic: 90deg x 60deg tiles, 4 longitude sectors x N/S band
WAC_TILE_LON_SPAN = 90.0
WAC_TILE_LAT_SPAN = 60.0
WAC_LAT_LIMIT = 60.0                # equatorial tiles stop at 60 deg


def lola_tile(lat_min: float, lat_max: float, lon_min: float):
    """Return (stem, lat_top, lon_west) of the LOLA LDEM_256 tile for a box."""
    if lat_min >= 0:
        stem_lat, lat_top = "00n_90n", 90.0
    else:
        stem_lat, lat_top = "90s_00s", 0.0
    if lon_min < 180.0:
        stem_lon, lon_west = "000_180", 0.0
    else:
        stem_lon, lon_west = "180_360", 180.0
    return f"ldem_256_{stem_lat}_{stem_lon}_float", lat_top, lon_west


def wac_tile(lat_min: float, lon_min: float) -> str:
    """Return the WAC_GLOBAL tile id (e.g. 'E300N0450') covering a box."""
    hemi = "N" if lat_min >= 0 else "S"
    centre = int(lon_min // WAC_TILE_LON_SPAN) * 90 + 45
    return f"E300{hemi}{centre * 10:04d}"


def load_region(region_id: str) -> dict:
    """Load and validate a region config, adding derived geometry fields.

    Parameters
    ----------
    region_id : str
        Key in configs/regions.yaml (e.g. ``"posidonius"``).

    Returns
    -------
    dict
        Region config extended with:
        - ``id``: the region key
        - ``intermediate_dir``, ``benchmark_dir``: resolved Path objects
        - ``diviner_row_start/end``, ``diviner_col_start/end``
        - ``lola_row_start/end``, ``lola_col_start/end``
        - ``sources``: dict with PDS URLs and WAC local filename
    """

    config_path = ROOT / "configs" / "regions.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Region registry not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    available = list(cfg.get("regions", {}).keys())
    if region_id not in cfg.get("regions", {}):
        raise ValueError(
            f"Unknown region '{region_id}'.  "
            f"Available regions: {available}"
        )

    region = dict(cfg["regions"][region_id])
    region["id"] = region_id
    region["sources"] = cfg.get("sources", {})

    lat_min = region["lat_min"]
    lat_max = region["lat_max"]
    lon_min = region["lon_min"]
    lon_max = region["lon_max"]

    # ------------------------------------------------------------------
    # Validate extents
    # ------------------------------------------------------------------
    step = 1.0 / TARGET_PPD
    for v in (lat_min, lat_max, lon_min, lon_max):
        assert abs(v / step - round(v / step)) < 1e-9, (
            f"{region_id}: bounds must be multiples of 1/{TARGET_PPD} deg"
        )
    assert abs((lat_max - lat_min) - GRID_SIZE_DEG) < 1e-9,         "Region must be exactly 2deg in latitude"
    assert abs((lon_max - lon_min) - GRID_SIZE_DEG) < 1e-9,         "Region must be exactly 2deg in longitude"
    assert -WAC_LAT_LIMIT <= lat_min < lat_max <= WAC_LAT_LIMIT,         "Region must lie within the +-60 deg WAC equatorial tiles"
    assert 0.0 <= lon_min < lon_max <= 360.0
    # A region must not straddle a WAC tile edge, the equator, or the
    # 0/180 E seam between LOLA tiles.
    assert int(lon_min // WAC_TILE_LON_SPAN) == int((lon_max - 1e-9) // WAC_TILE_LON_SPAN),         f"{region_id}: straddles a WAC longitude tile edge"
    assert (lat_min >= 0) or (lat_max <= 0),         f"{region_id}: straddles the equator (WAC/LOLA N/S tile edge)"
    assert int(lon_min // 180) == int((lon_max - 1e-9) // 180),         f"{region_id}: straddles the 180 E LOLA tile seam"

    # ------------------------------------------------------------------
    # Output directories (region-namespaced)
    # ------------------------------------------------------------------
    region["intermediate_dir"] = ROOT / "data" / "intermediate" / region_id
    region["benchmark_dir"] = ROOT / "data" / "benchmark" / region_id
    region["output_dir"] = ROOT / "outputs" / region_id

    # ------------------------------------------------------------------
    # Common-grid dimensions
    # ------------------------------------------------------------------
    region["grid_height"] = GRID_PIXELS   # 256
    region["grid_width"] = GRID_PIXELS    # 256
    region["resolution_m"] = TARGET_RESOLUTION_M
    region["pixels_per_degree"] = TARGET_PPD

    # ------------------------------------------------------------------
    # Diviner window indices
    # (origin = 70N top, 0E left; 128 PPD; integer pixel boundaries)
    # ------------------------------------------------------------------
    region["diviner_row_start"] = round((DIVINER_LAT_TOP - lat_max) * DIVINER_PPD)
    region["diviner_row_end"] = round((DIVINER_LAT_TOP - lat_min) * DIVINER_PPD)
    region["diviner_col_start"] = round((lon_min - DIVINER_LON_ORIGIN) * DIVINER_PPD)
    region["diviner_col_end"] = round((lon_max - DIVINER_LON_ORIGIN) * DIVINER_PPD)
    region["diviner_height"] = region["diviner_row_end"] - region["diviner_row_start"]
    region["diviner_width"] = region["diviner_col_end"] - region["diviner_col_start"]

    # ------------------------------------------------------------------
    # LOLA window indices (256 PPD -> native window is 512x512) and the
    # tile-specific source URL / local filename
    # ------------------------------------------------------------------
    stem, lola_top, lola_west = lola_tile(lat_min, lat_max, lon_min)
    region["lola_row_start"] = round((lola_top - lat_max) * LOLA_PPD)
    region["lola_row_end"] = round((lola_top - lat_min) * LOLA_PPD)
    region["lola_col_start"] = round((lon_min - lola_west) * LOLA_PPD)
    region["lola_col_end"] = round((lon_max - lola_west) * LOLA_PPD)
    region["lola_height"] = region["lola_row_end"] - region["lola_row_start"]
    region["lola_width"] = region["lola_col_end"] - region["lola_col_start"]
    region["lola_tile"] = stem
    region["lola_url"] = region["sources"]["lola_base_url"].rstrip("/") + f"/{stem}.img"
    region["lola_local_filename"] = stem.upper() + ".IMG"

    tile = wac_tile(lat_min, lon_min)
    region["wac_tile"] = tile
    region["wac_url"] = region["sources"]["wac_base_url"].rstrip("/") + f"/WAC_GLOBAL_{tile}_100M.TIF"
    region["wac_local_filename"] = f"WAC_GLOBAL_{tile}_100M.TIF"

    # Sanity checks
    assert region["lola_height"] == 512, (
        f"LOLA height should be 512, got {region['lola_height']}"
    )
    assert region["lola_width"] == 512, (
        f"LOLA width should be 512, got {region['lola_width']}"
    )
    assert region["diviner_height"] == GRID_PIXELS
    assert region["diviner_width"] == GRID_PIXELS

    return region


def list_regions() -> list:
    """Return all available region IDs."""
    config_path = ROOT / "configs" / "regions.yaml"
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return list(cfg.get("regions", {}).keys())
