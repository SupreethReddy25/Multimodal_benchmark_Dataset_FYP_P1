"""Load benchmark tiles into stacked arrays for the evaluation tasks."""

from pathlib import Path

import numpy as np
import yaml
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
LAYERS = ["optical", "optical_coverage", "dem", "dem_valid_mask", "slope",
          "hillshade", "thermal", "thermal_valid_mask"]


def load_splits() -> dict:
    with open(ROOT / "configs" / "splits.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_region_tiles(region_id: str) -> dict:
    """Return {layer: (16, 64, 64) array, 'lat': (16, 64, 64) latitude grid}."""
    tiles_dir = ROOT / "data" / "benchmark" / region_id / "tiles"
    tile_dirs = sorted(p for p in tiles_dir.iterdir() if p.name.startswith("tile_"))
    out = {k: [] for k in LAYERS}
    lat = []
    import json
    for td in tile_dirs:
        for k in LAYERS:
            out[k].append(np.load(td / f"{k}.npy"))
        meta = json.loads((td / "metadata.json").read_text(encoding="utf-8"))
        ext = meta["geographic_extent"]
        la = np.linspace(ext["latitude_max_deg"], ext["latitude_min_deg"], 64)
        lat.append(np.repeat(la[:, None], 64, axis=1))
    out = {k: np.stack(v) for k, v in out.items()}
    out["lat"] = np.stack(lat).astype(np.float32)
    return out


def valid_mask(t: dict) -> np.ndarray:
    """Pixels where every modality holds a genuine observation."""
    return ((t["optical_coverage"] >= 0.99) & t["dem_valid_mask"]
            & t["thermal_valid_mask"])


def pixel_features(t: dict, use_lat: bool = False) -> dict:
    """Named per-pixel feature stacks, each (N, 64, 64)."""
    opt = t["optical"].astype(np.float32)
    dem = t["dem"].astype(np.float32)
    dem_rel = dem - dem.mean(axis=(1, 2), keepdims=True)

    def smooth(a, s):
        return np.stack([ndimage.gaussian_filter(x, s, mode="reflect") for x in a])

    f = {
        "optical": opt,
        "optical_s2": smooth(opt, 2), "optical_s4": smooth(opt, 4),
        "dem_rel": dem_rel, "dem_rel_s4": smooth(dem_rel, 4),
        "slope": t["slope"].astype(np.float32),
        "slope_s2": smooth(t["slope"].astype(np.float32), 2),
        "hillshade": t["hillshade"].astype(np.float32),
    }
    if use_lat:
        f["lat"] = t["lat"]
    return f


VARIANTS = {
    "optical": ["optical", "optical_s2", "optical_s4"],
    "terrain": ["dem_rel", "dem_rel_s4", "slope", "slope_s2", "hillshade"],
    "optical+terrain": ["optical", "optical_s2", "optical_s4", "dem_rel",
                        "dem_rel_s4", "slope", "slope_s2", "hillshade"],
}
