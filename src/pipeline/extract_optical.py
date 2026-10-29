"""Step 3 — Extract LROC WAC optical window for a region.

The WAC GeoTIFF (~473 MB) is downloaded once to data/raw/optical/ and
reused for all regions.  The pilot AOI window is extracted using
rasterio and saved as a raw numpy array.

Usage
-----
    python -m src.pipeline.extract_optical --region posidonius
"""

import argparse
from pathlib import Path

import numpy as np
import requests
import rasterio
from rasterio.windows import from_bounds
from tqdm import tqdm

from .config import load_region
from .io_utils import save_npy_atomic

ROOT = Path(__file__).resolve().parents[2]
MOON_RADIUS_M = 1_737_400.0


def _lon_to_x(lon_deg: float) -> float:
    import math
    return MOON_RADIUS_M * math.radians(lon_deg)


def _lat_to_y(lat_deg: float) -> float:
    import math
    return MOON_RADIUS_M * math.radians(lat_deg)


def _remote_size(url: str) -> int:
    """Total file size via a 1-byte Range request (follows the PDS redirect)."""
    r = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=60)
    r.raise_for_status()
    cr = r.headers.get("Content-Range", "")
    if "/" not in cr:
        raise RuntimeError(f"Server did not report a total size ({cr!r}).")
    return int(cr.rsplit("/", 1)[1])


def _ensure_wac(region: dict) -> Path:
    """Download the WAC GeoTIFF (resumable, size-verified) if not cached."""
    raw_optical = ROOT / "data" / "raw" / "optical"
    raw_optical.mkdir(parents=True, exist_ok=True)

    local_name = region["wac_local_filename"]
    local_path = raw_optical / local_name
    part_path = local_path.with_name(local_name + ".part")
    url = region["wac_url"]

    if local_path.exists():
        print(f"  [optical] WAC file cached at {local_path}")
        return local_path

    total = _remote_size(url)
    have = part_path.stat().st_size if part_path.exists() else 0
    if have > total:
        part_path.unlink()
        have = 0
    print(f"  [optical] Downloading WAC GeoTIFF ({total / 1e6:.0f} MB, one-time"
          f"{', resuming at %.0f MB' % (have / 1e6) if have else ''}) ...")

    if have < total:
        headers = {"Range": f"bytes={have}-"} if have else {}
        with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
            if resp.status_code not in (200, 206):
                resp.raise_for_status()
            if have and resp.status_code != 206:
                have = 0   # server ignored Range -> restart
            with open(part_path, "ab" if have else "wb") as fh, tqdm(
                total=total, initial=have, unit="B", unit_scale=True,
                desc="  WAC download",
            ) as bar:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
                        bar.update(len(chunk))

    if part_path.stat().st_size != total:
        raise RuntimeError(
            f"WAC download incomplete ({part_path.stat().st_size}/{total} bytes). "
            "Re-run to resume."
        )
    part_path.replace(local_path)
    print(f"  [optical] WAC saved to {local_path}")
    return local_path


def run(region_id: str) -> None:
    region = load_region(region_id)

    out_dir = region["intermediate_dir"] / "optical"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "wac_pilot.npy"

    if out_path.exists():
        print(f"  [optical] {out_path.name} already exists — skipping.")
        return

    wac_path = _ensure_wac(region)

    lat_min = region["lat_min"]
    lat_max = region["lat_max"]
    lon_min = region["lon_min"]
    lon_max = region["lon_max"]

    left = _lon_to_x(lon_min)
    right = _lon_to_x(lon_max)
    bottom = _lat_to_y(lat_min)
    top = _lat_to_y(lat_max)

    print(f"  [optical] Extracting window  "
          f"lat [{lat_min},{lat_max}]  lon [{lon_min},{lon_max}]")

    with rasterio.open(wac_path) as src:
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        window = window.round_offsets().round_lengths()
        data = src.read(1, window=window)

    save_npy_atomic(out_path, data)

    print(f"  [optical] Saved {out_path}  "
          f"shape={data.shape}  dtype={data.dtype}  "
          f"min={data.min()}  max={data.max()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    args = ap.parse_args()
    run(args.region)
