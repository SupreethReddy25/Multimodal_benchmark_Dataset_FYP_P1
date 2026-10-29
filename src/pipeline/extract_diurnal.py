"""Supplementary layer -- tile-level diurnal temperature curve (Diviner GCP).

The pixel-level thermal layer is a single local time (midnight).  To give the
benchmark some day/night information we add, for every 64x64 tile, the
Diviner *bolometric brightness temperature versus local time* from the PDS
Global Cumulative Products (Williams et al. 2017): all nadir observations from
2009-07-05 to 2015-04-01 binned at 0.5 deg x 0.5 deg x 0.25 h of local time.

One 0.5 deg bin is exactly one 64x64 tile (region edges are multiples of
0.5 deg), so the curve is a tile-level value (~15 km), NOT a pixel-aligned
map.  Bins without observations are NaN (-9999 in the source).

Output (per region): data/benchmark/<region>/diurnal_tbol.npz with
    tbol[16, 96]   K, tile index = tile_RR_CC order (row-major, north first)
    local_time[96] hours (bin centres 0.125 ... 23.875)
    source         product description

Rows in the .tab files are fixed length (113 bytes) with the order
longitude (720) > latitude (20, south to north) > local time (96), after one
113-byte header line, so a single (lon, lat-band) column is one Range read.
"""

import io
from pathlib import Path

import numpy as np
import requests

from .config import load_region, list_regions
from .io_utils import fetch_range, save_npy_atomic  # noqa: F401  (save_npy_atomic kept for symmetry)

BASE = ("https://pds-geosciences.wustl.edu/lro/urn-nasa-pds-lro_diviner_derived1/"
        "data_derived_gcp/global_cumul_avg_cyl_{band}_002.tab")
ROW = 113
N_LT, N_LAT = 96, 20


def band_name(lo: int) -> str:
    hi = lo + 10
    if lo >= 0:
        return f"{lo:02d}n{hi:02d}n"
    return f"{abs(lo):02d}s{abs(hi):02d}s"


def _column(session, band_lo, lon_idx):
    """All (lat, ltim) rows of one longitude bin of one band -> (20, 96, 8)."""
    start = ROW + ROW * lon_idx * N_LAT * N_LT
    raw = fetch_range(session, BASE.format(band=band_name(band_lo)), start,
                      ROW * N_LAT * N_LT)
    arr = np.genfromtxt(io.BytesIO(raw), delimiter=",")
    arr = arr.reshape(N_LAT, N_LT, -1)
    return arr


def extract_region(rid: str, out_dir: Path) -> Path:
    r = load_region(rid)
    out = out_dir / rid / "diurnal_tbol.npz"
    if out.exists():
        return out
    sess = requests.Session()
    tbol = np.full((4, 4, N_LT), np.nan, dtype=np.float32)
    ltim = None
    for i in range(4):                       # tile row, north first
        lat_c = r["lat_max"] - 0.5 * (i + 0.5)
        band_lo = int(np.floor(lat_c / 10.0)) * 10
        lat_idx = int(round((lat_c - (band_lo + 0.25)) / 0.5))
        for j in range(4):                   # tile column, west first
            lon_c = r["lon_min"] + 0.5 * (j + 0.5)
            lon_c = lon_c - 360.0 if lon_c > 180 else lon_c
            lon_idx = int(round((lon_c + 179.75) / 0.5))
            col = _column(sess, band_lo, lon_idx)
            # sanity: the bin centres in the file must match what we asked for
            assert abs(col[lat_idx, 0, 0] - lon_c) < 1e-3, (rid, i, j, col[lat_idx, 0, 0], lon_c)
            assert abs(col[lat_idx, 0, 1] - lat_c) < 1e-3, (rid, i, j, col[lat_idx, 0, 1], lat_c)
            v = col[lat_idx, :, -1].astype(np.float32)   # tbol is the last column
            v[v <= -9000] = np.nan
            tbol[i, j] = v
            ltim = col[lat_idx, :, 2]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, tbol=tbol.reshape(16, N_LT), local_time=ltim,
             source="Diviner GCP global_cumul_avg_cyl_*_002 (Williams et al. 2017), "
                    "0.5 deg x 0.25 h bins, 2009-07-05..2015-04-01, tbol column")
    return out


def main():
    ROOT = Path(__file__).resolve().parents[2]
    out_dir = ROOT / "data" / "benchmark"
    for rid in list_regions():
        p = extract_region(rid, out_dir)
        d = np.load(p, allow_pickle=False)
        t = d["tbol"]
        print(f"{rid:<18} tiles with a noon value: "
              f"{int(np.isfinite(t[:, 47:49]).any(axis=1).sum())}/16  "
              f"midnight mean {np.nanmean(t[:, [0, 95]]):.1f} K", flush=True)


if __name__ == "__main__":
    main()
