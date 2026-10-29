"""Step 2 — Extract LOLA DEM window for a region.

Reads the 512x512 float32 window from LDEM_256_00N_90N_000_180_FLOAT.IMG.

If the ~4.2 GB file is already in data/raw/dem/ it is read locally;
otherwise only the required rows (512 x 2 KB) are streamed from the NASA PDS
server with HTTP Range requests, so no multi-GB download is needed.

Usage
-----
    python -m src.pipeline.extract_dem --region posidonius
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import load_region, LOLA_RECORD_BYTES
from .io_utils import save_npy_atomic, stream_rows

ROOT = Path(__file__).resolve().parents[2]

BYTES_PER_SAMPLE = 4


def _read_local(path: Path, row_offsets, row_bytes: int) -> np.ndarray:
    out = np.empty((len(row_offsets), row_bytes // 4), dtype=np.float32)
    with open(path, "rb") as fh:
        for i, off in enumerate(tqdm(row_offsets, desc="  LOLA rows (local)",
                                     unit="row")):
            fh.seek(off)
            raw = fh.read(row_bytes)
            if len(raw) != row_bytes:
                raise RuntimeError(f"Short read at offset {off}")
            out[i, :] = np.frombuffer(raw, dtype="<f4")
    return out


def run(region_id: str) -> None:
    region = load_region(region_id)

    out_dir = region["intermediate_dir"] / "dem"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "lola_ldem256_pilot.npy"

    if out_path.exists():
        print(f"  [dem] {out_path.name} already exists — skipping.")
        return

    row_start = region["lola_row_start"]
    row_end = region["lola_row_end"]
    col_start = region["lola_col_start"]
    width = region["lola_width"]          # 512
    row_bytes = width * BYTES_PER_SAMPLE

    offsets = [
        r * LOLA_RECORD_BYTES + col_start * BYTES_PER_SAMPLE
        for r in range(row_start, row_end)
    ]

    local_path = ROOT / "data" / "raw" / "dem" / region["lola_local_filename"]
    if local_path.exists():
        print(f"  [dem] Reading LOLA window from local file  "
              f"rows {row_start}-{row_end}, cols {col_start}-{col_start + width}")
        dem = _read_local(local_path, offsets, row_bytes)
    else:
        print(f"  [dem] Streaming LOLA window over HTTP  "
              f"rows {row_start}-{row_end}, cols {col_start}-{col_start + width}")
        dem = stream_rows(region["lola_url"], offsets, row_bytes,
                          desc="  LOLA rows")

    if dem.shape != (region["lola_height"], width):
        raise RuntimeError(f"Unexpected LOLA window shape {dem.shape}")

    finite = np.isfinite(dem)
    if not finite.any():
        raise RuntimeError("LOLA window contains no finite values — "
                           "check the window indices / source URL.")
    # LDEM heights are km relative to the 1737.4 km reference sphere;
    # anything outside +-20 km means we read the wrong bytes.
    lo, hi = float(dem[finite].min()), float(dem[finite].max())
    if lo < -20.0 or hi > 20.0:
        raise RuntimeError(
            f"LOLA heights {lo:.2f}..{hi:.2f} km are implausible — "
            "wrong file offset?"
        )

    save_npy_atomic(out_path, dem)
    print(f"  [dem] Saved {out_path}  shape={dem.shape}  "
          f"finite={finite.sum()}/{dem.size}  min={lo:.4f} km  max={hi:.4f} km")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    args = ap.parse_args()
    run(args.region)
