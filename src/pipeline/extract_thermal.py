"""Step 1 — Extract Diviner thermal window for a region.

Streams only the required 256x256 float32 window from the NASA PDS
Diviner GHRM mosaic using HTTP Range requests.  No full-file download.

Usage
-----
    python -m src.pipeline.extract_thermal --region posidonius
"""

import argparse

import numpy as np

from .config import load_region, DIVINER_TOTAL_COLS
from .io_utils import save_npy_atomic, stream_rows

BYTES_PER_SAMPLE = 4


def run(region_id: str) -> None:
    region = load_region(region_id)

    out_dir = region["intermediate_dir"] / "thermal"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diviner_tbol_sam_pilot.npy"

    if out_path.exists():
        print(f"  [thermal] {out_path.name} already exists — skipping.")
        return

    row_start = region["diviner_row_start"]
    row_end = region["diviner_row_end"]
    col_start = region["diviner_col_start"]
    width = region["diviner_width"]

    print(f"  [thermal] Downloading Diviner window  "
          f"rows {row_start}-{row_end}, cols {col_start}-{region['diviner_col_end']}")

    offsets = [
        r * DIVINER_TOTAL_COLS * BYTES_PER_SAMPLE + col_start * BYTES_PER_SAMPLE
        for r in range(row_start, row_end)
    ]
    thermal = stream_rows(region["sources"]["diviner_url"], offsets,
                          width * BYTES_PER_SAMPLE, desc="  Diviner rows")

    finite = np.isfinite(thermal)
    if finite.sum() == 0:
        raise RuntimeError("Diviner window contains no finite values.")

    save_npy_atomic(out_path, thermal)
    print(f"  [thermal] Saved {out_path}  shape={thermal.shape}  "
          f"finite={finite.sum()}/{thermal.size}  "
          f"nan={np.count_nonzero(~finite)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", required=True)
    args = ap.parse_args()
    run(args.region)
