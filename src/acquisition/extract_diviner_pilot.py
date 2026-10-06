from pathlib import Path

import numpy as np
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

URL = (
    "https://pds-geosciences.wustl.edu/"
    "lro/urn-nasa-pds-lro_diviner_derived1/"
    "data_derived_ghrm/img/"
    "dghrm_tbol_sam_70s70n_img.img"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "thermal"
    / "diviner_tbol_sam_pilot.npy"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# Source geometry from PDS label
# ============================================================

TOTAL_COLUMNS = 46080
BYTES_PER_SAMPLE = 4

ROW_START = 3072
ROW_END = 3328

COL_START = 3456
COL_END = 3712

HEIGHT = ROW_END - ROW_START
WIDTH = COL_END - COL_START

EXPECTED_BYTES = WIDTH * BYTES_PER_SAMPLE


print("Diviner source:")
print(URL)

print("\nPilot window:")
print(f"Rows    : {ROW_START} -> {ROW_END}")
print(f"Columns : {COL_START} -> {COL_END}")
print(f"Shape   : {HEIGHT} x {WIDTH}")


# ============================================================
# Allocate output
# ============================================================

thermal = np.empty(
    (HEIGHT, WIDTH),
    dtype=np.float32
)


# ============================================================
# Download one contiguous row segment at a time
# ============================================================

with requests.Session() as session:

    for i, row in enumerate(
        range(ROW_START, ROW_END)
    ):

        byte_start = (
            row * TOTAL_COLUMNS
            * BYTES_PER_SAMPLE
            + COL_START
            * BYTES_PER_SAMPLE
        )

        byte_end = (
            byte_start
            + EXPECTED_BYTES
            - 1
        )

        headers = {
            "Range": f"bytes={byte_start}-{byte_end}"
        }

        response = session.get(
            URL,
            headers=headers,
            timeout=60
        )

        if response.status_code != 206:
            raise RuntimeError(
                f"Expected HTTP 206 for row {row}, "
                f"received {response.status_code}"
            )

        if len(response.content) != EXPECTED_BYTES:
            raise RuntimeError(
                f"Unexpected byte count at row {row}: "
                f"{len(response.content)}"
            )

        row_data = np.frombuffer(
            response.content,
            dtype="<f4"
        )

        if row_data.size != WIDTH:
            raise RuntimeError(
                f"Unexpected sample count at row {row}: "
                f"{row_data.size}"
            )

        thermal[i, :] = row_data

        if (i + 1) % 32 == 0:
            print(
                f"Downloaded {i + 1}/{HEIGHT} rows"
            )


# ============================================================
# Save
# ============================================================

np.save(
    OUTPUT_PATH,
    thermal
)


# ============================================================
# Statistics
# ============================================================

finite = np.isfinite(thermal)

print("\nExtraction complete.")
print(f"Output : {OUTPUT_PATH}")
print(f"Shape  : {thermal.shape}")
print(f"Dtype  : {thermal.dtype}")
print(
    f"Finite : "
    f"{finite.sum()} / {thermal.size}"
)

if finite.any():

    values = thermal[finite]

    print(
        f"Min    : {values.min():.6f}"
    )

    print(
        f"Max    : {values.max():.6f}"
    )

    print(
        f"Mean   : {values.mean():.6f}"
    )

    print(
        f"Std    : {values.std():.6f}"
    )

nan_count = np.count_nonzero(
    ~finite
)

print(f"NaN    : {nan_count}")