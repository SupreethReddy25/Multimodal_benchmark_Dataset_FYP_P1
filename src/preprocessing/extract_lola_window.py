from pathlib import Path
import numpy as np

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMG_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dem"
    / "LDEM_256_00N_90N_000_180_FLOAT.IMG"
)

OUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "intermediate"
    / "dem"
    / "lola_ldem256_pilot.npy"
)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Source geometry from PDS label
# --------------------------------------------------
PIXELS_PER_DEGREE = 256
TOTAL_COLUMNS = 46080
RECORD_BYTES = 184320
BYTES_PER_SAMPLE = 4

LINE_PROJECTION_OFFSET = 23039.5
SAMPLE_PROJECTION_OFFSET = 46079.5

CENTER_LONGITUDE = 180.0

# --------------------------------------------------
# Pilot AOI
# --------------------------------------------------
LAT_MIN = 44.0
LAT_MAX = 46.0

LON_MIN = 27.0
LON_MAX = 29.0

# --------------------------------------------------
# Convert geographic coordinates to image coordinates
# --------------------------------------------------
row_north = round(
    LINE_PROJECTION_OFFSET - LAT_MAX * PIXELS_PER_DEGREE
)

row_south = round(
    LINE_PROJECTION_OFFSET - LAT_MIN * PIXELS_PER_DEGREE
)

col_west = round(
    SAMPLE_PROJECTION_OFFSET
    - CENTER_LONGITUDE * PIXELS_PER_DEGREE
    + LON_MIN * PIXELS_PER_DEGREE
)

col_east = round(
    SAMPLE_PROJECTION_OFFSET
    - CENTER_LONGITUDE * PIXELS_PER_DEGREE
    + LON_MAX * PIXELS_PER_DEGREE
)

height = row_south - row_north
width = col_east - col_west

print("DEM source:")
print(IMG_PATH)

print("\nExtracted window:")
print(f"Rows : {row_north} -> {row_south}")
print(f"Cols : {col_west} -> {col_east}")
print(f"Shape: {height} x {width}")

# --------------------------------------------------
# Safety checks
# --------------------------------------------------
if not IMG_PATH.exists():
    raise FileNotFoundError(f"DEM not found: {IMG_PATH}")

if width != 512 or height != 512:
    raise RuntimeError(
        f"Unexpected pilot dimensions: {height} x {width}"
    )

# --------------------------------------------------
# Read only required rows
# --------------------------------------------------
window = np.empty((height, width), dtype=np.float32)

with open(IMG_PATH, "rb") as f:

    for i, source_row in enumerate(
        range(row_north, row_south)
    ):
        byte_offset = (
            source_row * RECORD_BYTES
            + col_west * BYTES_PER_SAMPLE
        )

        f.seek(byte_offset)

        row = np.fromfile(
            f,
            dtype="<f4",
            count=width
        )

        if row.size != width:
            raise RuntimeError(
                f"Unexpected row length at source row "
                f"{source_row}: {row.size}"
            )

        window[i, :] = row

# --------------------------------------------------
# Save
# --------------------------------------------------
np.save(OUT_PATH, window)

# --------------------------------------------------
# Statistics
# --------------------------------------------------
finite = np.isfinite(window)

print("\nExtraction complete.")
print(f"Output: {OUT_PATH}")
print(f"Shape : {window.shape}")
print(f"Dtype : {window.dtype}")
print(f"Finite pixels: {finite.sum()} / {window.size}")

if finite.any():
    values = window[finite]

    print(f"Min  : {values.min():.6f} km")
    print(f"Max  : {values.max():.6f} km")
    print(f"Mean : {values.mean():.6f} km")
    print(f"Std  : {values.std():.6f} km")