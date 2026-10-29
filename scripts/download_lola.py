"""Download the LOLA LDEM_256 binary for the Northern Hemisphere.

File size  : ~4.2 GB
Format     : flat binary, float32 little-endian, row-major
Coverage   : 0-90 N, 0-180 E at 256 pixels/degree
Resolution : ~118.5 m/pixel

Usage
-----
    python scripts/download_lola.py

The file is saved to:
    data/raw/dem/LDEM_256_00N_90N_000_180_FLOAT.IMG
"""

import sys
from pathlib import Path

import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DEST_DIR  = ROOT / "data" / "raw" / "dem"
DEST_FILE = DEST_DIR / "LDEM_256_00N_90N_000_180_FLOAT.IMG"

# Try URLs in order until one works
# NOTE: pds-geosciences.wustl.edu is sometimes unavailable.
# MIT imbrium is the primary mirror for LOLA GDR products.
CANDIDATE_URLS = [
    # MIT Imbrium — LOLA Science Team archive (most reliable)
    "https://imbrium.mit.edu/DATA/LOLA_GDR/POLAR/IMG/LDEM_256_00N_90N_000_180_FLOAT.IMG",
    # MIT Imbrium — cylindrical layout variant
    "https://imbrium.mit.edu/DATA/LOLA_GDR/CYLINDRICAL/IMG/LDEM_256_00N_90N_000_180_FLOAT.IMG",
    # PDS Geosciences Node (Washington U) — polar path
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/float/north/ldem_256_00n_90n_000_180_float.img",
    # PDS Geosciences Node — uppercase variant
    "https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/float/north/LDEM_256_00N_90N_000_180_FLOAT.IMG",
]

EXPECTED_BYTES = 23_040 * 46_080 * 4   # 4 246 732 800


def _find_url() -> str:
    """Return the first URL that returns HTTP 200 (HEAD request)."""
    print("  Probing candidate URLs ...")
    for url in CANDIDATE_URLS:
        try:
            r = requests.head(url, timeout=15, allow_redirects=True)
            if r.status_code == 200:
                content_length = int(r.headers.get("content-length", 0))
                print(f"  Found: {url}")
                if content_length:
                    print(f"  Remote size: {content_length / 1e9:.2f} GB")
                return url
            else:
                print(f"  HTTP {r.status_code}: {url}")
        except requests.RequestException as exc:
            print(f"  Error probing {url}: {exc}")

    return None


def _download(url: str) -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = DEST_FILE.with_suffix(".tmp")

    # Resume support: check how many bytes already downloaded
    start_byte = tmp_path.stat().st_size if tmp_path.exists() else 0
    if start_byte:
        print(f"  Resuming from {start_byte / 1e9:.2f} GB ...")

    headers = {"Range": f"bytes={start_byte}-"} if start_byte else {}

    with requests.get(url, headers=headers, stream=True, timeout=60) as resp:
        if resp.status_code not in (200, 206):
            raise RuntimeError(
                f"Unexpected HTTP {resp.status_code} from {url}"
            )

        total = int(resp.headers.get("content-length", EXPECTED_BYTES))
        mode  = "ab" if start_byte else "wb"

        with open(tmp_path, mode) as fh, tqdm(
            total=total + start_byte,
            initial=start_byte,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="  LOLA download",
        ) as bar:
            for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))

    # Verify size
    actual = tmp_path.stat().st_size
    print(f"\n  Downloaded {actual / 1e9:.2f} GB")
    if actual < EXPECTED_BYTES * 0.99:   # allow 1% tolerance
        raise RuntimeError(
            f"File appears truncated: {actual} bytes "
            f"(expected ~{EXPECTED_BYTES}). "
            "Re-run the script to resume the download."
        )

    tmp_path.rename(DEST_FILE)
    print(f"  Saved to: {DEST_FILE}")


def main() -> int:
    print("=" * 60)
    print("  LOLA LDEM_256 Download  (Northern Hemisphere, 0-180 E)")
    print("=" * 60)

    if DEST_FILE.exists():
        size = DEST_FILE.stat().st_size
        if size >= EXPECTED_BYTES * 0.99:
            print(f"\n  File already exists ({size / 1e9:.2f} GB).")
            print(f"  Location: {DEST_FILE}")
            print("  Nothing to do.")
            return 0
        else:
            print(f"\n  Incomplete file found ({size / 1e9:.2f} GB) — resuming.")

    url = _find_url()
    if not url:
        print("\n" + "=" * 60)
        print("  ERROR: Could not find the LOLA file at any candidate URL.")
        print("=" * 60)
        print("""
  Please download the file manually:

  1. Go to: https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/
             lrolol_1xxx/data/lola_gdr/polar/float/north/

  2. Download: LDEM_256_00N_90N_000_180_FLOAT.IMG  (~4.2 GB)

  3. Place it in:
       """ + str(DEST_DIR) + """

  4. Re-run the pipeline:
       python run_pipeline.py --skip-lacus-mortis --all
""")
        return 1

    print(f"\n  Starting download (~4.2 GB) ...")
    print("  This will take 5-30 min depending on your connection.")
    print("  The download is resumable — safe to Ctrl+C and restart.\n")

    _download(url)

    print("\n" + "=" * 60)
    print("  Download complete! Now run:")
    print("  python run_pipeline.py --skip-lacus-mortis --all")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
