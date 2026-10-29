"""Shared I/O helpers for the multi-region pipeline.

* ``fetch_range`` / ``stream_rows``  -- resilient HTTP Range reads (retries,
  keep-alive session, thread pool) used for Diviner and LOLA streaming.
* ``save_npy_atomic``                -- write-then-rename so an interrupted
  run never leaves a truncated file that a later run mistakes for complete.
* ``fill_gaps``                      -- nearest-valid-neighbour fill for the
  handful of NaN pixels some products contain (provenance is kept in the
  accompanying valid-mask / coverage layers).
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from scipy import ndimage
from tqdm import tqdm


def save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, path)


def fetch_range(session: requests.Session, url: str, start: int, length: int,
                retries: int = 5, timeout: int = 120) -> bytes:
    """Return exactly ``length`` bytes at ``start`` using an HTTP Range request."""
    last = None
    for attempt in range(retries):
        try:
            resp = session.get(
                url,
                headers={"Range": f"bytes={start}-{start + length - 1}"},
                timeout=timeout,
            )
            if resp.status_code != 206:
                raise RuntimeError(f"expected HTTP 206, got {resp.status_code}")
            if len(resp.content) != length:
                raise RuntimeError(
                    f"expected {length} bytes, got {len(resp.content)}"
                )
            return resp.content
        except (requests.RequestException, RuntimeError) as exc:
            last = exc
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Range read failed after {retries} attempts "
                       f"(offset {start}): {last}")


def stream_rows(url: str, row_offsets, row_bytes: int, desc: str,
                workers: int = 8) -> np.ndarray:
    """Fetch many equally-sized byte runs in parallel -> (n_rows, row_bytes//4) f32.

    ``row_offsets`` are absolute byte offsets, one per output row (order kept).
    """
    n = len(row_offsets)
    out = np.empty((n, row_bytes // 4), dtype=np.float32)

    with requests.Session() as session:
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=workers)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        def _one(i):
            raw = fetch_range(session, url, row_offsets[i], row_bytes)
            out[i, :] = np.frombuffer(raw, dtype="<f4")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(tqdm(pool.map(_one, range(n)), total=n, desc=desc, unit="row"))
    return out


def fill_gaps(arr: np.ndarray):
    """Fill non-finite pixels with the nearest finite value.

    Returns ``(filled, valid_mask)`` where ``valid_mask`` is True for pixels
    that held a genuine observation.  Raises if there is nothing to copy from.
    """
    valid = np.isfinite(arr)
    if valid.all():
        return arr.astype(np.float32, copy=True), valid
    if not valid.any():
        raise ValueError("array has no finite pixels to fill from")
    idx = ndimage.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    filled = arr[tuple(idx)].astype(np.float32)
    return filled, valid
