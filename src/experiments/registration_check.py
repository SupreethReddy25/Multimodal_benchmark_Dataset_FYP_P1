"""Per-region cross-modal registration check.

Both WAC and LOLA are georeferenced and are mapped to the common grid with
exact pixel-centre geometry (verified in the pipeline).  This script gives an
*empirical* complement: it asks whether the optical and terrain layers are
most consistent at zero shift, using the same idea as the pilot's
cross_modal_shift_validation.py (shift one layer by -5..+5 px and look at the
agreement) but for every region.

Agreement measure : normalized cross-correlation (NCC) between the gradient
                    magnitude of the optical layer and the LOLA slope.
Reported per region: NCC at zero shift, best shift (dy, dx) in pixels and its
                    ground distance, and the zero-shift NCC as a fraction of
                    the maximum.

Caveat (stated in the paper): albedo/topography correlation is weak on mare
plains, so a flat NCC surface is uninformative there; the table flags regions
where the zero-shift value is not distinguishable from the best shift.

Usage
-----
    python -m src.experiments.registration_check
"""

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..pipeline.config import list_regions, load_region

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "experiments"
MAX_SHIFT = 5


def _grad_mag(a):
    a = ndimage.gaussian_filter(a.astype(np.float64), 1.0)
    gy, gx = np.gradient(a)
    return np.hypot(gx, gy)


def _ncc(a, b, m):
    a, b = a[m], b[m]
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _shift_overlap(ref, mov, dy, dx):
    """Overlap of ref with mov shifted by (dy, dx) (down/right positive)."""
    h, w = ref.shape
    ys, ye = max(dy, 0), h + min(dy, 0)
    xs, xe = max(dx, 0), w + min(dx, 0)
    return (ref[ys:ye, xs:xe],
            mov[ys - dy:ye - dy, xs - dx:xe - dx])


def check_region(rid: str) -> dict:
    r = load_region(rid)
    i = r["intermediate_dir"]
    opt = np.load(i / "optical" / "wac_common_grid.npy")
    cov = np.load(i / "optical" / "wac_common_coverage.npy")
    slope = np.load(i / "dem" / "lola_common_slope_deg.npy")

    g_opt = _grad_mag(opt)
    valid = ndimage.binary_erosion(cov >= 0.99, iterations=3)

    surf = np.zeros((2 * MAX_SHIFT + 1, 2 * MAX_SHIFT + 1))
    for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):
        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):
            a, b = _shift_overlap(g_opt, slope, dy, dx)
            m_ref, m_mov = _shift_overlap(valid, valid, dy, dx)
            surf[dy + MAX_SHIFT, dx + MAX_SHIFT] = _ncc(a, b, m_ref & m_mov)

    zero = surf[MAX_SHIFT, MAX_SHIFT]
    iy, ix = np.unravel_index(np.argmax(surf), surf.shape)
    dy, dx = int(iy - MAX_SHIFT), int(ix - MAX_SHIFT)
    px_m = r["resolution_m"]
    return {
        "region": rid,
        "ncc_zero_shift": float(zero),
        "ncc_best": float(surf.max()),
        "best_shift_px": [dy, dx],
        "best_shift_m": float(np.hypot(dy, dx) * px_m),
        "zero_over_best": float(zero / surf.max()) if surf.max() > 0 else None,
        "zero_rank_of_121": int((surf > zero).sum() + 1),
        "optical_valid_fraction": float(valid.mean()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    print(f"{'region':<18}{'NCC@0':>8}{'NCCbest':>9}{'shift(dy,dx)':>14}"
          f"{'rank/121':>10}")
    for rid in list_regions():
        try:
            row = check_region(rid)
        except FileNotFoundError:
            print(f"{rid:<18} (not processed)")
            continue
        rows.append(row)
        print(f"{rid:<18}{row['ncc_zero_shift']:>8.3f}{row['ncc_best']:>9.3f}"
              f"{str(tuple(row['best_shift_px'])):>14}{row['zero_rank_of_121']:>10}")
    with open(OUT / "registration_check.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nSaved {OUT / 'registration_check.json'}")


if __name__ == "__main__":
    main()
