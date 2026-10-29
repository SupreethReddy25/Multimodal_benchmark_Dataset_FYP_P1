"""Crater-based cross-layer alignment check (independent of the NCC shift check).

For craters of the Robbins (2019) global database that lie fully inside a
region, the centre of each crater is estimated *separately in each layer* as
the position that maximises radial symmetry of that layer around the crater
(fraction of variance in a window of 1.3 radii explained by the azimuthal mean
profile).  The per-crater offsets  optical - dem  and  thermal - dem  (in
pixels) are then summarised over all craters.  A systematic registration error
between layers would show up as a non-zero median offset; random scatter only
widens the confidence interval.

This does not use the slope/gradient statistic of registration_check.py and
does not depend on the catalogue positions being exact: the catalogue only
provides the approximate crater centre and radius used to place the window.
Craters are asymmetric (ejecta, illumination), so single-crater offsets are
noisy and only the median over many craters is interpretable.

Usage
-----
    python -m src.experiments.crater_alignment
"""

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from ..pipeline.config import list_regions, load_region

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "experiments"
CAT = ROOT / "data" / "external" / "robbins" / "lunar_crater_database_robbins_2018_bundle" / "data" / "lunar_crater_database_robbins_2018.csv"

PPD = 128
KM_PER_PX = 2 * np.pi * 1737.4 / 360.0 / PPD
SEARCH = 6                  # +- px
D_MIN, D_MAX = 2.0, 30.0    # km
WINDOW = 1.3                # radii
MIN_SYMMETRY = 0.25         # per-layer reliability threshold
EDGE_MARGIN = 2             # reject peaks within this distance of the search edge


def _load_catalogue():
    cols = ["CRATER_ID", "LAT_CIRC_IMG", "LON_CIRC_IMG", "DIAM_CIRC_IMG"]
    df = pd.read_csv(CAT, usecols=cols)
    return df[(df.DIAM_CIRC_IMG >= D_MIN) & (df.DIAM_CIRC_IMG <= D_MAX)]


def _symmetry(img, valid, cy, cx, ry, rx):
    """Fraction of variance explained by the azimuthal mean profile."""
    h, w = img.shape
    y0, y1 = int(max(0, cy - WINDOW * ry - 1)), int(min(h, cy + WINDOW * ry + 2))
    x0, x1 = int(max(0, cx - WINDOW * rx - 1)), int(min(w, cx + WINDOW * rx + 2))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rho = np.hypot((yy - cy) / ry, (xx - cx) / rx)
    sel = (rho <= WINDOW) & valid[y0:y1, x0:x1]
    if sel.sum() < 0.8 * np.pi * ry * rx * WINDOW ** 2:
        return np.nan
    v = img[y0:y1, x0:x1][sel].astype(np.float64)
    b = np.minimum((rho[sel] / WINDOW * 40).astype(int), 39)
    cnt = np.bincount(b, minlength=40)
    mean = np.bincount(b, weights=v, minlength=40) / np.maximum(cnt, 1)
    tot = ((v - v.mean()) ** 2).sum()
    if tot <= 0:
        return np.nan
    res = ((v - mean[b]) ** 2).sum()
    return 1.0 - res / tot


def _best_centre(img, valid, cy, cx, ry, rx):
    s = np.full((2 * SEARCH + 1, 2 * SEARCH + 1), np.nan)
    for dy in range(-SEARCH, SEARCH + 1):
        for dx in range(-SEARCH, SEARCH + 1):
            s[dy + SEARCH, dx + SEARCH] = _symmetry(img, valid, cy + dy, cx + dx, ry, rx)
    if np.isnan(s).any():
        return None
    iy, ix = np.unravel_index(np.argmax(s), s.shape)
    if min(iy, ix, 2 * SEARCH - iy, 2 * SEARCH - ix) < EDGE_MARGIN:
        return None
    if s[iy, ix] < MIN_SYMMETRY:
        return None
    oy, ox = float(iy - SEARCH), float(ix - SEARCH)
    # parabolic sub-pixel refinement
    for axis in (0, 1):
        line = s[:, ix] if axis == 0 else s[iy, :]
        k = iy if axis == 0 else ix
        den = line[k - 1] - 2 * line[k] + line[k + 1]
        if den < 0:
            d = 0.5 * (line[k - 1] - line[k + 1]) / den
            if abs(d) <= 1:
                if axis == 0:
                    oy += d
                else:
                    ox += d
    return oy, ox, float(s[iy, ix])


def region_craters(rid, cat):
    r = load_region(rid)
    i = r["intermediate_dir"]
    dem = np.load(i / "dem" / "lola_dem_common_grid.npy")
    dem_ok = np.load(i / "dem" / "lola_dem_common_valid_mask.npy").astype(bool)
    opt = np.load(i / "optical" / "wac_common_grid.npy")
    cov = np.load(i / "optical" / "wac_common_coverage.npy") >= 0.99
    # gradient magnitude of the optical layer: rim edges appear on both sides of
    # a crater whatever the sun direction, unlike the raw brightness
    g = ndimage.gaussian_filter(opt.astype(np.float64), 1.0)
    gm = np.hypot(*np.gradient(g))
    th = np.load(i / "thermal" / "diviner_tbol_sam_common_grid.npy")
    th_ok = np.load(i / "thermal" / "diviner_tbol_sam_common_valid_mask.npy").astype(bool)
    lat_min, lat_max = r["lat_min"], r["lat_max"]
    lon_min, lon_max = r["lon_min"], r["lon_max"]
    sub = cat[(cat.LAT_CIRC_IMG > lat_min) & (cat.LAT_CIRC_IMG < lat_max) &
              (cat.LON_CIRC_IMG > lon_min) & (cat.LON_CIRC_IMG < lon_max)]
    rows = []
    for c in sub.itertuples():
        cy = (lat_max - c.LAT_CIRC_IMG) * PPD - 0.5
        cx = (c.LON_CIRC_IMG - lon_min) * PPD - 0.5
        R = c.DIAM_CIRC_IMG / 2.0
        ry = R / KM_PER_PX
        rx = ry / np.cos(np.radians(c.LAT_CIRC_IMG))
        # whole search window must fit inside the region
        if (cy - WINDOW * ry - SEARCH - 2 < 0 or cy + WINDOW * ry + SEARCH + 2 > dem.shape[0]
                or cx - WINDOW * rx - SEARCH - 2 < 0 or cx + WINDOW * rx + SEARCH + 2 > dem.shape[1]):
            continue
        d = _best_centre(dem, dem_ok, cy, cx, ry, rx)
        if d is None:
            continue
        o = _best_centre(opt, cov, cy, cx, ry, rx)
        t = _best_centre(th, th_ok, cy, cx, ry, rx)
        og = _best_centre(gm, cov, cy, cx, ry, rx)
        rows.append({
            "region": rid, "crater_id": c.CRATER_ID, "diam_km": float(c.DIAM_CIRC_IMG),
            "dem_dy": d[0], "dem_dx": d[1], "dem_sym": d[2],
            "opt_dy": o[0] if o else np.nan, "opt_dx": o[1] if o else np.nan,
            "opt_sym": o[2] if o else np.nan,
            "th_dy": t[0] if t else np.nan, "th_dx": t[1] if t else np.nan,
            "th_sym": t[2] if t else np.nan,
            "og_dy": og[0] if og else np.nan, "og_dx": og[1] if og else np.nan,
            "og_sym": og[2] if og else np.nan,
        })
    return rows


def _summ(x, rng):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return {"n": int(len(x)), "median": None, "ci95": None}
    boots = [np.median(rng.choice(x, len(x))) for _ in range(2000)]
    return {"n": int(len(x)), "median": float(np.median(x)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cat = _load_catalogue()
    rows = []
    for rid in list_regions():
        try:
            got = region_craters(rid, cat)
        except FileNotFoundError:
            continue
        rows.extend(got)
        print(f"{rid:<18} craters used: {len(got)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "crater_alignment_craters.csv", index=False)
    rng = np.random.default_rng(0)
    res = {"n_regions_with_craters": int(df.region.nunique()), "n_craters_dem": int(len(df)),
           "px_m": 236.9, "criteria": {"D_km": [D_MIN, D_MAX], "search_px": SEARCH,
                                       "min_symmetry": MIN_SYMMETRY}}
    for name, a, b in (("optical_minus_dem", "opt", "dem"), ("optical_gradient_minus_dem", "og", "dem"),
                         ("thermal_minus_dem", "th", "dem")):
        res[name] = {"dy": _summ(df[f"{a}_dy"] - df[f"{b}_dy"], rng),
                     "dx": _summ(df[f"{a}_dx"] - df[f"{b}_dx"], rng)}
        for ax in ("dy", "dx"):
            v = (df[f"{a}_{ax}"] - df[f"{b}_{ax}"]).dropna()
            res[name][ax]["mad_px"] = float(np.median(np.abs(v - v.median()))) if len(v) else None
        # per-region medians, to see whether any single region is off
        per = df.assign(dy=df[f"{a}_dy"] - df[f"{b}_dy"], dx=df[f"{a}_dx"] - df[f"{b}_dx"]) \
                .dropna(subset=["dy", "dx"]).groupby("region").agg(n=("dy", "size"), dy=("dy", "median"), dx=("dx", "median"))
        per.to_csv(OUT / f"crater_alignment_{name}_per_region.csv")
    with open(OUT / "crater_alignment.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
