"""Dataset statistics tables and figures for the paper.

Writes:
  outputs/tables/region_statistics.csv            (+ paper/tables/*.tex)
  outputs/figures/region_map.png                  (geographic coverage)
  outputs/figures/region_montage.png              (optical | hillshade | thermal)

Usage
-----
    python -m src.experiments.dataset_stats
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..pipeline.config import list_regions, load_region
from .data import load_region_tiles, load_splits, valid_mask

ROOT = Path(__file__).resolve().parents[2]


def _side(lon_c: float) -> str:
    return "nearside" if (lon_c <= 90 or lon_c >= 270) else "farside"


def _stitch(a):   # (16,64,64) -> (256,256)
    return a.reshape(4, 4, 64, 64).transpose(0, 2, 1, 3).reshape(256, 256)


def build_table() -> pd.DataFrame:
    split_of = {}
    for k, v in load_splits().items():
        if isinstance(v, list):
            for rid in v:
                split_of[rid] = k
    rows = []
    for rid in list_regions():
        r = load_region(rid)
        t = load_region_tiles(rid)
        m = valid_mask(t)
        dem, sl, th = t["dem"], t["slope"], t["thermal"]
        lat_c = (r["lat_min"] + r["lat_max"]) / 2
        lon_c = (r["lon_min"] + r["lon_max"]) / 2
        rows.append({
            "region": rid, "name": r["name"], "terrain": r.get("terrain", ""),
            "lat_c": lat_c, "lon_c": lon_c,
            "hemisphere": "N" if lat_c >= 0 else "S", "side": _side(lon_c),
            "split": split_of.get(rid, ""),
            "elev_mean_km": float(dem.mean()),
            "relief_km": float(np.percentile(dem, 99) - np.percentile(dem, 1)),
            "slope_mean_deg": float(sl.mean()),
            "slope_p95_deg": float(np.percentile(sl, 95)),
            "optical_mean": float(t["optical"][m].mean()),
            "optical_std": float(t["optical"][m].std()),
            "optical_cov": float(t["optical_coverage"].mean()),
            "thermal_mean_K": float(th.mean()),
            "thermal_std_K": float(th.std()),
            "thermal_min_K": float(th.min()), "thermal_max_K": float(th.max()),
            "valid_frac": float(m.mean()),
        })
    return pd.DataFrame(rows)


def region_map(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = {"train": "#2a6fb0", "val": "#e0a21b", "test": "#c0392b"}
    for _, r in df.iterrows():
        ax.add_patch(plt.Rectangle((r.lon_c - 1, r.lat_c - 1), 2, 2,
                                   color=colors.get(r.split, "gray")))
        ax.annotate(r.region, (r.lon_c, r.lat_c + 2.2), fontsize=6,
                    ha="center", rotation=35)
    ax.set_xlim(0, 360); ax.set_ylim(-70, 70)
    ax.set_xlabel("Longitude (deg E)"); ax.set_ylabel("Latitude (deg)")
    ax.set_title("Benchmark regions (2 deg x 2 deg, boxes enlarged for visibility)"
                 " - blue: train, amber: val, red: test")
    ax.axhline(0, lw=0.4, color="k"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def montage(df: pd.DataFrame, path: Path) -> None:
    n = len(df)
    fig, axes = plt.subplots(n, 4, figsize=(8, 2 * n))
    for i, rid in enumerate(df.region):
        t = load_region_tiles(rid)
        panels = [("optical", _stitch(t["optical"]), "gray"),
                  ("dem", _stitch(t["dem"]), "terrain"),
                  ("hillshade", _stitch(t["hillshade"]), "gray"),
                  ("thermal", _stitch(t["thermal"]), "inferno")]
        for j, (name, arr, cm) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(arr, cmap=cm, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(name, fontsize=8)
            if j == 0:
                ax.set_ylabel(rid, fontsize=6, rotation=0, ha="right", va="center")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    (ROOT / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "outputs" / "figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "paper" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "paper" / "figures").mkdir(parents=True, exist_ok=True)

    df = build_table()
    df.to_csv(ROOT / "outputs" / "tables" / "region_statistics.csv", index=False)

    cols = ["region", "terrain", "lat_c", "lon_c", "side", "split",
            "relief_km", "slope_mean_deg", "thermal_mean_K", "optical_cov"]
    tex = df[cols].to_latex(index=False, float_format="%.2f", escape=True)
    (ROOT / "paper" / "tables" / "region_statistics.tex").write_text(tex, encoding="utf-8")

    for sub in ("outputs/figures", "paper/figures"):
        region_map(df, ROOT / sub / "region_map.png")
    montage(df, ROOT / "outputs" / "figures" / "region_montage.png")
    montage(df, ROOT / "paper" / "figures" / "region_montage.png")

    print(df[["region", "side", "hemisphere", "split", "relief_km",
              "slope_mean_deg", "thermal_mean_K", "optical_cov"]]
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
