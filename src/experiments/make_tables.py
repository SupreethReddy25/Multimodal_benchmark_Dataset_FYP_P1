"""Turn outputs/experiments/*.json into CSV + LaTeX tables for the paper.

Every number in the manuscript tables comes from here; nothing is typed by hand.

Writes (outputs/tables/ and paper/tables/):
  baseline_summary.csv / baseline_<task>.tex   mean +- std over seeds
  per_region_<task>.csv                        best-by-validation-free reporting:
                                               all models, per test region
  registration_check.csv / .tex

Usage
-----
    python -m src.experiments.make_tables
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "outputs" / "experiments"
TAB = ROOT / "outputs" / "tables"
PTAB = ROOT / "paper" / "tables"
METRICS = ["rmse", "mae", "r2", "anomaly_r2"]


def _fmt(m, s):
    return f"{m:.3f}" if (s is None or np.isnan(s) or s == 0) else f"{m:.3f} $\\pm$ {s:.3f}"


def baseline_tables():
    data = json.loads((EXP / "baseline_results.json").read_text(encoding="utf-8"))
    rows = []
    for r in data["results"]:
        rows.append({"task": r["task"], "variant": r["variant"],
                     "model": r["model"], "seed": r["seed"], **r["mean"]})
    df = pd.DataFrame(rows)
    g = df.groupby(["task", "variant", "model"], sort=False)
    mean, std, n = g[METRICS].mean(), g[METRICS].std(ddof=0), g.size()
    summ = mean.add_suffix("_mean").join(std.add_suffix("_std")).join(n.rename("n_seeds"))
    summ = summ.reset_index()
    summ.to_csv(TAB / "baseline_summary.csv", index=False)

    for task, sub in summ.groupby("task", sort=False):
        out = pd.DataFrame({
            "Inputs": sub["variant"].str.replace("_", r"\_"),
            "Model": sub["model"].str.replace("_", r"\_"),
            "RMSE": [_fmt(a, b) for a, b in zip(sub.rmse_mean, sub.rmse_std)],
            "MAE": [_fmt(a, b) for a, b in zip(sub.mae_mean, sub.mae_std)],
            "R2": [_fmt(a, b) for a, b in zip(sub.r2_mean, sub.r2_std)],
            "Anomaly R2": [_fmt(a, b) for a, b in
                           zip(sub.anomaly_r2_mean, sub.anomaly_r2_std)],
        })
        (PTAB / f"baseline_{task}.tex").write_text(
            out.to_latex(index=False, escape=False), encoding="utf-8")

        # per-region detail (seed-averaged)
        recs = []
        for r in data["results"]:
            if r["task"] != task:
                continue
            for rid, m in r["per_region"].items():
                recs.append({"variant": r["variant"], "model": r["model"],
                             "region": rid, **m})
        pr = (pd.DataFrame(recs).groupby(["variant", "model", "region"], sort=False)
              [METRICS].mean().reset_index())
        pr.to_csv(TAB / f"per_region_{task}.csv", index=False)
    return summ


def registration_table():
    p = EXP / "registration_check.json"
    if not p.exists():
        return
    df = pd.DataFrame(json.loads(p.read_text(encoding="utf-8")))
    df["best_dy"] = df.best_shift_px.map(lambda v: v[0])
    df["best_dx"] = df.best_shift_px.map(lambda v: v[1])
    keep = df[["region", "ncc_zero_shift", "ncc_best", "best_dy", "best_dx",
               "best_shift_m", "zero_rank_of_121"]]
    keep.to_csv(TAB / "registration_check.csv", index=False)
    (PTAB / "registration_check.tex").write_text(
        keep.to_latex(index=False, float_format="%.2f", escape=True),
        encoding="utf-8")


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    PTAB.mkdir(parents=True, exist_ok=True)
    summ = baseline_tables()
    registration_table()
    print(summ[["task", "variant", "model", "n_seeds", "rmse_mean", "rmse_std",
                "anomaly_r2_mean", "anomaly_r2_std"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
