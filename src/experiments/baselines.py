"""Benchmark tasks and baselines.

Tasks (target <- inputs), all evaluated on held-out REGIONS:
  thermal : Diviner midnight bolometric temperature <- optical and/or terrain
  slope   : LOLA-derived slope                       <- optical only

Baselines: train mean, latitude-only (thermal), Ridge, HistGradientBoosting
(multi-scale pixel features), and a small CNN on 64x64 tiles.

Metrics (per test region, then averaged over test regions):
  RMSE / MAE in target units, R^2, and 'anomaly R^2' -- R^2 after removing each
  tile's mean from prediction and target, i.e. how well the within-tile spatial
  structure is recovered independent of the large latitude/region offset.

Usage
-----
    python -m src.experiments.baselines            # official split, 3 seeds
    python -m src.experiments.baselines --logo     # + leave-one-region-out
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ..pipeline.config import list_regions
from .data import (VARIANTS, load_region_tiles, load_splits, pixel_features,
                   valid_mask)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "experiments"
MAX_TRAIN_PIX = 400_000

TASKS = {
    "thermal": {"target": "thermal", "unit": "K",
                "variants": ["optical", "terrain", "optical+terrain"]},
    "slope": {"target": "slope", "unit": "deg", "variants": ["optical"]},
}


# --------------------------------------------------------------------------
# data assembly
# --------------------------------------------------------------------------

_CACHE = {}


def region_data(rid: str, target: str) -> dict:
    key = (rid, target)
    if key not in _CACHE:
        t = load_region_tiles(rid)
        _CACHE[key] = {
            "feats": pixel_features(t, use_lat=True),
            "y": t[target].astype(np.float32),
            "mask": valid_mask(t),
        }
    return _CACHE[key]


def stack_regions(regions, target, names):
    X, Y, M = [], [], []
    for rid in regions:
        d = region_data(rid, target)
        X.append(np.stack([d["feats"][n] for n in names], axis=-1))
        Y.append(d["y"])
        M.append(d["mask"])
    return np.concatenate(X), np.concatenate(Y), np.concatenate(M)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _tile_mean(a, m):
    return (np.where(m, a, 0).sum((1, 2), keepdims=True)
            / np.maximum(m.sum((1, 2), keepdims=True), 1))


def metrics(pred, y, mask):
    p, t = pred[mask], y[mask]
    err = p - t
    ss_tot = ((t - t.mean()) ** 2).sum()
    pa, ya = pred - _tile_mean(pred, mask), y - _tile_mean(y, mask)
    ea, ta = (pa - ya)[mask], ya[mask]
    return {
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mae": float(np.abs(err).mean()),
        "r2": float(1 - (err ** 2).sum() / max(ss_tot, 1e-12)),
        "anomaly_r2": float(1 - (ea ** 2).sum() / max((ta ** 2).sum(), 1e-12)),
    }


# --------------------------------------------------------------------------
# pixel models
# --------------------------------------------------------------------------

class MeanModel:
    def fit(self, X, y):
        self.m = float(y.mean())
        return self

    def predict(self, X):
        return np.full(len(X), self.m, dtype=np.float32)


def make_sklearn(name, seed):
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "latitude":
        return make_pipeline(PolynomialFeatures(2), StandardScaler(),
                             Ridge(alpha=1e-3))
    if name == "hist_gb":
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,
                                             random_state=seed)
    if name == "mean":
        return MeanModel()
    raise ValueError(name)


def run_pixel_model(name, names, train_regs, test_regs, target, seed):
    use = ["lat"] if name == "latitude" else list(names)
    Xtr, ytr, mtr = stack_regions(train_regs, target, use)
    Xf, yf = Xtr[mtr], ytr[mtr]
    if len(Xf) > MAX_TRAIN_PIX:
        sel = np.random.default_rng(seed).choice(len(Xf), MAX_TRAIN_PIX,
                                                 replace=False)
        Xf, yf = Xf[sel], yf[sel]
    model = make_sklearn(name, seed).fit(Xf, yf)
    res = {}
    for rid in test_regs:
        X, y, m = stack_regions([rid], target, use)
        pred = model.predict(X.reshape(-1, X.shape[-1])).reshape(y.shape)
        res[rid] = metrics(pred, y, m)
    return res


# --------------------------------------------------------------------------
# CNN
# --------------------------------------------------------------------------

class TinyCNN(nn.Module):
    def __init__(self, cin, width=32):
        super().__init__()

        def blk(i, o, d):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=d, dilation=d),
                                 nn.BatchNorm2d(o), nn.ReLU())

        self.net = nn.Sequential(blk(cin, width, 1), blk(width, width, 2),
                                 blk(width, width, 4), blk(width, width, 8),
                                 nn.Conv2d(width, 1, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def run_cnn(names, train_regs, val_regs, test_regs, target, seed, epochs=80):
    torch.manual_seed(seed)
    Xtr, ytr, mtr = stack_regions(train_regs, target, names)
    Xva, yva, mva = stack_regions(val_regs, target, names)
    mu, sd = Xtr[mtr].mean(0), Xtr[mtr].std(0) + 1e-6
    ymu, ysd = ytr[mtr].mean(), ytr[mtr].std() + 1e-6

    def prep(X, y, m):
        Xt = torch.from_numpy(((X - mu) / sd).astype(np.float32)).permute(0, 3, 1, 2)
        yt = torch.from_numpy(((y - ymu) / ysd).astype(np.float32))
        return Xt, yt, torch.from_numpy(m)

    Xt, yt, mt = prep(Xtr, ytr, mtr)
    Xv, yv, mv = prep(Xva, yva, mva)
    model = TinyCNN(Xt.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 16):
            b = perm[i:i + 16]
            opt.zero_grad()
            loss = ((model(Xt[b]) - yt[b]) ** 2)[mt[b]].mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(((model(Xv) - yv) ** 2)[mv].mean())
        if vl < best:   # model selection on the validation regions only
            best = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    res = {}
    for rid in test_regs:
        X, y, m = stack_regions([rid], target, names)
        Xte, _, _ = prep(X, y, m)
        with torch.no_grad():
            pred = model(Xte).numpy() * ysd + ymu
        res[rid] = metrics(pred, y, m)
    return res


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------

def mean_over_regions(per_region):
    keys = next(iter(per_region.values())).keys()
    return {k: float(np.mean([v[k] for v in per_region.values()])) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--logo", action="store_true",
                    help="also run leave-one-region-out for hist_gb")
    ap.add_argument("--no-cnn", action="store_true")
    args = ap.parse_args()

    sp = load_splits()
    train, val, test = sp["train"], sp["val"], sp["test"]
    OUT.mkdir(parents=True, exist_ok=True)
    results = []

    def add(task, variant, model, seed, per_region):
        m = mean_over_regions(per_region)
        results.append({"task": task, "variant": variant, "model": model,
                        "seed": seed, "per_region": per_region, "mean": m})
        print(f"  {task:<8}{variant:<22}{model:<9}seed={seed}  "
              f"RMSE={m['rmse']:.3f}  R2={m['r2']:.3f}  "
              f"anomR2={m['anomaly_r2']:.3f}", flush=True)

    for task, spec in TASKS.items():
        tgt = spec["target"]
        print(f"\n== task: {task} ({spec['unit']})")
        add(task, "-", "mean", 0,
            run_pixel_model("mean", ["optical"], train + val, test, tgt, 0))
        if task == "thermal":
            add(task, "latitude only", "latitude", 0,
                run_pixel_model("latitude", [], train + val, test, tgt, 0))
        for variant in spec["variants"]:
            names = VARIANTS[variant]
            add(task, variant, "ridge", 0,
                run_pixel_model("ridge", names, train + val, test, tgt, 0))
            for seed in range(args.seeds):
                add(task, variant, "hist_gb", seed,
                    run_pixel_model("hist_gb", names, train + val, test, tgt, seed))
                if not args.no_cnn:
                    add(task, variant, "cnn", seed,
                        run_cnn(names, train, val, test, tgt, seed))

    if args.logo:
        print("\n== leave-one-region-out (hist_gb, last variant per task)")
        regs = list_regions()
        for task, spec in TASKS.items():
            variant = spec["variants"][-1]
            per = {}
            for rid in regs:
                per.update(run_pixel_model("hist_gb", VARIANTS[variant],
                                           [r for r in regs if r != rid], [rid],
                                           spec["target"], 0))
            add(task, variant + " [LORO]", "hist_gb", 0, per)
            per_mean = {}
            for rid in regs:
                per_mean.update(run_pixel_model(
                    "mean", ["optical"], [r for r in regs if r != rid], [rid],
                    spec["target"], 0))
            add(task, variant + " [LORO]", "mean", 0, per_mean)

    with open(OUT / "baseline_results.json", "w", encoding="utf-8") as fh:
        json.dump({"split": sp, "results": results}, fh, indent=2)
    print(f"\nSaved {OUT / 'baseline_results.json'}")


if __name__ == "__main__":
    main()
