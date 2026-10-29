# PROJECT GUIDE — read this first

Audience: the project owner and teammates. It explains what the project is, how to
learn it, how to read the results, what changed from the original pilot, and what is
left before submission. Everything stated here is checked against files in this
repository; where something is a judgement or an open item it is marked as such.

---

## 1. One-paragraph summary

We built a **benchmark dataset for the Moon**: for 28 small areas (2° × 2°, about
60 km) three NASA LRO measurements are placed on one pixel grid — **optical** (LROC
WAC), **elevation** (LOLA), **thermal** (Diviner, local midnight) — plus derived
slope and hillshade, validity masks, and a tile-level day/night temperature curve.
Evaluation is by held-out region. We include simple baselines (train mean, latitude
only, ridge, gradient boosting, small CNN) only as **reference points**. The paper's
contribution is the aligned dataset, its documented quality, and the benchmark
protocol — **not** a modelling result.

## 2. Reading order (how to learn the project in ~2 hours)

1. This file, sections 3–6 (15 min).
2. `DATASET_CARD.md` — formal description and limitations (10 min). Read before writing anything.
3. Look at `paper/figures/region_montage.png` and `region_map.png` (10 min).
4. `configs/regions.yaml` and `configs/splits.yaml` — the 28 regions and the split (10 min).
5. Run the loading snippet in section 7 and plot one tile (20 min).
6. `paper/manuscript/main.tex` (open in Overleaf) — the draft (30 min).
7. `src/pipeline/` in this order: `config.py` → `run_pipeline.py` → `harmonize.py` → `derive_terrain.py` → `create_tiles.py` → `validate.py`.
8. `src/experiments/` : `data.py` → `baselines.py` → `registration_check.py` → `crater_alignment.py`.

## 3. What is in the dataset

Per region (`data/benchmark/<region>/`):

| Item | What it is | Unit / shape |
|:--|:--|:--|
| `tiles/tile_RR_CC/optical.npy` | LROC WAC brightness | 0–255, 64×64 |
| `.../dem.npy` | LOLA elevation relative to the 1737.4 km sphere | km |
| `.../thermal.npy` | Diviner bolometric temperature, local **midnight** (slope-adjusted) | K |
| `.../slope.npy` | steepness from the DEM (central differences, `numpy.gradient`) | degrees |
| `.../hillshade.npy` | shaded relief, sun azimuth 315°, altitude 45° | 0–1 |
| `.../*_coverage.npy`, `*_valid_mask.npy` | which pixels are real data | 0–1 / bool |
| `.../metadata.json` | exact lat/lon extent of the tile | — |
| `diurnal_tbol.npz` | `tbol[16, 96]` temperature vs local time per tile, `local_time[96]` | K, hours |
| `metadata/tile_index.json` | index of the 16 tiles | — |

* Common grid: equirectangular, radius 1,737,400 m, 128 px/deg (≈ 236.9 m), 256 × 256 px
  per region, 16 tiles of 64 × 64. 28 regions → **448 tiles**.
* Latitudes are limited to ±60° (the WAC tiles used).
* **Always apply the masks** (`valid_mask` in `src/experiments/data.py`).
* `diurnal_tbol.npz` is **tile-level only** (one 0.5° bin = one tile, ≈ 15 km). It is
  not pixel-aligned and no baseline uses it.

Splits (`configs/splits.yaml`, by region, never by tile):
train 17 · val 3 (hyginus, endymion, plato) · test 8 (taurus_littrow, mare_crisium,
tycho, tsiolkovsky, copernicus, ingenii, moscoviense, korolev).

## 4. How to interpret the results

All numbers below are on the **8 held-out test regions** (`outputs/tables/`, `paper/tables/`).

| Task | Train mean | Best baseline | Reading |
|:--|:--|:--|:--|
| Thermal (midnight, K RMSE) | 7.59 | optical CNN 5.78 ± 0.11 | optical helps; terrain alone ≈ mean (7.69); optical+terrain 6.60 |
| Slope from optical (° RMSE) | 6.77 | CNN 4.21 ± 0.11 | learnable; anomaly R² 0.43 |

Metrics:
* **RMSE / MAE** — absolute error, computed per region then averaged.
* **R²** — dominated by between-region offsets; can be very negative. Do not headline it.
* **Anomaly R²** — remove each tile's mean from prediction and target, then R². It asks
  "is the *pattern inside a tile* right?". **Use this to judge spatial structure.**

Key honest findings (these go in the paper as they are):
1. Thermal: **no learned model has positive anomaly R²** on unseen regions. They predict
   regional offsets but not within-tile structure. This is the benchmark's open problem.
2. The optical CNN is **worse than the train mean on 5 of 8 test regions** (the
   low-variance ones: mare_crisium, taurus_littrow, ingenii, moscoviense, korolev).
3. Adding terrain to optical does not help thermal prediction.
4. Leave-one-region-out (LORO) thermal gain is only ~3 % (6.27 K vs 6.49 K).
5. Seeds: 3 for CNN/boosting; seed std ≈ 0.1 K, so differences below that are noise.
   Only 8 test regions: read per-region results (`outputs/tables/per_region_*.csv`).

Alignment evidence (two independent checks; both are *consistency checks*, not proofs):
* Gradient/slope NCC shift test (`registration_check.py`): best shift ≤ 3 px in 25/28
  regions; dx sign mixed (16 +, 10 −, 2 zero); zero shift best in only 1 region;
  weakest region `proclus`.
* Crater-symmetry test (`crater_alignment.py`, 106 craters, 23 regions, Robbins 2019
  catalogue): thermal − elevation median offset ≈ 0.0 px (95 % CI within ±0.4 px);
  optical-gradient − elevation within ≈ 1 px. Raw optical brightness is biased by the
  fixed sun direction; the gradient variant was added *after* seeing that bias (stated in the paper).

Midnight vs the diurnal curve: the cumulative-product midnight value matches our
thermal tile means with correlation 0.993, RMSE 1.15 K (product 0.73 K warmer; it is
not slope-adjusted and covers a different period).

## 5. What changed from the original project

The original (git commit `b11f14b`, "complete multimodal lunar benchmark pipeline…") was
a **single-region pilot** (Lacus Mortis – Bürg, 16 tiles) with pilot scripts, a README
and `Notes.txt`; no paper, no baselines, no tests, no splits.

| Area | Original | Now |
|:--|:--|:--|
| Regions | 1 (Lacus Mortis) | **28** in a registry (`configs/regions.yaml`), terrain-diverse, both hemispheres, near/far side |
| Region validity | n/a | Boxes checked against published feature coordinates (Endymion box is off-centre, documented) |
| Code | one-off scripts per step (`src/acquisition … validation`) | region-aware `src/pipeline/` + `run_pipeline.py` (`--region`, `--all`, `--list`); resumable downloads, atomic writes, non-zero exit codes on failure |
| Data access | LOLA/Diviner pilot windows | LOLA and Diviner **HTTP range-streamed** (no 4 GB download); 5 WAC tiles cached |
| Gaps | — | nearest-filled and flagged in masks; coverage layers for WAC/LOLA/Diviner |
| Slope | docs said "Horn"; code was `numpy.gradient` | **documentation corrected** to match code (central differences, latitude-corrected spacing). Method unchanged |
| Splits | none | region-level train/val/test (`configs/splits.yaml`) |
| Evaluation | none | `src/experiments/`: tasks, metrics incl. anomaly R², 5 baselines, LORO |
| Alignment | pilot-only shift/MI scripts | per-region NCC shift check **plus** crater-symmetry check |
| Thermal | midnight only | midnight pixel map **plus** tile-level diurnal curve (documented as coarse) |
| Docs | README, Notes | `DATASET_CARD.md`, this guide, corrected README |
| Paper | none | `paper/manuscript/main.tex`, `refs.bib` (all 11 references verified against Crossref/DataCite), generated tables/figures |
| Tests | none | `tests/` (6 offline tests) |

Things we deliberately did **not** do: model tuning/development; a second pixel-level
Diviner local time (the other single-band PDS maps are all midnight-derived); Horn's
slope (kept consistent with the pilot); claims of sub-pixel registration.

### Cleanup record (this finalisation)
Removed because they were byte-identical duplicates of `lacus_mortis`, regenerable, or
caches — all pilot files are also in git commit `b11f14b`
(`git show b11f14b:<path>` restores any of them):
* `data/intermediate/{dem,optical,thermal,registration}` (pilot copies; arrays identical to `data/intermediate/lacus_mortis`; 3 native-resolution pilot-only derivatives were regenerable)
* `data/benchmark/{tiles,masks,terrain,metadata}` (pilot copy of lacus_mortis tiles)
* `outputs/validation/` (pilot QC figures; per-region QC lives in `outputs/<region>/validation/`)
* `data/external/robbins/robbins.zip` (CSV kept; re-downloadable from USGS Astropedia)
* `__pycache__`, `.pytest_cache`, `TEAM_GUIDE.md` (merged into this file)

Deliberately **kept**: `data/raw/optical` (3.3 GB WAC tiles, needed to rebuild; ignored by git),
`data/intermediate/<region>` (needed by `registration_check`/`crater_alignment`),
original pilot scripts in `src/{acquisition,preprocessing,registration,terrain,tiling,validation,metadata}`
(provenance; **not used** by the current pipeline and they expect the removed pilot folders),
`Notes.txt`, `outputs/*.xml` / `ode_rest.html` (source-discovery records), all logs.

## 6. Where every number in the paper comes from

| Paper content | File |
|:--|:--|
| Region table | `outputs/tables/region_statistics.csv` → `paper/tables/region_statistics.tex` |
| Baseline tables | `outputs/experiments/baseline_results.json` → `outputs/tables/baseline_summary.csv` → `paper/tables/baseline_*.tex` |
| Per-region results | `outputs/tables/per_region_thermal.csv`, `per_region_slope.csv` |
| NCC alignment table | `outputs/experiments/registration_check.json` → `paper/tables/registration_check.tex` |
| Crater alignment | `outputs/experiments/crater_alignment.json` (+ per-crater CSV) |
| Figures | `paper/figures/region_montage.png`, `region_map.png` (made by `dataset_stats`) |
| References | `paper/manuscript/refs.bib` |

Rule: if a number in `main.tex` disagrees with these files, the files win.

## 7. Using the data (teammates)

```python
from src.experiments.data import load_region_tiles, load_splits, valid_mask
splits = load_splits()                 # {'train': [...], 'val': [...], 'test': [...]}
t = load_region_tiles("tycho")         # each layer: (16, 64, 64)
m = valid_mask(t)                      # True where every modality is real
temp = t["thermal"][m]                 # trustworthy thermal pixels, Kelvin

import numpy as np
d = np.load("data/benchmark/tycho/diurnal_tbol.npz")
d["tbol"].shape, d["local_time"].shape  # (16, 96), (96,)
```

Rules: use the official splits; report anomaly R² as well as RMSE; apply masks;
never claim pixel-level day/night behaviour; never random-split tiles.

## 8. Commands (from the project root, venv at `../.venv`)

```bash
python -m pytest tests                                 # offline checks
python run_pipeline.py --list                          # regions
python run_pipeline.py --region <id>                   # build one region (resumable)
python run_pipeline.py --all                           # rebuild everything (downloads WAC tiles)
python -m src.pipeline.extract_diurnal                 # tile-level diurnal curves
python -m src.experiments.dataset_stats                # tables, montage, map
python -m src.experiments.registration_check           # NCC alignment check
python -m src.experiments.crater_alignment             # crater alignment check (needs Robbins CSV)
python -m src.experiments.baselines --logo             # baselines + LORO (~2.5 h CPU)
python -m src.experiments.make_tables                  # CSV + LaTeX tables
```

Add a region: add it to `configs/regions.yaml` (2° box, edges on multiples of 0.5°,
|lat| ≤ 60°, not straddling the equator), run the pipeline for it, assign a split,
re-run `dataset_stats`, `baselines --logo`, `make_tables`, and update the paper numbers.
The Robbins CSV (for `crater_alignment`) is at `data/external/robbins/`; source:
USGS Astropedia "Moon Crater Database v1 Robbins".

## 9. Limitations to state in the paper (all also in `DATASET_CARD.md`)

1. Pixel-level thermal is one local time (midnight); the diurnal curve is tile-level.
2. Grid (≈ 237 m) is coarser than native WAC (100 m); slope is computed at 237 m.
3. 28 boxes of ~60 km: results describe these terrains, not the whole Moon; 8 test regions.
4. Alignment is by georeferencing; checks are consistency checks, ≈ 1 px bound, not sub-pixel proof.
5. WAC illumination ≠ synthetic hillshade illumination.
6. Nearest-filled gap pixels and WAC nodata: always use masks.

## 10. Before submission (what is still open)

Needs the authors / owner:
* Author list, affiliations, venue-specific template and page limit.
* License text (code is MIT; data derive from NASA PDS public-domain products), archive DOI (Zenodo / Hugging Face), code URL — placeholders marked `\TODO` in `main.tex`.
* Confirm the novelty claim in Related Work (marked `\TODO`); do a final literature check.
* Compile in Overleaf (no LaTeX on the dev machine) and fix any layout issues; proof-read.
* Decide whether to publish the data as a dataset archive (the `.npy` tiles are gitignored by design).

Not done, and not claimed: sub-pixel registration proof, second pixel-level thermal time,
Horn slope, tuned or state-of-the-art models.
