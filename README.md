# Multimodal Lunar Benchmark Dataset

A rigorously harmonized, multi-region benchmark dataset for multimodal lunar remote sensing, co-registering **Optical (LROC WAC)**, **Topographic (LOLA LDEM)**, and **Thermal (Diviner Radiometer)** observations into synchronized, analysis-ready 64x64 tiles across 28 scientifically diverse lunar regions (northern/southern hemisphere, nearside and farside).

---

## Overview

Multimodal sensor fusion on planetary bodies requires addressing vast discrepancies in spatial resolution, projections, illumination geometry, and observational footprints.

This benchmark defines an end-to-end reproducible framework to:
1. Ingest official Planetary Data System (PDS) products from the **Lunar Reconnaissance Orbiter (LRO)** using HTTP Range-request streaming (no multi-GB downloads required).
2. Establish a unified 256x256 common cartographic grid (128 pixels/degree, equirectangular) over any 2deg x 2deg target region.
3. Compute physically consistent terrain derivatives (slope angle, hillshade).
4. Partition harmonized grids into 16 synchronized 64x64 benchmark tiles with pixel-wise correspondence, valid masks, and structured JSON metadata.
5. Validate spatial registration using cross-modal mutual information, edge gradient Dice similarity, and structural integrity checks.

---

## Benchmark Regions (v1.0)

Region definitions live in `configs/regions.yaml` (2 deg x 2 deg boxes on the 128 px/deg grid).
Source tiles (LOLA N/S x 0-180/180-360 E, WAC 90 x 60 deg tiles) are chosen automatically per region.

| # | Region ID | Terrain class | Box (lat, lon E) | Split |
|:--|:--|:--|:--|:--|
| 1 | `lacus_mortis` | mare + complex crater (pilot) | 44-46N, 27-29 | train |
| 2 | `posidonius` | floor-fractured crater | 31-33N, 28-30 | train |
| 3 | `montes_apenninus` | mountain range + rille | 25-27N, 3-5 | train |
| 4 | `aristoteles` | rugged crater ejecta | 49-51N, 16-18 | train |
| 5 | `endymion` | dark mare-filled crater | 52-54N, 54-56 | val |
| 6 | `mare_crisium` | flat mare basin | 15-17N, 58-60 | test |
| 7 | `taurus_littrow` | highland massifs + dark mantle valley (Apollo 17) | 19-21N, 30-32 | test |
| 8 | `tranquillitatis` | equatorial mare plain (Apollo 11) | 0-2N, 23-25 | train |
| 9 | `proclus` | young bright crater + rays | 15-17N, 46-48 | train |
| 10 | `hyginus` | volcanic rille + pit craters | 7-9N, 5-7 | val |
| 11 | `atlas` | northern highland crater | 46-48N, 43-45 | train |
| 12 | `humboldtianum` | high-latitude low-sun mare | 56-58N, 80-82 | train |
| 13 | `descartes` | southern highlands (Apollo 16) | 10-8S, 15-17 | train |
| 14 | `tycho` | young complex crater (south) | 44.5-42.5S, 347.5-349.5 | test |
| 15 | `tsiolkovsky` | farside crater + mare fill | 21.5-19.5S, 128-130 | test |
| 16 | `copernicus` | young complex crater (nearside mare) | 8.5-10.5N, 339-341 | test |
| 17 | `aristarchus` | bright young crater + volcanic plateau | 22.5-24.5N, 311.5-313.5 | train |
| 18 | `orientale` | multi-ring basin interior | 20.5-18.5S, 266-268 | train |
| 19 | `ingenii` | farside mare + magnetic swirls | 34.5-32.5S, 162.5-164.5 | test |
| 20 | `fra_mauro` | highland ejecta formation (Apollo 14) | 7-5S, 342-344 | train |
| 21 | `sinus_iridum` | mare bay + mountain arc | 44-46N, 327.5-329.5 | train |
| 22 | `moscoviense` | farside mare basin | 26.5-28.5N, 147-149 | test |
| 23 | `korolev` | farside highland basin crater | 5-3S, 201.5-203.5 | test |
| 24 | `kepler` | young bright crater + rays (Procellarum) | 7-9N, 321-323 | train |
| 25 | `grimaldi` | dark-floored basin | 6.5-4.5S, 290.5-292.5 | train |
| 26 | `alphonsus` | crater with pyroclastic dark halos | 14.5-12.5S, 356-358 | train |
| 27 | `plato` | high-latitude dark-floored crater | 50.5-52.5N, 349.5-351.5 | val |
| 28 | `apollo_basin` | farside basin interior | 37-35S, 207-209 | train |

**Splits are by region, never by tile** (`configs/splits.yaml`): the test set holds out
a southern young crater, a farside crater and a farside swirl mare, a young nearside
crater, Apollo-17 massifs and a flat mare basin. See `PROJECT_GUIDE.md` for how to view
and load the data.

---

## Sensor Modalities & Specifications

| Modality | Instrument / Product | Native Resolution | Common Grid Resolution | Resampling |
| :--- | :--- | :--- | :--- | :--- |
| **Optical** | LROC WAC Morphology Mosaic | 100 m/px | ~236.9 m (128 px/deg) | Area-Averaged |
| **DEM** | LOLA LDEM_256 | ~118.5 m (256 px/deg) | ~236.9 m (128 px/deg) | 2x2 Block Average |
| **Slope & Hillshade** | Derived from LOLA DEM | -- | ~236.9 m | central differences (numpy.gradient) |
| **Thermal** | Diviner TBOL_SAM (midnight) | ~236.9 m (128 px/deg) | ~236.9 m | Direct native-grid map |
| **Diurnal (tile-level)** | Diviner global cumulative products (0.5 deg x 0.25 h bins) | 0.5 deg | one value per 64x64 tile | `diurnal_tbol.npz`, `tbol[16, 96]` K; not pixel-aligned |

**Common grid:** 256x256 pixels, 128 px/deg, Lunar equirectangular (R = 1,737,400 m)
**Tiles:** 16 x (64x64 px) per region, 448 tiles total across 28 regions

---

## Repository Structure

```
Multimodal_Lunar_Benchmark/
|-- configs/
|   |-- benchmark_config.yaml      # Original pilot config
|   `-- regions.yaml               # All 28 region definitions + PDS source URLs
|-- data/
|   |-- raw/                       # Cached PDS downloads (gitignored, auto-downloaded)
|   |   |-- optical/               # WAC GeoTIFF (~473 MB, one-time download)
|   |   |-- dem/                   # Optional local LOLA .IMG (~4.2 GB); if absent, windows are HTTP-streamed
|   |   `-- thermal/               # Diviner .IMG (streamed via HTTP, no download needed)
|   |-- intermediate/
|   |   `-- {region_id}/           # Per-region intermediate arrays (gitignored)
|   |       |-- optical/
|   |       |-- dem/
|   |       |-- thermal/
|   |       `-- registration/
|   `-- benchmark/
|       |-- global_manifest.json   # Cross-region index (tracked in git)
|       `-- {region_id}/
|           |-- metadata/
|           |   `-- tile_index.json
|           `-- tiles/
|               `-- tile_{row}_{col}/
|                   |-- optical.npy, dem.npy, thermal.npy
|                   |-- slope.npy, hillshade.npy
|                   |-- *_valid_mask.npy, *_coverage.npy
|                   `-- metadata.json
|-- outputs/
|   `-- {region_id}/
|       `-- validation/
|           |-- common_grid_composite.png
|           |-- cross_modal_mi.png
|           `-- validation_report.json
|-- src/
|   |-- pipeline/                  # Multi-region pipeline (region-aware, improved)
|   |   |-- config.py              # Region loader from configs/regions.yaml
|   |   |-- extract_thermal.py     # Diviner HTTP range streaming
|   |   |-- extract_dem.py         # LOLA HTTP range streaming (no 4.2 GB download)
|   |   |-- extract_optical.py     # WAC cached download + AOI extraction
|   |   |-- create_grid.py         # Common cartographic grid
|   |   |-- harmonize.py           # All three modality harmonizations
|   |   |-- derive_terrain.py      # Slope + hillshade
|   |   |-- create_tiles.py        # 16 synchronized benchmark tiles
|   |   |-- validate.py            # Structural + MI + edge-Dice + QC figures
|   |   `-- manifest.py            # Global manifest updater
|   |-- acquisition/ preprocessing/ registration/ terrain/ tiling/ validation/ metadata/   # ORIGINAL pilot scripts (provenance; not used by src/pipeline)
|-- scripts/
|   `-- create_directories.py      # One-time directory scaffolding
|-- run_pipeline.py                # Master pipeline orchestrator
|-- requirements.txt
`-- .gitignore
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Madhav2246/Multimodal_benchmark_Dataset_FYP_P1.git
cd Multimodal_benchmark_Dataset_FYP_P1
```

### 2. Create and activate virtual environment
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Create directory structure
```bash
python scripts/create_directories.py
```

---

## Running the Pipeline

The `run_pipeline.py` orchestrator is **fully automated and idempotent** (skips steps whose outputs already exist). It replaces the previous 14-command manual sequence.

### Process a single region
```bash
python run_pipeline.py --region posidonius
```

### Process every region
```bash
python run_pipeline.py --all
```

### Force re-run all steps for a region
```bash
python run_pipeline.py --region aristoteles --force
```

### List all available regions
```bash
python run_pipeline.py --list
```

### Pipeline Steps (per region)

| Step | Module | Description |
|:--|:--|:--|
| 1 | `extract_thermal` | HTTP Range-stream 256x256 Diviner window (~0.25 MB) |
| 2 | `extract_dem` | 512x512 LOLA window, HTTP Range-streamed (~1 MB); uses the local .IMG instead if present |
| 3 | `extract_optical` | Download WAC GeoTIFF once (~473 MB, cached), extract AOI |
| 4 | `create_grid` | Compute 256x256 common coordinate mesh |
| 5 | `harmonize` | Area-average WAC, 2x2-aggregate LOLA, direct-map Diviner |
| 6 | `derive_terrain` | Slope (deg, central differences) + NW hillshade |
| 7 | `create_tiles` | Partition into 16 synchronized 64x64 tiles with metadata JSON |
| 8 | `validate` | Structural integrity + cross-modal MI + edge Dice + QC figures |
| 9 | `manifest` | Update `data/benchmark/global_manifest.json` |

> **Key improvement:** Step 1 uses HTTP Range requests to stream only the required Diviner window (~0.25 MB per region) — no full file download needed. LOLA is streamed the same way (the 4.2 GB file is optional), and the 497 MB WAC GeoTIFF is downloaded once, resumably, and size-verified. Gap pixels (rare NaNs) are nearest-filled and flagged `False` in the valid-mask/coverage layers.

---

## Evaluation Layer

```bash
python -m src.experiments.dataset_stats        # tables + region map/montage
python -m src.experiments.registration_check   # per-region cross-modal shift check
python -m src.experiments.baselines --logo     # tasks, baselines, leave-one-region-out
python -m pytest tests                         # offline sanity tests
```

Tasks: (1) Diviner midnight temperature from optical and/or terrain, (2) slope from optical alone.
Baselines: train mean, latitude-only, Ridge, HistGradientBoosting, small CNN.
Metrics: RMSE, MAE, R^2 and *anomaly R^2* (tile-demeaned, i.e. within-tile structure only).

---

## Benchmark Validation Metrics

| Metric | Description |
|:--|:--|
| **Structural integrity** | All 10 layer files present, correct shape (64x64), no NaN values |
| **Cross-modal MI** | Histogram mutual information (nats) across all modality pairs |
| **Edge Dice score** | Dice similarity of salient-gradient binary masks (90th percentile threshold) |
| **Coverage fraction** | Mean proportion of valid native-resolution pixels per common-grid cell |

---

## Data Access & Citation

Raw data originates from NASA's Lunar Reconnaissance Orbiter (LRO) and is openly distributed via the NASA Planetary Data System (PDS). Processed benchmark tiles will be archived on Zenodo/Hugging Face for public access upon paper publication.

**Instruments:** LROC (Lunar Reconnaissance Orbiter Camera), LOLA (Lunar Orbiter Laser Altimeter), Diviner Lunar Radiometer Experiment.

---

## License

This repository is released under the MIT License. NASA PDS data products are subject to NASA's open data policy.
