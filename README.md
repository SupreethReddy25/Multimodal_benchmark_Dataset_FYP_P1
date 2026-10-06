# Multimodal Lunar Benchmark

A comprehensive, rigorously harmonized benchmark dataset and reproducible processing pipeline for multimodal lunar remote sensing, coregistering **Optical (LROC WAC)**, **Topographic (LOLA LDEM)**, and **Thermal (Diviner Radiometer)** observations over lunar terrains.

---

## 🌒 Overview

Multimodal sensor fusion on planetary bodies requires addressing vast discrepancies in spatial resolution, projections, illumination geometry, and observational footprints. 

This benchmark defines an end-to-end framework to:
1. Ingest official Planetary Data System (PDS) products from the **Lunar Reconnaissance Orbiter (LRO)**.
2. Establish a unified, area-weighted common cartographic grid centered on the **Lacus Mortis – Bürg** pilot region (44.0°N–46.0°N, 27.0°E–29.0°E).
3. Compute physically consistent morphological and terrain derivatives (slope angle, hillshade).
4. Partition harmonized continuous grids into synchronized $64 \times 64$ benchmark tiles with pixel-wise correspondence, valid masks, and structured JSON metadata.
5. Provide rigorous spatial registration validation using cross-modal mutual information, edge gradient correlation, and sub-pixel phase shifts.

---

## 🛰️ Sensor Modalities & Dataset Specifications

| Modality | Instrument / Product | Native Resolution | Harmonized Grid Resolution | Grid Resampling Method |
| :--- | :--- | :--- | :--- | :--- |
| **Optical** | LROC WAC Morphology Mosaic (`WAC_GLOBAL_E300N0450_100M`) | 100 m / pixel | ~236.9 m (128 px/deg) | Conservative Area-Average |
| **Topography (DEM)** | LOLA Digital Elevation Model (`LDEM_256`) | 118.5 m (256 px/deg) | ~236.9 m (128 px/deg) | Conservative Area-Average |
| **Terrain Derivatives** | Derived Slope & Hillshade from LOLA DEM | Calculated at grid | ~236.9 m (128 px/deg) | Analytic Horn / Finite Difference |
| **Thermal** | Diviner Lunar Radiometer (`DGHRM_TBOL_SAM`) | 236.9 m (128 px/deg) | ~236.9 m (128 px/deg) | Direct Native-to-Grid Binning |

### Pilot Area of Interest (AOI)
- **Region Name:** Lacus Mortis – Bürg Crater Pilot
- **Latitude Extent:** 44.0° N to 46.0° N
- **Longitude Extent:** 27.0° E to 29.0° E
- **Center Coordinates:** 45.0° N, 28.0° E
- **Footprint:** ~60.65 km × 60.65 km (2° × 2° angular span)
- **Projection:** Lunar Equirectangular ($R_{Moon} = 1,737,400\text{ m}$)

---

## 📁 Repository Structure

```
Multimodal_Lunar_Benchmark/
├── configs/
│   └── benchmark_config.yaml           # Master configuration (grid specs, tile dimensions, paths)
├── data/
│   ├── raw/                            # External raw PDS products (kept empty / gitignored)
│   │   ├── dem/                        # LOLA LDEM_256 raw image and labels
│   │   ├── optical/                    # LROC WAC GeoTIFFs
│   │   └── thermal/                    # Diviner TBOL SAM datasets
│   ├── intermediate/                   # Spatially resampled intermediate pilot grids (.npy)
│   │   ├── dem/                        # Pilot LOLA DEM, slope, hillshade, valid masks
│   │   ├── optical/                    # Pilot WAC optical numpy arrays & cropped GeoTIFF
│   │   ├── registration/               # Target coordinate meshes (lat, lon, meter x/y)
│   │   └── thermal/                    # Resampled Diviner bolometric thermal fields
│   └── benchmark/                      # Final packaged benchmark dataset
│       ├── metadata/                   # Manifests & spatial tile indices
│       └── tiles/                      # 16 synchronized 64x64 multimodal tiles
│           ├── tile_00_00/ ... tile_03_03/
│           │   ├── dem.npy
│           │   ├── optical.npy
│           │   ├── thermal.npy
│           │   ├── slope.npy
│           │   ├── hillshade.npy
│           │   ├── dem_valid_mask.npy
│           │   ├── optical_coverage.npy
│           │   ├── thermal_valid_mask.npy
│           │   └── metadata.json
├── outputs/
│   ├── figures/                        # High-resolution composites and quality maps
│   ├── tables/                         # Per-tile distribution statistics (.csv, .json)
│   └── validation/                     # Cross-modal correlation & alignment metrics
├── src/
│   ├── acquisition/                    # Raw product inspection & pilot clipping routines
│   ├── preprocessing/                  # Spatial projection and cleaning
│   ├── registration/                   # Grid harmonization & coordinate transformations
│   ├── terrain/                        # Slope & hillshade derivative generators
│   ├── tiling/                         # Multi-modal synchronized tile generation
│   └── validation/                     # Mutual info, edge correlation, and QC reporting
├── Notes.txt                           # Detailed engineering logs, PDS links, and math derivations
├── requirements.txt                    # Python environment specifications
└── .gitignore                          # Excludes heavy raw binary rasters (>100MB)
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Madhav2246/Multimodal_benchmark_Dataset_FYP_P1.git
cd Multimodal_benchmark_Dataset_FYP_P1
```

### 2. Create and activate a virtual environment
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

---

## 📥 Downloading Raw PDS Data (Optional for Full Reproduction)

The processed benchmark tiles and intermediate validation arrays are included in this repository. If you wish to re-execute the pipeline from raw satellite data:

1. **LROC WAC Optical Mosaic:**
   - Download `WAC_GLOBAL_E300N0450_100M.TIF` from [LROC PDS Archive](https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/EXTRAS/BROWSE/WAC_GLOBAL/WAC_GLOBAL_E300N0450_100M.TIF).
   - Place into `data/raw/optical/`.

2. **LOLA Digital Elevation Model (LDEM 256):**
   - Download `LDEM_256_00N_90N_000_180_FLOAT.IMG` & `.LBL` from the [PDS Geosciences Node](https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/lola_gdr/polar/float/north/).
   - Place into `data/raw/dem/`.

3. **Diviner Lunar Radiometer:**
   - Download the bolometric brightness temperature raster (`DGHRM_TBOL_SAM`) from the PDS Geosciences Node.
   - Place into `data/raw/thermal/`.

---

## 🚀 Running the Pipeline

Execute the modules in order:

```bash
# 1. Establish common coordinate mesh
python src/registration/create_common_grid.py

# 2. Resample and harmonize individual modalities
python src/registration/harmonize_wac_to_common.py
python src/registration/harmonize_lola_to_common.py
python src/registration/map_diviner_to_common.py

# 3. Derive morphometric terrain features
python src/terrain/derive_common_grid_terrain.py

# 4. Generate synchronized 64x64 benchmark tiles
python src/tiling/create_benchmark_tiles.py

# 5. Run validation & spatial quality benchmarks
python src/validation/validate_benchmark_tiles.py
python src/validation/cross_modal_mutual_information.py
python src/validation/cross_modal_edge_validation.py
python src/validation/create_final_multimodal_composite.py
```

---

## 📊 Benchmark Validation Summary

- **Tile Count:** 16 non-overlapping tiles ($4 \times 4$ grid, 64 × 64 pixels each)
- **Spatial Alignment Error:** Mean estimated sub-pixel registration shift $< 0.15\text{ pixels}$ across optical-DEM baselines.
- **Data Coverage:** 100% complete across all 16 tiles for Optical, LOLA DEM, Hillshade, Slope, and Diviner Thermal channels.
- **Physical Consistency:** High mutual information between hillshade and optical illumination patterns confirms geometric fidelity across crater boundaries.

---

## 📜 License & Acknowledgments

This research utilizes data from NASA's Lunar Reconnaissance Orbiter (LRO) mission instruments: LROC, LOLA, and Diviner. Datasets are distributed under open scientific access policies by NASA PDS.
