# Dataset Card — Multimodal Lunar Benchmark (v1.1)

## Summary
Pixel-aligned optical (LROC WAC), elevation (LOLA LDEM_256) and thermal
(Diviner TBOL) observations of 28 lunar regions, released as 64x64 tiles on a
common equirectangular grid (128 px/deg, ~236.9 m/px) together with derived
slope/hillshade layers and per-pixel validity/coverage masks.

## Composition
- 28 regions x 2 deg x 2 deg = 28 x 16 = 448 tiles, each with 10 arrays
  (optical, dem, thermal, slope, hillshade, optical/dem/thermal coverage,
  dem/thermal valid masks) plus a `metadata.json`. Units: optical = WAC
  8-bit-scaled reflectance (0-255), dem = km relative to the 1737.4 km sphere,
  thermal = K, slope = degrees, hillshade = 0-1.
- Northern and southern hemispheres; nearside and farside; mare, highland,
  young-crater, rille/pit, multi-ring basin, swirl, Apollo-landing and
  high-latitude terrain. Latitudes are limited to +-60 deg by the WAC tiles used.
- Per region, `data/benchmark/<region>/diurnal_tbol.npz`: Diviner bolometric
  temperature vs local time, `tbol[16, 96]` (tile order, 0.25 h bins,
  K, NaN = no observations) plus `local_time[96]`. Source: PDS Diviner global
  cumulative products (Williams et al. 2017), 0.5 deg bins, 2009-2015. Tile
  level only; the midnight bins agree with the TBOL_SAM tile means
  (r = 0.993, RMSE 1.15 K, product 0.73 K warmer).
- Region definitions: `configs/regions.yaml`. Official splits: `configs/splits.yaml`
  (17 train / 3 val / 8 test regions).
- Region centres were checked against published feature coordinates (IAU names as
  tabulated on Wikipedia). All boxes contain their named feature; the only
  off-centre box is `endymion` (feature centre 56.5E, box 54-56E), which covers
  the western part of that 125 km crater floor. Box centres of the larger
  features (basins, plateaus) are approximate by design.

## Sources (all NASA PDS, public domain)
| Layer | Product | Native | Mapping to common grid |
|:--|:--|:--|:--|
| optical | LROC WAC global morphology mosaic, 100 m | 100 m | area-averaged; nodata (0) excluded via coverage |
| dem | LOLA LDEM_256 (float, km rel. 1737.4 km) | ~118.5 m | exact 2x2 block mean |
| thermal | Diviner TBOL_SAM (local midnight) | ~236.9 m | direct, pixel centres verified equal |
| slope, hillshade | derived from common-grid DEM | — | central differences (`numpy.gradient`, latitude-corrected E-W spacing); NW sun, 45 deg |

## Intended use
Benchmarking cross-modal learning, multimodal registration/validation methods
and terrain-aware thermal modelling on the Moon. Evaluation is by held-out
region (never random tiles).

## Known limitations (state these in any paper using the data)
1. **Pixel-level thermal is a single local-time product** (midnight), so it
   captures thermophysical/thermal-inertia contrast, not diurnal behaviour.
   A tile-level diurnal curve (`diurnal_tbol.npz`, see Composition) gives
   day/night context at ~15 km only; it is not pixel-aligned.
2. **Common-grid resolution (~237 m) is coarser than WAC's native 100 m;**
   fine optical detail is averaged away by design. Slope is scale-dependent and
   is computed at this 237 m scale, not at LOLA's native scale.
3. **Region count is small (28 boxes of ~60 km).** Results describe these
   terrains, not the whole Moon; the pipeline supports adding regions. Only 8
   regions are in the test set, so per-region results should be read
   individually.
4. **Registration is exact by georeferencing**, not by image-based
   co-registration. Residual error is bounded by each product's own
   geolocation accuracy; an empirical NCC shift check is reported per region,
   and is uninformative on low-relief mare where albedo and topography are
   weakly correlated. A second, crater-based check (106 Robbins-catalogue
   craters, 23 regions) puts the median thermal-minus-elevation offset at
   about 0.0 px (95% CI within +-0.4 px) and the optical-gradient-minus-
   elevation offset within about 1 px. These are consistency checks, not a
   proof of sub-pixel alignment.
5. **Illumination differs between layers:** WAC is acquired under its own sun
   angle; hillshade uses a fixed synthetic sun and is not a reflectance model.
6. **Gap pixels** (rare NaNs) are nearest-filled and flagged invalid in the
   mask/coverage layers; always apply the masks when computing metrics.
7. WAC nodata (0) appears at some mosaic seams / low-signal areas; use
   `optical_coverage` to exclude them.
8. Planetocentric coordinates; sphere radius 1,737,400 m.

## Reproducibility
`python run_pipeline.py --all` rebuilds everything from public URLs (HTTP
range streaming for LOLA/Diviner, one resumable download per WAC tile).
`python -m pytest tests` runs offline sanity checks.
`PROJECT_GUIDE.md` explains the whole project and how to inspect and load the data.

## License / citation
Code: MIT. Data derive from NASA PDS products (open data policy); cite the
LROC, LOLA and Diviner instrument papers and the PDS archives.
