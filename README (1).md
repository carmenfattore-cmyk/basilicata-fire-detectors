# Sentinel-2 forest-fire detectors validated against the Basilicata fire register

Code accompanying the article:

> Fattore, C. *Mapping forest fires in Basilicata (southern Italy) with Sentinel-2: unsupervised, machine-learning and deep-learning detectors validated against the regional fire register.* Submitted to *Remote Sensing Applications: Society and Environment*.

The workflow compares five detectors (dNBR, standardized multitemporal PCA, autoencoder, Random Forest, U-Net) on two Sentinel-2 inputs (summer median composites and pre/post fire-season composites), trains them on the 2018–2021 register, chooses every threshold by leave-one-year-out cross-validation, and tests them on the 426 wooded fires of 2022–2024. It also computes the interannual NBR trends and the post-fire trajectories, 2017–2025.

## Repository content

| Folder / file | Content |
|---|---|
| `gee/01_annual_composites_forest_mask_pca.js` | Earth Engine: summer median NBR/NDVI 2017–2025, forest mask, Hansen loss, exploratory PCA, regional boundary |
| `gee/02_seasonal_composites.js` | Earth Engine: pre-season (May–June) and post-season (mid Sep–Oct) NBR/NDVI, clear-observation counts |
| `data/incendi_basilicata_2016_2025.gpkg` | Regional fire register 2016–2025, merged in one layer (`incendi_2016_2025`) and reprojected to EPSG:32633; perimeters unchanged |
| `scripts/config.py` | All paths (edit only if your data are elsewhere) |
| `scripts/00_download_fire_register_qgis.py` | Downloads the yearly register layers from the RSDI Basilicata WFS (QGIS Python console) |
| `scripts/01_unet_cv.py` | U-Net, leave-one-year-out training on 2018–2021 and final models |
| `scripts/02_compare_methods.py` | All detectors on both inputs, thresholds, per-pixel / per-fire / per-patch validation |
| `scripts/03_statistics.py` | Bootstrap intervals, sensitivity, detection by month, union of the two inputs, register statistics |
| `scripts/04_missed_fires.py` | Diagnosis of the fires ≥ 5 ha missed by all methods |
| `scripts/05_trends.py`, `06_trend_figure.py` | Clear-observation statistics, Theil–Sen / Mann–Kendall trends, cohort trajectories |
| `scripts/07`–`10_fig_*.py` | Figures of the article |

## How to reproduce

1. **Earth Engine.** Run the two scripts in `gee/` in the [Earth Engine Code Editor](https://code.earthengine.google.com/) and start all exports from the *Tasks* tab. Download the Google Drive folder `GEE_Basilicata_PCA` into `data/` (about 0.7 GB; not stored on GitHub).
2. **Python environment.**
   ```
   conda env create -f environment.yml
   conda activate incendi
   ```
   For the U-Net on a GPU, install the CUDA build of PyTorch that matches your driver (see pytorch.org).
3. **Run the scripts in order** (`01` to `10`) from Spyder (F5) or from a terminal (`python scripts/01_unet_cv.py`). Results are written to `results/`.
   - `02_compare_methods.py` needs the U-Net models from `01`.
   - `03`, `04` and `08` need the score maps and tables written by `02`.

The original runs used Windows (64-bit), an Intel Core i9-13900HX, 32 GB RAM and an NVIDIA GeForce RTX 4070 Laptop GPU (8 GB). Random seeds are fixed; small differences can arise from GPU non-determinism.

## Data sources

- Sentinel-2 Level-2A surface reflectance, Copernicus programme, through the Earth Engine catalogue (`COPERNICUS/S2_SR_HARMONIZED`). Contains modified Copernicus Sentinel data (2017–2025).
- Fire register (aree percorse dal fuoco): Regione Basilicata, RSDI Basilicata, open data (WMS/WFS).
- ESA WorldCover 2020 v100; Hansen et al. Global Forest Change v1.12; FAO GAUL 2015 (regional boundary).

## Licence

Code: MIT licence (see `LICENSE`). The fire register remains the property of Regione Basilicata and is redistributed under the terms of its open-data licence.

## Citation

If you use this code, please cite the article above (and the archived release, DOI to be added).
