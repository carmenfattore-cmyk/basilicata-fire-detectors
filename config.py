"""Paths shared by all scripts. Edit ROOT only if the data are not inside this repository.

Expected layout (default: ROOT = repository folder):
  data/GEE_Basilicata_PCA/                 GeoTIFF/CSV/GeoJSON exported by the two Earth Engine scripts
  data/incendi_basilicata_2016_2025.gpkg   regional fire register 2016-2025 (layer 'incendi_2016_2025')
  results/                                 created by the scripts
"""
import os
ROOT = os.environ.get('BASILICATA_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEE_DIR = os.path.join(ROOT, 'data', 'GEE_Basilicata_PCA')
FIRES = os.path.join(ROOT, 'data', 'incendi_basilicata_2016_2025.gpkg')
RESULTS = os.path.join(ROOT, 'results')
UNET_DIR = os.path.join(RESULTS, 'unet_cv')            # 01: U-Net out-of-fold scores and final models
COMPARE_DIR = os.path.join(RESULTS, 'comparison_cv')   # 02: tables and binary maps of all detectors
SCORE_DIR = os.path.join(COMPARE_DIR, 'scores')        # 02: continuous score maps of the test years (.npy)
STATS_DIR = os.path.join(RESULTS, 'statistics')        # 03
DIAG_DIR = os.path.join(RESULTS, 'missed_fires')       # 04
TREND_DIR = os.path.join(RESULTS, 'trends')            # 05-06
FIG_DIR = os.path.join(RESULTS, 'figures')             # 05-10
for d in (RESULTS, UNET_DIR, COMPARE_DIR, SCORE_DIR, STATS_DIR, DIAG_DIR, TREND_DIR, FIG_DIR): os.makedirs(d, exist_ok=True)
