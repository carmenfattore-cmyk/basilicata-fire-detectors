`incendi_basilicata_2016_2025.gpkg`: burned-area register of Regione Basilicata, 2016–2025, downloaded from the RSDI Basilicata WFS service (see `scripts/00_download_fire_register_qgis.py`). Yearly layers plus the merged layer `incendi_2016_2025`, reprojected to WGS 84 / UTM 33N (EPSG:32633). Perimeters and original attributes are unchanged; added fields: `anno_gee` (year), `mese` (month), `data_iso` (date), `area_ha` (polygon area), `categoria` (land-cover category).

Place the Earth Engine exports in `data/GEE_Basilicata_PCA/`.
