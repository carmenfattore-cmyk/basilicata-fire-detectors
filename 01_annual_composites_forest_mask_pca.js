/**
 * Forest disturbances in Basilicata (Italy), 2017-2025
 * Step 1 - Multi-temporal standardized PCA of summer Sentinel-2 NBR/NDVI over forest areas
 *          + exports for the autoencoder (Python) and reference data for validation
 *
 * HOW TO USE
 *  1. Paste in the Google Earth Engine Code Editor and press Run.
 *  2. Console: explained variance and loadings (eigenvectors) of each principal component.
 *  3. Map: PC1 (mean state), PC2-PC4 (anomalies), forest mask, Hansen forest loss.
 *  4. Tasks tab -> run the exports (Drive folder GEE_Basilicata_PCA):
 *       - Bas_NBR_summer_2017_2025      annual summer NBR stack, forest only, 30 m, Int16 (x10000)  -> autoencoder input
 *       - Bas_NDVI_summer_2017_2025     same for NDVI
 *       - Bas_PCA_NBR                   first 5 standardized PCs of NBR, 30 m
 *       - Bas_PCA_NBR_loadings          eigenvalues, explained variance, loadings (CSV)
 *       - Bas_forest_mask               forest mask used (30 m)
 *       - Bas_Hansen_lossyear           Hansen GFC forest-loss year 2017-2024 (validation)
 *       - Bas_boundary                  regional boundary (GeoJSON)
 */

// ------------------------------------------------------------------ settings
var Y0 = 2017, Y1 = 2025;                 // Sentinel-2 L2A over Italy is consistent from 2017
var SUMMER = [6, 15, 9, 15];              // 15 June - 15 September (leaf-on, dry season)
var SCALE_STATS = 100;                    // scale for PCA statistics (sampling), m
var SCALE_OUT = 30;                       // export scale, m
var DRIVE = 'GEE_Basilicata_PCA';

var region = ee.FeatureCollection('FAO/GAUL/2015/level1')
  .filter(ee.Filter.eq('ADM1_NAME', 'Basilicata'));
var geom = region.geometry();
Map.centerObject(region, 8);

// ------------------------------------------------------------------ forest mask
// tree cover in ESA WorldCover 2020 OR Hansen tree cover 2000 >= 30% (keeps stands disturbed before 2020)
var wc = ee.Image('ESA/WorldCover/v100/2020').select('Map').eq(10);
var gfc = ee.Image('UMD/hansen/global_forest_change_2024_v1_12');
var forest = wc.or(gfc.select('treecover2000').gte(30)).rename('forest').clip(geom);
forest = forest.updateMask(forest);
print('Forest area (km2)', ee.Image.pixelArea().divide(1e6).updateMask(forest)
  .reduceRegion({reducer: ee.Reducer.sum(), geometry: geom, scale: 100, maxPixels: 1e10, tileScale: 4}).get('area'));

// ------------------------------------------------------------------ Sentinel-2 summer composites
function maskS2(img) {
  var scl = img.select('SCL');
  var ok = scl.eq(4).or(scl.eq(5)).or(scl.eq(7));     // vegetation, bare soil, unclassified
  return img.updateMask(ok);
}
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(geom)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .map(maskS2)
  .map(function (img) {
    return img.addBands(img.normalizedDifference(['B8', 'B12']).rename('NBR'))
              .addBands(img.normalizedDifference(['B8', 'B4']).rename('NDVI'));
  });

var years = ee.List.sequence(Y0, Y1);
function stack(index) {
  var col = ee.ImageCollection(years.map(function (y) {
    y = ee.Number(y);
    var c = s2.filterDate(ee.Date.fromYMD(y, SUMMER[0], SUMMER[1]), ee.Date.fromYMD(y, SUMMER[2], SUMMER[3]));
    return c.select(index).median().rename(ee.String(index).cat('_').cat(y.format('%d')))
            .set('n_scenes', c.size());
  }));
  return col.toBands().regexpRename('^[0-9]+_', '').updateMask(forest);
}
var nbr = stack('NBR');
var ndvi = stack('NDVI');
var bandNames = nbr.bandNames();
print('NBR bands', bandNames);

// ------------------------------------------------------------------ standardized PCA (correlation matrix)
function standardizedPCA(img, nameRoot) {
  var names = img.bandNames();
  var stats = img.reduceRegion({
    reducer: ee.Reducer.mean().combine(ee.Reducer.stdDev(), '', true),
    geometry: geom, scale: SCALE_STATS, maxPixels: 1e10, tileScale: 8, bestEffort: true
  });
  var means = ee.Image.constant(names.map(function (b) { return stats.get(ee.String(b).cat('_mean')); })).rename(names);
  var sds   = ee.Image.constant(names.map(function (b) { return stats.get(ee.String(b).cat('_stdDev')); })).rename(names);
  var z = img.subtract(means).divide(sds);
  var arr = z.toArray();
  var cov = arr.reduceRegion({
    reducer: ee.Reducer.centeredCovariance(), geometry: geom, scale: SCALE_STATS,
    maxPixels: 1e10, tileScale: 8, bestEffort: true
  });
  var eig = ee.Array(cov.get('array')).eigen();       // rows sorted by eigenvalue, descending
  var values = eig.slice(1, 0, 1);                     // [n,1]
  var vectors = eig.slice(1, 1);                       // [n,n]
  var nPC = names.size();
  var pcNames = ee.List.sequence(1, nPC).map(function (i) { return ee.String(nameRoot).cat(ee.Number(i).format('%d')); });
  var pcs = ee.Image(vectors).matrixMultiply(arr.toArray(1)).arrayProject([0]).arrayFlatten([pcNames]);
  var total = values.reduce(ee.Reducer.sum(), [0]).get([0, 0]);
  var explained = values.divide(total).multiply(100);
  return {pcs: pcs, values: values, vectors: vectors, explained: explained, names: names, pcNames: pcNames};
}

var pca = standardizedPCA(nbr, 'PC');
print('NBR PCA - explained variance (%) per component', pca.explained.project([0]));
print('NBR PCA - loadings (rows = PCs, columns = years)', pca.vectors);
// Reading the loadings: PC1 has similar positive loadings on all years (mean condition);
// a component with a strong loading on a single year, or a sign change between years,
// marks an abrupt disturbance in that year (fire, clear-cut); smooth gradients mark trends.

// ------------------------------------------------------------------ reference: Hansen forest loss 2017-2024
var loss = gfc.select('lossyear').add(2000).updateMask(gfc.select('lossyear').gte(17)).clip(geom).rename('lossyear');

// ------------------------------------------------------------------ map
var nbrVis = {min: -0.1, max: 0.6, palette: ['7f0000', 'd73027', 'fdae61', 'ffffbf', 'a6d96a', '1a9850']};
Map.addLayer(forest, {palette: ['1a9850']}, 'Forest mask', false);
Map.addLayer(nbr.select('NBR_2021'), nbrVis, 'NBR summer 2021', false);
Map.addLayer(pca.pcs.select('PC1'), {min: -6, max: 6}, 'PC1 (mean state)');
Map.addLayer(pca.pcs.select(['PC2', 'PC3', 'PC4']), {min: -3, max: 3}, 'PC2-PC3-PC4 (anomalies, RGB)');
Map.addLayer(loss, {min: 2017, max: 2024, palette: ['ffff00', 'ff8800', 'ff0000', '880088']}, 'Hansen loss 2017-2024', false);
Map.addLayer(region.style({color: 'black', fillColor: '00000000', width: 1}), {}, 'Basilicata');

// ------------------------------------------------------------------ exports
var loadTable = ee.FeatureCollection(ee.List.sequence(0, pca.names.size().subtract(1)).map(function (i) {
  i = ee.Number(i);
  var row = pca.vectors.slice(0, i, i.add(1)).project([1]).toList();
  var props = ee.Dictionary.fromLists(pca.names, row)
    .set('component', ee.String('PC').cat(i.add(1).format('%d')))
    .set('eigenvalue', pca.values.get([i, 0]))
    .set('explained_pct', pca.explained.get([i, 0]));
  return ee.Feature(null, props);
}));
Export.table.toDrive({collection: loadTable, description: 'Bas_PCA_NBR_loadings', folder: DRIVE, fileFormat: 'CSV'});

var CRS = 'EPSG:32633';   // WGS 84 / UTM 33N
Export.image.toDrive({image: nbr.multiply(10000).round().toInt16(), description: 'Bas_NBR_summer_2017_2025',
  folder: DRIVE, region: geom, scale: SCALE_OUT, crs: CRS, maxPixels: 1e11});
Export.image.toDrive({image: ndvi.multiply(10000).round().toInt16(), description: 'Bas_NDVI_summer_2017_2025',
  folder: DRIVE, region: geom, scale: SCALE_OUT, crs: CRS, maxPixels: 1e11});
Export.image.toDrive({image: pca.pcs.select(['PC1', 'PC2', 'PC3', 'PC4', 'PC5']).toFloat(), description: 'Bas_PCA_NBR',
  folder: DRIVE, region: geom, scale: SCALE_OUT, crs: CRS, maxPixels: 1e11});
Export.image.toDrive({image: forest.toByte(), description: 'Bas_forest_mask',
  folder: DRIVE, region: geom, scale: SCALE_OUT, crs: CRS, maxPixels: 1e11});
Export.image.toDrive({image: loss.toInt16(), description: 'Bas_Hansen_lossyear',
  folder: DRIVE, region: geom, scale: SCALE_OUT, crs: CRS, maxPixels: 1e11});
Export.table.toDrive({collection: region, description: 'Bas_boundary', folder: DRIVE, fileFormat: 'GeoJSON'});
