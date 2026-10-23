/**
 * Forest disturbances in Basilicata, 2017-2025 - Step 2 export
 * PRE- and POST-fire-season Sentinel-2 composites for every year, so that the change caused by
 * July-September fires falls BETWEEN two composites instead of inside a single summer median.
 *   pre-season : 1 May  - 30 June   (before most fires)
 *   post-season: 15 Sep - 31 October (after most fires, before leaf fall)
 *
 * Tasks (Drive folder GEE_Basilicata_PCA):
 *   Bas_NBR_pre_post_2017_2025   18 bands: NBRpre_2017, NBRpost_2017, ... (Int16 x10000, 30 m)
 *   Bas_NDVI_pre_post_2017_2025  18 bands, same for NDVI
 *   Bas_nobs_pre_post            number of clear observations per composite (quality control)
 */
var Y0 = 2017, Y1 = 2025;
var PRE = [5, 1, 6, 30];      // month, day, month, day
var POST = [9, 15, 10, 31];
var DRIVE = 'GEE_Basilicata_PCA';
var CRS = 'EPSG:32633', SCALE = 30;

var region = ee.FeatureCollection('FAO/GAUL/2015/level1').filter(ee.Filter.eq('ADM1_NAME', 'Basilicata'));
var geom = region.geometry();
var wc = ee.Image('ESA/WorldCover/v100/2020').select('Map').eq(10);
var gfc = ee.Image('UMD/hansen/global_forest_change_2024_v1_12');
var forest = wc.or(gfc.select('treecover2000').gte(30)).clip(geom);

function maskS2(img) {
  var scl = img.select('SCL');
  return img.updateMask(scl.eq(4).or(scl.eq(5)).or(scl.eq(7)));
}
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(geom).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
  .map(maskS2)
  .map(function (img) {
    return img.addBands(img.normalizedDifference(['B8', 'B12']).rename('NBR'))
              .addBands(img.normalizedDifference(['B8', 'B4']).rename('NDVI'));
  });

function composite(y, win, index, tag) {
  var c = s2.filterDate(ee.Date.fromYMD(y, win[0], win[1]), ee.Date.fromYMD(y, win[2], win[3]).advance(1, 'day'));
  var img = c.select(index).median().rename(ee.String(index).cat(tag).cat('_').cat(ee.Number(y).format('%d')));
  var n = c.select(index).count().rename(ee.String('n').cat(index).cat(tag).cat('_').cat(ee.Number(y).format('%d')));
  return [img, n];
}
function build(index) {
  var imgs = [], nobs = [];
  for (var y = Y0; y <= Y1; y++) {
    var a = composite(y, PRE, index, 'pre'), b = composite(y, POST, index, 'post');
    imgs.push(a[0], b[0]); nobs.push(a[1], b[1]);
  }
  return {img: ee.Image.cat(imgs).updateMask(forest), n: ee.Image.cat(nobs).updateMask(forest)};
}
var nbr = build('NBR'), ndvi = build('NDVI');
print('NBR bands', nbr.img.bandNames());
Map.centerObject(region, 8);
Map.addLayer(nbr.img.select('NBRpre_2021').subtract(nbr.img.select('NBRpost_2021')),
  {min: -0.1, max: 0.5, palette: ['1a9850', 'ffffbf', 'd73027']}, 'NBR pre-post 2021');

Export.image.toDrive({image: nbr.img.multiply(10000).round().toInt16(), description: 'Bas_NBR_pre_post_2017_2025',
  folder: DRIVE, region: geom, scale: SCALE, crs: CRS, maxPixels: 1e11});
Export.image.toDrive({image: ndvi.img.multiply(10000).round().toInt16(), description: 'Bas_NDVI_pre_post_2017_2025',
  folder: DRIVE, region: geom, scale: SCALE, crs: CRS, maxPixels: 1e11});
Export.image.toDrive({image: nbr.n.toByte(), description: 'Bas_nobs_pre_post',
  folder: DRIVE, region: geom, scale: SCALE, crs: CRS, maxPixels: 1e11});
