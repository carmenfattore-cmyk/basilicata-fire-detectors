"""Diagnosis of register forest fires (test years) missed by all methods + panels of the two large 2022 fires."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import os, csv, json, numpy as np, rasterio, fiona
from rasterio.windows import from_bounds, Window
from rasterio.features import rasterize
from shapely.geometry import shape
from scipy.ndimage import binary_dilation, shift as ndshift
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

G = GEE_DIR
PERF = os.path.join(COMPARE_DIR, 'confronto_per_incendio.csv')
OUT = DIAG_DIR; os.makedirs(OUT, exist_ok=True)
src = {k: rasterio.open(os.path.join(G, f)) for k, f in [('ann', 'Bas_NBR_summer_2017_2025.tif'), ('sea', 'Bas_NBR_pre_post_2017_2025.tif'),
                                                         ('for', 'Bas_forest_mask.tif'), ('nd', 'Bas_NDVI_pre_post_2017_2025.tif')]}

feats = []
with fiona.open(FIRES, layer='incendi_2016_2025') as c:
    for f in c:
        p = f['properties']; feats.append(dict(geom=f['geometry'], **{k: p[k] for k in ('anno_gee', 'mese', 'data_iso', 'area_ha', 'categoria', 'com', 'denominazi', 'loc', 'supcalcb', 'supcalcnb')}))

# detection by any method (from the full comparison on the user's PC)
det = {}
with open(PERF, encoding='utf-8') as fh:
    for r in csv.DictReader(fh, delimiter=';'):
        k = (int(r['year']), r['comune'], r['data'], round(float(r['area_ha']), 3))
        det.setdefault(k, {})[f"{r['method']} ({r['input']})"] = (int(r['detected']), float(r['flagged_frac']))

def window_for(geom, buf=600):
    b = shape(geom).bounds
    return from_bounds(b[0] - buf, b[1] - buf, b[2] + buf, b[3] + buf, src['ann'].transform).round_offsets().round_lengths()

def rd(k, band, w): return src[k].read(band, window=w).astype(np.float32) / 10000.

def analyse(f):
    y = f['anno_gee']; w = window_for(f['geom']); tr = src['ann'].window_transform(w); sh = (w.height, w.width)
    poly = rasterize([(f['geom'], 1)], out_shape=sh, transform=tr, fill=0).astype(bool)
    other = rasterize([(g['geom'], 1) for g in feats if g is not f and g['anno_gee'] in (y - 1, y, y + 1)] or [(f['geom'], 0)],
                      out_shape=sh, transform=tr, fill=0).astype(bool)
    forest = src['for'].read(1, window=w) == 1
    ring = binary_dilation(poly, iterations=7) & ~poly & ~other & forest           # 0-210 m outside, other fires excluded
    a = {yy: rd('ann', yy - 2017 + 1, w) for yy in (y - 1, y, y + 1)}
    pre = rd('sea', 2 * (y - 2017) + 1, w); post = rd('sea', 2 * (y - 2017) + 2, w); pren = rd('sea', 2 * (y + 1 - 2017) + 1, w)
    d_ann = a[y - 1] - np.minimum(a[y], a[y + 1]); d_sea = pre - post; d_1y = pre - pren
    inside = poly & forest & (pre != 0) & (post != 0)
    def m(x, msk): return float(np.nanmean(x[msk])) if msk.sum() else np.nan
    res = dict(n_forest_px=int(inside.sum()), forest_share=float((poly & forest).sum() / max(poly.sum(), 1)),
               dNBR_sea_in=m(d_sea, inside), dNBR_sea_ring=m(d_sea, ring), dNBR_ann_in=m(d_ann, inside), dNBR_ann_ring=m(d_ann, ring),
               dNBR_1y_in=m(d_1y, inside), dNBR_1y_ring=m(d_1y, ring),
               frac_burn_sea=float((d_sea[inside] > 0.1).mean()) if inside.sum() else np.nan,
               frac_burn_ann=float((d_ann[inside] > 0.1).mean()) if inside.sum() else np.nan,
               NBR_pre_in=m(pre, inside))
    # displacement test: best overlap of the dNBR>0.1 patch with the polygon within +-10 px (300 m)
    burn = (np.maximum(d_sea, d_ann) > 0.1) & forest
    best = (0, 0, float((burn & poly).sum()))
    for dy in range(-10, 11):
        for dx in range(-10, 11):
            ov = float((burn & ndshift(poly.astype(np.uint8), (dy, dx), order=0).astype(bool)).sum())
            if ov > best[2] * 1.0001: best = (dy, dx, ov)
    res.update(shift_m=float(np.hypot(*best[:2]) * 30), overlap_gain=best[2] / max(float((burn & poly).sum()), 1))
    return res, dict(poly=poly, forest=forest, pre=pre, post=post, d_sea=d_sea, d_ann=d_ann, a_prev=a[y - 1], tr=tr)

def classify(r, mese):
    sig_sea = r['dNBR_sea_in'] - r['dNBR_sea_ring']; sig_ann = r['dNBR_ann_in'] - r['dNBR_ann_ring']
    sig = max(sig_sea, sig_ann)
    if r['forest_share'] < 0.5: return 'low tree cover in polygon (<50% forest)'
    if r['shift_m'] >= 90 and r['overlap_gain'] > 1.5: return 'burn scar displaced from polygon'
    if sig < 0.03: return 'no detectable canopy change (surface fire / polygon not burned in canopy)'
    if sig < 0.08: return 'weak change (low-severity fire)'
    return 'clear change, missed by thresholds / smoothing'

rows = []
for f in feats:
    y = f['anno_gee']
    if y not in (2022, 2023, 2024) or not f['categoria'].lower().startswith('bosc') or f['area_ha'] < 5: continue
    k = (y, f['com'], f['data_iso'], round(f['area_ha'], 3))
    if k not in det: continue
    anyd = any(v[0] for v in det[k].values()); r, _ = analyse(f)
    rows.append(dict(year=y, comune=f['denominazi'], loc=f['loc'], data=f['data_iso'], mese=f['mese'], area_ha=f['area_ha'],
                     supcalcb=f['supcalcb'], detected_any=int(anyd), detected_by=sum(v[0] for v in det[k].values()),
                     best_frac=max(v[1] for v in det[k].values()), **r, cause='' if anyd else classify(r, f['mese'])))
with open(os.path.join(OUT, 'diag_incendi_5ha.csv'), 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter=';'); w.writeheader(); w.writerows(rows)
print(len(rows), 'fires >= 5 ha;', sum(r['detected_any'] for r in rows), 'detected by at least one method')
from collections import Counter
print(Counter(r['cause'] for r in rows if not r['detected_any']))
for r in rows:
    if r['area_ha'] >= 50: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})

# ------------------------------------------------------------------ panels for the two large 2022 fires
targets = [f for f in feats if f['anno_gee'] == 2022 and f['denominazi'] in ('Pomarico', 'Miglionico') and f['area_ha'] > 80]
fig, ax = plt.subplots(len(targets), 4, figsize=(16, 4.2 * len(targets)), constrained_layout=True)
for i, f in enumerate(targets):
    r, D = analyse(f)
    ext = None
    panels = [(D['a_prev'], 'NBR summer 2021', 'RdYlGn', (-0.1, 0.6)), (D['pre'], 'NBR pre-season (May-Jun 2022)', 'RdYlGn', (-0.1, 0.6)),
              (D['post'], 'NBR post-season (Sep-Oct 2022)', 'RdYlGn', (-0.1, 0.6)), (D['d_sea'], 'dNBR pre - post 2022', 'magma_r', (-0.05, 0.4))]
    for j, (img, t, cm, (lo, hi)) in enumerate(panels):
        A = ax[i, j]; img = np.where(img == 0, np.nan, img)
        im = A.imshow(img, cmap=cm, vmin=lo, vmax=hi)
        A.contour(D['poly'], levels=[0.5], colors='cyan', linewidths=1.2)
        A.contour(D['forest'], levels=[0.5], colors='white', linewidths=0.4, alpha=0.6)
        A.set_xticks([]); A.set_yticks([]); A.set_title(t, fontsize=10)
        if j == 3: plt.colorbar(im, ax=A, fraction=0.04)
    ax[i, 0].set_ylabel(f"{f['denominazi']} ({f['loc']})\n{f['data_iso']}, {f['area_ha']:.0f} ha", fontsize=11)
fig.suptitle('Large 2022 forest fires missed by all methods - cyan: register polygon, white: forest mask', fontsize=12)
fig.savefig(os.path.join(OUT, 'tavola_incendi_2022.png'), dpi=150)
print('saved')
