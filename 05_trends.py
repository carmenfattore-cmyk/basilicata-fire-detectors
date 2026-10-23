"""Satellite data statistics (clear observations), NBR trend 2017-2025 (Theil-Sen + Mann-Kendall) and post-fire recovery by cohort."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import json, numpy as np, rasterio, fiona, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from rasterio.features import rasterize
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5})
G = GEE_DIR; OUT = FIG_DIR
NOBS = os.path.join(GEE_DIR, 'Bas_nobs_pre_post.tif')
R = {}
with rasterio.open(f'{G}/Bas_forest_mask.tif') as s: F = s.read(1) == 1; TR = s.transform; SH = F.shape; b = s.bounds

# ---- clear observations per seasonal composite (forest pixels)
with rasterio.open(NOBS) as s:
    names = list(s.descriptions); nob = {}
    for i in range(s.count):
        a = s.read(i + 1)[F].astype(np.float32)
        nob[i] = (float(np.median(a)), float(np.percentile(a, 10)), float((a < 3).mean() * 100), float((a == 0).mean() * 100))
R['nobs_bands'] = names
R['nobs'] = {f'{2017 + i // 2}_{"pre" if i % 2 == 0 else "post"}': v for i, v in nob.items()}
print(R['nobs'], flush=True)

# ---- annual NBR stack
with rasterio.open(f'{G}/Bas_NBR_summer_2017_2025.tif') as s: NBR = s.read().astype(np.float32) / 1e4
V = F & (NBR != 0).all(0)
idx = np.flatnonzero(V.ravel()); X = NBR.reshape(9, -1)[:, idx]               # 9 x n
yrs = np.arange(2017, 2026)
# Theil-Sen slope and Mann-Kendall S (n = 9, 36 pairs)
slopes = []; S = np.zeros(X.shape[1], np.int16)
for i in range(9):
    for j in range(i + 1, 9):
        d = X[j] - X[i]; slopes.append((d / (j - i)).astype(np.float32)); S += np.sign(d).astype(np.int16)
slope = np.median(np.stack(slopes), axis=0); del slopes
sig = np.abs(S) >= 20                                                         # two-sided p < 0.05 (normal approx., var(S) = 92)
R['n_valid'] = int(len(idx))
R['share_sig_neg'] = float(((slope < 0) & sig).mean() * 100); R['share_sig_pos'] = float(((slope > 0) & sig).mean() * 100)
R['median_slope_per_decade'] = float(np.median(slope) * 10)
# relation with fires
feats = []
with fiona.open(FIRES, layer='incendi_2016_2025') as c:
    for f in c:
        p = f['properties']; feats.append((f['geometry'], p['anno_gee'], str(p['categoria'])))
yearmap = np.zeros(SH, np.int16); count = np.zeros(SH, np.int8)
for y in range(2016, 2026):
    m = rasterize([(g, 1) for g, yy, c in feats if yy == y and c.startswith('Bosc')] or [(feats[0][0], 0)], out_shape=SH, transform=TR, fill=0).astype(bool)
    yearmap[m] = y; count += m.astype(np.int8)
anyfire = rasterize([(g, 1) for g, yy, c in feats], out_shape=SH, transform=TR, fill=0).astype(bool)
fy = yearmap.ravel()[idx]; cnt = count.ravel()[idx]; unb = ~anyfire.ravel()[idx]
R['unburned'] = dict(sig_neg=float(((slope < 0) & sig)[unb].mean() * 100), sig_pos=float(((slope > 0) & sig)[unb].mean() * 100))
R['by_fire_year'] = {}
for y in range(2017, 2025):
    m = (fy == y) & (cnt == 1)
    if m.sum(): R['by_fire_year'][y] = dict(n=int(m.sum()), med_slope=float(np.median(slope[m]) * 10), sig_neg=float(((slope < 0) & sig)[m].mean() * 100), sig_pos=float(((slope > 0) & sig)[m].mean() * 100))
# regional mean NBR of unburned forest by year
R['unburned_mean_nbr'] = {int(y): float(X[i][unb].mean()) for i, y in enumerate(yrs)}
# cohort anomalies: mean NBR of cohort - mean NBR of unburned forest, per year
coh = {}
for y in range(2017, 2025):
    m = (fy == y) & (cnt == 1)
    if m.sum() > 200: coh[y] = [float(X[i][m].mean() - X[i][unb].mean()) for i in range(9)]
R['cohorts'] = coh
json.dump(R, open(os.path.join(TREND_DIR, 'trends.json'), 'w'), indent=1)

# ---- figure: trend map + cohort anomalies
sm = np.full(V.size, np.nan, np.float32); sm[idx] = np.where(sig, slope * 10, np.nan)
sm = sm.reshape(SH)
# ---- figure A: trend map (own figure, true aspect)
ext = [b.left, b.right, b.bottom, b.top]
fig, ax0 = plt.subplots(figsize=(6.0, 6.6), dpi=600)
ax0.imshow(np.where(F, 0, np.nan)[::2, ::2], cmap=matplotlib.colors.ListedColormap(['#e6e6e6']), extent=ext, interpolation='nearest')
im = ax0.imshow(sm[::2, ::2], cmap='RdYlGn', vmin=-0.3, vmax=0.3, extent=ext, interpolation='nearest')
import pyproj as _pp
from shapely.geometry import shape as _sh
from shapely.ops import transform as _tf
_t = _pp.Transformer.from_crs(4326, 32633, always_xy=True).transform
for _f in json.load(open(os.path.join(GEE_DIR, 'Bas_boundary.geojson')))['features']:
    _g = _tf(_t, _sh(_f['geometry']))
    for _p in getattr(_g, 'geoms', [_g]):
        _x, _y = _p.exterior.xy; ax0.plot(_x, _y, color='#444', lw=0.6)
ax0.set_aspect('equal'); ax0.set_xticks([]); ax0.set_yticks([]); [sp.set_visible(False) for sp in ax0.spines.values()]
x0 = b.left + 6000; y0 = b.bottom + 6000; ax0.plot([x0, x0 + 20000], [y0, y0], color='k', lw=1.5); ax0.text(x0 + 10000, y0 + 2500, '20 km', ha='center', fontsize=8)
ax0.annotate('N', xy=(b.right - 12000, b.top - 8000), xytext=(b.right - 12000, b.top - 22000), ha='center', fontsize=10, arrowprops=dict(arrowstyle='-|>', color='k'))
cb = fig.colorbar(im, ax=ax0, orientation='horizontal', fraction=0.035, pad=0.02, shrink=0.75)
cb.set_label('Theil\u2013Sen slope of summer NBR per decade (Mann\u2013Kendall p < 0.05)\ngrey: forest without significant trend', fontsize=7.5); cb.ax.tick_params(labelsize=7.5)
plt.savefig(f'{OUT}/FigT_map.png', dpi=600, bbox_inches='tight'); plt.close()
# ---- figure B: cohort trajectories
fig, ax1 = plt.subplots(figsize=(6.4, 3.6), dpi=600)
sel = [y for y in (2017, 2019, 2020, 2021, 2022) if y in coh]
cols = dict(zip(sel, ['#1b5e20', '#1f6fb4', '#7b1fa2', '#d4472a', '#c9a227']))
for y in sel:
    v = coh[y]; rel = yrs - y
    ax1.plot(rel, v, color=cols[y], lw=1.4, marker='o', ms=3, label=f'{y} (n = {R["by_fire_year"][y]["n"]:,} px)')
ax1.axhline(0, color='#999', lw=0.6); ax1.axvline(0, color='#bbb', lw=0.6, ls='--')
ax1.set_xlim(-5.5, 8.5); ax1.set_xticks(range(-5, 9))
ax1.set_xlabel('Years since fire'); ax1.set_ylabel('Summer NBR of burned stands\nminus unburned forest')
ax1.legend(title='Fire year', fontsize=7, title_fontsize=7.5, frameon=False, loc='lower right')
for s_ in ('top', 'right'): ax1.spines[s_].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/FigT_traj.png', dpi=600); plt.close()
print('figs done')
