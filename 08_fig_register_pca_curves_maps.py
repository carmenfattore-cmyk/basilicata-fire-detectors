"""Extra figures for the manuscript: register statistics, exploratory PCA, ROC/PR curves, detection maps."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import os, csv, numpy as np, rasterio, fiona, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rasterio.features import rasterize
from rasterio.windows import from_bounds
from shapely.geometry import shape
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5})
G = GE = GEE_DIR
SC = SCORE_DIR
OUT = FIG_DIR
feats = []
with fiona.open(FIRES, layer='incendi_2016_2025') as c:
    for f in c:
        p = f['properties']; feats.append(dict(geom=f['geometry'], year=p['anno_gee'], mese=p['mese'], ha=p['area_ha'], cat=str(p['categoria']),
                                               com=p['denominazi'], loc=p['loc'], data=p['data_iso']))
bosc = lambda f: f['cat'].startswith('Bosc')

# ---------------- Fig: register statistics
cats = [('Bosc', 'Wooded', '#2e7d32'), ('Non', 'Non-wooded', '#c9a227'), ('Pasc', 'Pasture', '#8d6e63'), ('N.C', 'Not classified', '#bdbdbd')]
yrs = list(range(2016, 2026))
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.2), dpi=600, gridspec_kw=dict(width_ratios=[1.55, 1]))
bot = np.zeros(len(yrs))
for key, lab, col in cats:
    v = np.array([sum(f['ha'] for f in feats if f['year'] == y and f['cat'].startswith(key)) for y in yrs])
    ax[0].bar(yrs, v, bottom=bot, color=col, label=lab, width=0.75); bot += v
for y, t in zip(yrs, bot): ax[0].text(y, t + 80, f'{t:,.0f}', ha='center', fontsize=6.5)
for y0, y1, lab in ((2017.6, 2021.4, 'training'), (2021.6, 2024.4, 'test')):
    ax[0].annotate('', xy=(y0, 7300), xytext=(y1, 7300), arrowprops=dict(arrowstyle='<->', lw=0.7, color='#555'))
    ax[0].text((y0 + y1) / 2, 7450, lab, ha='center', fontsize=7, color='#555')
ax[0].set_ylim(0, 7900); ax[0].set_xticks(yrs); ax[0].set_xticklabels([str(y) for y in yrs], rotation=45)
ax[0].set_ylabel('Burned area (ha)'); ax[0].legend(fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4); ax[0].set_title('(a) Burned area by year and land cover', fontsize=8.5, loc='left')
mon = np.array([sum(f['ha'] for f in feats if bosc(f) and f['mese'] == m) for m in range(1, 13)])
ax[1].bar(range(1, 13), mon, color='#2e7d32', width=0.75)
ax[1].set_xticks(range(1, 13)); ax[1].set_xticklabels(list('JFMAMJJASOND')); ax[1].set_ylabel('Burned woodland (ha)')
ax[1].set_title('(b) Burned woodland by month', fontsize=8.5, loc='left')
for a in ax:
    for s in ('top', 'right'): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/FigR_register.png', dpi=600); plt.close()

# ---------------- Fig: exploratory PCA of summer NBR
rows = list(csv.DictReader(open(f'{GE}/Bas_PCA_NBR_loadings.csv')))
years = [str(y) for y in range(2017, 2026)]
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.8), dpi=600, gridspec_kw=dict(width_ratios=[1, 1.6]))
ev = [float(r['explained_pct']) for r in rows]
ax[0].bar(range(1, len(ev) + 1), ev, color=['#1f6fb4'] + ['#9e9e9e'] * (len(ev) - 1)); ax[0].set_yscale('log')
for i, e in enumerate(ev[:3]): ax[0].text(i + 1, e * 1.15, f'{e:.1f}%', ha='center', fontsize=6.5)
ax[0].set_xticks(range(1, len(ev) + 1)); ax[0].set_xlabel('Component'); ax[0].set_ylabel('Explained variance (%, log scale)')
ax[0].set_title('(a) Explained variance', fontsize=8.5, loc='left')
sty = {'PC1': ('#1f6fb4', '-'), 'PC2': ('#555', '--'), 'PC3': ('#d4472a', '-'), 'PC4': ('#7b1fa2', ':')}
for r in rows:
    if r['component'] in sty:
        c, ls = sty[r['component']]; v = [float(r[f'NBR_{y}']) for y in years]
        sgn = -1 if r['component'] == 'PC1' else 1
        ax[1].plot([int(y) for y in years], [sgn * x for x in v], color=c, ls=ls, marker='o', ms=3, lw=1.1, label=r['component'] + (' (sign flipped)' if sgn < 0 else ''))
ax[1].axhline(0, color='#bbb', lw=0.6); ax[1].set_xticks([int(y) for y in years]); ax[1].set_ylabel('Loading')
ax[1].legend(fontsize=6.8, frameon=False, ncol=2, loc='lower left'); ax[1].set_title('(b) Loadings of the summer NBR of each year', fontsize=8.5, loc='left')
for a in ax:
    for s in ('top', 'right'): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/FigP_pca.png', dpi=600); plt.close()
print('register + pca done', flush=True)

# ---------------- masks (band by band)
with rasterio.open(f'{G}/Bas_forest_mask.tif') as s: FOREST = s.read(1) == 1; TR = s.transform; SH = FOREST.shape
HANSEN = rasterio.open(f'{G}/Bas_Hansen_lossyear.tif').read(1)
COMMON = FOREST.copy()
for f in ('Bas_NBR_summer_2017_2025.tif', 'Bas_NDVI_summer_2017_2025.tif', 'Bas_NBR_pre_post_2017_2025.tif', 'Bas_NDVI_pre_post_2017_2025.tif'):
    with rasterio.open(f'{G}/{f}') as s:
        for b in range(1, s.count + 1): COMMON &= s.read(b) != 0
ANY = rasterize([(f['geom'], 1) for f in feats], out_shape=SH, transform=TR, fill=0).astype(bool)
NEVER = COMMON & ~ANY & (HANSEN == 0)
rng = np.random.default_rng(3); allneg = np.flatnonzero(NEVER); NEG = rng.choice(allneg, 300000, replace=False); W = len(allneg) / len(NEG)
POS = {y: np.flatnonzero((rasterize([(f['geom'], 1) for f in feats if f['year'] == y and bosc(f)], out_shape=SH, transform=TR, fill=0) > 0) & COMMON) for y in (2022, 2023, 2024)}

def thr(k, m):
    a, s, w = np.load(f'{SC}/trainsample_{k}_{m}.npy'); p, r, t = precision_recall_curve(a, s, sample_weight=w)
    f1 = 2 * p * r / np.maximum(p + r, 1e-9); return float(t[min(int(np.argmax(f1[:-1])), len(t) - 1)])

# ---------------- Fig: ROC and precision-recall curves (test years pooled)
names = {'dNBR': 'dNBR', 'PCA': 'PCA', 'AE': 'Autoencoder', 'RF': 'Random Forest', 'UNet': 'U-Net'}
cols = {'dNBR': '#757575', 'PCA': '#2e7d32', 'AE': '#c9a227', 'RF': '#7b1fa2', 'UNet': '#1f6fb4'}
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3), dpi=600)
for m, ls in (('annual', '--'), ('seasonal', '-')):
    for k in names:
        yy, ss, ww = [], [], []
        for y in (2022, 2023, 2024):
            mp = np.load(f'{SC}/score_{k}_{m}_{y}.npy').astype(np.float32).ravel()
            yy.append(np.r_[np.ones(len(POS[y])), np.zeros(len(NEG))]); ss.append(np.r_[mp[POS[y]], mp[NEG]]); ww.append(np.r_[np.ones(len(POS[y])), np.full(len(NEG), W)])
        a, s, w = np.concatenate(yy), np.concatenate(ss), np.concatenate(ww)
        fpr, tpr, _ = roc_curve(a, s); ax[0].plot(fpr, tpr, color=cols[k], ls=ls, lw=1.1)
        p, r, _ = precision_recall_curve(a, s, sample_weight=w); ax[1].plot(r, p, color=cols[k], ls=ls, lw=1.1)
        t = thr(k, m); pr = s >= t; tp = (pr & (a == 1)).sum(); fp = (w * (pr & (a == 0))).sum()
        ax[1].plot(tp / (a == 1).sum(), tp / max(tp + fp, 1), marker='o', ms=3.5, color=cols[k], mfc=cols[k] if m == 'seasonal' else 'white')
ax[0].plot([0, 1], [0, 1], color='#ccc', lw=0.6); ax[0].set_xlabel('False positive rate'); ax[0].set_ylabel('True positive rate'); ax[0].set_title('(a) ROC curves', fontsize=8.5, loc='left')
ax[1].set_xlabel('Recall'); ax[1].set_ylabel('Precision'); ax[1].set_ylim(0, 1); ax[1].set_xlim(0, 1); ax[1].set_title('(b) Precision–recall curves (dots: chosen thresholds)', fontsize=8.5, loc='left')
h = [Line2D([], [], color=cols[k], lw=1.4, label=names[k]) for k in names] + [Line2D([], [], color='k', ls='--', lw=1, label='annual input'), Line2D([], [], color='k', ls='-', lw=1, label='seasonal input')]
fig.legend(handles=h, loc='lower center', ncol=7, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))
for x in ax:
    for s in ('top', 'right'): x.spines[s].set_visible(False)
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig(f'{OUT}/FigC_curves.png', dpi=600); plt.close()
print('curves done', flush=True)

# ---------------- Fig: detection maps, 2024 examples
T = {(k, m): thr(k, m) for k, m in (('dNBR', 'annual'), ('RF', 'annual'), ('RF', 'seasonal'), ('UNet', 'seasonal'))}
maps = {key: np.load(f'{SC}/score_{key[0]}_{key[1]}_2024.npy').astype(np.float32) >= T[key] for key in T}
with rasterio.open(f'{G}/Bas_NBR_pre_post_2017_2025.tif') as s: post = s.read(2 * (2024 - 2017) + 2).astype(np.float32) / 1e4
targets = [f for f in feats if f['year'] == 2024 and bosc(f) and f['com'] in ('Tursi', 'Accettura') and f['ha'] > 50]
panels = [(None, 'Post-season NBR 2024'), (('dNBR', 'annual'), 'dNBR (annual)'), (('RF', 'annual'), 'Random Forest (annual)'),
          (('RF', 'seasonal'), 'Random Forest (seasonal)'), (('UNet', 'seasonal'), 'U-Net (seasonal)')]
fig, ax = plt.subplots(len(targets), 5, figsize=(7.6, 1.75 * len(targets) + 0.3), dpi=600)
reg24 = [f for f in feats if f['year'] == 2024]
for i, t in enumerate(targets):
    cx, cy = shape(t['geom']).centroid.coords[0]; half = 2600
    win = from_bounds(cx - half, cy - half, cx + half, cy + half, TR).round_offsets().round_lengths()
    r0, c0 = int(win.row_off), int(win.col_off); sl = (slice(r0, r0 + int(win.height)), slice(c0, c0 + int(win.width)))
    wtr = rasterio.windows.transform(win, TR); sh = (int(win.height), int(win.width))
    poly = rasterize([(f['geom'], 1) for f in reg24], out_shape=sh, transform=wtr, fill=0)
    fm = COMMON[sl]
    for j, (key, title) in enumerate(panels):
        A = ax[i, j]
        if key is None:
            A.imshow(np.where(fm, post[sl], np.nan), cmap='RdYlGn', vmin=-0.1, vmax=0.6, interpolation='nearest')
        else:
            img = np.full(sh + (3,), 1.0); img[fm] = (0.88, 0.93, 0.86); img[maps[key][sl] & fm] = (0.83, 0.18, 0.13)
            A.imshow(img, interpolation='nearest')
        A.contour(poly, levels=[0.5], colors='#1f6fb4' if key else 'k', linewidths=0.7)
        A.set_xticks([]); A.set_yticks([])
        if i == 0: A.set_title(title, fontsize=7.2)
    ax[i, 0].set_ylabel(f"{t['com']} ({t['loc'].split(' - ')[0].title()})\n{t['data']}, {t['ha']:.0f} ha", fontsize=6.8)
    ax[i, 4].plot([sh[1] - 12 - 1000 / 30, sh[1] - 12], [sh[0] - 10, sh[0] - 10], color='k', lw=1.5); ax[i, 4].text(sh[1] - 12 - 500 / 30, sh[0] - 16, '1 km', ha='center', fontsize=6)
fig.legend(handles=[Line2D([], [], color='#1f6fb4', lw=1, label='register perimeters 2024'), matplotlib.patches.Patch(color=(0.83, 0.18, 0.13), label='flagged as burned'),
                    matplotlib.patches.Patch(color=(0.88, 0.93, 0.86), label='forest, not flagged')], loc='lower center', ncol=3, fontsize=7, frameon=False)
plt.tight_layout(rect=(0, 0.06, 1, 1)); plt.savefig(f'{OUT}/FigM_maps.png', dpi=600); plt.close()
print('maps done', [(t['com'], t['loc'], t['ha']) for t in targets])
