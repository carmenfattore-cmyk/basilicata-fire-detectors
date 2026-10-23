"""Points 3-6: bootstrap CIs, sensitivity, detection by month, regional statistics (uses score maps saved by the comparison run)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import os, csv, json, numpy as np, rasterio, fiona
from rasterio.features import rasterize
from scipy.ndimage import uniform_filter
from sklearn.metrics import roc_auc_score, precision_recall_curve
from collections import defaultdict

G = GEE_DIR
SC = SCORE_DIR; OUT = STATS_DIR; os.makedirs(OUT, exist_ok=True)
TEST = [2022, 2023, 2024]; rng = np.random.default_rng(1)
METHODS = [(k, m) for m in ('annual', 'seasonal') for k in ('dNBR', 'PCA', 'AE', 'RF', 'UNet')]

# ---------- masks (band by band, low memory)
with rasterio.open(os.path.join(G, 'Bas_forest_mask.tif')) as s: FOREST = s.read(1) == 1; TR = s.transform; SH = FOREST.shape
HANSEN = rasterio.open(os.path.join(G, 'Bas_Hansen_lossyear.tif')).read(1)
COMMON = FOREST.copy()
for f in ('Bas_NBR_summer_2017_2025.tif', 'Bas_NDVI_summer_2017_2025.tif', 'Bas_NBR_pre_post_2017_2025.tif', 'Bas_NDVI_pre_post_2017_2025.tif'):
    with rasterio.open(os.path.join(G, f)) as s:
        for b in range(1, s.count + 1): COMMON &= s.read(b) != 0
feats = []
with fiona.open(FIRES, layer='incendi_2016_2025') as c:
    for f in c:
        p = f['properties']; feats.append(dict(geom=f['geometry'], year=p['anno_gee'], mese=p['mese'], ha=p['area_ha'], cat=p['categoria'],
                                               com=p['denominazi'], data=p['data_iso']))
ANY = rasterize([(f['geom'], 1) for f in feats], out_shape=SH, transform=TR, fill=0).astype(bool)
NEVER = COMMON & ~ANY & (HANSEN == 0)
bosc = lambda f: f['cat'].lower().startswith('bosc')
FIRE = {}
for y in TEST:
    L = [f for f in feats if f['year'] == y and bosc(f)]
    FIRE[y] = (L, rasterize([(f['geom'], i + 1) for i, f in enumerate(L)], out_shape=SH, transform=TR, fill=0, dtype='int32'))
neg_all = np.flatnonzero(NEVER); NEG = rng.choice(neg_all, 300000, replace=False); WNEG = len(neg_all) / len(NEG)
print('COMMON', COMMON.sum(), 'NEVER', len(neg_all), flush=True)

def load(k, m, y): return np.load(os.path.join(SC, f'score_{k}_{m}_{y}.npy')).astype(np.float32)
def thresholds(k, m):
    a, s, w = np.load(os.path.join(SC, f'trainsample_{k}_{m}.npy'))
    p, r, t = precision_recall_curve(a, s, sample_weight=w); f1 = 2 * p * r / np.maximum(p + r, 1e-9)
    t_f1 = float(t[min(int(np.argmax(f1[:-1])), len(t) - 1)])
    t_r50 = float(t[np.where(r[:-1] >= 0.5)[0].max()])          # highest threshold with training recall >= 0.5
    return t_f1, t_r50

# ---------- collect per-pixel samples with fire id, and per-fire flagged fractions
D = {}
for k, m in METHODS:
    t_f1, t_r50 = thresholds(k, m); rec = dict(t_f1=t_f1, t_r50=t_r50, years={})
    for y in TEST:
        mp = load(k, m, y); L, fid = FIRE[y]
        pos = np.flatnonzero((fid > 0) & COMMON); ids = fid.ravel()[pos]
        rec['years'][y] = dict(sp=mp.ravel()[pos], ids=ids, sn=mp.ravel()[NEG])
        sel = COMMON & (fid > 0); npx = np.bincount(fid[sel], minlength=len(L) + 1)
        fr = {}
        for name, t in (('f1', t_f1), ('r50', t_r50)):
            fr[name] = np.bincount(fid[sel], weights=(mp[sel] >= t).astype(float), minlength=len(L) + 1) / np.maximum(npx, 1)
            rec['years'][y]['flag_ha_' + name] = float(((mp >= t) & COMMON).sum()) * 0.09
        rec['years'][y].update(npx=npx, frac=fr)
    D[(k, m)] = rec; print(k, m, 'loaded', flush=True)

def pix_metrics(sp, sn, t):
    tp = (sp >= t).sum(); fn = len(sp) - tp; fp = (sn >= t).sum() * WNEG
    return 2 * tp / max(2 * tp + fp + fn, 1), tp / max(tp + fp, 1), tp / max(tp + fn, 1)

# ---------- 3. bootstrap over fires (cluster bootstrap) + iid over negatives
B = 300; boot = defaultdict(list)
fire_lists = {y: np.arange(1, len(FIRE[y][0]) + 1) for y in TEST}
for b in range(B):
    choice = {y: rng.choice(fire_lists[y], len(fire_lists[y]), replace=True) for y in TEST}
    negi = rng.integers(0, len(NEG), len(NEG))
    for key, rec in D.items():
        aucs, f1s = [], []
        for y in TEST:
            R = rec['years'][y]; cnt = np.bincount(choice[y], minlength=len(FIRE[y][0]) + 1)
            w = cnt[R['ids']]; keep = w > 0
            sp = np.repeat(R['sp'][keep], w[keep]); sn = R['sn'][negi]
            if len(sp) == 0: continue
            sub = rng.choice(len(sp), min(len(sp), 60000), replace=False)
            sp = sp[sub]
            aucs.append(roc_auc_score(np.r_[np.ones(len(sp)), np.zeros(len(sn))], np.r_[sp, sn]))
            f1s.append(pix_metrics(sp, sn, rec['t_f1'])[0])
        boot[key].append((np.mean(aucs), np.mean(f1s)))
    if b % 50 == 0: print('bootstrap', b, flush=True)
point = {}
for key, rec in D.items():
    aucs, f1s = [], []
    for y in TEST:
        R = rec['years'][y]
        aucs.append(roc_auc_score(np.r_[np.ones(len(R['sp'])), np.zeros(len(R['sn']))], np.r_[R['sp'], R['sn']]))
        f1s.append(pix_metrics(R['sp'], R['sn'], rec['t_f1'])[0])
    point[key] = (np.mean(aucs), np.mean(f1s))
rows = []
for key in D:
    bb = np.array(boot[key])
    rows.append(dict(method=key[0], input=key[1], AUC=point[key][0], AUC_lo=np.percentile(bb[:, 0], 2.5), AUC_hi=np.percentile(bb[:, 0], 97.5),
                     F1=point[key][1], F1_lo=np.percentile(bb[:, 1], 2.5), F1_hi=np.percentile(bb[:, 1], 97.5)))
ref_pairs = [(('UNet', 'seasonal'), ('dNBR', 'annual')), (('UNet', 'seasonal'), ('RF', 'seasonal')), (('RF', 'annual'), ('dNBR', 'annual')),
             (('PCA', 'annual'), ('dNBR', 'annual')), (('UNet', 'seasonal'), ('UNet', 'annual')), (('AE', 'annual'), ('PCA', 'annual'))]
pairs = []
for a, b in ref_pairs:
    d = np.array(boot[a]) - np.array(boot[b])
    pairs.append(dict(A=' '.join(a), B=' '.join(b), dAUC=point[a][0] - point[b][0], dAUC_lo=np.percentile(d[:, 0], 2.5), dAUC_hi=np.percentile(d[:, 0], 97.5),
                      dF1=point[a][1] - point[b][1], dF1_lo=np.percentile(d[:, 1], 2.5), dF1_hi=np.percentile(d[:, 1], 97.5)))

# ---------- 4/5. per-fire detection: DET_FRAC and threshold sensitivity, by month and size
det_rows = []; month = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for key, rec in D.items():
    for thr in ('f1', 'r50'):
        for df in (0.1, 0.2, 0.5):
            n = d = 0; big = [0, 0]
            for y in TEST:
                L = FIRE[y][0]; R = rec['years'][y]
                for i, f in enumerate(L):
                    if R['npx'][i + 1] == 0: continue
                    hit = R['frac'][thr][i + 1] >= df; n += 1; d += hit
                    if f['ha'] >= 5: big[0] += 1; big[1] += hit
                    if thr == 'f1' and df == 0.2: month[key][f['mese']][0] += 1; month[key][f['mese']][1] += hit
            det_rows.append(dict(method=key[0], input=key[1], threshold=thr, det_frac=df, detected=d, fires=n, rate=d / n,
                                 detected_5ha=big[1], fires_5ha=big[0], rate_5ha=big[1] / big[0],
                                 flagged_ha=sum(rec['years'][y]['flag_ha_' + thr] for y in TEST)))
# pixel metrics at the recall-oriented threshold
r50 = []
for key, rec in D.items():
    ms = [pix_metrics(rec['years'][y]['sp'], rec['years'][y]['sn'], rec['t_r50']) for y in TEST]
    r50.append(dict(method=key[0], input=key[1], F1=np.mean([x[0] for x in ms]), precision=np.mean([x[1] for x in ms]), recall=np.mean([x[2] for x in ms])))

# smoothing on/off for dNBR (raw maps recomputed from the composites)
def raw_dnbr(m, y):
    if m == 'annual':
        with rasterio.open(os.path.join(G, 'Bas_NBR_summer_2017_2025.tif')) as s:
            a = [s.read(y - 2017 + i).astype(np.float32) / 1e4 for i in (0, 1, 2)]
        return a[0] - np.minimum(a[1], a[2])
    with rasterio.open(os.path.join(G, 'Bas_NBR_pre_post_2017_2025.tif')) as s:
        return (s.read(2 * (y - 2017) + 1).astype(np.float32) - s.read(2 * (y - 2017) + 2).astype(np.float32)) / 1e4
smooth_rows = []
for m in ('annual', 'seasonal'):
    for sm in (False, True):
        aucs = []
        for y in TEST:
            d = raw_dnbr(m, y); d[~COMMON] = 0
            if sm:
                v = uniform_filter(COMMON.astype(np.float32), 5); d = uniform_filter(d, 5) / np.maximum(v, 1e-6)
            R = D[('dNBR', m)]['years'][y]; L, fid = FIRE[y]
            pos = np.flatnonzero((fid > 0) & COMMON)
            aucs.append(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(NEG))], np.r_[d.ravel()[pos], d.ravel()[NEG]]))
        smooth_rows.append(dict(input=m, smoothing='5x5' if sm else 'none', AUC=np.mean(aucs), **{f'AUC_{y}': a for y, a in zip(TEST, aucs)}))

# ---------- 6. regional statistics from the register
reg = defaultdict(lambda: defaultdict(float)); comuni = defaultdict(float); cnt = defaultdict(int)
for f in feats:
    reg[f['year']][f['cat']] += f['ha']; cnt[(f['year'], f['cat'])] += 1
    if bosc(f): comuni[f['com']] += f['ha']
regional = [dict(year=y, **{c: round(reg[y][c], 1) for c in ('Boscata', 'Non boscata', 'Pascolo', 'N.C.')},
                 n_forest_fires=cnt[(y, 'Boscata')]) for y in sorted(reg)]
top = sorted(comuni.items(), key=lambda x: -x[1])[:15]
months_all = defaultdict(lambda: [0, 0.0])
for f in feats:
    if bosc(f): months_all[f['mese']][0] += 1; months_all[f['mese']][1] += f['ha']

def w(rows, name):
    with open(os.path.join(OUT, name), 'w', newline='', encoding='utf-8') as fh:
        ww = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter=';'); ww.writeheader()
        for r in rows: ww.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
w(rows, 'bootstrap_ci.csv'); w(pairs, 'bootstrap_differenze.csv'); w(det_rows, 'sensibilita_rilevamento.csv'); w(r50, 'pixel_soglia_recall50.csv')
w(smooth_rows, 'smoothing_dNBR.csv'); w(regional, 'catasto_per_anno.csv')
w([dict(comune=c, forest_ha=round(h, 1)) for c, h in top], 'catasto_top_comuni.csv')
w([dict(method=k[0], input=k[1], **{f'm{mm}': f"{v[1]}/{v[0]}" for mm, v in sorted(month[k].items())}) for k in D], 'rilevamento_per_mese.csv')
w([dict(mese=mm, n=v[0], ha=round(v[1], 1)) for mm, v in sorted(months_all.items())], 'catasto_per_mese.csv')
# ---------- 7. union of an annual and a seasonal method (complementarity by month)
def union_eval(k1, k2):
    out = dict(pair=f'{k1[0]} {k1[1]} + {k2[0]} {k2[1]}')
    ms, det_n, det_d, mon = [], 0, 0, defaultdict(lambda: [0, 0])
    for y in TEST:
        a = load(*k1, y) >= D[k1]['t_f1']; b = load(*k2, y) >= D[k2]['t_f1']; fl = (a | b) & COMMON
        L, fid = FIRE[y]; pos = np.flatnonzero((fid > 0) & COMMON)
        tp = fl.ravel()[pos].sum(); fn = len(pos) - tp; fp = fl.ravel()[NEG].sum() * WNEG
        ms.append((2 * tp / max(2 * tp + fp + fn, 1), tp / max(tp + fp, 1), tp / max(tp + fn, 1)))
        sel = COMMON & (fid > 0); npx = np.bincount(fid[sel], minlength=len(L) + 1)
        fr = np.bincount(fid[sel], weights=fl[sel].astype(float), minlength=len(L) + 1) / np.maximum(npx, 1)
        for i, f in enumerate(L):
            if npx[i + 1] == 0: continue
            hit = fr[i + 1] >= 0.2; det_n += 1; det_d += hit; mon[f['mese']][0] += 1; mon[f['mese']][1] += hit
    out.update(F1=np.mean([m[0] for m in ms]), precision=np.mean([m[1] for m in ms]), recall=np.mean([m[2] for m in ms]),
               detected=det_d, fires=det_n, **{f'm{mm}': f'{v[1]}/{v[0]}' for mm, v in sorted(mon.items())})
    return out
U = [union_eval(a, b) for a, b in [(('RF', 'annual'), ('RF', 'seasonal')), (('dNBR', 'annual'), ('UNet', 'seasonal')),
                                    (('RF', 'annual'), ('UNet', 'seasonal')), (('PCA', 'annual'), ('UNet', 'seasonal'))]]
w(U, 'unione_annuale_stagionale.csv')
print('done')
