# -*- coding: utf-8 -*-
"""
Forest-fire detection in Basilicata - comparison of ALL methods on the SAME pixels

Methods, each on the 'annual' (summer medians) and 'seasonal' (pre/post fire-season) inputs:
  dNBR 5x5   - NBR drop, 5x5 mean (no training)
  PCA        - multitemporal standardized PCA, residual after the first K_PCA components (unsupervised)
  AE         - dense autoencoder, reconstruction error (unsupervised, non-linear analogue of PCA)
  RF         - Random Forest on the input channels + their 5x5 means (supervised)
  U-Net      - the models already trained by 01_unet_cv.py (loaded, NOT retrained)

Training years 2018-2021, test years 2022-2024 (never seen). The threshold of every method is chosen on
OUT-OF-FOLD scores (leave-one-year-out within 2018-2021), so no model is ever thresholded on data it was trained on;
the U-Net out-of-fold maps and final models come from 01_unet_cv.py (run it first).
All methods are evaluated on the same mask: forest pixels valid in BOTH inputs.

Validation
  1. pixel level : AUC, precision, recall, F1 (threshold = max F1 on the out-of-fold training-year scores)
  2. per fire    : share of register forest fires detected (>= DET_FRAC of their pixels flagged), by size class
  3. per object  : flagged patches (>= MIN_OBJ_PX pixels) that overlap a register fire of the same year

HOW TO RUN: open in Spyder (conda environment from environment.yml) and press F5, or run 'python <script>'. Paths are set in config.py.
Outputs in OUT_DIR: confronto_sintesi.txt, confronto_pixel.csv, confronto_per_incendio.csv,
                    confronto_oggetti.csv, det_<method>_<input>_<year>.tif (1 = flagged; open in QGIS)
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import os, time, csv
import numpy as np
import rasterio, fiona
from rasterio.windows import Window
from rasterio.features import rasterize
from scipy.ndimage import uniform_filter, label as cc_label
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import torch, torch.nn as nn, torch.nn.functional as F

# =============================== SETTINGS ===============================
BASE = ROOT
MODEL_DIR = UNET_DIR       # from 01_unet_cv.py: unet_<input>.pt + oof_<input>_<year>.npy
OUT_DIR = COMPARE_DIR
TRAIN = [2018, 2019, 2020, 2021]   # fitting (final models) and leave-one-year-out threshold choice
TEST = [2022, 2023, 2024]
INPUTS = ['annual', 'seasonal']
K_PCA = 2               # principal components kept as "normal" variability
DET_FRAC = 0.20         # a register fire is "detected" if >= 20% of its forest pixels are flagged
MIN_OBJ_PX = 5          # flagged patches smaller than this (5 px = 0.45 ha) are ignored in the object check
RF_TREES = 200
SAVE_MAPS = True        # binary maps of the test years (small uint8 GeoTIFFs)
SEED = 0
CROP = None             # e.g. (1500, 2500, 1500, 2500) = rows, cols window for a quick test; None = whole region
# ========================================================================
for k in ('BASE', 'GEE_DIR', 'FIRES', 'MODEL_DIR', 'OUT_DIR'):                     # optional overrides
    if os.environ.get('CONF_' + k): globals()[k] = os.environ['CONF_' + k]
if os.environ.get('CONF_CROP'): CROP = tuple(int(v) for v in os.environ['CONF_CROP'].split(','))
if os.environ.get('CONF_RF_TREES'): RF_TREES = int(os.environ['CONF_RF_TREES'])
SCORE_DIR = os.environ.get('CONF_SCORE_DIR', SCORE_DIR)   # continuous score maps of the test years (.npy), used by scripts 03 and 08
if SCORE_DIR: os.makedirs(SCORE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED); torch.manual_seed(SEED)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T0 = time.time()
def log(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)
log('Device:', DEV, torch.cuda.get_device_name(0) if DEV.type == 'cuda' else '')

# ------------------------------------------------------------------ data
WIN = Window(CROP[2], CROP[0], CROP[3] - CROP[2], CROP[1] - CROP[0]) if CROP else None
def read(fname):
    with rasterio.open(os.path.join(GEE_DIR, fname)) as s:
        a = s.read(window=WIN)
        prof = s.profile.copy()
        tr = s.window_transform(WIN) if WIN else s.transform
    return a, prof, tr

FILES = {'annual': ('Bas_NBR_summer_2017_2025.tif', 'Bas_NDVI_summer_2017_2025.tif'),
         'seasonal': ('Bas_NBR_pre_post_2017_2025.tif', 'Bas_NDVI_pre_post_2017_2025.tif')}
f, PROFILE, TRANSFORM = read('Bas_forest_mask.tif'); FOREST = f[0] == 1
HANSEN = read('Bas_Hansen_lossyear.tif')[0][0]
SHAPE = FOREST.shape
PROFILE.update(height=SHAPE[0], width=SHAPE[1], transform=TRANSFORM)
RUN_INPUTS = os.environ.get('CONF_RUN_INPUTS', ','.join(INPUTS)).split(',')   # low-RAM option: run one input at a time
DATA, VALID = {}, {}
for m in INPUTS:
    nbr, ndvi = read(FILES[m][0])[0], read(FILES[m][1])[0]
    VALID[m] = FOREST & (nbr != 0).all(0) & (ndvi != 0).all(0)
    if m in RUN_INPUTS: DATA[m] = (nbr, ndvi)
    del nbr, ndvi
COMMON = np.logical_and.reduce([VALID[m] for m in INPUTS])
INPUTS = [m for m in INPUTS if m in RUN_INPUTS]
log(f'Raster {SHAPE} | forest {int(FOREST.sum())} px | ' +
    ' | '.join(f'valid {m} {int(VALID[m].sum())}' for m in INPUTS) + f' | COMMON (evaluation mask) {int(COMMON.sum())}')

def b(arr, i): return arr[i].astype(np.float32) / 10000.

def stack(y, m):
    """input channels of year y for input m (identical to unet_basilicata_spyder.py)"""
    NBR, NDVI = DATA[m]
    if m == 'seasonal':
        def ix(kind, yy): return 2 * (yy - 2017) + (0 if kind == 'pre' else 1)
        keys = [('post', y - 1), ('pre', y), ('post', y), ('pre', y + 1)]
        nb = [b(NBR, ix(*k)) for k in keys]; nd = [b(NDVI, ix(*k)) for k in keys]
        ch = nb + nd + [nb[1] - nb[2], nb[1] - nb[3]]
    else:
        i = y - 2017
        n = [b(NBR, i - 1), b(NBR, i), b(NBR, i + 1)]; d = [b(NDVI, i - 1), b(NDVI, i), b(NDVI, i + 1)]
        ch = n + d + [n[0] - n[1], n[1] - n[2], n[0] - np.minimum(n[1], n[2])]
    S = np.stack(ch).astype(np.float32); S[:, ~VALID[m]] = 0
    return S

def smooth(a, valid):
    a = np.where(valid, a, 0).astype(np.float32)
    w = uniform_filter(valid.astype(np.float32), 5)
    return np.where(valid, uniform_filter(a, 5) / np.maximum(w, 1e-6), 0).astype(np.float32)

# ------------------------------------------------------------------ fire register -> labels
with fiona.open(FIRES, layer='incendi_2016_2025') as c:
    FEATS = []
    for ft in c:
        p = ft['properties']
        FEATS.append(dict(geom=ft['geometry'], year=int(p['anno_gee']), cat=str(p['categoria']),
                          ha=float(p['area_ha']) if p.get('area_ha') is not None else np.nan,
                          com=str(p.get('com', '')), data=str(p.get('data_iso', ''))))
def ras(sel, values=None):
    g = [(f['geom'], 1 if values is None else v) for f, v in zip(FEATS, values or [None] * len(FEATS)) if sel(f)]
    if not g: return np.zeros(SHAPE, np.int32)
    return rasterize(g, out_shape=SHAPE, transform=TRANSFORM, fill=0, dtype='int32')
ANYFIRE = ras(lambda f: True) > 0
NEVER = COMMON & ~ANYFIRE & (HANSEN == 0)
def is_forest(f): return f['cat'].lower().startswith('bosc')
POS = {y: (ras(lambda f, y=y: f['year'] == y and is_forest(f)) > 0) & COMMON for y in TRAIN + TEST}
FIRE_Y = {y: ras(lambda f, y=y: f['year'] == y) > 0 for y in TEST}            # any category, for the object check

SAMPLE = {}                                                                   # fixed pixel sample per year
for y in TRAIN + TEST:
    pos = np.flatnonzero(POS[y]); neg_all = np.flatnonzero(NEVER)
    neg = rng.choice(neg_all, min(300000, len(neg_all)), replace=False)
    SAMPLE[y] = (pos, neg, len(neg_all) / len(neg))
    log(f'  {y}: {len(pos)} burned forest px (register) | {len(neg_all)} never-burned forest px')

# ------------------------------------------------------------------ models
class Blk(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.c = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(s, x): return s.c(x)
class UNet(nn.Module):
    def __init__(s, cin, b=32):
        super().__init__()
        s.e1, s.e2, s.e3, s.e4 = Blk(cin, b), Blk(b, 2*b), Blk(2*b, 4*b), Blk(4*b, 8*b)
        s.u3, s.u2, s.u1 = Blk(12*b, 4*b), Blk(6*b, 2*b), Blk(3*b, b)
        s.drop = nn.Dropout2d(0.1); s.o = nn.Conv2d(b, 1, 1)
    def forward(s, x):
        e1 = s.e1(x); e2 = s.e2(F.max_pool2d(e1, 2)); e3 = s.e3(F.max_pool2d(e2, 2)); e4 = s.drop(s.e4(F.max_pool2d(e3, 2)))
        d3 = s.u3(torch.cat([F.interpolate(e4, scale_factor=2), e3], 1))
        d2 = s.u2(torch.cat([F.interpolate(d3, scale_factor=2), e2], 1))
        d1 = s.u1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        return s.o(d1)

def unet_predict(net, mean, std, S, valid, tile=512, pad=48):
    H, W = SHAPE; out = np.zeros(SHAPE, np.float32); n = tile + 2 * pad
    with torch.no_grad():
        for y0 in range(0, H, tile):
            for x0 in range(0, W, tile):
                ya, yb, xa, xb = max(y0 - pad, 0), min(y0 - pad + n, H), max(x0 - pad, 0), min(x0 - pad + n, W)
                blk = (S[:, ya:yb, xa:xb] - mean) / std                       # tile-wise: no full-size copies
                pw = ((0, 0), (ya - (y0 - pad), (y0 - pad + n) - yb), (xa - (x0 - pad), (x0 - pad + n) - xb))
                blk = np.pad(blk, pw, mode='reflect' if min(blk.shape[1:]) > max(max(p) for p in pw) else 'edge')
                t = torch.from_numpy(blk.astype(np.float32))[None].to(DEV)
                p = torch.sigmoid(net(t))[0, 0, pad:pad + tile, pad:pad + tile].cpu().numpy()
                out[y0:y0 + tile, x0:x0 + tile] = p[:min(tile, H - y0), :min(tile, W - x0)]
    out[~valid] = 0
    return out

class AE(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.enc = nn.Sequential(nn.Linear(c, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, K_PCA))
        s.dec = nn.Sequential(nn.Linear(K_PCA, 16), nn.ReLU(), nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, c))
    def forward(s, x): return s.dec(s.enc(x))

def rf_features(S, valid, idx):
    """(n_pixels, 2*C) features for the flat pixel indices idx: channels + their 5x5 means (low memory)"""
    F_ = np.empty((len(idx), 2 * S.shape[0]), np.float32)
    for c in range(S.shape[0]):
        F_[:, c] = S[c].ravel()[idx]; F_[:, S.shape[0] + c] = smooth(S[c], valid).ravel()[idx]
    return F_

class Method:
    """fit on the training years, then score(S, y) -> map"""
    def __init__(s, kind, m): s.kind, s.m, s.name = kind, m, f'{kind} ({m})'

    def fit(s, years=TRAIN):
        m, valid = s.m, VALID[s.m]
        if s.kind == 'dNBR':
            return
        if s.kind == 'U-Net':
            ck = torch.load(os.path.join(MODEL_DIR, f'unet_{m}.pt'), map_location='cpu', weights_only=False)
            st = ck['state']; cin = st['e1.c.0.weight'].shape[1]; bch = st['e1.c.0.weight'].shape[0]
            s.net = UNet(cin, bch).to(DEV); s.net.load_state_dict(st); s.net.eval()
            s.mean, s.std = np.asarray(ck['mean'], np.float32), np.asarray(ck['std'], np.float32)
            return
        X, Y = [], []
        for y in years:
            S = stack(y, m)
            if s.kind == 'RF':
                pos, neg, _ = SAMPLE[y]
                pos = pos if len(pos) <= 40000 else rng.choice(pos, 40000, replace=False)
                neg = neg[:80000]
                X.append(rf_features(S, valid, np.r_[pos, neg])); Y.append(np.r_[np.ones(len(pos)), np.zeros(len(neg))])
            else:                                            # PCA / AE: unsupervised, random forest pixels
                idx = rng.choice(np.flatnonzero(valid), 60000, replace=False)
                X.append(S.reshape(S.shape[0], -1)[:, idx].T)
        X = np.concatenate(X).astype(np.float32)
        if s.kind == 'RF':
            s.rf = None; import gc; gc.collect()          # free the previous fold's forest before fitting a new one
            s.rf = RandomForestClassifier(RF_TREES, min_samples_leaf=5, max_features='sqrt', n_jobs=-1, random_state=SEED)
            s.rf.fit(X, np.concatenate(Y))
            return
        s.mu, s.sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - s.mu) / s.sd
        if s.kind == 'PCA':
            s.pca = PCA(n_components=K_PCA, random_state=SEED).fit(Z)
            ev = PCA().fit(Z).explained_variance_ratio_
            log(f'    {s.name}: explained variance of PC1..PC4 = ' + ', '.join(f'{100 * v:.1f}%' for v in ev[:4]))
        else:
            s.ae = AE(Z.shape[1]).to(DEV); opt = torch.optim.Adam(s.ae.parameters(), 1e-3)
            Zt = torch.from_numpy(Z).to(DEV)
            for it in range(4000):
                bt = Zt[torch.randint(0, len(Zt), (2048,), device=DEV)]
                l = F.mse_loss(s.ae(bt), bt); opt.zero_grad(); l.backward(); opt.step()
            s.ae.eval()

    def score_year(s, y):
        """score map of year y; for RF the input stack is freed before prediction (saves RAM)"""
        if s.kind != 'RF': return s.score(stack(y, s.m))
        valid = VALID[s.m]; idx = np.flatnonzero(valid); S = stack(y, s.m)
        Fe = rf_features(S, valid, idx); del S
        out = np.zeros(SHAPE[0] * SHAPE[1], np.float32)
        for i in range(0, len(idx), 500000):
            out[idx[i:i + 500000]] = s.rf.predict_proba(Fe[i:i + 500000])[:, 1]
        return out.reshape(SHAPE)

    def score(s, S):
        m, valid = s.m, VALID[s.m]
        if s.kind == 'dNBR':
            return smooth(S[8], valid)       # seasonal: NBRpre(y) - NBRpost(y) ; annual: NBR(y-1) - min(NBR y, y+1)
        if s.kind == 'U-Net':
            return unet_predict(s.net, s.mean, s.std, S, valid)
        idx = np.flatnonzero(valid); out = np.zeros(SHAPE[0] * SHAPE[1], np.float32)
        if s.kind == 'RF':
            Fe = rf_features(S, valid, idx)
            for i in range(0, len(idx), 500000):
                out[idx[i:i + 500000]] = s.rf.predict_proba(Fe[i:i + 500000])[:, 1]
            return out.reshape(SHAPE)
        Z = (S.reshape(S.shape[0], -1)[:, idx].T - s.mu) / s.sd
        if s.kind == 'PCA':
            R = Z - s.pca.inverse_transform(s.pca.transform(Z))
            out[idx] = (R ** 2).sum(1)
        else:
            with torch.no_grad():
                for i in range(0, len(Z), 500000):
                    zt = torch.from_numpy(Z[i:i + 500000].astype(np.float32)).to(DEV)
                    out[idx[i:i + 500000]] = ((s.ae(zt) - zt) ** 2).sum(1).cpu().numpy()
        return smooth(out.reshape(SHAPE), valid)

# ------------------------------------------------------------------ evaluation helpers
SIZE_CLASSES = [(0, 1, '<1 ha'), (1, 5, '1-5 ha'), (5, 20, '5-20 ha'), (20, 100, '20-100 ha'), (100, 1e9, '>100 ha')]
FIRE_IDX = {y: [i for i, f in enumerate(FEATS) if f['year'] == y and is_forest(f)] for y in TEST}
FIRE_ID = {}
for y in TEST:
    ids = FIRE_IDX[y]
    FIRE_ID[y] = (rasterize([(FEATS[i]['geom'], k + 1) for k, i in enumerate(ids)], out_shape=SHAPE,
                            transform=TRANSFORM, fill=0, dtype='int32') if ids else np.zeros(SHAPE, np.int32))

def sample_scores(mp, y):
    pos, neg, w = SAMPLE[y]; fl = mp.ravel()
    return np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[fl[pos], fl[neg]], np.r_[np.ones(len(pos)), np.full(len(neg), w)]

def save_map(flag, name):
    out = np.full(SHAPE, 255, np.uint8); out[COMMON] = flag[COMMON].astype(np.uint8)
    prof = PROFILE.copy(); prof.update(count=1, dtype='uint8', nodata=255, compress='deflate')
    with rasterio.open(os.path.join(OUT_DIR, name), 'w', **prof) as dst: dst.write(out, 1)

PIX, FIRE_ROWS, OBJ = [], [], []
KINDS = os.environ.get('CONF_METHODS', 'dNBR,PCA,AE,RF,U-Net').split(',')
for m in INPUTS:
    for kind in KINDS:
        M = Method(kind, m)
        # threshold = max F1 on out-of-fold scores of the training years (negatives re-weighted to their true share)
        log(f'{M.name}: leave-one-year-out scores for the threshold ...')
        yy, ss, ww = [], [], []
        for y in TRAIN:
            if kind == 'dNBR':                                   # no fitting: the score itself is out-of-sample
                mp = M.score_year(y)
            elif kind == 'U-Net':                                # out-of-fold maps from 01_unet_cv.py
                mp = np.load(os.path.join(MODEL_DIR, f'oof_{m}_{y}.npy')).astype(np.float32)
                if CROP: mp = mp[CROP[0]:CROP[1], CROP[2]:CROP[3]]
            else:
                M.fit([t for t in TRAIN if t != y]); mp = M.score_year(y)
            a, sc, w = sample_scores(mp, y); yy.append(a); ss.append(sc); ww.append(w)
        log(f'{M.name}: fitting final model on {TRAIN} ...'); M.fit(TRAIN)
        p, r, t = precision_recall_curve(np.concatenate(yy), np.concatenate(ss), sample_weight=np.concatenate(ww))
        f1 = 2 * p * r / np.maximum(p + r, 1e-9); thr = float(t[min(int(np.argmax(f1[:-1])), len(t) - 1)])
        if SCORE_DIR:
            np.save(os.path.join(SCORE_DIR, f'trainsample_{kind.replace("-", "")}_{m}.npy'),
                    np.stack([np.concatenate(yy), np.concatenate(ss), np.concatenate(ww)]).astype(np.float32))
        for y in TEST:
            mp = M.score_year(y); flag = (mp >= thr) & COMMON
            if SCORE_DIR: np.save(os.path.join(SCORE_DIR, f'score_{kind.replace("-", "")}_{m}_{y}.npy'), mp.astype(np.float16) if kind in ('U-Net', 'RF') else mp)
            a, sc, w = sample_scores(mp, y); pr = sc >= thr
            tp = float(((a == 1) & pr).sum()); fn = float(((a == 1) & ~pr).sum()); fp = float((w * ((a == 0) & pr)).sum())
            PIX.append(dict(method=kind, input=m, year=y, AUC=roc_auc_score(a, sc), precision=tp / max(tp + fp, 1),
                            recall=tp / max(tp + fn, 1), F1=2 * tp / max(2 * tp + fp + fn, 1),
                            flagged_ha=float(flag.sum()) * 0.09, register_forest_ha=float(POS[y].sum()) * 0.09,
                            threshold=thr))
            # per fire
            fid = FIRE_ID[y]; sel = COMMON & (fid > 0)
            npx = np.bincount(fid[sel], minlength=len(FIRE_IDX[y]) + 1)
            nfl = np.bincount(fid[sel], weights=flag[sel].astype(np.float64), minlength=len(FIRE_IDX[y]) + 1)
            for k, i in enumerate(FIRE_IDX[y]):
                if npx[k + 1] == 0: continue
                fr = FEATS[i]
                FIRE_ROWS.append(dict(method=kind, input=m, year=y, comune=fr['com'], data=fr['data'], area_ha=fr['ha'],
                                      forest_px=int(npx[k + 1]), flagged_frac=nfl[k + 1] / npx[k + 1],
                                      detected=int(nfl[k + 1] / npx[k + 1] >= DET_FRAC)))
            # per object
            lab, n = cc_label(flag, structure=np.ones((3, 3)))
            size = np.bincount(lab.ravel(), minlength=n + 1); hit = np.bincount(lab.ravel(), weights=FIRE_Y[y].ravel(), minlength=n + 1)
            keep = np.arange(n + 1)[1:][size[1:] >= MIN_OBJ_PX]
            OBJ.append(dict(method=kind, input=m, year=y, objects=len(keep), confirmed=int((hit[keep] > 0).sum()),
                            object_precision=float((hit[keep] > 0).mean()) if len(keep) else np.nan))
            if SAVE_MAPS: save_map(flag, f'det_{kind.replace("-", "")}_{m}_{y}.tif')
            r_ = PIX[-1]
            log(f'    {y}: AUC {r_["AUC"]:.3f}  F1 {r_["F1"]:.3f}  P {r_["precision"]:.3f}  R {r_["recall"]:.3f} | '
                f'fires detected {sum(d["detected"] for d in FIRE_ROWS if d["method"] == kind and d["input"] == m and d["year"] == y)}'
                f'/{sum(1 for d in FIRE_ROWS if d["method"] == kind and d["input"] == m and d["year"] == y)} | '
                f'object precision {OBJ[-1]["object_precision"]:.2f}')
        del M
        if DEV.type == 'cuda': torch.cuda.empty_cache()

# ------------------------------------------------------------------ outputs
def write_csv(rows, name):
    with open(os.path.join(OUT_DIR, name), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter=';'); w.writeheader(); w.writerows(rows)
write_csv(PIX, 'confronto_pixel.csv'); write_csv(FIRE_ROWS, 'confronto_per_incendio.csv'); write_csv(OBJ, 'confronto_oggetti.csv')

L = [f'Train {TRAIN} (threshold: leave-one-year-out) | test {TEST} | evaluation mask: {int(COMMON.sum())} forest px valid in both inputs | '
     f'fire detected if >= {int(DET_FRAC * 100)}% of its pixels flagged', '',
     f'{"Method":<20}{"AUC":>7}{"F1":>7}{"Prec":>7}{"Rec":>7}{"Fires det.":>12}{"Obj.prec":>10}']
for m in INPUTS:
    for kind in KINDS:
        P = [r for r in PIX if r['method'] == kind and r['input'] == m]
        Fr = [r for r in FIRE_ROWS if r['method'] == kind and r['input'] == m]
        O = [r for r in OBJ if r['method'] == kind and r['input'] == m]
        L.append(f'{kind + " (" + m + ")":<20}' + ''.join(f'{np.mean([r[k] for r in P]):7.3f}' for k in ('AUC', 'F1', 'precision', 'recall')) +
                 f'{sum(r["detected"] for r in Fr):>6}/{len(Fr):<5}' + f'{np.nanmean([r["object_precision"] for r in O]):10.2f}')
L += ['', 'Fires detected by size class (test years, all three pooled)', f'{"Method":<20}' + ''.join(f'{c[2]:>12}' for c in SIZE_CLASSES)]
for m in INPUTS:
    for kind in KINDS:
        Fr = [r for r in FIRE_ROWS if r['method'] == kind and r['input'] == m]
        cells = []
        for lo, hi, _ in SIZE_CLASSES:
            g = [r for r in Fr if lo <= r['area_ha'] < hi]
            cells.append(f'{sum(r["detected"] for r in g)}/{len(g)}' if g else '-')
        L.append(f'{kind + " (" + m + ")":<20}' + ''.join(f'{c:>12}' for c in cells))
txt = '\n'.join(L); print('\n' + txt)
open(os.path.join(OUT_DIR, 'confronto_sintesi.txt'), 'w', encoding='utf-8').write(txt)
log('Done. Results in', OUT_DIR)
