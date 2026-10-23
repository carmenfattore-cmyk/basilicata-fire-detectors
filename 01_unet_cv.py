# -*- coding: utf-8 -*-
"""
U-Net for forest-fire detection in Basilicata - leave-one-year-out cross-validation version

Why: the threshold of a supervised model must be chosen on data the model has NOT been trained on.
Holding out 2021 as a validation year removes 55% of the burned training pixels (2021 is by far the
largest training year), so instead every training year is held out once:

  for each year Y in 2018-2021: train on the other three years, predict Y  -> out-of-fold maps (oof_*.npy)
  final model: train on all four years 2018-2021                          -> unet_<input>.pt
  The comparison script picks the threshold on the out-of-fold maps and tests the final model on 2022-2024.

HOW TO RUN: open in Spyder (conda environment from environment.yml) and press F5, or run 'python <script>'. Both inputs (annual, seasonal) are
processed in one run: 10 trainings, several hours on a laptop GPU. Paths are set in config.py.
Outputs in OUT_DIR: oof_<input>_<year>.npy, unet_<input>.pt, metrics_cv_<input>.txt
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import os, time, gc
import numpy as np
import rasterio, fiona
from rasterio.features import rasterize
from scipy.ndimage import uniform_filter
from sklearn.metrics import roc_auc_score, precision_recall_curve
import torch, torch.nn as nn, torch.nn.functional as F

# =============================== SETTINGS ===============================
BASE = ROOT
INPUTS = ['annual', 'seasonal']
TRAIN = [2018, 2019, 2020, 2021]
TEST = [2022, 2023, 2024]
PATCH = 128; BATCH = 32; EPOCHS = 40; STEPS_PER_EPOCH = 250; BASE_CH = 32
OUT_DIR = UNET_DIR
SEED = 0
# ========================================================================
for k in ('BASE', 'GEE_DIR', 'FIRES', 'OUT_DIR', 'EPOCHS', 'STEPS_PER_EPOCH', 'BASE_CH'):
    if os.environ.get('UNET_' + k): globals()[k] = type(globals()[k])(os.environ['UNET_' + k])
if os.environ.get('UNET_INPUTS'): INPUTS = os.environ['UNET_INPUTS'].split(',')
os.makedirs(OUT_DIR, exist_ok=True)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T0 = time.time()
def log(*a): print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)
log('Device:', DEV, torch.cuda.get_device_name(0) if DEV.type == 'cuda' else '')

FILES = {'annual': ('Bas_NBR_summer_2017_2025.tif', 'Bas_NDVI_summer_2017_2025.tif'),
         'seasonal': ('Bas_NBR_pre_post_2017_2025.tif', 'Bas_NDVI_pre_post_2017_2025.tif')}
with rasterio.open(os.path.join(GEE_DIR, 'Bas_forest_mask.tif')) as s: FOREST = s.read(1) == 1; TRANSFORM = s.transform
HANSEN = rasterio.open(os.path.join(GEE_DIR, 'Bas_Hansen_lossyear.tif')).read(1)
SHAPE = FOREST.shape
with fiona.open(FIRES, layer='incendi_2016_2025') as c:
    FEATS = [(f['geometry'], f['properties']['anno_gee'], str(f['properties']['categoria'])) for f in c]
def ras(cond):
    g = [(geom, 1) for geom, y, cat in FEATS if cond(y, cat)]
    return rasterize(g, out_shape=SHAPE, transform=TRANSFORM, fill=0).astype(bool) if g else np.zeros(SHAPE, bool)
ANYFIRE = ras(lambda y, c: True)

# ------------------------------------------------------------------ model
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

def loss_fn(logit, y, posw):
    m = (y >= 0).float(); t = y.clamp(min=0)
    bce = (F.binary_cross_entropy_with_logits(logit, t, pos_weight=posw, reduction='none') * m).sum() / m.sum().clamp(min=1)
    p = torch.sigmoid(logit) * m; dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + (t * m).sum() + 1)
    return bce + 0.5 * dice

def run_input(INPUT):
    NBR = rasterio.open(os.path.join(GEE_DIR, FILES[INPUT][0])).read()
    NDVI = rasterio.open(os.path.join(GEE_DIR, FILES[INPUT][1])).read()
    VALID = FOREST & (NBR != 0).all(0) & (NDVI != 0).all(0)
    NEVER = VALID & ~ANYFIRE & (HANSEN == 0)
    log(f'=== input {INPUT}: valid forest px {int(VALID.sum())}')

    def b(a, i): return a[i].astype(np.float32) / 10000.
    def stack(y):                                   # identical to unet_basilicata_spyder.py
        if INPUT == 'seasonal':
            def ix(kind, yy): return 2 * (yy - 2017) + (0 if kind == 'pre' else 1)
            keys = [('post', y - 1), ('pre', y), ('post', y), ('pre', y + 1)]
            nb = [b(NBR, ix(*k)) for k in keys]; nd = [b(NDVI, ix(*k)) for k in keys]
            ch = nb + nd + [nb[1] - nb[2], nb[1] - nb[3]]
        else:
            i = y - 2017
            n = [b(NBR, i - 1), b(NBR, i), b(NBR, i + 1)]; d = [b(NDVI, i - 1), b(NDVI, i), b(NDVI, i + 1)]
            ch = n + d + [n[0] - n[1], n[1] - n[2], n[0] - np.minimum(n[1], n[2])]
        S = np.stack(ch).astype(np.float32); S[:, ~VALID] = 0
        return S
    def labels(y):
        L = np.full(SHAPE, -1, np.int8); L[NEVER] = 0
        L[ras(lambda yy, c, y=y: yy == y and c.startswith('Bosc')) & VALID] = 1
        return L

    def train(years, tag):
        torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
        TR = {}
        for y in years:
            S = stack(y); L = labels(y)
            TR[y] = dict(S=S.astype(np.float16), L=L, pos=np.argwhere(L == 1), val=np.argwhere(VALID))
            del S
        C = TR[years[0]]['S'].shape[0]
        samp = np.concatenate([TR[y]['S'][:, ::7, ::7].reshape(C, -1) for y in years], 1).astype(np.float32)
        MEAN = samp.mean(1)[:, None, None].astype(np.float32); STD = samp.std(1)[:, None, None].astype(np.float32) + 1e-6
        del samp
        H, W = SHAPE
        def batch():
            X = np.empty((BATCH, C, PATCH, PATCH), np.float32); Y = np.empty((BATCH, PATCH, PATCH), np.float32)
            for k in range(BATCH):
                y = years[rng.integers(len(years))]; d = TR[y]
                arr = d['pos'] if (rng.random() < 0.6 and len(d['pos'])) else d['val']
                cy, cx = arr[rng.integers(len(arr))]
                y0 = int(np.clip(cy - PATCH // 2 + rng.integers(-PATCH // 3, PATCH // 3), 0, H - PATCH))
                x0 = int(np.clip(cx - PATCH // 2 + rng.integers(-PATCH // 3, PATCH // 3), 0, W - PATCH))
                X[k] = (d['S'][:, y0:y0 + PATCH, x0:x0 + PATCH].astype(np.float32) - MEAN) / STD
                Y[k] = d['L'][y0:y0 + PATCH, x0:x0 + PATCH]
            X = torch.from_numpy(X); Y = torch.from_numpy(Y)
            if rng.random() < 0.5: X, Y = X.flip(-1), Y.flip(-1)
            if rng.random() < 0.5: X, Y = X.flip(-2), Y.flip(-2)
            k = int(rng.integers(4)); X, Y = torch.rot90(X, k, (-2, -1)), torch.rot90(Y, k, (-2, -1))
            return X.to(DEV), Y.to(DEV)
        net = UNet(C, BASE_CH).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * STEPS_PER_EPOCH)
        posw = torch.tensor(3.0, device=DEV)
        for ep in range(EPOCHS):
            net.train(); tot = 0
            for _ in range(STEPS_PER_EPOCH):
                X, Y = batch(); l = loss_fn(net(X)[:, 0], Y, posw)
                opt.zero_grad(); l.backward(); opt.step(); sched.step(); tot += l.item()
            if (ep + 1) % 10 == 0: log(f'  {tag}: epoch {ep + 1}/{EPOCHS} loss {tot / STEPS_PER_EPOCH:.4f}')
        del TR; gc.collect()
        net.eval()
        return net, MEAN, STD

    def predict(net, MEAN, STD, y, tile=512, pad=48):
        S = stack(y); H, W = SHAPE; out = np.zeros(SHAPE, np.float32); n = tile + 2 * pad
        with torch.no_grad():
            for y0 in range(0, H, tile):
                for x0 in range(0, W, tile):
                    ya, yb, xa, xb = max(y0 - pad, 0), min(y0 - pad + n, H), max(x0 - pad, 0), min(x0 - pad + n, W)
                    blk = (S[:, ya:yb, xa:xb] - MEAN) / STD
                    pw = ((0, 0), (ya - (y0 - pad), (y0 - pad + n) - yb), (xa - (x0 - pad), (x0 - pad + n) - xb))
                    blk = np.pad(blk, pw, mode='reflect' if min(blk.shape[1:]) > max(max(p) for p in pw) else 'edge')
                    p = torch.sigmoid(net(torch.from_numpy(blk.astype(np.float32))[None].to(DEV)))[0, 0, pad:pad + tile, pad:pad + tile].cpu().numpy()
                    out[y0:y0 + tile, x0:x0 + tile] = p[:min(tile, H - y0), :min(tile, W - x0)]
        out[~VALID] = 0
        return out

    rng_eval = np.random.default_rng(SEED)
    def sample(m, y):
        L = labels(y).ravel(); pos = np.flatnonzero(L == 1); allneg = np.flatnonzero(L == 0)
        neg = rng_eval.choice(allneg, min(300000, len(allneg)), replace=False)
        return np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[m.ravel()[pos], m.ravel()[neg]], np.r_[np.ones(len(pos)), np.full(len(neg), len(allneg) / len(neg))]

    # 1) out-of-fold predictions
    yy, ss, ww = [], [], []
    for f in TRAIN:
        net, MEAN, STD = train([y for y in TRAIN if y != f], f'fold {f}')
        m = predict(net, MEAN, STD, f)
        np.save(os.path.join(OUT_DIR, f'oof_{INPUT}_{f}.npy'), m.astype(np.float16))
        a, s_, w_ = sample(m, f); yy.append(a); ss.append(s_); ww.append(w_)
        log(f'  out-of-fold {f}: AUC {roc_auc_score(a, s_):.3f}')
        del net; torch.cuda.empty_cache() if DEV.type == 'cuda' else None
    p, r, t = precision_recall_curve(np.concatenate(yy), np.concatenate(ss), sample_weight=np.concatenate(ww))
    f1 = 2 * p * r / np.maximum(p + r, 1e-9); thr = float(t[min(int(np.argmax(f1[:-1])), len(t) - 1)])
    # 2) final model on all training years
    net, MEAN, STD = train(TRAIN, 'final')
    torch.save({'state': net.state_dict(), 'mean': MEAN, 'std': STD, 'input': INPUT, 'train': TRAIN, 'threshold_oof': thr},
               os.path.join(OUT_DIR, f'unet_{INPUT}.pt'))
    lines = [f'Input {INPUT} | final model trained on {TRAIN} | threshold (max F1 on out-of-fold maps) = {thr:.3f}']
    for y in TEST:
        m = predict(net, MEAN, STD, y); a, s_, w_ = sample(m, y); pr = s_ >= thr
        tp = float((pr & (a == 1)).sum()); fn = float((~pr & (a == 1)).sum()); fp = float((w_ * (pr & (a == 0))).sum())
        lines.append(f'  {y}: AUC {roc_auc_score(a, s_):.3f}  F1 {2 * tp / max(2 * tp + fp + fn, 1):.3f}  '
                     f'precision {tp / max(tp + fp, 1):.3f}  recall {tp / max(tp + fn, 1):.3f}')
    txt = '\n'.join(lines); log('\n' + txt)
    open(os.path.join(OUT_DIR, f'metrics_cv_{INPUT}.txt'), 'w').write(txt)
    del NBR, NDVI, net; gc.collect()

for INPUT in INPUTS:
    run_input(INPUT)
log('Done. Now run confronto_metodi_spyder.py')
