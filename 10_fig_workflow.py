"""Workflow figure: numbered steps 1-5 matching Sections 3.1-3.5, top to bottom."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
plt.rcParams.update({'font.family': 'DejaVu Sans'})
BLUE = '#1f6fb4'; GREY = '#8a8a8a'; FILL_DATA = '#f3f3f3'; FILL_STEP = '#eaf2fb'
fig, ax = plt.subplots(figsize=(7.6, 9.6), dpi=600)
ax.set_xlim(0, 100); ax.set_ylim(128, 0); ax.axis('off')
RX = 90  # register column (right)

def box(x0, y0, w, h, fill='white', ec=GREY, lw=0.9):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0,rounding_size=1.2', fc=fill, ec=ec, lw=lw))
def title(x, y, t, fs=7.4): ax.text(x, y, t, fontsize=fs, fontweight='bold', va='center')
def lines(x, y, ls, fs=6.3, dy=2.5, c='#333'):
    for i, l in enumerate(ls): ax.text(x, y + i * dy, l, fontsize=fs, va='center', color=c)
def step(x, y, n):
    ax.add_patch(Circle((x, y), 2.0, fc=BLUE, ec='none')); ax.text(x, y + 0.1, str(n), color='white', fontsize=7.5, fontweight='bold', ha='center', va='center')
def arrow(p, q):
    ax.annotate('', xy=q, xytext=p, arrowprops=dict(arrowstyle='-|>', color='#666', lw=0.9, mutation_scale=8))
def poly(pts):
    xs, ys = zip(*pts); ax.plot(xs, ys, color='#666', lw=0.9)
def note(x, y, t, ha='left'): ax.text(x, y, t, fontsize=6, color='#555', style='italic', ha=ha, va='center')

# ---------------- data (Section 2)
ax.text(1, 2.5, 'Data (Section 2)', fontsize=7.8, fontweight='bold', color='#555', va='center')
D = [(1, 'Sentinel-2 L2A', ['2017–2025, Earth Engine', 'cloud mask; NBR, NDVI']),
     (24, 'Median composites', ['summer: 15 Jun–15 Sep', 'pre: May–Jun, post: Sep–Oct']),
     (49, 'Forest mask, 30 m', ['WorldCover 2020 trees', 'or GFC cover ≥ 30%']),
     (77, 'Fire register', ['2016–2025, RSDI', 'open data, yearly'])]
for x0, t, ls in D:
    w = {1: 21.5, 24: 23, 49: 20}.get(x0, 22)
    box(x0, 5, w, 13, fill=FILL_DATA); title(x0 + 1.4, 8.6, t, fs=6.9); lines(x0 + 1.4, 12.2, ls, fs=6.0)
arrow((22.5, 11.5), (24, 11.5))

# ---------------- step 1 (3.1)
box(1, 24, 72, 15, fill=FILL_STEP, ec=BLUE, lw=1.1); step(4.5, 28, 1)
title(8, 28, 'Input stack for each fire year y  (Section 3.1)')
lines(4, 32.2, ['annual input: NBR, NDVI of y−1, y, y+1 and three NBR drops (9 channels)', 'seasonal input: post y−1, pre y, post y, pre y+1 and two NBR drops (10 channels)'], fs=6.2)
poly([(35.5, 18), (35.5, 21)]); poly([(59, 18), (59, 21)]); poly([(35.5, 21), (59, 21)]); arrow((40, 21), (40, 24))

# ---------------- step 2 (3.2)
box(1, 45, 72, 26, fill=FILL_STEP, ec=BLUE, lw=1.1); step(4.5, 49, 2)
title(8, 49, 'Five detectors, each run on both inputs  (Section 3.2)')
B = [(3.5, 'Index', ['dNBR', 'no training']), (26.5, 'Unsupervised', ['PCA residual', 'autoencoder']), (49.5, 'Supervised', ['Random Forest', 'U-Net'])]
for x0, t, ls in B:
    box(x0, 53, 21, 15); title(x0 + 1.3, 56.5, t, fs=6.7); lines(x0 + 1.3, 60.3, ls, fs=6.2)
arrow((40, 39), (40, 45))

# ---------------- step 3 (3.3)
box(1, 77, 72, 15, fill=FILL_STEP, ec=BLUE, lw=1.1); step(4.5, 81, 3)
title(8, 81, 'Labels and experimental design  (Section 3.3)')
lines(4, 85.2, ['fit on 2018–2021; threshold = max F1 on leave-one-year-out scores;', 'final models applied to the test years 2022–2024'], fs=6.2)
arrow((40, 71), (40, 77))

# ---------------- step 4 (3.4)
box(1, 98, 98, 17, fill=FILL_STEP, ec=BLUE, lw=1.1); step(4.5, 102, 4)
title(8, 102, 'Validation against the register, 2022–2024  (Section 3.4)')
V = [(3.5, 'Per pixel', ['AUC, precision,', 'recall, F1']), (27.5, 'Per fire', ['≥ 20% flagged;', 'size and month']),
     (51.5, 'Per flagged patch', ['patch precision,', 'area flagged']), (75.5, 'Uncertainty', ['bootstrap; missed', 'fires; inputs OR'])]
for x0, t, ls in V:
    box(x0, 105.5, 21, 8.5); title(x0 + 1.3, 108, t, fs=6.5); lines(x0 + 1.3, 110.8, ls, fs=5.9, dy=2.2)
arrow((40, 92), (40, 98))

# ---------------- register column feeding steps 2-4
poly([(RX, 18), (RX, 94.5)])
arrow((RX, 61), (73, 61)); note(82, 58.6, 'burned / unburned', ha='center'); note(82, 63.4, 'labels', ha='center')
arrow((RX, 84), (73, 84)); note(81.5, 82, 'fire years', ha='center')
arrow((RX, 94.5), (RX, 98)); note(RX + 1, 96.2, 'reference')

# ---------------- step 5 (3.5)
box(1, 120, 98, 7.5, fill=FILL_STEP, ec=BLUE, lw=1.1); step(4.5, 123.7, 5)
title(8, 122.2, 'Interannual trends and post-fire trajectories  (Section 3.5)')
lines(8, 125.3, ['Theil–Sen and Mann–Kendall on summer NBR 2017–2025; each fire-year cohort against never-burned forest'], fs=6.0)
arrow((40, 115), (40, 120))
plt.savefig(os.path.join(FIG_DIR, 'Fig_workflow.png'), dpi=600, bbox_inches='tight'); print('ok')
