"""Trend graph: (a) yearly mean summer NBR of never-burned forest with Theil-Sen line; (b) share of significant greening/browning by fire-year cohort."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import json, numpy as np, itertools, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5})
R = json.load(open(os.path.join(TREND_DIR, 'trends.json')))
yrs = np.arange(2017, 2026); m = np.array([R['unburned_mean_nbr'][str(y)] for y in yrs])
pairs = list(itertools.combinations(range(9), 2))
sl = np.median([(m[j] - m[i]) / (j - i) for i, j in pairs]); S = int(sum(np.sign(m[j] - m[i]) for i, j in pairs))
b0 = np.median(m - sl * (yrs - yrs[0]))
print(f'regional TS slope {sl*10:+.4f} per decade, S={S}')
json.dump(dict(slope_dec=float(sl * 10), S=S), open(os.path.join(TREND_DIR, 'trend_series.json'), 'w'))
fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0), dpi=600, gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.42))
a.plot(yrs, m, 'o-', color='#1b5e20', lw=1.4, ms=4, label='never-burned forest, mean')
a.plot(yrs, b0 + sl * (yrs - yrs[0]), '--', color='#555', lw=1, label=f'Theil–Sen: {sl*10:+.3f} per decade (S = {S})')
a.set_xticks(yrs); a.set_xticklabels([f"'{str(y)[2:]}" for y in yrs]); a.set_ylabel('Summer NBR'); a.set_xlabel('Year')
a.set_ylim(0.44, 0.545); a.legend(frameon=False, fontsize=7, loc='lower center'); a.set_title('(a) Regional summer NBR, 2017–2025', fontsize=8.5, loc='left')
labs = ['Never burned'] + [str(y) for y in range(2017, 2025)]
pos = [R['unburned']['sig_pos']] + [R['by_fire_year'][str(y)]['sig_pos'] for y in range(2017, 2025)]
neg = [R['unburned']['sig_neg']] + [R['by_fire_year'][str(y)]['sig_neg'] for y in range(2017, 2025)]
yy = np.arange(len(labs))[::-1]
b.barh(yy, pos, color='#4c9a2a', height=0.62, label='significant greening')
b.barh(yy, [-v for v in neg], color='#c0392b', height=0.62, label='significant browning')
for k in range(len(labs)):
    b.text(pos[k] + 0.6, yy[k], f'{pos[k]:.1f}', va='center', fontsize=6.5); b.text(-neg[k] - 0.6, yy[k], f'{neg[k]:.1f}', va='center', ha='right', fontsize=6.5)
b.set_yticks(yy); b.set_yticklabels(labs); b.axvline(0, color='#444', lw=0.6)
b.set_xlim(-12, 31); b.set_xticks([-10, 0, 10, 20, 30]); b.set_xticklabels(['10', '0', '10', '20', '30'])
b.set_xlabel('Share of pixels (%)')
b.legend(frameon=False, fontsize=7, loc='lower right'); b.set_title('(b) Significant trends, by fire year of the stand', fontsize=8.5, loc='left')
for ax in (a, b):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
plt.savefig(os.path.join(FIG_DIR, 'FigT_series.png'), dpi=600, bbox_inches='tight')
