"""Figures drawn from the result tables: fires detected vs area flagged, and detection by month (values from results/comparison_cv and results/statistics)."""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
GREY='#9a9a9a'; BLUE='#1f6fb4'
# ---------- Fig 3: trade-off
rows=[('dNBR','annual',14011,97),('PCA','annual',19418,110),('Autoencoder','annual',18743,47),('Random Forest','annual',27866,129),('U-Net','annual',4298,48),
      ('dNBR','seasonal',915,32),('PCA','seasonal',30653,56),('Autoencoder','seasonal',5648,49),('Random Forest','seasonal',4241,80),('U-Net','seasonal',1309,47)]
lab={('dNBR','annual'):('l',0),('PCA','annual'):('l',0),('Autoencoder','annual'):('r',-7),('Random Forest','annual'):('l',0),('U-Net','annual'):('l',5),
     ('dNBR','seasonal'):('l',0),('PCA','seasonal'):('r',0),('Autoencoder','seasonal'):('r',5),('Random Forest','seasonal'):('r',0),('U-Net','seasonal'):('l',0)}
fig,ax=plt.subplots(figsize=(7.2,4.4),dpi=600)
ax.axvline(1812,color='#555',ls='--',lw=0.8); ax.text(1812*1.04,136,'Burned forest in the register: 1,812 ha',fontsize=8,color='#444',va='top')
for m,i,ha,f in rows:
    hit=(m,i)==('U-Net','seasonal'); c=BLUE if hit else GREY
    ax.scatter(ha,f,s=46 if hit else 38,facecolor=c if i=='seasonal' else 'white',edgecolor=c,linewidth=1.4,zorder=3)
    side,dy=lab[(m,i)]
    ax.annotate(f'{m} ({i})',(ha,f),xytext=((-7 if side=='l' else 7),dy),textcoords='offset points',ha='right' if side=='l' else 'left',va='center',fontsize=8,
                color='black' if hit else '#444',fontweight='bold' if hit else 'normal')
ax.set_xscale('log'); ax.set_xlim(500,60000); ax.set_ylim(0,140)
ax.set_xticks([1000,3000,10000,30000]); ax.set_xticklabels(['1,000','3,000','10,000','30,000'])
ax.set_xlabel('Area flagged, 2022–2024 (ha, log scale)'); ax.set_ylabel('Wooded register fires detected (of 426)')
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.grid(axis='y',color='#e5e5e5',lw=0.6); ax.set_axisbelow(True)
ax.scatter([],[],facecolor='white',edgecolor=GREY,label='annual input'); ax.scatter([],[],facecolor=GREY,edgecolor=GREY,label='seasonal input')
ax.legend(loc='lower right',fontsize=8,frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, 'Fig_tradeoff.png'),dpi=600); plt.close()
# ---------- Fig 4: by month
months=['June','July','August','September']; n=[44,120,116,60]
ann=[29,54,22,13]; sea=[2,15,35,28]; both=[29,62,50,30]
fig,ax=plt.subplots(figsize=(7.2,3.8),dpi=600); x=np.arange(4); w=0.26
for k,(v,c,a,l) in enumerate([(ann,'#cfcfcf',1,'Annual input (Random Forest)'),(sea,'#8f8f8f',1,'Seasonal input (Random Forest)'),(both,BLUE,1,'Both combined')]):
    r=np.array(v)/np.array(n); b=ax.bar(x+(k-1)*w,r*100,w*0.92,color=c,label=l)
    for xi,ri in zip(x+(k-1)*w,r): ax.text(xi,ri*100+1.2,f'{int(ri*100+0.5)}%',ha='center',fontsize=7.5,fontweight='bold' if k==2 else 'normal')
ax.set_xticks(x); ax.set_xticklabels([f'{m}\n({k} fires)' for m,k in zip(months,n)])
ax.set_ylabel('Fires detected (%)'); ax.set_ylim(0,78)
for s in ('top','right'): ax.spines[s].set_visible(False)
ax.legend(fontsize=8,frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.1))
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, 'Fig_months.png'),dpi=600); plt.close()
print('ok')
