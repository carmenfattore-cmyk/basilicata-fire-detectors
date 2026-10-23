import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import *
import json, numpy as np, rasterio, fiona, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import shape
from shapely.ops import transform
import pyproj
G=GEE_DIR
with rasterio.open(G+'/Bas_forest_mask.tif') as s:
    f=s.read(1)==1; b=s.bounds
fs=f[::3,::3]
ext=[b.left,b.right,b.bottom,b.top]
fig,ax=plt.subplots(figsize=(8,8.4),dpi=300)
ax.imshow(np.where(fs,1,np.nan),extent=ext,cmap=matplotlib.colors.ListedColormap(['#cfe3c6']),interpolation='nearest')
bd=json.load(open(G+'/Bas_boundary.geojson'))
tr=pyproj.Transformer.from_crs(4326,32633,always_xy=True).transform
for ft in bd['features']:
    g=transform(tr,shape(ft['geometry']))
    for p in getattr(g,'geoms',[g]):
        x,y=p.exterior.xy; ax.plot(x,y,color='#444',lw=0.8)
cols={'train':'#1f6fb4','test':'#d4472a','other':'#9a9a9a'}
with fiona.open(FIRES,layer='incendi_2016_2025') as c:
    for ft in c:
        p=ft['properties']
        if not str(p['categoria']).startswith('Bosc'): continue
        y=p['anno_gee']; k='train' if 2018<=y<=2021 else 'test' if 2022<=y<=2024 else 'other'
        g=shape(ft['geometry'])
        for q in getattr(g,'geoms',[g]):
            x,yy=q.exterior.xy; ax.fill(x,yy,color=cols[k],lw=0.3,ec=cols[k])
ax.set_aspect('equal')
inv=pyproj.Transformer.from_crs(4326,32633,always_xy=True)
xl=(b.left-3000,b.right+3000); yl=(b.bottom-3000,b.top+3000)
for lon in np.arange(15.25,17.0,0.25):
    la=np.linspace(39.7,41.3,60); X,Y=inv.transform(np.full_like(la,lon),la); ax.plot(X,Y,color='#bbb',lw=0.4,zorder=0)
for lat in np.arange(39.75,41.25,0.25):
    lo=np.linspace(15.0,17.2,60); X,Y=inv.transform(lo,np.full_like(lo,lat)); ax.plot(X,Y,color='#bbb',lw=0.4,zorder=0)
ax.set_xlim(*xl); ax.set_ylim(*yl)
xt=[inv.transform(l,yl and 40.0)[0] for l in (15.5,16.0,16.5)]
ax.set_xticks(xt); ax.set_xticklabels(['15.5°E','16.0°E','16.5°E'],fontsize=8)
yt=[inv.transform(15.9,l)[1] for l in (40.0,40.5,41.0)]
ax.set_yticks(yt); ax.set_yticklabels(['40.0°N','40.5°N','41.0°N'],fontsize=8)
for sp in ax.spines.values(): sp.set_color('#888'); sp.set_linewidth(0.6)
x0=b.left+8000; y0=b.bottom+8000
ax.plot([x0,x0+20000],[y0,y0],color='k',lw=2); ax.text(x0+10000,y0+2500,'20 km',ha='center',fontsize=9)
ax.annotate('N',xy=(b.right-12000,b.top-8000),xytext=(b.right-12000,b.top-22000),ha='center',fontsize=11,arrowprops=dict(arrowstyle='-|>',color='k'))
ax.legend(handles=[Patch(color='#cfe3c6',label='Forest mask (WorldCover 2020 ∪ GFC tree cover ≥ 30%)'),Patch(color=cols['train'],label='Wooded fires 2018–2021 (training)'),Patch(color=cols['test'],label='Wooded fires 2022–2024 (test)'),Patch(color=cols['other'],label='Wooded fires 2016–2017, 2025 (not used)')],loc='upper center',bbox_to_anchor=(0.5,-0.04),ncol=2,fontsize=8,frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR,'Fig1_study_area.png'),dpi=300,bbox_inches='tight'); print('ok')
