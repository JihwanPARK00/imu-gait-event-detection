"""fig_heatmap.py — Fig. 2 F1 히트맵 (발표용, 영어 라벨)

RE 라벨 기준 phase1_cells_*.csv 로부터 재생성.
5 sensor locations x 9 channels, 4 algorithm panels.

  python fig_heatmap.py          # HS
  python fig_heatmap.py TO       # TO
"""
import sys, glob
sys.path.insert(0,'.')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

EV = sys.argv[1] if len(sys.argv)>1 else 'HS'
d = pd.concat([pd.read_csv(f) for f in glob.glob('results/phase1_cells_*.csv')],
              ignore_index=True)
# 4명 평균
subj = d.groupby(['sensor','channel','algorithm','subject'])[f'F1_{EV}'].mean().reset_index()
cell = subj.groupby(['sensor','channel','algorithm'])[f'F1_{EV}'].mean().reset_index()

SENS = ['foot','shank','thigh_aff','thigh_sound','lumbar']
SLAB = ['Foot','Shank','Thigh (ipsi)','Thigh (contra)','Lumbar']
CH   = ['Gyro_X','Gyro_Y','Gyro_Z','Acc_X','Acc_Y','Acc_Z',
        'Euler_roll','Euler_pitch','Euler_yaw']
CLAB = ['Gyro X','Gyro Y','Gyro Z','Acc X','Acc Y','Acc Z',
        'Roll','Pitch','Yaw']
ALGO = ['TB_classic','TB_adaptive','BPF','TM']
ALAB = ['TB (fixed threshold)','TB (adaptive threshold)',
        'BPF (band-pass)','TM (template matching)']

cmap = LinearSegmentedColormap.from_list('rg',
        ['#8B0000','#D73027','#FDAE61','#FFFFBF','#A6D96A','#1A9850'])

plt.rcParams.update({'font.size':9,'figure.dpi':300})
fig, axes = plt.subplots(4,1, figsize=(7.2,10.2))
for ax, a, lab in zip(axes, ALGO, ALAB):
    M = np.full((len(SENS),len(CH)), np.nan)
    for i,s in enumerate(SENS):
        for j,c in enumerate(CH):
            v = cell[(cell.sensor==s)&(cell.channel==c)&(cell.algorithm==a)][f'F1_{EV}']
            if len(v): M[i,j]=v.values[0]
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect='auto')
    for i in range(len(SENS)):
        for j in range(len(CH)):
            if np.isnan(M[i,j]): continue
            ax.text(j,i,f'{M[i,j]:.2f}',ha='center',va='center',fontsize=7.5,
                    color='white' if (M[i,j]<0.28 or M[i,j]>0.72) else 'black',
                    fontweight='bold')
    ax.set_yticks(range(len(SENS))); ax.set_yticklabels(SLAB, fontsize=8)
    ax.set_xticks(range(len(CH)))
    ax.set_xticklabels(CLAB if ax is axes[-1] else ['']*len(CH),
                       rotation=45, ha='right', fontsize=8)
    ax.set_title(lab, fontsize=10, pad=5)
    ax.set_xticks(np.arange(-.5,len(CH),1), minor=True)
    ax.set_yticks(np.arange(-.5,len(SENS),1), minor=True)
    ax.grid(which='minor', color='white', lw=1.2); ax.tick_params(which='minor',length=0)
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label(f'{EV} F1-score (4-subject mean)', fontsize=9)
plt.savefig(f'results/fig_heatmap_{EV}.png', bbox_inches='tight')
print(f'saved results/fig_heatmap_{EV}.png')
