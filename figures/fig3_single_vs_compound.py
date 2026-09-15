"""
Fig. 3 - Single-product vs. measured compound DR
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS = Path('results')

compounds = ['C1\n(P1+P3)', 'C2\n(P1+P2)', 'C3\n(P2+P3)', 'C4\n(P1+P2+P3)']

data = {
    'TB_classic': {
        'product':  [0.98, 1.07, 1.10, 1.07],
        'measured': [0.95, 1.08, 1.10, 1.07],
    },
    'BPF': {
        'product':  [0.96, 0.95, 0.99, 0.95],
        'measured': [0.95, 0.97, 0.99, 0.97],
    },
    'TM': {
        'product':  [0.77, 0.74, 0.97, 0.75],
        'measured': [0.76, 0.74, 0.97, 0.74],
    },
}

colors = {'TB_classic': '#888888', 'BPF': '#1f77b4', 'TM': '#7e2bbd'}

group_spacing = 0.75
x = np.arange(len(compounds)) * group_spacing
width = 0.09
gap = 0.015

# figure 크기 키움
fig, ax = plt.subplots(figsize=(14, 7))

algs = ['TB_classic', 'BPF', 'TM']
for i, alg in enumerate(algs):
    pos_p = x + (i * 2 - 2.5) * (width + gap)
    pos_m = x + (i * 2 - 1.5) * (width + gap)
    ax.bar(pos_p, data[alg]['product'], width,
           facecolor='white', edgecolor=colors[alg], hatch='///',
           linewidth=1.8, label=f'{alg} (product)')
    ax.bar(pos_m, data[alg]['measured'], width,
           facecolor=colors[alg], edgecolor=colors[alg], linewidth=1.8,
           label=f'{alg} (measured)')

ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1.2)

# 폰트 크기 모두 키움
ax.set_xticks(x)
ax.set_xticklabels(compounds, fontsize=15)
ax.tick_params(axis='y', labelsize=13)
ax.set_ylabel('DR (HS)', fontsize=15)
ax.set_ylim(0, 1.35)
ax.grid(True, axis='y', alpha=0.3)

# legend 글자 크기 키움
ax.legend(loc='upper center', ncol=3, fontsize=13, framealpha=0.9)

plt.tight_layout()
out = RESULTS / 'fig3_single_vs_compound_HS.png'
plt.savefig(out, dpi=600, bbox_inches='tight')
plt.close()
print(f'[saved] {out}')
