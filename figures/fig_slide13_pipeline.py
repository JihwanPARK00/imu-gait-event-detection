"""fig_slide13_pipeline.py — 슬라이드 13: 딥러닝 입력 파이프라인 구조도

  결과: results/slide13_pipeline.png

  9채널 신호 → 200 ms 윈도우 → 1D-CNN / 1D-LSTM → 3분류

왼쪽 9채널은 실제 데이터를 쓴다 (foot, 한 주기).

  python fig_slide13_pipeline.py
"""
import sys
import glob
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

from src.perturbations_mc import IDX, CH_ORDER

TRIAL = sys.argv[1] if len(sys.argv) > 1 else '260519_PJH_3'
plt.rcParams.update({'font.size': 9, 'figure.dpi': 300})

z = np.load(f'results/raw_cache/{TRIAL}_foot.npz')
t, sig9 = z['t'], z['sig9']
gHS = np.sort(z['gt_HS'])
hs = gHS[len(gHS) // 2:][:2]
m = (t >= hs[0]) & (t <= hs[1])
seg = sig9[m]                       # (N, 9)

C_GYRO = '#1565C0'
C_ACC = '#2E7D32'
C_EUL = '#8C8C8C'
C_MODEL = '#E8710A'
C_BOX = '#DDDDDD'
GROUP = [C_GYRO] * 3 + [C_ACC] * 3 + [C_EUL] * 3

fig, ax = plt.subplots(figsize=(12.5, 3.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 34)
ax.axis('off')


def box(x, y, w, h, label, sub=None, fc='white', ec='#555555', lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.4',
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h + 1.6, label, ha='center', va='bottom',
            fontsize=11, fontweight='bold')
    if sub:
        ax.text(x + w / 2, y - 2.4, sub, ha='center', va='top',
                fontsize=9, color='#555555')


def arrow(x0, x1, y=17):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle='-|>',
                                 mutation_scale=18, lw=1.8, color='#444444',
                                 zorder=3))


# ── ① 9채널 입력 ────────────────────────────────
X0, W0 = 2, 22
box(X0, 5, W0, 24, '9-channel signals', 'Gyro 3 · Acc 3 · Euler 3')
for i in range(9):
    s = seg[:, i]
    s = (s - s.mean()) / (s.std() + 1e-9)
    yb = 27 - i * 2.5
    xs = np.linspace(X0 + 1.5, X0 + W0 - 1.5, len(s))
    ax.plot(xs, yb + s * 0.62, color=GROUP[i], lw=0.8, zorder=4)

arrow(25.5, 30.5)

# ── ② 200 ms 윈도우 ─────────────────────────────
X1, W1 = 32, 20
box(X1, 5, W1, 24, '200 ms window', 'shape (9, 18)')
for i in range(9):
    s = seg[:, i]
    s = (s - s.mean()) / (s.std() + 1e-9)
    yb = 27 - i * 2.5
    xs = np.linspace(X1 + 1.5, X1 + W1 - 1.5, len(s))
    ax.plot(xs, yb + s * 0.62, color=GROUP[i], lw=0.8, alpha=.32, zorder=4)
# 잘라내는 창
wx0 = X1 + W1 * 0.40
ww = W1 * 0.20
ax.add_patch(Rectangle((wx0, 5.6), ww, 22.8, fc=C_MODEL, alpha=.20,
                       ec=C_MODEL, lw=1.6, zorder=5))
for i in range(9):
    s = seg[:, i]
    s = (s - s.mean()) / (s.std() + 1e-9)
    yb = 27 - i * 2.5
    xs = np.linspace(X1 + 1.5, X1 + W1 - 1.5, len(s))
    k = (xs >= wx0) & (xs <= wx0 + ww)
    ax.plot(xs[k], yb + s[k] * 0.62, color=GROUP[i], lw=1.3, zorder=6)

arrow(53.5, 58.5)

# ── ③ 모델 ─────────────────────────────────────
X2, W2 = 60, 18
for k in range(3):
    ax.add_patch(FancyBboxPatch((X2 + k * 1.3, 5 + k * 1.3), W2 - 2.6, 21,
                                boxstyle='round,pad=0.4', fc='white',
                                ec=C_MODEL, lw=1.4, alpha=.95, zorder=2 + k))
ax.text(X2 + W2 / 2, 30.6, 'Deep model', ha='center', va='bottom',
        fontsize=11, fontweight='bold')
ax.text(X2 + W2 / 2, 20.5, '1D-CNN', ha='center', va='center',
        fontsize=12, fontweight='bold', color=C_MODEL, zorder=8)
ax.text(X2 + W2 / 2, 16.5, 'or', ha='center', va='center',
        fontsize=9.5, color='#777777', zorder=8)
ax.text(X2 + W2 / 2, 12.5, '1D-LSTM', ha='center', va='center',
        fontsize=12, fontweight='bold', color=C_MODEL, zorder=8)
ax.text(X2 + W2 / 2, 2.6, '~30K / ~20K parameters', ha='center', va='top',
        fontsize=9, color='#555555')

arrow(80, 85)

# ── ④ 3분류 출력 ────────────────────────────────
X3, W3 = 86, 12
box(X3, 5, W3, 24, '3-class output', 'softmax')
bars = [('HS', 0.88, '#C62828'), ('TO', 0.09, '#1565C0'),
        ('None', 0.03, '#999999')]
for j, (nm, v, c) in enumerate(bars):
    yb = 23 - j * 6.2
    ax.add_patch(Rectangle((X3 + 4.2, yb - 1.1), v * 7.0, 2.2,
                           fc=c, ec='none', zorder=5))
    ax.text(X3 + 3.4, yb, nm, ha='right', va='center', fontsize=9.5,
            fontweight='bold', color=c)

fig.text(0.5, 0.005,
         'Trained on clean data only — perturbations applied at test time',
         ha='center', va='bottom', fontsize=10, color='#333333')

plt.tight_layout()
plt.savefig('results/slide13_pipeline.png', bbox_inches='tight')
print('saved results/slide13_pipeline.png')