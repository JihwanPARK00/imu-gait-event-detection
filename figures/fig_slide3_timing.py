"""fig_slide3_timing.py — 슬라이드 3: 검출 타이밍이 보조 시점을 결정한다

  결과: results/slide3_timing.png

위: 검출이 정확할 때  → 보조 토크가 push-off 구간에 정확히 들어감
아래: 검출이 늦을 때   → 같은 토크가 swing 구간에 들어가 도움이 안 됨

실제 foot Gyro_Y 신호 한 주기를 사용한다. 숫자·수식 없이 형태로만 전달.

  python fig_slide3_timing.py
  python fig_slide3_timing.py 260522_KKH_4     # trial 지정
"""
import sys
import glob
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from src.perturbations_mc import IDX

TRIAL = sys.argv[1] if len(sys.argv) > 1 else '260519_PJH_3'
DELAY = 0.22          # 어긋난 경우의 지연 (초)

plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'axes.linewidth': 0.9})

path = f'results/raw_cache/{TRIAL}_foot.npz'
if not glob.glob(path):
    raise SystemExit(f'{path} 없음')

z = np.load(path)
t, sig = z['t'], z['sig9'][:, IDX['Gyro_Y']]
gHS, gTO = np.sort(z['gt_HS']), np.sort(z['gt_TO'])

# HS -> HS 한 주기
hs = gHS[len(gHS) // 2:][:2]
T0, T1 = hs[0], hs[1]
m = (t >= T0) & (t <= T1)
tt, ss = t[m] - T0, sig[m]
to = gTO[(gTO > T0) & (gTO < T1)]
TO = (to[0] - T0) if len(to) else (T1 - T0) * 0.68

C_SIG = '#333333'
C_GOOD = '#2E7D32'
C_BAD = '#C62828'
C_HS = '#C62828'
C_TO = '#1565C0'

fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.0), sharex=True)

for ax, correct in zip(axes, [True, False]):
    ax.plot(tt, ss, color=C_SIG, lw=1.7, zorder=3)
    ylo, yhi = ss.min(), ss.max()
    rng = yhi - ylo
    ax.set_ylim(ylo - rng * 0.52, yhi + rng * 0.34)

    # 이벤트 수직선
    for x, c in [(0.0, C_HS), (TO, C_TO), (tt[-1], C_HS)]:
        ax.axvline(x, color=c, lw=1.1, ls='--', alpha=.6, zorder=2)

    # 보조 토크 구간: push-off (TO 직전)에 들어가야 정상
    a0 = TO - 0.22
    a1 = TO + 0.04
    shift = 0.0 if correct else DELAY
    col = C_GOOD if correct else C_BAD
    ax.axvspan(a0 + shift, a1 + shift, color=col, alpha=.20, zorder=1)

    yb = ylo - rng * 0.24
    ax.annotate('', xy=(a1 + shift, yb), xytext=(a0 + shift, yb),
                arrowprops=dict(arrowstyle='-', color=col, lw=5,
                                alpha=.85, shrinkA=0, shrinkB=0))
    ax.text((a0 + a1) / 2 + shift, yb - rng * 0.06, 'assistance',
            ha='center', va='top', fontsize=9.5, color=col,
            fontweight='bold')

    ax.set_ylabel('Foot angular\nvelocity', fontsize=9.5)
    ax.grid(alpha=.20, lw=.5)
    ax.set_yticks([])

    ttl = ('Detection on time  —  assistance lands at push-off'
           if correct else
           'Detection late  —  assistance lands during swing')
    ax.set_title(ttl, fontsize=10.5, color=col, fontweight='bold', pad=22)

# 이벤트 이름은 위쪽 축에만
y0, y1 = axes[0].get_ylim()
yh = y1 - (y1 - y0) * 0.03
for x, name, c in [(0.0, 'HS', C_HS), (TO, 'TO', C_TO), (tt[-1], 'HS', C_HS)]:
    axes[0].text(x, yh, name, ha='center', va='top', fontsize=11,
                 fontweight='bold', color=c)

# 지연을 나타내는 화살표 (아래 축)
axes[1].annotate('', xy=(a0 + DELAY, axes[1].get_ylim()[1] * 0.72),
                 xytext=(a0, axes[1].get_ylim()[1] * 0.72),
                 arrowprops=dict(arrowstyle='-|>', color=C_BAD, lw=1.6))

axes[1].set_xlabel('one gait cycle', fontsize=10)
axes[1].set_xticks([])
plt.tight_layout()
plt.savefig('results/slide3_timing.png', bbox_inches='tight')
print('saved results/slide3_timing.png')