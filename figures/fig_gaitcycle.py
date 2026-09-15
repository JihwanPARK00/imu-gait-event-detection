"""fig_gaitcycle.py — 보행 주기 + 정답 라벨 그림 (슬라이드 2/4용)

  결과: results/fig_gaitcycle.png

foot Gyro_Y 신호 2주기에 수동 라벨링한 HS / TO 를 표시한다.
발표에서 "이게 우리가 검출하려는 것"을 한 장으로 보여주는 용도.

  python fig_gaitcycle.py                       # 기본 (PJH 3km)
  python fig_gaitcycle.py 260522_KKH_4          # trial 지정
  python fig_gaitcycle.py 260522_KKH_4 45       # 시작 시각(초) 지정
"""
import sys
import glob
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.perturbations_mc import IDX

TRIAL = sys.argv[1] if len(sys.argv) > 1 else '260519_PJH_3'
T0 = float(sys.argv[2]) if len(sys.argv) > 2 else None
N_CYCLE = 1

plt.rcParams.update({'font.size': 10, 'figure.dpi': 300, 'axes.linewidth': 0.9})

path = f'results/raw_cache/{TRIAL}_foot.npz'
if not glob.glob(path):
    cands = sorted(glob.glob('results/raw_cache/*_foot.npz'))
    raise SystemExit(f'{path} 없음. 사용 가능:\n  ' +
                     '\n  '.join(c.split("/")[-1] for c in cands[:8]))

z = np.load(path)
t, sig = z['t'], z['sig9'][:, IDX['Gyro_Y']]
gHS, gTO = np.sort(z['gt_HS']), np.sort(z['gt_TO'])

# 시작 시각: 지정 없으면 라벨 구간 중간쯤의 HS 에서 시작
if T0 is None:
    T0 = gHS[len(gHS) // 2]
hs_in = gHS[gHS >= T0][:N_CYCLE + 1]
if len(hs_in) < N_CYCLE + 1:
    hs_in = gHS[-(N_CYCLE + 1):]
    T0 = hs_in[0]
T1 = hs_in[-1]
pad = (T1 - T0) * 0.02   # 좌우 여분 최소화 (HS 로 시작, HS 로 끝)
m = (t >= T0 - pad) & (t <= T1 + pad)

# 신호 부호: HS 근처가 음의 피크가 되도록 (발등 Gyro_Y 관례)
seg_t, seg_s = t[m], sig[m]

HS = gHS[(gHS >= T0 - pad) & (gHS <= T1 + pad)]
TO = gTO[(gTO >= T0 - pad) & (gTO <= T1 + pad)]

C_HS, C_TO = '#C62828', '#1565C0'
fig, ax = plt.subplots(figsize=(8.6, 4.0))

ax.plot(seg_t - T0, seg_s, color='#333333', lw=1.6, zorder=3,
        label='Foot Gyro$_Y$ (sagittal angular velocity)')

ylo, yhi = seg_s.min(), seg_s.max()
rng = yhi - ylo
ax.set_ylim(ylo - rng * 0.34, yhi + rng * 0.26)


def mark(times, color, name):
    for k, tt in enumerate(times):
        ax.axvline(tt - T0, ymin=0.03, ymax=0.88, color=color,
                   lw=1.3, ls='--', alpha=.75, zorder=2)
        v = np.interp(tt, seg_t, seg_s)
        ax.plot(tt - T0, v, 'o', ms=10, mfc=color, mec='white', mew=1.6,
                zorder=5, label=name if k == 0 else None)
        ax.text(tt - T0, yhi + rng * 0.13, name, ha='center', va='bottom',
                fontsize=13, fontweight='bold', color=color)


mark(HS, C_HS, 'HS')
mark(TO, C_TO, 'TO')

# 보행 주기 구간 표시 (HS -> 다음 HS)
for a, b in zip(hs_in[:-1], hs_in[1:]):
    ya = ylo - rng * 0.15
    ax.annotate('', xy=(b - T0, ya), xytext=(a - T0, ya),
                arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.1))
    ax.text((a + b) / 2 - T0, ya - rng * 0.055, 'one gait cycle',
            ha='center', va='top', fontsize=10.5, color='#555555')

ax.set_xlabel('Time (s)')
ax.set_ylabel('Angular velocity (deg/s)')
ax.set_xlim((T0 - pad) - T0, (T1 + pad) - T0)
ax.grid(alpha=.22, lw=.6)
# 범례는 그래프 밖 하단으로 (HS/TO 라벨을 가리지 않도록)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=3,
          fontsize=9, frameon=False)
ax.set_title('Ground-truth gait events from sagittal video labeling',
             fontsize=11, pad=8)
plt.tight_layout()
plt.savefig('results/fig_gaitcycle.png', bbox_inches='tight')
print(f'saved results/fig_gaitcycle.png  ({TRIAL}, {T0:.2f}–{T1:.2f} s, '
      f'HS {len(HS)} / TO {len(TO)})')