"""fig_slide10_perturb.py — 슬라이드 10: 5종 교란을 실제 신호에 적용한 예시

  결과: results/slide10_perturbations.png

foot Gyro_Y 한 주기에 각 교란의 '가장 강한 레벨'을 적용해
원본(회색)과 교란 후(주황)를 겹쳐 그린다. 5패널 가로 배치.

숫자·수식은 넣지 않는다. 형태 변화만 보이면 충분하다.

  python fig_slide10_perturb.py
  python fig_slide10_perturb.py 260522_KKH_4
"""
import sys
import glob
sys.path.insert(0, '.')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.perturbations_mc import apply_perturbation_mc, IDX

TRIAL = sys.argv[1] if len(sys.argv) > 1 else '260519_PJH_3'
N_CYCLE = 2          # 2주기 — 교란 효과가 더 잘 보인다

plt.rcParams.update({'font.size': 9, 'figure.dpi': 300, 'axes.linewidth': 0.9})

z = np.load(f'results/raw_cache/{TRIAL}_foot.npz')
t, sig9 = z['t'], z['sig9']
gHS = np.sort(z['gt_HS'])
ci = IDX['Gyro_Y']

# 각 교란의 최강 레벨 (슬라이드 표와 동일)
# 시연용 과장 레벨. 실험에 쓴 최강 레벨보다 크게 잡아 형태 차이를 보이게 한다.
# (슬라이드 표에는 실제 실험 레벨을 적고, 그림 캡션에 exaggerated 를 명시)
PANELS = [
    ('P1', 160,  'P1  Sensor noise'),
    ('P2', 70,   'P2  Misalignment'),
    ('P4', 0.60, 'P3  Packet loss'),
    ('P6', 200,  'P4  Constant bias'),
    ('P7', 6,    'P5  Sampling rate'),
]

C_ORIG = '#8C8C8C'
C_PERT = '#E8710A'

fig, axes = plt.subplots(1, 5, figsize=(13.5, 2.6), sharey=True)

# 표시 구간: HS 기준 2주기
hs = gHS[len(gHS) // 2:][:N_CYCLE + 1]
T0, T1 = hs[0], hs[-1]
m = (t >= T0) & (t <= T1)
tt = t[m] - T0
orig = sig9[m, ci]

for ax, (pt, lv, title) in zip(axes, PANELS):
    pert = apply_perturbation_mc(pt, sig9, t, lv,
                                 ref_channel='Gyro_Y', scope='all9')[m, ci]
    ax.plot(tt, pert, color=C_PERT, lw=1.7, zorder=2, alpha=.9)
    ax.plot(tt, orig, color='#4A4A4A', lw=1.6, ls=(0, (4, 2)),
            zorder=4)

    # 패킷 손실은 보간으로 복원돼 형태 차이가 거의 없다.
    # 어느 샘플이 소실됐는지 점으로 표시해야 무엇이 일어났는지 보인다.
    if pt == 'P4':
        rng_ = np.random.default_rng(3)
        idx = np.sort(rng_.choice(len(tt), int(len(tt) * lv), replace=False))
        ax.plot(tt[idx], pert[idx], 'o', ms=2.2, color=C_PERT,
                mec='none', alpha=.55, zorder=4)
        ax.text(0.5, 0.02, 'lost samples interpolated',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=8, color=C_PERT)
    ax.set_title(title, fontsize=10.5, pad=7)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

axes[0].plot([], [], color='#4A4A4A', lw=1.8, ls=(0, (4, 2)), label='original')
axes[0].plot([], [], color=C_PERT, lw=2.0, label='perturbed')
h, lb = axes[0].get_legend_handles_labels()
fig.legend(h, lb, loc='upper center', ncol=2, fontsize=10, frameon=False,
           bbox_to_anchor=(0.5, 1.16))

lo, hi = orig.min(), orig.max()
rng = hi - lo
axes[0].set_ylim(lo - rng * 0.55, hi + rng * 1.05)

fig.text(0.5, -0.03, 'perturbation levels exaggerated for visibility',
         ha='center', va='top', fontsize=8.5, color='#777777',
         style='italic')
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('results/slide10_perturbations.png', bbox_inches='tight')
print('saved results/slide10_perturbations.png')