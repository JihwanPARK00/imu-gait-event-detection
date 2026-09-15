"""fig_slide11.py — 슬라이드 11: 정규화 모드가 겉보기 강건성에 미치는 영향

  결과: results/slide11_normalization.png

  per-signal : 교란된 신호 자체 통계로 z-score (전처리가 교란을 흡수)
  frozen     : clean 신호 통계를 고정 사용 (실시간 배포에 가까움)

축 처리
  값이 0.79~1.00 구간에 몰려 있어 0부터 그리면 차이가 안 보인다.
  그렇다고 0.70부터 시작하면 막대 길이가 왜곡된다.
  → 위/아래 두 축으로 실제 분할(broken axis)하고 경계에 사선을 넣었다.
    아래 축은 0을, 위 축은 0.70~1.06을 담당한다.

읽는 법
  같은 색 막대 두 개가 한 쌍(per-signal / 빗금 frozen).
  빨간 화살표는 per-signal 높이에서 frozen 높이까지 내려오며,
  숫자는 화살표 시작점(per-signal 높이) 옆에 붙는다.
  낙폭이 0.02 미만이면 표시하지 않는다.

선행 조건: run_phase4.py 실행 후 results/dl_p4_*.csv 존재
실행:      python fig_slide11.py
"""
import sys
import glob
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({'font.size': 9, 'figure.dpi': 300, 'axes.linewidth': 0.9})

LV = {'P1': [1, 5, 20, 50, 100], 'P2': [5, 10, 15, 20, 30],
      'P4': [.01, .05, .10, .20, .30], 'P6': [1, 5, 10, 25, 50],
      'P7': [60, 45, 30, 20, 10]}
PERTS = ['P1', 'P2', 'P4', 'P6', 'P7']
PLAB = ['P1\nNoise', 'P2\nMisalign', 'P3\nPacket loss', 'P4\nBias', 'P5\nSampling']
COL = {'CNN': '#D81B60', 'LSTM': '#F57C00'}
HI_LO, HI_HI = 0.70, 1.055     # 위쪽 축 범위 (라벨 여유 포함)
LBL_Y, LBL_DY, LBL_DX = 1.008, 0.030, 0.17   # 낙폭 라벨 배치
LO_LO, LO_HI = 0.0, 0.035      # 아래쪽 축 범위 (0 표시용)

files = [f for f in glob.glob('results/dl_p4_*.csv')
         if not f.endswith(('dr.csv', 'curve.csv', 'auc.csv'))]
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
base = (d[d.pert_type == 'baseline']
        .groupby(['model', 'position', 'trial_id', 'norm'])
        .F1_HS.mean().rename('b').reset_index())
m = d[d.pert_type != 'baseline'].merge(
    base, on=['model', 'position', 'trial_id', 'norm'])
m['DR'] = np.minimum(m.F1_HS / (m.b + 1e-9), 1.0)
m = m[m.scope == 'all9']


def auc(sub, pt):
    y = [sub[np.isclose(sub.pert_level, l)].DR.mean() for l in LV[pt]]
    return float(np.trapezoid(np.nan_to_num(y), np.linspace(0, 1, 5)))


ORDER = [('CNN', 'adaptive'), ('CNN', 'frozen'),
         ('LSTM', 'adaptive'), ('LSTM', 'frozen')]
w = 0.2
x = np.arange(len(PERTS))

fig = plt.figure(figsize=(9.8, 4.0))
gs = GridSpec(2, 2, height_ratios=[9, 1], hspace=0.06, wspace=0.10,
              figure=fig)
axes_hi, axes_lo = [], []
for c in range(2):
    ax_hi = fig.add_subplot(gs[0, c])
    ax_lo = fig.add_subplot(gs[1, c], sharex=ax_hi)
    axes_hi.append(ax_hi)
    axes_lo.append(ax_lo)

for ci, pos in enumerate(['shank', 'foot']):
    ax, axb = axes_hi[ci], axes_lo[ci]
    yv = {}
    for k, (mdl, nm) in enumerate(ORDER):
        sub = m[(m.model == mdl) & (m.position == pos) & (m.norm == nm)]
        if len(sub) == 0:
            continue
        y = [auc(sub[sub.pert_type == p], p) for p in PERTS]
        xs = x + (k - 1.5) * w
        yv[(mdl, nm)] = (y, xs)
        style = dict(color=COL[mdl], alpha=1.0 if nm == 'adaptive' else 0.42,
                     edgecolor='black', lw=.5,
                     hatch='' if nm == 'adaptive' else '///')
        ax.bar(xs, y, w,
               label=f'{mdl} — {"per-signal" if nm == "adaptive" else "frozen"}'
                     if pos == 'shank' else None, **style)
        axb.bar(xs, y, w, **style)      # 아래 축에도 같은 막대 (0에서 이어짐)

    # 낙폭: 숫자를 1.00 기준선 위에 두고, 거기서 frozen 막대까지 화살표
    drops = []
    for mdl in ['CNN', 'LSTM']:
        if (mdl, 'adaptive') not in yv or (mdl, 'frozen') not in yv:
            continue
        ya, _ = yv[(mdl, 'adaptive')]
        yf, xf = yv[(mdl, 'frozen')]
        for y0, y1, xi in zip(ya, yf, xf):
            if y1 - y0 < -0.02:
                drops.append((xi, y1, y1 - y0, mdl))
    # x 순으로 정렬 후, 가까운 라벨은 한 단씩 위로 올려 겹침 방지
    drops.sort(key=lambda t: t[0])
    placed = []
    for xi, y1, dv, mdl in drops:
        ly = LBL_Y
        while any(abs(xi - px) < LBL_DX and abs(ly - py) < LBL_DY
                  for px, py in placed):
            ly += LBL_DY
        placed.append((xi, ly))
        c = COL[mdl]
        ax.annotate('', xy=(xi, y1), xytext=(xi, ly - 0.010),
                    arrowprops=dict(arrowstyle='-|>', color=c,
                                    lw=1.2, shrinkA=0, shrinkB=0))
        ax.text(xi, ly, f'{dv:+.2f}', ha='center', va='bottom',
                fontsize=7.4, fontweight='bold', color=c)

    ax.set_ylim(HI_LO, HI_HI)
    ax.set_yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.axhline(1.0, color='gray', lw=.7, ls=':')
    ax.set_title(pos.capitalize(), fontsize=10.5)
    ax.grid(axis='y', alpha=.25, lw=.6)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(bottom=False, labelbottom=False)

    axb.set_ylim(LO_LO, LO_HI)
    axb.set_yticks([0.0])
    axb.spines['top'].set_visible(False)
    axb.set_xticks(x)
    axb.set_xticklabels(PLAB, fontsize=8.5)

    if ci == 1:
        ax.tick_params(labelleft=False)
        axb.tick_params(labelleft=False)

    # 축 분할 사선 (두 축 경계에 각각)
    dk = dict(marker=[(-1, -.6), (1, .6)], markersize=7, linestyle='none',
              color='k', mec='k', mew=1.1, clip_on=False)
    ax.plot([0, 1], [0, 0], transform=ax.transAxes, **dk)
    axb.plot([0, 1], [1, 1], transform=axb.transAxes, **dk)

axes_hi[0].set_ylabel('AUC of DR (HS)\nhigher is better')
axes_hi[0].yaxis.set_label_coords(-0.115, 0.40)
h, lb = axes_hi[0].get_legend_handles_labels()
fig.legend(h, lb, loc='upper center', ncol=4, fontsize=8.5, frameon=False,
           bbox_to_anchor=(0.5, 1.03))
plt.subplots_adjust(top=0.86, bottom=0.15, left=0.115, right=0.98)
plt.savefig('results/slide11_normalization.png', bbox_inches='tight')
print('saved results/slide11_normalization.png')

# ── 표 출력 ──
rows = []
for pos in ['shank', 'foot']:
    for mdl in ['CNN', 'LSTM']:
        for nm in ['adaptive', 'frozen']:
            sub = m[(m.model == mdl) & (m.position == pos) & (m.norm == nm)]
            if len(sub) == 0:
                continue
            r = {'position': pos, 'model': mdl,
                 'norm': 'per-signal' if nm == 'adaptive' else 'frozen'}
            for p, lab in zip(PERTS, ['P1', 'P2', 'P3', 'P4', 'P5']):
                r[lab] = round(auc(sub[sub.pert_type == p], p), 3)
            rows.append(r)
print('\nAUC of DR (HS), scope = all 9 channels')
print(pd.DataFrame(rows).to_string(index=False))