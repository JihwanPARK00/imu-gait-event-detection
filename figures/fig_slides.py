"""fig_slides.py — 발표용 그림 생성 (영어 라벨)

  slide8_sampling.png   샘플링 처리 3단 패널 (부정확 / 시간축 정정 / +Nyquist)
  slide9_dr_curves.png  교란 5종 DR 곡선

선행 조건: phase2_corrected.py 실행 후 results/p2c_*.csv 존재
실행:      python fig_slides.py
"""
import sys, glob
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'figure.dpi': 300,
                     'axes.linewidth': 0.9})

LV = {'P1': [1, 5, 20, 50, 100], 'P2': [5, 10, 15, 20, 30],
      'P4': [.01, .05, .10, .20, .30], 'P6': [1, 5, 10, 25, 50],
      'P7': [60, 45, 30, 20, 10]}
TITLE = {'P1': 'P1  Sensor noise', 'P2': 'P2  Misalignment',
         'P4': 'P3  Packet loss', 'P6': 'P4  Constant bias',
         'P7': 'P5  Sampling rate'}
XLAB = {'P1': r'$\sigma$  (deg/s)', 'P2': r'$\theta$  (deg)',
        'P4': 'loss ratio', 'P6': 'offset  (deg/s)',
        'P7': r'$f_s$  (Hz)'}
ALGO = ['TB_classic', 'BPF', 'TM']
ALAB = {'TB_classic': 'TB (fixed threshold)', 'BPF': 'BPF (band-pass)',
        'TM': 'TM (template matching)'}
C = {'TB_classic': '#5A5A5A', 'BPF': '#1565C0', 'TM': '#B8860B'}
MK = {'TB_classic': 'o', 'BPF': 's', 'TM': '^'}

d = pd.concat([pd.read_csv(f) for f in glob.glob('results/p2c_*.csv')],
              ignore_index=True)
d['subject'] = d.trial_id.str.split('_').str[1].str.upper()
base = (d[d.pert_type == 'baseline']
        .groupby(['subject', 'sensor', 'channel', 'algorithm'])
        .F1_HS.mean().rename('bH').reset_index())
# 이벤트별 셀 필터: HS baseline 이 0.5 이하인 셀은 HS DR 계산에서 제외
# (셀 선정 기준이 max(HS, TO) > 0.5 이므로 TO 만으로 통과한 셀이 섞임)
cellbase = (base.groupby(['sensor', 'channel', 'algorithm'])
            .bH.mean().rename('cH').reset_index())
VALID = cellbase[cellbase.cH > 0.5]
print(f'HS valid cells: {len(VALID)} / {len(cellbase)}')


def dr(sub):
    m = sub.merge(base, on=['subject', 'sensor', 'channel', 'algorithm'])
    m = m.merge(cellbase, on=['sensor', 'channel', 'algorithm'])
    m['DR'] = np.where((m.bH > 0.1) & (m.cH > 0.5),
                       np.minimum(m.F1_HS / m.bH, 1.0), np.nan)
    return m


# ══════════ Slide 8 — sampling handling ══════════
m7 = dr(d[d.pert_type == 'P7'])
MODES = [('orig', 'Index-based time axis'),
         ('native', '+ Time axis corrected'),
         ('native_nyq', '+ Nyquist-aware cutoff')]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
for ax, (mode, title) in zip(axes, MODES):
    for a in ALGO:
        s = m7[(m7.p7mode == mode) & (m7.algorithm == a)]
        y = [s[s.pert_level == l].DR.mean() for l in LV['P7']]
        ax.plot(range(5), y, marker=MK[a], color=C[a], ms=5, lw=1.8,
                label=ALAB[a] if ax is axes[0] else None)
    ax.set_xticks(range(5))
    ax.set_xticklabels(LV['P7'])
    ax.set_ylim(-0.04, 1.05)
    ax.set_xlabel(XLAB['P7'])
    ax.set_title(title, fontsize=10, pad=6)
    ax.grid(alpha=.28, lw=.6)
axes[0].set_ylabel('Degradation Ratio (HS)\nhigher is better')
axes[0].legend(fontsize=8, frameon=False, loc='lower left')
axes[0].annotate('identical at 60 and 30 Hz', xy=(2, 0.36), xytext=(1.1, 0.62),
                 fontsize=8, color='#C62828',
                 arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.1))
axes[1].annotate('fixed 12 Hz cutoff\nexceeds Nyquist', xy=(3.6, 0.03),
                 xytext=(1.5, 0.20), fontsize=8, color='#C62828',
                 arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.1))
plt.tight_layout()
plt.savefig('results/slide8_sampling.png', bbox_inches='tight')
plt.close()
print('saved results/slide8_sampling.png')

# ══════════ Slide 9 — DR curves ══════════
fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.0), sharey=True)
for ax, pt in zip(axes, ['P1', 'P2', 'P4', 'P6', 'P7']):
    sub = d[d.pert_type == pt]
    if pt == 'P7':
        sub = sub[sub.p7mode == 'native_nyq']
    m = dr(sub)
    for a in ALGO:
        s = m[m.algorithm == a]
        y = [s[np.isclose(s.pert_level, l)].DR.mean() for l in LV[pt]]
        ax.plot(range(5), y, marker=MK[a], color=C[a], ms=5, lw=1.8,
                label=ALAB[a] if ax is axes[0] else None)
    ax.set_xticks(range(5))
    ax.set_xticklabels(LV[pt], fontsize=8)
    ax.set_ylim(0.35, 1.03)
    ax.set_xlabel(XLAB[pt], fontsize=9)
    ax.set_title(TITLE[pt], fontsize=10, pad=6)
    ax.grid(alpha=.28, lw=.6)
axes[0].set_ylabel('Degradation Ratio (HS)\nhigher is better')
axes[0].legend(fontsize=8, frameon=False, loc='lower left')
plt.tight_layout()
plt.savefig('results/slide9_dr_curves.png', bbox_inches='tight')
plt.close()
print('saved results/slide9_dr_curves.png')