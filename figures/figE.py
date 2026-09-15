"""figE.py — P7 4단 패널 (27개 셀 평균)

  (a) Published implementation      기존 구현
  (b) Time-axis corrected           시간축 정정 (네이티브 저레이트)
  (c) + Nyquist-aware cutoff        + min(12, 0.4*fs) 컷오프
  (d) + resampled to 91 Hz          + 원 격자 재보간

figD 의 27셀 확장판이며 논문 그림으로는 이쪽을 쓰는 것을 권한다.

선행 조건: phase2_corrected.py 4회 + table3_final.py 실행
           → results/p7_levels_27cells.csv 생성
실행:      python figE.py
"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'figure.dpi':300})
m=pd.read_csv('results/p7_levels_27cells.csv')
lv=[60,45,30,20,10]; C={'TB_classic':'#888888','BPF':'#1565C0','TM':'#6A1B9A'}
T={'orig':'(a) Published implementation','native':'(b) Time-axis corrected',
   'native_nyq':'(c) + Nyquist-aware cutoff','resample':'(d) + resampled to 91 Hz'}
fig,ax=plt.subplots(1,4,figsize=(12,2.9),sharey=True)
for i,md in enumerate(['orig','native','native_nyq','resample']):
    for a in ['TB_classic','BPF','TM']:
        s=m[(m.p7mode==md)&(m.algorithm==a)]
        ax[i].plot(range(5),[s[s.pert_level==l].DR_HS.mean() for l in lv],'-o',
                   color=C[a],ms=4,label=a if i==0 else None)
    ax[i].set_xticks(range(5)); ax[i].set_xticklabels(lv); ax[i].set_ylim(-0.03,1.05)
    ax[i].set_title(T[md],fontsize=8.5); ax[i].grid(alpha=.25,lw=.5); ax[i].set_xlabel('$f_s$ (Hz)')
ax[0].set_ylabel('DR (HS), mean of 27 cells'); ax[0].legend(fontsize=7,frameon=False)
plt.tight_layout(); plt.savefig('results/figE_p7_27cells.png',bbox_inches='tight')
print('ok')