"""phase1_audit.py — trial 단위 품질 감사 (RE_ GT 기준)

각 trial × 위치에서 최적 sign 기준 F1과 검출-GT 계통오차를 계산해
동기화가 어긋난 trial 을 찾아낸다.
"""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, glob
from src.algorithms import ALGORITHMS
from src.pipeline import match_F1
from src.perturbations_mc import IDX
rows=[]
for pos,ch in [('shank','Gyro_Z'),('foot','Gyro_Y')]:
    ci=IDX[ch]
    for f in sorted(glob.glob(f'results/raw_cache/*_{pos}.npz')):
        z=np.load(f); t=z['t']; sig=z['sig9'][:,ci]; gH=z['gt_HS']; gT=z['gt_TO']
        fs=1/np.median(np.diff(t)); tid=f.split('/')[-1].replace(f'_{pos}.npz','')
        r={'trial':tid,'pos':pos,'nHS':len(gH)}
        for a in ['BPF','TB_adaptive']:
            best=(-1,0,0)
            for sgn in (1,-1):
                HS,TO=ALGORITHMS[a](sig*sgn,fs)
                if len(HS)==0: continue
                det=t[HS]; f1=match_F1(det,gH,0.15); f1t=match_F1(t[TO] if len(TO) else np.array([]),gT,0.15)
                off=np.median([gH[np.argmin(abs(gH-x))]-x for x in det])*1000
                if f1>best[0]: best=(f1,f1t,off)
            r[f'{a}_HS']=best[0]; r[f'{a}_TO']=best[1]; r[f'{a}_off']=best[2]
        rows.append(r)
d=pd.DataFrame(rows); d.to_csv('results/phase1_audit.csv',index=False)
for pos in ['shank','foot']:
    s=d[d.pos==pos]
    print(f'\n━━━ {pos} ━━━  BPF_HS 평균 {s.BPF_HS.mean():.3f}  TB_adapt_HS 평균 {s.TB_adaptive_HS.mean():.3f}')
    bad=s[(s.BPF_HS<0.7)|(abs(s.BPF_off)>60)]
    if len(bad): print('  [주의] 이상 trial:'); print(bad[['trial','BPF_HS','BPF_TO','BPF_off','nHS']].to_string(index=False))
    else: print('  이상 trial 없음')
    print(f'  검출오차 중앙값 {s.BPF_off.median():+.0f}ms  범위 {s.BPF_off.min():+.0f}~{s.BPF_off.max():+.0f}ms')
print('\n━━━ 전체 trial F1 하위 6개 (shank BPF) ━━━')
print(d[d.pos=='shank'].nsmallest(6,'BPF_HS')[['trial','BPF_HS','BPF_TO','BPF_off']].to_string(index=False))