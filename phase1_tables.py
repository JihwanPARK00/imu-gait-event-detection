"""phase1_tables.py — 논문 Table 1 / Table 2 재현 + surviving cell 선정"""
import sys; sys.path.insert(0,'.')
import glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon
d=pd.concat([pd.read_csv(f) for f in glob.glob('results/phase1_cells_*.csv')],ignore_index=True)
print(f'{len(d)}행 / 셀 {d.groupby(["sensor","channel","algorithm"]).ngroups}개 / trial {d.trial_id.nunique()}개')
subj=d.groupby(['sensor','channel','algorithm','subject'])[['F1_HS','F1_TO']].mean().reset_index()
cell=subj.groupby(['sensor','channel','algorithm'])[['F1_HS','F1_TO']].agg(['mean','std'])
cell.columns=['HS_mean','HS_sd','TO_mean','TO_sd']; cell=cell.reset_index()
print('\n═══ Table 1. Top 5 (4명 평균 HS F1) ═══')
t1=cell.nlargest(5,'HS_mean')[['sensor','channel','algorithm','HS_mean','HS_sd']]
print(t1.round(3).to_string(index=False))
print('\n논문 Table 1: foot/Gyro_Y/TB_adapt 0.853±0.021 | foot/Gyro_Y/BPF 0.850±0.008')
print('              shank/Gyro_Z/TB_adapt 0.848±0.007 | shank/Gyro_Z/BPF 0.847±0.007')
print('              shank/Gyro_Z/TB_classic 0.842±0.007')
t1.round(3).to_csv('results/TABLE1_new.csv',index=False)

print('\n═══ Table 2. TB_adaptive vs TB_classic (Wilcoxon, 32 trial 쌍) ═══')
res=[]
for (s,ch),g in d[d.algorithm.isin(['TB_classic','TB_adaptive'])].groupby(['sensor','channel']):
    p=g.pivot_table(index='trial_id',columns='algorithm',values=['F1_HS','F1_TO'])
    for ev in ['HS','TO']:
        a=p[(f'F1_{ev}','TB_adaptive')].values; c=p[(f'F1_{ev}','TB_classic')].values
        if np.allclose(a,c): pv=1.0
        else:
            try: pv=wilcoxon(a,c,alternative='greater').pvalue
            except Exception: pv=1.0
        res.append(dict(sensor=s,channel=ch,event=ev,dF1=a.mean()-c.mean(),p=pv))
r=pd.DataFrame(res)
print(f'유의 셀 (p<0.05): {(r.p<0.05).sum()} / {len(r)}   [논문: 30 / 90]')
print('\n상위 5개 ΔF1:'); print(r.nlargest(5,'dF1')[['sensor','channel','event','dF1','p']].round(4).to_string(index=False))
print('\n논문 Table 2: thigh_R/Euler_roll/TO +0.533 | foot/Acc_Z/HS +0.311')
print('              foot/Acc_X/HS +0.309 | foot/Gyro_Y/HS +0.053 | foot/Gyro_X/HS +0.041')
r.to_csv('results/TABLE2_new.csv',index=False)

print('\n═══ surviving cells (max(4명평균 HS, TO) > 0.5) ═══')
cell['max_F1']=cell[['HS_mean','TO_mean']].max(axis=1)
surv=cell[cell.max_F1>0.5]
print(f'{len(surv)}개  [논문: 27개]')
print(surv.groupby('algorithm').size().to_dict())
surv[['sensor','channel','algorithm']].to_csv('results/cells_new.csv',index=False)