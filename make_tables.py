"""make_tables.py — 전통 vs 딥러닝 통합 표 생성

입력
  results/trad_ref_raw.csv        (src.traditional_ref 산출)
  results/dl_p4_*.csv             (run_phase4.py 산출, dr/curve/auc 파일 제외)
출력
  results/trad_ref_dr.csv         전통 알고리즘 DR
  results/dl_p4_dr.csv            딥러닝 DR
  results/combined_dr.csv         통합 (make_figs.py 가 이 파일을 읽음)
  results/table3_corrected_HS.csv / _TO.csv   전통+딥러닝 AUC 표

DR = min(F1_pert / F1_base, 1.0),  AUC = trapz(레벨별 평균 DR) / 4
교란 표기는 논문 기준(P1~P5)으로 변환해 출력한다.

실행: python make_tables.py
"""
import sys; sys.path.insert(0,'.')
import glob, numpy as np, pandas as pd
LV={'P1':[1,5,20,50,100],'P2':[5,10,15,20,30],'P4':[.01,.05,.10,.20,.30],
    'P6':[1,5,10,25,50],'P7':[60,45,30,20,10]}
PAPER={'P1':'P1 Noise','P2':'P2 Misalign','P4':'P3 Dropout','P6':'P4 Bias','P7':'P5 Downsamp'}

# 전통
t=pd.read_csv('results/trad_ref_raw.csv')
tb=(t[t.pert_type=='baseline'].groupby(['algorithm','position','trial_id'])
    [['F1_HS','F1_TO']].mean().rename(columns={'F1_HS':'b_HS','F1_TO':'b_TO'}).reset_index())
td=t[t.pert_type!='baseline'].merge(tb,on=['algorithm','position','trial_id'])
td['DR_HS']=(td.F1_HS/(td.b_HS+1e-9)).clip(0,1); td['DR_TO']=(td.F1_TO/(td.b_TO+1e-9)).clip(0,1)
td['method']=td.algorithm; td['family']='Traditional'; td['scope']='ref1'; td['norm']='n/a'
td.to_csv('results/trad_ref_dr.csv',index=False)

# 딥러닝
d=pd.concat([pd.read_csv(f) for f in [f for f in glob.glob('results/dl_p4_*.csv') if not f.endswith(('dr.csv','curve.csv','auc.csv'))]],ignore_index=True)
base=(d[d.pert_type=='baseline'].groupby(['model','position','trial_id','norm'])
      [['F1_HS','F1_TO']].mean().rename(columns={'F1_HS':'b_HS','F1_TO':'b_TO'}).reset_index())
dd=d[d.pert_type!='baseline'].merge(base,on=['model','position','trial_id','norm'])
dd['DR_HS']=(dd.F1_HS/(dd.b_HS+1e-9)).clip(0,1); dd['DR_TO']=(dd.F1_TO/(dd.b_TO+1e-9)).clip(0,1)
dd['method']=dd.model; dd['family']='Deep'
dd.to_csv('results/dl_p4_dr.csv',index=False)
cols=['family','method','position','scope','norm','pert_type','pert_level','trial_id','DR_HS','DR_TO','F1_HS','F1_TO']
allr=pd.concat([td[cols],dd[cols]],ignore_index=True); allr.to_csv('results/combined_dr.csv',index=False)

def auc_table(sub,ev='HS'):
    rows=[]
    for key,g in sub.groupby(['position','method']):
        r=dict(zip(['position','method'],key))
        for pt,lv in LV.items():
            y=[g[(g.pert_type==pt)&(np.isclose(g.pert_level,l))][f'DR_{ev}'].mean() for l in lv]
            r[PAPER[pt]]=np.trapezoid(y,range(5))/4
        rows.append(r)
    return pd.DataFrame(rows).set_index(['position','method'])

sel=pd.concat([allr[allr.family=='Traditional'],
    allr[(allr.family=='Deep')&(allr.scope=='all9')&(allr.norm=='adaptive')]])
print('═══ Table 3′ (정정판). AUC of DR — HS, n=32 ═══')
print(auc_table(sel).round(2).to_string())
auc_table(sel).round(3).to_csv('results/table3_corrected_HS.csv')
print('\n═══ AUC of DR — TO ═══'); print(auc_table(sel,'TO').round(2).to_string())
auc_table(sel,'TO').round(3).to_csv('results/table3_corrected_TO.csv')

print('\n═══ 정규화 모드 (AUC of DR, HS) ═══')
nz=allr[(allr.family=='Deep')&(allr.scope=='all9')]
for nm in ['adaptive','frozen']:
    print(f'-- norm = {nm} --'); print(auc_table(nz[nz.norm==nm]).round(3).to_string())

print('\n═══ 교란 범위 scope (AUC of DR, HS, adaptive) ═══')
sc=allr[(allr.family=='Deep')&(allr.norm=='adaptive')]
for s in ['ref1','gyro3','all9']:
    print(f'-- scope = {s} --'); print(auc_table(sc[sc.scope==s]).round(3).to_string())