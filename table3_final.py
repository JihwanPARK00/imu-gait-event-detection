import sys; sys.path.insert(0,'.')
import glob, numpy as np, pandas as pd
PAPER={'P1':'P1 Noise','P2':'P2 Misalign','P4':'P3 Dropout','P6':'P4 Bias','P7':'P5 Downsamp'}
d=pd.concat([pd.read_csv(f) for f in glob.glob('results/p2c_*.csv')],ignore_index=True)
d['subject']=d.trial_id.str.split('_').str[1].str.upper()
print(f'{len(d)}행, 셀 {d.groupby(["sensor","channel","algorithm"]).ngroups}개, trial {d.trial_id.nunique()}개')

def dr_table(p7mode, clip):
    # pipeline_phase2 규칙: baseline 은 subject x cell 평균, DR=F1/base, base<=0.1 이면 NaN
    df=d[(d.p7mode=='all')|(d.p7mode==p7mode)].copy()
    base=(df[df.pert_type=='baseline'].groupby(['subject','sensor','channel','algorithm'])
          [['F1_HS','F1_TO']].mean().rename(columns={'F1_HS':'bH','F1_TO':'bT'}).reset_index())
    m=df[df.pert_type!='baseline'].merge(base,on=['subject','sensor','channel','algorithm'])
    # ── A안: 이벤트별로 셀 집합 분리 ──────────────────────────────
    # 셀 선정 기준이 max(HS, TO) > 0.5 이므로, HS 는 거의 검출되지 않는데
    # TO 만으로 통과한 셀이 섞인다 (예: thigh Euler_roll HS F1 = 0.03).
    # 그런 셀의 HS DR 은 의미가 없으므로 이벤트별로 따로 거른다.
    cellbase=(base.groupby(['sensor','channel','algorithm'])[['bH','bT']]
              .mean().rename(columns={'bH':'cH','bT':'cT'}).reset_index())
    m=m.merge(cellbase,on=['sensor','channel','algorithm'])
    for ev,b,cb in [('HS','bH','cH'),('TO','bT','cT')]:
        r=m[f'F1_{ev}']/m[b]
        if clip: r=np.minimum(r,1.0)
        m[f'DR_{ev}']=np.where((m[b]>0.1)&(m[cb]>0.5), r, np.nan)
    rows=[]
    for (a,pt),s in m.groupby(['algorithm','pert_type']):
        lv=sorted(s.pert_level.unique()); nx=np.linspace(0,1,len(lv))
        for ev in ['HS','TO']:
            mm=np.array([s[s.pert_level==l][f'DR_{ev}'].mean() for l in lv])
            rows.append({'algorithm':a,'pert':PAPER[pt],'event':ev,
                         'AUC':np.trapezoid(np.nan_to_num(mm),nx)})
    t=pd.DataFrame(rows)
    return {ev:t[t.event==ev].pivot(index='algorithm',columns='pert',values='AUC')[list(PAPER.values())]
            for ev in ['HS','TO']}

# 이벤트별 셀 수 보고
_b=(d[d.pert_type=='baseline'].groupby(['sensor','channel','algorithm'])
    [['F1_HS','F1_TO']].mean())
print(f'HS 유효 셀 {(_b.F1_HS>0.5).sum()}개 / TO 유효 셀 {(_b.F1_TO>0.5).sum()}개'
      f'  (전체 {len(_b)}개)')

paper=pd.DataFrame({'P1 Noise':[0.84,0.92,0.74],'P2 Misalign':[0.91,0.94,0.96],
  'P3 Dropout':[1.00,1.00,1.00],'P4 Bias':[0.86,1.00,0.90],'P5 Downsamp':[0.21,0.42,0.17]},
  index=['TB_classic','BPF','TM']).sort_index()
print('\n═══ 논문 Table 3 (게재값) ═══'); print(paper.to_string())
for mode,label in [('orig','M0 기존 구현 (논문 재현 검증)'),
                   ('native','M1 시간축 정정 (네이티브 저레이트)'),
                   ('native_nyq','M2 시간축 + Nyquist 컷오프'),
                   ('resample','M3 시간축 정정 + 91 Hz 재보간')]:
    for clip,cl in [(True,'클리핑'),(False,'비클리핑')]:
        if mode!='orig' and not clip: continue
        t=dr_table(mode,clip)['HS']
        print(f'\n═══ {label} / DR {cl} — HS ═══'); print(t.round(2).to_string())
        t.round(3).to_csv(f'results/table3_{mode}_{"clip" if clip else "noclip"}_HS.csv')
final=dr_table('native_nyq',True)
final['HS'].round(3).to_csv('results/TABLE3_FINAL_HS.csv')
final['TO'].round(3).to_csv('results/TABLE3_FINAL_TO.csv')
print('\n═══ 최종 정정판 (M2, 클리핑) — TO ═══'); print(final['TO'].round(2).to_string())
# P7 레벨별
df=d[d.pert_type=='P7']
base=(d[d.pert_type=='baseline'].groupby(['subject','sensor','channel','algorithm'])
      .F1_HS.mean().rename('bH').reset_index())
m=df.merge(base,on=['subject','sensor','channel','algorithm'])
m['DR_HS']=np.where(m.bH>0.1,np.minimum(m.F1_HS/m.bH,1.0),np.nan)
print('\n═══ P7 레벨별 평균 DR(HS), 27셀 ═══')
print(m.pivot_table(index=['p7mode','algorithm'],columns='pert_level',values='DR_HS').round(3).to_string())
m.to_csv('results/p7_levels_27cells.csv',index=False)