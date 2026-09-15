"""make_figs.py — 그림 A~D 생성

  figA_trad_vs_deep.png   DR 곡선 2행(shank/foot) x 5열(P1~P7)
                          전통 알고리즘 실선 / 딥러닝 점선
  figB_normalization.png  adaptive vs frozen 정규화 대비
  figC_scope.png          교란 채널 수 1/3/9 — 다채널 중복성
  figD_p7_bug.png         P7 3단 패널 (shank Gyro_Z 단일 셀)
                          ※ 선택 사항. p7_audit/p7_native/p7_nyquist CSV가
                            있을 때만 생성된다. 27셀 버전인 figE.py 가
                            상위 호환이므로 figD 는 없어도 무방하다.

선행 조건: make_tables.py 를 먼저 실행해 results/combined_dr.csv 생성
실행:      python make_figs.py
"""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'axes.linewidth':0.8,'figure.dpi':300})

df=pd.read_csv('results/combined_dr.csv')
PERT={'P1':('P1 Noise','$\\sigma$ (deg/s)',[1,5,20,50,100]),
      'P2':('P2 Misalignment','$\\theta$ (deg)',[5,10,15,20,30]),
      'P4':('P4 Dropout','loss ratio',[0.01,0.05,0.10,0.20,0.30]),
      'P6':('P6 Bias','offset (deg/s)',[1,5,10,25,50]),
      'P7':('P7 Downsampling','$f_s$ (Hz)',[60,45,30,20,10])}
TRAD=['TB_classic','TB_adaptive','BPF','TM']; DEEP=['CNN','LSTM']
C={'TB_classic':'#888888','TB_adaptive':'#2E7D32','BPF':'#1565C0','TM':'#6A1B9A',
   'CNN':'#D81B60','LSTM':'#F57C00'}

# ── Fig A: DR curves, traditional vs deep ──
fig,axes=plt.subplots(2,5,figsize=(11,4.6),sharey=True)
for r,pos in enumerate(['shank','foot']):
    for c,(pt,(title,xlab,levels)) in enumerate(PERT.items()):
        ax=axes[r,c]
        for m in TRAD:
            s=df[(df.method==m)&(df.position==pos)&(df.pert_type==pt)]
            y=[s[np.isclose(s.pert_level,l)].DR_HS.mean() for l in levels]
            ax.plot(range(5),y,'-o',color=C[m],ms=3,lw=1.2,label=m if (r==0 and c==0) else None)
        for m in DEEP:
            s=df[(df.method==m)&(df.position==pos)&(df.pert_type==pt)&
                 (df.scope=='all9')&(df.norm=='adaptive')]
            y=[s[np.isclose(s.pert_level,l)].DR_HS.mean() for l in levels]
            ax.plot(range(5),y,'--s',color=C[m],ms=3.5,lw=1.6,label=m if (r==0 and c==0) else None)
        ax.set_xticks(range(5)); ax.set_xticklabels(levels,fontsize=6.5)
        ax.set_ylim(0.3,1.03); ax.grid(alpha=.25,lw=.5)
        if r==0: ax.set_title(title,fontsize=8.5)
        if r==1: ax.set_xlabel(xlab,fontsize=7.5)
        if c==0: ax.set_ylabel(f'{pos}\nDR (HS)',fontsize=8.5)
fig.legend(loc='upper center',ncol=6,fontsize=7.5,frameon=False,bbox_to_anchor=(0.5,1.005))
fig.suptitle('',y=1)
plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig('results/figA_trad_vs_deep.png',bbox_inches='tight'); plt.close()

# ── Fig B: normalization ──
d=df[(df.family=='Deep')&(df.scope=='all9')]
fig,axes=plt.subplots(1,2,figsize=(8,3),sharey=True)
w=0.2; pts=list(PERT.keys())
for i,pos in enumerate(['shank','foot']):
    ax=axes[i]; x=np.arange(5)
    for j,(m,nm) in enumerate([('CNN','adaptive'),('CNN','frozen'),('LSTM','adaptive'),('LSTM','frozen')]):
        s=d[(d.method==m)&(d.position==pos)&(d.norm==nm)]
        y=[s[s.pert_type==p].DR_HS.mean() for p in pts]
        ax.bar(x+(j-1.5)*w,y,w,label=f'{m} / {nm}' if i==0 else None,
               color=C[m],alpha=1.0 if nm=='adaptive' else 0.45,
               edgecolor='k',lw=.4,hatch='' if nm=='adaptive' else '///')
    ax.set_xticks(x); ax.set_xticklabels(pts); ax.set_ylim(0.7,1.02)
    ax.set_title(pos,fontsize=9); ax.grid(axis='y',alpha=.25,lw=.5)
    if i==0: ax.set_ylabel('DR (HS)')
axes[0].legend(fontsize=6.5,ncol=2,frameon=False,loc='lower left')
plt.tight_layout(); plt.savefig('results/figB_normalization.png',bbox_inches='tight'); plt.close()

# ── Fig C: scope (multi-channel redundancy) ──
d=df[(df.family=='Deep')&(df.norm=='adaptive')]
fig,axes=plt.subplots(1,2,figsize=(8,3),sharey=True)
SC=['ref1','gyro3','all9']; SL=['1 ch','3 ch (gyro)','9 ch (all)']
for i,pos in enumerate(['shank','foot']):
    ax=axes[i]; x=np.arange(5)
    for j,m in enumerate(DEEP):
        for k,sc in enumerate(SC):
            s=d[(d.method==m)&(d.position==pos)&(d.scope==sc)]
            y=[s[s.pert_type==p].DR_HS.mean() for p in pts]
            ax.plot(x,y,marker='osd'[k],ls=['-','--',':'][k],color=C[m],ms=4,lw=1.2,
                    alpha=0.5+0.25*k,label=f'{m}: {SL[k]}' if i==0 else None)
    ax.set_xticks(x); ax.set_xticklabels(pts); ax.set_ylim(0.85,1.01)
    ax.set_title(pos,fontsize=9); ax.grid(alpha=.25,lw=.5)
    if i==0: ax.set_ylabel('DR (HS)')
axes[0].legend(fontsize=6.5,ncol=2,frameon=False,loc='lower left')
plt.tight_layout(); plt.savefig('results/figC_scope.png',bbox_inches='tight'); plt.close()

# ── Fig D: P7 bug evidence ──
import os
_need = ['results/p7_audit.csv','results/p7_native.csv','results/p7_nyquist.csv']
if all(os.path.exists(f) for f in _need):
    au=pd.read_csv('results/p7_audit.csv'); na=pd.read_csv('results/p7_native.csv')
    for c in ['old','new']: au[f'DR_{c}']=(au[f'{c}_HS']/(au.base_HS+1e-9)).clip(0,1)
    na['DR_nat']=(na.nat_HS/(na.base_HS+1e-9)).clip(0,1)
    ny=pd.read_csv('results/p7_nyquist.csv'); ny['DR']=(ny.nat_HS/(ny.base_HS+1e-9)).clip(0,1)
    fig,axes=plt.subplots(1,3,figsize=(10,2.9),sharey=True)
    lv=[60,45,30,20,10]
    for a in ['TB_classic','BPF','TM']:
        s=au[(au.position=='shank')&(au.algorithm==a)]
        axes[0].plot(range(5),[s[s.fs_new==l].DR_old.mean() for l in lv],'-o',color=C[a],ms=3.5,label=a)
        t=na[(na.position=='shank')&(na.algorithm==a)]
        axes[1].plot(range(5),[t[t.fs_new==l].DR_nat.mean() for l in lv],'-o',color=C[a],ms=3.5)
    for a,st,col in [('TB_classic','-o','#888888'),('TB_nyquist_fixed','-s','#C62828')]:
        s=ny[(ny.position=='shank')&(ny.algorithm==a)]
        axes[2].plot(range(5),[s[s.fs_new==l].DR.mean() for l in lv],st,color=col,ms=3.5,label=a)
    for i,t in enumerate(['(a) Published implementation','(b) Time-axis corrected','(c) + Nyquist-aware cutoff']):
        axes[i].set_xticks(range(5)); axes[i].set_xticklabels(lv); axes[i].set_ylim(-0.03,1.05)
        axes[i].set_title(t,fontsize=8.5); axes[i].grid(alpha=.25,lw=.5); axes[i].set_xlabel('$f_s$ (Hz)')
    axes[0].set_ylabel('DR (HS), shank Gyro_Z'); axes[0].legend(fontsize=7,frameon=False)
    axes[2].legend(fontsize=7,frameon=False)
    plt.tight_layout(); plt.savefig('results/figD_p7_bug.png',bbox_inches='tight'); plt.close()
    print('saved figA-figD')
else:
    print('figD 건너뜀 (p7_audit/p7_native/p7_nyquist CSV 없음) — figE.py 를 사용하세요)')
    print('saved figA-figC')