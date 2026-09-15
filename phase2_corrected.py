"""
phase2_corrected.py — 27개 셀 전체 재실행 (P7 정정)
====================================================
pipeline_phase2.py 를 그대로 재현하되 P7 처리만 4가지 방식으로 분해한다.

  M0 orig        기존 구현: 인덱스 기반 시간축 복원, 알고리즘은 fs_new 에서 실행
  M1 native      시간축 정정, 알고리즘은 fs_new 에서 실행 (컷오프 원본 12 Hz)
  M2 native_nyq  시간축 정정 + Nyquist 적응 컷오프 min(12, 0.4*fs)
  M3 resample    시간축 정정 + 원 91 Hz 격자로 재보간 (딥러닝과 동일 조건)

M3 에서는 알고리즘이 항상 91 Hz 를 보므로 Nyquist 문제가 발생하지 않는다.
P1/P2/P4/P6 는 수정 대상이 아니므로 한 번만 계산한다.

셀 목록·sign·DR·AUC 규칙은 pipeline_phase2.py 와 완전히 동일.
"""
import sys, time, glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT))
from src.config import TOL_MAIN_MS
from src.pipeline import match_F1
from src.algorithms import ALGORITHMS
from src.algorithms_patch import detect_TB_classic_nyq
from src.perturbations import apply_perturbation, apply_P7_downsample, PERTURBATIONS
from src.perturbations_mc import mc_P7_downsample, IDX, CH_ORDER

TOL=TOL_MAIN_MS/1000
LEVELS={p:PERTURBATIONS[p]['levels'] for p in ['P1','P2','P4','P6','P7']}
P7MODES=['orig','native','native_nyq','resample']

def _nyq_wrap(fn):
    """알고리즘 내부 butter 컷오프를 Nyquist 안전하게 감싼다 (TM/BPF용)."""
    import src.algorithms as A
    def wrapped(sig,fs):
        orig=A.butter
        def patched(N,Wn,*a,**kw):
            f=kw.get('fs')
            if f and np.isscalar(Wn) and Wn>=f/2: Wn=0.4*f
            return orig(N,Wn,*a,**kw)
        A.butter=patched
        try: return fn(sig,fs)
        finally: A.butter=orig
    return wrapped

def get_algo(name,nyq):
    if not nyq: return ALGORITHMS[name]
    if name=='TB_classic': return detect_TB_classic_nyq
    return _nyq_wrap(ALGORITHMS[name])

def native_p7(sig9,t,fs_new,ci):
    """시간축 정정 + 네이티브 저레이트 (알고리즘이 fs_new 에서 실행)"""
    from fractions import Fraction
    from scipy.signal import resample_poly
    fs=1/np.median(np.diff(t))
    t_uni=np.arange(t[0],t[-1],1/fs)
    s_uni=np.interp(t_uni,t,sig9[:,ci])
    fr=Fraction(int(round(fs_new*100)),int(round(fs*100))).limit_denominator(100)
    s=resample_poly(s_uni,fr.numerator,fr.denominator)
    fa=fs*fr.numerator/fr.denominator
    return s, t_uni[0]+np.arange(len(s))/fa, fa

def run(sensors):
    cells=pd.read_csv('results/cells27.csv').merge(
        pd.read_csv('results/sign_map27.csv'),on=['sensor','channel','algorithm'])
    rows=[]; t0=time.time()
    for sensor in sensors:
        sub=cells[cells.sensor==sensor]
        files=sorted(glob.glob(f'results/raw_cache/*_{sensor}.npz'))
        print(f'[{sensor}] 셀 {len(sub)} × trial {len(files)}',flush=True)
        for k,f in enumerate(files,1):
            z=np.load(f); t,sig9=z['t'],z['sig9']; gHS,gTO=z['gt_HS'],z['gt_TO']
            fs=1/np.median(np.diff(t)); tid=f.split('/')[-1].replace(f'_{sensor}.npz','')
            df9=pd.DataFrame(sig9,columns=CH_ORDER)
            for _,c in sub.iterrows():
                ci=IDX[c.channel]; sg=int(c['sign']); ch=c.channel
                def ev(fn,s,tt,ff):
                    try: HS,TO=fn(s*sg,ff)
                    except Exception: HS,TO=np.array([],int),np.array([],int)
                    return (match_F1(tt[HS] if len(HS) else np.array([]),gHS,TOL),
                            match_F1(tt[TO] if len(TO) else np.array([]),gTO,TOL))
                fn0=get_algo(c.algorithm,False)
                # baseline + P1/P2/P4/P6 (수정 대상 아님)
                for pt,lv in [('baseline',0)]+[(p,l) for p in ['P1','P2','P4','P6'] for l in LEVELS[p]]:
                    if pt=='baseline': s,tt,ff=sig9[:,ci],t,fs
                    else: s,tt,ff=apply_perturbation(pt,sig9[:,ci],t,fs,lv,df=df9,channel=ch)
                    fH,fT=ev(fn0,s,tt,ff)
                    rows.append(dict(p7mode='all',sensor=sensor,channel=ch,algorithm=c.algorithm,
                        trial_id=tid,pert_type=pt,pert_level=lv,F1_HS=fH,F1_TO=fT))
                # P7 4가지 방식
                for lv in LEVELS['P7']:
                    for m in P7MODES:
                        if m=='orig': s,tt,ff=apply_P7_downsample(sig9[:,ci],t,lv); fn=fn0
                        elif m=='native': s,tt,ff=native_p7(sig9,t,lv,ci); fn=fn0
                        elif m=='native_nyq': s,tt,ff=native_p7(sig9,t,lv,ci); fn=get_algo(c.algorithm,True)
                        else: s,tt,ff=mc_P7_downsample(sig9,t,lv,[ci])[:,ci],t,fs; fn=fn0
                        fH,fT=ev(fn,s,tt,ff)
                        rows.append(dict(p7mode=m,sensor=sensor,channel=ch,algorithm=c.algorithm,
                            trial_id=tid,pert_type='P7',pert_level=lv,F1_HS=fH,F1_TO=fT))
            print(f'  [{k:>2}/{len(files)}] {tid:<20} {len(rows):>7}행 {time.time()-t0:.0f}s',flush=True)
    return pd.DataFrame(rows)

if __name__=='__main__':
    out=run(sys.argv[1].split(','))
    tag=sys.argv[1].replace(',','-')
    out.to_csv(f'results/p2c_{tag}.csv',index=False)
    print(f'[저장] p2c_{tag}.csv ({len(out)}행)')