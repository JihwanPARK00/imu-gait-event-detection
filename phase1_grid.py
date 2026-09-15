"""phase1_grid.py — 1단계 전체 격자 재계산 (RE_ GT 기준)

5 위치 x 9 채널 x 4 알고리즘 x 32 trial.
논문 Table 1 (Top5) 과 Table 2 (TB_classic vs TB_adaptive, Wilcoxon) 를 재현한다.

sign 은 논문과 동일하게 trial 별로 F1HS + F1TO 를 최대화하는 쪽을 선택.

  python phase1_grid.py                 # 전체
  python phase1_grid.py shank foot      # 위치 지정 (분할 실행용)

출력: results/phase1_cells_{위치}.csv
"""
import sys, time, glob
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.config import TOL_MAIN_MS
from src.pipeline import match_F1
from src.algorithms import ALGORITHMS
from src.perturbations_mc import CH_ORDER

TOL = TOL_MAIN_MS / 1000
POS = sys.argv[1:] or ['foot', 'shank', 'thigh_aff', 'thigh_sound', 'lumbar']


def eval_cell(sig, t, fs, fn, gH, gT):
    best = (-1, -1, 1)
    for sgn in (1, -1):
        try:
            HS, TO = fn(sig * sgn, fs)
        except Exception:
            continue
        fH = match_F1(t[HS] if len(HS) else np.array([]), gH, TOL)
        fT = match_F1(t[TO] if len(TO) else np.array([]), gT, TOL)
        if fH + fT > best[0] + best[1]:
            best = (fH, fT, sgn)
    return best


for pos in POS:
    files = sorted(glob.glob(f'results/raw_cache/*_{pos}.npz'))
    if not files:
        print(f'[{pos}] 캐시 없음 — 건너뜀')
        continue
    rows = []
    t0 = time.time()
    for k, f in enumerate(files, 1):
        z = np.load(f)
        t, sig9, gH, gT = z['t'], z['sig9'], z['gt_HS'], z['gt_TO']
        fs = 1.0 / np.median(np.diff(t))
        tid = f.split('/')[-1].replace(f'_{pos}.npz', '')
        subj = tid.split('_')[1].upper()
        for ci, ch in enumerate(CH_ORDER):
            for a, fn in ALGORITHMS.items():
                fH, fT, sgn = eval_cell(sig9[:, ci], t, fs, fn, gH, gT)
                rows.append(dict(sensor=pos, channel=ch, algorithm=a,
                                 trial_id=tid, subject=subj,
                                 F1_HS=fH, F1_TO=fT, sign=sgn))
        print(f'  [{k:>2}/{len(files)}] {tid:<20} {time.time()-t0:.0f}s',
              flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(f'results/phase1_cells_{pos}.csv', index=False)
    print(f'[저장] phase1_cells_{pos}.csv ({len(d)}행)')