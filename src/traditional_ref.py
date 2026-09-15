"""
src/traditional_ref.py — 전통 알고리즘 기준선 (딥러닝과 동일 교란 정의)
=======================================================================
목적: Fig. "전통 vs 딥러닝 DR 곡선 오버레이"를 만들려면 두 쪽이
      **완전히 같은 교란 신호**를 봐야 한다.

그래서 여기서는 phase2의 단일채널 교란 함수를 쓰지 않고,
perturbations_mc 로 9채널을 교란한 뒤 **기준 채널 하나만 뽑아서**
전통 알고리즘에 넣는다. 이렇게 하면

  - P1/P6 : 기준 채널의 환산계수 = 1.0 이므로 phase2와 수치적으로 동일
  - P2    : phase2와 동일 (삼축 회전 후 1축 추출)
  - P4    : phase2와 동일 (해당 채널 인덱스 드롭)
  - P7    : **다름** — phase2는 새 시간축 반환, 여기서는 원 시간축 재보간.
            딥러닝이 길이를 유지해야 하므로 이 쪽에 맞춘 것.
            → 논문에는 "P7은 두 방식 모두 보고" 하거나 재보간 방식으로 통일

기준 셀 (1단계 best 단일채널):
  shank : Gyro_Z
  foot  : Gyro_Y
알고리즘: TB_classic, TB_adaptive, BPF, TM

출력: results/trad_ref_raw.csv
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TOL_MAIN_MS, RESULTS_DIR
from src.algorithms import ALGORITHMS
from src.pipeline import match_F1
from src.perturbations_mc import (PERTURBATIONS_MC, apply_perturbation_mc,
                                  IDX)

TOL = TOL_MAIN_MS / 1000
REF_CHANNEL = {'shank': 'Gyro_Z', 'foot': 'Gyro_Y'}


def best_sign(sig, t, fs, algo_fn, gt_HS, gt_TO):
    """clean 신호에서 sign(+1/-1) 결정 — phase2와 동일하게 이후 고정"""
    best, best_sgn = -1, 1
    for sgn in (1, -1):
        try:
            HS_i, TO_i = algo_fn(sig * sgn, fs)
        except Exception:
            continue
        f = (match_F1(t[HS_i] if len(HS_i) else np.array([]), gt_HS, TOL)
             + match_F1(t[TO_i] if len(TO_i) else np.array([]), gt_TO, TOL))
        if f > best:
            best, best_sgn = f, sgn
    return best_sgn


def eval_once(sig, t, fs, algo_fn, gt_HS, gt_TO, sign):
    try:
        HS_i, TO_i = algo_fn(sig * sign, fs)
    except Exception:
        return 0.0, 0.0
    HS_t = t[HS_i] if len(HS_i) else np.array([])
    TO_t = t[TO_i] if len(TO_i) else np.array([])
    return match_F1(HS_t, gt_HS, TOL), match_F1(TO_t, gt_TO, TOL)


def run(raw_trials, position, algos=None):
    algos = algos or list(ALGORITHMS.keys())
    ref_ch = REF_CHANNEL[position]
    ci = IDX[ref_ch]
    conds = [('baseline', 0)]
    for pt, info in PERTURBATIONS_MC.items():
        conds += [(pt, lv) for lv in info['levels']]

    rows = []
    t0 = time.time()
    for k, rt in enumerate(raw_trials, 1):
        t = rt['t']
        fs = 1.0 / np.median(np.diff(t))
        clean = rt['sig9'][:, ci]
        signs = {a: best_sign(clean, t, fs, ALGORITHMS[a],
                              rt['gt_HS'], rt['gt_TO']) for a in algos}
        for pt, lv in conds:
            sig_p = apply_perturbation_mc(pt, rt['sig9'], t, lv,
                                          ref_channel=ref_ch,
                                          scope='all9')[:, ci]
            for a in algos:
                f_hs, f_to = eval_once(sig_p, t, fs, ALGORITHMS[a],
                                       rt['gt_HS'], rt['gt_TO'], signs[a])
                rows.append({
                    'algorithm': a, 'position': position, 'channel': ref_ch,
                    'subject': rt['subject'], 'gender': rt['gender'],
                    'speed': rt['speed'], 'rep': rt['rep'],
                    'trial_id': rt['trial_id'],
                    'pert_type': pt, 'pert_level': lv,
                    'F1_HS': f_hs, 'F1_TO': f_to, 'sign': signs[a],
                })
        print(f'  [{k:>2}/{len(raw_trials)}] {rt["trial_id"]:<20} '
              f'({time.time()-t0:.0f}s)', flush=True)
    return pd.DataFrame(rows)


def main():
    from src.trials_adapter import get_trials
    from src.pipeline_phase4 import build_raw_trials
    trials = get_trials(verbose=False)
    dfs = []
    for pos in ['shank', 'foot']:
        print(f'\n=== 전통 알고리즘 기준선 | {pos} ({REF_CHANNEL[pos]}) ===')
        raw = build_raw_trials(trials, pos)
        dfs.append(run(raw, pos))
    df = pd.concat(dfs, ignore_index=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / 'trad_ref_raw.csv', index=False)
    print(f'\n[저장] trad_ref_raw.csv ({len(df)}행)')
    base = df[df['pert_type'] == 'baseline']
    print('\n━━━ baseline F1 (전체 trial 평균) ━━━')
    print(base.groupby(['position', 'algorithm'])
          [['F1_HS', 'F1_TO']].mean().round(3).to_string())


if __name__ == '__main__':
    main()