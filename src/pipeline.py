"""
src/pipeline.py — Euler 변환 + 격자 평가 + 통계 검증
======================================================
실행: python -m src.pipeline   (또는 run.py에서 호출)
출력: results/cell_results.csv, summary_*.csv, stats_*.csv
"""
import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.stats import wilcoxon

# 프로젝트 루트를 sys.path에 추가 (imu_loader.py 임포트용)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    SUBJECT_XLSX, SUBJECT_GENDER, SENSORS, CHANNELS, Q_CANDIDATES,
    EULER_CONVENTION, TOL_MAIN_MS, TOL_AUX_MS,
    GT_DIR, IMU_DIR, RESULTS_DIR,
)
from src.algorithms import ALGORITHMS

from imu_loader import load_imu_sheet


# ───────── Euler 변환 ─────────
def find_q_cols(df):
    for cand in Q_CANDIDATES:
        if all(c in df.columns for c in cand):
            return cand
    return None


def add_euler_channels(df):
    """Q→Euler 추가 (Q=(w,x,y,z), 컨벤션 xyz, degrees)"""
    q_cols = find_q_cols(df)
    if q_cols is None:
        return df
    Q = df[q_cols].values.astype(float)
    norms = np.linalg.norm(Q, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    Q = Q / norms
    # scipy.from_quat은 (x,y,z,w) 순서 입력
    Q_xyzw = np.roll(Q, -1, axis=1)
    try:
        euler = R.from_quat(Q_xyzw).as_euler(EULER_CONVENTION, degrees=True)
        for i in range(3):
            euler[:, i] = np.rad2deg(np.unwrap(np.deg2rad(euler[:, i])))
    except Exception as e:
        print(f'    [경고] Euler 변환 실패: {e}')
        return df
    df = df.copy()
    df['Euler_roll'] = euler[:, 0]
    df['Euler_pitch'] = euler[:, 1]
    df['Euler_yaw'] = euler[:, 2]
    return df


# ───────── F1 매칭 ─────────
def match_F1(det_t, gt_t, tol_s):
    if len(det_t) == 0:
        return 0.0 if len(gt_t) else 1.0
    if len(gt_t) == 0:
        return 0.0
    det_t = np.sort(det_t)
    gt_t = np.sort(gt_t)
    pairs = []
    for di, d in enumerate(det_t):
        dists = np.abs(gt_t - d)
        within = np.where(dists <= tol_s)[0]
        for gi in within:
            pairs.append((dists[gi], di, gi))
    pairs.sort()
    md, mg = set(), set()
    for _, di, gi in pairs:
        if di not in md and gi not in mg:
            md.add(di)
            mg.add(gi)
    TP = len(md)
    FP = len(det_t) - TP
    FN = len(gt_t) - len(mg)
    return 2 * TP / (2 * TP + FP + FN) if (TP + FP + FN) else 0.0


def evaluate_channel(sig, t, fs, algo_fn, gt_HS, gt_TO,
                     tol_main, tol_aux):
    """sign +1, -1 둘 다 시도, F1_HS+F1_TO 합 기준 best 선택"""
    best = {'F1_HS_main': 0.0, 'F1_TO_main': 0.0,
            'F1_HS_aux': 0.0, 'F1_TO_aux': 0.0,
            'sign': 1, 'n_det_HS': 0, 'n_det_TO': 0}
    for sgn in (1, -1):
        try:
            HS_i, TO_i = algo_fn(sig * sgn, fs)
        except Exception:
            continue
        HS_t = t[HS_i] if len(HS_i) else np.array([])
        TO_t = t[TO_i] if len(TO_i) else np.array([])
        f_hs_m = match_F1(HS_t, gt_HS, tol_main)
        f_to_m = match_F1(TO_t, gt_TO, tol_main)
        if f_hs_m + f_to_m > best['F1_HS_main'] + best['F1_TO_main']:
            f_hs_a = match_F1(HS_t, gt_HS, tol_aux)
            f_to_a = match_F1(TO_t, gt_TO, tol_aux)
            best = {'F1_HS_main': f_hs_m, 'F1_TO_main': f_to_m,
                    'F1_HS_aux': f_hs_a, 'F1_TO_aux': f_to_a,
                    'sign': sgn,
                    'n_det_HS': len(HS_t), 'n_det_TO': len(TO_t)}
    return best


# ───────── trial 평가 ─────────
def evaluate_trial(xlsx, sheet, gt_csv, tol_main, tol_aux):
    sensors_data, _ = load_imu_sheet(xlsx, sheet)
    gt = pd.read_csv(gt_csv)
    gt = gt[gt['in_window'] == True]
    gt_HS = gt[gt['event'] == 'HS']['imu_time'].values
    gt_TO = gt[gt['event'] == 'TO']['imu_time'].values

    rows = []
    for sensor_id, sensor_name in SENSORS.items():
        if sensor_id not in sensors_data:
            continue
        df = sensors_data[sensor_id]
        df = add_euler_channels(df)
        t = df['t'].values
        fs = 1 / np.median(np.diff(t))
        for ch in CHANNELS:
            if ch not in df.columns:
                continue
            sig = df[ch].values.astype(float)
            sig = pd.Series(sig).interpolate(
                limit_direction='both').fillna(0).values
            for algo_name, algo_fn in ALGORITHMS.items():
                res = evaluate_channel(sig, t, fs, algo_fn,
                                       gt_HS, gt_TO,
                                       tol_main, tol_aux)
                rows.append({
                    'sensor': sensor_name, 'channel': ch,
                    'algorithm': algo_name,
                    'F1_HS_150': res['F1_HS_main'],
                    'F1_TO_150': res['F1_TO_main'],
                    'F1_HS_50': res['F1_HS_aux'],
                    'F1_TO_50': res['F1_TO_aux'],
                    'sign': res['sign'],
                    'n_det_HS': res['n_det_HS'],
                    'n_det_TO': res['n_det_TO'],
                    'n_gt_HS': len(gt_HS), 'n_gt_TO': len(gt_TO),
                })
    return rows


# ───────── trial 목록 자동 스캔 ─────────
def build_trial_list():
    trials = []
    for csv in sorted(GT_DIR.glob('*_synced.csv')):
        stem = csv.stem.replace('_synced', '')
        m = re.match(r'(\d{6})_(\w+?)_(\d+)(?:_(\d+))?$', stem)
        if not m:
            continue
        date, subj, speed, rep = m.groups()
        # KKH 정규화 (대소문자 무시)
        subj_upper = subj.upper()
        if subj_upper in SUBJECT_XLSX:
            subj_norm = subj_upper
        elif subj in SUBJECT_XLSX:
            subj_norm = subj
        else:
            continue
        xlsx = IMU_DIR / SUBJECT_XLSX[subj_norm]
        trials.append({
            'subject': subj_norm,
            'speed': int(speed),
            'rep': rep or '1',
            'gender': SUBJECT_GENDER[subj_norm],
            'xlsx': str(xlsx),
            'sheet': stem,
            'csv': str(csv),
            'trial_id': stem,
        })
    return trials


# ───────── 격자 실행 ─────────
def run_grid(trials):
    tol_main = TOL_MAIN_MS / 1000
    tol_aux = TOL_AUX_MS / 1000
    all_rows = []
    t_start = time.time()
    for i, tr in enumerate(trials, 1):
        elapsed = time.time() - t_start
        eta = elapsed / i * (len(trials) - i) if i > 0 else 0
        print(f'  [{i:>2}/{len(trials)}] {tr["trial_id"]:<25} '
              f'(경과 {elapsed:.0f}s, 남은 ~{eta:.0f}s)', flush=True)
        try:
            rows = evaluate_trial(tr['xlsx'], tr['sheet'], tr['csv'],
                                  tol_main, tol_aux)
        except Exception as e:
            print(f'    [실패] {e}')
            continue
        for r in rows:
            r.update({
                'subject': tr['subject'],
                'gender': tr['gender'],
                'speed': tr['speed'],
                'rep': tr['rep'],
                'trial_id': tr['trial_id'],
            })
            all_rows.append(r)
    print(f'\n총 {time.time()-t_start:.0f}초, {len(all_rows)} 셀')
    return pd.DataFrame(all_rows)


# ───────── 요약 CSV 생성 ─────────
def run_summaries(df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 피험자별
    by_subj = (df.groupby(['subject', 'sensor', 'channel', 'algorithm'])
               [['F1_HS_150', 'F1_TO_150', 'F1_HS_50', 'F1_TO_50']]
               .mean().reset_index())
    by_subj.to_csv(out_dir / 'summary_subject.csv', index=False)

    # 2) 4명 평균 + 표준편차
    by_avg = (df.groupby(['sensor', 'channel', 'algorithm'])
              [['F1_HS_150', 'F1_TO_150']]
              .agg(['mean', 'std']).reset_index())
    by_avg.columns = ['_'.join(c).strip('_') for c in by_avg.columns]
    by_avg.to_csv(out_dir / 'summary_overall.csv', index=False)

    # 3) 속도별
    by_speed = (df.groupby(['subject', 'speed',
                             'sensor', 'channel', 'algorithm'])
                [['F1_HS_150', 'F1_TO_150']]
                .mean().reset_index())
    by_speed.to_csv(out_dir / 'summary_speed.csv', index=False)

    # 4) 성별 평균 + 표준편차
    by_gender = (df.groupby(['gender', 'sensor', 'channel', 'algorithm'])
                 [['F1_HS_150', 'F1_TO_150']]
                 .agg(['mean', 'std']).reset_index())
    by_gender.columns = ['_'.join(c).strip('_') for c in by_gender.columns]
    by_gender.to_csv(out_dir / 'summary_gender.csv', index=False)

    # 5) Top-20 (HS, TO)
    for event in ('HS', 'TO'):
        col = f'F1_{event}_150'
        top = (df.groupby(['sensor', 'channel', 'algorithm'])[col]
               .mean().reset_index()
               .sort_values(col, ascending=False).head(20))
        top.to_csv(out_dir / f'top20_{event}.csv', index=False)


# ───────── 통계 검증 (Wilcoxon paired) ─────────
def run_stats(df, out_dir):
    """TB_classic vs TB_adaptive (paired, alternative='greater')"""
    out_dir = Path(out_dir)
    rows = []
    for sensor in df['sensor'].unique():
        for channel in df['channel'].unique():
            sub = df[(df['sensor'] == sensor) & (df['channel'] == channel)]
            if len(sub) < 8:
                continue
            for event in ('HS', 'TO'):
                col = f'F1_{event}_150'
                pivot = sub.pivot_table(
                    index='trial_id', columns='algorithm',
                    values=col, aggfunc='mean')
                if 'TB_classic' not in pivot.columns or \
                   'TB_adaptive' not in pivot.columns:
                    continue
                paired = pivot[['TB_classic', 'TB_adaptive']].dropna()
                if len(paired) < 5:
                    continue
                d = paired['TB_adaptive'].values - paired['TB_classic'].values
                if np.all(d == 0):
                    p = 1.0
                    stat = 0.0
                else:
                    try:
                        stat, p = wilcoxon(d, alternative='greater')
                    except Exception:
                        continue
                rows.append({
                    'sensor': sensor,
                    'channel': channel,
                    'event': event,
                    'n_trials': len(paired),
                    'mean_classic': paired['TB_classic'].mean(),
                    'mean_adaptive': paired['TB_adaptive'].mean(),
                    'mean_diff': float(d.mean()),
                    'wilcoxon_stat': float(stat),
                    'p_value': float(p),
                    'significant': bool(p < 0.05),
                })
    stats_df = pd.DataFrame(rows).sort_values('p_value')
    stats_df.to_csv(out_dir / 'stats_wilcoxon.csv', index=False)
    return stats_df


# ───────── 메인 ─────────
def main():
    print(f'GT_DIR  = {GT_DIR}')
    print(f'IMU_DIR = {IMU_DIR}')
    print(f'RESULTS = {RESULTS_DIR}\n')

    trials = build_trial_list()
    print(f'trial: {len(trials)}개')
    if not trials:
        raise FileNotFoundError(f'synced csv 없음 in {GT_DIR}')

    df = run_grid(trials)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / 'cell_results.csv', index=False)
    print(f'\n[저장] {RESULTS_DIR/"cell_results.csv"} ({len(df)}행)')

    print('\n── 요약 CSV 생성 ──')
    run_summaries(df, RESULTS_DIR)
    print(f'  summary_subject.csv, summary_overall.csv,')
    print(f'  summary_speed.csv, summary_gender.csv,')
    print(f'  top20_HS.csv, top20_TO.csv')

    print('\n── 통계 검증 (Wilcoxon: TB_classic < TB_adaptive) ──')
    stats_df = run_stats(df, RESULTS_DIR)
    if len(stats_df):
        sig = stats_df[stats_df['significant']]
        print(f'  유의(p<0.05) 셀: {len(sig)}/{len(stats_df)}')
        print('\n  Top-5 가장 유의한 개선:')
        top_sig = stats_df.head(5)
        for _, r in top_sig.iterrows():
            print(f"    {r['sensor']:<12} {r['channel']:<14} "
                  f"{r['event']}  Δ={r['mean_diff']:+.3f}  "
                  f"p={r['p_value']:.4f}")

    return df


if __name__ == '__main__':
    main()
