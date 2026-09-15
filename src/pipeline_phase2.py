"""
src/pipeline_phase2.py — 2단계 교란 격자 평가
================================================
1단계 cell_results.csv에서 F1>0.5 셀 추출 → 5개 교란 × 5 레벨 적용
→ results/cell_results_phase2.csv

알고리즘 3개: TB_classic, BPF, TM (TB_adaptive는 1단계 결론으로 확정)
sign은 1단계 best 그대로 사용 (재계산 X)
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

# numpy 2.x 호환 (np.trapz → np.trapezoid)
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz', None)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    SUBJECT_XLSX, SUBJECT_GENDER, SENSORS,
    TOL_MAIN_MS, GT_DIR, IMU_DIR, RESULTS_DIR,
)
from src.algorithms import ALGORITHMS
from src.pipeline import add_euler_channels, match_F1, build_trial_list
from src.perturbations import (PERTURBATIONS, COMPOUND_PERTURBATIONS,
                                LEVEL_LABELS, apply_perturbation,
                                apply_compound)

from imu_loader import load_imu_sheet


# ─────────── 2단계 알고리즘 (TB_adaptive 제외) ───────────
PHASE2_ALGORITHMS = {
    'TB_classic': ALGORITHMS['TB_classic'],
    'BPF': ALGORITHMS['BPF'],
    'TM': ALGORITHMS['TM'],
}

F1_THRESHOLD = 0.5  # 살아남은 셀 기준
TOL = TOL_MAIN_MS / 1000  # 0.15s


# ─────────── 살아남은 셀 추출 ───────────
def load_surviving_cells(threshold=F1_THRESHOLD):
    """1단계 결과에서 F1>threshold 셀 (Phase 2 알고리즘만).

    Returns
    -------
    cells : list of dict — [{'sensor', 'channel', 'algorithm'}, ...]
    sign_map : dict — {(sensor, channel, algorithm): sign}
    """
    df = pd.read_csv(RESULTS_DIR / 'cell_results.csv')
    df = df[df['algorithm'].isin(PHASE2_ALGORITHMS.keys())]

    # 4명 평균 F1 계산
    by_combo = (df.groupby(['sensor', 'channel', 'algorithm'])
                [['F1_HS_150', 'F1_TO_150']].mean().reset_index())
    by_combo['max_F1'] = by_combo[['F1_HS_150', 'F1_TO_150']].max(axis=1)
    survivors = by_combo[by_combo['max_F1'] > threshold].copy()

    # sign은 가장 흔한 (mode) 값 사용
    sign_df = (df.groupby(['sensor', 'channel', 'algorithm'])['sign']
               .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else 1)
               .reset_index())
    sign_map = {(r['sensor'], r['channel'], r['algorithm']): int(r['sign'])
                for _, r in sign_df.iterrows()}

    cells = survivors[['sensor', 'channel', 'algorithm']].to_dict('records')
    return cells, sign_map


# ─────────── F1 평가 (sign 고정) ───────────
def evaluate_perturbed(sig, t, fs, algo_fn, gt_HS, gt_TO, sign):
    """sign 고정으로 HS/TO F1 계산"""
    try:
        HS_i, TO_i = algo_fn(sig * sign, fs)
    except Exception:
        return 0.0, 0.0
    HS_t = t[HS_i] if len(HS_i) else np.array([])
    TO_t = t[TO_i] if len(TO_i) else np.array([])
    return (match_F1(HS_t, gt_HS, TOL),
            match_F1(TO_t, gt_TO, TOL))


# ─────────── trial 단위 평가 ───────────
def evaluate_trial_phase2(xlsx, sheet, gt_csv, cells, sign_map):
    """한 trial에서 모든 셀 × 모든 교란 조건 평가."""
    sensors_data, _ = load_imu_sheet(xlsx, sheet)
    gt = pd.read_csv(gt_csv)
    gt = gt[gt['in_window'] == True]
    gt_HS = gt[gt['event'] == 'HS']['imu_time'].values
    gt_TO = gt[gt['event'] == 'TO']['imu_time'].values

    sensor_id_map = {v: k for k, v in SENSORS.items()}

    # 미리 센서별 add_euler_channels 한 번만 적용 (캐시)
    sensor_cache = {}
    for sensor_name in set(c['sensor'] for c in cells):
        sid = sensor_id_map.get(sensor_name)
        if sid is None or sid not in sensors_data:
            continue
        df = add_euler_channels(sensors_data[sid])
        sensor_cache[sensor_name] = df

    rows = []
    for cell in cells:
        sensor = cell['sensor']
        channel = cell['channel']
        algo = cell['algorithm']
        if sensor not in sensor_cache:
            continue
        df = sensor_cache[sensor]
        if channel not in df.columns:
            continue
        t = df['t'].values
        fs = 1 / np.median(np.diff(t))
        sig = df[channel].values.astype(float)
        sig = (pd.Series(sig).interpolate(limit_direction='both')
               .fillna(0).values)
        algo_fn = PHASE2_ALGORITHMS[algo]
        sign = sign_map.get((sensor, channel, algo), 1)

        # baseline
        f_hs, f_to = evaluate_perturbed(
            sig, t, fs, algo_fn, gt_HS, gt_TO, sign)
        rows.append({
            'sensor': sensor, 'channel': channel, 'algorithm': algo,
            'pert_type': 'baseline', 'pert_level': 0,
            'F1_HS': f_hs, 'F1_TO': f_to, 'sign': sign,
        })

        # 5 교란 × 5 레벨
        for pert_type, info in PERTURBATIONS.items():
            for level in info['levels']:
                sig_p, t_p, fs_p = apply_perturbation(
                    pert_type, sig, t, fs, level,
                    df=df, channel=channel)
                f_hs, f_to = evaluate_perturbed(
                    sig_p, t_p, fs_p, algo_fn, gt_HS, gt_TO, sign)
                rows.append({
                    'sensor': sensor, 'channel': channel, 'algorithm': algo,
                    'pert_type': pert_type, 'pert_level': level,
                    'F1_HS': f_hs, 'F1_TO': f_to, 'sign': sign,
                })
    return rows


# ─────────── 격자 실행 ───────────
def run_grid_phase2(trials, cells, sign_map):
    all_rows = []
    t_start = time.time()
    for i, tr in enumerate(trials, 1):
        elapsed = time.time() - t_start
        eta = elapsed / i * (len(trials) - i) if i > 0 else 0
        print(f'  [{i:>2}/{len(trials)}] {tr["trial_id"]:<25} '
              f'(경과 {elapsed:.0f}s, 남은 ~{eta:.0f}s)', flush=True)
        try:
            rows = evaluate_trial_phase2(
                tr['xlsx'], tr['sheet'], tr['csv'], cells, sign_map)
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
    print(f'\n총 {time.time()-t_start:.0f}초, {len(all_rows)} 행')
    return pd.DataFrame(all_rows)


# ─────────── 복합 교란 trial 단위 평가 ───────────
def evaluate_trial_compound(xlsx, sheet, gt_csv, cells, sign_map):
    """한 trial: 셀 × 4 복합 교란 × 3 레벨"""
    sensors_data, _ = load_imu_sheet(xlsx, sheet)
    gt = pd.read_csv(gt_csv)
    gt = gt[gt['in_window'] == True]
    gt_HS = gt[gt['event'] == 'HS']['imu_time'].values
    gt_TO = gt[gt['event'] == 'TO']['imu_time'].values

    sensor_id_map = {v: k for k, v in SENSORS.items()}
    sensor_cache = {}
    for sensor_name in set(c['sensor'] for c in cells):
        sid = sensor_id_map.get(sensor_name)
        if sid is None or sid not in sensors_data:
            continue
        df = add_euler_channels(sensors_data[sid])
        sensor_cache[sensor_name] = df

    rows = []
    for cell in cells:
        sensor = cell['sensor']
        channel = cell['channel']
        algo = cell['algorithm']
        if sensor not in sensor_cache:
            continue
        df = sensor_cache[sensor]
        if channel not in df.columns:
            continue
        t = df['t'].values
        fs = 1 / np.median(np.diff(t))
        sig = (df[channel].values.astype(float))
        sig = (pd.Series(sig).interpolate(limit_direction='both')
               .fillna(0).values)
        algo_fn = PHASE2_ALGORITHMS[algo]
        sign = sign_map.get((sensor, channel, algo), 1)

        for pert_type, info in COMPOUND_PERTURBATIONS.items():
            for level_idx, level_tuple in enumerate(info['levels']):
                sig_p, t_p, fs_p = apply_compound(
                    pert_type, sig, t, fs, level_tuple,
                    df=df, channel=channel)
                f_hs, f_to = evaluate_perturbed(
                    sig_p, t_p, fs_p, algo_fn, gt_HS, gt_TO, sign)
                rows.append({
                    'sensor': sensor, 'channel': channel, 'algorithm': algo,
                    'pert_type': pert_type,
                    'pert_level': level_idx,  # 0=Light, 1=Medium, 2=Heavy
                    'pert_level_label': LEVEL_LABELS[level_idx],
                    'pert_level_tuple': str(level_tuple),
                    'F1_HS': f_hs, 'F1_TO': f_to, 'sign': sign,
                })
    return rows


def run_compound_grid(trials, cells, sign_map):
    all_rows = []
    t_start = time.time()
    for i, tr in enumerate(trials, 1):
        elapsed = time.time() - t_start
        eta = elapsed / i * (len(trials) - i) if i > 0 else 0
        print(f'  [{i:>2}/{len(trials)}] {tr["trial_id"]:<25} '
              f'(경과 {elapsed:.0f}s, 남은 ~{eta:.0f}s)', flush=True)
        try:
            rows = evaluate_trial_compound(
                tr['xlsx'], tr['sheet'], tr['csv'], cells, sign_map)
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
    print(f'\n총 {time.time()-t_start:.0f}초, {len(all_rows)} 행')
    return pd.DataFrame(all_rows)


# ─────────── DR (Degradation Ratio) 계산 ───────────
def compute_dr(df):
    """baseline F1 대비 비율. DR = min(F1_pert / F1_base, 1.0).

    baseline F1<=0.1이면 DR=NaN (정의 불가, 저성능 셀 제외).
    DR은 [0,1]로 클리핑: 교란이 우연히 성능을 올린 경우(DR>1)는
    1.0으로 처리해 "성능 저하 정도"만 단조 해석 가능하게 함.
    (논문 2.4절 정의와 일치)
    """
    # baseline F1 추출 (피험자×조합별)
    baseline = (df[df['pert_type'] == 'baseline']
                .groupby(['subject', 'sensor', 'channel', 'algorithm'])
                [['F1_HS', 'F1_TO']].mean()
                .rename(columns={'F1_HS': 'F1_HS_base',
                                 'F1_TO': 'F1_TO_base'})
                .reset_index())
    df_merge = df.merge(baseline,
                         on=['subject', 'sensor', 'channel', 'algorithm'],
                         how='left')
    df_merge['DR_HS'] = np.where(
        df_merge['F1_HS_base'] > 0.1,
        np.minimum(df_merge['F1_HS'] / df_merge['F1_HS_base'], 1.0),
        np.nan)
    df_merge['DR_TO'] = np.where(
        df_merge['F1_TO_base'] > 0.1,
        np.minimum(df_merge['F1_TO'] / df_merge['F1_TO_base'], 1.0),
        np.nan)
    return df_merge


# ─────────── 요약 ───────────
def run_summaries_phase2(df, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 알고리즘 × 교란 × 레벨 평균 DR (핵심)
    by_algo = (df[df['pert_type'] != 'baseline']
               .groupby(['algorithm', 'pert_type', 'pert_level'])
               [['DR_HS', 'DR_TO']]
               .agg(['mean', 'std']).reset_index())
    by_algo.columns = ['_'.join(c).strip('_') for c in by_algo.columns]
    by_algo.to_csv(out_dir / 'phase2_by_algorithm.csv', index=False)

    # 2) 셀별 (각 sensor×channel×algorithm마다 교란 평균 DR)
    by_cell = (df[df['pert_type'] != 'baseline']
               .groupby(['sensor', 'channel', 'algorithm', 'pert_type'])
               [['DR_HS', 'DR_TO']]
               .mean().reset_index())
    by_cell.to_csv(out_dir / 'phase2_by_cell.csv', index=False)

    # 3) AUC (DR 곡선 아래 면적) — 알고리즘 종합 강건성
    auc_rows = []
    for (algo, pert), sub in df[df['pert_type'] != 'baseline'].groupby(
            ['algorithm', 'pert_type']):
        levels = sorted(sub['pert_level'].unique())
        # 정규화된 레벨 (0~1)
        norm_levels = np.linspace(0, 1, len(levels))
        for event in ('HS', 'TO'):
            dr_col = f'DR_{event}'
            means = [sub[sub['pert_level'] == lv][dr_col].mean()
                     for lv in levels]
            means = np.array(means)
            if np.all(np.isnan(means)):
                auc = np.nan
            else:
                auc = _trapz(np.nan_to_num(means), norm_levels)
            auc_rows.append({
                'algorithm': algo, 'pert_type': pert, 'event': event,
                'AUC_DR': auc,
            })
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(out_dir / 'phase2_auc.csv', index=False)

    return by_algo, by_cell, auc_df


# ─────────── 메인 ───────────
def main():
    print(f'GT_DIR  = {GT_DIR}')
    print(f'IMU_DIR = {IMU_DIR}')
    print(f'RESULTS = {RESULTS_DIR}\n')

    # 1단계 결과 확인
    cell_csv = RESULTS_DIR / 'cell_results.csv'
    if not cell_csv.exists():
        raise FileNotFoundError(
            f'{cell_csv} 없음. 1단계(pipeline.py)를 먼저 실행하세요.')

    # 캐시: cell_results_phase2.csv가 이미 있으면 격자 건너뛰기
    cache_csv = RESULTS_DIR / 'cell_results_phase2.csv'
    if cache_csv.exists():
        print(f'단일 격자 사용: {cache_csv}')
        print('(격자 다시 돌리려면 이 csv를 삭제)\n')
        df_single = pd.read_csv(cache_csv)
    else:
        # 살아남은 셀 추출
        cells, sign_map = load_surviving_cells(F1_THRESHOLD)
        print(f'1단계 살아남은 셀 (F1 > {F1_THRESHOLD}): {len(cells)}개')
        if not cells:
            raise ValueError(f'F1 > {F1_THRESHOLD} 셀이 없음.')

        algo_counts = pd.DataFrame(cells)['algorithm'].value_counts()
        print('  알고리즘별:')
        for algo, n in algo_counts.items():
            print(f'    {algo:<12} {n}개')

        n_conditions = 1 + sum(len(v['levels'])
                                for v in PERTURBATIONS.values())
        print(f'\n단일 교란: 1 baseline + 5 × 5 = {n_conditions}개')
        print(f'예상 평가: {len(cells)} × {n_conditions} × 32 = '
              f'{len(cells) * n_conditions * 32:,}개\n')

        trials = build_trial_list()
        print(f'trial: {len(trials)}개\n')

        print('━━ 단일 격자 실행 ━━')
        df_single = run_grid_phase2(trials, cells, sign_map)
        df_single.to_csv(cache_csv, index=False)
        print(f'\n[저장] {cache_csv} ({len(df_single)}행)')

    # ═════════════ 복합 교란 격자 ═════════════
    compound_csv = RESULTS_DIR / 'cell_results_phase2_compound.csv'
    if compound_csv.exists():
        print(f'\n복합 격자 사용: {compound_csv}')
        df_compound = pd.read_csv(compound_csv)
    else:
        cells, sign_map = load_surviving_cells(F1_THRESHOLD)
        trials = build_trial_list()
        n_c = sum(len(v['levels']) for v in COMPOUND_PERTURBATIONS.values())
        print(f'\n복합 교란: 4종 × 3 레벨 = {n_c}개')
        print(f'예상 평가: {len(cells)} × {n_c} × 32 = '
              f'{len(cells) * n_c * 32:,}개\n')
        print('━━ 복합 격자 실행 ━━')
        df_compound = run_compound_grid(trials, cells, sign_map)
        df_compound.to_csv(compound_csv, index=False)
        print(f'\n[저장] {compound_csv} ({len(df_compound)}행)')

    # ═════════════ 통합 (단일 + 복합) ═════════════
    common_cols = ['subject', 'gender', 'speed', 'rep', 'trial_id',
                    'sensor', 'channel', 'algorithm',
                    'pert_type', 'pert_level',
                    'F1_HS', 'F1_TO', 'sign']
    df = pd.concat([
        df_single[common_cols],
        df_compound[common_cols]
    ], ignore_index=True)

    # DR 계산
    print('\n── DR 계산 ──')
    df_dr = compute_dr(df)
    df_dr.to_csv(RESULTS_DIR / 'cell_results_phase2_dr.csv', index=False)

    # 요약
    print('\n── 요약 CSV ──')
    by_algo, by_cell, auc_df = run_summaries_phase2(df_dr, RESULTS_DIR)
    print('  phase2_by_algorithm.csv (알고리즘 × 교란 × 레벨)')
    print('  phase2_by_cell.csv      (셀별 교란 평균)')
    print('  phase2_auc.csv          (AUC 종합 강건성)')

    # AUC 요약 출력
    print('\n━━━━━ AUC (강건성 종합) — HS ━━━━━')
    pivot_hs = auc_df[auc_df['event'] == 'HS'].pivot(
        index='pert_type', columns='algorithm', values='AUC_DR')
    print(pivot_hs.round(3).to_string())
    print('\n━━━━━ AUC (강건성 종합) — TO ━━━━━')
    pivot_to = auc_df[auc_df['event'] == 'TO'].pivot(
        index='pert_type', columns='algorithm', values='AUC_DR')
    print(pivot_to.round(3).to_string())

    return df_dr


if __name__ == '__main__':
    main()