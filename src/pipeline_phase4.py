"""
src/pipeline_phase4.py — 4단계: 딥러닝 교란 강건성 평가
========================================================
S-A 시나리오: clean 학습 → 교란 테스트
  전통 알고리즘(TB/BPF/TM)이 튜닝 없이 교란을 맞았으므로,
  딥러닝도 clean으로만 학습해야 공정한 비교가 된다.

핵심: **재학습 0회.**
  fold별 모델을 한 번 학습해 저장해두면 교란 26조건은 전부 추론만 하면 된다.
  2 model x 2 position x 4 fold = 8회 학습.

출력:
  results/models/{model}_{position}_{testsubj}.pt   학습된 모델
  results/dl_phase2_raw.csv                          trial 단위 F1
  results/dl_phase2_dr.csv                           셀 단위 DR / DR-AUC

실행:
  python -m src.pipeline_phase4                 # 전체
  python -m src.pipeline_phase4 --quick         # 빠른 점검(에폭 축소)
"""
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SUBJECT_GENDER, SENSORS, CHANNELS, TOL_MAIN_MS, RESULTS_DIR
from src.pipeline import add_euler_channels, match_F1
from src.pipeline_phase3 import (
    GaitCNN, GaitLSTM, train_fold, cluster_events,
    WINDOW_MS, STRIDE_SAMPLES, LABEL_TOL_MS, INPUT_CHANNELS,
)
from src.perturbations_mc import (
    PERTURBATIONS_MC, apply_perturbation_mc, normalize, CH_ORDER,
)
from imu_loader import load_imu_sheet

TOL_MAIN = TOL_MAIN_MS / 1000

# 위치별 기준 단일 채널 (1단계 best = 전통 알고리즘 비교 대상)
REF_CHANNEL = {'shank': 'Gyro_Z', 'foot': 'Gyro_Y'}
POSITIONS = ['shank', 'foot']
NORM_MODES = ['adaptive', 'frozen']
SCOPES = ['all9', 'gyro3', 'ref1']
MODEL_DIR = RESULTS_DIR / 'models'


# ═══════════ 1. raw trial 캐시 ═══════════
CACHE_DIR = RESULTS_DIR / 'raw_cache'


def build_raw_trials(trials, position, use_cache=True):
    """trial별 raw 9채널 신호(정규화 이전) + GT + clean 통계를 보관.

    교란은 raw에 적용한 뒤 정규화해야 하므로, phase3처럼 정규화된
    윈도우를 미리 만들어두면 안 된다. 여기서는 '신호 전체'를 들고 있다가
    교란 후 윈도우를 잘라낸다.

    xlsx 읽기가 trial당 ~11초로 느리므로 npz 디스크 캐시를 둔다.
    """
    sid_map = {v: k for k, v in SENSORS.items()}
    sid = sid_map[position]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for tr in trials:
        cf = CACHE_DIR / f'{tr["trial_id"]}_{position}.npz'
        if use_cache and cf.exists():
            z = np.load(cf, allow_pickle=True)
            out.append({
                'trial_id': tr['trial_id'], 'subject': tr['subject'],
                'gender': tr['gender'], 'speed': tr['speed'], 'rep': tr['rep'],
                't': z['t'], 'sig9': z['sig9'],
                'gt_HS': z['gt_HS'], 'gt_TO': z['gt_TO'],
                'clean_stats': (z['mu'], z['sd']),
            })
            continue
        try:
            sensors_data, _ = load_imu_sheet(tr['xlsx'], tr['sheet'])
        except Exception as e:
            print(f'    [skip] {tr["sheet"]}: {e}')
            continue
        if sid not in sensors_data:
            continue
        df = add_euler_channels(sensors_data[sid])
        chs = [c for c in CH_ORDER if c in df.columns]
        if len(chs) < INPUT_CHANNELS:
            continue
        t = df['t'].values.astype(float)
        sig9 = df[chs].values.astype(float)
        sig9 = pd.DataFrame(sig9).interpolate(
            limit_direction='both').fillna(0).values

        gt = pd.read_csv(tr['csv'])
        gt = gt[gt['in_window'] == True]
        gt_HS = gt[gt['event'] == 'HS']['imu_time'].values
        gt_TO = gt[gt['event'] == 'TO']['imu_time'].values
        if len(gt_HS) == 0 and len(gt_TO) == 0:
            continue

        clean_stats = (sig9.mean(axis=0, keepdims=True),
                       sig9.std(axis=0, keepdims=True) + 1e-6)
        np.savez_compressed(cf, t=t, sig9=sig9, gt_HS=gt_HS, gt_TO=gt_TO,
                            mu=clean_stats[0], sd=clean_stats[1])
        out.append({
            'trial_id': tr['trial_id'], 'subject': tr['subject'],
            'gender': tr['gender'], 'speed': tr['speed'], 'rep': tr['rep'],
            't': t, 'sig9': sig9, 'gt_HS': gt_HS, 'gt_TO': gt_TO,
            'clean_stats': clean_stats,
        })
    return out


# ═══════════ 2. 윈도우 절단 ═══════════
def windows_from_signal(sig9_norm, t, gt_HS, gt_TO, with_labels=True):
    """정규화된 9채널 신호 → (N,9,W) 윈도우 + 라벨 + 중심시각"""
    fs = 1.0 / np.median(np.diff(t))
    W = int(WINDOW_MS / 1000 * fs)
    if W % 2 == 0:
        W += 1
    half = W // 2
    t_min = min(gt_HS.min() if len(gt_HS) else np.inf,
                gt_TO.min() if len(gt_TO) else np.inf) - 0.3
    t_max = max(gt_HS.max() if len(gt_HS) else -np.inf,
                gt_TO.max() if len(gt_TO) else -np.inf) + 0.3
    idx_in = np.where((t >= t_min) & (t <= t_max))[0]
    if len(idx_in) < W * 2:
        return None, None, None
    starts = np.arange(idx_in[0], idx_in[-1] - W, STRIDE_SAMPLES)
    X = np.stack([sig9_norm[s:s + W].T for s in starts]).astype(np.float32)
    tc = t[starts + half]
    y = None
    if with_labels:
        tol_s = LABEL_TOL_MS / 1000
        y = np.zeros(len(starts), dtype=np.int64)
        for i, c in enumerate(tc):
            d_hs = np.min(np.abs(gt_HS - c)) if len(gt_HS) else np.inf
            d_to = np.min(np.abs(gt_TO - c)) if len(gt_TO) else np.inf
            if d_hs <= tol_s and d_hs <= d_to:
                y[i] = 1
            elif d_to <= tol_s:
                y[i] = 2
    return X, y, tc


# ═══════════ 3. 학습 (clean, fold별 1회) ═══════════
def get_or_train_model(model_name, model_class, position, test_subj,
                       raw_trials, n_epochs=None, device='cpu'):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = MODEL_DIR / f'{model_name}_{position}_{test_subj}.pt'
    model = model_class(INPUT_CHANNELS, 3).to(device)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        print(f'    [캐시] {ckpt.name}')
        return model

    Xs, ys = [], []
    for rt in raw_trials:
        if rt['subject'] == test_subj:
            continue
        sn = normalize(rt['sig9'], 'adaptive')  # 학습은 phase3와 동일 조건
        X, y, _ = windows_from_signal(sn, rt['t'], rt['gt_HS'], rt['gt_TO'])
        if X is None:
            continue
        Xs.append(X)
        ys.append(y)
    X_train = np.concatenate(Xs)
    y_train = np.concatenate(ys)
    print(f'    학습 윈도우 {len(X_train):,} '
          f'(None={np.sum(y_train==0):,}, HS={np.sum(y_train==1):,}, '
          f'TO={np.sum(y_train==2):,})')

    import src.pipeline_phase3 as p3
    orig_epochs = p3.N_EPOCHS
    if n_epochs:
        p3.N_EPOCHS = n_epochs
    # train_fold는 (preds)만 반환하도록 되어 있으므로 더미 X_test를 주고
    # 내부 model을 얻기 위해 여기서 직접 학습 루프를 재현하지 않고,
    # phase3.train_fold를 패치 없이 쓰기 위해 model을 별도로 학습한다.
    model = _train_loop(X_train, y_train, model_class, device,
                        n_epochs=p3.N_EPOCHS)
    p3.N_EPOCHS = orig_epochs
    torch.save(model.state_dict(), ckpt)
    print(f'    [저장] {ckpt.name}')
    return model


def _train_loop(X_train, y_train, model_class, device, n_epochs):
    """phase3.train_fold와 동일한 설정의 학습 루프 (모델을 반환)"""
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from src.pipeline_phase3 import GaitDataset, BATCH_SIZE, LR

    ds = GaitDataset(X_train, y_train)
    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    w = 1.0 / (counts + 1)
    w = w / w.sum() * 3
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model = model_class(INPUT_CHANNELS, 3).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss(weight=torch.from_numpy(w).to(device))
    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        tot = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item()
        if (ep + 1) % 5 == 0:
            print(f'      epoch {ep+1:>2}/{n_epochs} loss {tot/len(loader):.4f} '
                  f'({time.time()-t0:.0f}s)')
    model.eval()
    return model


# ═══════════ 4. 추론 + F1 ═══════════
@torch.no_grad()
def infer_f1(model, X, tc, gt_HS, gt_TO, device='cpu'):
    preds = []
    Xt = torch.from_numpy(X)
    for i in range(0, len(Xt), 4096):
        preds.append(model(Xt[i:i + 4096].to(device)).argmax(1).cpu().numpy())
    p = np.concatenate(preds)
    HS_t = cluster_events(tc[p == 1], min_gap=0.3)
    TO_t = cluster_events(tc[p == 2], min_gap=0.3)
    return (match_F1(HS_t, gt_HS, TOL_MAIN), match_F1(TO_t, gt_TO, TOL_MAIN),
            len(HS_t), len(TO_t))


# ═══════════ 5. 메인 격자 ═══════════
def run_phase4(trials, models, positions=POSITIONS, norm_modes=NORM_MODES,
               scopes=SCOPES, n_epochs=None, device='cpu'):
    rows = []
    t_start = time.time()
    for position in positions:
        ref_ch = REF_CHANNEL[position]
        print(f'\n{"="*60}\n  POSITION = {position}  (ref channel = {ref_ch})\n{"="*60}')
        print('  raw trial 로딩...')
        raw_trials = build_raw_trials(trials, position)
        print(f'  {len(raw_trials)} trial 확보')
        subjects = sorted({rt['subject'] for rt in raw_trials})

        for model_name, model_class in models:
            for test_subj in subjects:
                print(f'\n  ── {model_name} | fold test={test_subj} ──')
                model = get_or_train_model(model_name, model_class, position,
                                           test_subj, raw_trials,
                                           n_epochs=n_epochs, device=device)
                test_trials = [rt for rt in raw_trials
                               if rt['subject'] == test_subj]

                # 조건 목록: baseline + 5교란 x 5레벨
                conds = [('baseline', 0)]
                for pt, info in PERTURBATIONS_MC.items():
                    conds += [(pt, lv) for lv in info['levels']]

                for rt in test_trials:
                    for scope in scopes:
                        for norm_mode in norm_modes:
                            for pt, lv in conds:
                                # baseline은 scope와 무관 → all9에서만 1회
                                if pt == 'baseline' and scope != 'all9':
                                    continue
                                sig_p = apply_perturbation_mc(
                                    pt, rt['sig9'], rt['t'], lv,
                                    ref_channel=ref_ch, scope=scope)
                                sn = normalize(sig_p, norm_mode,
                                               rt['clean_stats'])
                                X, _, tc = windows_from_signal(
                                    sn, rt['t'], rt['gt_HS'], rt['gt_TO'],
                                    with_labels=False)
                                if X is None:
                                    continue
                                f_hs, f_to, n_hs, n_to = infer_f1(
                                    model, X, tc, rt['gt_HS'], rt['gt_TO'],
                                    device)
                                rows.append({
                                    'model': model_name, 'position': position,
                                    'subject': rt['subject'],
                                    'gender': rt['gender'],
                                    'speed': rt['speed'], 'rep': rt['rep'],
                                    'trial_id': rt['trial_id'],
                                    'scope': scope, 'norm': norm_mode,
                                    'pert_type': pt, 'pert_level': lv,
                                    'F1_HS': f_hs, 'F1_TO': f_to,
                                    'n_det_HS': n_hs, 'n_det_TO': n_to,
                                })
                    print(f'    {rt["trial_id"]:<20} 완료 '
                          f'(누적 {len(rows)}행, {time.time()-t_start:.0f}s)',
                          flush=True)
    return pd.DataFrame(rows)


# ═══════════ 6. DR 계산 ═══════════
def compute_dr(df):
    """DR = F1_baseline / F1_perturbed  (기존 논문 정의와 동일)

    baseline은 scope='all9', 동일 norm 모드 기준.
    DR-AUC는 레벨 인덱스(1..5)에 대한 사다리꼴 적분 / 4 (평균 DR).
    """
    base = (df[df['pert_type'] == 'baseline']
            .groupby(['model', 'position', 'subject', 'trial_id', 'norm'])
            [['F1_HS', 'F1_TO']].mean()
            .rename(columns={'F1_HS': 'F1_HS_base', 'F1_TO': 'F1_TO_base'})
            .reset_index())
    d = df[df['pert_type'] != 'baseline'].merge(
        base, on=['model', 'position', 'subject', 'trial_id', 'norm'],
        how='left')
    eps = 1e-6
    d['DR_HS'] = d['F1_HS_base'] / (d['F1_HS'] + eps)
    d['DR_TO'] = d['F1_TO_base'] / (d['F1_TO'] + eps)
    return d


def dr_auc_table(d):
    """셀(model x position x scope x norm x pert_type) 단위 평균 DR 곡선 + AUC"""
    lvl_rank = (d.groupby(['pert_type'])['pert_level']
                .rank(method='dense').astype(int))
    d = d.assign(level_rank=lvl_rank)
    g = (d.groupby(['model', 'position', 'scope', 'norm',
                    'pert_type', 'level_rank', 'pert_level'])
         [['DR_HS', 'DR_TO', 'F1_HS', 'F1_TO']].mean().reset_index())
    aucs = []
    for key, sub in g.groupby(['model', 'position', 'scope', 'norm',
                               'pert_type']):
        sub = sub.sort_values('level_rank')
        x = sub['level_rank'].values
        rec = dict(zip(['model', 'position', 'scope', 'norm', 'pert_type'], key))
        for ev in ['HS', 'TO']:
            y = sub[f'DR_{ev}'].values
            rec[f'DRAUC_{ev}'] = (np.trapezoid(y, x) / (x[-1] - x[0])
                                  if len(x) > 1 else np.nan)
            rec[f'DRmax_{ev}'] = y.max()
        aucs.append(rec)
    return g, pd.DataFrame(aucs)


# ═══════════ main ═══════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='에폭 축소 점검용')
    ap.add_argument('--positions', nargs='+', default=POSITIONS)
    ap.add_argument('--models', nargs='+', default=['CNN', 'LSTM'])
    ap.add_argument('--scopes', nargs='+', default=SCOPES)
    ap.add_argument('--local', action='store_true',
                    help='컨테이너용 trial 빌더 사용')
    args = ap.parse_args()

    if args.local:
        from src.trials_adapter import get_trials
        trials = get_trials(verbose=False)
    else:
        from src.pipeline import build_trial_list
        trials = build_trial_list()
    print(f'trial {len(trials)}개')

    model_map = {'CNN': GaitCNN, 'LSTM': GaitLSTM}
    models = [(m, model_map[m]) for m in args.models]

    df = run_phase4(trials, models, positions=args.positions,
                    scopes=args.scopes,
                    n_epochs=3 if args.quick else None)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / 'dl_phase2_raw.csv', index=False)
    print(f'\n[저장] dl_phase2_raw.csv ({len(df)}행)')

    d = compute_dr(df)
    curve, auc = dr_auc_table(d)
    d.to_csv(RESULTS_DIR / 'dl_phase2_dr.csv', index=False)
    curve.to_csv(RESULTS_DIR / 'dl_phase2_curve.csv', index=False)
    auc.to_csv(RESULTS_DIR / 'dl_phase2_auc.csv', index=False)
    print('[저장] dl_phase2_dr.csv / dl_phase2_curve.csv / dl_phase2_auc.csv')

    print('\n━━━ DR-AUC 요약 (scope=all9) ━━━')
    print(auc[auc['scope'] == 'all9']
          .pivot_table(index=['position', 'model', 'norm'],
                       columns='pert_type',
                       values='DRAUC_HS').round(3).to_string())
    return df


if __name__ == '__main__':
    main()