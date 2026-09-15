"""
src/pipeline_phase3.py — 3단계: 1D CNN (multi-channel)
========================================================
shank + foot 두 위치, 각 9채널 (Gyro 3 + Acc 3 + Euler 3)
3-class 분류 (HS, TO, None), Leave-One-Subject-Out 4 fold

1단계 best 단일 채널과 직접 비교:
- shank Gyro_Z TB_adaptive vs shank 9채널 CNN
- foot  Gyro_Y TB_adaptive vs foot  9채널 CNN

결과: results/cnn_results.csv
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    SUBJECT_GENDER, SENSORS, CHANNELS,
    TOL_MAIN_MS, RESULTS_DIR,
)
from src.pipeline import add_euler_channels, match_F1, build_trial_list
from imu_loader import load_imu_sheet


# ─────────── 설정 ───────────
CNN_POSITIONS = ['shank', 'foot']
INPUT_CHANNELS = 9
WINDOW_MS = 200
STRIDE_SAMPLES = 8     # 윈도우 stride (작을수록 데이터 많아짐)
N_EPOCHS = 20
BATCH_SIZE = 512
LR = 1e-3
LABEL_TOL_MS = 60      # 라벨 생성 시 (윈도우 중심 ±tol_s 내 이벤트가 있으면 그 클래스)
TOL_MAIN = TOL_MAIN_MS / 1000  # F1 평가 ±150ms

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────── 모델 ───────────
class GaitCNN(nn.Module):
    """경량 1D CNN — 약 30K parameters"""
    def __init__(self, in_channels=9, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = self.gap(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class GaitLSTM(nn.Module):
    """경량 1-layer LSTM — 약 20K parameters"""
    def __init__(self, in_channels=9, n_classes=3, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden,
                            num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # x: (batch, 9, window) → (batch, window, 9)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # 마지막 timestep
        last = self.dropout(last)
        return self.fc(last)


# ─────────── 데이터셋 빌드 ───────────
def build_dataset(trials, position):
    """모든 trial에서 윈도우 추출.

    Returns
    -------
    X : (N, 9, window_samples) float32
    y : (N,) int64 — 0=None, 1=HS, 2=TO
    subj : (N,) str
    trial_id : (N,) str
    t_center : (N,) float — 윈도우 중심 시간 (IMU time)
    gt_dict : {trial_id: (gt_HS, gt_TO)}
    """
    sensor_id_map = {v: k for k, v in SENSORS.items()}
    sensor_id = sensor_id_map[position]

    X_all, y_all = [], []
    subj_all, trial_all, t_center_all = [], [], []
    gt_dict = {}

    for tr in trials:
        try:
            sensors_data, _ = load_imu_sheet(tr['xlsx'], tr['sheet'])
        except Exception as e:
            print(f'    [skip] {tr["sheet"]}: {e}')
            continue
        if sensor_id not in sensors_data:
            continue
        df = add_euler_channels(sensors_data[sensor_id])
        t = df['t'].values
        fs = 1 / np.median(np.diff(t))
        window_samples = int(WINDOW_MS / 1000 * fs)
        if window_samples % 2 == 0:
            window_samples += 1
        half = window_samples // 2

        chs = [c for c in CHANNELS if c in df.columns]
        if len(chs) < INPUT_CHANNELS:
            continue
        sig = df[chs[:INPUT_CHANNELS]].values.astype(np.float32)
        # z-score 정규화 (per channel)
        mu = sig.mean(axis=0, keepdims=True)
        sd = sig.std(axis=0, keepdims=True) + 1e-6
        sig = (sig - mu) / sd
        sig = np.nan_to_num(sig)

        gt = pd.read_csv(tr['csv'])
        gt = gt[gt['in_window'] == True]
        gt_HS = gt[gt['event'] == 'HS']['imu_time'].values
        gt_TO = gt[gt['event'] == 'TO']['imu_time'].values
        gt_dict[tr['trial_id']] = (gt_HS, gt_TO)

        # in_window 범위
        t_min = gt['imu_time'].min() - 0.3
        t_max = gt['imu_time'].max() + 0.3
        idx_in = np.where((t >= t_min) & (t <= t_max))[0]
        if len(idx_in) < window_samples * 2:
            continue

        starts = np.arange(idx_in[0], idx_in[-1] - window_samples,
                            STRIDE_SAMPLES)
        tol_s = LABEL_TOL_MS / 1000
        for s in starts:
            window = sig[s:s + window_samples]  # (W, 9)
            tc = t[s + half]
            # 라벨: 윈도우 중심에 가장 가까운 이벤트
            d_hs = np.min(np.abs(gt_HS - tc)) if len(gt_HS) else np.inf
            d_to = np.min(np.abs(gt_TO - tc)) if len(gt_TO) else np.inf
            if d_hs <= tol_s and d_hs <= d_to:
                lbl = 1
            elif d_to <= tol_s:
                lbl = 2
            else:
                lbl = 0
            X_all.append(window.T)  # (9, W)
            y_all.append(lbl)
            subj_all.append(tr['subject'])
            trial_all.append(tr['trial_id'])
            t_center_all.append(tc)

    X = np.stack(X_all).astype(np.float32)
    y = np.array(y_all, dtype=np.int64)
    subj = np.array(subj_all)
    trial_id = np.array(trial_all)
    t_center = np.array(t_center_all, dtype=np.float32)
    return X, y, subj, trial_id, t_center, gt_dict


# ─────────── 학습 ───────────
class GaitDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_fold(X_train, y_train, X_test, model_class=GaitCNN, device='cpu',
                verbose=True):
    train_ds = GaitDataset(X_train, y_train)
    # Class weight (imbalance — None이 대부분)
    counts = np.bincount(y_train, minlength=3).astype(np.float32)
    weights = 1.0 / (counts + 1)
    weights = weights / weights.sum() * 3
    weights_t = torch.from_numpy(weights).to(device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)

    model = model_class(INPUT_CHANNELS, 3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=weights_t)

    t_start = time.time()
    for epoch in range(N_EPOCHS):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose and (epoch + 1) % 5 == 0:
            print(f'      epoch {epoch+1:>2}/{N_EPOCHS}, '
                  f'loss {total_loss/len(train_loader):.4f}, '
                  f'elapsed {time.time()-t_start:.0f}s')

    # 평가
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X_test).to(device)
        # batch별 처리 (메모리)
        all_preds = []
        for i in range(0, len(Xt), 2048):
            logits = model(Xt[i:i+2048])
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
        preds = np.concatenate(all_preds)
    return preds


# ─────────── 윈도우 예측 → 이벤트 시간 ───────────
def cluster_events(times, min_gap=0.3):
    """인접 예측 클러스터링 → 평균 시간"""
    if len(times) == 0:
        return np.array([])
    times = np.sort(times)
    clusters = [[times[0]]]
    for t in times[1:]:
        if t - clusters[-1][-1] < min_gap:
            clusters[-1].append(t)
        else:
            clusters.append([t])
    return np.array([np.mean(c) for c in clusters])


def evaluate_predictions(preds, trial_ids_test, t_centers_test,
                          gt_dict, tol=TOL_MAIN):
    """trial별 F1 평가"""
    rows = []
    for tid in np.unique(trial_ids_test):
        mask = trial_ids_test == tid
        p = preds[mask]
        tc = t_centers_test[mask]
        gt_HS, gt_TO = gt_dict[tid]

        HS_t = cluster_events(tc[p == 1], min_gap=0.3)
        TO_t = cluster_events(tc[p == 2], min_gap=0.3)
        f1_hs = match_F1(HS_t, gt_HS, tol)
        f1_to = match_F1(TO_t, gt_TO, tol)
        rows.append({
            'trial_id': tid,
            'F1_HS_150': f1_hs,
            'F1_TO_150': f1_to,
            'n_det_HS': len(HS_t),
            'n_det_TO': len(TO_t),
            'n_gt_HS': len(gt_HS),
            'n_gt_TO': len(gt_TO),
        })
    return rows


# ─────────── 한 위치 LOSO ───────────
def run_loso(position, trials, model_class=GaitCNN, model_name='CNN',
              cached_data=None):
    """cached_data: (X, y, subj, trial_id, t_center, gt_dict) — dataset 재사용용"""
    print(f'\n━━━ {model_name} | Position: {position} ━━━')
    t_pos_start = time.time()

    if cached_data is None:
        print(f'  Building dataset...')
        X, y, subj, trial_id, t_center, gt_dict = build_dataset(trials, position)
    else:
        X, y, subj, trial_id, t_center, gt_dict = cached_data
        print(f'  (dataset 재사용)')
    print(f'  Total windows: {len(X):,}')
    print(f'  Label: None={np.sum(y==0):,}, '
          f'HS={np.sum(y==1):,}, TO={np.sum(y==2):,}')

    subjects = sorted(set(subj))
    all_rows = []
    for fold_idx, test_subj in enumerate(subjects, 1):
        print(f'\n  Fold {fold_idx}/{len(subjects)}: test = {test_subj}')
        train_mask = subj != test_subj
        test_mask = subj == test_subj
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]

        preds = train_fold(X_train, y_train, X_test,
                            model_class=model_class)
        rows = evaluate_predictions(preds,
                                    trial_id[test_mask],
                                    t_center[test_mask],
                                    gt_dict, tol=TOL_MAIN)
        for r in rows:
            r.update({
                'model': model_name,
                'subject': test_subj,
                'gender': SUBJECT_GENDER[test_subj],
                'position': position,
            })
            all_rows.append(r)

        df_fold = pd.DataFrame(rows)
        print(f'    {test_subj}: HS F1 mean={df_fold["F1_HS_150"].mean():.3f}, '
              f'TO F1 mean={df_fold["F1_TO_150"].mean():.3f}')

    print(f'\n  [{model_name} {position} 완료] {time.time()-t_pos_start:.0f}s')
    return all_rows, (X, y, subj, trial_id, t_center, gt_dict)


# ─────────── 메인 ───────────
def main():
    print(f'GT_DIR  = {RESULTS_DIR.parent / "output" / "synced"}')
    print(f'RESULTS = {RESULTS_DIR}\n')
    print(f'딥러닝 설정: {INPUT_CHANNELS}채널, '
          f'윈도우 {WINDOW_MS}ms, stride {STRIDE_SAMPLES}샘플')
    print(f'             {N_EPOCHS} epochs, batch {BATCH_SIZE}, '
          f'LR {LR}, LOSO 4 fold')
    print(f'PyTorch: {torch.__version__}, '
          f'device: cpu (CUDA: {torch.cuda.is_available()})\n')

    trials = build_trial_list()
    print(f'trial: {len(trials)}개')

    # 모델 목록
    models = [
        ('CNN', GaitCNN),
        ('LSTM', GaitLSTM),
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    t_start = time.time()

    # 위치별로 dataset 빌드 한 번 → 두 모델 모두에 재사용
    for position in CNN_POSITIONS:
        position_cache = None  # 첫 모델에서 데이터셋 빌드 후 재사용
        for model_name, model_class in models:
            cache_csv = RESULTS_DIR / f'{model_name.lower()}_results.csv'
            existing = []
            # 부분 캐시: 다른 위치 결과는 보존
            if cache_csv.exists():
                df_existing = pd.read_csv(cache_csv)
                if position in df_existing['position'].unique():
                    print(f'\n[캐시] {model_name} | {position} 기존 결과 사용')
                    continue
                existing = [df_existing]

            rows, position_cache = run_loso(
                position, trials,
                model_class=model_class,
                model_name=model_name,
                cached_data=position_cache,
            )
            df_new = pd.DataFrame(rows)
            df_combined = pd.concat(existing + [df_new], ignore_index=True)
            df_combined.to_csv(cache_csv, index=False)

    print(f'\n총 {time.time()-t_start:.0f}초')

    # 모든 결과 통합
    for model_name, _ in models:
        cache_csv = RESULTS_DIR / f'{model_name.lower()}_results.csv'
        if cache_csv.exists():
            all_dfs.append(pd.read_csv(cache_csv))
    df = pd.concat(all_dfs, ignore_index=True)
    df.to_csv(RESULTS_DIR / 'dl_results.csv', index=False)
    print(f'[저장] {RESULTS_DIR / "dl_results.csv"} ({len(df)}행)')

    # ─── 요약 ───
    print('\n━━━━━ 모델 × 위치 4명 평균 ━━━━━')
    pivot = (df.groupby(['model', 'position'])
             [['F1_HS_150', 'F1_TO_150']]
             .agg(['mean', 'std']))
    print(pivot.round(3).to_string())

    print('\n━━━━━ 모델 × 위치 × 피험자 ━━━━━')
    by_subj = (df.groupby(['model', 'position', 'subject'])
                [['F1_HS_150', 'F1_TO_150']].mean())
    print(by_subj.round(3).to_string())

    # 1단계 비교
    print('\n━━━━━ 비교: 1단계 단일 채널 best vs CNN vs LSTM ━━━━━')
    try:
        df_phase1 = pd.read_csv(RESULTS_DIR / 'cell_results.csv')
        for position in CNN_POSITIONS:
            sub = df_phase1[df_phase1['sensor'] == position]
            best_HS = (sub.groupby(['channel', 'algorithm'])['F1_HS_150']
                        .mean().sort_values(ascending=False).head(1))
            best_TO = (sub.groupby(['channel', 'algorithm'])['F1_TO_150']
                        .mean().sort_values(ascending=False).head(1))
            print(f'\n  [{position}]')
            print(f'    1단계 best HS: {best_HS.index[0]} → {best_HS.iloc[0]:.3f}')
            for model_name, _ in models:
                sub_m = df[(df['position'] == position) &
                            (df['model'] == model_name)]
                if len(sub_m):
                    cnn_hs = sub_m['F1_HS_150'].mean()
                    print(f'    {model_name:<5}      HS:                  {cnn_hs:.3f}')
            print(f'    1단계 best TO: {best_TO.index[0]} → {best_TO.iloc[0]:.3f}')
            for model_name, _ in models:
                sub_m = df[(df['position'] == position) &
                            (df['model'] == model_name)]
                if len(sub_m):
                    cnn_to = sub_m['F1_TO_150'].mean()
                    print(f'    {model_name:<5}      TO:                  {cnn_to:.3f}')
    except Exception as e:
        print(f'  비교 실패: {e}')

    return df


if __name__ == '__main__':
    main()
