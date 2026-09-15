"""
sync_video_imu.py (timestamp 기반)
===================================
영상 라벨(HS/TO/JUMP/CLAP)과 IMU를 JUMP 기준으로 동기화하고,
2번째 박수(CLAP2) 기준 분석 윈도우를 산출한다. 종료 박수로 drift 검증.

핵심 안전 설계:
  - imu_loader 사용 → timestamp 기반 시간축 (행번호 X, 90.9Hz 자동, 드롭 무관)
  - 발등(100-0) 가속도 jerk로 점프 자동 검출 (영상 JUMP ±JUMP_SEARCH초)
  - offset = imu_jump_t - video_jump_t  (1점 동기화)
  - 종료 박수 이후 보행 신호가 실제로 멈추는지로 offset 유효성(=drift 無) 검증

[입력]
  imu/      피험자별 xlsx (260519_PJH_IMU.xlsx 등, 시트 = trial)
  labels/   trial별 csv (<시트명>_labels.csv)

[출력]
  output/synced/<trial>_synced.csv     라벨별 imu_time, in_window
  output/synced/<trial>_sync_check.png 점프정렬 + 라벨 + 윈도우 + drift검증
  output/synced/sync_summary.csv

[의존성] imu_loader.py 가 같은 폴더에 있어야 함
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from imu_loader import load_imu_sheet
import warnings
warnings.filterwarnings('ignore')

# ==================== 설정 ====================
IMU_DIR = 'imu'
LABEL_DIR = 'labels'
OUTPUT_DIR = 'output/synced'

FOOT_SENSOR = '100-0'      # 점프 검출용 발등
JUMP_SEARCH = 2.0          # 점프 탐색 반경 (영상 JUMP ±, 초)
WIN_OFFSET = 5             # CLAP2 이후 분석 시작 (초)
WIN_DURATION = 120         # 분석 길이 (초)

# 피험자 코드 별칭: IMU 시트명 코드 → 라벨 파일 코드
# (라벨/IMU 코드가 다를 때만 사용. 현재는 KKH로 통일되어 비어있음)
SUBJECT_ALIAS = {}

mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False


# ==================== 라벨 매칭 ====================
def find_label(sheet_name, label_dir):
    """시트명으로 라벨 csv 찾기. 직접 매칭 실패 시 별칭 적용."""
    p = Path(label_dir) / f'{sheet_name}_labels.csv'
    if p.is_file():
        return p
    for imu_code, lbl_code in SUBJECT_ALIAS.items():
        if imu_code in sheet_name:
            alt = sheet_name.replace(imu_code, lbl_code)
            p2 = Path(label_dir) / f'{alt}_labels.csv'
            if p2.is_file():
                return p2
    return None


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    return df.sort_values('time_sec').reset_index(drop=True)


# ==================== 점프 검출 ====================
def detect_jump(foot, video_jump_t, search=JUMP_SEARCH):
    """
    발등 Acc magnitude의 절대 최대점으로 점프(착지) 시각 검출.
    라벨 정의 = '발가락 닿아 트레드밀 꺼지는 순간' = 착지 충격 = |Acc| 정점.
    절대 최대를 쓰므로 봉우리 뒤 작은 spike에 헷갈리지 않음(가장 큰 봉우리 선택).
    """
    t = foot['t'].values
    mag = np.sqrt(foot['Acc_X']**2 + foot['Acc_Y']**2 + foot['Acc_Z']**2).values
    lo, hi = video_jump_t - search, video_jump_t + search
    win = (t >= lo) & (t <= hi)
    if win.sum() < 3:
        win = np.ones(len(t), dtype=bool)
    mag_w = np.where(win, mag, -np.inf)
    peak = int(np.argmax(mag_w))
    return t[peak], {'t': t, 'mag': mag, 'peak': peak, 'lo': lo, 'hi': hi}


# ==================== drift 검증 ====================
def verify_sync(foot, labels, offset, window):
    """
    동기화 품질 검증 (제대로 된 방식).
    분석 윈도우 안 HS들을 IMU 시간으로 변환 → 각 HS 주변 발등 자이로 파형을 모음.
      - sharpness: HS-triggered 평균 파형의 선명도 (HS가 일관 위상이면 큼)
      - drift_lag_ms: 초반 HS 평균파형 vs 후반 HS 평균파형의 위상차
                      (drift 있으면 후반이 밀려서 lag 커짐. ±20ms 이내면 양호)
    """
    hs = labels[labels['event'] == 'HS']['time_sec'].values + offset
    hs = hs[(hs >= window[0]) & (hs <= window[1])]
    if len(hs) < 12:
        return None
    t = foot['t'].values
    gmag = np.sqrt(foot['Gyro_X']**2 + foot['Gyro_Y']**2 + foot['Gyro_Z']**2).values

    half = 0.25                      # HS 주변 ±0.25초 (보행주기 절반 수준)
    n_grid = 50
    xi = np.linspace(-half, half, n_grid)
    segs = []
    for h in hs:
        m = (t >= h - half) & (t <= h + half)
        if m.sum() > 5:
            segs.append(np.interp(xi, t[m] - h, gmag[m]))
    if len(segs) < 12:
        return None
    segs = np.array(segs)
    avg = segs.mean(0)
    sharpness = avg.std() / (segs.std(0).mean() + 1e-9)

    # 초반 1/3 vs 후반 1/3 평균파형 위상차
    k = len(segs) // 3
    early = segs[:k].mean(0)
    late = segs[-k:].mean(0)
    ea, la = early - early.mean(), late - late.mean()
    xcorr = np.correlate(ea, la, 'full')
    lag = xcorr.argmax() - (len(la) - 1)
    dt_grid = (2 * half) / (n_grid - 1)
    drift_lag_ms = lag * dt_grid * 1000

    return {'sharpness': round(float(sharpness), 3),
            'drift_lag_ms': round(float(drift_lag_ms), 1),
            'n_hs_used': len(segs)}


# ==================== 윈도우 ====================
def compute_window(labels, offset):
    claps = sorted(labels[labels['event'] == 'CLAP']['time_sec'].tolist())
    if len(claps) < 2:
        return None, claps
    clap2_imu = claps[1] + offset
    return (clap2_imu + WIN_OFFSET, clap2_imu + WIN_OFFSET + WIN_DURATION), claps


# ==================== 검증 그림 ====================
def plot_sync_check(foot, labels, offset, jdiag, window, drift, label, save):
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    t = jdiag['t']

    # (1) 점프 정렬
    ax = axes[0]
    lo, hi = jdiag['lo'], jdiag['hi']
    z = (t >= lo - 1) & (t <= hi + 1)
    ax.plot(t[z], jdiag['mag'][z], lw=1, label='발등 |Acc|')
    ij = t[jdiag['peak']]
    ax.axvline(ij, color='red', ls='--', lw=2, label=f'IMU 점프 @{ij:.3f}s')
    vj = labels[labels['event'] == 'JUMP']['time_sec'].iloc[0]
    ax.axvline(vj + offset, color='green', ls=':', lw=2,
               label=f'영상JUMP+offset @{vj + offset:.3f}s')
    ax.axvspan(lo, hi, alpha=0.1, color='orange')
    ax.set_title(f'{label} — 점프 동기화  offset={offset:+.3f}s')
    ax.set_xlabel('IMU time (s)'); ax.set_ylabel('|Acc|')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (2) 전체 + 라벨 + 윈도우 + drift
    ax = axes[1]
    gmag = np.sqrt(foot['Gyro_X']**2 + foot['Gyro_Y']**2 + foot['Gyro_Z']**2).values
    ax.plot(foot['t'], gmag, lw=0.4, color='gray', label='발등 |Gyro|')
    for ev, c in [('HS', 'tab:blue'), ('TO', 'tab:green')]:
        et = labels[labels['event'] == ev]['time_sec'].values + offset
        ax.plot(et, np.full_like(et, gmag.max() * 1.05), '|', color=c, ms=8,
                label=f'{ev}({len(et)})')
    for ct in labels[labels['event'] == 'CLAP']['time_sec']:
        ax.axvline(ct + offset, color='orange', lw=1, alpha=0.7)
    if window:
        ax.axvspan(window[0], window[1], alpha=0.15, color='red',
                   label=f'분석윈도우({window[0]:.0f}~{window[1]:.0f}s)')
    if drift:
        ax.plot([], [], ' ',
                label=f'sync: lag={drift["drift_lag_ms"]:.0f}ms sharp={drift["sharpness"]:.2f}')
    ax.set_title('전체 신호 + 라벨(IMU시간) + 분석윈도우 + drift검증')
    ax.set_xlabel('IMU time (s)'); ax.set_ylabel('|Gyro|')
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save, dpi=110, bbox_inches='tight')
    plt.close(fig)


# ==================== trial 처리 ====================
def process_trial(xlsx, sheet, label_dir, out_dir):
    lp = find_label(sheet, label_dir)
    if lp is None:
        print('라벨 없음'); return None

    sensors, qdf = load_imu_sheet(xlsx, sheet)
    if FOOT_SENSOR not in sensors:
        print(f'{FOOT_SENSOR} 없음'); return None
    foot = sensors[FOOT_SENSOR]
    labels = load_labels(lp)

    jr = labels[labels['event'] == 'JUMP']
    if jr.empty:
        print('JUMP 라벨 없음'); return None
    vj = jr['time_sec'].iloc[0]
    ij, jdiag = detect_jump(foot, vj)
    offset = ij - vj

    window, claps = compute_window(labels, offset)
    if window is None:
        print(f'CLAP<2 ({len(claps)}), fallback (40,160)')
        window = (40, 160)
    drift = verify_sync(foot, labels, offset, window)

    synced = labels.copy()
    synced['imu_time'] = synced['time_sec'] + offset
    synced['in_window'] = ((synced['imu_time'] >= window[0]) &
                           (synced['imu_time'] <= window[1]))
    synced.to_csv(Path(out_dir) / f'{sheet}_synced.csv',
                  index=False, encoding='utf-8-sig')
    plot_sync_check(foot, labels, offset, jdiag, window, drift,
                    sheet, Path(out_dir) / f'{sheet}_sync_check.png')

    n_hs = int(((synced['event'] == 'HS') & synced['in_window']).sum())
    n_to = int(((synced['event'] == 'TO') & synced['in_window']).sum())
    rate = qdf['rate_Hz'].median()
    lag_ms = drift['drift_lag_ms'] if drift else np.nan
    sharp = drift['sharpness'] if drift else np.nan
    print(f'offset={offset:+.3f}s rate={rate:.1f}Hz HS_win={n_hs} '
          f'TO_win={n_to} drift_lag={lag_ms}ms sharp={sharp}')

    return {
        'trial': sheet, 'offset_s': round(offset, 4),
        'imu_rate_Hz': round(rate, 2), 'n_clap': len(claps),
        'win_start': round(window[0], 2), 'win_end': round(window[1], 2),
        'n_HS_win': n_hs, 'n_TO_win': n_to,
        'drift_lag_ms': lag_ms,
        'sync_sharpness': sharp,
        'max_drop_sensor': qdf.loc[qdf['n_drops'].idxmax(), 'sensor'],
        'max_n_drops': int(qdf['n_drops'].max()),
    }


def main():
    out = Path(OUTPUT_DIR); out.mkdir(parents=True, exist_ok=True)
    xlsx_files = sorted(Path(IMU_DIR).glob('*.xlsx'))
    if not xlsx_files:
        print(f'{IMU_DIR}에 xlsx 없음'); return

    summary = []
    for xlsx in xlsx_files:
        print(f'\n### {xlsx.name}')
        for sheet in pd.ExcelFile(xlsx).sheet_names:
            print(f'  [{sheet}] ', end='')
            try:
                r = process_trial(xlsx, sheet, LABEL_DIR, out)
                if r:
                    summary.append(r)
            except Exception as e:
                print(f'에러: {e}')

    if summary:
        sdf = pd.DataFrame(summary)
        sdf.to_csv(out / 'sync_summary.csv', index=False, encoding='utf-8-sig')
        print(f'\n{"="*70}\n요약: {out / "sync_summary.csv"}')
        print(sdf[['trial', 'offset_s', 'imu_rate_Hz', 'n_HS_win',
                   'drift_lag_ms', 'sync_sharpness', 'max_n_drops']].to_string(index=False))
        print(f'\noffset mean={sdf["offset_s"].mean():+.3f} '
              f'std={sdf["offset_s"].std():.3f}')
        if sdf['offset_s'].std() > 0.5:
            print('  경고: offset 편차 큼 — 일부 점프검출 확인 필요')
        bad = sdf[sdf['drift_lag_ms'].abs() > 20]
        if len(bad):
            print(f'  주의: drift_lag>20ms trial (sync_check 확인): {bad["trial"].tolist()}')
        weak = sdf[sdf['sync_sharpness'] < 0.3]
        if len(weak):
            print(f'  주의: sharpness<0.3 (HS 위상 흐림): {weak["trial"].tolist()}')


if __name__ == '__main__':
    main()