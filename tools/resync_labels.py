"""
resync_labels.py — 걷기 신호로 GT 라벨 자동 재동기화
=====================================================
점프 싱크가 trial마다 "반 걸음"쯤 밀리는 문제를, 종아리 보행 파형에 GT HS를
자동 정렬(교차상관)해서 보정한다. manual_sync.py 슬라이더의 자동화 버전.

원리
----
정상 보행은 발뒤꿈치 닿을 때(HS)마다 IMU 신호 모양이 항상 똑같다.
→ 여러 trial의 "HS 정렬 평균 파형"이 서로 겹친다(같은 위상).
→ 다수(in-phase) trial의 평균을 '정답 템플릿'으로 삼고, 각 trial을 이 템플릿에
  가장 잘 맞는 시간만큼 밀어(shift) imu_time을 보정. (2-pass로 템플릿 정제)
trial 내부 시간은 일정(drift 없음)하므로 상수 shift 하나면 전체가 맞는다.

전제: 32개 중 절반 이상이 올바르게 싱크돼 있어야 템플릿이 옳다(현재 데이터 충족).

입출력
------
입력 : IMU xlsx (imu_loader.load_imu_sheet) + 기존 *_synced.csv
출력 : OUT_DIR/*_synced.csv (imu_time 보정본, 원본 손대지 않음)
       OUT_DIR/resync_report.csv (trial별 shift / 보정 전후 상관 / 플래그)

사용
----
imu_loader.py 가 같은 폴더(또는 PYTHONPATH)에 있어야 함.  →  python resync_labels.py
경로/기준센서는 아래 '설정'에서 수정.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root
import glob
from pathlib import Path
import numpy as np
import pandas as pd

from imu_loader import load_imu_sheet

# ─────────────── 설정 ───────────────
IMU_DIR   = 'imu'            # xlsx 폴더
SYNC_DIR  = 'output/synced'  # *_synced.csv 폴더
OUT_DIR   = 'output/synced_resync'
REF_SENSOR  = '100-1'               # 기준 센서: 종아리 (정렬 파형이 가장 깨끗)
REF_CHANNEL = 'Gyro_Z'             # 기준 채널 (발등 정렬을 원하면 '100-0','Gyro_Y')
PRE, POST   = 0.4, 0.6             # HS 정렬 평균 창 [HS-PRE, HS+POST] (초)
SEARCH      = 0.75                 # 오프셋 탐색 범위 ±초 (한 주기보다 작아 앨리어싱 없음)
STEP        = 0.005                # 탐색 간격 (초)
VOTE_CORR   = 0.5                  # 템플릿 합의: 이 상관 이상이면 '같은 위상 이웃'
MIN_CORR_OK = 0.85                 # 보정 후 상관이 이 미만이면 수동확인(CHECK)
FIX_THRESH_MS = 50                 # 이 이상 움직인 trial을 '보정됨'으로 집계
N_REFINE    = 2                    # 템플릿 정제 반복 횟수


# ─────────────── 핵심 함수 ───────────────
def hs_triggered_avg(t, sig, HS, shift=0.0, fs=None):
    """HS(+shift) 순간에 정렬해 신호를 평균 → z-정규화 파형."""
    if fs is None:
        fs = 1.0 / np.median(np.diff(t))
    a, b = int(PRE * fs), int(POST * fs)
    segs = []
    for e in np.asarray(HS) + shift:
        c = int(np.argmin(np.abs(t - e)))
        if c - a >= 0 and c + b < len(sig):
            segs.append(sig[c - a:c + b])
    if not segs:
        return None
    w = np.mean(segs, axis=0)
    return (w - w.mean()) / (w.std() + 1e-9)


def consensus_template(waves):
    """파형 리스트에서 in-phase 다수의 평균을 템플릿으로."""
    keys = [k for k in waves if waves[k] is not None]
    M = np.array([waves[k] for k in keys])
    if len(M) == 1:
        t = M[0]
        return (t - t.mean()) / (t.std() + 1e-9), 1, 1
    C = np.corrcoef(M)
    votes = (C > VOTE_CORR).sum(axis=1)
    keep = votes >= np.median(votes)
    t = M[keep].mean(axis=0)
    return (t - t.mean()) / (t.std() + 1e-9), int(keep.sum()), len(keys)


def best_shift(t, sig, HS, template, fs):
    """template에 가장 잘 맞는 shift, (보정 전, 보정 후) 상관."""
    shifts = np.arange(-SEARCH, SEARCH + 1e-9, STEP)
    corrs = np.full(len(shifts), -1.0)
    for i, s in enumerate(shifts):
        w = hs_triggered_avg(t, sig, HS, shift=s, fs=fs)
        if w is not None:
            corrs[i] = np.corrcoef(w, template)[0, 1]
    cmax = corrs.max()
    # 최대상관과 0.02 이내인 후보 중 |shift|가 가장 작은 것 선택 (주기 앨리어싱 방지)
    cand = np.where(corrs >= cmax - 0.02)[0]
    bi = int(cand[np.argmin(np.abs(shifts[cand]))])
    c0 = corrs[int(np.argmin(np.abs(shifts)))]
    return shifts[bi], float(c0), float(corrs[bi])


def resync(data):
    """2-pass: 템플릿 만들고 정렬 → 잘 맞은 것으로 템플릿 재구성 → 재정렬."""
    waves = {k: data[k]['w0'] for k in data}
    tmpl, nk, nt = consensus_template(waves)
    print(f'  [pass 1] 템플릿 합의 {nk}/{nt}')
    result = {}
    for _ in range(N_REFINE):
        for k, d in data.items():
            s, c0, cb = best_shift(d['t'], d['sig'], d['HS'], tmpl, d['fs'])
            result[k] = {'shift': s, 'c0': c0, 'cb': cb,
                         'w': hs_triggered_avg(d['t'], d['sig'], d['HS'], s, d['fs'])}
        # 잘 맞은(cb 높은) trial의 '정렬된' 파형으로 템플릿 재구성
        good = {k: r['w'] for k, r in result.items() if r['cb'] >= MIN_CORR_OK}
        if len(good) >= 3:
            tmpl, nk, nt = consensus_template(good)
    return result


# ─────────────── I/O ───────────────
def scan_trials(imu_dir=IMU_DIR, sync_dir=SYNC_DIR):
    trials = []
    for xlsx in sorted(glob.glob(str(Path(imu_dir) / '*.xlsx'))):
        try:
            sheets = pd.ExcelFile(xlsx).sheet_names
        except Exception as e:
            print(f'  [skip xlsx] {xlsx}: {e}');  continue
        for sh in sheets:
            csv = Path(sync_dir) / f'{sh}_synced.csv'
            if csv.is_file():
                trials.append((xlsx, str(sh), csv))
    return trials


def load_ref_signal(xlsx, sheet):
    sensors, _ = load_imu_sheet(xlsx, sheet)
    if REF_SENSOR not in sensors:
        raise KeyError(f'{REF_SENSOR} 센서 없음 in {sheet}')
    d = sensors[REF_SENSOR]
    return d['t'].values, d[REF_CHANNEL].values.astype(float)


def read_HS(csv):
    gt = pd.read_csv(csv)
    hs_all = gt[gt['event'] == 'HS']['imu_time'].values
    if 'in_window' in gt.columns:
        hs_w = gt[(gt['event'] == 'HS') & (gt['in_window'] == True)]['imu_time'].values
        if len(hs_w) >= 5:
            return hs_w
    return hs_all


def apply_offset(csv, shift):
    gt = pd.read_csv(csv)
    gt['imu_time'] = gt['imu_time'] + shift
    if 'in_window' in gt.columns and gt['in_window'].any():
        lo = gt.loc[gt['in_window'], 'imu_time'].min()
        hi = gt.loc[gt['in_window'], 'imu_time'].max()
        gt['in_window'] = (gt['imu_time'] >= lo) & (gt['imu_time'] <= hi)
    return gt


# ─────────────── 메인 ───────────────
def main():
    trials = scan_trials()
    if not trials:
        print(f'trial 없음. IMU_DIR={IMU_DIR}, SYNC_DIR={SYNC_DIR} 확인.');  return
    print(f'{len(trials)} trial 발견\n[1/3] 신호 로드 + HS 정렬 평균')

    data = {}
    for xlsx, sh, csv in trials:
        try:
            t, sig = load_ref_signal(xlsx, sh)
            HS = read_HS(csv)
            fs = 1.0 / np.median(np.diff(t))
            data[sh] = {'xlsx': xlsx, 'csv': csv, 't': t, 'sig': sig, 'HS': HS,
                        'fs': fs, 'w0': hs_triggered_avg(t, sig, HS, 0.0, fs)}
            print(f'  {sh:26} HS={len(HS)}')
        except Exception as e:
            print(f'  [skip] {sh}: {e}')

    print(f'\n[2/3] 2-pass 자동 정렬 (기준 {REF_SENSOR} {REF_CHANNEL})')
    result = resync(data)

    print('\n[3/3] 보정 CSV 저장')
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    report = []
    for sh, r in result.items():
        gt = apply_offset(data[sh]['csv'], r['shift'])
        gt.to_csv(Path(OUT_DIR) / f'{sh}_synced.csv', index=False, encoding='utf-8-sig')
        flag = '' if r['cb'] >= MIN_CORR_OK else 'CHECK'
        if abs(abs(r['shift']) - SEARCH) < STEP:
            flag = (flag + ' EDGE').strip()
        report.append({'trial': sh, 'shift_ms': round(r['shift'] * 1000, 1),
                       'corr_before': round(r['c0'], 3), 'corr_after': round(r['cb'], 3),
                       'flag': flag})
        print(f'  {sh:26} shift {r["shift"]*1000:+7.1f}ms  '
              f'corr {r["c0"]:+.2f}→{r["cb"]:+.2f}  {flag}')

    rep = pd.DataFrame(report).sort_values('shift_ms')
    rep.to_csv(Path(OUT_DIR) / 'resync_report.csv', index=False, encoding='utf-8-sig')
    n_fix = int((rep['shift_ms'].abs() > FIX_THRESH_MS).sum())
    n_check = int(rep['flag'].str.contains('CHECK').sum())
    print(f'\n완료 → {OUT_DIR}/  (보정 CSV {len(rep)}개 + resync_report.csv)')
    print(f'  {FIX_THRESH_MS}ms 이상 보정된 trial: {n_fix} / {len(rep)}')
    if n_check:
        print(f'  ⚠ 수동확인 필요(CHECK) {n_check}개: '
              f'{rep[rep.flag.str.contains("CHECK")].trial.tolist()}')


if __name__ == '__main__':
    main()