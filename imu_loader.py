"""
imu_loader.py
=============
EBIMU 무선 IMU 데이터 공통 로더. 모든 분석 코드가 이걸 import해서 사용.

핵심 원칙: 행 번호가 아니라 timestamp를 시간축으로 사용
  → 샘플레이트 편차(파일마다 다름)와 패킷 드롭에 robust

기능:
  1. timestamp(ms, 0~60000 wrap) unwrap     → 연속 시간 복원
  2. 실제 샘플레이트 자동 측정               → 파일/센서마다 다를 수 있음
  3. 5개 센서를 공통 절대 시간축에 정렬       → 수신기 단일 클럭 기준 (3ms 차 반영)
  4. 패킷 드롭 검출 + 품질 리포트
  5. (옵션) 균일 격자 리샘플                  → CNN 다채널 입력용

매뉴얼 근거: SET OUTPUT TIME STAMP = ms 단위, 0~60000(1분) wrap
"""

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

TS_WRAP = 60000.0          # timestamp wrap 주기 (ms)
TS_COL = 'Timestamp'
ID_COL = 'Sensor_ID'
# Quaternion 컬럼: 매뉴얼 출력순서 z,y,x,w → 컬럼명 Qz,Qy,Qx,Qw
QUAT_COLS = ['Qz', 'Qy', 'Qx', 'Qw']
SIGNAL_COLS = QUAT_COLS + ['Gyro_X', 'Gyro_Y', 'Gyro_Z',
                           'Acc_X', 'Acc_Y', 'Acc_Z']


def quat_to_euler(df):
    """
    Qz,Qy,Qx,Qw → roll/pitch/yaw (deg) 변환.
    scipy는 (x,y,z,w) 순서를 받음. 매뉴얼: pitch -90~+90.
    df에 Euler_roll, Euler_pitch, Euler_yaw 컬럼 추가해서 반환.
    """
    if not all(c in df.columns for c in QUAT_COLS):
        return df
    quat = np.column_stack([df['Qx'].values, df['Qy'].values,
                            df['Qz'].values, df['Qw'].values]).astype(float)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    quat = quat / norm
    eul = R.from_quat(quat).as_euler('xyz', degrees=True)
    df['Euler_roll'] = eul[:, 0]
    df['Euler_pitch'] = eul[:, 1]
    df['Euler_yaw'] = eul[:, 2]
    return df


def unwrap_timestamp(ts, wrap=TS_WRAP):
    """60000ms에서 0으로 wrap하는 카운터를 연속 ms로 복원"""
    ts = np.asarray(ts, dtype=float)
    out = ts.copy()
    add = 0.0
    for i in range(1, len(out)):
        if ts[i] < ts[i - 1] - wrap / 2:    # 큰 음의 점프 = wrap 발생
            add += wrap
        out[i] = ts[i] + add
    return out


def load_imu_sheet(xlsx_path, sheet_name, resample_hz=None, drop_gap_factor=1.8):
    """
    한 시트에서 Sensor_ID별 DataFrame dict 반환.
    각 DataFrame에 't'(초, 공통 절대 시간축), '_ts_ms'(unwrap된 ms) 컬럼 추가.

    Parameters
    ----------
    resample_hz    : None이면 raw timestamp 유지, 값 주면 균일 격자 보간
    drop_gap_factor: 정상 간격의 이 배수 초과 시 드롭으로 판정

    Returns
    -------
    sensors : dict {sensor_id: DataFrame}
    quality : DataFrame  (센서별 레이트/드롭 리포트)
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    if TS_COL not in df.columns:
        raise ValueError(f"'{TS_COL}' 컬럼 없음. timestamp가 포함된 원본인지 확인.")

    # 신호/timestamp 컬럼을 숫자로 강제 변환 (깨진 값 → NaN)
    for c in [TS_COL] + [s for s in SIGNAL_COLS if s in df.columns]:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    n_bad_ts = int(df[TS_COL].isna().sum())
    if n_bad_ts:
        print(f'    (정보) Timestamp 결측 {n_bad_ts}행 발견 → 센서별 보간 처리')

    sensors, raw_unwrap = {}, {}
    for sid, g in df.groupby(ID_COL, sort=False):
        g = g.reset_index(drop=True).copy()

        # 1) timestamp 결측: 센서 내 위치기반 선형보간 (양끝은 외삽)
        if g[TS_COL].isna().any():
            ts_interp = g[TS_COL].interpolate(method='linear',
                                              limit_direction='both')
            g[TS_COL] = ts_interp
        # 2) 신호 결측: 선형보간
        for c in [s for s in SIGNAL_COLS if s in g.columns]:
            if g[c].isna().any():
                g[c] = g[c].interpolate(method='linear', limit_direction='both')

        ts_un = unwrap_timestamp(g[TS_COL].values)
        g['_ts_ms'] = ts_un
        g = quat_to_euler(g)          # Euler_roll/pitch/yaw 추가
        sensors[str(sid)] = g
        raw_unwrap[str(sid)] = ts_un

    t0 = min(ts.min() for ts in raw_unwrap.values())

    quality = []
    for sid, g in sensors.items():
        g['t'] = (g['_ts_ms'].values - t0) / 1000.0
        dt = np.diff(g['_ts_ms'].values)
        med = float(np.median(dt)) if len(dt) else np.nan
        rate = 1000.0 / med if med and med > 0 else np.nan
        drops = int(np.sum(dt > med * drop_gap_factor)) if len(dt) else 0
        quality.append({
            'sensor': sid,
            'n_samples': len(g),
            'duration_s': round(float(g['t'].iloc[-1] - g['t'].iloc[0]), 2) if len(g) else 0,
            'rate_Hz': round(rate, 2),
            'median_dt_ms': round(med, 2),
            'max_gap_ms': round(float(dt.max()), 1) if len(dt) else 0,
            'n_drops': drops,
        })
    qdf = pd.DataFrame(quality)

    if resample_hz:
        sensors = {sid: resample_to_grid(g, resample_hz) for sid, g in sensors.items()}

    return sensors, qdf


def resample_to_grid(g, hz, cols=None):
    """불균일 timestamp 신호를 hz 균일 격자로 선형보간 (CNN 다채널용)"""
    cols = cols or [c for c in SIGNAL_COLS if c in g.columns]
    t = g['t'].values
    t_grid = np.arange(t[0], t[-1], 1.0 / hz)
    out = pd.DataFrame({'t': t_grid})
    for c in cols:
        out[c] = np.interp(t_grid, t, g[c].values)
    out['Sensor_ID'] = g['Sensor_ID'].iloc[0]
    return out


def quick_report(qdf, label=''):
    """품질 리포트 콘솔 출력 + 경고"""
    print(f'  [품질] {label}')
    print(qdf.to_string(index=False))
    rates = qdf['rate_Hz']
    if rates.std() > 2:
        print(f'    경고: 센서간 레이트 편차 큼 (std={rates.std():.1f}Hz)')
    heavy = qdf[qdf['n_drops'] > qdf['n_samples'] * 0.02]
    if len(heavy):
        print(f'    경고: 드롭 2% 초과 센서: {heavy["sensor"].tolist()}')


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else '260519_PJH_2_IMU.xlsx'
    sheet = sys.argv[2] if len(sys.argv) > 2 else None
    xls = pd.ExcelFile(path)
    sheet = sheet or xls.sheet_names[0]
    sensors, qdf = load_imu_sheet(path, sheet)
    quick_report(qdf, f'{path}::{sheet}')