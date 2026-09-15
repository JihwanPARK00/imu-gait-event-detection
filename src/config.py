"""
src/config.py — 모든 설정 한 곳
=================================
이 파일만 수정하면 다른 모듈에 자동 반영됨.
"""
from pathlib import Path

# ─────────── 폴더 경로 ───────────
ROOT = Path(__file__).resolve().parent.parent  # ICCAS/
IMU_DIR = ROOT / 'imu'                         # xlsx는 ICCAS/imu/ 안에
GT_DIR = ROOT / 'output' / 'synced'            # synced csv
RESULTS_DIR = ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'

# ─────────── 피험자 ───────────
SUBJECT_XLSX = {
    'PJH': '260519_PJH_IMU.xlsx',
    'KKH': '260522_KKH_IMU.xlsx',
    'LNH': '260522_LNH_IMU.xlsx',
    'YHS': '260523_YHS_IMU.xlsx',  # 260528_YHS는 재실험본, synced csv 없어 자동 무시
}
SUBJECT_GENDER = {
    'PJH': 'M', 'KKH': 'M',
    'LNH': 'F', 'YHS': 'F',
}
SUBJECT_ORDER = ['PJH', 'KKH', 'LNH', 'YHS']  # 남자 먼저, 여자 다음

# ─────────── 센서 ───────────
SENSORS = {
    '100-0': 'foot',         # 환측 발등
    '100-1': 'shank',        # 환측 종아리
    '100-2': 'thigh_aff',    # 환측 허벅지
    '100-3': 'thigh_sound',  # 건측 허벅지
    '100-4': 'lumbar',       # 배꼽
}
SENSOR_ORDER = ['foot', 'shank', 'thigh_aff', 'thigh_sound', 'lumbar']

# ─────────── 채널 (9개) ───────────
CHANNELS = [
    'Gyro_X', 'Gyro_Y', 'Gyro_Z',
    'Acc_X', 'Acc_Y', 'Acc_Z',
    'Euler_roll', 'Euler_pitch', 'Euler_yaw',
]

# Quaternion 컬럼명 후보 (자동 탐지)
Q_CANDIDATES = [
    ['Q1', 'Q2', 'Q3', 'Q4'],
    ['Q0', 'Q1', 'Q2', 'Q3'],
    ['Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z'],
    ['qw', 'qx', 'qy', 'qz'],
]
# Quaternion → Euler 변환: Q=(w,x,y,z), 컨벤션 xyz (verify_quaternion에서 확정)
EULER_CONVENTION = 'xyz'

# ─────────── 평가 ───────────
TOL_MAIN_MS = 150  # 기존 논문 기준
TOL_AUX_MS = 50    # 엄격 (부가)

# ─────────── 속도 ───────────
SPEEDS = [2, 3, 4, 5]  # km/h

# ─────────── 시각화 색상 ───────────
SUBJECT_COLORS = {
    'PJH': '#1f77b4',
    'KKH': '#2ca02c',
    'LNH': '#ff7f0e',
    'YHS': '#d62728',
}
GENDER_COLORS = {'M': '#4C72B0', 'F': '#DD8452'}
ALGO_COLORS = {
    'TB_classic': '#888888',
    'TB_adaptive': '#2E7D32',
    'BPF': '#1565C0',
    'TM': '#6A1B9A',
}
