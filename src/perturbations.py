"""
src/perturbations.py — 5개 합성 교란 함수
==========================================
P1 노이즈   : 가우시안 노이즈 더하기
P2 오정렬   : 센서 부착 회전 시뮬 (Gyro/Acc는 3축 회전 매트릭스, Euler는 근사)
P4 드롭     : 랜덤 NaN 후 선형 보간
P6 bias     : Constant offset 더하기
P7 샘플링   : 안티앨리어싱 후 다운샘플

재현성: 시드 고정 (level + 채널 hash 조합으로 동일 결과)
"""
import numpy as np
import pandas as pd
from fractions import Fraction
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation as R


# ─────────── 교란 명세 ───────────
PERTURBATIONS = {
    'P1': {
        'name': 'noise',
        'levels': [1, 5, 20, 50, 100],
        'unit': 'deg/s',
        'desc': '가우시안 노이즈 σ',
    },
    'P2': {
        'name': 'misalign',
        'levels': [5, 10, 15, 20, 30],
        'unit': 'deg',
        'desc': '센서 부착 회전',
    },
    'P4': {
        'name': 'drop',
        'levels': [0.01, 0.05, 0.10, 0.20, 0.30],
        'unit': 'ratio',
        'desc': '패킷 드롭 비율',
    },
    'P6': {
        'name': 'bias',
        'levels': [1, 5, 10, 25, 50],
        'unit': 'deg/s',
        'desc': 'Constant offset',
    },
    'P7': {
        'name': 'downsample',
        'levels': [60, 45, 30, 20, 10],
        'unit': 'Hz',
        'desc': '다운샘플링 fs',
    },
}


def _seeded_rng(key):
    """key 기반 결정적 RNG (재현성)"""
    h = hash(key) & 0xFFFFFFFF
    return np.random.default_rng(h)


# ─────────── P1 노이즈 ───────────
def apply_P1_noise(sig, fs, sigma, seed_key='P1'):
    """가우시안 노이즈 N(0, σ²) 더하기"""
    rng = _seeded_rng((seed_key, sigma, len(sig)))
    return sig + rng.normal(0, sigma, size=len(sig))


# ─────────── P2 오정렬 ───────────
def apply_P2_misalignment(df, theta_deg, channel):
    """센서 부착 오정렬 시뮬.

    Gyro/Acc: 3축을 X축 주위 theta_deg 회전 후 채널 추출 (정확)
    Euler   : 해당 채널에 theta_deg/3 더하기 (작은 각도 근사)
    """
    if channel.startswith('Gyro_'):
        group = ['Gyro_X', 'Gyro_Y', 'Gyro_Z']
    elif channel.startswith('Acc_'):
        group = ['Acc_X', 'Acc_Y', 'Acc_Z']
    elif channel.startswith('Euler_'):
        # 근사: 작은 각도면 Euler 각도에 직접 영향
        return df[channel].values.astype(float) + theta_deg / 3
    else:
        return df[channel].values.astype(float)

    if not all(g in df.columns for g in group):
        return df[channel].values.astype(float)

    v = df[group].values.astype(float)  # (N, 3)
    rot = R.from_euler('x', theta_deg, degrees=True)
    v_rot = rot.apply(v)
    idx = group.index(channel)
    return v_rot[:, idx]


# ─────────── P4 드롭 ───────────
def apply_P4_drop(sig, rate, seed_key='P4'):
    """랜덤 비율로 NaN 후 선형 보간"""
    rng = _seeded_rng((seed_key, rate, len(sig)))
    s = sig.copy().astype(float)
    n = len(s)
    n_drop = int(n * rate)
    if n_drop > 0:
        drop_idx = rng.choice(n, size=n_drop, replace=False)
        s[drop_idx] = np.nan
    return pd.Series(s).interpolate(limit_direction='both').fillna(0).values


# ─────────── P6 bias ───────────
def apply_P6_bias(sig, bias):
    """Constant offset 더하기"""
    return sig + bias


# ─────────── P7 샘플링 ───────────
def apply_P7_downsample(sig, t, fs_new):
    """안티앨리어싱 다운샘플 후 새 시간축 생성."""
    fs_old = 1 / np.median(np.diff(t))
    if fs_new >= fs_old:
        return sig, t, fs_old
    # 정수 비율로 근사 (안전)
    frac = Fraction(int(fs_new * 100), int(fs_old * 100)).limit_denominator(100)
    up, down = frac.numerator, frac.denominator
    if up == 0 or down == 0:
        return sig, t, fs_old
    try:
        s_new = resample_poly(sig, up, down)
    except Exception:
        return sig, t, fs_old
    # 새 시간축
    fs_actual = fs_old * up / down
    t_new = np.arange(len(s_new)) / fs_actual + t[0]
    return s_new, t_new, fs_actual


# ─────────── dispatch ───────────
def apply_perturbation(pert_type, sig, t, fs, level, df=None, channel=None):
    """디스패처. df, channel은 P2에만 필요."""
    if pert_type == 'P1':
        return apply_P1_noise(sig, fs, level), t, fs
    elif pert_type == 'P2':
        if df is None or channel is None:
            return sig, t, fs
        return apply_P2_misalignment(df, level, channel), t, fs
    elif pert_type == 'P4':
        return apply_P4_drop(sig, level), t, fs
    elif pert_type == 'P6':
        return apply_P6_bias(sig, level), t, fs
    elif pert_type == 'P7':
        return apply_P7_downsample(sig, t, level)
    else:
        raise ValueError(f'Unknown perturbation: {pert_type}')


# ═════════════════ 복합 교란 ═════════════════
COMPOUND_PERTURBATIONS = {
    'C1': {
        'name': 'noise_drop',
        'components': ['P1', 'P4'],
        'levels': [(5, 0.05), (20, 0.10), (50, 0.20)],
        'desc': 'P1 노이즈 + P4 드롭',
    },
    'C2': {
        'name': 'noise_misalign',
        'components': ['P1', 'P2'],
        'levels': [(5, 10), (20, 15), (50, 20)],
        'desc': 'P1 노이즈 + P2 오정렬',
    },
    'C3': {
        'name': 'misalign_drop',
        'components': ['P2', 'P4'],
        'levels': [(10, 0.05), (15, 0.10), (20, 0.20)],
        'desc': 'P2 오정렬 + P4 드롭',
    },
    'C4': {
        'name': 'all_three',
        'components': ['P1', 'P2', 'P4'],
        'levels': [(5, 10, 0.05), (20, 15, 0.10), (50, 20, 0.20)],
        'desc': '셋 다',
    },
}

LEVEL_LABELS = ['Light', 'Medium', 'Heavy']


def apply_compound(pert_type, sig, t, fs, level_tuple, df, channel):
    """복합 교란 적용.

    순서: P2 (있으면 먼저, 3축 회전 후 단축 추출) → P1 → P4
    P2가 단축 신호를 baseline df로부터 회전해서 추출하기 때문에
    먼저 적용해야 함.
    """
    info = COMPOUND_PERTURBATIONS[pert_type]
    components = info['components']

    # P2 분리 (있으면 먼저)
    p2_level = None
    other = []
    for comp, lv in zip(components, level_tuple):
        if comp == 'P2':
            p2_level = lv
        else:
            other.append((comp, lv))

    if p2_level is not None and df is not None and channel is not None:
        s = apply_P2_misalignment(df, p2_level, channel)
    else:
        s = sig.copy()

    # 나머지 (P1, P4)
    for comp, lv in other:
        if comp == 'P1':
            s = apply_P1_noise(s, fs, lv)
        elif comp == 'P4':
            s = apply_P4_drop(s, lv)

    return s, t, fs
