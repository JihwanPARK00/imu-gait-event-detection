"""
src/perturbations_mc.py — 9채널 동시 교란 (딥러닝 강건성 평가용)
================================================================
기존 src/perturbations.py 는 '단일 채널'에 교란을 적용함.
CNN/LSTM은 9채널을 동시에 입력받으므로 9채널 텐서 전체에 물리적으로
일관된 교란을 적용해야 함. 이 모듈이 그 역할을 함.

기존 perturbations.py 와의 차이 (논문 Methods에 반드시 명시할 것)
-----------------------------------------------------------------
P1 noise      : 채널마다 단위가 다름(deg/s, g, deg) → 기준 채널 대비
                std 비율로 σ를 환산 (SNR-matched). 기준 채널의 σ는
                phase2와 동일한 deg/s 값.
P2 misalign   : Gyro 3축·Acc 3축을 '동일한' 회전행렬로 회전 (물리적으로
                정확). Euler 3채널은 phase2와 동일하게 θ/3 오프셋 근사.
                → 단일채널 버전보다 더 정확한 시뮬레이션.
P4 drop       : 패킷 손실은 9채널이 '함께' 사라짐 → 행(row) 단위 드롭.
                (phase2는 채널별 독립 드롭이었음)
P6 bias       : P1과 동일한 std 비율 환산.
P7 downsample : 다운샘플 후 '원래 시간축으로 재보간'하여 길이·fs 유지.
                딥러닝 입력 윈도우 길이가 fs에 묶여 있기 때문.
                → "저사양 센서를 기존 91 Hz 파이프라인에 그대로 꽂은" 상황.

교란 범위(scope)
----------------
'all9'   : 9채널 전부           — 전통 알고리즘과 동등 비교
'gyro3'  : Gyro 3축만           — 나머지 6채널의 보완 효과 측정
'ref1'   : 기준 단일 채널만     — 다채널 중복성(redundancy)의 직접 증거
"""
import numpy as np
import pandas as pd
from fractions import Fraction
from scipy.signal import resample_poly
from scipy.spatial.transform import Rotation as R

# 채널 순서 (src.config.CHANNELS 와 동일해야 함)
CH_ORDER = ['Gyro_X', 'Gyro_Y', 'Gyro_Z',
            'Acc_X', 'Acc_Y', 'Acc_Z',
            'Euler_roll', 'Euler_pitch', 'Euler_yaw']
IDX = {c: i for i, c in enumerate(CH_ORDER)}
GYRO_IDX = [0, 1, 2]
ACC_IDX = [3, 4, 5]
EULER_IDX = [6, 7, 8]

SCOPES = {
    'all9': list(range(9)),
    'gyro3': GYRO_IDX,
    # 'ref1' 은 런타임에 기준 채널명으로 결정
}

# phase2 와 동일한 교란 레벨 (src/perturbations.py PERTURBATIONS 와 일치)
PERTURBATIONS_MC = {
    'P1': {'name': 'noise', 'levels': [1, 5, 20, 50, 100], 'unit': 'deg/s'},
    'P2': {'name': 'misalign', 'levels': [5, 10, 15, 20, 30], 'unit': 'deg'},
    'P4': {'name': 'drop', 'levels': [0.01, 0.05, 0.10, 0.20, 0.30], 'unit': 'ratio'},
    'P6': {'name': 'bias', 'levels': [1, 5, 10, 25, 50], 'unit': 'deg/s'},
    'P7': {'name': 'downsample', 'levels': [60, 45, 30, 20, 10], 'unit': 'Hz'},
}


def _seeded_rng(key):
    """결정적 RNG — 재현성 확보 (파이썬 hash 랜덤화 회피 위해 문자열 고정)"""
    s = repr(key).encode()
    h = int.from_bytes(s[:0] + __import__('hashlib').md5(s).digest()[:4], 'little')
    return np.random.default_rng(h)


def scope_indices(scope, ref_channel):
    if scope == 'ref1':
        return [IDX[ref_channel]]
    return SCOPES[scope]


def channel_scales(sig9, ref_channel):
    """각 채널 std / 기준 채널 std → 단위 환산 계수 (SNR-matched)"""
    sd = sig9.std(axis=0)
    sd = np.where(sd < 1e-9, 1e-9, sd)
    ref_sd = sd[IDX[ref_channel]]
    return sd / ref_sd


# ─────────── P1 노이즈 ───────────
def mc_P1_noise(sig9, sigma, ref_channel, cols, seed_key='P1'):
    rng = _seeded_rng((seed_key, sigma, sig9.shape[0], tuple(cols)))
    out = sig9.copy()
    scales = channel_scales(sig9, ref_channel)
    for c in cols:
        out[:, c] = out[:, c] + rng.normal(0, sigma * scales[c], size=sig9.shape[0])
    return out


# ─────────── P2 오정렬 ───────────
def mc_P2_misalign(sig9, theta_deg, cols):
    """Gyro/Acc 삼축을 X축 기준 동일 회전. Euler는 θ/3 오프셋 근사."""
    out = sig9.copy()
    rot = R.from_euler('x', theta_deg, degrees=True)
    colset = set(cols)
    for triad in (GYRO_IDX, ACC_IDX):
        if not colset & set(triad):
            continue
        v = sig9[:, triad]
        v_rot = rot.apply(v)
        # scope에 포함된 축만 교체 (ref1 같은 부분 scope 지원)
        for k, c in enumerate(triad):
            if c in colset:
                out[:, c] = v_rot[:, k]
    for c in EULER_IDX:
        if c in colset:
            out[:, c] = sig9[:, c] + theta_deg / 3
    return out


# ─────────── P4 드롭 ───────────
def mc_P4_drop(sig9, rate, cols, seed_key='P4'):
    """행 단위 패킷 손실 → 선형 보간 복원 (scope 채널만 손실)"""
    n = sig9.shape[0]
    rng = _seeded_rng((seed_key, rate, n))
    n_drop = int(n * rate)
    out = sig9.copy()
    if n_drop <= 0:
        return out
    drop_idx = rng.choice(n, size=n_drop, replace=False)
    for c in cols:
        s = out[:, c].astype(float)
        s[drop_idx] = np.nan
        out[:, c] = (pd.Series(s).interpolate(limit_direction='both')
                     .fillna(0).values)
    return out


# ─────────── P6 bias ───────────
def mc_P6_bias(sig9, bias, ref_channel, cols):
    out = sig9.copy()
    scales = channel_scales(sig9, ref_channel)
    for c in cols:
        out[:, c] = out[:, c] + bias * scales[c]
    return out


# ─────────── P7 다운샘플 (길이 보존) ───────────
def mc_P7_downsample(sig9, t, fs_new, cols):
    """다운샘플 → 원래 시간축으로 재보간 (입력 길이/fs 유지).

    ⚠ 시간축 정합성 (중요)
    ----------------------
    EBIMU 타임스탬프는 균일하지 않다. 실측에서 샘플 인덱스 기반
    균일 격자와 실제 t 사이에 **최대 345 ms** 편차가 존재한다
    (dt가 10 ms / 11 ms로 섞여 있고 분포가 한쪽에 쏠려 있음).
    ±150 ms 허용창보다 크므로, 신호를 '균일 샘플링'으로 간주하고
    resample_poly 후 인덱스로 시간을 되돌리면 검출 시각이 통째로
    어긋나 모든 레벨에서 F1이 붕괴한다.

    따라서 반드시 다음 순서로 처리한다.
      1) 실제 t → 공칭 fs 균일 격자로 보간          (시간축 정규화)
      2) 균일 격자에서 fs_new 로 안티앨리어싱 다운샘플
      3) fs_new 격자를 '같은 시간 구간'에 매핑
      4) 원래 t 로 다시 보간                         (길이·시간축 복원)

    딥러닝 윈도우 길이 = WINDOW_MS x fs 이므로 fs가 바뀌면 모델에
    넣을 수 없다. 실제 배포에서도 저레이트 센서를 기존 파이프라인에
    올릴 때 업샘플 보간이 들어가므로 현실적인 모델링이기도 하다.
    """
    out = sig9.copy()
    fs_nom = 1.0 / np.median(np.diff(t))
    if fs_new >= fs_nom:
        return out
    # 1) 균일 격자
    t_uni = np.arange(t[0], t[-1], 1.0 / fs_nom)
    if len(t_uni) < 10:
        return out
    frac = Fraction(int(round(fs_new * 100)),
                    int(round(fs_nom * 100))).limit_denominator(100)
    up, down = frac.numerator, frac.denominator
    if up == 0 or down == 0:
        return out
    for c in cols:
        s_uni = np.interp(t_uni, t, sig9[:, c])
        try:
            s_dn = resample_poly(s_uni, up, down)          # 2)
        except Exception:
            continue
        # 3) 같은 시간 구간에 균일 매핑
        t_dn = t_uni[0] + np.arange(len(s_dn)) * (down / up) / fs_nom
        out[:, c] = np.interp(t, t_dn, s_dn)               # 4)
    return out


# ─────────── dispatcher ───────────
def apply_perturbation_mc(pert_type, sig9, t, level,
                          ref_channel='Gyro_Z', scope='all9'):
    """9채널 교란 적용. sig9: (N,9) raw(정규화 이전). 반환도 (N,9)."""
    cols = scope_indices(scope, ref_channel)
    if pert_type == 'baseline':
        return sig9.copy()
    if pert_type == 'P1':
        return mc_P1_noise(sig9, level, ref_channel, cols)
    if pert_type == 'P2':
        return mc_P2_misalign(sig9, level, cols)
    if pert_type == 'P4':
        return mc_P4_drop(sig9, level, cols)
    if pert_type == 'P6':
        return mc_P6_bias(sig9, level, ref_channel, cols)
    if pert_type == 'P7':
        return mc_P7_downsample(sig9, t, level, cols)
    raise ValueError(f'Unknown perturbation: {pert_type}')


# ─────────── 정규화 (두 가지 모드) ───────────
def normalize(sig9, mode, clean_stats=None):
    """
    mode='adaptive' : 교란된 신호 자체 통계로 z-score
                      (현재 pipeline_phase3 과 동일한 동작)
    mode='frozen'   : clean 신호에서 계산한 통계를 그대로 사용
                      (공장 캘리브레이션 고정 → P6 bias 영향이 보존됨)

    P6(bias)에서 adaptive를 쓰면 오프셋이 정규화로 완전히 소거되어
    DR=1.00 이 나옴. 이는 모델의 강건성이 아니라 전처리 아티팩트이므로
    두 모드를 반드시 병기해야 함.
    """
    if mode == 'frozen':
        assert clean_stats is not None
        mu, sd = clean_stats
    else:
        mu = sig9.mean(axis=0, keepdims=True)
        sd = sig9.std(axis=0, keepdims=True) + 1e-6
    out = (sig9 - mu) / sd
    return np.nan_to_num(out).astype(np.float32)