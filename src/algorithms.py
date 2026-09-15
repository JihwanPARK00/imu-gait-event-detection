"""
src/algorithms.py — 4 보행 검출 알고리즘
==========================================
- TB_classic   : 기존 논문 (LP 12Hz + peak prom=30)
- TB_adaptive  : 개선안 (prom 신호 진폭에 적응) — KKH 이중봉우리 대응
- BPF          : 0.5–3Hz bandpass
- TM           : self-bootstrap 정규화 상호상관 (3km/h 템플릿 X)

검출 후처리는 공통: peak 직전 -→+ ZC = TO, peak 직후 +→- ZC = HS
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks


def _zero_cross_from_peaks(s, peaks, fs, win_sec=0.5):
    """peak 기준 zero-crossing → HS/TO 인덱스"""
    win = int(win_sec * fs)
    HS, TO = [], []
    for p in peaks:
        # TO: peak 직전 -→+ 교차 (마지막)
        a = max(0, p - win)
        seg_pre = s[a:p]
        if len(seg_pre) > 1:
            zc = np.where(np.diff(np.sign(seg_pre)) > 0)[0]
            if len(zc):
                TO.append(a + zc[-1])
        # HS: peak 직후 +→- 교차 (첫)
        b = min(len(s), p + win)
        seg_post = s[p:b]
        if len(seg_post) > 1:
            zc = np.where(np.diff(np.sign(seg_post)) < 0)[0]
            if len(zc):
                HS.append(p + zc[0])
    return np.array(HS, dtype=int), np.array(TO, dtype=int)


def detect_TB_classic(sig, fs):
    """기존 논문 TB"""
    sos = butter(4, 12, 'lp', fs=fs, output='sos')
    s = sosfiltfilt(sos, sig)
    peaks, _ = find_peaks(s, height=30, distance=int(0.5 * fs),
                          prominence=30)
    return _zero_cross_from_peaks(s, peaks, fs)


def detect_TB_adaptive(sig, fs):
    """prom을 신호 진폭(peak prominence 중앙값 × 40%)에 자동 조정"""
    sos = butter(4, 12, 'lp', fs=fs, output='sos')
    s = sosfiltfilt(sos, sig)
    pre_peaks, pre_props = find_peaks(s, height=10,
                                       distance=int(0.4 * fs),
                                       prominence=10)
    if len(pre_peaks) < 5:
        peaks, _ = find_peaks(s, height=30, distance=int(0.5 * fs),
                              prominence=30)
    else:
        big = float(np.median(pre_props['prominences']))
        new_prom = max(20, big * 0.4)
        new_height = max(20, big * 0.3)
        peaks, _ = find_peaks(s, height=new_height,
                              distance=int(0.5 * fs),
                              prominence=new_prom)
    return _zero_cross_from_peaks(s, peaks, fs)


def detect_BPF(sig, fs):
    """0.5–3Hz bandpass + peak/zero-crossing"""
    nyq = fs / 2
    if 3 >= nyq:
        sos = butter(4, 0.5, 'hp', fs=fs, output='sos')
    else:
        sos = butter(4, [0.5, 3], 'bp', fs=fs, output='sos')
    s = sosfiltfilt(sos, sig)
    amp = float(np.std(s))
    h = max(5, amp * 1.0)
    peaks, _ = find_peaks(s, height=h, distance=int(0.5 * fs),
                          prominence=h * 0.5)
    return _zero_cross_from_peaks(s, peaks, fs)


def detect_TM(sig, fs):
    """Self-bootstrap 정규화 상호상관 (Template Matching).

    ─ 시나리오 ─
    같은 trial 안의 첫 5 stride로 200ms 템플릿을 만든다 → 단일 사용자
    적응(user-adaptive) 시나리오. 보조 로봇 등이 켜진 직후 사용자 자신의
    초기 stride로 calibration 한 뒤 그 사용자에게 적용하는 방식과 일치.

    ─ 기존 논문(Park et al. 2025)과의 차이 ─
    기존 논문은 Leave-One-Subject-Out (LOSO): 다른 N-1명으로 템플릿 학습 후
    1명에 테스트. 우리는 LOSO 채택 안 함. 이유:
      (1) 4명 데이터로 LOSO는 표본 크기 제약 (3명으로 학습).
      (2) 4명의 보행 특성 변동성 큼 (KKH 이중봉우리, YHS 5–12Hz 진동) →
          타인 평균 템플릿이 잘 맞지 않을 가능성.
      (3) TB/BPF는 같은 trial 내 정보만 쓰는데 TM만 cross-subject 외부
          학습을 받으면 비교가 공정하지 않음 (unfair advantage).
      → "단일 사용자 적응" 시나리오가 우리 데이터에서 더 정직한 평가이며,
        실제 사용 환경(개인 보정)에도 부합.
    """
    sos = butter(4, 12, 'lp', fs=fs, output='sos')
    s = sosfiltfilt(sos, sig)
    init_peaks, _ = find_peaks(s, height=30, distance=int(0.5 * fs),
                                prominence=30)
    if len(init_peaks) < 5:
        return np.array([], dtype=int), np.array([], dtype=int)
    half = int(0.1 * fs)  # 템플릿 200ms
    segs = []
    for p in init_peaks[:5]:
        if p - half >= 0 and p + half < len(s):
            segs.append(s[p - half:p + half])
    if not segs:
        return np.array([], dtype=int), np.array([], dtype=int)
    template = np.mean(segs, axis=0)
    template = (template - template.mean()) / (template.std() + 1e-9)
    n = len(template)
    s_pad = pd.Series(s)
    win_std = s_pad.rolling(n).std().values
    corr_raw = np.correlate(s, template, mode='valid')
    denom = n * win_std[n - 1:]
    denom = np.where(denom < 1e-6, np.inf, denom)
    corr = corr_raw / denom
    corr_peaks, _ = find_peaks(corr, height=0.4,
                                distance=int(0.5 * fs),
                                prominence=0.2)
    peaks = corr_peaks + half
    peaks = peaks[(peaks >= 0) & (peaks < len(s))]
    return _zero_cross_from_peaks(s, peaks, fs)


ALGORITHMS = {
    'TB_classic': detect_TB_classic,
    'TB_adaptive': detect_TB_adaptive,
    'BPF': detect_BPF,
    'TM': detect_TM,
}
