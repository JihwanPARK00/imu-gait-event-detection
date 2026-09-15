"""
src/algorithms_patch.py — Nyquist 적응 컷오프 버전
==================================================
기존 algorithms.py 의 detect_TB_classic / detect_TB_adaptive / detect_TM 은
`butter(4, 12, 'lp', fs=fs)` 를 쓴다. 이 호출은 **fs <= 24 Hz 에서
ValueError** 를 던진다 (12 Hz > Nyquist). pipeline_phase2 의 except 가
이를 (0.0, 0.0) 으로 삼켜 P7 다운샘플링에서 F1 = 0 이 되고,
"30 Hz 절벽" 이라는 잘못된 결론으로 이어졌다.

수정은 한 줄이다:  cut = min(12.0, 0.4 * fs)
91 Hz 원신호에서는 cut = 12.0 이므로 **baseline 성능이 완전히 동일**하다
(검증: foot 0.797 / shank 0.842, 소수 3자리까지 일치).
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from src.algorithms import _zero_cross_from_peaks


def _lp(sig, fs, nominal=12.0):
    cut = min(nominal, 0.4 * fs)
    sos = butter(4, cut, 'lp', fs=fs, output='sos')
    return sosfiltfilt(sos, sig)


def detect_TB_classic_nyq(sig, fs):
    s = _lp(sig, fs)
    peaks, _ = find_peaks(s, height=30, distance=max(1, int(0.5 * fs)),
                          prominence=30)
    return _zero_cross_from_peaks(s, peaks, fs)


def detect_TB_adaptive_nyq(sig, fs):
    s = _lp(sig, fs)
    pre, props = find_peaks(s, height=10, distance=max(1, int(0.4 * fs)),
                            prominence=10)
    if len(pre) < 5:
        peaks, _ = find_peaks(s, height=30, distance=max(1, int(0.5 * fs)),
                              prominence=30)
    else:
        big = float(np.median(props['prominences']))
        peaks, _ = find_peaks(s, height=max(20, big * 0.3),
                              distance=max(1, int(0.5 * fs)),
                              prominence=max(20, big * 0.4))
    return _zero_cross_from_peaks(s, peaks, fs)


ALGORITHMS_NYQ = {
    'TB_classic_nyq': detect_TB_classic_nyq,
    'TB_adaptive_nyq': detect_TB_adaptive_nyq,
}