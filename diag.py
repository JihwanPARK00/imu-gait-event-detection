"""diag.py — baseline F1 저하 원인 진단

증상: 전통 알고리즘 baseline F1 이 논문 Table 1(0.85 내외) 대비
      0.54~0.60 으로 균일하게 낮음. 모든 알고리즘·위치에서 동일하게 발생.

이 스크립트는 신호와 GT 중 어느 쪽이 어긋났는지 좁힌다.
실행: python diag.py
"""
import sys, glob, hashlib
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

print('=' * 62)
print('1. 캐시 상태')
print('=' * 62)
for pos in ['shank', 'foot', 'thigh_aff', 'lumbar']:
    n = len(glob.glob(f'results/raw_cache/*_{pos}.npz'))
    print(f'  {pos:<12} {n:>3} 개  {"OK" if n == 32 else "<<< 32개가 아님"}')
for f in sorted(glob.glob('results/p2c_*.csv')):
    import os
    print(f'  {f:<32} {os.path.getsize(f):>10,} bytes')

print()
print('=' * 62)
print('2. imu_loader.py 버전 확인')
print('=' * 62)
h = hashlib.md5(open('imu_loader.py', 'rb').read()).hexdigest()
print(f'  md5 = {h}')
print('  (제 컨테이너: 6560 bytes)')
import os
print(f'  크기 = {os.path.getsize("imu_loader.py")} bytes')

print()
print('=' * 62)
print('3. 캐시 1개 상세 — 신호와 GT 범위')
print('=' * 62)
f = sorted(glob.glob('results/raw_cache/*_shank.npz'))[0]
z = np.load(f)
t, sig9, gHS, gTO = z['t'], z['sig9'], z['gt_HS'], z['gt_TO']
print(f'  파일       {f}')
print(f'  sig9       {sig9.shape}   (기대: (약 18000, 9))')
print(f'  t 범위     {t[0]:.3f} ~ {t[-1]:.3f} s   fs={1/np.median(np.diff(t)):.2f} Hz')
print(f'  GT HS      {len(gHS)}개, {gHS.min():.2f} ~ {gHS.max():.2f} s')
print(f'  GT TO      {len(gTO)}개, {gTO.min():.2f} ~ {gTO.max():.2f} s')
print(f'  Gyro_Z std {sig9[:, 2].std():.1f} deg/s   (기대: 90~110)')
if gHS.max() > t[-1] or gHS.min() < t[0]:
    print('  <<< GT 가 신호 시간 범위를 벗어남 — 페어링 오류 가능성')

print()
print('=' * 62)
print('4. 검출 시각과 GT 사이의 계통 오프셋 (핵심 진단)')
print('=' * 62)
from src.algorithms import ALGORITHMS
from src.pipeline import match_F1
fs = 1 / np.median(np.diff(t))
best = None
for sgn in (1, -1):
    HS_i, _ = ALGORITHMS['BPF'](sig9[:, 2] * sgn, fs)
    if len(HS_i) == 0:
        continue
    det = t[HS_i]
    d = np.array([gHS[np.argmin(np.abs(gHS - x))] - x for x in det])
    f1 = match_F1(det, gHS, 0.15)
    print(f'  sign={sgn:+d}  검출 {len(det):>3}개 / GT {len(gHS)}개  '
          f'F1={f1:.3f}  offset 중앙값={np.median(d)*1000:+.0f} ms  '
          f'IQR={np.percentile(d,75)*1000-np.percentile(d,25)*1000:.0f} ms')
    if best is None or f1 > best[0]:
        best = (f1, np.median(d) * 1000)
print(f'\n  → 최선 F1 {best[0]:.3f} (기대 0.85 내외)')
print(f'  → 계통 오프셋 {best[1]:+.0f} ms')
if abs(best[1]) > 40:
    print('  <<< 오프셋이 큼: GT 동기화(imu_time) 문제')
elif best[0] < 0.7:
    print('  <<< 오프셋은 작은데 F1 낮음: GT-신호 페어링(trial 짝) 문제 의심')

print()
print('=' * 62)
print('5. trial 페어링 확인 — sheet 와 csv 가 같은 시행인지')
print('=' * 62)
from src.trials_adapter import get_trials
tr = get_trials(verbose=False)
for x in tr[:6]:
    gt = pd.read_csv(x['csv'])
    gt = gt[gt['in_window'] == True]
    print(f'  {x["trial_id"]:<20} sheet={x["sheet"]:<20} '
          f'csv={x["csv"].split(chr(92))[-1].split("/")[-1]:<28} '
          f'HS={len(gt[gt["event"]=="HS"]):>3}')