"""
apply_manual_offsets.py — 눈으로 맞춘 offset을 synced 파일에 적용
================================================================
manual_sync.py로 만든 manual_offsets.csv를 읽어,
각 trial의 synced.csv를 manual_offset으로 재생성한다.
(원본 synced는 _auto 백업으로 보관)

실행: python apply_manual_offsets.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root
import pandas as pd
from pathlib import Path

SYNC_DIR = 'output/synced'
MANUAL_CSV = 'output/synced/manual_offsets.csv'

def main():
    man = pd.read_csv(MANUAL_CSV).set_index('trial')
    print(f'{len(man)} trial 보정 적용')
    for trial, row in man.iterrows():
        sp = Path(SYNC_DIR) / f'{trial}_synced.csv'
        if not sp.is_file():
            print(f'  ⚠ {sp.name} 없음'); continue
        s = pd.read_csv(sp)
        # 백업
        bak = Path(SYNC_DIR) / f'{trial}_synced_auto.csv'
        if not bak.is_file():
            s.to_csv(bak, index=False, encoding='utf-8-sig')
        # manual offset으로 imu_time 재계산
        new_off = row['manual_offset']
        s['imu_time'] = s['time_sec'] + new_off
        # 분석 윈도우(in_window)는 CLAP2 기준이었으니 유지하되 imu_time 갱신됐으니 재계산
        # 간단히: 기존 in_window True 구간의 새 imu_time 범위로
        if 'in_window' in s.columns and s['in_window'].any():
            lo = s.loc[s['in_window'], 'imu_time'].min()
            hi = s.loc[s['in_window'], 'imu_time'].max()
            s['in_window'] = (s['imu_time'] >= lo) & (s['imu_time'] <= hi)
        s.to_csv(sp, index=False, encoding='utf-8-sig')
        print(f'  {trial}: offset {row["auto_offset"]:+.3f} → {new_off:+.3f} (Δ{row["delta_ms"]:+.0f}ms)')
    print('완료. grid_baseline.py 다시 실행하세요.')

if __name__ == '__main__':
    main()