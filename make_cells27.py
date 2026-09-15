"""make_cells27.py — 27개 셀 목록과 sign_map 추출

phase2_corrected.py 가 필요로 하는 두 파일을 기존 결과에서 만든다.
  results/cells27.csv    : sensor, channel, algorithm
  results/sign_map27.csv : + sign (phase1 mode sign)

입력: results/cell_results_phase2_dr.csv (기존 2단계 산출물)
"""
import sys
from pathlib import Path
import pandas as pd

src = Path(sys.argv[1] if len(sys.argv) > 1
           else 'results/cell_results_phase2_dr.csv')
d = pd.read_csv(src)
c = (d.groupby(['sensor', 'channel', 'algorithm']).size()
     .reset_index()[['sensor', 'channel', 'algorithm']])
s = (d.groupby(['sensor', 'channel', 'algorithm'])['sign']
     .agg(lambda x: x.mode().iloc[0]).reset_index())
Path('results').mkdir(exist_ok=True)
c.to_csv('results/cells27.csv', index=False)
s.to_csv('results/sign_map27.csv', index=False)
print(f'셀 {len(c)}개 / sign +1:{(s["sign"]==1).sum()} -1:{(s["sign"]==-1).sum()}')