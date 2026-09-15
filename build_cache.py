"""build_cache.py — raw 9채널 npz 캐시 생성 (최초 1회, 가장 오래 걸림)

xlsx 로딩이 trial당 ~11초라 매번 읽으면 시간이 낭비된다.
5개 위치 x 32 trial = 160개 npz 를 만들어 두면 이후 전부 즉시 로드된다.

  python build_cache.py                 # 5개 위치 전부 (~25분)
  python build_cache.py shank foot      # 딥러닝 실험만 필요하면 2개 (~10분)
"""
import sys, time
sys.path.insert(0, '.')
from src.trials_adapter import get_trials
from src.pipeline_phase4 import build_raw_trials

POS = sys.argv[1:] or ['foot', 'shank', 'thigh_aff', 'thigh_sound', 'lumbar']
trials = get_trials()
for p in POS:
    t0 = time.time()
    r = build_raw_trials(trials, p)
    print(f'{p:<14} {len(r):>3} trial cached ({time.time()-t0:.0f}s)', flush=True)