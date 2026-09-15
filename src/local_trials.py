"""
src/local_trials.py — 컨테이너 환경용 trial 목록 빌더
=====================================================
local PC와 달리 이 컨테이너의 xlsx는 속도별로 분리되어 있고
(260519_PJH_IMU_2km.xlsx 등), synced csv의 피험자 표기는
소문자가 섞여 있음 (260522_kKH_...). 이 어댑터가 그 차이를 흡수함.

PC에서 실행할 때는 src.pipeline.build_trial_list() 를 그대로 쓰면 됨.
"""
import re
from pathlib import Path
import openpyxl

ROOT = Path(__file__).resolve().parent.parent
IMU_DIR = ROOT / 'imu'
GT_DIR = ROOT / 'output' / 'synced'

SUBJECT_GENDER = {'PJH': 'M', 'KKH': 'M', 'LNH': 'F', 'YHS': 'F'}


def _sheet_index():
    """{정규화된 시트명(대문자): (xlsx경로, 실제 시트명)}"""
    idx = {}
    for f in sorted(IMU_DIR.glob('*.xlsx')):
        wb = openpyxl.load_workbook(f, read_only=True)
        for sh in wb.sheetnames:
            idx[sh.upper()] = (str(f), sh)
        wb.close()
    return idx


def build_trial_list_local(verbose=True):
    sheets = _sheet_index()
    trials, missing = [], []
    for csv in sorted(GT_DIR.glob('*_synced.csv')):
        stem = csv.stem.replace('_synced', '')
        m = re.match(r'(\d{6})_(\w+?)_(\d+)(?:_(\d+))?$', stem)
        if not m:
            continue
        date, subj, speed, rep = m.groups()
        subj = subj.upper()
        if subj not in SUBJECT_GENDER:
            continue
        key = stem.upper()
        if key not in sheets:
            missing.append(stem)
            continue
        xlsx, sheet = sheets[key]
        trials.append({
            'subject': subj,
            'speed': int(speed),
            'rep': rep or '1',
            'gender': SUBJECT_GENDER[subj],
            'xlsx': xlsx,
            'sheet': sheet,
            'csv': str(csv),
            'trial_id': stem.upper(),
        })
    if verbose:
        print(f'[trial] 매칭 {len(trials)}개, 누락 {len(missing)}개')
        if missing:
            print(f'        누락 목록: {missing}')
    return trials, missing


if __name__ == '__main__':
    tr, miss = build_trial_list_local()
    import collections
    c = collections.Counter((t['subject'], t['speed']) for t in tr)
    for k in sorted(c):
        print(f'  {k[0]} {k[1]}km : {c[k]} trial')