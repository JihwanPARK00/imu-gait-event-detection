"""
src/trials_adapter.py — 실행 환경 자동 감지
============================================
local PC:  xlsx 가 피험자별로 통합 → src.pipeline.build_trial_list() 사용
컨테이너:   xlsx 가 속도별로 분리 + 피험자 표기 소문자 혼재
            → src.local_trials.build_trial_list_local() 사용

모든 스크립트는 이 함수만 호출하면 되므로, PC/컨테이너 어느 쪽에서도
코드를 고칠 필요가 없다.
"""


def get_trials(verbose=True):
    trials = None
    try:
        from src.pipeline import build_trial_list
        trials = build_trial_list()
        if trials and len(trials) > 0:
            if verbose:
                print(f'[trials] src.pipeline.build_trial_list() → {len(trials)}개')
            return trials
    except Exception as e:
        if verbose:
            print(f'[trials] build_trial_list() 실패 ({e}) → 로컬 어댑터로 전환')
    from src.local_trials import build_trial_list_local
    trials, missing = build_trial_list_local(verbose=verbose)
    return trials