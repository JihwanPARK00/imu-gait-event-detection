"""
src/run.py — 한 번 실행으로 전체 분석
=======================================
사용법 (ICCAS/ 폴더에서):
    python -m src.run                    # 1단계
    python -m src.run --phase 2          # 2단계 (교란)
    python -m src.run --phase 3          # 3단계 (CNN)
    python -m src.run --phase 1+2        # 1+2
    python -m src.run --no-figures       # 격자만
    python -m src.run --figures-only     # 그림만
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_phase1(no_fig, fig_only):
    if not fig_only:
        print('━━━━━━━━━━━━ Phase 1: 격자 + 통계 ━━━━━━━━━━━━')
        from src.pipeline import main as run_pipeline
        run_pipeline()
        print()
    if not no_fig:
        print('━━━━━━━━━━━━ Phase 1: 그림 ━━━━━━━━━━━━')
        from src.visualize import main as run_visualize
        run_visualize()


def run_phase2(no_fig, fig_only):
    if not fig_only:
        print('━━━━━━━━━━━━ Phase 2: 교란 격자 ━━━━━━━━━━━━')
        from src.pipeline_phase2 import main as run_pipeline2
        run_pipeline2()
        print()
    if not no_fig:
        print('━━━━━━━━━━━━ Phase 2: 그림 ━━━━━━━━━━━━')
        from src.visualize_phase2 import main as run_visualize2
        run_visualize2()


def run_phase3():
    print('━━━━━━━━━━━━ Phase 3: CNN (1D, LOSO) ━━━━━━━━━━━━')
    from src.pipeline_phase3 import main as run_pipeline3
    run_pipeline3()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['1', '2', '3', '1+2', '1+2+3'],
                    default='1', help='실행 단계 (기본: 1)')
    ap.add_argument('--no-figures', action='store_true',
                    help='격자만, 그림 X')
    ap.add_argument('--figures-only', action='store_true',
                    help='격자 건너뛰고 그림만')
    args = ap.parse_args()

    if '1' in args.phase.split('+'):
        run_phase1(args.no_figures, args.figures_only)
        if '+' in args.phase:
            print()

    if '2' in args.phase.split('+'):
        run_phase2(args.no_figures, args.figures_only)
        if args.phase.endswith('+3'):
            print()

    if '3' in args.phase.split('+'):
        run_phase3()


if __name__ == '__main__':
    main()
