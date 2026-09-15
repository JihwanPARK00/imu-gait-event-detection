"""
src/visualize_phase2.py — 2단계 그림 생성
==========================================
cell_results_phase2_dr.csv 읽어서 그림 생성.

생성 그림:
  Fig P1. dr_curves_{HS,TO}.png      — 교란×알고리즘 DR 곡선 (5 패널)
  Fig P2. f1_curves_{HS,TO}.png      — 절대 F1 곡선 (DR 비교용)
  Fig P3. dr_heatmap_{HS,TO}.png     — 셀×교란 DR 평균 히트맵
  Fig P4. auc_bars.png               — AUC 알고리즘 비교
  콘솔: 강건성 ranking
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, FIGURES_DIR, ALGO_COLORS
from src.perturbations import PERTURBATIONS, COMPOUND_PERTURBATIONS, LEVEL_LABELS

for font in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']:
    try:
        rcParams['font.family'] = font
        break
    except Exception:
        pass
rcParams['axes.unicode_minus'] = False

PHASE2_ALGO_ORDER = ['TB_classic', 'BPF', 'TM']
PERT_ORDER = ['P1', 'P2', 'P4', 'P6', 'P7']
COMPOUND_ORDER = ['C1', 'C2', 'C3', 'C4']


def load_results():
    csv = RESULTS_DIR / 'cell_results_phase2_dr.csv'
    if not csv.exists():
        raise FileNotFoundError(
            f'{csv} 없음. 먼저 pipeline_phase2.py를 돌리세요.')
    return pd.read_csv(csv)


# ═════════ Fig P5: 복합 교란 DR 곡선 ═════════
def fig_compound_dr(df, out_dir):
    df_c = df[df['pert_type'].isin(COMPOUND_ORDER)]
    if len(df_c) == 0:
        return
    for event in ('HS', 'TO'):
        col = f'DR_{event}'
        fig, axes = plt.subplots(1, len(COMPOUND_ORDER),
                                 figsize=(4 * len(COMPOUND_ORDER), 4.5),
                                 sharey=True)
        for ax, pert in zip(axes, COMPOUND_ORDER):
            info = COMPOUND_PERTURBATIONS[pert]
            for algo in PHASE2_ALGO_ORDER:
                sub = df_c[(df_c['pert_type'] == pert) &
                           (df_c['algorithm'] == algo)]
                if len(sub) == 0:
                    continue
                levels = sorted(sub['pert_level'].unique())
                means = [sub[sub['pert_level'] == lv][col].mean()
                         for lv in levels]
                stds = [sub[sub['pert_level'] == lv][col].std()
                        for lv in levels]
                ax.errorbar(levels, means, yerr=stds,
                            marker='o', lw=2, ms=8, capsize=3,
                            label=algo, color=ALGO_COLORS.get(algo))
            ax.set_xticks(range(len(LEVEL_LABELS)))
            ax.set_xticklabels(LEVEL_LABELS)
            ax.set_xlabel(f'{pert}: {info["desc"]}', fontsize=9)
            ax.set_title(f'{pert} {info["name"]}', fontweight='bold')
            ax.set_ylim(0, 1.2)
            ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.6)
            ax.grid(alpha=0.3)
            ax.legend(loc='lower left', fontsize=8)
        axes[0].set_ylabel(f'DR — {event}')
        fig.suptitle(f'{event} — 복합 교란 강건성', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(out_dir / f'compound_dr_{event}.png',
                    dpi=140, bbox_inches='tight')
        plt.close(fig)


# ═════════ Fig P6: 단일 vs 복합 비교 ═════════
def fig_single_vs_compound(df, out_dir):
    """C1 medium의 DR vs (P1 medium × P4 medium) 곱 비교.

    교란 효과가 곱셈인지 덧셈인지 확인.
    Medium 레벨만 사용 (대표값).
    """
    # 컴포넌트 단일 교란 레벨 (Medium = C 복합의 두번째 = idx 1)
    single_levels = {
        'P1': 20, 'P2': 15, 'P4': 0.10,
    }
    pairs = []
    for cpert, info in COMPOUND_PERTURBATIONS.items():
        components = info['components']
        if all(c in single_levels for c in components):
            pairs.append((cpert, components,
                          info['levels'][1]))  # Medium tuple

    for event in ('HS', 'TO'):
        col = f'DR_{event}'
        fig, ax = plt.subplots(figsize=(11, 5.5))
        x = np.arange(len(pairs))
        w = 0.22

        for i, algo in enumerate(PHASE2_ALGO_ORDER):
            multiplied = []  # 단일 DR의 곱
            actual = []      # 실제 복합 DR
            for cpert, comps, lv_tuple in pairs:
                # 단일 DR (medium)
                drs = []
                for c, lv in zip(comps, lv_tuple):
                    s = df[(df['pert_type'] == c) &
                           (df['pert_level'] == single_levels[c]) &
                           (df['algorithm'] == algo)]
                    drs.append(s[col].mean() if len(s) else np.nan)
                mult = np.nanprod(drs)
                multiplied.append(mult)
                # 복합 DR
                s_c = df[(df['pert_type'] == cpert) &
                         (df['pert_level'] == 1) &
                         (df['algorithm'] == algo)]
                actual.append(s_c[col].mean() if len(s_c) else np.nan)

            offset = (i - 1) * w
            color = ALGO_COLORS.get(algo, 'gray')
            # 단일 곱 (속이 빈 막대)
            ax.bar(x + offset - w / 4, multiplied, w / 2,
                   label=f'{algo} (단일 곱)',
                   facecolor='none', edgecolor=color, hatch='//',
                   linewidth=1.5)
            # 복합 실제 (속이 찬 막대)
            ax.bar(x + offset + w / 4, actual, w / 2,
                   label=f'{algo} (복합 실제)',
                   color=color)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{p[0]}\n{"+".join(p[1])}' for p in pairs])
        ax.set_ylabel(f'DR — {event}')
        ax.set_ylim(0, 1.3)
        ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.6)
        ax.set_title(
            f'{event} — 단일 교란 곱 vs 복합 실제 (Medium 레벨)',
            fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=3)
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f'single_vs_compound_{event}.png',
                    dpi=140, bbox_inches='tight')
        plt.close(fig)


# ═════════ Fig P1: DR 곡선 (5 교란 × 알고리즘) ═════════
def fig_dr_curves(df, out_dir):
    for event in ('HS', 'TO'):
        col = f'DR_{event}'
        fig, axes = plt.subplots(1, len(PERT_ORDER),
                                 figsize=(4 * len(PERT_ORDER), 4.5),
                                 sharey=True)
        for ax, pert in zip(axes, PERT_ORDER):
            info = PERTURBATIONS[pert]
            for algo in PHASE2_ALGO_ORDER:
                sub = df[(df['pert_type'] == pert) &
                         (df['algorithm'] == algo)]
                if len(sub) == 0:
                    continue
                levels = sorted(sub['pert_level'].unique())
                means = []
                stds = []
                for lv in levels:
                    ss = sub[sub['pert_level'] == lv][col].dropna()
                    means.append(ss.mean() if len(ss) else np.nan)
                    stds.append(ss.std() if len(ss) else np.nan)
                means = np.array(means)
                stds = np.array(stds)
                ax.errorbar(levels, means, yerr=stds,
                            marker='o', lw=2, ms=7, capsize=3,
                            label=algo, color=ALGO_COLORS.get(algo))
            ax.set_xlabel(f'{pert}: {info["desc"]} ({info["unit"]})',
                          fontsize=10)
            ax.set_title(f'{pert} {info["name"]}',
                         fontweight='bold')
            ax.set_ylim(0, 1.2)
            ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.6)
            ax.grid(alpha=0.3)
            if pert == 'P4':
                ax.set_xscale('linear')
            ax.legend(loc='lower left', fontsize=8)
        axes[0].set_ylabel(f'DR (F1_pert / F1_base) — {event}')
        fig.suptitle(f'{event} — 교란 강건성 (DR = 1.0이 손실 없음)',
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(out_dir / f'dr_curves_{event}.png',
                    dpi=140, bbox_inches='tight')
        plt.close(fig)


# ═════════ Fig P2: 절대 F1 곡선 (DR 보조) ═════════
def fig_f1_curves(df, out_dir):
    for event in ('HS', 'TO'):
        col = f'F1_{event}'
        fig, axes = plt.subplots(1, len(PERT_ORDER),
                                 figsize=(4 * len(PERT_ORDER), 4.5),
                                 sharey=True)
        for ax, pert in zip(axes, PERT_ORDER):
            info = PERTURBATIONS[pert]
            for algo in PHASE2_ALGO_ORDER:
                sub = df[(df['pert_type'] == pert) &
                         (df['algorithm'] == algo)]
                if len(sub) == 0:
                    continue
                levels = sorted(sub['pert_level'].unique())
                means = [sub[sub['pert_level'] == lv][col].mean()
                         for lv in levels]
                stds = [sub[sub['pert_level'] == lv][col].std()
                        for lv in levels]
                ax.errorbar(levels, means, yerr=stds,
                            marker='o', lw=2, ms=7, capsize=3,
                            label=algo, color=ALGO_COLORS.get(algo))
            # baseline 점선 (4명·셀 평균)
            base = (df[df['pert_type'] == 'baseline']
                    .groupby('algorithm')[col].mean())
            for algo in PHASE2_ALGO_ORDER:
                if algo in base.index:
                    ax.axhline(base[algo], color=ALGO_COLORS.get(algo),
                               ls=':', lw=0.8, alpha=0.5)
            ax.set_xlabel(f'{pert}: {info["desc"]} ({info["unit"]})',
                          fontsize=10)
            ax.set_title(f'{pert} {info["name"]}', fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.grid(alpha=0.3)
            ax.legend(loc='lower left', fontsize=8)
        axes[0].set_ylabel(f'F1 — {event}')
        fig.suptitle(
            f'{event} — 절대 F1 (점선 = baseline, ±150ms)', fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(out_dir / f'f1_curves_{event}.png',
                    dpi=140, bbox_inches='tight')
        plt.close(fig)


# ═════════ Fig P3: 셀×교란 DR 히트맵 ═════════
def fig_dr_heatmap(df, out_dir):
    for event in ('HS', 'TO'):
        col = f'DR_{event}'
        # 셀×교란 평균 DR (모든 레벨 평균)
        pivot = (df[df['pert_type'] != 'baseline']
                 .groupby(['sensor', 'channel', 'algorithm', 'pert_type'])
                 [col].mean()
                 .unstack('pert_type'))
        # 알고리즘별 정렬
        pivot = pivot.reindex(columns=PERT_ORDER)
        # row index: 알고리즘 → 위치 → 채널
        pivot = pivot.reset_index()
        pivot['row_label'] = (pivot['algorithm'] + ' | '
                              + pivot['sensor'] + ' '
                              + pivot['channel'])
        pivot = pivot.sort_values(['algorithm', 'sensor', 'channel'])
        mat = pivot[PERT_ORDER].values
        labels = pivot['row_label'].values

        fig, ax = plt.subplots(
            figsize=(7, max(5, 0.3 * len(labels))))
        im = ax.imshow(mat, cmap='RdYlGn', vmin=0, vmax=1.2,
                       aspect='auto')
        ax.set_xticks(range(len(PERT_ORDER)))
        ax.set_xticklabels(PERT_ORDER)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.2f}', ha='center',
                            va='center', fontsize=6,
                            color='white' if (v < 0.4 or v > 1.1)
                                  else 'black')
        fig.colorbar(im, ax=ax, label='DR (평균)', fraction=0.04)
        ax.set_title(f'{event} — 셀×교란 평균 DR',
                     fontsize=12, fontweight='bold')
        fig.tight_layout()
        fig.savefig(out_dir / f'dr_heatmap_{event}.png',
                    dpi=140, bbox_inches='tight')
        plt.close(fig)


# ═════════ Fig P4: AUC 막대 ═════════
def fig_auc_bars(out_dir):
    auc_csv = RESULTS_DIR / 'phase2_auc.csv'
    if not auc_csv.exists():
        return
    auc = pd.read_csv(auc_csv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, event in zip(axes, ['HS', 'TO']):
        sub = auc[auc['event'] == event]
        pivot = sub.pivot(index='pert_type', columns='algorithm',
                          values='AUC_DR').reindex(PERT_ORDER)
        pivot = pivot[PHASE2_ALGO_ORDER]
        x = np.arange(len(PERT_ORDER))
        w = 0.25
        for i, algo in enumerate(PHASE2_ALGO_ORDER):
            offset = (i - 1) * w
            bars = ax.bar(x + offset, pivot[algo], w,
                          label=algo, color=ALGO_COLORS.get(algo))
            for b, v in zip(bars, pivot[algo]):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                            f'{v:.2f}', ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(PERT_ORDER)
        ax.set_ylim(0, 1.3)
        ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.6,
                   label='완전 강건 (DR=1.0)')
        ax.set_ylabel(f'AUC of DR — {event}')
        ax.set_title(f'{event} — 알고리즘별 강건성 종합',
                     fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    fig.suptitle('AUC (DR 곡선 아래 면적) — 값이 클수록 강건',
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / 'auc_bars.png',
                dpi=140, bbox_inches='tight')
    plt.close(fig)


# ═════════ 콘솔: 강건성 ranking ═════════
def print_robustness_ranking():
    auc_csv = RESULTS_DIR / 'phase2_auc.csv'
    if not auc_csv.exists():
        return
    auc = pd.read_csv(auc_csv)
    for event in ('HS', 'TO'):
        sub = auc[auc['event'] == event]
        # 알고리즘별 평균 AUC (5 교란)
        rank = (sub.groupby('algorithm')['AUC_DR']
                .mean().sort_values(ascending=False))
        print(f'\n━━━━━ {event} 강건성 종합 ranking (평균 AUC) ━━━━━')
        for i, (algo, auc_val) in enumerate(rank.items(), 1):
            print(f'  {i}. {algo:<14} AUC = {auc_val:.3f}')


def main():
    print(f'결과 폴더: {RESULTS_DIR}')
    df = load_results()
    print(f'cell_results_phase2_dr.csv: {len(df)}행\n')

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print('Fig P1: DR 곡선...')
    fig_dr_curves(df, FIGURES_DIR)

    print('Fig P2: F1 절대값 곡선...')
    fig_f1_curves(df, FIGURES_DIR)

    print('Fig P3: 셀×교란 DR 히트맵...')
    fig_dr_heatmap(df, FIGURES_DIR)

    print('Fig P4: AUC 막대...')
    fig_auc_bars(FIGURES_DIR)

    # 복합 교란 (있을 때만)
    has_compound = df['pert_type'].isin(COMPOUND_ORDER).any()
    if has_compound:
        print('Fig P5: 복합 교란 DR...')
        fig_compound_dr(df, FIGURES_DIR)
        print('Fig P6: 단일 vs 복합 비교...')
        fig_single_vs_compound(df, FIGURES_DIR)

    print_robustness_ranking()

    print(f'\n[완료] 그림 저장: {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
