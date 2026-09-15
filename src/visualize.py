"""
src/visualize.py — 모든 그림 생성
==================================
cell_results.csv 읽어서 논문급 그림 생성.
격자 재실행 없이 그림만 빨리 만들 수 있음.

생성 그림:
  Fig 1. heatmap_avg_{HS,TO}.png       — 4명 평균, 알고리즘 × 센서·채널
  Fig 2. heatmap_{algo}_{HS,TO}.png    — 피험자별 (8장)
  Fig 3. algo_compare.png              — 알고리즘 비교 (foot Gyro_Y)
  Fig 4. gender_{HS,TO}.png            — 성별 비교 (top 6 조합)
  Fig 5. speed_{HS,TO}.png             — 속도별 라인 (top 4 조합)
  콘솔: Top-10 조합표
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

from src.config import (
    SENSOR_ORDER, CHANNELS, SUBJECT_ORDER, SUBJECT_GENDER,
    SUBJECT_COLORS, GENDER_COLORS, ALGO_COLORS,
    SPEEDS, RESULTS_DIR, FIGURES_DIR, TOL_MAIN_MS,
)
from src.algorithms import ALGORITHMS

for font in ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']:
    try:
        rcParams['font.family'] = font
        break
    except Exception:
        pass
rcParams['axes.unicode_minus'] = False

ALGO_ORDER = list(ALGORITHMS.keys())


def load_results():
    csv = RESULTS_DIR / 'cell_results.csv'
    if not csv.exists():
        raise FileNotFoundError(
            f'{csv} 없음. 먼저 pipeline.py를 돌려 격자를 생성하세요.')
    return pd.read_csv(csv)


# ═════════ Fig 1: 4명 평균 히트맵 ═════════
def fig_heatmap_avg(df, out_dir):
    for event in ('HS', 'TO'):
        col = f'F1_{event}_150'
        fig, axes = plt.subplots(1, len(ALGO_ORDER),
                                 figsize=(5 * len(ALGO_ORDER), 4.5),
                                 sharey=True)
        for ax, algo in zip(axes, ALGO_ORDER):
            sub = df[df['algorithm'] == algo]
            pivot = (sub.groupby(['sensor', 'channel'])[col]
                     .mean().unstack('channel'))
            pivot = pivot.reindex(SENSOR_ORDER)
            cols_p = [c for c in CHANNELS if c in pivot.columns]
            pivot = pivot[cols_p]
            im = ax.imshow(pivot.values, cmap='RdYlGn',
                           vmin=0, vmax=1, aspect='auto')
            ax.set_xticks(range(len(cols_p)))
            ax.set_xticklabels(cols_p, rotation=45, ha='right',
                               fontsize=8)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_title(algo, fontsize=11, fontweight='bold')
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = pivot.values[i, j]
                    if np.isfinite(v):
                        ax.text(j, i, f'{v:.2f}', ha='center',
                                va='center', fontsize=7,
                                color='white' if v < 0.4 else 'black')
        fig.colorbar(im, ax=axes, fraction=0.015, pad=0.02,
                     label=f'F1 ({event})')
        fig.suptitle(
            f'{event} F1 — 4명 평균 (±{TOL_MAIN_MS}ms)',
            fontsize=13)
        fig.savefig(out_dir / f'heatmap_avg_{event}.png',
                    dpi=140, bbox_inches='tight')
        plt.close(fig)


# ═════════ Fig 2: 피험자별 히트맵 ═════════
def fig_heatmap_by_subject(df, out_dir):
    for event in ('HS', 'TO'):
        col = f'F1_{event}_150'
        for algo in ALGO_ORDER:
            sub = df[df['algorithm'] == algo]
            fig, axes = plt.subplots(1, len(SUBJECT_ORDER),
                                     figsize=(5 * len(SUBJECT_ORDER), 4.5),
                                     sharey=True)
            for ax, subj in zip(axes, SUBJECT_ORDER):
                ss = sub[sub['subject'] == subj]
                pivot = (ss.groupby(['sensor', 'channel'])[col]
                         .mean().unstack('channel'))
                pivot = pivot.reindex(SENSOR_ORDER)
                cols_p = [c for c in CHANNELS if c in pivot.columns]
                pivot = pivot[cols_p]
                im = ax.imshow(pivot.values, cmap='RdYlGn',
                               vmin=0, vmax=1, aspect='auto')
                ax.set_xticks(range(len(cols_p)))
                ax.set_xticklabels(cols_p, rotation=45, ha='right',
                                   fontsize=8)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index)
                g = SUBJECT_GENDER[subj]
                ax.set_title(f'{subj} ({g})', fontsize=11,
                             fontweight='bold',
                             color=GENDER_COLORS[g])
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        v = pivot.values[i, j]
                        if np.isfinite(v):
                            ax.text(j, i, f'{v:.2f}', ha='center',
                                    va='center', fontsize=6,
                                    color='white' if v < 0.4 else 'black')
            fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02,
                         label=f'F1 ({event})')
            fig.suptitle(
                f'{algo} — {event} F1 — 피험자별 (±{TOL_MAIN_MS}ms)',
                fontsize=12)
            fig.savefig(out_dir / f'heatmap_{algo}_{event}.png',
                        dpi=130, bbox_inches='tight')
            plt.close(fig)


# ═════════ Fig 3: 알고리즘 비교 (foot Gyro_Y) ═════════
def fig_algo_compare(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, event in zip(axes, ['HS', 'TO']):
        col = f'F1_{event}_150'
        sub = df[(df['sensor'] == 'foot') & (df['channel'] == 'Gyro_Y')]
        pivot = (sub.groupby(['subject', 'algorithm'])[col]
                 .mean().unstack('algorithm'))
        pivot = pivot.reindex(SUBJECT_ORDER)
        pivot = pivot[ALGO_ORDER]
        x = np.arange(len(pivot.index))
        w = 0.18
        for i, algo in enumerate(ALGO_ORDER):
            offset = (i - (len(ALGO_ORDER) - 1) / 2) * w
            bars = ax.bar(x + offset, pivot[algo], w,
                          label=algo, color=ALGO_COLORS.get(algo))
            for b, v in zip(bars, pivot[algo]):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                            f'{v:.2f}', ha='center', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s}\n({SUBJECT_GENDER[s]})'
                            for s in pivot.index])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f'{event} F1 (±{TOL_MAIN_MS}ms)')
        ax.set_title(f'foot Gyro_Y — {event}', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)
    fig.suptitle('알고리즘 비교 — 발등 Gyro_Y (4명)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / 'algo_compare.png',
                dpi=140, bbox_inches='tight')
    plt.close(fig)


# ═════════ Fig 4: 성별 비교 ═════════
def fig_gender_compare(df, out_dir):
    """top 6 조합에서 M (PJH+KKH) vs F (LNH+YHS) 평균±std"""
    for event in ('HS', 'TO'):
        col = f'F1_{event}_150'
        top = (df.groupby(['sensor', 'channel', 'algorithm'])[col]
               .mean().sort_values(ascending=False).head(6))
        combos = list(top.index)

        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(combos))
        w = 0.35
        m_means, m_stds, f_means, f_stds = [], [], [], []
        for sensor, channel, algo in combos:
            sub = df[(df['sensor'] == sensor) &
                     (df['channel'] == channel) &
                     (df['algorithm'] == algo)]
            m = sub[sub['gender'] == 'M']
            f = sub[sub['gender'] == 'F']
            m_means.append(m[col].mean())
            m_stds.append(m[col].std())
            f_means.append(f[col].mean())
            f_stds.append(f[col].std())

        b1 = ax.bar(x - w / 2, m_means, w, yerr=m_stds,
                    label='남 (PJH, KKH)',
                    color=GENDER_COLORS['M'], capsize=4)
        b2 = ax.bar(x + w / 2, f_means, w, yerr=f_stds,
                    label='여 (LNH, YHS)',
                    color=GENDER_COLORS['F'], capsize=4)
        for bs, vals in [(b1, m_means), (b2, f_means)]:
            for b, v in zip(bs, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.03,
                            f'{v:.2f}', ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s}\n{c}\n{a}' for s, c, a in combos],
                           fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(f'{event} F1 (±{TOL_MAIN_MS}ms)')
        ax.set_title(f'{event} — 성별 비교 (top 6 조합)',
                     fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f'gender_{event}.png',
                    dpi=140, bbox_inches='tight')
        plt.close(fig)


# ═════════ Fig 5: 속도별 라인 ═════════
def fig_speed_compare(df, out_dir):
    for event in ('HS', 'TO'):
        col = f'F1_{event}_150'
        top = (df.groupby(['sensor', 'channel', 'algorithm'])[col]
               .mean().sort_values(ascending=False).head(4))
        combos = list(top.index)

        fig, axes = plt.subplots(1, len(combos),
                                 figsize=(4 * len(combos), 4),
                                 sharey=True)
        for ax, (sensor, channel, algo) in zip(axes, combos):
            sub = df[(df['sensor'] == sensor) &
                     (df['channel'] == channel) &
                     (df['algorithm'] == algo)]
            for subj in SUBJECT_ORDER:
                ss = sub[sub['subject'] == subj]
                vals = ss.groupby('speed')[col].mean().reindex(SPEEDS)
                ax.plot(SPEEDS, vals.values, 'o-',
                        label=f'{subj} ({SUBJECT_GENDER[subj]})',
                        color=SUBJECT_COLORS[subj], lw=2, ms=8)
            ax.set_title(f'{sensor}\n{channel} / {algo}',
                         fontsize=10, fontweight='bold')
            ax.set_xlabel('속도 (km/h)')
            ax.set_xticks(SPEEDS)
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
            ax.legend(loc='lower right', fontsize=8)
        axes[0].set_ylabel(f'{event} F1 (±{TOL_MAIN_MS}ms)')
        fig.suptitle(f'{event} F1 — 속도별 (top 4 조합)',
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fig.savefig(out_dir / f'speed_{event}.png',
                    dpi=140, bbox_inches='tight')
        plt.close(fig)


# ═════════ 콘솔: Top-10 표 ═════════
def print_top_combinations(df):
    for event in ('HS', 'TO'):
        col = f'F1_{event}_150'
        top = (df.groupby(['sensor', 'channel', 'algorithm'])[col]
               .agg(['mean', 'std']).reset_index()
               .sort_values('mean', ascending=False).head(10))
        print(f'\n━━━━━ Top 10 {event} (4명 평균, ±{TOL_MAIN_MS}ms) ━━━━━')
        print(f'{"#":>2} {"sensor":<12} {"channel":<14} '
              f'{"algorithm":<14} {"mean":>6} {"std":>6}')
        for i, row in enumerate(top.itertuples(), 1):
            print(f'{i:>2} {row.sensor:<12} {row.channel:<14} '
                  f'{row.algorithm:<14} {row.mean:>6.3f} {row.std:>6.3f}')


def main():
    print(f'결과 폴더: {RESULTS_DIR}')
    df = load_results()
    print(f'cell_results.csv: {len(df)}행\n')

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print('Fig 1: 4명 평균 히트맵...')
    fig_heatmap_avg(df, FIGURES_DIR)

    print('Fig 2: 피험자별 히트맵...')
    fig_heatmap_by_subject(df, FIGURES_DIR)

    print('Fig 3: 알고리즘 비교 (foot Gyro_Y)...')
    fig_algo_compare(df, FIGURES_DIR)

    print('Fig 4: 성별 비교 (top 6)...')
    fig_gender_compare(df, FIGURES_DIR)

    print('Fig 5: 속도별 라인 (top 4)...')
    fig_speed_compare(df, FIGURES_DIR)

    print_top_combinations(df)

    print(f'\n[완료] 그림 저장: {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
