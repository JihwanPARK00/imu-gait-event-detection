"""
라벨 패치 + heatmap 재생성 (matplotlib only)

ICCAS 폴더에서 실행:
    python patch_labels_and_regenerate.py
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from pathlib import Path

RESULTS = Path('results')
CSV = RESULTS / 'cell_results.csv'
BACKUP = RESULTS / 'cell_results_backup.csv'

# ─────────── Step 1: 백업 ───────────
if not BACKUP.exists():
    shutil.copy(CSV, BACKUP)
    print(f'[백업] {BACKUP}')
else:
    print(f'[백업 이미 존재] {BACKUP}')

# ─────────── Step 2: 라벨 통일 ───────────
df = pd.read_csv(CSV)
rename_map = {
    'thigh_aff': 'thigh_R',
    'thigh_sound': 'thigh_L',
    'lumbar': 'abdomen',
}
before = sorted(df['sensor'].unique().tolist())
df['sensor'] = df['sensor'].replace(rename_map)
after = sorted(df['sensor'].unique().tolist())
print(f'[라벨] {before} -> {after}')
df.to_csv(CSV, index=False)
print(f'[저장] {CSV}')

# ─────────── Step 3: 2x2 heatmap 재생성 ───────────
SENSOR_ORDER = ['foot', 'shank', 'thigh_R', 'thigh_L', 'abdomen']
CHANNEL_ORDER = ['Gyro_X', 'Gyro_Y', 'Gyro_Z',
                 'Acc_X', 'Acc_Y', 'Acc_Z',
                 'Euler_roll', 'Euler_pitch', 'Euler_yaw']
ALGORITHMS = ['TB_classic', 'TB_adaptive', 'BPF', 'TM']


def draw_heatmap(ax, data, title, show_ylabel=True):
    """matplotlib imshow + annotation, aspect=auto (subplot 영역 채움)"""
    cmap = plt.cm.RdYlGn
    norm = Normalize(vmin=0, vmax=1)
    im = ax.imshow(data.values, cmap=cmap, norm=norm, aspect='auto')

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            color = 'white' if (v < 0.25 or v > 0.85) else 'black'
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    color=color, fontsize=11, fontweight='bold')

    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=45, ha='right', fontsize=11)
    ax.set_yticks(range(len(data.index)))
    if show_ylabel:
        ax.set_yticklabels(data.index, fontsize=11)
    else:
        ax.set_yticklabels([])
    ax.set_title(title, fontsize=14, fontweight='bold')
    return im


for event in ['HS', 'TO']:
    col = f'F1_{event}_150'
    if col not in df.columns:
        print(f'[skip] {col} 컬럼 없음')
        continue

    # GridSpec: 2x2 subplot + 우측 colorbar (전체 높이)
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig,
                  width_ratios=[1, 1, 0.035],
                  wspace=0.04, hspace=0.28,
                  left=0.05, right=0.94,
                  top=0.96, bottom=0.07)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    cbar_ax = fig.add_subplot(gs[:, 2])  # 양 행 가로지르는 colorbar

    last_im = None
    for i, (ax, alg) in enumerate(zip(axes, ALGORITHMS)):
        sub = df[df['algorithm'] == alg]
        pivot = (sub.groupby(['sensor', 'channel'])[col]
                 .mean().unstack())
        pivot = pivot.reindex(index=SENSOR_ORDER, columns=CHANNEL_ORDER)
        # 좌측 subplot(0, 2)만 y라벨 표시
        show_y = (i % 2 == 0)
        last_im = draw_heatmap(ax, pivot, alg, show_ylabel=show_y)

    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label(f'F1 ({event})', fontsize=11)

    out = RESULTS / f'heatmap_F1_2x2_{event}.png'
    plt.savefig(out, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'[saved] {out}')

print('\n완료. 본 논문에 사용:')
print(f'  Fig. 2: {RESULTS / "heatmap_F1_2x2_HS.png"}')
