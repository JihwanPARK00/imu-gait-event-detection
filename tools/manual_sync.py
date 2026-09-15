"""
manual_sync.py — GyroY 보행 봉우리에 GT 직접 맞추기
====================================================
점프 동기화가 trial마다 부정확(진동 뭉개짐)하므로 점프를 버리고,
발등 Gyro_Y 보행 파형에 GT HS/TO를 직접 정렬한다.
trial 내 시간은 일정(drift 없음 확인)하므로 한 위치에 맞추면 전체가 맞음.

조작:
  슬라이더 / ←→(1ms) / Shift+←→(10ms) : offset Δ 조정
  a/d : 보는 구간 좌우 이동 (다른 봉우리 구간 확인)
  Enter 또는 Save→Next : 저장하고 다음
  r : 리셋,  q : 저장 후 종료

출력: output/synced/manual_offsets.csv
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider, Button
from pathlib import Path
from imu_loader import load_imu_sheet
from scipy.signal import butter, sosfiltfilt, find_peaks
import warnings
warnings.filterwarnings('ignore')

IMU_DIR = 'imu'; LABEL_DIR = 'labels'; SYNC_DIR = 'output/synced'
OUT_CSV = 'output/synced/manual_offsets.csv'
FOOT = '100-0'
VIEW_SECONDS = 5        # 한 화면에 보여줄 보행 구간 길이 (봉우리 4~5개)
SIGN = +1               # 발등 Gyro_Y + (EBterminal 확인)

mpl.rcParams['font.family'] = 'Malgun Gothic'
mpl.rcParams['axes.unicode_minus'] = False


def list_trials():
    out = []
    for xlsx in sorted(Path(IMU_DIR).glob('*.xlsx')):
        for sheet in pd.ExcelFile(xlsx).sheet_names:
            lp = Path(LABEL_DIR) / f'{sheet}_labels.csv'
            sp = Path(SYNC_DIR) / f'{sheet}_synced.csv'
            if lp.is_file() and sp.is_file():
                out.append((xlsx, sheet, lp, sp))
    return out


def get_auto_offset(sp):
    s = pd.read_csv(sp)
    if 'imu_time' in s.columns and 'time_sec' in s.columns:
        return float((s['imu_time'] - s['time_sec']).median())
    return 0.0


def main():
    trials = list_trials()
    if not trials:
        print('trial 없음'); return
    print(f'{len(trials)}개 trial. GyroY 봉우리에 GT 맞추고 Enter로 다음.')

    results = {}
    state = {'i': 0, 'view0': None}
    cache = {}

    fig, ax = plt.subplots(figsize=(15, 6))
    plt.subplots_adjust(bottom=0.20)
    ax_sl = plt.axes([0.15, 0.08, 0.6, 0.03])
    ax_save = plt.axes([0.80, 0.07, 0.08, 0.05])
    ax_reset = plt.axes([0.80, 0.01, 0.08, 0.05])
    slider = Slider(ax_sl, 'offset Δ(s)', -3.0, 3.0, valinit=0, valstep=0.001)
    btn_save = Button(ax_save, 'Save→Next')
    btn_reset = Button(ax_reset, 'Reset')

    def load(i):
        xlsx, sheet, lp, sp = trials[i]
        if sheet in cache:
            return cache[sheet]
        sensors, _ = load_imu_sheet(xlsx, sheet)
        foot = sensors[FOOT]
        t = foot['t'].values
        fs = 1000.0 / np.median(np.diff(foot['_ts_ms'].values))
        gy = sosfiltfilt(butter(4, 12, 'lp', fs=fs, output='sos'),
                         foot['Gyro_Y'].values * SIGN)
        lab = pd.read_csv(lp)
        auto = get_auto_offset(sp)
        win = (lab['time_sec'].min() + auto, lab['time_sec'].max() + auto)
        center = (win[0] + win[1]) / 2
        d = dict(sheet=sheet, t=t, gy=gy, lab=lab, auto=auto, center=center,
                 rng=(gy.max() - gy.min()))
        cache[sheet] = d
        return d

    def draw():
        ax.clear()
        i = state['i']; d = load(i); delta = slider.val; off = d['auto'] + delta
        t, gy = d['t'], d['gy']
        v0 = state['view0'] if state['view0'] is not None else d['center']
        m = (t >= v0) & (t <= v0 + VIEW_SECONDS)
        ax.plot(t[m], gy[m], 'k', lw=1.2, label='발등 Gyro_Y')
        # 봉우리/골짜기 표시 (참고)
        pk, _ = find_peaks(gy, distance=int(0.4 * 90), prominence=d['rng'] * 0.2)
        tr, _ = find_peaks(-gy, distance=int(0.4 * 90), prominence=d['rng'] * 0.2)
        for arr, mk, c in [(pk, '^', 'm'), (tr, 'v', 'c')]:
            xt = t[arr]; xt = xt[(xt >= v0) & (xt <= v0 + VIEW_SECONDS)]
            ax.plot(xt, gy[np.searchsorted(t, xt)], mk, color=c, ms=7, alpha=0.6)
        # GT HS/TO (offset 적용)
        lab = d['lab']
        hs = lab[lab.event == 'HS']['time_sec'].values + off
        to = lab[lab.event == 'TO']['time_sec'].values + off
        for x in hs[(hs >= v0) & (hs <= v0 + VIEW_SECONDS)]:
            ax.axvline(x, color='green', ls=':', lw=2)
        for x in to[(to >= v0) & (to <= v0 + VIEW_SECONDS)]:
            ax.axvline(x, color='red', ls=':', lw=1.5)
        ax.axvline(np.nan, color='green', ls=':', lw=2, label='GT HS')
        ax.axvline(np.nan, color='red', ls=':', lw=1.5, label='GT TO')
        ax.plot([], [], 'm^', label='봉우리'); ax.plot([], [], 'cv', label='골짜기')
        ax.set_title(f'[{i+1}/{len(trials)}] {d["sheet"]}   '
                     f'auto={d["auto"]:+.3f} Δ={delta:+.3f} 최종={off:+.3f}s\n'
                     f'GT HS(초록)/TO(빨강)를 봉우리·골짜기의 일정 위치에 맞추세요 (a/d=구간이동)')
        ax.set_xlabel('IMU time (s)'); ax.set_ylabel('Gyro_Y')
        ax.legend(loc='upper right', fontsize=9); ax.grid(alpha=0.3)
        fig.canvas.draw_idle()

    def on_save(event=None):
        i = state['i']; d = load(i)
        results[d['sheet']] = {'trial': d['sheet'], 'auto_offset': round(d['auto'], 4),
                               'manual_offset': round(d['auto'] + slider.val, 4),
                               'delta_ms': round(slider.val * 1000, 1)}
        print(f'  저장 [{i+1}/{len(trials)}] {d["sheet"]}: Δ={slider.val*1000:+.0f}ms')
        if i + 1 < len(trials):
            state['i'] += 1; state['view0'] = None; slider.reset(); draw()
        else:
            print('완료! q로 종료.'); save_csv()

    def on_reset(event=None):
        state['view0'] = None; slider.reset(); draw()

    def on_key(e):
        d = load(state['i'])
        v0 = state['view0'] if state['view0'] is not None else d['center']
        if e.key == 'enter': on_save()
        elif e.key == 'right': slider.set_val(min(3.0, slider.val + 0.001))
        elif e.key == 'left': slider.set_val(max(-3.0, slider.val - 0.001))
        elif e.key == 'shift+right': slider.set_val(min(3.0, slider.val + 0.010))
        elif e.key == 'shift+left': slider.set_val(max(-3.0, slider.val - 0.010))
        elif e.key == 'd': state['view0'] = v0 + VIEW_SECONDS * 0.8; draw()
        elif e.key == 'a': state['view0'] = v0 - VIEW_SECONDS * 0.8; draw()
        elif e.key == 'r': on_reset()
        elif e.key == 'q': save_csv(); plt.close(fig)

    def save_csv():
        if results:
            df = pd.DataFrame(list(results.values()))
            Path(SYNC_DIR).mkdir(parents=True, exist_ok=True)
            df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
            print(f'\n저장: {OUT_CSV} ({len(df)} trial)')

    slider.on_changed(lambda v: draw())
    btn_save.on_clicked(on_save)
    btn_reset.on_clicked(on_reset)
    fig.canvas.mpl_connect('key_press_event', on_key)
    draw(); plt.show(); save_csv()


if __name__ == '__main__':
    main()