"""
video_labeler.py (v2 - CLAP 이벤트 추가)
==========================================
영상에서 Heel Strike(HS), Toe Off(TO), 동기화 이벤트(JUMP, CLAP) 수동 레이블링.

사용법:
    python video_labeler.py <video_path>
    python video_labeler.py <video_path> --out labels.csv
    python video_labeler.py <video_path> --resume

키 조작:
    SPACE       : 재생 / 일시정지
    , / .       : -1 / +1 프레임 (한 프레임씩 정밀 이동)
    [ / ]       : -10 / +10 프레임
    < / >       : -1초 / +1초
    h           : 현재 프레임을 HS 로 기록
    t           : 현재 프레임을 TO 로 기록
    j           : 현재 프레임을 JUMP(점프 동기화)로 기록
    c           : 현재 프레임을 CLAP(박수 동기화)로 기록  ← v2 추가
    u           : 마지막 레이블 1개 삭제 (undo)
    r           : 현재 프레임에 가까운(±3 frame) 레이블 모두 삭제
    + / -       : 재생 속도 증가/감소
    s           : 현재까지의 레이블 저장
    q 또는 ESC  : 저장하고 종료

CLAP 사용 권장 시점:
    - 1회차: 목표 속도 도달 직후 (정상 보행 시작점)
    - 2회차: 정상 보행 종료 직전 (속도 감속 시작점)
    → 두 박수 사이가 분석 가능 구간

출력 CSV 컬럼:
    event, frame, time_sec
    (event ∈ {HS, TO, JUMP, CLAP})
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))  # repo root
import argparse
import csv
import os
import sys
from collections import deque

import cv2


# ---------- 색상/스타일 ----------
COLOR_BG = (24, 24, 28)
COLOR_HS = (80, 200, 255)
COLOR_TO = (110, 255, 130)
COLOR_JUMP = (80, 120, 255)
COLOR_CLAP = (255, 180, 80)         # v2 추가: 주황색
COLOR_TEXT = (240, 240, 240)
COLOR_DIM = (160, 160, 160)
COLOR_HILITE = (50, 220, 255)

EVENT_COLOR = {"HS": COLOR_HS, "TO": COLOR_TO, "JUMP": COLOR_JUMP, "CLAP": COLOR_CLAP}


def draw_overlay(frame, info):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 78), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    bar_y = 70
    progress = info["frame_idx"] / max(info["total_frames"] - 1, 1)
    cv2.rectangle(frame, (10, bar_y), (w - 10, bar_y + 4), (60, 60, 70), -1)
    cv2.rectangle(
        frame,
        (10, bar_y),
        (10 + int((w - 20) * progress), bar_y + 4),
        COLOR_HILITE,
        -1,
    )

    for ev, frame_no in info["events_for_bar"]:
        x = 10 + int((w - 20) * (frame_no / max(info["total_frames"] - 1, 1)))
        col = EVENT_COLOR.get(ev, COLOR_TEXT)
        cv2.line(frame, (x, bar_y - 6), (x, bar_y + 10), col, 1)

    line1 = (
        f"Frame {info['frame_idx']:>6d} / {info['total_frames']-1}    "
        f"t = {info['time_sec']:7.3f}s    "
        f"fps={info['fps']:.1f}    "
        f"playspeed x{info['play_speed']:.1f}    "
        f"{'PLAY' if info['playing'] else 'PAUSE'}"
    )
    cv2.putText(frame, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

    counts = info["counts"]
    line2 = (
        f"HS={counts['HS']}  TO={counts['TO']}  JUMP={counts['JUMP']}  CLAP={counts['CLAP']}    "
        f"out={os.path.basename(info['out_path'])}"
    )
    cv2.putText(frame, line2, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, COLOR_DIM, 1, cv2.LINE_AA)

    hist = info["recent"]
    for i, (ev, fr, t) in enumerate(reversed(list(hist))):
        y = h - 14 - i * 18
        if y < 90:
            break
        col = EVENT_COLOR.get(ev, COLOR_TEXT)
        cv2.putText(
            frame,
            f"{ev:4s}  f={fr:6d}  t={t:7.3f}s",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            col,
            1,
            cv2.LINE_AA,
        )

    helps = [
        "SPACE play  , . frame  < > 10f  [ ] 1s",
        "h HS   t TO   j JUMP   c CLAP   u undo   r remove",
        "+ - speed   s save   q quit",
    ]
    for i, line in enumerate(helps):
        cv2.putText(
            frame,
            line,
            (w - 420, h - 14 - (len(helps) - 1 - i) * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            COLOR_DIM,
            1,
            cv2.LINE_AA,
        )

    flash = info.get("flash_event")
    if flash:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), EVENT_COLOR[flash], 6)


def save_csv(out_path, labels):
    labels_sorted = sorted(labels, key=lambda x: x[1])
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "frame", "time_sec"])
        for ev, fr, t in labels_sorted:
            w.writerow([ev, fr, f"{t:.6f}"])


def load_existing(path):
    if not os.path.isfile(path):
        return []
    labels = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            labels.append((row["event"], int(row["frame"]), float(row["time_sec"])))
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="영상 파일 경로")
    ap.add_argument("--out", default=None, help="레이블 저장 경로 (기본: <video>_labels.csv)")
    ap.add_argument("--resume", action="store_true", help="기존 레이블 CSV가 있으면 이어서 작업")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        print(f"[ERR] not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or os.path.splitext(args.video)[0] + "_labels.csv"

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERR] cannot open video: {args.video}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] video: {args.video}")
    print(f"[INFO] {width}x{height}, fps={fps:.3f}, total_frames={total_frames}")
    print(f"[INFO] labels -> {out_path}")

    labels = load_existing(out_path) if args.resume else []
    if labels:
        print(f"[INFO] resumed with {len(labels)} existing labels")

    win = "video_labeler"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if width > 1280:
        cv2.resizeWindow(win, 1280, int(1280 * height / width))
    else:
        cv2.resizeWindow(win, width, height)

    def on_trackbar(v):
        pass

    cv2.createTrackbar("frame", win, 0, max(total_frames - 1, 1), on_trackbar)

    playing = False
    play_speed = 1.0
    recent = deque(maxlen=12)
    flash_event = None
    flash_until = 0

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        return ok, fr

    cur_idx = 0
    ok, frame = read_frame(cur_idx)
    if not ok:
        print("[ERR] failed to read first frame")
        sys.exit(1)

    tick_freq = cv2.getTickFrequency()

    while True:
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                save_csv(out_path, labels)
                print(f"[INFO] window closed - saved {len(labels)} labels -> {out_path}")
                break
        except cv2.error:
            save_csv(out_path, labels)
            print(f"[INFO] window destroyed - saved {len(labels)} labels -> {out_path}")
            break

        tb_idx = cv2.getTrackbarPos("frame", win)
        if tb_idx != cur_idx and not playing:
            cur_idx = tb_idx
            ok, frame = read_frame(cur_idx)
            if not ok:
                cur_idx = max(0, cur_idx - 1)
                ok, frame = read_frame(cur_idx)

        if playing:
            next_idx = cur_idx + 1
            if next_idx >= total_frames:
                playing = False
                next_idx = total_frames - 1
            ok, frame = read_frame(next_idx)
            if ok:
                cur_idx = next_idx
                cv2.setTrackbarPos("frame", win, cur_idx)

        disp = frame.copy()

        counts = {"HS": 0, "TO": 0, "JUMP": 0, "CLAP": 0}
        for ev, _, _ in labels:
            if ev in counts:
                counts[ev] += 1

        events_for_bar = [(ev, fr) for ev, fr, _ in labels]
        if len(events_for_bar) > 600:
            step = len(events_for_bar) // 600
            events_for_bar = events_for_bar[::step]

        flash_now = None
        if flash_event is not None and cv2.getTickCount() < flash_until:
            flash_now = flash_event
        else:
            flash_event = None

        info = {
            "frame_idx": cur_idx,
            "total_frames": total_frames,
            "time_sec": cur_idx / fps,
            "fps": fps,
            "play_speed": play_speed,
            "playing": playing,
            "counts": counts,
            "recent": recent,
            "events_for_bar": events_for_bar,
            "out_path": out_path,
            "flash_event": flash_now,
        }
        draw_overlay(disp, info)
        cv2.imshow(win, disp)

        if playing:
            base_delay = max(1, int(1000.0 / (fps * play_speed)))
        else:
            base_delay = 30
        key = cv2.waitKey(base_delay) & 0xFF

        if key == 255:
            continue

        def add_label(ev):
            nonlocal flash_event, flash_until
            t = cur_idx / fps
            labels.append((ev, cur_idx, t))
            recent.append((ev, cur_idx, t))
            flash_event = ev
            flash_until = cv2.getTickCount() + int(0.18 * tick_freq)
            print(f"  + {ev:4s}  frame={cur_idx:6d}  t={t:.4f}s")

        def remove_near(window=3):
            removed = [lab for lab in labels if abs(lab[1] - cur_idx) <= window]
            if removed:
                for r in removed:
                    labels.remove(r)
                    if r in recent:
                        recent.remove(r)
                print(f"  - removed {len(removed)} label(s) near frame {cur_idx}")
            else:
                print(f"  (no label within ±{window} frames of {cur_idx})")

        if key in (ord("q"), 27):
            save_csv(out_path, labels)
            print(f"[INFO] saved {len(labels)} labels -> {out_path}")
            break

        elif key == ord(" "):
            playing = not playing

        elif key == ord(","):
            playing = False
            cur_idx = max(0, cur_idx - 1)
            cv2.setTrackbarPos("frame", win, cur_idx)
            ok, frame = read_frame(cur_idx)
        elif key == ord("."):
            playing = False
            cur_idx = min(total_frames - 1, cur_idx + 1)
            cv2.setTrackbarPos("frame", win, cur_idx)
            ok, frame = read_frame(cur_idx)
        elif key == ord("["):
            playing = False
            cur_idx = max(0, cur_idx - 10)
            cv2.setTrackbarPos("frame", win, cur_idx)
            ok, frame = read_frame(cur_idx)
        elif key == ord("]"):
            playing = False
            cur_idx = min(total_frames - 1, cur_idx + 10)
            cv2.setTrackbarPos("frame", win, cur_idx)
            ok, frame = read_frame(cur_idx)
        elif key == ord("<"):
            playing = False
            cur_idx = max(0, cur_idx - int(round(fps)))
            cv2.setTrackbarPos("frame", win, cur_idx)
            ok, frame = read_frame(cur_idx)
        elif key == ord(">"):
            playing = False
            cur_idx = min(total_frames - 1, cur_idx + int(round(fps)))
            cv2.setTrackbarPos("frame", win, cur_idx)
            ok, frame = read_frame(cur_idx)

        elif key == ord("h"):
            add_label("HS")
        elif key == ord("t"):
            add_label("TO")
        elif key == ord("j"):
            add_label("JUMP")
        elif key == ord("c"):              # v2 추가
            add_label("CLAP")

        elif key == ord("u"):
            if labels:
                last = labels.pop()
                if last in recent:
                    recent.remove(last)
                print(f"  - undo {last[0]} frame={last[1]}")
            else:
                print("  (nothing to undo)")
        elif key == ord("r"):
            remove_near(3)

        elif key in (ord("+"), ord("=")):
            play_speed = min(8.0, play_speed * 1.25)
        elif key == ord("-"):
            play_speed = max(0.125, play_speed / 1.25)

        elif key == ord("s"):
            save_csv(out_path, labels)
            print(f"  saved {len(labels)} labels -> {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()