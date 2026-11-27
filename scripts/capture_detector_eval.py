"""검출기 비교(Ablation)용 탑뷰 원본 프레임 고속 캡처 GUI (카메라 전용, 로봇 구동 없음).

실행:
  python scripts/capture_detector_eval.py
  python scripts/capture_detector_eval.py --target 20

절차 (약 3~5분):
  1) 작업대를 비운 상태(기본 조명)에서 [B] 키 -> 빈 배경(bg_empty.png) 1장 저장
  2) 조건별로 블록을 놓으면서 화면 위 블록 중심을 [마우스 좌클릭] -> 원본 PNG + 정답 중심(GT) 즉시 저장
     - [1] base         : 기본 위치, 파란 블록, 기본 조명
     - [2] new_pos      : 새 위치(구석·상자 그림자 근처 포함), 파란 블록, 기본 조명
     - [3] color_change : 다른 색 블록(빨강/노랑 등), 기본 조명
     - [4] light_change : 파란 블록, 조명 어둡게/밝게/그림자 변화
  3) [Q]로 종료 후 `python scripts/eval_detectors.py` 실행 -> 표·300 DPI 그래프 자동 생성
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from so101.camera import open_camera, read_frame, require_gui  # noqa: E402
from so101.detect_ablation import (  # noqa: E402
    AblationParams,
    build_workspace_mask,
    run_detector,
)
from so101.experiment import CONDITIONS  # noqa: E402
from so101.homography import TopViewRayCalib  # noqa: E402

COND_LIST = list(CONDITIONS.keys())
MANIFEST_FIELDS = ["filename", "condition", "trial", "gt_u", "gt_v", "block_color", "mean_v", "timestamp"]


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def compute_workspace_poly() -> tuple[np.ndarray | None, tuple[int, int] | None]:
    if not (config.CAMERA_INTRINSICS.exists() and config.CAMERA_EXTRINSICS.exists()):
        return None, None
    ray = TopViewRayCalib.from_files(config.CAMERA_INTRINSICS, config.CAMERA_EXTRINSICS)
    corners = [
        [config.WORKSPACE_X_M[0], config.WORKSPACE_Y_M[0], config.BLOCK_HEIGHT_M],
        [config.WORKSPACE_X_M[1], config.WORKSPACE_Y_M[0], config.BLOCK_HEIGHT_M],
        [config.WORKSPACE_X_M[1], config.WORKSPACE_Y_M[1], config.BLOCK_HEIGHT_M],
        [config.WORKSPACE_X_M[0], config.WORKSPACE_Y_M[1], config.BLOCK_HEIGHT_M],
    ]
    ws_poly = np.array([ray.project(p) for p in corners], dtype=np.int32)
    box_px = tuple(int(round(v)) for v in ray.project([config.BOX_XYZ_M[0], config.BOX_XYZ_M[1], 0.03]))
    return ws_poly, box_px


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=config.DATA_DIR / "detector_ablation")
    ap.add_argument("--target", type=int, default=20, help="조건당 목표 캡처 장수 (기본: 20)")
    ap.add_argument("--condition", choices=COND_LIST, default="base", help="시작 조건")
    ap.add_argument("--no-lock", action="store_true", help="lock_camera.sh 자동 실행 생략")
    args = ap.parse_args()

    require_gui()

    out_dir: Path = args.out_dir
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    bg_path = out_dir / "bg_empty.png"

    hsv_file = config.ROOT / "calib" / "hsv_blue.json"
    hsv_raw = json.loads(hsv_file.read_text())["ranges"]
    hsv_ranges = tuple((tuple(lo), tuple(hi)) for lo, hi in hsv_raw)
    preview_params = AblationParams(hsv_ranges=hsv_ranges)

    ws_poly, box_px = compute_workspace_poly()
    roi_mask = build_workspace_mask((config.CAM_HEIGHT, config.CAM_WIDTH), ws_poly)

    cap = open_camera(config.TOP_CAM)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not args.no_lock:
        lock_script = config.ROOT / "tools" / "lock_camera.sh"
        if lock_script.exists():
            subprocess.run([str(lock_script)], check=False)
    for _ in range(8):
        cap.grab()

    rows = load_manifest(manifest_path)
    bg_img = cv2.imread(str(bg_path)) if bg_path.exists() else None
    cond_idx = COND_LIST.index(args.condition)
    show_overlay = True
    mouse_pos = [-1, -1]
    pending_click: list[tuple[int, int]] = []
    flash_until = 0.0
    status_msg = (
        "Press [B] with EMPTY workspace first to capture background!"
        if bg_img is None
        else "Ready! Left-click block center to capture raw frame + GT."
    )

    def on_mouse(event, x, y, _flags, _param):
        mouse_pos[0], mouse_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= x < config.CAM_WIDTH and 0 <= y < config.CAM_HEIGHT:
            pending_click.append((x, y))

    win = "Detector Ablation Capture (B: Empty BG | Click: Save | 1-4/C: Cond | U: Undo | Q: Quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 820, 760)
    cv2.setMouseCallback(win, on_mouse)

    print("=== 검출기 Ablation 원본 프레임 캡처 시작 (로봇 구동 없음) ===")
    print("1) 먼저 빈 작업대(기본 조명)에서 [B] 키를 눌러 배경을 저장하세요.")
    print("2) 블록을 놓고 화면의 블록 중심을 [좌클릭]하면 원본 PNG와 정답 좌표(GT)가 저장됩니다.")
    print("3) 완료 후 [Q]로 종료하고 `python scripts/eval_detectors.py` 를 실행하세요.")

    try:
        while True:
            raw_frame = read_frame(cap)
            cond = COND_LIST[cond_idx]

            # Handle click capture on raw_frame (zero overlay!)
            if pending_click:
                gx, gy = pending_click.pop(0)
                pending_click.clear()
                same = [int(r["trial"]) for r in rows if r["condition"] == cond]
                trial_no = (max(same) + 1) if same else 1
                fname = f"{cond}_{trial_no:02d}.png"
                cv2.imwrite(str(frames_dir / fname), raw_frame)
                hsv_v = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2HSV)[:, :, 2]
                mean_v = float(hsv_v[roi_mask > 0].mean()) if roi_mask is not None else float(hsv_v.mean())
                rec = {
                    "filename": f"frames/{fname}",
                    "condition": cond,
                    "trial": str(trial_no),
                    "gt_u": f"{gx:.1f}",
                    "gt_v": f"{gy:.1f}",
                    "block_color": "other" if cond == "color_change" else "blue",
                    "mean_v": f"{mean_v:.1f}",
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                rows.append(rec)
                save_manifest(manifest_path, rows)
                flash_until = time.perf_counter() + 0.14
                status_msg = f"Saved [{cond} #{trial_no}] GT=({gx}, {gy}) meanV={mean_v:.1f}"
                print(f"  [저장] {fname} | GT=({gx}, {gy}) | meanV={mean_v:.1f}")

            view = raw_frame.copy()
            if ws_poly is not None:
                cv2.polylines(view, [ws_poly], True, (200, 180, 0), 1)
            if box_px is not None:
                cv2.drawMarker(view, box_px, (0, 0, 255), cv2.MARKER_DIAMOND, 12, 2)

            # Live preview of HSV(orig) vs RGB-diff vs BG-diff
            if show_overlay:
                det_hsv, _ = run_detector("hsv_orig", raw_frame, preview_params)
                det_rgb, _ = run_detector("rgb_diff", raw_frame, preview_params)
                det_bg, _ = run_detector("bg_diff", raw_frame, preview_params, bg_bgr=bg_img, roi_mask=roi_mask)

                if det_hsv is not None:
                    cv2.drawContours(view, [det_hsv.box.astype(np.int32)], 0, (0, 255, 0), 2)
                if det_rgb is not None:
                    cv2.drawContours(view, [det_rgb.box.astype(np.int32)], 0, (255, 255, 0), 1)
                    cv2.circle(view, (int(det_rgb.u), int(det_rgb.v)), 3, (255, 255, 0), -1)
                if det_bg is not None:
                    cv2.drawContours(view, [det_bg.box.astype(np.int32)], 0, (0, 165, 255), 1)

                hsv_txt = f"HSV:OK({det_hsv.area:.0f})" if det_hsv else "HSV:MISS"
                rgb_txt = f"RGBdiff:OK({det_rgb.area:.0f})" if det_rgb else "RGBdiff:MISS"
                bg_txt = (
                    f"BGdiff:OK({det_bg.area:.0f})"
                    if det_bg
                    else ("BGdiff:NO_BG" if bg_img is None else "BGdiff:MISS")
                )
                cv2.putText(view, hsv_txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if det_hsv else (0, 80, 255), 2)
                cv2.putText(view, rgb_txt, (185, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0) if det_rgb else (0, 80, 255), 2)
                cv2.putText(view, bg_txt, (410, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255) if det_bg else (140, 140, 140), 2)

            # Draw past GT dots for current condition
            cond_rows = [r for r in rows if r["condition"] == cond]
            for r in cond_rows:
                pu, pv = int(round(float(r["gt_u"]))), int(round(float(r["gt_v"])))
                cv2.circle(view, (pu, pv), 4, (255, 80, 255), -1)
                cv2.putText(view, r["trial"], (pu + 5, pv - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 160, 255), 1)

            # Mouse crosshair for accurate block-center clicking
            mx, my = mouse_pos
            if 0 <= mx < config.CAM_WIDTH and 0 <= my < config.CAM_HEIGHT:
                cv2.line(view, (mx - 14, my), (mx + 14, my), (255, 255, 255), 1)
                cv2.line(view, (mx, my - 14), (mx, my + 14), (255, 255, 255), 1)

            if time.perf_counter() < flash_until:
                cv2.rectangle(view, (2, 2), (config.CAM_WIDTH - 3, config.CAM_HEIGHT - 3), (0, 255, 120), 4)

            # Bottom HUD
            hud = np.full((135, config.CAM_WIDTH, 3), 26, dtype=np.uint8)
            counts_str = "  ".join(
                f"{'*' if c == cond else ''}{c}:{sum(1 for r in rows if r['condition'] == c)}/{args.target}"
                for c in COND_LIST
            )
            bg_badge = "BG:OK" if bg_img is not None else "BG:MISSING (press B!)"
            bg_col = (80, 255, 120) if bg_img is not None else (0, 180, 255)
            cv2.putText(
                hud,
                f"COND: {cond.upper()} [{len(cond_rows)}/{args.target}]   |   {bg_badge}",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                bg_col,
                2,
            )
            cv2.putText(hud, counts_str, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (210, 230, 255), 1)
            cv2.putText(hud, status_msg[:78], (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (245, 245, 245), 1)
            cv2.putText(
                hud,
                "[LEFT-CLICK] Save frame+GT   [B] Capture Empty BG (5-frame median)   [U] Undo",
                (12, 104),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (180, 220, 255),
                1,
            )
            cv2.putText(
                hud,
                "[1] base  [2] new_pos  [3] color_change  [4] light_change  [C] Next  [V] Overlay  [Q] Quit",
                (12, 124),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (165, 165, 165),
                1,
            )

            cv2.imshow(win, np.vstack([view, hud]))
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue

            if key in (ord("q"), 27):
                break
            elif key == ord("b"):
                stack = [read_frame(cap) for _ in range(5)]
                bg_img = np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)
                cv2.imwrite(str(bg_path), bg_img)
                flash_until = time.perf_counter() + 0.18
                status_msg = f"Saved empty background -> {bg_path.relative_to(config.ROOT)} (5-frame median)"
                print(f"  [배경 저장] {bg_path} (5프레임 중앙값)")
            elif key == ord("c"):
                cond_idx = (cond_idx + 1) % len(COND_LIST)
                status_msg = f"Condition -> {COND_LIST[cond_idx]} ({CONDITIONS[COND_LIST[cond_idx]]})"
                print(f"[조건 전환] {COND_LIST[cond_idx]}")
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")):
                cond_idx = key - ord("1")
                status_msg = f"Condition -> {COND_LIST[cond_idx]} ({CONDITIONS[COND_LIST[cond_idx]]})"
                print(f"[조건 전환] {COND_LIST[cond_idx]}")
            elif key == ord("v"):
                show_overlay = not show_overlay
                status_msg = f"Live detector overlay: {'ON' if show_overlay else 'OFF'}"
            elif key == ord("u"):
                if rows:
                    removed = rows.pop()
                    fpath = out_dir / removed["filename"]
                    if fpath.exists():
                        fpath.unlink()
                    save_manifest(manifest_path, rows)
                    status_msg = f"Undone: {removed['filename']}"
                    print(f"  [취소] {removed['filename']}")
                else:
                    status_msg = "Nothing to undo."
    finally:
        cap.release()
        cv2.destroyAllWindows()
        total = len(rows)
        print(f"\n=== 캡처 종료 (총 {total}장 저장, 배경={'있음' if bg_path.exists() else '없음'}) ===")
        for c in COND_LIST:
            print(f"  - {c:13s}: {sum(1 for r in rows if r['condition'] == c)}장")
        if total > 0:
            print("\n다음 명령으로 정량 평가 표와 보고서용 그래프를 생성하세요:")
            print(f"  python scripts/eval_detectors.py --data-dir {out_dir}")


if __name__ == "__main__":
    main()
