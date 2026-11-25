"""시스템 A 연속 비교 실험 GUI 도우미 (실시간 탑뷰 + 원키 실행·기록).

실행:
  python scripts/eval_system_a_live.py
  python scripts/eval_system_a_live.py --condition base --target 20

조작 (카메라 창을 클릭해 포커스를 둔 상태에서 키 입력):
  [대기 상태]
    Space / Enter : 픽앤플레이스 1회 실행
    C             : 조건 전환 (base -> new_pos -> color_change -> light_change)
    U             : 방금 저장한 마지막 기록 1건 취소 (Undo)
    S             : 집계표(data/experiments/summary.md) 출력 및 저장
    Q / Esc       : 종료 (로봇 안전 해제)

  [시행 완료 후 판정 상태]
    Space / Enter / Y : 추천 결과 그대로 저장 (로봇 성공 -> 성공 저장, 로봇 실패 -> 해당 실패원인 저장)
    1 : 실패 - 위치/좌표 오차 (coord_error)
    2 : 실패 - 파지 실패 (grasp_fail)
    3 : 실패 - 이송 중 낙하 (drop_fail)
    4 : 실패 - 검출 실패 (detection_fail)
    5 : 실패 - 작업 범위 밖 (out_of_range)
    X : 이번 시행 기록하지 않고 무시 (재시도)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402
from so101.camera import open_camera, read_frame, require_gui  # noqa: E402
from so101.detect import detect_block, draw_detection  # noqa: E402
from so101.experiment import (  # noqa: E402
    CONDITIONS,
    FAILURE_CAUSES,
    FIELDNAMES,
    TrialRecord,
    append_trial,
    format_markdown_report,
    load_trials,
)
from so101.fk import SO101Kinematics  # noqa: E402
from so101.homography import HomographyCalib, TopViewRayCalib  # noqa: E402
from so101.pickplace import PickPlaceResult, WorkspaceConfig, execute_pick_and_place  # noqa: E402
from so101.robot import connect_follower  # noqa: E402

COND_LIST = list(CONDITIONS.keys())
CAUSE_KEYS = {
    ord("1"): "coord_error",
    ord("2"): "grasp_fail",
    ord("3"): "drop_fail",
    ord("4"): "detection_fail",
    ord("5"): "out_of_range",
}


def load_joint_offsets() -> dict[str, float] | None:
    if config.CAMERA_EXTRINSICS.exists():
        return json.loads(config.CAMERA_EXTRINSICS.read_text()).get("joint_offset_deg")
    return None


def save_summary(csv_path: Path, summary_path: Path) -> None:
    records = load_trials(csv_path)
    md = format_markdown_report(records)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(md, encoding="utf-8")


def undo_last_trial(csv_path: Path, summary_path: Path) -> TrialRecord | None:
    records = load_trials(csv_path)
    if not records:
        return None
    removed = records.pop()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))
    save_summary(csv_path, summary_path)
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--color", default="blue", help="HSV 기준 색상 (기본: blue)")
    ap.add_argument("--condition", choices=COND_LIST, default="base", help="시작 실험 조건")
    ap.add_argument("--target", type=int, default=20, help="조건당 목표 시행 수 (기본: 20)")
    ap.add_argument("--ik", choices=["analytic", "numerical", "hybrid"], default="hybrid")
    ap.add_argument("--csv", type=Path, default=config.EXPERIMENT_CSV)
    ap.add_argument("--summary", type=Path, default=config.DATA_DIR / "experiments" / "summary.md")
    args = ap.parse_args()

    require_gui()

    hsv_file = config.ROOT / "calib" / f"hsv_{args.color}.json"
    if not hsv_file.exists():
        raise SystemExit(f"HSV 파일 없음: {hsv_file}")
    hsv_ranges = json.loads(hsv_file.read_text())["ranges"]

    calib = HomographyCalib.load(config.HOMOGRAPHY_CALIB, intrinsics_path=config.CAMERA_INTRINSICS)
    ray = (
        TopViewRayCalib.from_files(config.CAMERA_INTRINSICS, config.CAMERA_EXTRINSICS)
        if config.CAMERA_EXTRINSICS.exists()
        else None
    )
    kin = SO101Kinematics(offset_deg=load_joint_offsets())
    ws = WorkspaceConfig(
        table_z_m=config.TABLE_Z_M,
        block_height_m=config.BLOCK_HEIGHT_M,
        box_xyz_m=tuple(config.BOX_XYZ_M),
        x_range_m=config.WORKSPACE_X_M,
        y_range_m=config.WORKSPACE_Y_M,
    )

    cap = open_camera(config.TOP_CAM)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    lock_script = config.ROOT / "tools" / "lock_camera.sh"
    if lock_script.exists():
        subprocess.run([str(lock_script)], check=False)
    for _ in range(8):
        cap.grab()

    robot = connect_follower(max_relative_target=15.0)

    cond_idx = COND_LIST.index(args.condition)
    state = "IDLE"  # IDLE | RUNNING | CONFIRM
    pending_res: PickPlaceResult | None = None
    pending_cond: str = COND_LIST[cond_idx]
    status_msg = "Ready. Place block and press SPACE to run."
    worker_lock = threading.Lock()
    snap_dir = config.DATA_DIR / "system_a"
    snap_dir.mkdir(parents=True, exist_ok=True)
    cached_trials = load_trials(args.csv)

    # Precompute workspace polygon & box marker in raw pixels
    ws_poly = None
    box_px = None
    if ray is not None:
        corners = [
            [ws.x_range_m[0], ws.y_range_m[0], config.BLOCK_HEIGHT_M],
            [ws.x_range_m[1], ws.y_range_m[0], config.BLOCK_HEIGHT_M],
            [ws.x_range_m[1], ws.y_range_m[1], config.BLOCK_HEIGHT_M],
            [ws.x_range_m[0], ws.y_range_m[1], config.BLOCK_HEIGHT_M],
        ]
        ws_poly = np.array([ray.project(p) for p in corners], dtype=np.int32)
        box_px = tuple(int(round(v)) for v in ray.project([ws.box_xyz_m[0], ws.box_xyz_m[1], 0.03]))

    def run_worker(frame_snapshot: np.ndarray, det_snapshot, cond_name: str) -> None:
        nonlocal state, pending_res, pending_cond, status_msg
        stamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(snap_dir / f"{stamp}_{cond_name}_det.jpg"), draw_detection(frame_snapshot.copy(), det_snapshot))
        try:
            res = execute_pick_and_place(
                robot=robot,
                kin=kin,
                calib=calib,
                det=det_snapshot,
                ws=ws,
                ik_method=args.ik,
                realtime=True,
            )
        except Exception as exc:  # noqa: BLE001
            res = PickPlaceResult(ok=False, failure_cause="coord_error", detail=f"예외: {exc}")
        with worker_lock:
            pending_res = res
            pending_cond = cond_name
            state = "CONFIRM"
            if res.ok:
                status_msg = f"DONE {res.elapsed_s:.1f}s (grip={res.gripper_opening}%) | Y/SPACE=OK | 1:coord 2:grasp 3:drop | X:skip"
                print(f"[완료] {cond_name} | 로봇 판정: 성공 ({res.elapsed_s:.2f}s, grip={res.gripper_opening}%) -> 창에서 Y/Space(성공) 또는 1~5(실패) 입력")
            else:
                cause = res.failure_cause or "grasp_fail"
                status_msg = f"FAIL({cause}) {res.elapsed_s:.1f}s | SPACE/Y=Save fail | 1-5=Change cause | X=Skip"
                print(f"[완료] {cond_name} | 로봇 판정: 실패({FAILURE_CAUSES.get(cause, cause)}, {res.elapsed_s:.2f}s) -> 창에서 Space(실패 저장) 또는 X(무시)")

    def commit_trial(success: int, cause: str) -> None:
        nonlocal state, pending_res, status_msg, cached_trials
        assert pending_res is not None
        res = pending_res
        rec = append_trial(
            args.csv,
            TrialRecord(
                timestamp="",
                system="A",
                condition=pending_cond,
                trial=0,
                success=success,
                elapsed_s=res.elapsed_s,
                failure_cause="" if success else cause,
                block_color=args.color if pending_cond != "color_change" else "other",
                u_px=round(res.u, 1) if res.u is not None else "",
                v_px=round(res.v, 1) if res.v is not None else "",
                target_x_m=res.target_x if res.target_x is not None else "",
                target_y_m=res.target_y if res.target_y is not None else "",
                note=res.detail,
            ),
        )
        save_summary(args.csv, args.summary)
        cached_trials = load_trials(args.csv)
        tag = "OK" if success else f"FAIL({cause})"
        status_msg = f"Saved [{pending_cond} #{rec.trial}] {tag} {rec.elapsed_s:.1f}s | Place next block & press SPACE"
        print(f"  -> 저장 완료: {pending_cond} #{rec.trial} | {tag} | {rec.elapsed_s:.2f}s")
        pending_res = None
        state = "IDLE"

    win = "System A Live Eval (SPACE: Run | C: Cond | U: Undo | Q: Quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 800, 740)

    print("=== 시스템 A 연속 실험 러너 시작 ===")
    print("카메라 창을 클릭한 뒤 Space를 누르면 바로 실행됩니다.")

    try:
        while True:
            frame = read_frame(cap)
            with worker_lock:
                cur_state = state
                cur_msg = status_msg
            cond = COND_LIST[cond_idx]
            cond_trials = [r for r in cached_trials if r.system == "A" and r.condition == cond]
            n_done = len(cond_trials)
            n_ok = sum(1 for r in cond_trials if r.success == 1)

            view = frame.copy()
            # Draw workspace boundary & drop box
            if ws_poly is not None:
                cv2.polylines(view, [ws_poly], True, (200, 180, 0), 1)
            if box_px is not None:
                cv2.drawMarker(view, box_px, (0, 0, 255), cv2.MARKER_DIAMOND, 14, 2)

            # Plot past trial positions for current condition so user can spread 20 positions cleanly
            for r in cond_trials:
                if r.u_px != "" and r.v_px != "":
                    pu, pv = int(round(float(r.u_px))), int(round(float(r.v_px)))
                    col = (0, 220, 0) if r.success else (0, 60, 255)
                    cv2.circle(view, (pu, pv), 5, col, -1 if r.success else 2)
                    cv2.putText(view, str(r.trial), (pu + 6, pv - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

            # Live detection preview when IDLE
            det = None
            if cur_state == "IDLE":
                det = detect_block(frame, hsv_ranges)
                draw_detection(view, det)
                if det is not None:
                    bx, by = calib.pixel_to_xy(det.u, det.v)
                    in_ws = ws.is_reachable_xy(bx, by)
                    col = (0, 255, 0) if in_ws else (0, 0, 255)
                    txt = f"XY=({bx*1000:.0f},{by*1000:.0f})mm {'IN' if in_ws else 'OUT_OF_RANGE'}"
                    cv2.putText(view, txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2)

            # Build bottom HUD panel (clean dark bar, zero overlap with workspace)
            hud = np.full((125, 640, 3), 28, dtype=np.uint8)
            rate_str = f"{100.0 * n_ok / n_done:.0f}%" if n_done else "-"
            hdr_color = (0, 230, 255) if cur_state == "CONFIRM" else ((0, 180, 255) if cur_state == "RUNNING" else (80, 255, 120))
            cv2.putText(
                hud,
                f"COND: {cond.upper()}  [{n_done}/{args.target}]   OK: {n_ok}/{n_done} ({rate_str})   STATE: {cur_state}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                hdr_color,
                2,
            )
            cv2.putText(hud, cur_msg[:78], (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1)
            if cur_state == "CONFIRM":
                help1 = "[SPACE / Y] Save recommended   [1] coord  [2] grasp  [3] drop  [4] detect  [5] range"
                help2 = "[X] Discard (do not save)      Note: check if block stayed inside box >= 2s"
            elif cur_state == "RUNNING":
                help1 = "Robot moving... please keep hands clear of the workspace."
                help2 = ""
            else:
                extra = "  * Place OTHER COLOR block (HSV stays blue)" if cond == "color_change" else ""
                help1 = f"[SPACE] Run trial #{n_done + 1}   [C] Next cond   [U] Undo last   [S] Summary   [Q] Quit"
                help2 = f"All trials saved -> {args.csv.relative_to(config.ROOT)}{extra}"
            cv2.putText(hud, help1, (12, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1)
            if help2:
                cv2.putText(hud, help2, (12, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)

            canvas = np.vstack([view, hud])
            cv2.imshow(win, canvas)

            key = cv2.waitKey(25) & 0xFF
            if key == 255:
                continue

            if cur_state == "IDLE":
                if key in (ord("q"), 27):
                    break
                elif key in (ord(" "), 13):
                    with worker_lock:
                        state = "RUNNING"
                        status_msg = f"Running {cond} #{n_done + 1}..."
                    snap = frame.copy()
                    det_now = detect_block(snap, hsv_ranges)
                    threading.Thread(target=run_worker, args=(snap, det_now, cond), daemon=True).start()
                elif key == ord("c"):
                    cond_idx = (cond_idx + 1) % len(COND_LIST)
                    status_msg = f"Switched condition -> {COND_LIST[cond_idx]}"
                    print(f"[조건 전환] {COND_LIST[cond_idx]} ({CONDITIONS[COND_LIST[cond_idx]]})")
                elif key == ord("u"):
                    rem = undo_last_trial(args.csv, args.summary)
                    if rem:
                        cached_trials = load_trials(args.csv)
                        status_msg = f"Undone: {rem.condition} #{rem.trial}"
                        print(f"[취소됨] {rem.condition} #{rem.trial}")
                    else:
                        status_msg = "Nothing to undo."
                elif key == ord("s"):
                    save_summary(args.csv, args.summary)
                    print("\n" + format_markdown_report(load_trials(args.csv)))
                    status_msg = f"Summary saved -> {args.summary.relative_to(config.ROOT)}"

            elif cur_state == "CONFIRM":
                with worker_lock:
                    res = pending_res
                if res is None:
                    continue
                if key in (ord(" "), 13, ord("y")):
                    if res.ok:
                        commit_trial(1, "")
                    else:
                        commit_trial(0, res.failure_cause or "grasp_fail")
                elif key in CAUSE_KEYS:
                    commit_trial(0, CAUSE_KEYS[key])
                elif key == ord("n"):
                    commit_trial(0, res.failure_cause or "coord_error")
                elif key == ord("x"):
                    with worker_lock:
                        pending_res = None
                        state = "IDLE"
                        status_msg = "Discarded last trial. Place block and press SPACE."
                    print("  -> 기록 무시됨 (재시도 가능)")
                elif key in (ord("q"), 27):
                    break
    finally:
        save_summary(args.csv, args.summary)
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("\n=== 종료: 최종 집계표 ===")
        print(format_markdown_report(load_trials(args.csv)))


if __name__ == "__main__":
    main()
