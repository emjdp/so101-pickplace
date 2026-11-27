"""시스템 B(ACT) 연속 비교 실험 GUI 도우미 (실시간 탑뷰+손목뷰 + 원키 실행·판정·기록).

실행:
  python scripts/eval_system_b_live.py
  python scripts/eval_system_b_live.py --condition base --target 20 --duration 30

조작 (창을 클릭해 포커스를 둔 상태에서 키 입력):
  [대기 상태 (IDLE)]
    Space / Enter : ACT 정책 1회 실행 시작
    C             : 조건 전환 (base -> new_pos -> color_change -> light_change)
    U             : 직전 저장 기록 1건 취소 (Undo)
    S             : A vs B 통합 집계표(data/experiments/summary.md) 출력 및 저장
    Q / Esc       : 안전 종료

  [실행 중 (RUNNING)]
    Space / Y     : 성공 판정으로 즉시 중단 (소요 시간 기록 후 rest 복귀)
    F / N / Esc   : 실패 판정으로 즉시 중단 (rest 복귀 후 실패 원인 선택)

  [판정 상태 (CONFIRM)]
    Space / Enter / Y : 추천 결과 저장
    1 : 정책 이탈/정지 (policy_drift)
    2 : 파지 실패 (grasp_fail)
    3 : 이송 중 낙하 (drop_fail)
    4 : 시간 초과 (timeout)
    5 : 위치/좌표 오차 (coord_error)
    X : 이번 시행 무시 (저장 안 함)
"""

from __future__ import annotations

import argparse
import csv
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
from so101.act import ACTRunner, run_act_episode  # noqa: E402
from so101.camera import open_camera, read_frame, require_gui  # noqa: E402
from so101.experiment import (  # noqa: E402
    CONDITIONS,
    FAILURE_CAUSES,
    FIELDNAMES,
    TrialRecord,
    append_trial,
    format_markdown_report,
    load_trials,
)
from so101.pickplace import DEFAULT_POSES, move_smooth  # noqa: E402
from so101.robot import connect_follower  # noqa: E402

COND_LIST = list(CONDITIONS.keys())
CAUSE_KEYS = {
    ord("1"): "policy_drift",
    ord("2"): "grasp_fail",
    ord("3"): "drop_fail",
    ord("4"): "timeout",
    ord("5"): "coord_error",
}


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
    ap.add_argument("--checkpoint", type=Path, default=config.ACT_CHECKPOINT_DIR)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--duration", type=float, default=30.0, help="시행당 최대 시간 [s]")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--condition", choices=COND_LIST, default="base")
    ap.add_argument("--target", type=int, default=20)
    ap.add_argument("--csv", type=Path, default=config.EXPERIMENT_CSV)
    ap.add_argument("--summary", type=Path, default=config.DATA_DIR / "experiments" / "summary.md")
    args = ap.parse_args()

    require_gui()

    print("ACT 정책 로딩 중...")
    runner = ACTRunner(args.checkpoint, device=args.device)
    print(f"[로드 완료] {runner.checkpoint_dir} | cameras={runner.camera_keys} | chunk={runner.policy.config.chunk_size}")

    # Open required cameras (top=YUYV, wrist=MJPG to avoid C270 Corrupt JPEG spam)
    cam_devs = {"top": config.TOP_CAM, "wrist": config.WRIST_CAM}
    caps: dict[str, cv2.VideoCapture] = {}
    for k in runner.camera_keys:
        dev = cam_devs.get(k, f"/dev/soarm_cam_{k}")
        fourcc = "MJPG" if k == "wrist" else config.CAM_FOURCC
        cap = open_camera(dev=dev, fourcc=fourcc)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        caps[k] = cap

    lock_script = config.ROOT / "tools" / "lock_camera.sh"
    if lock_script.exists():
        subprocess.run([str(lock_script)], check=False)
    for cap in caps.values():
        for _ in range(8):
            cap.grab()

    robot = connect_follower(max_relative_target=15.0)
    move_smooth(robot, {**DEFAULT_POSES["rest"], "gripper": 15.0}, speed_deg_s=50.0)

    cond_idx = COND_LIST.index(args.condition)
    state = "IDLE"  # IDLE | RUNNING | CONFIRM
    stop_reason = ""  # "" | "ok" | "fail" | "timeout"
    run_start_t = 0.0
    pending: dict | None = None
    status_msg = "Ready. Place block and press SPACE to start ACT rollout."
    lock = threading.Lock()
    frame_lock = threading.Lock()
    latest_frames: dict[str, np.ndarray] = {}
    cached_trials = load_trials(args.csv)

    # Pre-read initial frames
    for k, cap in caps.items():
        latest_frames[k] = read_frame(cap)

    def get_frame(name: str) -> np.ndarray:
        with frame_lock:
            return latest_frames[name].copy()

    cam_callables = {k: (lambda n=k: get_frame(n)) for k in runner.camera_keys}

    def run_worker(cond_name: str) -> None:
        nonlocal state, stop_reason, pending, status_msg
        t0 = time.perf_counter()

        def on_step(_step, _joints, _goal, _frames):
            with lock:
                return stop_reason == ""

        try:
            res = run_act_episode(
                runner=runner,
                robot=robot,
                cameras=cam_callables,
                max_duration_s=args.duration,
                fps=args.fps,
                on_step=on_step,
            )
            elapsed = res.elapsed_s
            steps = res.steps
            err_detail = ""
        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.perf_counter() - t0, 2)
            steps = 0
            err_detail = f"예외: {exc}"

        with lock:
            reason = stop_reason
        if not reason:
            reason = "timeout"

        # Smoothly return to rest pose for consistent next start
        try:
            move_smooth(robot, {**DEFAULT_POSES["rest"], "gripper": 15.0}, speed_deg_s=50.0)
        except Exception:  # noqa: BLE001
            pass

        with lock:
            rec_ok = 1 if reason == "ok" else 0
            rec_cause = "" if rec_ok else ("timeout" if reason == "timeout" else "policy_drift")
            pending = {
                "cond": cond_name,
                "elapsed_s": elapsed,
                "steps": steps,
                "rec_ok": rec_ok,
                "rec_cause": rec_cause,
                "note": err_detail or f"steps={steps}",
            }
            state = "CONFIRM"
            if rec_ok:
                status_msg = f"DONE {elapsed:.1f}s ({steps} steps) | SPACE/Y=Save OK | 1-5=Fail | X=Skip"
                print(f"[완료] {cond_name} | 성공 ({elapsed:.2f}s, {steps} steps) -> Space/Y 저장")
            else:
                status_msg = f"STOP({rec_cause}) {elapsed:.1f}s | SPACE/Y=Save {rec_cause} | 1-5=Change | X=Skip"
                print(f"[완료] {cond_name} | 실패({FAILURE_CAUSES.get(rec_cause, rec_cause)}, {elapsed:.2f}s) -> Space 저장 또는 1~5 변경")

    def commit_trial(success: int, cause: str) -> None:
        nonlocal state, pending, status_msg, cached_trials
        assert pending is not None
        p = pending
        rec = append_trial(
            args.csv,
            TrialRecord(
                timestamp="",
                system="B",
                condition=p["cond"],
                trial=0,
                success=success,
                elapsed_s=p["elapsed_s"],
                failure_cause="" if success else cause,
                block_color="blue" if p["cond"] != "color_change" else "other",
                note=p["note"],
            ),
        )
        save_summary(args.csv, args.summary)
        cached_trials = load_trials(args.csv)
        tag = "OK" if success else f"FAIL({cause})"
        status_msg = f"Saved [B / {p['cond']} #{rec.trial}] {tag} {rec.elapsed_s:.1f}s | Press SPACE for next"
        print(f"  -> 저장 완료: B / {p['cond']} #{rec.trial} | {tag} | {rec.elapsed_s:.2f}s")
        pending = None
        state = "IDLE"

    win = "System B (ACT) Live Eval (SPACE: Run/Success | F: Fail | C: Cond | Q: Quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 500)

    print("=== 시스템 B (ACT) 연속 실험 러너 시작 ===")
    print("창을 클릭한 뒤 Space로 시작, 상자에 들어가면 Space(성공) 또는 F(실패)를 누르세요.")

    try:
        while True:
            # Refresh live camera frames in main thread
            views = []
            for k in ("top", "wrist"):
                if k in caps:
                    fr = read_frame(caps[k])
                    with frame_lock:
                        latest_frames[k] = fr
                    tile = cv2.resize(fr, (480, 360))
                    cv2.putText(tile, k.upper(), (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    views.append(tile)
            panel_top = np.hstack(views) if len(views) > 1 else views[0]
            pw = panel_top.shape[1]

            with lock:
                cur_state = state
                cur_msg = status_msg
                cur_t0 = run_start_t
            cond = COND_LIST[cond_idx]
            cond_trials = [r for r in cached_trials if r.system == "B" and r.condition == cond]
            n_done = len(cond_trials)
            n_ok = sum(1 for r in cond_trials if r.success == 1)

            hud = np.full((130, pw, 3), 28, dtype=np.uint8)
            rate_str = f"{100.0 * n_ok / n_done:.0f}%" if n_done else "-"
            if cur_state == "RUNNING":
                el = time.perf_counter() - cur_t0
                hdr = f"SYS B (ACT) | COND: {cond.upper()} [{n_done}/{args.target}] | RUNNING {el:.1f}s / {args.duration:.0f}s"
                hdr_col = (0, 190, 255)
                h1 = "[SPACE / Y] Block in box -> STOP & SUCCESS     [F / N / ESC] STOP & FAIL"
                h2 = "Robot returns to rest pose automatically after stop."
            elif cur_state == "CONFIRM":
                hdr = f"SYS B (ACT) | COND: {cond.upper()} [{n_done}/{args.target}] | OK: {n_ok}/{n_done} ({rate_str}) | CONFIRM"
                hdr_col = (0, 235, 255)
                h1 = "[SPACE / Y] Save recommended   [1] drift  [2] grasp  [3] drop  [4] timeout  [5] coord"
                h2 = "[X] Discard (do not save)"
            else:
                hdr = f"SYS B (ACT) | COND: {cond.upper()} [{n_done}/{args.target}] | OK: {n_ok}/{n_done} ({rate_str}) | IDLE"
                hdr_col = (80, 255, 120)
                h1 = f"[SPACE] Run trial #{n_done + 1}   [C] Next cond   [U] Undo last   [S] Summary   [Q] Quit"
                h2 = f"Saves to {args.csv.relative_to(config.ROOT)} & updates A vs B summary.md"

            cv2.putText(hud, hdr, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, hdr_col, 2)
            cv2.putText(hud, cur_msg[:105], (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1)
            cv2.putText(hud, h1, (14, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 220, 255), 1)
            cv2.putText(hud, h2, (14, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

            cv2.imshow(win, np.vstack([panel_top, hud]))
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue

            if cur_state == "IDLE":
                if key in (ord("q"), 27):
                    break
                elif key in (ord(" "), 13):
                    with lock:
                        state = "RUNNING"
                        stop_reason = ""
                        run_start_t = time.perf_counter()
                        status_msg = f"ACT running {cond} #{n_done + 1}... Press SPACE on success, F on fail."
                    threading.Thread(target=run_worker, args=(cond,), daemon=True).start()
                elif key == ord("c"):
                    cond_idx = (cond_idx + 1) % len(COND_LIST)
                    status_msg = f"Switched condition -> {COND_LIST[cond_idx]}"
                    print(f"[조건 전환] {COND_LIST[cond_idx]} ({CONDITIONS[COND_LIST[cond_idx]]})")
                elif key == ord("u"):
                    rem = undo_last_trial(args.csv, args.summary)
                    if rem:
                        cached_trials = load_trials(args.csv)
                        status_msg = f"Undone: {rem.system}/{rem.condition} #{rem.trial}"
                        print(f"[취소됨] {rem.system}/{rem.condition} #{rem.trial}")
                    else:
                        status_msg = "Nothing to undo."
                elif key == ord("s"):
                    save_summary(args.csv, args.summary)
                    print("\n" + format_markdown_report(load_trials(args.csv)))
                    status_msg = f"Summary saved -> {args.summary.relative_to(config.ROOT)}"

            elif cur_state == "RUNNING":
                if key in (ord(" "), 13, ord("y")):
                    with lock:
                        stop_reason = "ok"
                        status_msg = "Success! Returning to rest pose..."
                elif key in (ord("f"), ord("n"), 27):
                    with lock:
                        stop_reason = "fail"
                        status_msg = "Stopped (fail). Returning to rest pose..."

            elif cur_state == "CONFIRM":
                with lock:
                    p = pending
                if p is None:
                    continue
                if key in (ord(" "), 13, ord("y")):
                    commit_trial(p["rec_ok"], p["rec_cause"])
                elif key in CAUSE_KEYS:
                    commit_trial(0, CAUSE_KEYS[key])
                elif key == ord("n"):
                    commit_trial(0, "policy_drift")
                elif key == ord("x"):
                    with lock:
                        pending = None
                        state = "IDLE"
                        status_msg = "Discarded last trial. Press SPACE to run again."
                    print("  -> 기록 무시됨")
                elif key in (ord("q"), 27):
                    break
    finally:
        save_summary(args.csv, args.summary)
        for cap in caps.values():
            cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("\n=== 종료: A vs B 최종 집계표 ===")
        print(format_markdown_report(load_trials(args.csv)))


if __name__ == "__main__":
    main()
