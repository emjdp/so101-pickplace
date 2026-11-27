"""시스템 B: ACT 모방학습 정책 실행 및 평가 스크립트 (계획서 3.3절).

학습한 ACT 체크포인트를 팔로워암에서 실행하고 비교 실험 결과를 CSV에 기록한다.

사용법:
  # 1) 기본 체크포인트로 실기 1회 실행 (기본 25초)
  python scripts/run_system_b.py --checkpoint outputs/act_pickplace/pretrained_model

  # 2) 비교 실험 조건 지정 및 결과 기록
  python scripts/run_system_b.py --checkpoint outputs/act_pickplace/pretrained_model --condition base --log

  # 3) 로봇·카메라 없이 체크포인트 로딩 및 추론 루프 점검 (--dry-run)
  python scripts/run_system_b.py --checkpoint outputs/act_pickplace/pretrained_model --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402

import config  # noqa: E402
from so101.act import ACTRunner, run_act_episode  # noqa: E402
from so101.camera import open_camera  # noqa: E402
from so101.experiment import CONDITIONS, FAILURE_CAUSES, TrialRecord, append_trial  # noqa: E402
from so101.pickplace import DEFAULT_POSES, DryRunRobot, move_smooth  # noqa: E402
from so101.robot import connect_follower  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=config.ACT_CHECKPOINT_DIR, help="ACT pretrained_model 경로")
    ap.add_argument("--device", default="cpu", help="추론 장치 (cpu 또는 cuda)")
    ap.add_argument("--duration", type=float, default=25.0, help="1회 시행 최대 시간 [s]")
    ap.add_argument("--fps", type=float, default=30.0, help="제어 주기 [Hz]")
    ap.add_argument("--condition", choices=list(CONDITIONS), default="base", help="비교 실험 조건")
    ap.add_argument("--color", default="blue", help="블록 색상 (실험 기록용)")
    ap.add_argument("--log", action="store_true", help="실행 결과를 data/experiments/trials.csv 에 기록")
    ap.add_argument("--csv", type=Path, default=config.EXPERIMENT_CSV, help="기록할 CSV 경로")
    ap.add_argument("--dry-run", action="store_true", help="로봇·카메라 없이 가짜 입력으로 추론 루프만 점검")
    ap.add_argument("--dry-steps", type=int, default=15, help="--dry-run 실행 스텝 수")
    args = ap.parse_args()

    runner = ACTRunner(args.checkpoint, device=args.device)
    print(f"[ACT 정책 로드 완료] 경로: {runner.checkpoint_dir}")
    print(f"  장치: {runner.device} | 요구 카메라: {runner.camera_keys} | chunk_size: {runner.policy.config.chunk_size}")

    cams = {}
    caps = []
    if args.dry_run:
        robot = DryRunRobot()
        dummy_img = np.full((config.CAM_HEIGHT, config.CAM_WIDTH, 3), 128, dtype=np.uint8)
        for k in runner.camera_keys:
            cams[k] = lambda: dummy_img
    else:
        cam_devs = {"top": config.TOP_CAM, "wrist": config.WRIST_CAM}
        for k in runner.camera_keys:
            dev = cam_devs.get(k, f"/dev/soarm_cam_{k}")
            fourcc = "MJPG" if k == "wrist" else config.CAM_FOURCC
            cap = open_camera(dev=dev, fourcc=fourcc)
            caps.append(cap)
            cams[k] = cap
        robot = connect_follower(max_relative_target=12.0)

    try:
        if args.dry_run:
            res = run_act_episode(
                runner=runner,
                robot=robot,
                cameras=cams,
                max_duration_s=args.duration,
                fps=1000.0,
                on_step=lambda step, *_: step < args.dry_steps,
            )
        else:
            print("ACT 정책 실행 시작 (중단하려면 Ctrl+C)...")
            try:
                res = run_act_episode(
                    runner=runner,
                    robot=robot,
                    cameras=cams,
                    max_duration_s=args.duration,
                    fps=args.fps,
                )
            except KeyboardInterrupt:
                print("\n사용자 중단 (Ctrl+C)")
                res = None
            print("휴식 자세(rest)로 복귀합니다...")
            move_smooth(robot, {**DEFAULT_POSES["rest"], "gripper": 20.0}, speed_deg_s=45.0)

        if res is not None:
            print(f"[시스템 B 실행 완료] 스텝: {res.steps}, 경과 시간: {res.elapsed_s:.2f} s")

        if args.log and res is not None:
            if args.dry_run:
                success, cause, note = 1, "", "dry-run"
            else:
                ans = input("성공 여부 [y/N]: ").strip().lower()
                success = 1 if ans in ("y", "yes", "1") else 0
                cause = ""
                if not success:
                    print("실패 원인 코드:", ", ".join(f"{k}({v})" for k, v in FAILURE_CAUSES.items() if k))
                    cause = input("실패 원인 코드 [policy_drift]: ").strip() or "policy_drift"
                note = input("메모 (선택): ").strip()
            rec = append_trial(
                args.csv,
                TrialRecord(
                    timestamp="",
                    system="B",
                    condition=args.condition,
                    trial=0,
                    success=success,
                    elapsed_s=res.elapsed_s,
                    failure_cause=cause,
                    block_color=args.color,
                    note=note,
                ),
            )
            print(f"[기록 저장] {args.csv} (시행 #{rec.trial}, 성공={rec.success})")
    finally:
        for cap in caps:
            cap.release()
        robot.disconnect()


if __name__ == "__main__":
    main()
